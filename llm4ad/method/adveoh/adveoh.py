from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import math
import os
import random
import time
import traceback
from collections import deque
from threading import Lock, Thread
from typing import Optional, List, Literal

import numpy as np

from .population import Population, HallOfFame
from .profiler import AdvEoHProfiler
from .prompt import HeuristicPrompt, InstancePrompt
from .sampler import AdvEoHSampler
from ...base import (
    Evaluation, LLM, Function, Program, TextFunctionProgramConverter, SecureEvaluator
)
from ...tools.profiler import ProfilerBase


# =============================================================================
# Failure-Mode Accumulation imports (lightweight, no LLM dependency)
# =============================================================================

from .failure_memory import FailureModeMemory, FailureMode
from .failure_analyzer import FailureAnalyzer


class AdvEoH:
    """Adversarial Evolution of Heuristics (AdvEoH) — generic, task-agnostic.

    Idea
    ----
    Two LLMs co-evolve in an adversarial loop:

    * ``llm_heu``  — the *heuristic* LLM. It evolves programs that solve a
      black-box optimization task; its goal is to maximise solution quality.
    * ``llm_inst`` — the *instance* LLM. It evolves programs that *generate*
      problem instances; its goal is to produce instances on which the current
      best heuristics perform as poorly as possible.

    Each side is therefore the other side's fitness landscape: heuristics are
    selected for doing well on the hardest instances seen so far, while instance
    generators are selected for exposing the heuristics' weaknesses. A held-out
    score on the task's real/default dataset is reported every heuristic
    generation and is treated as the true progress signal — it is decoupled from
    the adversarial training signal, which is non-stationary and can overfit to
    generated instances.

    The method itself knows nothing about the task. The user supplies two
    task-specific *adapter* evaluations (see ``llm4ad/method/adveoh/adapters/``):

    * ``heu_evaluation``  — an ``Adv<XXX>Evaluation`` whose ``evaluate_program``
      accepts an ``instances=None`` kwarg. When ``instances`` is ``None`` it
      falls back to the task's default dataset (used for warm-up and held-out
      reporting); otherwise it scores the heuristic on the supplied generated
      instances.
    * ``inst_evaluation`` — an ``<XXX>InstanceGenEvaluation`` that validates the
      output of ``generate_instance(seed)`` and returns the instance data, or
      ``None`` on failure.

    Per-generation loop
    -------------------
    Phase A (evolve heuristics).
        Heuristics are re-scored on a *frozen* phase instance set — a
        deterministic fixed train prefix plus instances drawn from the instance
        HoF/population — so that all scores within a generation are comparable.
        New offspring are sampled with EoH-style operators (e1/e2/m1/m2) and
        survive against the current population.

    Phase B (evolve instances).
        Triggered either on a fixed interval or when the held-out score
        plateaus. Instance generators are scored against a *frozen* heuristic
        reference set (top heuristics by fixed score + by adversarial score +
        the best held-out heuristic) and evolved for a few inner rounds.

    Fitness
    -------
    * Heuristic fitness = score on the frozen phase instance set (higher is
      better; re-evaluated each phase).
    * Generator fitness = minimax: ``-mean_seed( max_h score(h, instance_seed) )``.
      A generator is fit if even the strongest current heuristics do poorly on
      its instances; the ``max`` over heuristics prevents single-hero
      overfitting, and the ``mean`` over ``n_seed`` seeds rewards stable
      difficulty.

    Adversarial communication
    -------------------------
    Each LLM is shown natural-language descriptions (the ``algorithm`` field) of
    the other side's strongest individuals and explicitly told to counter them.
    ``n_opponent_desc`` controls how many descriptions are injected.

    Populations, HoF and selection
    ------------------------------
    * Two populations (heuristic, instance) with *code-only* deduplication:
      different heuristics may legitimately tie on the same instances, so
      score-based dedup is too aggressive.
    * A sliding-window Hall of Fame per side (``hof_max_gen`` generations,
      ``hof_top_k`` elites per generation) supplies stable, stratified opponents
      across phases.
    * Parent selection follows multiplicative-weights (MWU) with per-side
      learning rates ``mwu_eta_heu`` / ``mwu_eta_inst``.

    Phase scheduling (G update trigger)
    -----------------------------------
    * Fixed-interval mode (``use_plateau_trigger=False``): refresh instances
      every ``inst_update_interval`` heuristic generations.
    * Plateau-trigger mode (``use_plateau_trigger=True``): refresh instances
      when the held-out score fails to improve by more than ``plateau_epsilon``
      for ``plateau_window`` consecutive heuristic generations, but no sooner
      than ``min_heu_generations_per_phase``.
    On a trigger, instances are evolved for ``inst_update_inner_rounds`` inner
    rounds against a frozen heuristic reference set, after which a new phase
    starts (frozen instance set rebuilt, plateau counters reset).

    Failure-Mode Accumulation (FMA)
    -------------------------------
    When ``use_fma=True``, the system maintains a persistent FailureModeMemory.
    After each evaluation round, a FailureAnalyzer extracts abstract weakness
    descriptions from instances where heuristics performed poorly. These failure
    modes are stored in memory and influence future evolution:

    * Generator objective: reward = difficulty + lambda * novelty_penalty, where
      novelty_penalty = 1 - max_similarity(description, archived_failures).
    * Heuristic objective: fitness weighted by coverage over archived failure
      modes (fraction of modes the heuristic survives).
    * Failure descriptions are injected into prompts so both sides are aware of
      historical weaknesses.

    Robustness & concurrency
    ------------------------
    * ``n_seed``              — seeds per generator (fitness averaged across seeds).
    * ``n_inst_sample``       — instance generators sampled from the HoF for
      heuristic evaluation.
    * ``n_fixed_inst_sample`` — size of the deterministic fixed train prefix
      (stable across runs and resumes).
    * ``num_samplers``        — parallel sampler threads (concurrency only; does
      not change the per-generation sample budget).
    * ``num_evaluators``      — parallel evaluation workers (thread or process
      pool via ``multi_thread_or_process_eval``).
    * Heuristic evaluations are cached by (function id, instance set) to avoid
      re-scoring the same (heuristic, instances) pair.

    Resume
    ------
    ``resume_adveoh`` rebuilds both populations, HoFs, MWU weights and phase
    state (phase id, phase best, no-improve count, phase heuristic generations)
    plus the best held-out heuristic from a previous run's logs. The FMA memory
    is also persisted and reloaded from the log directory.

    Args:
        llm_heu             : LLM for generating heuristics.
        llm_inst            : LLM for generating instance generators.
        heu_evaluation      : Adv<XXX>Evaluation instance (task-specific).
        inst_evaluation     : <XXX>InstanceGenEvaluation instance (task-specific).
        profiler            : AdvEoHProfiler or ProfilerBase instance.
        max_generations     : number of adversarial (heuristic) generations.
        max_sample_nums     : terminate after this many sampled functions
                              across both LLMs.
        pop_size_heu        : heuristic population size.
        pop_size_inst       : instance-generator population size.
        samples_per_gen_heu : heuristic samples per generation (Phase A).
        samples_per_gen_inst: instance samples per generation (Phase B inner round).
        num_samplers        : parallel sampler threads (concurrency, not budget).
        n_inst_sample       : instance generators sampled from the HoF for heu eval.
        n_fixed_inst_sample : size of the deterministic fixed train prefix.
        n_seed              : seeds per instance generator (robustness).
        n_opponent_desc     : opponent descriptions injected into prompts.
        selection_num       : parents per crossover operator.
        use_e2/m1/m2        : toggle individual operators.
        hof_max_gen         : Hall of Fame sliding-window size (generations).
        hof_top_k           : elites archived per generation in the HoF.
        use_plateau_trigger : use plateau-based G updates instead of a fixed interval.
        inst_update_interval: fixed-interval G update period (heuristic generations).
        plateau_window      : consecutive non-improving heuristic generations
                              needed to trigger a plateau G update.
        min_heu_generations_per_phase: minimum heuristic generations before a
                              plateau trigger is allowed.
        inst_update_inner_rounds: instance evolution inner rounds per G update.
        plateau_epsilon     : minimum held-out improvement to reset the plateau count.
        mwu_eta_heu         : MWU learning rate for the heuristic population.
        mwu_eta_inst        : MWU learning rate for the instance population.
        num_evaluators      : parallel evaluation workers.
        multi_thread_or_process_eval: 'thread' or 'process' evaluation pool.
        resume_mode         : internal flag; prefer ``resume_adveoh`` to resume.
        debug_mode          : print detailed information.

    FMA-specific args:
        use_fma                 : enable Failure-Mode Accumulation.
        fma_lambda              : weight of novelty penalty in generator reward.
        fma_memory_size         : max failure modes retained in memory.
        fma_min_delta           : minimum performance delta to record a failure.
        fma_top_k_failures      : number of top failure descriptions to inject
                                  into prompts.
        fma_heuristic_coverage_weight: weight of failure-mode coverage in
                                  heuristic fitness.
        fma_archive_path        : path for persisting failure memory (defaults to
                                  log_dir/failure_memory.json when profiler is
                                  available).
    """

    def __init__(self,
                 llm_heu: LLM,
                 llm_inst: LLM,
                 heu_evaluation: Evaluation,
                 inst_evaluation: Evaluation,
                 profiler: ProfilerBase = None,
                 max_generations: int = 10,
                 max_sample_nums: Optional[int] = None,
                 pop_size_heu: int = 10,
                 pop_size_inst: int = 10,
                 samples_per_gen_heu: int = 8,
                 samples_per_gen_inst: int = 4,
                 n_inst_sample: int = 1,
                 n_fixed_inst_sample: int = 32,
                 n_seed: int = 8,
                 n_opponent_desc: int = 3,
                 selection_num: int = 2,
                 use_e2: bool = True,
                 use_m1: bool = True,
                 use_m2: bool = True,
                 hof_max_gen: int = 5,
                 hof_top_k: int = 5,
                 use_plateau_trigger: bool = False,
                 inst_update_interval: int = 4,
                 plateau_window: int = 3,
                 min_heu_generations_per_phase: int = 3,
                 inst_update_inner_rounds: int = 2,
                 plateau_epsilon: float = 0.0,
                 mwu_eta_heu: float = 2.5,
                 mwu_eta_inst: float = 1.6,
                 num_evaluators: int = 1,
                 num_samplers: int = 1,
                 *,
                 resume_mode: bool = False,
                 debug_mode: bool = False,
                 multi_thread_or_process_eval: Literal['thread', 'process'] = 'thread',
                 # --- FMA parameters ---
                 use_fma: bool = True,
                 fma_lambda: float = 0.3,
                 fma_memory_size: int = 100,
                 fma_min_delta: float = 0.05,
                 fma_top_k_failures: int = 3,
                 fma_heuristic_coverage_weight: float = 0.2,
                 fma_archive_path: Optional[str] = None,
                 **kwargs):

        

        self._debug_mode = debug_mode
        self._resume_mode = resume_mode
        llm_heu.debug_mode = debug_mode
        llm_inst.debug_mode = debug_mode

        # --- Generic evaluations (user supplies task-specific adapters) ---
        self._heu_evaluation = heu_evaluation
        self._inst_evaluation = inst_evaluation

        # --- Hyperparameters ---
        self._max_generations = max_generations
        self._max_sample_nums = max_sample_nums
        self._pop_size_heu = pop_size_heu
        self._pop_size_inst = pop_size_inst
        self._samples_per_gen_heu = samples_per_gen_heu
        self._samples_per_gen_inst = samples_per_gen_inst
        self._num_samplers = max(1, int(num_samplers))
        self._n_inst_sample = n_inst_sample
        self._n_fixed_inst_sample = n_fixed_inst_sample
        self._n_seed = n_seed
        self._n_opponent_desc = n_opponent_desc
        self._selection_num = selection_num
        self._use_e2 = use_e2
        self._use_m1 = use_m1
        self._use_m2 = use_m2
        self._use_plateau_trigger = use_plateau_trigger
        self._inst_update_interval = max(1, inst_update_interval)
        self._plateau_window = max(1, plateau_window)
        self._min_heu_generations_per_phase = max(1, min_heu_generations_per_phase)
        self._inst_update_inner_rounds = max(1, inst_update_inner_rounds)
        self._plateau_epsilon = plateau_epsilon
        self._mwu_eta_heu = mwu_eta_heu
        self._mwu_eta_inst = mwu_eta_inst

        # --- FMA configuration ---
        self._use_fma = use_fma
        self._fma_lambda = fma_lambda
        self._fma_memory_size = fma_memory_size
        self._fma_min_delta = fma_min_delta
        self._fma_top_k_failures = fma_top_k_failures
        self._fma_heuristic_coverage_weight = fma_heuristic_coverage_weight

        # --- Parse heuristic template ---
        self._template_program_heu_str = heu_evaluation.template_program
        self._task_description_heu_str = heu_evaluation.task_description
        self._function_to_evolve_heu: Function = TextFunctionProgramConverter.text_to_function(
            self._template_program_heu_str
        )
        self._template_program_heu: Program = TextFunctionProgramConverter.text_to_program(
            self._template_program_heu_str
        )

        # --- Parse instance template ---
        self._template_program_inst_str = inst_evaluation.template_program
        self._task_description_inst_str = inst_evaluation.task_description
        self._function_to_evolve_inst: Function = TextFunctionProgramConverter.text_to_function(
            self._template_program_inst_str
        )
        self._template_program_inst: Program = TextFunctionProgramConverter.text_to_program(
            self._template_program_inst_str
        )

        # --- Populations ---
        self._heu_pop = Population(pop_size=pop_size_heu, eta=mwu_eta_heu)
        self._inst_pop = Population(pop_size=pop_size_inst, eta=mwu_eta_inst)

        # --- Hall of Fame ---
        self._heu_hof = HallOfFame(max_gen=hof_max_gen, top_k=hof_top_k)
        self._inst_hof = HallOfFame(max_gen=hof_max_gen, top_k=hof_top_k)

        # --- Samplers ---
        self._sampler_heu = AdvEoHSampler(llm_heu, self._template_program_heu_str)
        self._sampler_inst = AdvEoHSampler(llm_inst, self._template_program_inst_str)

        # --- Evaluators ---
        self._heu_evaluator = SecureEvaluator(heu_evaluation, debug_mode=debug_mode, **kwargs)
        self._inst_evaluator = SecureEvaluator(inst_evaluation, debug_mode=debug_mode, **kwargs)

        # --- Profiler ---
        self._profiler = profiler

        # --- Evaluation executor ---
        assert multi_thread_or_process_eval in ['thread', 'process']
        if multi_thread_or_process_eval == 'thread':
            self._evaluation_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=num_evaluators
            )
        else:
            self._evaluation_executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=num_evaluators
            )

        # --- Statistics ---
        self._tot_sample_nums = 0
        self._best_heldout_function = None
        self._best_heldout_score = float('-inf')
        self._instance_cache_by_code = {}
        self._heuristic_eval_cache = {}
        self._heuristic_eval_cache_lock = Lock()
        self._resume_heu_generation = 0
        self._resume_inst_generation = 0
        self._resume_phase_id = 1
        self._resume_phase_heu_gens = 0
        self._resume_phase_best = float('-inf')
        self._resume_no_improve_count = 0
        self._skip_initial_heuristic_population_once = resume_mode
        self._skip_initial_instance_population_once = resume_mode

        # --- FMA: Failure-Mode Accumulation ---
        if fma_archive_path is None and profiler is not None:
            log_dir = getattr(profiler, '_log_dir', None)
            if log_dir:
                fma_archive_path = os.path.join(log_dir, 'failure_memory.json')
        self._fma_memory = FailureModeMemory(path=fma_archive_path)
        self._fma_analyzer = FailureAnalyzer(task=self._task_description_heu_str[:64])
        # Track which failure signatures each heuristic has survived
        self._heuristic_coverage: dict[str, set[str]] = {}
        # Instance evaluations that may reveal failures (populated during scoring)
        self._pending_failure_analysis: list[dict] = []
        self._pending_failure_lock = Lock()

        # --- Pass parameters to profiler ---
        if profiler is not None:
            self._profiler.record_parameters(llm_heu, heu_evaluation, self)

        if use_fma:
            self._fma_analyzer = FailureAnalyzer(
                task=self._task_description_heu_str[:120],
                llm_call_fn=self._fma_llm_call,   # see _fma_llm_call below
                model='claude-haiku-4-5-20251001',
                max_tokens=80,
            )
            
    # ================================================================
    #  FMA: Failure-Mode Accumulation helpers
    # ================================================================

    def _get_task_name(self) -> str:
        """Derive a short task name from the heuristic task description."""
        desc = self._task_description_heu_str[:60].strip()
        # Simple heuristic: first few meaningful words
        words = desc.replace(',', ' ').replace('.', ' ').split()
        key_words = [w for w in words if len(w) > 3][:4]
        return '_'.join(key_words).lower() if key_words else 'task'

    def _fma_individual_id(self, func: Function) -> str:
        """Get or compute a deterministic SHA1 for a function."""
        return Population._individual_id(func)

    def _fma_heuristic_coverage_key(self, heu_func: Function) -> str:
        return self._fma_individual_id(heu_func)

    def _fma_llm_call(self, prompt: str, system: str) -> Optional[str]:
        """Thin adapter: call AdvEoH's heuristic LLM for failure description generation.
    
        Bypasses the full sampler/population machinery — fires a single raw call.
        Returns the text response or None on failure.
        """
        try:
            llm = self._sampler_heu.llm
            # LLM's portable interface is draw_sample(); not every backend
            # implements provider-specific generate/system arguments.
            response = llm.draw_sample(f'{system}\n\n{prompt}')
            return response.strip() if response else None
        except Exception:
            return None
    
    
# ---------------------------------------------------------------------------
# Fix 3a: replace _fma_mark_heuristic_survived
# ---------------------------------------------------------------------------
 
    def _fma_try_mark_covered(
        self,
        heu_func,          # Function
        generator_func,    # Function  ← NEW: need the generator to gate coverage
        heuristic_score: float,
        optimal_value: Optional[float],
    ) -> None:
        """Credit heuristic coverage ONLY for failure modes triggered by this generator.
    
        Fix 3: the old implementation looped over all archived modes and credited
        coverage whenever the heuristic scored well on *any* instance, regardless
        of which generator produced it. This produced spurious coverage credits.
    
        The corrected version passes generator_id to FailureModeMemory.try_mark_covered,
        which only credits modes whose generator_id matches the current generator.
        """
        if not self._use_fma or len(self._fma_memory) == 0:
            return
    
        generator_id = self._fma_individual_id(generator_func)
        heu_key = self._fma_heuristic_coverage_key(heu_func)
    
        newly_covered = self._fma_memory.try_mark_covered(
            heu_func_id=heu_key,
            generator_id=generator_id,
            heuristic_score=heuristic_score,
            optimal_value=optimal_value,
            min_delta=self._fma_min_delta,
        )
    
        if newly_covered:
            if heu_key not in self._heuristic_coverage:
                self._heuristic_coverage[heu_key] = set()
            self._heuristic_coverage[heu_key].update(newly_covered)
    
    
    # ---------------------------------------------------------------------------
    # Fix 3b: replace _score_instances_against_reference
    # ---------------------------------------------------------------------------
    
    def _score_instances_against_reference(
        self,
        instances: list,
        reference_heuristics: list,
        *,
        _generation: int = 0,
        _generator_func=None,   # Function | None
    ) -> Optional[float]:
        """Minimax generator score: -mean_seed(max_h score(h, instance_seed)).
    
        Fixes applied:
        - Passes _generator_func to _fma_try_mark_covered so coverage is only
        credited for failure modes triggered by THIS generator (Fix 3).
        - Failure analysis queue entries now include generator_func for accurate
        generator_id tracking.
        """
        if not instances or not reference_heuristics:
            return None
    
        per_seed_best_scores = []
    
        for inst_idx, inst in enumerate(instances):
            seed_scores = []
    
            for heu_func in reference_heuristics:
                score = self._evaluate_heuristic_function(heu_func, instances=[inst])
                if score is None:
                    continue
    
                seed_scores.append(score)
    
                if self._use_fma and _generator_func is not None:
                    # Queue failure analysis for poor-performing heuristics
                    if score < 0.5:
                        with self._pending_failure_lock:
                            self._pending_failure_analysis.append({
                                'instance': inst,
                                'heuristic_score': score,
                                'optimal_value': None,
                                'generator_func': _generator_func,   # preserved
                                'heuristic_func': heu_func,
                                'generation': _generation,
                                'instance_seed': inst_idx,
                            })
    
                    # Fix 3: pass generator_func so coverage is correctly gated
                    self._fma_try_mark_covered(
                        heu_func=heu_func,
                        generator_func=_generator_func,
                        heuristic_score=score,
                        optimal_value=None,
                    )
    
            if seed_scores:
                per_seed_best_scores.append(max(seed_scores))
    
        if not per_seed_best_scores:
            return None
        return -float(np.mean(per_seed_best_scores))
    
    
    # ---------------------------------------------------------------------------
    # Fix 1 (also update): _fma_maybe_record_failure
    # Unchanged in logic — but FailureAnalyzer now calls LLM internally,
    # so no changes needed here. Included for completeness / clarity.
    # ---------------------------------------------------------------------------
    
    def _fma_maybe_record_failure(
        self,
        instance,
        heuristic_score: float,
        optimal_value: Optional[float],
        generator_func,
        heuristic_func,
        generation: int,
        instance_seed: int,
    ) -> None:
        """Run failure analysis and record any new mode.
    
        Fix 1 is transparent here: FailureAnalyzer.analyze() internally calls
        the LLM to produce the description. The rest of this method is unchanged.
        """
        if not self._use_fma:
            return
    
        delta = (optimal_value - heuristic_score) if optimal_value is not None else None
        if delta is not None and delta < self._fma_min_delta:
            return
    
        mode = self._fma_analyzer.analyze(
            instance=instance,
            heuristic_score=heuristic_score,
            optimal_value=optimal_value,
            generator_description=getattr(generator_func, 'algorithm', ''),
            heuristic_description=getattr(heuristic_func, 'algorithm', ''),
            generator_id=self._fma_individual_id(generator_func),
            heuristic_id=self._fma_individual_id(heuristic_func),
            generation=generation,
            instance_seed=instance_seed,
        )
        if mode is None:
            return
    
        added = self._fma_memory.add(mode)
        if self._debug_mode and added:
            print(f'  [FMA] New failure mode: {mode.description[:80]}')
    
        # Prune memory if over capacity (unchanged)
        if len(self._fma_memory) > self._fma_memory_size:
            all_modes = self._fma_memory.get_all()
            sorted_modes = sorted(
                all_modes,
                key=lambda m: (m.strength, m.normalized_delta),
                reverse=True,
            )
            self._fma_memory._modes = {
                m.signature: m for m in sorted_modes[:self._fma_memory_size]
            }
            self._fma_memory._save()
    


    def _fma_collect_failure_analysis(self):
        """Process any pending failure analysis items accumulated during
        instance scoring."""
        with self._pending_failure_lock:
            items = list(self._pending_failure_analysis)
            self._pending_failure_analysis.clear()

        for item in items:
            self._fma_maybe_record_failure(
                instance=item.get('instance'),
                heuristic_score=item.get('heuristic_score'),
                optimal_value=item.get('optimal_value'),
                generator_func=item.get('generator_func'),
                heuristic_func=item.get('heuristic_func'),
                generation=item.get('generation', 0),
                instance_seed=item.get('instance_seed', 0),
            )

    def _fma_generator_novelty_reward(self, description: str) -> float:
        """Compute the novelty reward for a generator.

        novelty = 1 - max_similarity(description, archived_failures)

        A generator that exposes a weakness unlike any in memory receives
        a high novelty bonus.
        """
        if not self._use_fma or len(self._fma_memory) == 0:
            return 1.0
        return self._fma_memory.novelty_vs_memory(description)

    def _fma_heuristic_coverage_bonus(self, heu_func: Function) -> float:
        """Compute coverage bonus for a heuristic.

        Returns the fraction of archived failure modes that this heuristic
        has survived (verified robust against).
        """
        if not self._use_fma or len(self._fma_memory) == 0:
            return 0.0
        heu_key = self._fma_heuristic_coverage_key(heu_func)
        survived = self._heuristic_coverage.get(heu_key, set())
        return self._fma_memory.coverage_fraction(survived)

    def _fma_failure_descriptions_for_prompt(self) -> list[str]:
        """Get the top-k most threatening failure descriptions for prompt
        injection into the generator and heuristic LLMs."""
        if not self._use_fma or len(self._fma_memory) == 0:
            return []
        return self._fma_memory.get_descriptions(k=self._fma_top_k_failures)

    def _fma_modified_generator_fitness(
        self,
        base_fitness: float,
        generator_func: Function,
        instances: list,
    ) -> float:
        """Modified generator fitness: difficulty + lambda * novelty.

        ``base_fitness`` is the original minimax score (negative = hard).
        We re-interpret it as difficulty and add the novelty bonus.
        """
        if not self._use_fma:
            return base_fitness

        # Difficulty: negate base_fitness so that more negative = harder = higher difficulty
        difficulty = -base_fitness if base_fitness is not None else 0.0

        # Novelty: how different are the generated instances' failure patterns
        # from what's already in memory?
        gen_desc = getattr(generator_func, 'algorithm', '')
        novelty = self._fma_generator_novelty_reward(gen_desc)

        modified = difficulty + self._fma_lambda * novelty
        return modified

    # ================================================================
    #  Core: sample -> evaluate -> register
    # ================================================================

    def _multi_threaded_sampling(self, sample_budget: int, sample_fn: callable) -> None:
        """Run up to sample_budget sampler calls using self._num_samplers threads."""
        if sample_budget <= 0:
            return

        if self._num_samplers <= 1:
            for sample_index in range(sample_budget):
                if not self._continue_loop():
                    break
                try:
                    sample_fn(sample_index)
                except Exception:
                    if self._debug_mode:
                        traceback.print_exc()
            return

        counter_lock = Lock()
        next_sample = 0

        def worker():
            nonlocal next_sample
            while True:
                with counter_lock:
                    if next_sample >= sample_budget or not self._continue_loop():
                        return
                    sample_index = next_sample
                    next_sample += 1
                try:
                    sample_fn(sample_index)
                except Exception:
                    if self._debug_mode:
                        traceback.print_exc()

        sampler_threads = [
            Thread(target=worker)
            for _ in range(self._num_samplers)
        ]
        for thread in sampler_threads:
            thread.start()
        for thread in sampler_threads:
            thread.join()

    def _continue_loop(self) -> bool:
        return self._max_sample_nums is None or self._tot_sample_nums < self._max_sample_nums

    def _heuristic_eval_cache_key(self, heu_func: Function, instances: list | None):
        func_key = Population._individual_id(heu_func)
        if instances is None:
            instances_key = ('default',)
        else:
            instances_key = tuple(id(inst) for inst in instances)
        return func_key, instances_key

    def _get_cached_heuristic_score(self, heu_func: Function, instances: list | None):
        key = self._heuristic_eval_cache_key(heu_func, instances)
        with self._heuristic_eval_cache_lock:
            return key, self._heuristic_eval_cache.get(key, None), key in self._heuristic_eval_cache

    def _cache_heuristic_score(self, heu_func: Function, instances: list | None, score) -> None:
        key = self._heuristic_eval_cache_key(heu_func, instances)
        with self._heuristic_eval_cache_lock:
            self._heuristic_eval_cache[key] = score

    def _record_rejected_token_usage(self, token_usage: dict | None, *,
                                     role: str,
                                     operator: str,
                                     reason: str) -> None:
        if token_usage is None:
            return
        if isinstance(self._profiler, AdvEoHProfiler):
            self._profiler.register_token_usage(
                token_usage,
                role=role,
                operator=operator,
                accepted=False,
                reason=reason,
            )

    def _sample_evaluate_register_heuristic(self, prompt: str,
                                            instances: list | None = None,
                                            operator: str = 'Unknown'):
        if not self._continue_loop():
            return
        sample_start = time.time()
        thought, func, token_usage = self._sampler_heu.get_thought_function_and_usage(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            self._record_rejected_token_usage(
                token_usage,
                role='heuristic',
                operator=operator,
                reason='parse_failed',
            )
            return
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program_heu)
        if program is None:
            self._record_rejected_token_usage(
                token_usage,
                role='heuristic',
                operator=operator,
                reason='program_conversion_failed',
            )
            return
        _cache_key, cached_score, found = self._get_cached_heuristic_score(func, instances)
        if found:
            score = cached_score
            eval_time = 0.0
        else:
            score, eval_time = self._evaluation_executor.submit(
                self._heu_evaluator.evaluate_program_record_time,
                program,
                instances=instances
            ).result()
            self._cache_heuristic_score(func, instances, score)
        func.score = score
        func.evaluate_time = eval_time
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator
        self._tot_sample_nums += 1
        if self._profiler is not None:
            if isinstance(self._profiler, AdvEoHProfiler):
                self._profiler.register_function(
                    func,
                    program=str(program),
                    role='heuristic',
                    token_usage=token_usage,
                )
            else:
                self._profiler.register_function(func, program=str(program))
        self._heu_pop.register_function(func)

    def _sample_evaluate_register_instance(self, prompt: str, operator: str = 'Unknown',
                                           reference_heuristics: list[Function] | None = None):
        if not self._continue_loop():
            return
        sample_start = time.time()
        thought, func, token_usage = self._sampler_inst.get_thought_function_and_usage(prompt)
        sample_time = time.time() - sample_start
        if thought is None or func is None:
            self._record_rejected_token_usage(
                token_usage,
                role='instance',
                operator=operator,
                reason='parse_failed',
            )
            print(f'[AdvEoH] Invalid instance generator ({operator}): LLM response could not be parsed.')
            return
        program = TextFunctionProgramConverter.function_to_program(func, self._template_program_inst)
        if program is None:
            self._record_rejected_token_usage(
                token_usage,
                role='instance',
                operator=operator,
                reason='program_conversion_failed',
            )
            print(f'[AdvEoH] Invalid instance generator ({operator}): parsed function could not be converted to a program.')
            return

        # --- Generate instances with n_seed seeds ---
        instances = []
        for seed in range(self._n_seed):
            inst_data, _ = self._evaluation_executor.submit(
                self._inst_evaluator.evaluate_program_record_time,
                program,
                seed=seed
            ).result()
            if inst_data is None:
                self._record_rejected_token_usage(
                    token_usage,
                    role='instance',
                    operator=operator,
                    reason=f'instance_validation_failed_seed_{seed}',
                )
                print(
                    f'[AdvEoH] Invalid instance generator ({operator}): '
                    f'failed execution/validation at seed {seed}.'
                )
                return
            instances.append(inst_data)

        inst_fitness = self._score_instances_against_reference(
            instances,
            reference_heuristics or self._build_heuristic_reference_set(),
        )
        if inst_fitness is None:
            self._record_rejected_token_usage(
                token_usage,
                role='instance',
                operator=operator,
                reason='reference_scoring_failed',
            )
            print(f'[AdvEoH] Invalid instance generator ({operator}): could not score against heuristic references.')
            return

        # --- FMA: modify generator fitness with novelty reward ---
        if self._use_fma:
            modified_fitness = self._fma_modified_generator_fitness(
                inst_fitness, func, instances
            )
            if self._debug_mode:
                print(
                    f'  [FMA] Generator fitness: base={inst_fitness:.4f} -> '
                    f'modified={modified_fitness:.4f} '
                    f'(lambda={self._fma_lambda})'
                )
            func.score = modified_fitness
        else:
            func.score = inst_fitness

        func.evaluate_time = 0.0
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator
        func.instances = instances
        func._fma_base_fitness = inst_fitness  # store original for reference
        self._instance_cache_by_code[str(func)] = instances
        self._tot_sample_nums += 1

        if self._profiler is not None:
            if isinstance(self._profiler, AdvEoHProfiler):
                self._profiler.register_function(
                    func,
                    program=str(program),
                    role='instance',
                    token_usage=token_usage,
                )
            else:
                self._profiler.register_function(func, program=str(program))
        self._inst_pop.register_function(func)

    def _evaluate_heuristic_function(self, heu_func: Function, instances: list | None = None) -> float | None:
        _cache_key, cached_score, found = self._get_cached_heuristic_score(heu_func, instances)
        if found:
            return cached_score
        try:
            heu_program = TextFunctionProgramConverter.function_to_program(
                heu_func, self._template_program_heu
            )
            if heu_program is None:
                self._cache_heuristic_score(heu_func, instances, None)
                return None
            score, _ = self._evaluation_executor.submit(
                self._heu_evaluator.evaluate_program_record_time,
                heu_program,
                instances=instances
            ).result()
            self._cache_heuristic_score(heu_func, instances, score)
            return score
        except Exception:
            self._cache_heuristic_score(heu_func, instances, None)
            return None

    @staticmethod
    def _dedupe_functions(funcs: list[Function]) -> list[Function]:
        deduped = []
        seen = set()
        for func in funcs:
            key = str(func)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(func)
        return deduped

    def _build_heuristic_reference_set(self) -> list[Function]:
        """Freeze strong heuristics for minimax generator scoring."""
        valid = [
            f for f in self._heu_pop.population
            if f.score is not None and f.score != float('-inf')
        ]
        if not valid:
            return []

        by_fixed = sorted(
            valid,
            key=lambda f: getattr(f, 'fixed_score', f.score if f.score is not None else float('-inf')),
            reverse=True,
        )
        by_adv = sorted(
            valid,
            key=lambda f: f.score if f.score is not None else float('-inf'),
            reverse=True,
        )
        refs = by_fixed[:5] + by_adv[:2]
        if self._best_heldout_function is not None:
            refs.append(self._best_heldout_function)
        return self._dedupe_functions(refs)

    def _reevaluate_inst_pop(self, reference_heuristics: list[Function]):
        """Refresh generator scores against the frozen current heuristic reference set."""
        for inst_func in self._inst_pop.population:
            instances = self._ensure_instance_cache(inst_func)
            score = self._score_instances_against_reference(
                instances,
                reference_heuristics,
                _generation=self._gen_counter(),
                _generator_func=inst_func,
            )
            if score is not None:
                if self._use_fma:
                    inst_func.score = self._fma_modified_generator_fitness(
                        score, inst_func, instances
                    )
                else:
                    inst_func.score = score
            else:
                inst_func.score = float('-inf')

    def _gen_counter(self) -> int:
        """Approximate current generation counter."""
        return getattr(self, '_resume_heu_generation', 0) + getattr(self, '_phase_heu_gens', 0)

    # ================================================================
    #  Initialization
    # ================================================================

    def _init_heuristic_population(self):
        """Warmup: init heuristic population on the task's default dataset."""
        if self._skip_initial_heuristic_population_once:
            self._skip_initial_heuristic_population_once = False
            return

        max_tries = 2 * self._pop_size_heu

        def sample_init(_sample_index: int):
            prompt = HeuristicPrompt.get_prompt_i1(
                self._task_description_heu_str, self._function_to_evolve_heu
            )
            self._sample_evaluate_register_heuristic(prompt, instances=None, operator='i1')

        self._multi_threaded_sampling(max_tries, sample_init)
        if self._heu_pop.generation == 0 and len(self._heu_pop._next_gen_pop) > 0:
            self._heu_pop.survival()

    def _valid_inst_pop_size(self) -> int:
        return sum(
            1 for func in self._inst_pop.population
            if Population._is_valid_score(func.score)
        )

    def _valid_pending_inst_pop_size(self) -> int:
        return sum(
            1 for func in self._inst_pop._next_gen_pop
            if Population._is_valid_score(func.score)
        )

    def _init_instance_population(self, min_size: int | None = None):
        """Init/top-up instance population until enough valid generators exist."""
        if self._skip_initial_instance_population_once:
            self._skip_initial_instance_population_once = False
            return

        min_size = min_size or self._selection_num
        n_tries = 0
        while self._valid_inst_pop_size() < min_size:
            if (
                self._valid_inst_pop_size() + self._valid_pending_inst_pop_size() >= min_size
                and len(self._inst_pop._next_gen_pop) > 0
            ):
                self._inst_pop.survival()
                if self._valid_inst_pop_size() >= min_size:
                    break

            def sample_init(_sample_index: int):
                opponent_descs = self._heu_hof.get_top_k_descriptions(self._n_opponent_desc)
                # --- FMA: inject failure descriptions ---
                if self._use_fma:
                    failure_descs = self._fma_failure_descriptions_for_prompt()
                    if failure_descs:
                        opponent_descs = list(opponent_descs) if opponent_descs else []
                        opponent_descs.extend([
                            f'Archived failure mode to exploit: {d}'
                            for d in failure_descs
                        ])
                prompt = InstancePrompt.get_prompt_i1(
                    self._task_description_inst_str,
                    self._function_to_evolve_inst,
                    opponent_descriptions=opponent_descs,
                )
                self._sample_evaluate_register_instance(prompt, operator='i1')

            self._multi_threaded_sampling(self._num_samplers, sample_init)
            n_tries += self._num_samplers
            if n_tries % 10 == 0 or n_tries - self._num_samplers < 10 <= n_tries:
                print(
                    f'[AdvEoH] Still initializing instance generators: '
                    f'{self._valid_inst_pop_size()}/{min_size} valid '
                    f'({self._valid_pending_inst_pop_size()} pending valid).'
                )

        if (
            self._valid_inst_pop_size() < min_size
            and self._valid_pending_inst_pop_size() > 0
        ):
            self._inst_pop.survival()

    # ================================================================
    #  Evolution phases
    # ================================================================

    def _evolve_heuristics(self, gen: int, instances: list | None = None):
        """Phase A: evolve heuristics against a frozen phase instance set."""
        if instances is None:
            instances = self._build_phase_instances()

        if instances is not None and len(self._heu_pop) > 0:
            self._reevaluate_heu_pop(instances)
        self._heu_pop.update_weights()

        opponent_descs = self._inst_hof.get_top_k_descriptions(self._n_opponent_desc)

        # --- FMA: inject failure descriptions as "historical weaknesses to survive" ---
        if self._use_fma:
            failure_descs = self._fma_failure_descriptions_for_prompt()
            if failure_descs:
                heu_opponent_descs = list(opponent_descs) if opponent_descs else []
                heu_opponent_descs.extend([
                    f'Historical weakness that heuristics should survive: {d}'
                    for d in failure_descs
                ])
                opponent_descs = heu_opponent_descs

        if self._debug_mode and opponent_descs:
            print(f'  [heu] opponent descs: {opponent_descs}')

        operators = ['e1']
        if self._use_e2:
            operators.append('e2')
        if self._use_m1:
            operators.append('m1')
        if self._use_m2:
            operators.append('m2')

        def sample_heuristic(sample_index: int):
            operator = operators[sample_index % len(operators)]
            if operator == 'e1':
                indivs = [self._heu_pop.selection() for _ in range(self._selection_num)]
                prompt = HeuristicPrompt.get_prompt_e1(
                    self._task_description_heu_str, indivs, self._function_to_evolve_heu,
                    opponent_descriptions=opponent_descs
                )
                self._sample_evaluate_register_heuristic(prompt, instances=instances, operator='e1')
                return

            if operator == 'e2':
                indivs = [self._heu_pop.selection() for _ in range(self._selection_num)]
                prompt = HeuristicPrompt.get_prompt_e2(
                    self._task_description_heu_str, indivs, self._function_to_evolve_heu,
                    opponent_descriptions=opponent_descs
                )
                self._sample_evaluate_register_heuristic(prompt, instances=instances, operator='e2')
                return

            if operator == 'm1':
                indiv = self._heu_pop.selection()
                prompt = HeuristicPrompt.get_prompt_m1(
                    self._task_description_heu_str, indiv, self._function_to_evolve_heu,
                    opponent_descriptions=opponent_descs
                )
                self._sample_evaluate_register_heuristic(prompt, instances=instances, operator='m1')
                return

            indiv = self._heu_pop.selection()
            prompt = HeuristicPrompt.get_prompt_m2(
                self._task_description_heu_str, indiv, self._function_to_evolve_heu,
                opponent_descriptions=opponent_descs
            )
            self._sample_evaluate_register_heuristic(prompt, instances=instances, operator='m2')

        self._multi_threaded_sampling(self._samples_per_gen_heu, sample_heuristic)

        if len(self._heu_pop._next_gen_pop) > 0:
            self._heu_pop.survival()

        # --- FMA: add heuristic coverage bonus to scores ---
        if self._use_fma and len(self._fma_memory) > 0:
            for heu_func in self._heu_pop.population:
                if heu_func.score is not None and heu_func.score != float('-inf'):
                    coverage_bonus = self._fma_heuristic_coverage_bonus(heu_func)
                    if coverage_bonus > 0:
                        bonus_score = (
                            heu_func.score
                            + self._fma_heuristic_coverage_weight * coverage_bonus * abs(heu_func.score)
                        )
                        if self._debug_mode:
                            print(
                                f'  [FMA] Heuristic coverage bonus: '
                                f'base={heu_func.score:.4f}, coverage={coverage_bonus:.3f}, '
                                f'modified={bonus_score:.4f}'
                            )
                        heu_func.score = bonus_score

    def _evolve_instances(self, gen: int, reference_heuristics: list[Function] | None = None):
        """Phase B: evolve instance generators against a frozen heuristic reference set."""
        reference_heuristics = reference_heuristics or self._build_heuristic_reference_set()
        if not reference_heuristics:
            return
        self._reevaluate_inst_pop(reference_heuristics)
        self._inst_pop.update_weights()

        opponent_descs = self._heu_hof.get_top_k_descriptions(self._n_opponent_desc)

        # --- FMA: inject failure descriptions as weaknesses to exploit ---
        if self._use_fma:
            failure_descs = self._fma_failure_descriptions_for_prompt()
            if failure_descs:
                inst_opponent_descs = list(opponent_descs) if opponent_descs else []
                inst_opponent_descs.extend([
                    f'Previously discovered weakness to re-exploit: {d}'
                    for d in failure_descs
                ])
                opponent_descs = inst_opponent_descs

        if self._debug_mode and opponent_descs:
            print(f'  [inst] opponent descs: {opponent_descs}')

        operators = ['e1']
        if self._use_e2:
            operators.append('e2')
        if self._use_m1:
            operators.append('m1')
        if self._use_m2:
            operators.append('m2')

        def sample_instance(sample_index: int):
            operator = operators[sample_index % len(operators)]
            if operator == 'e1':
                indivs = [self._inst_pop.selection() for _ in range(self._selection_num)]
                prompt = InstancePrompt.get_prompt_e1(
                    self._task_description_inst_str, indivs, self._function_to_evolve_inst,
                    opponent_descriptions=opponent_descs
                )
                self._sample_evaluate_register_instance(prompt, operator='e1',
                                                        reference_heuristics=reference_heuristics)
                return

            if operator == 'e2':
                indivs = [self._inst_pop.selection() for _ in range(self._selection_num)]
                prompt = InstancePrompt.get_prompt_e2(
                    self._task_description_inst_str, indivs, self._function_to_evolve_inst,
                    opponent_descriptions=opponent_descs
                )
                self._sample_evaluate_register_instance(prompt, operator='e2',
                                                        reference_heuristics=reference_heuristics)
                return

            if operator == 'm1':
                indiv = self._inst_pop.selection()
                prompt = InstancePrompt.get_prompt_m1(
                    self._task_description_inst_str, indiv, self._function_to_evolve_inst,
                    opponent_descriptions=opponent_descs
                )
                self._sample_evaluate_register_instance(prompt, operator='m1',
                                                        reference_heuristics=reference_heuristics)
                return

            indiv = self._inst_pop.selection()
            prompt = InstancePrompt.get_prompt_m2(
                self._task_description_inst_str, indiv, self._function_to_evolve_inst,
                opponent_descriptions=opponent_descs
            )
            self._sample_evaluate_register_instance(prompt, operator='m2',
                                                    reference_heuristics=reference_heuristics)

        self._multi_threaded_sampling(self._samples_per_gen_inst, sample_instance)

        if len(self._inst_pop._next_gen_pop) > 0:
            self._inst_pop.survival()

    # ================================================================
    #  Helpers
    # ================================================================

    def _fixed_instances_for_phase(self) -> list:
        datasets = list(getattr(self._heu_evaluation, '_datasets', []) or [])
        if not datasets:
            return []
        # Use a deterministic prefix so the fixed train subset is stable across runs.
        return datasets[:min(self._n_fixed_inst_sample, len(datasets))]

    def _ensure_instance_cache(self, inst_func: Function) -> list:
        cached = getattr(inst_func, 'instances', None)
        if cached:
            return cached

        cache_key = str(inst_func)
        if cache_key in self._instance_cache_by_code:
            inst_func.instances = self._instance_cache_by_code[cache_key]
            return inst_func.instances

        program = TextFunctionProgramConverter.function_to_program(
            inst_func, self._template_program_inst
        )
        if program is None:
            inst_func.instances = []
            self._instance_cache_by_code[cache_key] = inst_func.instances
            return inst_func.instances

        instances = []
        for seed in range(self._n_seed):
            inst_data, _ = self._evaluation_executor.submit(
                self._inst_evaluator.evaluate_program_record_time,
                program,
                seed=seed,
            ).result()
            if inst_data is not None:
                instances.append(inst_data)

        inst_func.instances = instances
        self._instance_cache_by_code[cache_key] = instances
        return instances

    def _build_phase_instances(self) -> list | None:
        """Freeze fixed train + current/historical generated instances for one H phase."""
        instances = []
        instances.extend(self._fixed_instances_for_phase())

        inst_funcs = list(self._inst_pop.population)
        if len(self._inst_hof) > 0 and self._n_inst_sample > 0:
            inst_funcs.extend(self._inst_hof.sample(self._n_inst_sample, stratified=True))

        for inst_func in self._dedupe_functions(inst_funcs):
            cached = self._ensure_instance_cache(inst_func)
            if cached:
                instances.extend(cached)

        return instances or None

    def _sample_instances_for_heu_eval(self) -> list | None:
        """Stratified sample of instances from the inst HoF.
        Returns a flat list of instance tuples (in the task's native format),
        or None if HoF is empty (in which case the default dataset is used).
        """
        if len(self._inst_hof) == 0:
            return None
        inst_funcs = self._inst_hof.sample(self._n_inst_sample, stratified=True)
        if not inst_funcs:
            return None
        instances = []
        for inst_func in inst_funcs:
            cached = self._ensure_instance_cache(inst_func)
            if cached:
                instances.extend(cached)
        if not instances:
            return None
        return instances

    def _reevaluate_heu_pop(self, instances: list):
        """Re-evaluate all heuristics in heu_pop on the current instance set
        so that scores are comparable within this generation.
        """
        for heu_func in self._heu_pop.population:
            score = self._evaluate_heuristic_function(heu_func, instances=instances)
            if score is not None:
                heu_func.score = score
            else:
                heu_func.score = float('-inf')

    def _log_heldout(self, gen: int) -> float | None:
        """Evaluate all heuristics on the default fixed/train dataset and update best_h."""
        if len(self._heu_pop) == 0:
            return None

        best_score = float('-inf')
        best_func = None
        for heu_func in self._heu_pop.population:
            score = self._evaluate_heuristic_function(heu_func, instances=None)
            if score is None:
                heu_func.fixed_score = float('-inf')
                continue
            heu_func.fixed_score = score
            if score > best_score:
                best_score = score
                best_func = heu_func

        if best_func is None:
            return None
        if best_score > self._best_heldout_score:
            self._best_heldout_score = best_score
            self._best_heldout_function = copy.deepcopy(best_func)
        return best_score

    def _record_best_heldout_function(self) -> None:
        if self._best_heldout_function is None:
            return
        if not isinstance(self._profiler, AdvEoHProfiler):
            return
        program = TextFunctionProgramConverter.function_to_program(
            self._best_heldout_function,
            self._template_program_heu,
        )
        if program is None:
            return
        self._profiler.record_best_heldout_heuristic(
            self._best_heldout_function,
            program=str(program),
            heldout_score=self._best_heldout_score,
        )

    # ================================================================
    #  Main loop
    # ================================================================

    def run(self):
        # === 1. Warmup: init heuristic population on default instances ===
        print('[AdvEoH] === Phase 0: Warmup — initializing heuristic population ===')
        self._init_heuristic_population()
        if len(self._heu_pop) < self._selection_num:
            print(f'[AdvEoH] ERROR: heuristic init failed. '
                  f'Only {len(self._heu_pop)} valid samples.')
            self._finish()
            return
        self._heu_hof.update(self._heu_pop)
        if not self._continue_loop():
            print(f'[AdvEoH] Reached max_sample_nums={self._max_sample_nums} after heuristic warmup.')
            self._finish()
            return

        # === 2. Init instance population ===
        print('[AdvEoH] === Phase 0: Initializing instance population ===')
        self._init_instance_population()
        if self._valid_inst_pop_size() < self._selection_num:
            print(f'[AdvEoH] WARNING: instance init failed. '
                  f'Only {self._valid_inst_pop_size()} valid samples. '
                  f'Continuing with default instances.')
        if not self._continue_loop():
            print(f'[AdvEoH] Reached max_sample_nums={self._max_sample_nums} after instance warmup.')
            self._finish()
            return

        # === 3. Initial G refinement against the initialized H population ===
        self._inst_hof.update(self._inst_pop)
        if not self._resume_mode and self._valid_inst_pop_size() >= self._selection_num:
            reference_heuristics = self._build_heuristic_reference_set()
            print(
                f'[AdvEoH] === Phase 0: Refining instance generators for '
                f'{self._inst_update_inner_rounds} inner rounds with '
                f'{len(reference_heuristics)} initialized H references ==='
            )
            for inner_round in range(1, self._inst_update_inner_rounds + 1):
                print(
                    f'[AdvEoH] Phase 0-B: Evolving instances inner round '
                    f'{inner_round}/{self._inst_update_inner_rounds}...'
                )
                self._evolve_instances(0, reference_heuristics=reference_heuristics)
                self._inst_hof.update(self._inst_pop)
                # --- FMA: collect failure analysis after instance evolution ---
                if self._use_fma:
                    self._fma_collect_failure_analysis()
                if not self._continue_loop():
                    print(f'[AdvEoH] Reached max_sample_nums={self._max_sample_nums} during initial G refinement.')
                    break

        # === 4. Async adversarial loop ===
        phase_id = 1
        phase_instances = self._build_phase_instances()
        phase_best = float('-inf')
        no_improve_count = 0
        phase_heu_gens = 0
        inst_gen = 0
        start_gen = 1
        if self._resume_mode:
            start_gen = self._resume_heu_generation + 1
            phase_id = self._resume_phase_id
            phase_best = self._resume_phase_best
            no_improve_count = self._resume_no_improve_count
            phase_heu_gens = self._resume_phase_heu_gens
            inst_gen = self._resume_inst_generation
            phase_instances = self._build_phase_instances()
            print(
                f'[AdvEoH] === Resuming from H Gen {self._resume_heu_generation}, '
                f'G Gen {self._resume_inst_generation}; next H Gen {start_gen} ==='
            )

        for gen in range(start_gen, self._max_generations + 1):
            print(f'\n[AdvEoH] === H Generation {gen}/{self._max_generations} | Phase {phase_id} ===')
            print('[AdvEoH] Phase A: Evolving heuristics on frozen phase instances...')
            self._evolve_heuristics(gen, instances=phase_instances)
            if not self._continue_loop():
                print(f'[AdvEoH] Reached max_sample_nums={self._max_sample_nums} after H Generation {gen}.')
                break
            phase_heu_gens += 1

            self._heu_hof.update(self._heu_pop)

            heldout = self._log_heldout(gen)
            print(f'[AdvEoH] Gen {gen} fixed/held-out score: {heldout}')
            if heldout is not None and heldout > phase_best + self._plateau_epsilon:
                phase_best = heldout
                no_improve_count = 0
            else:
                no_improve_count += 1

            if isinstance(self._profiler, AdvEoHProfiler):
                self._profiler.register_heuristic_generation(
                    self._heu_pop, gen, heldout
                )

            # --- FMA: collect and report failure analysis ---
            if self._use_fma:
                self._fma_collect_failure_analysis()
                if self._debug_mode and len(self._fma_memory) > 0:
                    top_f = self._fma_memory.most_threatening(3)
                    print(
                        f'  [FMA] Memory: {len(self._fma_memory)} modes, '
                        f'top strength: '
                        f'{", ".join(f"{m.description[:40]}... (s={m.strength:.2f})" for m in top_f[:2])}'
                    )

            plateau_reached = (
                phase_heu_gens >= self._min_heu_generations_per_phase and
                no_improve_count >= self._plateau_window
            )
            fixed_interval_reached = phase_heu_gens >= self._inst_update_interval
            update_g = plateau_reached if self._use_plateau_trigger else fixed_interval_reached
            if not update_g:
                continue

            if self._valid_inst_pop_size() < self._selection_num:
                print(
                    f'[AdvEoH] G update triggered, but instance population has only '
                    f'{self._valid_inst_pop_size()}/{self._selection_num} valid generators. '
                    f'Re-initializing until enough generators are available...'
                )
                self._init_instance_population(min_size=self._selection_num)
                self._inst_hof.update(self._inst_pop)

            reference_heuristics = self._build_heuristic_reference_set()
            trigger_msg = (
                f'plateau after {phase_heu_gens} H generations'
                if self._use_plateau_trigger
                else f'fixed interval of {self._inst_update_interval} H generations'
            )
            print(
                f'[AdvEoH] G update triggered by {trigger_msg}. '
                f'Updating G for {self._inst_update_inner_rounds} inner rounds '
                f'with {len(reference_heuristics)} frozen H references...'
            )
            for inner_round in range(1, self._inst_update_inner_rounds + 1):
                print(f'[AdvEoH] Phase B: Evolving instances inner round {inner_round}/{self._inst_update_inner_rounds}...')
                self._evolve_instances(gen, reference_heuristics=reference_heuristics)
                self._inst_hof.update(self._inst_pop)
                # --- FMA: collect failure analysis after instance evolution ---
                if self._use_fma:
                    self._fma_collect_failure_analysis()
                inst_gen += 1
                if isinstance(self._profiler, AdvEoHProfiler):
                    self._profiler.register_instance_generation(
                        self._inst_pop, inst_gen, trigger_heuristic_gen=gen
                    )
                if not self._continue_loop():
                    print(f'[AdvEoH] Reached max_sample_nums={self._max_sample_nums} during G update.')
                    break

            if not self._continue_loop():
                break

            phase_id += 1
            phase_instances = self._build_phase_instances()
            phase_best = float('-inf')
            no_improve_count = 0
            phase_heu_gens = 0
            print(f'[AdvEoH] Rebuilt frozen phase instance set for phase {phase_id}.')

        # === 5. Final report ===
        print('\n[AdvEoH] === Final held-out evaluation ===')
        final_heldout = self._log_heldout(self._max_generations)
        print(f'[AdvEoH] Final held-out score: {final_heldout}')
        if self._use_fma:
            print(f'[AdvEoH] FMA: {len(self._fma_memory)} failure modes accumulated.')
            top_threats = self._fma_memory.most_threatening(5)
            if top_threats:
                print('[AdvEoH] FMA: Top threats:')
                for i, m in enumerate(top_threats):
                    print(f'  {i + 1}. [{m.severity:.1f}] {m.description[:80]}... '
                          f'(strength={m.strength:.2f}, coverage={m.coverage})')
        self._record_best_heldout_function()

        self._finish()

    def _finish(self):
        try:
            self._evaluation_executor.shutdown(cancel_futures=True)
        except Exception:
            pass
        if self._profiler is not None:
            self._profiler.finish()
        self._sampler_heu.llm.close()
        self._sampler_inst.llm.close()
