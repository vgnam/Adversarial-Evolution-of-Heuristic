from __future__ import annotations

import concurrent.futures
import copy
import time
import traceback
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
    plus the best held-out heuristic from a previous run's logs.

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

        # --- Pass parameters to profiler ---
        if profiler is not None:
            self._profiler.record_parameters(llm_heu, heu_evaluation, self)

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
            reference_heuristics or self._build_heuristic_reference_set()
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

        func.score = inst_fitness
        func.evaluate_time = 0.0
        func.algorithm = thought
        func.sample_time = sample_time
        func.operator = operator
        func.instances = instances
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

    def _score_instances_against_reference(
            self,
            instances: list,
            reference_heuristics: list[Function],
    ) -> float | None:
        """Minimax generator score: -mean_seed(max_h score(h, instance_seed))."""
        if not instances or not reference_heuristics:
            return None

        per_seed_best_scores = []
        for inst in instances:
            seed_scores = []
            for heu_func in reference_heuristics:
                score = self._evaluate_heuristic_function(heu_func, instances=[inst])
                if score is not None:
                    seed_scores.append(score)
            if seed_scores:
                per_seed_best_scores.append(max(seed_scores))

        if not per_seed_best_scores:
            return None
        return -float(np.mean(per_seed_best_scores))

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
            score = self._score_instances_against_reference(instances, reference_heuristics)
            if score is not None:
                inst_func.score = score
            else:
                inst_func.score = float('-inf')

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

    def _evolve_instances(self, gen: int, reference_heuristics: list[Function] | None = None):
        """Phase B: evolve instance generators against a frozen heuristic reference set."""
        reference_heuristics = reference_heuristics or self._build_heuristic_reference_set()
        if not reference_heuristics:
            return
        self._reevaluate_inst_pop(reference_heuristics)
        self._inst_pop.update_weights()

        opponent_descs = self._heu_hof.get_top_k_descriptions(self._n_opponent_desc)
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
