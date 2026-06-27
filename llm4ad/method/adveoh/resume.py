from __future__ import annotations

import copy
import json
import math
import os
import re
from typing import Any

from tqdm.auto import tqdm

from .adveoh import AdvEoH
from .population import HallOfFame, Population
from .profiler import AdvEoHProfiler
from ...base import Function, TextFunctionProgramConverter as tfpc


def _score_is_valid(score: Any) -> bool:
    try:
        return score is not None and math.isfinite(float(score))
    except (TypeError, ValueError):
        return False


def _pop_order(filename: str) -> int | None:
    match = re.fullmatch(r'pop_(\d+)\.json', filename)
    if not match:
        return None
    return int(match.group(1))


def _list_pop_jsons(log_path: str, dirname: str) -> list[tuple[int, str]]:
    pop_dir = os.path.join(log_path, dirname)
    if not os.path.isdir(pop_dir):
        return []
    paths = []
    for filename in os.listdir(pop_dir):
        order = _pop_order(filename)
        if order is not None:
            paths.append((order, os.path.join(pop_dir, filename)))
    return sorted(paths, key=lambda item: item[0])


def _latest_pop_json(log_path: str, dirname: str) -> tuple[int, str] | None:
    paths = _list_pop_jsons(log_path, dirname)
    return paths[-1] if paths else None


def _load_records(path: str) -> list[dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        records = json.load(f)
    return records if isinstance(records, list) else []


def _record_to_function(record: dict[str, Any], template_func: Function) -> Function:
    func_text = record.get('function') or record.get('program') or ''
    func = tfpc.text_to_function(func_text)
    if func is None:
        func = copy.deepcopy(template_func)
        func.body = '    pass'
    func.score = record.get('score')
    func.algorithm = record.get('algorithm', '')
    func.operator = record.get('operator', 'Unknown')
    return func


def _load_population_file(
        path: str,
        pop_size: int,
        generation: int,
        template_func: Function,
        eta: float,
) -> Population:
    funcs = [
        _record_to_function(record, template_func)
        for record in _load_records(path)
    ]
    return Population(pop_size=pop_size, generation=generation, pop=funcs, eta=eta)


def _restore_instance_cache(
        adveoh: AdvEoH,
        funcs: list[Function],
        desc: str,
        cache: dict[str, list] | None = None,
):
    cache = cache if cache is not None else {}
    for func in tqdm(funcs, desc=desc):  # noqa
        cache_key = str(func)
        if cache_key in cache:
            func.instances = cache[cache_key]
            continue

        program = tfpc.function_to_program(func, adveoh._template_program_inst)
        if program is None:
            func.instances = []
            cache[cache_key] = func.instances
            continue

        instances = []
        for seed in range(adveoh._n_seed):
            inst_data, _ = adveoh._evaluation_executor.submit(
                adveoh._inst_evaluator.evaluate_program_record_time,
                program,
                seed=seed,
            ).result()
            if inst_data is not None:
                instances.append(inst_data)
        func.instances = instances
        cache[cache_key] = instances


def _load_latest_population(
        adveoh: AdvEoH,
        log_path: str,
        dirname: str,
        pop_size: int,
        template_func: Function,
        eta: float,
        *,
        required: bool,
        instance_cache: dict[str, list] | None = None,
        restore_instances: bool = False,
) -> tuple[Population | None, int]:
    latest = _latest_pop_json(log_path, dirname)
    if latest is None:
        if required:
            raise FileNotFoundError(
                f'No {dirname} checkpoints found under {log_path}'
            )
        return None, 0

    generation, path = latest
    pop = _load_population_file(path, pop_size, generation, template_func, eta)
    if restore_instances and dirname == 'instance_pop':
        _restore_instance_cache(
            adveoh,
            pop.population,
            f'Resume AdvEoH {dirname} instances',
            instance_cache,
        )
    return pop, generation


def _rebuild_hof(
        adveoh: AdvEoH,
        log_path: str,
        dirname: str,
        pop_size: int,
        template_func: Function,
        eta: float,
        hof: HallOfFame,
        instance_cache: dict[str, list] | None = None,
        restore_instances: bool = False,
) -> HallOfFame:
    max_gen = getattr(hof, '_max_gen', 5)
    top_k = getattr(hof, '_top_k', 5)
    restored = HallOfFame(max_gen=max_gen, top_k=top_k)
    checkpoint_paths = _list_pop_jsons(log_path, dirname)[-max_gen:]

    for generation, path in checkpoint_paths:
        pop = _load_population_file(path, pop_size, generation, template_func, eta)
        if restore_instances and dirname == 'instance_pop':
            top_funcs = sorted(
                pop.population,
                key=lambda func: func.score if func.score is not None else float('-inf'),
                reverse=True,
            )[:top_k]
            _restore_instance_cache(
                adveoh,
                top_funcs,
                f'Resume AdvEoH HoF {dirname} gen {generation}',
                instance_cache,
            )
        restored.update(pop)

    return restored


def _sample_files(log_path: str) -> list[str]:
    sample_dir = os.path.join(log_path, 'samples')
    if not os.path.isdir(sample_dir):
        return []

    def order(filename: str) -> int:
        match = re.search(r'samples_(\d+)~', filename)
        return int(match.group(1)) if match else 0

    files = [
        filename for filename in os.listdir(sample_dir)
        if filename.startswith('samples_') and 'best' not in filename
    ]
    return [os.path.join(sample_dir, filename) for filename in sorted(files, key=order)]


def _load_sample_records(log_path: str) -> list[dict[str, Any]]:
    records = []
    for path in _sample_files(log_path):
        try:
            records.extend(_load_records(path))
        except (OSError, json.JSONDecodeError):
            continue
    return records


def _resume_profiler(log_path: str, profiler: AdvEoHProfiler | None):
    if profiler is None:
        return 0

    records = _load_sample_records(log_path)
    max_order = max((record.get('sample_order', 0) for record in records), default=0)
    profiler._num_samples = max_order
    profiler._evaluate_success_program_num = sum(
        1 for record in records if record.get('score') is not None
    )
    profiler._evaluate_failed_program_num = sum(
        1 for record in records if record.get('score') is None
    )

    if hasattr(profiler, '_best_role_score'):
        profiler._best_role_score = {
            'heuristic': float('-inf'),
            'instance': float('-inf'),
        }
        profiler._best_role_sample_order = {
            'heuristic': None,
            'instance': None,
        }
        for record in records:
            role = record.get('role', 'heuristic')
            if role not in profiler._best_role_score:
                continue
            score = record.get('score')
            if _score_is_valid(score) and score > profiler._best_role_score[role]:
                profiler._best_role_score[role] = score
                profiler._best_role_sample_order[role] = record.get('sample_order')

    if hasattr(profiler, 'load_token_usage_log'):
        profiler.load_token_usage_log()

    print(f'RESUME AdvEoH: Sample order: {max_order}.', flush=True)
    return max_order


def _load_generation_metrics(log_path: str) -> list[dict[str, Any]]:
    path = os.path.join(log_path, 'generation_metrics.json')
    if not os.path.isfile(path):
        return []
    try:
        return _load_records(path)
    except (OSError, json.JSONDecodeError):
        return []


def _resume_loop_state(adveoh: AdvEoH, log_path: str, heu_generation: int, inst_generation: int):
    metrics = _load_generation_metrics(log_path)
    h_events = [
        item for item in metrics
        if item.get('event') == 'heuristic_generation'
        or (item.get('heu_generation') is not None and item.get('inst_generation') is None)
    ]
    i_events = [
        item for item in metrics
        if item.get('event') == 'instance_generation'
        or item.get('inst_generation') is not None
    ]

    if h_events:
        heu_generation = max(int(item.get('heu_generation') or item.get('generation') or 0) for item in h_events)
    if i_events:
        inst_generation = max(int(item.get('inst_generation') or item.get('generation') or 0) for item in i_events)

    trigger_gens = sorted({
        int(item.get('heu_generation') or 0)
        for item in i_events
        if item.get('heu_generation') is not None
    })
    last_trigger_gen = trigger_gens[-1] if trigger_gens else 0
    phase_events = [
        item for item in h_events
        if int(item.get('heu_generation') or item.get('generation') or 0) > last_trigger_gen
    ]

    phase_best = float('-inf')
    no_improve_count = 0
    for item in sorted(
            phase_events,
            key=lambda event: int(event.get('heu_generation') or event.get('generation') or 0),
    ):
        heldout = item.get('heldout_score')
        if _score_is_valid(heldout) and heldout > phase_best + adveoh._plateau_epsilon:
            phase_best = heldout
            no_improve_count = 0
        else:
            no_improve_count += 1

    adveoh._resume_heu_generation = heu_generation
    adveoh._resume_inst_generation = inst_generation
    adveoh._resume_phase_id = len(trigger_gens) + 1
    adveoh._resume_phase_heu_gens = len(phase_events)
    adveoh._resume_phase_best = phase_best
    adveoh._resume_no_improve_count = no_improve_count

    heldout_scores = [
        item.get('heldout_score')
        for item in h_events
        if _score_is_valid(item.get('heldout_score'))
    ]
    if heldout_scores:
        adveoh._best_heldout_score = max(heldout_scores)
        valid_current = [
            func for func in adveoh._heu_pop.population
            if _score_is_valid(func.score)
        ]
        if valid_current:
            adveoh._best_heldout_function = copy.deepcopy(
                max(valid_current, key=lambda func: func.score)
            )


def resume_adveoh(
        adveoh: AdvEoH,
        path: str,
        *,
        restore_instances: bool = False,
        evaluate_heldout: bool = False,
):
    """Resume an AdvEoH run from a log directory.

    The latest heuristic and instance populations are restored from checkpoint
    JSON files. By default, instance generator outputs are rebuilt lazily when
    the resumed run first needs them. Pass ``restore_instances=True`` to rebuild
    those caches immediately during resume. Pass ``evaluate_heldout=True`` to
    refresh the current fixed/held-out score immediately.
    """
    log_path = path
    adveoh._resume_mode = True
    adveoh._skip_initial_heuristic_population_once = True
    adveoh._skip_initial_instance_population_once = True
    instance_cache: dict[str, list] = {}

    heu_pop, heu_generation = _load_latest_population(
        adveoh,
        log_path,
        'heuristic_pop',
        adveoh._pop_size_heu,
        adveoh._function_to_evolve_heu,
        getattr(adveoh._heu_pop, 'eta', adveoh._mwu_eta_heu),
        required=True,
    )
    inst_pop, inst_generation = _load_latest_population(
        adveoh,
        log_path,
        'instance_pop',
        adveoh._pop_size_inst,
        adveoh._function_to_evolve_inst,
        getattr(adveoh._inst_pop, 'eta', adveoh._mwu_eta_inst),
        required=False,
        instance_cache=instance_cache,
        restore_instances=restore_instances,
    )

    adveoh._heu_pop = heu_pop
    if inst_pop is not None:
        adveoh._inst_pop = inst_pop

    adveoh._heu_hof = _rebuild_hof(
        adveoh,
        log_path,
        'heuristic_pop',
        adveoh._pop_size_heu,
        adveoh._function_to_evolve_heu,
        getattr(adveoh._heu_pop, 'eta', adveoh._mwu_eta_heu),
        adveoh._heu_hof,
    )
    adveoh._inst_hof = _rebuild_hof(
        adveoh,
        log_path,
        'instance_pop',
        adveoh._pop_size_inst,
        adveoh._function_to_evolve_inst,
        getattr(adveoh._inst_pop, 'eta', adveoh._mwu_eta_inst),
        adveoh._inst_hof,
        instance_cache=instance_cache,
        restore_instances=restore_instances,
    )

    sample_order = _resume_profiler(
        log_path,
        adveoh._profiler if isinstance(adveoh._profiler, AdvEoHProfiler) else None,
    )
    adveoh._tot_sample_nums = sample_order

    _resume_loop_state(adveoh, log_path, heu_generation, inst_generation)

    heldout = None
    if evaluate_heldout:
        heldout = adveoh._log_heldout(adveoh._resume_heu_generation)
    elif _score_is_valid(adveoh._best_heldout_score):
        heldout = adveoh._best_heldout_score

    print(
        f'RESUME AdvEoH: H generations={adveoh._resume_heu_generation}, '
        f'G generations={adveoh._resume_inst_generation}, '
        f'best logged heldout={heldout}.',
        flush=True,
    )
