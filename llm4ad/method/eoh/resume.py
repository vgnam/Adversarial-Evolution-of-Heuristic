from __future__ import annotations

import copy
import json
import os.path
import re
from typing import Any

from tqdm.auto import tqdm

from .eoh import EoH
from .profiler import EoHProfiler
from .population import Population
from ...base import TextFunctionProgramConverter as tfpc, Function


def _score_or_neg_inf(score: Any):
    return float('-inf') if score is None else score


def _pop_order(filename: str) -> int | None:
    match = re.fullmatch(r'pop_(\d+)\.json', filename)
    if not match:
        return None
    return int(match.group(1))


def _get_latest_pop_json(log_path: str):
    path = os.path.join(log_path, 'population')
    orders = [
        order
        for filename in os.listdir(path)
        if (order := _pop_order(filename)) is not None
    ]
    if not orders:
        raise FileNotFoundError(f'No population checkpoints found in {path}')
    max_o = max(orders)
    return os.path.join(path, f'pop_{max_o}.json'), max_o


def _get_all_samples_and_scores(path, get_algorithm=True):
    file_dir = os.path.join(path, 'samples')
    # get all file directories
    all_files = os.listdir(file_dir)
    # filter `samples_*.json` files and ignore best snapshots
    sample_files = [
        f for f in all_files
        if f.startswith('samples_') and 'best' not in f
    ]

    def extract_number(filename):
        # match the first number of the filename
        match = re.search(r'samples_(\d+)~', filename)
        if match:
            return int(match.group(1))
        return 0

    sorted_files = sorted(sample_files, key=extract_number)

    all_func = []
    all_score = []
    all_algorithm = []
    max_o = 0  # the max sample orders

    for file in sorted_files:
        file_path = os.path.join(file_dir, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            samples = json.load(f)
            for sample in samples:
                func = sample.get('function') or sample.get('program')
                if not func:
                    continue
                acc = _score_or_neg_inf(sample.get('score'))
                all_func.append(func)
                all_score.append(acc)
                all_algorithm.append(sample.get('algorithm', ''))
                max_o = max(max_o, sample.get('sample_order', 0))

    if get_algorithm:
        return all_func, all_score, max_o, all_algorithm
    return all_func, all_score, max_o


# def _get_all_samples_and_scores(path):
#     path = os.path.join(path, 'samples')
#
#     def path_to_int(path):
#         num = int(path.split('.')[0].split('_')[1])
#         return num
#
#     all_func = []
#     all_score = []
#     dirs = list(os.listdir(path))
#     dirs = sorted(dirs, key=path_to_int)
#     max_o = path_to_int(dirs[-1])
#
#     for dir in dirs:
#         file_name = os.path.join(path, dir)
#         with open(file_name, 'r') as f:
#             sample = json.load(f)
#         func = sample['function']
#         acc = sample['score'] if sample['score'] else float('-inf')
#         all_func.append(func)
#         all_score.append(acc)
#
#     return all_func, all_score, max_o


def _resume_pop(log_path: str, pop_size) -> Population:
    path, max_gen = _get_latest_pop_json(log_path)
    print(f'RESUME EoH: Generations: {max_gen}.', flush=True)
    with open(path, 'r') as f:
        data = json.load(f)
    funcs = []
    for d in data:
        func = d['function']
        func = tfpc.text_to_function(func)
        if func is None:
            continue
        score = _score_or_neg_inf(d.get('score'))
        algorithm = d.get('algorithm', '')
        func.score = score
        func.algorithm = algorithm
        func.operator = d.get('operator', getattr(func, 'operator', 'Unknown'))
        funcs.append(func)
    return Population(pop_size=pop_size, generation=max_gen, pop=funcs)


def _resume_text2func(f, s, template_func: Function):
    temp = copy.deepcopy(template_func)
    f = tfpc.text_to_function(f)
    if f is None:
        temp.body = '    pass'
        temp.score = None
        return temp
    else:
        f.score = s
        return f


def _resume_pf(log_path: str, pf: EoHProfiler, template_func):
    if pf is None:
        return
    _, db_max_order = _get_latest_pop_json(log_path)
    funcs, scores, sample_max_order, algorithms = _get_all_samples_and_scores(log_path)
    print(f'RESUME EoH: Sample order: {sample_max_order}.', flush=True)
    if hasattr(pf, '_cur_gen'):
        pf._cur_gen = db_max_order
    for i in tqdm(range(len(funcs)), desc='Resume EoH Profiler'):  # noqa
        f, s, algo = funcs[i], scores[i], algorithms[i]
        f = _resume_text2func(f, s, template_func)
        f.algorithm = algo
        pf.register_function(f, resume_mode=True)
    pf._num_samples = sample_max_order


def resume_eoh(eoh: EoH, path):
    eoh._resume_mode = True
    pf = eoh._profiler
    log_path = path
    # resume program database
    pop = _resume_pop(log_path, eoh._pop_size)
    eoh._population = pop
    # resume profiler
    template_func = eoh._function_to_evolve
    _resume_pf(log_path, pf, template_func)
    # resume eoh
    _, _, sample_max_order, _ = _get_all_samples_and_scores(log_path)
    eoh._tot_sample_nums = sample_max_order
