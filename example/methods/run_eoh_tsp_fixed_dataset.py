"""Example: run EoH on TSP using fixed train/test datasets.

Training loads one fixed-size dataset from ``train_dataset.pkl``. Testing can
load multiple held-out datasets with different sizes from
``test_datasets/size_<N>.pkl``.

Prerequisite:
    python llm4ad/task/optimization/generate_fixed_datasets.py --task tsp_construct --split all

The single-line difference vs the original ``run_eoh_tsp.py`` is:
    task = TSPEvaluation(load_from_file=True, dataset_split='train')

After training finishes, the best heuristic is evaluated on all fixed test
datasets found in ``test_datasets/size_*.pkl``.
"""
import json
import os
import sys

sys.path.append('../../../')  # for finding all the modules

import llm4ad.task.optimization.tsp_construct as tsp_construct
from llm4ad.base import SecureEvaluator
from llm4ad.task.optimization.tsp_construct import TSPEvaluation
from llm4ad.task.optimization._dataset_loader import list_test_dataset_sizes
from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.eoh import EoH
from llm4ad.tools.profiler import ProfilerBase


def test_best_on_full_dataset(best_program, profiler: ProfilerBase | None = None) -> dict[int, float | None]:
    task_folder = os.path.dirname(tsp_construct.__file__)
    test_sizes = list_test_dataset_sizes(task_folder)
    if not test_sizes:
        print('[Test] No fixed test datasets found. Run generate_fixed_datasets.py --task tsp_construct --split test')
        return {}

    print('\n[Test] Evaluating best heuristic on full fixed test dataset...')
    results = {}
    for size in test_sizes:
        test_task = TSPEvaluation(load_from_file=True, dataset_split='test', dataset_size=size)
        score = SecureEvaluator(test_task).evaluate_program(best_program)
        results[size] = score
        print(f'[Test] size={size}: score={score}')

    valid_scores = [score for score in results.values() if score is not None]
    if valid_scores:
        print(f'[Test] mean score across sizes: {sum(valid_scores) / len(valid_scores)}')

    log_dir = getattr(profiler, '_log_dir', None) if profiler is not None else None
    if log_dir:
        out_path = os.path.join(log_dir, 'full_test_results.json')
        with open(out_path, 'w') as f:
            json.dump({str(size): score for size, score in results.items()}, f, indent=4)
        print(f'[Test] results saved to {out_path}')

    return results


def main():
    api_key = os.environ['LLM_API_KEY']
    llm = HttpsApi(host=os.environ.get('LLM_HOST', 'api.bltcy.ai'),
                   key=api_key,
                   model=os.environ.get('LLM_MODEL', 'gemini-2.5-flash'),
                   timeout=120)

    # --- Train on the single fixed-size training dataset ---
    task = TSPEvaluation(load_from_file=True, dataset_split='train')
    profiler = ProfilerBase(log_dir='logs/eoh_tsp_fixed', log_style='simple')

    method = EoH(llm=llm,
                 profiler=profiler,
                 evaluation=task,
                 max_sample_nums=100,
                 max_generations=None,
                 pop_size=16,
                 num_samplers=8,
                 num_evaluators=8,
                 debug_mode=False)

    method.run()
    if method.best_program is None:
        print('[Test] No valid heuristic was found during training; skip full test.')
        return
    test_best_on_full_dataset(method.best_program, profiler)


if __name__ == '__main__':
    main()
