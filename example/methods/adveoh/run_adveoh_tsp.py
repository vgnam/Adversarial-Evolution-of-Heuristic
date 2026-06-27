"""Example: run AdvEoH on the TSP (constructive) task.

This demonstrates the generic adapter pattern: swap the two evaluations to
run AdvEoH on a different task — no method code changes needed.
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.tools.llm.llm_api_https import HttpsApi
from llm4ad.method.adveoh import AdvEoH
from llm4ad.method.adveoh.profiler import AdvEoHProfiler
from llm4ad.method.adveoh.adapters.tsp_construct import (
    AdvTSPEvaluation, TSPInstanceGenEvaluation,
)

DEFAULT_TRAIN_SIZE = 20


def parse_args():
    parser = argparse.ArgumentParser(description='Run AdvEoH on fixed-size TSP data.')
    parser.add_argument(
        '--train-size',
        type=int,
        default=DEFAULT_TRAIN_SIZE,
        help='Use tsp_construct/train_datasets/size_<N>.pkl as the fixed train set.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Heuristic LLM ---
    llm_heu = HttpsApi(host='api.bltcy.ai',
                       key='sk-xxx',
                       model='gemini-2.5-flash',
                       timeout=120)

    # --- Instance LLM ---
    llm_inst = HttpsApi(host='api.bltcy.ai',
                        key='sk-xxx',
                        model='gpt-4o-mini',
                        timeout=120)

    # --- Task-specific evaluations ---
    heu_evaluation = AdvTSPEvaluation(
        load_from_file=True,
        dataset_split='train',
        dataset_size=args.train_size,
    )
    inst_evaluation = TSPInstanceGenEvaluation(problem_size=args.train_size)

    # --- Method ---
    method = AdvEoH(
        llm_heu=llm_heu,
        llm_inst=llm_inst,
        heu_evaluation=heu_evaluation,
        inst_evaluation=inst_evaluation,
        profiler=AdvEoHProfiler(log_dir=f'logs/adveoh_tsp_size_{args.train_size}', log_style='simple'),
        max_generations=10,
        pop_size_heu=10,
        pop_size_inst=10,
        samples_per_gen_heu=8,
        samples_per_gen_inst=4,
        n_inst_sample=5,
        n_seed=3,
        n_opponent_desc=3,
        num_evaluators=4,
        debug_mode=False,
    )

    method.run()


if __name__ == '__main__':
    main()
