"""Evaluate logged TSP heuristics on explicit mixed-OOD dataset files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.base import SecureEvaluator
from llm4ad.task.optimization.fixed_test_eval import _load_best_record
from llm4ad.task.optimization.tsp_construct import TSPEvaluation


DEFAULT_DATASET_DIR = Path(__file__).resolve().parent / 'ood_test_datasets' / 'mixture'
DEFAULT_RUNS = {
    'EoH': (
        'logs/20260625_141515_TSPEvaluation_EoH_split_run1',
        'logs/20260625_141515_TSPEvaluation_EoH_split_run2',
        'logs/20260625_141519_TSPEvaluation_EoH',
    ),
    'AdvEoH': (
        'logs/20260624_231534_TSPEvaluation_AdvEoH',
        'logs/20260624_231537_TSPEvaluation_AdvEoH',
        'logs/20260624_231540_TSPEvaluation_AdvEoH',
    ),
    'MCTS_AHD': (
        'logs/20260624_203220_TSPEvaluation_MCTS_AHD',
        'logs/20260624_203225_TSPEvaluation_MCTS_AHD',
        'logs/20260624_203230_TSPEvaluation_MCTS_AHD',
    ),
}


# NetworkX computes zero unweighted betweenness for every node in a complete
# graph. This replacement removes that no-op O(n^3+) calculation from one
# known logged heuristic while preserving its node-ranking expression.
SLOW_MCTS_SHA256 = '9253142a21f3f8676b350b51d158fde8fe382c78681c6f2125548718013a686e'
OPTIMIZED_EQUIVALENT_MCTS = '''\
import numpy as np
def select_next_node(current_node: int, destination_node: int, unvisited_nodes: np.ndarray, distance_matrix: np.ndarray) -> int:
    import numpy as np

    total_nodes = distance_matrix.shape[0]
    progress_ratio = (total_nodes - len(unvisited_nodes)) / total_nodes
    sub_mat = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    max_sub_distance = np.max(sub_mat) + 1e-12
    eccentricities = np.max(sub_mat, axis=1)
    node_importance = 0.4 * eccentricities / max_sub_distance

    distances_to_current = distance_matrix[current_node, unvisited_nodes]
    inv_dists = 1 / (distances_to_current + 1e-12)
    probs_diversity = inv_dists / np.sum(inv_dists)
    entropy = -np.sum(probs_diversity * np.log(probs_diversity + 1e-12))
    max_entropy = np.log(len(unvisited_nodes) + 1e-12)

    weight_explore = max(0.1, 0.6 - 0.5 * progress_ratio)
    weight_importance = min(0.8, 0.3 + 0.5 * progress_ratio)
    weight_diversity = max(0, min(1, 1 - weight_explore - weight_importance))
    max_dist_current = np.max(distances_to_current) + 1e-12
    diversity_score = entropy / max_entropy

    scores = (
        weight_explore * distances_to_current / max_dist_current
        + weight_importance * (1 - node_importance)
        + weight_diversity * (1 - diversity_score)
    )
    return unvisited_nodes[int(np.argmin(scores))]
'''


def _parse_run(value: str) -> tuple[str, Path]:
    try:
        method, path = value.split('=', 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('run must have the form METHOD=LOG_DIR') from exc
    return method, Path(path)


def _resolve_runs(run_args: list[tuple[str, Path]] | None) -> dict[str, list[Path]]:
    runs: dict[str, list[Path]] = defaultdict(list)
    if run_args:
        for method, path in run_args:
            runs[method].append(path if path.is_absolute() else REPO_ROOT / path)
    else:
        for method, paths in DEFAULT_RUNS.items():
            runs[method].extend(REPO_ROOT / path for path in paths)
    return dict(runs)


def _program_for_run(log_dir: Path) -> tuple[dict[str, Any], str, bool]:
    record = _load_best_record(str(log_dir))
    if record is None:
        raise FileNotFoundError(f'No best heuristic record found in {log_dir}')
    program = record.get('program') or record.get('function')
    if not program:
        raise ValueError(f'Best record has no program in {log_dir}')
    digest = hashlib.sha256(program.encode()).hexdigest()
    if digest == SLOW_MCTS_SHA256:
        return record, OPTIMIZED_EQUIVALENT_MCTS, True
    return record, program, False


def _dataset_sizes(dataset_dir: Path) -> list[int]:
    sizes = []
    for path in dataset_dir.glob('size_*.pkl'):
        try:
            sizes.append(int(path.stem.removeprefix('size_')))
        except ValueError:
            continue
    return sorted(sizes)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    methods = sorted({row['method'] for row in results})
    sizes = sorted({row['size'] for row in results})
    for method in methods:
        method_rows = [row for row in results if row['method'] == method]
        by_size = []
        for size in sizes:
            lengths = [
                row['tour_length'] for row in method_rows
                if row['size'] == size and row['tour_length'] is not None
            ]
            by_size.append({
                'size': size,
                'mean_tour_length': statistics.fmean(lengths) if lengths else None,
                'stdev_tour_length': statistics.stdev(lengths) if len(lengths) > 1 else 0.0 if lengths else None,
                'valid_runs': len(lengths),
                'total_runs': sum(row['size'] == size for row in method_rows),
            })
        all_lengths = [row['tour_length'] for row in method_rows if row['tour_length'] is not None]
        run_means = []
        for run in sorted({row['run'] for row in method_rows}):
            run_lengths = [
                row['tour_length'] for row in method_rows
                if row['run'] == run and row['tour_length'] is not None
            ]
            if run_lengths:
                run_means.append({'run': run, 'mean_tour_length': statistics.fmean(run_lengths)})
        aggregate[method] = {
            'by_size': by_size,
            'run_means': run_means,
            'overall_mean_tour_length': statistics.fmean(all_lengths) if all_lengths else None,
            'valid_evaluations': len(all_lengths),
            'total_evaluations': len(method_rows),
        }
    return aggregate


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        '# TSP OOD evaluation',
        '',
        f"- Distribution: {summary.get('distribution', 'unknown')}",
        f"- Instances per size: {summary['n_instances']}",
        f"- Sizes: {', '.join(map(str, summary['sizes']))}",
        '- Metric: mean tour length (lower is better).',
        '- Aggregation: arithmetic mean over three independent runs.',
        '',
        '| Method | ' + ' | '.join(f'n={size}' for size in summary['sizes']) + ' | Overall mean |',
        '|---|' + '|'.join('---:' for _ in summary['sizes']) + '|---:|',
    ]
    for method, item in summary['aggregate'].items():
        cells = []
        by_size = {row['size']: row for row in item['by_size']}
        for size in summary['sizes']:
            row = by_size[size]
            mean = row['mean_tour_length']
            stdev = row['stdev_tour_length']
            cells.append('FAILED' if mean is None else f'{mean:.6f} +/- {stdev:.6f}')
        overall = item['overall_mean_tour_length']
        lines.append(
            f"| {method} | " + ' | '.join(cells) + f" | {overall:.6f} |"
        )
    metadata = summary.get('dataset_metadata') or {}
    datasets = metadata.get('datasets') or []
    if datasets:
        lines.extend(['', '## Distribution composition', ''])
        for dataset in datasets:
            counts = dataset.get('distribution_counts') or {}
            count_text = ', '.join(f'{name}={count}' for name, count in counts.items())
            lines.append(f"- n={dataset.get('size')}: {count_text}")
    lines.extend(['', '## Run means', ''])
    for method, item in summary['aggregate'].items():
        lines.append(f'### {method}')
        lines.append('')
        for row in item['run_means']:
            lines.append(f"- `{row['run']}`: {row['mean_tour_length']:.6f}")
        lines.append('')
    optimized = [row for row in summary['results'] if row['optimized_equivalent']]
    if optimized:
        lines.extend([
            '## Evaluation note',
            '',
            'The known MCTS sample 320 uses unweighted betweenness centrality on a complete graph,',
            'which is identically zero but expensive. Its behavior-equivalent vectorized form was',
            'used to avoid timeouts; the source-program SHA-256 is recorded in the JSON report.',
            '',
        ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate logged heuristics on TSP OOD data.')
    parser.add_argument('--dataset-dir', type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--n-instances', type=int, default=64)
    parser.add_argument('--timeout-seconds', type=float, default=240.0)
    parser.add_argument('--run', action='append', type=_parse_run, help='METHOD=LOG_DIR; repeat as needed')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    sizes = _dataset_sizes(dataset_dir)
    if not sizes:
        parser.error(f'no size_*.pkl datasets found in {dataset_dir}')
    runs = _resolve_runs(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for method, log_dirs in runs.items():
        for log_dir in log_dirs:
            record, program, optimized = _program_for_run(log_dir)
            for size in sizes:
                dataset_path = dataset_dir / f'size_{size}.pkl'
                task = TSPEvaluation(
                    timeout_seconds=args.timeout_seconds,
                    n_instance=args.n_instances,
                    problem_size=size,
                    load_from_file=True,
                    dataset_file=str(dataset_path),
                )
                # These are previously logged local heuristics, not untrusted
                # responses. Running them in-process avoids Windows spawn
                # overhead/deadlocks across the 36 fixed benchmark evaluations.
                task.safe_evaluate = False
                started = time.perf_counter()
                score = SecureEvaluator(task).evaluate_program(program)
                elapsed = time.perf_counter() - started
                score = float(score) if isinstance(score, (int, float, np.number)) and math.isfinite(score) else None
                row = {
                    'method': method,
                    'run': log_dir.name,
                    'log_dir': str(log_dir),
                    'best_sample_order': record.get('sample_order'),
                    'size': size,
                    'score': score,
                    'tour_length': -score if score is not None else None,
                    'elapsed_seconds': elapsed,
                    'optimized_equivalent': optimized,
                    'source_program_sha256': hashlib.sha256(
                        (record.get('program') or record.get('function')).encode()
                    ).hexdigest(),
                }
                results.append(row)
                value = 'FAILED' if score is None else f'{-score:.6f}'
                print(f'[OOD] {method:8s} {log_dir.name} n={size}: {value} ({elapsed:.2f}s)')

    metadata_path = dataset_dir / 'metadata.json'
    metadata = (
        json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata_path.is_file() else None
    )
    summary = {
        'distribution': (
            metadata.get('distribution', 'explicit_ood')
            if metadata else 'explicit_ood'
        ),
        'dataset_dir': str(dataset_dir),
        'dataset_metadata': metadata,
        'n_instances': args.n_instances,
        'sizes': sizes,
        'timeout_seconds': args.timeout_seconds,
        'results': results,
        'aggregate': _aggregate(results),
    }
    json_path = args.output_dir / 'results.json'
    markdown_path = args.output_dir / 'report.md'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_markdown(summary, markdown_path)
    print(f'[OOD] JSON report: {json_path}')
    print(f'[OOD] Markdown report: {markdown_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
