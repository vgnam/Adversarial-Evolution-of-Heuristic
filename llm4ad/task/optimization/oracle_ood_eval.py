"""Final-population Oracle@10 evaluation on constructive-task OOD datasets."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import pickle
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.base import Evaluation, SecureEvaluator


TASK_EVALUATIONS = {
    'tsp_construct': ('llm4ad.task.optimization.tsp_construct', 'TSPEvaluation'),
    'cvrp_construct': ('llm4ad.task.optimization.cvrp_construct', 'CVRPEvaluation'),
    'ovrp_construct': ('llm4ad.task.optimization.ovrp_construct', 'OVRPEvaluation'),
    'vrptw_construct': ('llm4ad.task.optimization.vrptw_construct', 'VRPTWEvaluation'),
    'bp_1d_construct': ('llm4ad.task.optimization.bp_1d_construct', 'BP1DEvaluation'),
    'bp_2d_construct': ('llm4ad.task.optimization.bp_2d_construct', 'BP2DEvaluation'),
    'knapsack_construct': ('llm4ad.task.optimization.knapsack_construct', 'KnapsackEvaluation'),
    'jssp_construct': ('llm4ad.task.optimization.jssp_construct', 'JSSPEvaluation'),
    'qap_construct': ('llm4ad.task.optimization.qap_construct', 'QAPEvaluation'),
    'cflp_construct': ('llm4ad.task.optimization.cflp_construct', 'CFLPEvaluation'),
    'set_cover_construct': ('llm4ad.task.optimization.set_cover_construct', 'SCPEvaluation'),
}
HERE = Path(__file__).resolve().parent


def _generation(path: Path) -> int:
    match = re.search(r'(\d+)$', path.stem)
    return int(match.group(1)) if match else -1


def _is_heuristic(record: dict[str, Any]) -> bool:
    role = record.get('role')
    program = record.get('program') or record.get('function') or ''
    return role in (None, 'heuristic') and 'def generate_instance(' not in program


def load_final_population(log_dir: str | Path, k: int = 10) -> tuple[Path, list[dict[str, Any]]]:
    """Load and deterministically select K unique heuristics from the last checkpoint."""
    log_dir = Path(log_dir)
    candidates: list[Path] = []
    for dirname in ('heuristic_pop', 'population'):
        candidates.extend((log_dir / dirname).glob('pop_*.json'))
    if not candidates:
        raise FileNotFoundError(f'No final population checkpoint under {log_dir}')
    checkpoint = max(candidates, key=lambda path: (_generation(path), path.stat().st_mtime_ns))
    records = json.loads(checkpoint.read_text(encoding='utf-8'))
    if not isinstance(records, list):
        raise ValueError(f'Population must be a JSON list: {checkpoint}')

    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, dict) or not _is_heuristic(record):
            continue
        program = record.get('program') or record.get('function')
        if not program:
            continue
        digest = hashlib.sha256(program.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            unique.append(record)

    def rank(record: dict[str, Any]) -> float:
        score = record.get('score')
        return float(score) if isinstance(score, (int, float)) and math.isfinite(score) else float('-inf')

    if len(unique) > k:
        unique = sorted(enumerate(unique), key=lambda pair: (-rank(pair[1]), pair[0]))
        unique = [record for _, record in unique[:k]]
    if len(unique) != k:
        raise ValueError(f'Oracle@{k} requires {k} unique final-population heuristics; found {len(unique)} in {checkpoint}')
    return checkpoint, unique


def _task_name(evaluation_cls: type[Evaluation]) -> str:
    folder = Path(inspect.getfile(evaluation_cls)).resolve().parent.name
    if folder not in TASK_EVALUATIONS:
        raise ValueError(f'OOD Oracle evaluation is not registered for {folder}')
    return folder


def _evaluation_class(task: str) -> type[Evaluation]:
    module_name, class_name = TASK_EVALUATIONS[task]
    return getattr(importlib.import_module(module_name), class_name)


def _clean_params(params: dict[str, Any]) -> dict[str, Any]:
    signature_exclusions = {'name', 'dataset_split', 'dataset_size', 'dataset_file', 'load_from_file', 'n_instance'}
    return {key: value for key, value in params.items() if key not in signature_exclusions}


def _dataset_files(task: str, suites: list[str] | None, sizes: list[int] | None) -> list[tuple[str, int, Path]]:
    root = HERE / task / 'ood_test_datasets'
    if not root.is_dir():
        suite_names = []
    else:
        suite_names = suites or sorted(path.name for path in root.iterdir() if path.is_dir())
    files = []
    for suite in suite_names:
        for path in sorted((root / suite).glob('size_*.pkl'), key=_generation):
            size = _generation(path)
            if sizes is None or size in sizes:
                files.append((suite, size, path))
    if not files:
        raise FileNotFoundError(f'No OOD datasets found for {task} under {root}')
    return files


def _numeric_score(value: Any) -> float | None:
    if isinstance(value, (int, float, np.number)):
        value = float(value)
        return value if math.isfinite(value) else None
    return None


def _evaluate_program_instances(
    evaluation_cls: type[Evaluation],
    evaluation_params: dict[str, Any],
    program: str,
    dataset_path: Path,
    safe_evaluate: bool,
) -> tuple[list[float | None], float]:
    with dataset_path.open('rb') as stream:
        instances = pickle.load(stream)
    params = _clean_params(evaluation_params)
    params.update(load_from_file=True, dataset_file=str(dataset_path), n_instance=len(instances))
    task = evaluation_cls(**params)
    task.safe_evaluate = safe_evaluate
    evaluator = SecureEvaluator(task)
    scores: list[float | None] = []
    started = time.perf_counter()
    for instance in instances:
        task._datasets = [instance]
        task.n_instance = 1
        scores.append(_numeric_score(evaluator.evaluate_program(program)))
    return scores, time.perf_counter() - started


def evaluate_final_population_oracle_ood(
    evaluation_cls: type[Evaluation],
    evaluation_params: dict[str, Any],
    log_dir: str | Path,
    method_name: str,
    *,
    k: int = 10,
    suites: list[str] | None = None,
    sizes: list[int] | None = None,
    safe_evaluate: bool = False,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate the best per-instance score among K final-population members."""
    task_name = _task_name(evaluation_cls)
    checkpoint, records = load_final_population(log_dir, k=k)
    datasets = _dataset_files(task_name, suites, sizes)
    programs = [record.get('program') or record.get('function') for record in records]
    rows = []
    for suite, size, dataset_path in datasets:
        matrix: list[list[float | None]] = []
        elapsed = 0.0
        for index, program in enumerate(programs):
            scores, seconds = _evaluate_program_instances(
                evaluation_cls, evaluation_params, program, dataset_path, safe_evaluate
            )
            matrix.append(scores)
            elapsed += seconds
            print(f'[Oracle@{k}] {method_name} {task_name}/{suite} n={size} individual={index + 1}/{k}')

        score_array = np.array([[np.nan if value is None else value for value in row] for row in matrix])
        oracle_values: list[float | None] = []
        winner_ids: list[int] = []
        for column in score_array.T:
            valid_ids = np.flatnonzero(~np.isnan(column))
            if not len(valid_ids):
                oracle_values.append(None)
                continue
            winner = int(valid_ids[np.argmax(column[valid_ids])])
            winner_ids.append(winner)
            oracle_values.append(float(column[winner]))
        valid = [value for value in oracle_values if value is not None]
        rows.append({
            'suite': suite, 'size': size, 'dataset': str(dataset_path),
            'n_instances': len(oracle_values), 'k': k,
            'oracle_mean_score': statistics.fmean(valid) if valid else None,
            'oracle_instance_scores': oracle_values,
            'winner_counts': {str(i): winner_ids.count(i) for i in sorted(set(winner_ids))},
            'failed_program_instance_evaluations': int(np.isnan(score_array).sum()),
            'elapsed_seconds': elapsed,
        })

    summary = {
        'metric': f'final-population Oracle@{k} (higher evaluation score is better)',
        'task': task_name, 'method': method_name, 'run': Path(log_dir).name,
        'checkpoint': str(checkpoint), 'k': k, 'rows': rows,
    }
    target = Path(output_dir) if output_dir else Path(log_dir) / 'eval' / f'ood_oracle_at_{k}'
    target.mkdir(parents=True, exist_ok=True)
    (target / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_report(summary, target / 'report.md')
    return summary


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        f"# {summary['task']} OOD Oracle@{summary['k']}", '',
        f"- Method: {summary['method']}", f"- Run: `{summary['run']}`",
        f"- Final population: `{summary['checkpoint']}`",
        '- Oracle selection is independent per OOD instance; higher score is better.', '',
        '| Suite | Size | Oracle mean score | Failed evaluations |',
        '|---|---:|---:|---:|',
    ]
    for row in summary['rows']:
        mean = row['oracle_mean_score']
        lines.append(f"| {row['suite']} | {row['size']} | {'FAILED' if mean is None else f'{mean:.6f}'} | {row['failed_program_instance_evaluations']} |")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def _aggregate_runs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({(row['suite'], row['size']) for summary in summaries for row in summary['rows']})
    methods = sorted({summary['method'] for summary in summaries})
    aggregate: dict[str, Any] = {}
    for method in methods:
        cells = []
        for suite, size in keys:
            values = [
                row['oracle_mean_score']
                for summary in summaries if summary['method'] == method
                for row in summary['rows']
                if row['suite'] == suite and row['size'] == size and row['oracle_mean_score'] is not None
            ]
            cells.append({
                'suite': suite, 'size': size, 'runs': len(values),
                'mean': statistics.fmean(values) if values else None,
                'stdev': statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
            })
        aggregate[method] = cells
    return {'metric': summaries[0]['metric'] if summaries else None, 'methods': methods, 'cells': aggregate}


def _write_comparison(comparison: dict[str, Any], path: Path) -> None:
    keys = sorted({(cell['suite'], cell['size']) for cells in comparison['cells'].values() for cell in cells})
    lines = ['# OOD final-population Oracle@10 comparison', '', '- Higher evaluation score is better.', '']
    for suite in sorted({suite for suite, _ in keys}):
        sizes = [size for key_suite, size in keys if key_suite == suite]
        lines.extend([
            f'## {suite}', '',
            '| Method | ' + ' | '.join(f'n={size}' for size in sizes) + ' |',
            '|---|' + '|'.join('---:' for _ in sizes) + '|',
        ])
        for method in comparison['methods']:
            lookup = {(cell['suite'], cell['size']): cell for cell in comparison['cells'][method]}
            values = []
            for size in sizes:
                cell = lookup[(suite, size)]
                values.append('FAILED' if cell['mean'] is None else f"{cell['mean']:.6f} +/- {cell['stdev']:.6f}")
            lines.append(f"| {method} | " + ' | '.join(values) + ' |')
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')


def _parse_run(value: str) -> tuple[str, Path]:
    if '=' not in value:
        raise argparse.ArgumentTypeError('--run must be METHOD=LOG_DIR')
    method, path = value.split('=', 1)
    return method, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate final-population Oracle@10 on OOD data.')
    parser.add_argument('--task', required=True, choices=tuple(TASK_EVALUATIONS))
    parser.add_argument('--run', action='append', required=True, type=_parse_run, help='METHOD=LOG_DIR; repeat for every method/run.')
    parser.add_argument('--suite', action='append', help='OOD suite; default all available suites.')
    parser.add_argument('--sizes', type=int, nargs='+')
    parser.add_argument('--k', type=int, default=10)
    parser.add_argument('--safe-evaluate', action='store_true')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    evaluation_cls = _evaluation_class(args.task)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_summaries = []
    for method, log_dir in args.run:
        output = args.output_root / method / log_dir.name if args.output_root else None
        summary = evaluate_final_population_oracle_ood(
            evaluation_cls, {}, log_dir, method, k=args.k, suites=args.suite,
            sizes=args.sizes, safe_evaluate=args.safe_evaluate, output_dir=output,
        )
        grouped[method].append(summary)
        all_summaries.append(summary)
    if args.output_root:
        args.output_root.mkdir(parents=True, exist_ok=True)
        comparison = _aggregate_runs(all_summaries)
        (args.output_root / 'index.json').write_text(
            json.dumps({'runs': grouped, 'comparison': comparison}, indent=2), encoding='utf-8'
        )
        _write_comparison(comparison, args.output_root / 'report.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
