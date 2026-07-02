"""Evaluate Oracle@10 for existing EoH TSP runs using top-10 train-score samples.

Because the requested EoH runs were executed with pop_size=4, their final population
only contains 4 individuals.  To obtain an Oracle@10 comparison we instead select the
10 highest-scoring *unique* samples from each run's sample history (the ``samples_*.json``
files).  This is a best-effort proxy for final-population Oracle@10; it is **not**
identical to re-running EoH with pop_size=10.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

from llm4ad.task.optimization.tsp_construct.evaluate_oracle_ood import (
    REPO_ROOT,
    DATA_ROOT,
    DEFAULT_SUITES,
    DEFAULT_SIZES,
    _evaluate_individual,
    _validate_optimisations,
    _aggregate,
    _write_report,
)

# The three EoH runs requested by the user.
EOH_RUNS = (
    'logs/20260627_205347_459872_pid4616_8723467b_TSPEvaluation_EoH',
    'logs/20260627_205305_823062_pid7748_5fa4b7b2_TSPEvaluation_EoH',
    'logs/20260625_141515_TSPEvaluation_EoH',
)

EXISTING_RESULTS_JSON = REPO_ROOT / 'logs' / '20260627_TSP_OOD_OracleK_AdvEoH_MCTS_AHD' / 'results.json'
TOP_K = 10


def _load_top_k_samples(log_dir: Path, k: int = TOP_K) -> tuple[list[dict[str, Any]], int]:
    """Return the top-k unique samples from a run's sample history, plus total unique count."""
    samples_dir = log_dir / 'samples'
    all_samples: list[dict[str, Any]] = []
    for path in sorted(samples_dir.glob('samples_*.json')):
        if path.name == 'samples_best.json':
            continue
        all_samples.extend(json.loads(path.read_text(encoding='utf-8')))

    valid = [
        s for s in all_samples
        if s.get('score') is not None
        and (s.get('program') or s.get('function'))
    ]

    # Deduplicate by program content, keeping the best (most negative) score seen.
    by_hash: dict[str, dict[str, Any]] = {}
    for sample in valid:
        program = sample.get('program') or sample.get('function')
        digest = hashlib.sha256(program.encode()).hexdigest()
        if digest not in by_hash or sample['score'] < by_hash[digest]['score']:
            by_hash[digest] = sample

    unique = sorted(by_hash.values(), key=lambda s: s['score'])
    return unique[:k], len(unique)


def main() -> int:
    suites = list(DEFAULT_SUITES)
    sizes = list(DEFAULT_SIZES)

    dataset_paths = {
        f'{suite}/size_{size}': str((DATA_ROOT / suite / f'size_{size}.pkl').resolve())
        for suite in suites for size in sizes
    }
    for path in dataset_paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    output_dir = REPO_ROOT / 'logs' / '20260627_TSP_OOD_OracleK10_EoH_top10_samples'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing baseline raw results.
    existing_raw: list[dict[str, Any]] = []
    existing_runs: list[dict[str, Any]] = []
    existing_validations: list[dict[str, Any]] = []
    if EXISTING_RESULTS_JSON.is_file():
        existing = json.loads(EXISTING_RESULTS_JSON.read_text(encoding='utf-8'))
        existing_raw = existing.get('raw_individual_results', [])
        existing_runs = existing.get('runs', [])
        existing_validations = existing.get('optimisation_validations', [])
        print(f'[EoH Top10] Loaded {len(existing_raw)} existing baseline raw results')
    else:
        print('[EoH Top10] Warning: existing baseline results not found')

    # Build EoH Oracle@10 tasks from top-10 samples.
    eoh_runs: list[dict[str, Any]] = []
    eoh_tasks: list[dict[str, Any]] = []
    for relative_dir in EOH_RUNS:
        log_dir = REPO_ROOT / relative_dir
        top_samples, n_unique = _load_top_k_samples(log_dir, TOP_K)
        if len(top_samples) < TOP_K:
            print(f'[EoH Top10] Warning: only {len(top_samples)} unique samples available '
                  f'for {log_dir.name} (out of {n_unique} unique)')
        programs = [s.get('program') or s.get('function') for s in top_samples]
        if any(not program for program in programs):
            raise ValueError(f'Missing program in top-{TOP_K} samples of {log_dir.name}')
        if len(set(programs)) != len(programs):
            raise ValueError(f'Duplicate program in top-{TOP_K} samples of {log_dir.name}')
        checkpoint_path = log_dir / 'samples' / f'top_{TOP_K}_samples.json'
        checkpoint_path.write_text(json.dumps(top_samples, indent=2), encoding='utf-8')
        eoh_runs.append({
            'method': 'EoH',
            'run': log_dir.name,
            'checkpoint': str(checkpoint_path),
            'k': len(programs),
            'unique_samples_available': n_unique,
        })
        for individual, program in enumerate(programs):
            for dataset_key, dataset_path in dataset_paths.items():
                task_id = f"EoH|{log_dir.name}|{individual}|{dataset_key}"
                eoh_tasks.append({
                    'task_id': task_id,
                    'method': 'EoH',
                    'run': log_dir.name,
                    'individual': individual,
                    'program': program,
                    'dataset_key': dataset_key,
                    'dataset_path': dataset_path,
                })

    eoh_validations = _validate_optimisations(
        eoh_tasks,
        DATA_ROOT / 'diagonal' / 'size_20.pkl',
    )

    all_runs = existing_runs + eoh_runs
    all_validations = existing_validations + eoh_validations

    # Resume support.
    partial_path = output_dir / 'partial_results.json'
    eoh_raw: list[dict[str, Any]] = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding='utf-8'))
        if isinstance(partial, list):
            eoh_raw = partial
    completed_ids = {item['task_id'] for item in eoh_raw}
    pending_tasks = [task for task in eoh_tasks if task['task_id'] not in completed_ids]
    if completed_ids:
        print(f'[EoH Top10] Resuming {len(completed_ids)} completed EoH tasks; {len(pending_tasks)} remain.')

    print(f'[EoH Top10] Evaluating {len(pending_tasks)} EoH tasks across 3 runs (top-{TOP_K} samples each).')

    max_workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(_evaluate_individual, task) for task in pending_tasks]
        for newly_completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            eoh_raw.append(result)
            completed = len(completed_ids) + newly_completed
            print(
                f"[EoH Top10] {completed:02d}/{len(eoh_tasks)} {result['method']} "
                f"{result['run']} individual={result['individual']} "
                f"dataset={result['dataset_key']} error={result['error'] is not None} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if newly_completed % 10 == 0:
                partial_path.write_text(json.dumps(eoh_raw), encoding='utf-8')
        partial_path.write_text(json.dumps(eoh_raw), encoding='utf-8')

    all_raw = existing_raw + eoh_raw
    per_run, aggregate = _aggregate(all_raw, all_runs, suites, sizes)
    failed = sum(item['error'] is not None for item in all_raw)

    summary = {
        'metric': 'per-instance Oracle@K mean tour length',
        'note': (
            f'EoH uses top-{TOP_K} unique samples by train score as a proxy for Oracle@10 '
            'because the evaluated runs were executed with pop_size=4.'
        ),
        'suites': suites,
        'sizes': sizes,
        'runs': all_runs,
        'per_run': per_run,
        'aggregate': aggregate,
        'optimisation_validations': all_validations,
        'failed_evaluations': failed,
        'raw_individual_results': all_raw,
    }

    json_path = output_dir / 'results.json'
    report_path = output_dir / 'report.md'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_report(summary, report_path)

    # Combined Oracle@K folder.
    oracle_dir = output_dir / 'oracle'
    oracle_dir.mkdir(parents=True, exist_ok=True)
    (oracle_dir / 'results.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_report(summary, oracle_dir / 'report.md')

    # Per-method sub-folders.
    folder_map = {
        'AdvEoH': 'adveoh',
        'EoH': 'eoh',
        'MCTS_AHD': 'mcts',
    }
    for method, folder_name in folder_map.items():
        if method not in aggregate:
            continue
        method_dir = output_dir / folder_name
        method_dir.mkdir(parents=True, exist_ok=True)
        method_summary = {
            'metric': summary['metric'],
            'note': summary.get('note'),
            'suites': suites,
            'sizes': sizes,
            'method': method,
            'runs': [r for r in all_runs if r['method'] == method],
            'per_run': [r for r in per_run if r['method'] == method],
            'aggregate': aggregate[method],
            'failed_evaluations': sum(
                item['error'] is not None for item in all_raw if item['method'] == method
            ),
        }
        (method_dir / 'results.json').write_text(
            json.dumps(method_summary, indent=2), encoding='utf-8'
        )

    # AHD is represented by MCTS_AHD in this codebase.
    ahd_dir = output_dir / 'ahd'
    ahd_dir.mkdir(parents=True, exist_ok=True)
    (ahd_dir / 'results.json').write_text(
        (output_dir / 'mcts' / 'results.json').read_text(encoding='utf-8'),
        encoding='utf-8',
    )
    (ahd_dir / 'README.md').write_text(
        '# AHD results\n\nIn this evaluation the AHD baseline is represented by the `MCTS_AHD` '
        'method, so the JSON here is identical to the `mcts` folder.\n',
        encoding='utf-8',
    )

    print(f'[EoH Top10] JSON: {json_path}')
    print(f'[EoH Top10] Report: {report_path}')
    print(f'[EoH Top10] Failed evaluations: {failed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
