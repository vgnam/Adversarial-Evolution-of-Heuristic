"""Evaluate final-population Oracle@K for 3 EoH TSP runs and combine with existing AdvEoH/MCTS_AHD results."""
from __future__ import annotations

import json
import concurrent.futures
import os
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np

# Re-use helpers from the original Oracle@K evaluator.
from llm4ad.task.optimization.tsp_construct.evaluate_oracle_ood import (
    REPO_ROOT,
    DATA_ROOT,
    DEFAULT_SUITES,
    DEFAULT_SIZES,
    _load_final_population,
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

# Previously computed Oracle@K results for the baseline methods.
EXISTING_RESULTS_JSON = REPO_ROOT / 'logs' / '20260627_TSP_OOD_OracleK_AdvEoH_MCTS_AHD' / 'results.json'


def _build_run_info(method: str, log_dir: Path, checkpoint: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        'method': method,
        'run': log_dir.name,
        'checkpoint': str(checkpoint),
        'k': len(records),
    }


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

    output_dir = REPO_ROOT / 'logs' / '20260627_TSP_OOD_OracleK_with_EoH'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load existing baseline raw results if available.
    existing_raw: list[dict[str, Any]] = []
    existing_runs: list[dict[str, Any]] = []
    existing_validations: list[dict[str, Any]] = []
    if EXISTING_RESULTS_JSON.is_file():
        existing = json.loads(EXISTING_RESULTS_JSON.read_text(encoding='utf-8'))
        existing_raw = existing.get('raw_individual_results', [])
        existing_runs = existing.get('runs', [])
        existing_validations = existing.get('optimisation_validations', [])
        print(f'[EoH Oracle] Loaded {len(existing_raw)} existing raw results from {EXISTING_RESULTS_JSON}')
    else:
        print('[EoH Oracle] Warning: existing baseline results not found; EoH will be evaluated standalone.')

    # Build EoH tasks.
    eoh_runs: list[dict[str, Any]] = []
    eoh_tasks: list[dict[str, Any]] = []
    for relative_dir in EOH_RUNS:
        log_dir = REPO_ROOT / relative_dir
        checkpoint, records = _load_final_population('EoH', log_dir)
        programs = [record.get('program') or record.get('function') for record in records]
        if any(not program for program in programs):
            raise ValueError(f'Missing function in {checkpoint}')
        if len(set(programs)) != len(programs):
            raise ValueError(f'Duplicate program in {checkpoint}')
        eoh_runs.append(_build_run_info('EoH', log_dir, checkpoint, records))
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

    # Validate optimisations on a representative subset of EoH programs.
    eoh_validations = _validate_optimisations(
        eoh_tasks,
        DATA_ROOT / 'diagonal' / 'size_20.pkl',
    )

    all_runs = existing_runs + eoh_runs
    all_validations = existing_validations + eoh_validations

    # Resume support for EoH evaluation.
    partial_path = output_dir / 'partial_results.json'
    eoh_raw: list[dict[str, Any]] = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding='utf-8'))
        if isinstance(partial, list):
            eoh_raw = partial
    completed_ids = {item['task_id'] for item in eoh_raw}
    pending_tasks = [task for task in eoh_tasks if task['task_id'] not in completed_ids]
    if completed_ids:
        print(f'[EoH Oracle] Resuming {len(completed_ids)} completed EoH tasks; {len(pending_tasks)} remain.')

    print(f'[EoH Oracle] Evaluating {len(pending_tasks)} EoH tasks across {len(eoh_runs)} runs '
          f'({len(existing_runs)} baseline runs already loaded).')

    max_workers = min(8, os.cpu_count() or 1)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(_evaluate_individual, task) for task in pending_tasks]
        for newly_completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            eoh_raw.append(result)
            completed = len(completed_ids) + newly_completed
            print(
                f"[EoH Oracle] {completed:02d}/{len(eoh_tasks)} {result['method']} "
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

    # Per-method sub-folders as requested (mcts, ahd, adveoh, plus eoh).
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

    # AHD is represented by the MCTS_AHD baseline in this codebase.
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

    print(f'[EoH Oracle] JSON: {json_path}')
    print(f'[EoH Oracle] Report: {report_path}')
    print(f'[EoH Oracle] Failed evaluations: {failed}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
