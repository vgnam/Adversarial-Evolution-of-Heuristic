from __future__ import annotations

import inspect
import json
import os
from typing import Any

import numpy as np

from llm4ad.base import Evaluation, SecureEvaluator
from llm4ad.task.optimization._dataset_loader import list_test_dataset_sizes


DEFAULT_TEST_N_INSTANCE = 64
DEFAULT_TEST_TIMEOUT_SECONDS = 240


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _task_folder_for_evaluation(evaluation_cls: type[Evaluation]) -> str:
    return os.path.dirname(inspect.getfile(evaluation_cls))


def _is_heuristic_record(record: dict[str, Any]) -> bool:
    role = record.get('role')
    function = record.get('function') or record.get('program') or ''
    if role not in (None, 'heuristic'):
        return False
    return 'def generate_instance(' not in function


def _load_json_records(path: str) -> list[dict[str, Any]]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            records = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return records if isinstance(records, list) else []


def _load_best_record(log_dir: str) -> dict[str, Any] | None:
    samples_dir = os.path.join(log_dir, 'samples')
    if not os.path.isdir(samples_dir):
        return None

    best_record = None
    for filename in ('samples_best_heuristic.json', 'samples_best.json'):
        records = [
            record for record in _load_json_records(os.path.join(samples_dir, filename))
            if _is_heuristic_record(record)
        ]
        if records:
            heldout_records = [
                record for record in records
                if record.get('selection_metric') == 'heldout_score'
            ]
            best_record = heldout_records[-1] if heldout_records else records[-1]
            break

    if best_record is None:
        history_records = []
        for filename in sorted(os.listdir(samples_dir)):
            if not filename.startswith('samples_') or 'best' in filename:
                continue
            for record in _load_json_records(os.path.join(samples_dir, filename)):
                if _is_heuristic_record(record) and isinstance(record.get('score'), (int, float)):
                    history_records.append(record)
        if not history_records:
            return None
        best_record = max(history_records, key=lambda record: record['score'])

    if not best_record:
        return None
    if best_record.get('program'):
        return best_record

    sample_order = best_record.get('sample_order')
    if sample_order is None or not os.path.isdir(samples_dir):
        return best_record

    for filename in sorted(os.listdir(samples_dir)):
        if not filename.startswith('samples_') or 'best' in filename:
            continue
        path = os.path.join(samples_dir, filename)
        for record in _load_json_records(path):
            if (
                record.get('sample_order') == sample_order
                and record.get('program')
                and _is_heuristic_record(record)
            ):
                merged = dict(best_record)
                merged['program'] = record['program']
                return merged

    return best_record


def _format_score(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f'{value:.6f}'
    return str(value)


def _print_eval_summary(summary: dict[str, Any]) -> None:
    print('[Eval] Fixed test summary:')
    print(f"[Eval]   best_sample_order: {summary.get('best_sample_order')}")
    print(f"[Eval]   train_score: {_format_score(summary.get('train_score'))}")
    print(
        f"[Eval]   n_test_instance: {summary.get('n_test_instance')}, "
        f"timeout_seconds: {summary.get('timeout_seconds')}"
    )
    print('[Eval]   size    score')
    for item in summary.get('test_results', []):
        print(
            f"[Eval]   {str(item.get('size')).rjust(4)}    "
            f"{_format_score(item.get('score'))}"
        )
    print(f"[Eval]   mean_test_score: {_format_score(summary.get('mean_test_score'))}")


def evaluate_best_on_fixed_test_datasets(
        evaluation_cls: type[Evaluation],
        evaluation_params: dict[str, Any],
        log_dir: str | None,
        n_test_instance: int = DEFAULT_TEST_N_INSTANCE,
        timeout_seconds: int | float | None = DEFAULT_TEST_TIMEOUT_SECONDS,
) -> dict[str, Any] | None:
    """Evaluate the final best logged heuristic on all fixed test datasets.

    This follows the EoH paper protocol: train/evolve on the training split,
    select the best heuristic, then report that heuristic's average score on
    each held-out test size.
    """
    if not log_dir:
        return None

    task_folder = _task_folder_for_evaluation(evaluation_cls)
    test_sizes = list_test_dataset_sizes(task_folder)
    if not test_sizes:
        return None

    best_record = _load_best_record(log_dir)
    if best_record is None:
        return None

    program = best_record.get('program') or best_record.get('function')
    if not program:
        return None

    eval_dir = os.path.join(log_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    base_params = {
        key: value
        for key, value in evaluation_params.items()
        if key not in {'name', 'dataset_split', 'dataset_size', 'dataset_file', 'n_instance'}
    }
    base_params['load_from_file'] = True
    base_params['n_instance'] = n_test_instance
    if timeout_seconds is not None:
        base_params['timeout_seconds'] = timeout_seconds

    results = []
    for size in test_sizes:
        test_params = dict(base_params)
        test_params.update(dataset_split='test', dataset_size=size)
        test_task = evaluation_cls(**test_params)
        score = SecureEvaluator(test_task).evaluate_program(program)
        results.append({
            'size': size,
            'score': _jsonable(score),
        })

    numeric_scores = [
        item['score']
        for item in results
        if isinstance(item['score'], (int, float))
    ]
    summary = {
        'best_sample_order': best_record.get('sample_order'),
        'train_score': _jsonable(best_record.get('score')),
        'n_test_instance': n_test_instance,
        'timeout_seconds': timeout_seconds,
        'test_results': results,
        'mean_test_score': (
            sum(numeric_scores) / len(numeric_scores)
            if numeric_scores else None
        ),
    }

    out_path = os.path.join(eval_dir, 'eval_results.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4)

    print(f'[Eval] Fixed test results saved to {out_path}')
    _print_eval_summary(summary)
    return summary
