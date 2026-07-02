from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from llm4ad.task.optimization import oracle_ood_eval
from llm4ad.task.optimization.generate_ood_datasets import (
    TASK_FAMILIES,
    generate_dataset,
    generate_instance,
)
from llm4ad.task.optimization.tsp_construct import TSPEvaluation


@pytest.mark.parametrize('task', sorted(TASK_FAMILIES))
def test_every_constructive_task_generates_all_ood_families(task: str) -> None:
    rng = np.random.default_rng(42)
    for family in TASK_FAMILIES[task]:
        instance = generate_instance(task, size=8, family=family, rng=rng)
        assert instance is not None


def test_oracle_at_10_uses_final_population_per_instance(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / 'task_root'
    dataset_path = data_root / 'tsp_construct' / 'ood_test_datasets' / 'mixture' / 'size_8.pkl'
    generate_dataset('tsp_construct', 8, 3, 123, 'mixture', dataset_path)
    monkeypatch.setattr(oracle_ood_eval, 'HERE', data_root)

    log_dir = tmp_path / 'run'
    population_dir = log_dir / 'population'
    population_dir.mkdir(parents=True)
    records = []
    for variant in range(10):
        records.append({
            'score': -float(variant),
            'function': f'''import numpy as np
def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    variant = {variant}
    distances = distance_matrix[current_node, unvisited_nodes] + variant * 0.0
    return unvisited_nodes[int(np.argmin(distances))]
''',
        })
    (population_dir / 'pop_9.json').write_text(json.dumps(records), encoding='utf-8')

    summary = oracle_ood_eval.evaluate_final_population_oracle_ood(
        TSPEvaluation,
        {},
        log_dir,
        'EoH',
        k=10,
        output_dir=tmp_path / 'output',
    )

    assert summary['k'] == 10
    assert len(summary['rows']) == 1
    assert summary['rows'][0]['n_instances'] == 3
    assert summary['rows'][0]['failed_program_instance_evaluations'] == 0
    assert (tmp_path / 'output' / 'results.json').is_file()


def test_oracle_at_10_rejects_smaller_final_population(tmp_path: Path) -> None:
    population_dir = tmp_path / 'population'
    population_dir.mkdir()
    records = [{'function': f'def f(x):\n    return x + {i}\n'} for i in range(9)]
    (population_dir / 'pop_1.json').write_text(json.dumps(records), encoding='utf-8')

    with pytest.raises(ValueError, match='requires 10 unique'):
        oracle_ood_eval.load_final_population(tmp_path, k=10)
