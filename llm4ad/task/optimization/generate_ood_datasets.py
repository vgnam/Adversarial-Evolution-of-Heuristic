"""Generate reproducible multi-distribution OOD data for constructive tasks.

Training data is intentionally untouched: every method continues to train on
the existing ``train_dataset.pkl``/``train_datasets`` ID split.  This command
writes held-out data to ``<task>/ood_test_datasets/<suite>/size_<N>.pkl``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llm4ad.task.optimization.generate_fixed_datasets import TASK_SPECS


HERE = Path(__file__).resolve().parent
DEFAULT_SEED = 20260702
DEFAULT_N_INSTANCES = 64
UPPER_BOUND = np.nextafter(1.0, 0.0)


def _clip(points: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(points, dtype=float), 0.0, UPPER_BOUND)


def _clustered(n: int, rng: np.random.Generator) -> np.ndarray:
    k = min(max(2, n // 12), 6)
    centers = rng.uniform(0.12, 0.88, size=(k, 2))
    labels = np.arange(n) % k
    rng.shuffle(labels)
    return _clip(centers[labels] + rng.normal(0, 0.035, size=(n, 2)))


def _rings(n: int, rng: np.random.Generator) -> np.ndarray:
    ring_ids = np.arange(n) % 3
    rng.shuffle(ring_ids)
    radii = np.array([0.12, 0.28, 0.44])[ring_ids]
    angles = rng.uniform(0, 2 * np.pi, size=n)
    points = 0.5 + np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    return _clip(points + rng.normal(0, 0.006, size=(n, 2)))


def _boundary(n: int, rng: np.random.Generator) -> np.ndarray:
    side = np.arange(n) % 4
    rng.shuffle(side)
    along = rng.uniform(0.02, 0.98, size=n)
    inward = np.minimum(rng.exponential(0.018, size=n), 0.07)
    points = np.empty((n, 2))
    points[side == 0] = np.column_stack((along[side == 0], inward[side == 0]))
    points[side == 1] = np.column_stack((1 - inward[side == 1], along[side == 1]))
    points[side == 2] = np.column_stack((along[side == 2], 1 - inward[side == 2]))
    points[side == 3] = np.column_stack((inward[side == 3], along[side == 3]))
    return _clip(points)


def _diagonal(n: int, rng: np.random.Generator) -> np.ndarray:
    x = rng.uniform(0.03, 0.97, size=n)
    y = np.where(rng.random(n) < 0.5, x, 1 - x)
    return _clip(np.column_stack((x, y)) + rng.normal(0, 0.018, size=(n, 2)))


def _radial(n: int, rng: np.random.Generator) -> np.ndarray:
    spokes = 5
    ids = np.arange(n) % spokes
    rng.shuffle(ids)
    angles = ids * 2 * np.pi / spokes + rng.normal(0, 0.02, size=n)
    radii = rng.uniform(0.04, 0.48, size=n)
    return _clip(0.5 + np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None])


def _core_outliers(n: int, rng: np.random.Generator) -> np.ndarray:
    n_out = max(2, round(0.15 * n))
    core = rng.normal(0.5, 0.055, size=(n - n_out, 2))
    angles = rng.uniform(0, 2 * np.pi, size=n_out)
    out = 0.5 + np.column_stack((np.cos(angles), np.sin(angles))) * rng.uniform(0.4, 0.49, size=(n_out, 1))
    points = np.vstack((core, out))
    rng.shuffle(points)
    return _clip(points)


COORDINATE_FAMILIES: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    'clustered': _clustered,
    'concentric_rings': _rings,
    'boundary': _boundary,
    'crossing_diagonals': _diagonal,
    'radial_spokes': _radial,
    'dense_core_outliers': _core_outliers,
}

TASK_FAMILIES = {
    'tsp_construct': tuple(COORDINATE_FAMILIES),
    'cvrp_construct': tuple(COORDINATE_FAMILIES),
    'ovrp_construct': tuple(COORDINATE_FAMILIES),
    'vrptw_construct': tuple(COORDINATE_FAMILIES),
    'bp_1d_construct': ('uniform_small', 'bimodal', 'near_capacity', 'lognormal'),
    'bp_2d_construct': ('slender', 'large_squares', 'bimodal', 'small_dense'),
    'knapsack_construct': ('strongly_correlated', 'inverse_correlated', 'bimodal', 'heavy_tailed'),
    'jssp_construct': ('bimodal', 'machine_bottleneck', 'job_correlated', 'heavy_tailed'),
    'qap_construct': ('sparse', 'block', 'heavy_tailed', 'geometric'),
    'cflp_construct': ('tight_capacity', 'clustered_cost', 'bimodal_demand', 'sparse_cheap_edges'),
    'set_cover_construct': ('sparse', 'dense', 'clustered', 'power_law'),
}


def _distance(points: np.ndarray) -> np.ndarray:
    return np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)


def _demands(n: int, family: str, rng: np.random.Generator) -> np.ndarray:
    values = {
        'clustered': rng.integers(1, 5, size=n),
        'concentric_rings': rng.choice([1, 2, 8, 9], size=n),
        'boundary': rng.integers(6, 10, size=n),
        'crossing_diagonals': np.clip(np.rint(np.linspace(1, 9, n) + rng.normal(0, 1, n)), 1, 9).astype(int),
        'radial_spokes': rng.integers(1, 10, size=n),
        'dense_core_outliers': np.where(np.arange(n) < max(2, n // 6), 9, rng.integers(1, 5, size=n)),
    }[family]
    values[0] = 0
    return values


def _routing_instance(task: str, size: int, family: str, rng: np.random.Generator) -> tuple:
    n = size if task == 'tsp_construct' else size + 1
    points = COORDINATE_FAMILIES[family](n, rng)
    distances = _distance(points)
    if task == 'tsp_construct':
        return points, distances
    demands = _demands(n, family, rng)
    capacity = 40
    if task in {'cvrp_construct', 'ovrp_construct'}:
        return points, distances, demands, capacity

    service = np.zeros(n)
    service[1:] = rng.uniform(0.12, 0.22, size=n - 1)
    width = 0.10 if family in {'boundary', 'crossing_diagonals'} else 0.30
    d0 = distances[0, 1:]
    # Every customer remains reachable from the depot and can return before
    # the depot closes, even for the tight-window OOD families.
    latest_start = np.maximum(d0, 4.6 - service[1:] - width - d0)
    early = d0 + rng.random(n - 1) * np.maximum(0, latest_start - d0)
    windows = np.column_stack((early, early + width))
    windows = np.vstack(([0.0, 4.6], windows))
    return points, distances, demands, capacity, service, windows


def _bp1(size: int, family: str, rng: np.random.Generator) -> tuple[list[int], int]:
    capacity = 100
    if family == 'uniform_small':
        weights = rng.integers(3, 31, size=size)
    elif family == 'bimodal':
        weights = rng.choice(np.r_[5:16, 65:91], size=size)
    elif family == 'near_capacity':
        weights = rng.integers(70, 100, size=size)
    else:
        weights = np.clip(rng.lognormal(3.4, 0.65, size=size), 2, 99).astype(int)
    return weights.tolist(), capacity


def _bp2(size: int, family: str, rng: np.random.Generator) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    if family == 'slender':
        w = rng.integers(4, 16, size=size); h = rng.integers(55, 96, size=size)
        swap = rng.random(size) < 0.5; w[swap], h[swap] = h[swap], w[swap].copy()
    elif family == 'large_squares':
        w = rng.integers(55, 91, size=size); h = rng.integers(55, 91, size=size)
    elif family == 'bimodal':
        large = rng.random(size) < 0.5
        w = np.where(large, rng.integers(55, 91, size=size), rng.integers(5, 21, size=size))
        h = np.where(large, rng.integers(55, 91, size=size), rng.integers(5, 21, size=size))
    else:
        w = rng.integers(3, 25, size=size); h = rng.integers(3, 25, size=size)
    return list(zip(w.tolist(), h.tolist())), (100, 100)


def _knapsack(size: int, family: str, rng: np.random.Generator) -> tuple[list[int], list[int], int]:
    weights = rng.integers(5, 60, size=size)
    if family == 'strongly_correlated':
        values = weights + rng.integers(8, 14, size=size)
    elif family == 'inverse_correlated':
        values = 110 - weights + rng.integers(-5, 6, size=size)
    elif family == 'bimodal':
        weights = rng.choice(np.r_[5:16, 45:71], size=size)
        values = rng.choice(np.r_[5:21, 80:121], size=size)
    else:
        weights = np.clip(rng.lognormal(2.8, 0.8, size=size), 2, 90).astype(int)
        values = np.clip(rng.lognormal(3.4, 0.9, size=size), 1, 180).astype(int)
    return weights.astype(int).tolist(), np.maximum(values, 1).astype(int).tolist(), 100


def _jssp(size: int, family: str, rng: np.random.Generator) -> tuple[list[list[int]], int, int]:
    machines = 10
    if family == 'bimodal':
        times = rng.choice(np.r_[1:11, 90:151], size=(size, machines))
    elif family == 'machine_bottleneck':
        times = rng.integers(5, 35, size=(size, machines)); times[:, 0] = rng.integers(100, 181, size=size)
    elif family == 'job_correlated':
        base = rng.integers(5, 130, size=(size, 1)); times = np.clip(base + rng.integers(-4, 5, size=(size, machines)), 1, None)
    else:
        times = np.clip(rng.lognormal(3.2, 0.9, size=(size, machines)), 1, 220).astype(int)
    return times.astype(int).tolist(), size, machines


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=int)
    matrix = (matrix + matrix.T) // 2
    np.fill_diagonal(matrix, 0)
    return matrix


def _qap(size: int, family: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if family == 'sparse':
        flow = np.where(rng.random((size, size)) < 0.12, rng.integers(40, 151, (size, size)), 0)
        dist = rng.integers(1, 101, (size, size))
    elif family == 'block':
        labels = np.arange(size) % max(2, min(5, size // 3)); rng.shuffle(labels)
        same = labels[:, None] == labels[None, :]
        flow = np.where(same, rng.integers(70, 151, (size, size)), rng.integers(1, 16, (size, size)))
        dist = rng.integers(1, 101, (size, size))
    elif family == 'heavy_tailed':
        flow = np.clip(rng.lognormal(3.0, 1.1, (size, size)), 0, 250).astype(int)
        dist = np.clip(rng.lognormal(2.5, 1.0, (size, size)), 0, 200).astype(int)
    else:
        a = rng.uniform(0, 1, (size, 2)); b = rng.uniform(0, 1, (size, 2))
        flow = np.rint(_distance(a) * 100); dist = np.rint(_distance(b) * 100)
    return _symmetric(flow), _symmetric(dist)


def _cflp(size: int, family: str, rng: np.random.Generator) -> dict[str, Any]:
    demands = rng.integers(5, 21, size=size)
    capacities = rng.integers(40, 101, size=size)
    costs = rng.integers(5, 51, size=(size, size))
    if family == 'tight_capacity':
        target = int(math.ceil(demands.sum() * 1.05))
        capacities[:] = max(5, target // size)
        capacities[:target - capacities.sum()] += 1
    elif family == 'clustered_cost':
        groups = np.arange(size) % max(2, min(5, size // 4)); rng.shuffle(groups)
        costs = np.where(groups[:, None] == groups[None, :], rng.integers(1, 9, (size, size)), rng.integers(60, 121, (size, size)))
    elif family == 'bimodal_demand':
        demands = rng.choice(np.r_[2:7, 25:41], size=size); capacities = rng.integers(60, 141, size=size)
    else:
        costs = rng.integers(70, 151, size=(size, size)); cheap = rng.random((size, size)) < 0.12; costs[cheap] = rng.integers(1, 9, cheap.sum())
    if capacities.sum() < demands.sum():
        capacities[0] += int(demands.sum() - capacities.sum())
    return {'facility_capacities': capacities.tolist(), 'customer_demands': demands.tolist(), 'assignment_costs': costs.tolist()}


def _set_cover(size: int, family: str, rng: np.random.Generator) -> tuple[list[int], list[list[int]]]:
    universe = list(range(1, size + 1))
    subsets: list[list[int]] = []
    for i in range(size):
        if family == 'sparse': count = rng.integers(1, max(2, size // 12) + 1)
        elif family == 'dense': count = rng.integers(max(1, size // 2), size + 1)
        elif family == 'clustered':
            block = np.arange(i % 4, size, 4) + 1; count = rng.integers(1, len(block) + 1)
            subsets.append(rng.choice(block, size=count, replace=False).tolist()); continue
        else: count = min(size, max(1, int(rng.zipf(1.8))))
        subsets.append(rng.choice(universe, size=count, replace=False).tolist())
    covered = {item for subset in subsets for item in subset}
    for item in set(universe) - covered:
        subsets[int(rng.integers(0, len(subsets)))].append(item)
    return universe, subsets


def generate_instance(task: str, size: int, family: str, rng: np.random.Generator) -> Any:
    if task in {'tsp_construct', 'cvrp_construct', 'ovrp_construct', 'vrptw_construct'}:
        return _routing_instance(task, size, family, rng)
    functions = {
        'bp_1d_construct': _bp1, 'bp_2d_construct': _bp2,
        'knapsack_construct': _knapsack, 'jssp_construct': _jssp,
        'qap_construct': _qap, 'cflp_construct': _cflp,
        'set_cover_construct': _set_cover,
    }
    return functions[task](size, family, rng)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def generate_dataset(task: str, size: int, n_instances: int, seed: int, suite: str, output: Path) -> dict[str, Any]:
    families = TASK_FAMILIES[task]
    if suite != 'mixture':
        if suite not in families:
            raise ValueError(f'{suite!r} is not valid for {task}; choices: {families}')
        families = (suite,)
    rng = np.random.default_rng(np.random.SeedSequence([seed, size, sum(map(ord, task))]))
    assignments = [families[i % len(families)] for i in range(n_instances)]
    rng.shuffle(assignments)
    instances = [generate_instance(task, size, family, rng) for family in assignments]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        pickle.dump(instances, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        'size': size, 'path': str(output.resolve()), 'n_instances': n_instances,
        'sha256': _sha256(output), 'bytes': output.stat().st_size,
        'distribution_counts': dict(sorted(Counter(assignments).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate multi-distribution OOD datasets.')
    parser.add_argument('--task', action='append', choices=tuple(TASK_FAMILIES), help='Repeat to select tasks; default all constructive tasks.')
    parser.add_argument('--sizes', type=int, nargs='+', help='Override each task default benchmark sizes.')
    parser.add_argument('--n-instances', type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--suite', default='mixture', help='mixture or one task-specific family.')
    parser.add_argument('--output-root', type=Path, default=HERE)
    args = parser.parse_args()
    if args.n_instances <= 0:
        parser.error('--n-instances must be positive')

    specs = {item['name']: item for item in TASK_SPECS}
    tasks = args.task or list(TASK_FAMILIES)
    for task in tasks:
        sizes = args.sizes or list(specs[task]['test_sizes'])
        out_dir = args.output_root / task / 'ood_test_datasets' / args.suite
        datasets = []
        for size in sizes:
            item = generate_dataset(task, size, args.n_instances, args.seed, args.suite, out_dir / f'size_{size}.pkl')
            datasets.append(item)
            print(f"[OOD] {task}/{args.suite} size={size}: {item['distribution_counts']}")
        metadata = {
            'task': task, 'suite': args.suite, 'split': 'ood_test',
            'reference_split': 'train (ID)', 'seed': args.seed,
            'families': list(TASK_FAMILIES[task]), 'datasets': datasets,
        }
        (out_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
