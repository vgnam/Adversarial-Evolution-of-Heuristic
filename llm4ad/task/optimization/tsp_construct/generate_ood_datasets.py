"""Generate reproducible mixed-distribution OOD datasets for constructive TSP.

The regular TSP data uses iid uniform coordinates in ``[0, 1]^2``.  Every
instance generated here independently draws one distribution family from a
uniform categorical distribution, then draws the point cloud from that family.
The chosen family is recorded in ``metadata.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np


DEFAULT_SIZES = (20, 50, 100, 200)
DEFAULT_N_INSTANCES = 64
DEFAULT_SEED = 20260627
UPPER_BOUND = np.nextafter(1.0, 0.0)

Instance = tuple[np.ndarray, np.ndarray]
DistributionFn = Callable[[int, np.random.Generator], np.ndarray]


def _clip(coordinates: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(coordinates, dtype=float), 0.0, UPPER_BOUND)


def _clustered_gaussian(n: int, rng: np.random.Generator) -> np.ndarray:
    n_clusters = int(rng.integers(3, min(7, n + 1)))
    centers = rng.uniform(0.12, 0.88, size=(n_clusters, 2))
    assignments = np.arange(n) % n_clusters
    rng.shuffle(assignments)
    std = rng.uniform(0.025, 0.075)
    return _clip(centers[assignments] + rng.normal(0.0, std, size=(n, 2)))


def _concentric_rings(n: int, rng: np.random.Generator) -> np.ndarray:
    n_rings = int(rng.integers(2, 5))
    center = np.array([0.5, 0.5]) + rng.normal(0.0, 0.025, size=2)
    radii = np.linspace(rng.uniform(0.08, 0.14), rng.uniform(0.36, 0.47), n_rings)
    ring_ids = np.arange(n) % n_rings
    rng.shuffle(ring_ids)
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    noisy_radii = radii[ring_ids] + rng.normal(0.0, 0.008, size=n)
    points = center + np.column_stack((np.cos(angles), np.sin(angles))) * noisy_radii[:, None]
    return _clip(points)


def _jittered_grid(n: int, rng: np.random.Generator) -> np.ndarray:
    side = math.ceil(math.sqrt(n))
    axis = np.linspace(0.06, 0.94, side)
    grid = np.array([(x, y) for x in axis for y in axis], dtype=float)
    chosen = rng.choice(len(grid), size=n, replace=False)
    jitter = rng.normal(0.0, rng.uniform(0.004, 0.018), size=(n, 2))
    return _clip(grid[chosen] + jitter)


def _crossing_diagonals(n: int, rng: np.random.Generator) -> np.ndarray:
    t = rng.uniform(0.03, 0.97, size=n)
    use_anti_diagonal = rng.random(n) < 0.5
    y = np.where(use_anti_diagonal, 1.0 - t, t)
    noise = rng.normal(0.0, rng.uniform(0.012, 0.035), size=(n, 2))
    return _clip(np.column_stack((t, y)) + noise)


def _radial_spokes(n: int, rng: np.random.Generator) -> np.ndarray:
    n_spokes = int(rng.integers(3, 8))
    center = np.array([0.5, 0.5]) + rng.normal(0.0, 0.02, size=2)
    base_angles = rng.uniform(0.0, 2.0 * np.pi) + np.arange(n_spokes) * 2.0 * np.pi / n_spokes
    spoke_ids = np.arange(n) % n_spokes
    rng.shuffle(spoke_ids)
    angles = base_angles[spoke_ids] + rng.normal(0.0, 0.025, size=n)
    radii = rng.uniform(0.05, 0.48, size=n)
    points = center + np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    return _clip(points)


def _boundary_frame(n: int, rng: np.random.Generator) -> np.ndarray:
    sides = np.arange(n) % 4
    rng.shuffle(sides)
    along = rng.uniform(0.02, 0.98, size=n)
    inward = np.minimum(rng.exponential(scale=0.018, size=n), 0.08)
    points = np.empty((n, 2), dtype=float)
    points[sides == 0] = np.column_stack((along[sides == 0], inward[sides == 0]))
    points[sides == 1] = np.column_stack((1.0 - inward[sides == 1], along[sides == 1]))
    points[sides == 2] = np.column_stack((along[sides == 2], 1.0 - inward[sides == 2]))
    points[sides == 3] = np.column_stack((inward[sides == 3], along[sides == 3]))
    return _clip(points)


def _dense_core_with_outliers(n: int, rng: np.random.Generator) -> np.ndarray:
    n_outliers = max(2, round(0.18 * n))
    n_core = n - n_outliers
    center = rng.uniform(0.38, 0.62, size=2)
    core = center + rng.normal(0.0, rng.uniform(0.035, 0.075), size=(n_core, 2))
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_outliers)
    radii = rng.uniform(0.36, 0.49, size=n_outliers)
    outliers = center + np.column_stack((np.cos(angles), np.sin(angles))) * radii[:, None]
    return _clip(np.vstack((core, outliers)))


def _noisy_spiral(n: int, rng: np.random.Generator) -> np.ndarray:
    phase = rng.uniform(0.0, 2.0 * np.pi)
    theta = np.linspace(0.0, 4.5 * np.pi, n) + phase
    radius = np.linspace(0.035, 0.47, n)
    points = np.column_stack((np.cos(theta), np.sin(theta))) * radius[:, None] + 0.5
    points += rng.normal(0.0, rng.uniform(0.004, 0.014), size=(n, 2))
    rng.shuffle(points)
    return _clip(points)


DISTRIBUTIONS: dict[str, DistributionFn] = {
    'clustered_gaussian': _clustered_gaussian,
    'concentric_rings': _concentric_rings,
    'jittered_grid': _jittered_grid,
    'crossing_diagonals': _crossing_diagonals,
    'radial_spokes': _radial_spokes,
    'boundary_frame': _boundary_frame,
    'dense_core_with_outliers': _dense_core_with_outliers,
    'noisy_spiral': _noisy_spiral,
}

SUITES: dict[str, tuple[str, ...]] = {
    'clustered': ('clustered_gaussian',),
    'diagonal': ('crossing_diagonals',),
    'mixture': tuple(DISTRIBUTIONS),
}


def generate_mixed_instances(
    n_instances: int,
    n_cities: int,
    rng: np.random.Generator,
    families: tuple[str, ...] | None = None,
) -> tuple[list[Instance], list[str]]:
    """Draw each instance from one of ``families`` with uniform probability."""
    names = families or tuple(DISTRIBUTIONS)
    instances: list[Instance] = []
    assignments: list[str] = []
    for _ in range(n_instances):
        name = names[int(rng.integers(0, len(names)))]
        coordinates = DISTRIBUTIONS[name](n_cities, rng)
        if coordinates.shape != (n_cities, 2) or not np.isfinite(coordinates).all():
            raise ValueError(f'{name} produced invalid coordinates: {coordinates.shape}')
        distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
        instances.append((coordinates, distances))
        assignments.append(name)
    return instances, assignments


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate explicit OOD TSP dataset suites.')
    parser.add_argument('--sizes', type=int, nargs='+', default=list(DEFAULT_SIZES))
    parser.add_argument('--n-instances', type=int, default=DEFAULT_N_INSTANCES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--suite', choices=tuple(SUITES), default='mixture')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.n_instances <= 0:
        parser.error('--n-instances must be positive')
    if any(size < 4 for size in args.sizes):
        parser.error('each size must be at least 4')

    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / 'ood_test_datasets' / args.suite
    args.output_dir.mkdir(parents=True, exist_ok=True)
    families = SUITES[args.suite]
    seed_sequence = np.random.SeedSequence(args.seed)
    size_sequences = seed_sequence.spawn(len(args.sizes))
    generated = []
    assignments_by_size: dict[str, list[str]] = {}

    for size, size_sequence in zip(args.sizes, size_sequences):
        rng = np.random.default_rng(size_sequence)
        instances, assignments = generate_mixed_instances(
            args.n_instances,
            size,
            rng,
            families=families,
        )
        output_path = args.output_dir / f'size_{size}.pkl'
        with output_path.open('wb') as f:
            pickle.dump(instances, f, protocol=pickle.HIGHEST_PROTOCOL)
        counts = dict(sorted(Counter(assignments).items()))
        assignments_by_size[str(size)] = assignments
        generated.append({
            'size': size,
            'path': str(output_path.resolve()),
            'bytes': output_path.stat().st_size,
            'sha256': _sha256(output_path),
            'distribution_counts': counts,
        })
        print(f'[OOD] suite={args.suite}, {args.n_instances} instances, size={size}: {counts}')

    metadata = {
        'suite': args.suite,
        'distribution': (
            f'{families[0]}_ood'
            if len(families) == 1 else 'uniform_categorical_mixture_ood'
        ),
        'reference_distribution': 'iid_uniform_[0,1]^2',
        'seed': args.seed,
        'n_instances': args.n_instances,
        'selection': (
            'fixed family for every instance'
            if len(families) == 1
            else 'independent uniform categorical draw per instance'
        ),
        'distribution_families': list(families),
        'coordinate_bounds': [0.0, 1.0],
        'assignments_by_size': assignments_by_size,
        'datasets': generated,
    }
    metadata_path = args.output_dir / 'metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(f'[OOD] Metadata: {metadata_path.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
