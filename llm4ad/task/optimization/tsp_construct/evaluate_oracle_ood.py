"""Evaluate final-population Oracle@K on explicit TSP OOD suites.

Oracle@K is computed per instance: every final-population heuristic is run on
the same instance, the shortest tour is selected, and only then are the 64
per-instance minima averaged.  It is an upper bound on portfolio quality, not a
deployable selector score.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import json
import math
import os
import pickle
import re
import statistics
import time
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = Path(__file__).resolve().parent / 'ood_test_datasets'
DEFAULT_SUITES = ('clustered', 'diagonal', 'mixture')
DEFAULT_SIZES = (20, 50, 100, 200)
DEFAULT_RUNS = {
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

# This final-population individual computes weighted betweenness on a graph that
# is constant throughout one tour.  Cache that exact calculation per instance.
WEIGHTED_BETWEENNESS_SHA256 = (
    '455d9317738de19fe1c853d017360a39eb10f45e0ae90dc5042855714cd85db5'
)
FINITE_NEIGHBOR_DENSITY_SHA256 = (
    '5ac6ea159116e72287f10f2e47df5116fef7d9081f91ebe658152026741eb71f'
)
FINITE_NEIGHBOR_MAP_SHA256 = (
    '1886bd93c22dbcd18dfef752b0cf9ef46d8b9b450d1248fd5bedcc25090de213'
)
COMPLETE_CLUSTERING_LIST_SHA256 = (
    'd3e1ee29dbd86b38a28ef78b0d5c36eb48ab1172204c49758d79460be6e9e3e0'
)


def _generation(path: Path) -> int:
    match = re.search(r'(\d+)$', path.stem)
    return int(match.group(1)) if match else -1


def _load_final_population(method: str, log_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    population_dir = log_dir / ('heuristic_pop' if method == 'AdvEoH' else 'population')
    checkpoints = sorted(population_dir.glob('pop_*.json'), key=_generation)
    if not checkpoints:
        raise FileNotFoundError(f'No population checkpoint in {population_dir}')
    checkpoint = checkpoints[-1]
    records = json.loads(checkpoint.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError(f'Invalid or empty population: {checkpoint}')
    return checkpoint, records


def _weighted_betweenness_cache(program: str) -> str:
    start = program.index('    G = nx.Graph()')
    marker = "    betweenness_centrality = nx.betweenness_centrality(G, weight='weight', endpoints=False)"
    end = program.index(marker, start) + len(marker)
    replacement = '''    _cache = getattr(select_next_node, '_centrality_cache', {})
    _cache_key = id(distance_matrix)
    _cached = _cache.get(_cache_key)
    if _cached is None:
        from scipy.sparse.csgraph import shortest_path
        _weights = 1 / (distance_matrix + 1e-5)
        np.fill_diagonal(_weights, 0.0)
        _, _predecessors = shortest_path(
            _weights, directed=False, method='D', return_predecessors=True
        )
        _counts = np.zeros(num_nodes, dtype=float)
        for _source in range(num_nodes):
            for _target in range(_source + 1, num_nodes):
                _node = _target
                while _predecessors[_source, _node] not in (-9999, _source):
                    _node = int(_predecessors[_source, _node])
                    _counts[_node] += 1.0
        _scale = 2.0 / ((num_nodes - 1) * (num_nodes - 2))
        degree_centrality = {i: 1.0 for i in range(num_nodes)}
        betweenness_centrality = {i: _counts[i] * _scale for i in range(num_nodes)}
        _cache[_cache_key] = (degree_centrality, betweenness_centrality)
        select_next_node._centrality_cache = _cache
    else:
        degree_centrality, betweenness_centrality = _cached'''
    return program[:start] + replacement + program[end:]


def _finite_neighbor_density(program: str) -> str:
    old = '''        neighbors = np.where(distance_matrix[node] < np.inf)[0]
        unvisited_neighbors = [n for n in neighbors if n in unvisited_nodes]
        local_density = len(unvisited_neighbors)'''
    new = '''        local_density = len(unvisited_nodes)'''
    if old not in program:
        raise ValueError('finite-neighbor density pattern not found')
    return program.replace(old, new)


def _finite_neighbor_map(program: str) -> str:
    start = program.index('    # Precompute cluster density')
    end = program.index('    total_unvisited = len(unvisited_nodes)', start)
    replacement = '''    # Every off-diagonal distance is finite, so each node is
    # connected to every other node in the complete distance matrix.
    _all_neighbor_count = distance_matrix.shape[0] - 1
    _future_density = (len(unvisited_nodes) - 1) / (_all_neighbor_count + 1e-5)

'''
    program = program[:start] + replacement + program[end:]
    old = '''        # Future potential: how many unvisited neighbors
        future_density = len(neighbors_map[node]) / (len(get_neighbors(node)) + 1e-5)

        # Clustering density: how connected the node is within unvisited set
        cluster_density = future_density'''
    new = '''        # Clustering density in the complete finite-distance graph.
        cluster_density = _future_density'''
    if old not in program:
        raise ValueError('finite-neighbor map pattern not found')
    return program.replace(old, new)


def _complete_clustering_list(program: str) -> str:
    start = program.index('    clustering_coeffs = []')
    end = program.index('    max_clustering = max(clustering_coeffs) + 1e-12', start)
    replacement = '''    _clustering = 1.0 if len(unvisited_nodes) >= 3 else 0.0
    clustering_coeffs = [_clustering] * len(unvisited_nodes)
'''
    return program[:start] + replacement + program[end:]


class _CompleteGraphSimplifier(ast.NodeTransformer):
    """Apply exact complete-graph identities used by known MCTS individuals."""

    def __init__(self, total_nodes_name: str):
        self.total_nodes_name = total_nodes_name

    @staticmethod
    def _attribute_name(node: ast.AST) -> str | None:
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            return None
        return f'{node.value.id}.{node.attr}'

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        name = self._attribute_name(node.func)
        if name == 'nx.betweenness_centrality':
            weighted = any(keyword.arg == 'weight' for keyword in node.keywords)
            if not weighted:
                expression = (
                    "{i: 0.0 for i in range(" + self.total_nodes_name + ")}"
                )
                return ast.copy_location(ast.parse(expression, mode='eval').body, node)
        if name == 'G.degree':
            expression = 'enumerate(np.sum(distance_matrix, axis=1))'
            return ast.copy_location(ast.parse(expression, mode='eval').body, node)
        return node


def _optimise_complete_graph_program(program: str) -> tuple[str, list[str]]:
    digest = hashlib.sha256(program.encode()).hexdigest()
    if digest == WEIGHTED_BETWEENNESS_SHA256:
        return _weighted_betweenness_cache(program), ['scipy_exact_weighted_betweenness']
    if digest == FINITE_NEIGHBOR_DENSITY_SHA256:
        return _finite_neighbor_density(program), ['simplify_finite_neighbor_density']
    if digest == FINITE_NEIGHBOR_MAP_SHA256:
        return _finite_neighbor_map(program), ['simplify_finite_neighbor_map']
    if digest == COMPLETE_CLUSTERING_LIST_SHA256:
        return _complete_clustering_list(program), ['simplify_complete_clustering_list']
    if 'networkx' not in program:
        return program, []

    tree = ast.parse(program)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assigned_names = {
        target.id
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    total_nodes_name = 'total_nodes' if 'total_nodes' in assigned_names else 'num_nodes'
    function = _CompleteGraphSimplifier(total_nodes_name).visit(function)
    ast.fix_missing_locations(function)

    changes: list[str] = []
    body = function.body

    # On these continuous OOD datasets every pair of distinct cities has a
    # non-zero distance.  The hand-written complete-neighbor clustering loop is
    # therefore exactly 1 when at least three nodes remain, otherwise 0.
    for index in range(len(body) - 1):
        node, next_node = body[index:index + 2]
        target_names = {
            target.id for target in getattr(node, 'targets', []) if isinstance(target, ast.Name)
        }
        if 'clustering_coeffs' in target_names and isinstance(next_node, ast.For):
            replacement = ast.parse(
                "clustering_coeffs = {int(node): (1.0 if len(unvisited_nodes) >= 3 else 0.0) "
                "for node in unvisited_nodes}"
            ).body[0]
            body[index:index + 2] = [replacement]
            changes.append('simplify_complete_neighbor_clustering')
            break

    # Remove graph construction if all consumers were simplified away.
    graph_indices = []
    for index, node in enumerate(body):
        is_graph_assignment = (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == 'G' for target in node.targets)
        )
        if is_graph_assignment:
            graph_indices = [index]
            if index + 1 < len(body) and isinstance(body[index + 1], ast.For):
                graph_indices.append(index + 1)
            break
    if graph_indices:
        remainder = [node for i, node in enumerate(body) if i not in graph_indices]
        if not any(
            isinstance(node, ast.Name) and node.id == 'G' and isinstance(node.ctx, ast.Load)
            for statement in remainder for node in ast.walk(statement)
        ):
            body[:] = remainder
            changes.append('remove_dead_complete_graph_build')

    ast.fix_missing_locations(tree)
    optimised = ast.unparse(tree)
    if optimised != program and not changes:
        changes.append('simplify_unweighted_complete_graph_centrality')
    elif optimised != program:
        changes.append('simplify_unweighted_complete_graph_centrality')
    return optimised, changes


def _callable(program: str):
    namespace: dict[str, Any] = {}
    exec('import numpy as np\n' + program, namespace)
    function = namespace.get('select_next_node')
    if not callable(function):
        raise ValueError('select_next_node was not defined')
    return function


def _tour_lengths(program: str, dataset_path: str, limit: int | None = None) -> list[float]:
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    if limit is not None:
        dataset = dataset[:limit]
    heuristic = _callable(program)
    lengths: list[float] = []

    for coordinates, distance_matrix in dataset:
        n = len(coordinates)
        neighbor_matrix = np.argsort(distance_matrix, axis=1)
        route = np.full(n, -1, dtype=int)
        route[0] = 0
        current_node = 0
        for step in range(1, n - 1):
            near_nodes = neighbor_matrix[current_node][1:]
            unvisited = near_nodes[~np.isin(near_nodes, route[:step])]
            selected = heuristic(current_node, 0, unvisited, distance_matrix)
            try:
                selected = int(selected)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f'invalid selected node {selected!r}') from exc
            if not np.any(unvisited == selected):
                raise ValueError(f'selected node {selected} is not unvisited')
            route[step] = selected
            current_node = selected
        remaining = np.setdiff1d(np.arange(n), route[:-1], assume_unique=False)
        if len(remaining) != 1:
            raise ValueError(f'invalid route left {len(remaining)} remaining nodes')
        route[-1] = int(remaining[0])
        tour_length = float(np.sum(distance_matrix[route, np.roll(route, -1)]))
        if not math.isfinite(tour_length):
            raise ValueError('non-finite tour length')
        lengths.append(tour_length)
    return lengths


def _evaluate_individual(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    program = task['program']
    optimised, optimisations = _optimise_complete_graph_program(program)
    error = None
    try:
        tour_lengths = _tour_lengths(optimised, task['dataset_path'])
    except Exception as exc:
        tour_lengths = None
        error = f'{type(exc).__name__}: {exc}'
    return {
        'task_id': task['task_id'],
        'method': task['method'],
        'run': task['run'],
        'individual': task['individual'],
        'dataset_key': task['dataset_key'],
        'program_sha256': hashlib.sha256(program.encode()).hexdigest(),
        'optimisations': optimisations,
        'tour_lengths': tour_lengths,
        'error': error,
        'elapsed_seconds': time.perf_counter() - started,
    }


def _validate_optimisations(tasks: list[dict[str, Any]], dataset_path: Path) -> list[dict[str, Any]]:
    validations = []
    seen = set()
    for task in tasks:
        program = task['program']
        digest = hashlib.sha256(program.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        optimised, changes = _optimise_complete_graph_program(program)
        if not changes:
            continue
        original = _tour_lengths(program, str(dataset_path), limit=2)
        transformed = _tour_lengths(optimised, str(dataset_path), limit=2)
        matched = bool(np.allclose(original, transformed, rtol=0.0, atol=1e-12))
        validations.append({
            'program_sha256': digest,
            'changes': changes,
            'matched': matched,
            'original': original,
            'transformed': transformed,
        })
        if not matched:
            raise AssertionError(f'Optimisation changed behavior for {digest}')
    return validations


def _aggregate(
    raw: list[dict[str, Any]],
    run_info: list[dict[str, Any]],
    suites: list[str],
    sizes: list[int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_run = []
    by_run = {(item['method'], item['run']): item for item in run_info}
    result_map = {
        (item['method'], item['run'], item['individual'], item['dataset_key']): item
        for item in raw
    }

    for (method, run), info in by_run.items():
        k = info['k']
        for suite in suites:
            for size in sizes:
                key = f'{suite}/size_{size}'
                rows = []
                valid_individuals = []
                for individual in range(k):
                    values = result_map[(method, run, individual, key)]['tour_lengths']
                    if values is not None:
                        rows.append(values)
                        valid_individuals.append(individual)
                if not rows:
                    oracle_costs = None
                    winner_counts = {}
                    oracle_mean = None
                else:
                    matrix = np.asarray(rows, dtype=float)
                    winners = np.argmin(matrix, axis=0)
                    oracle = np.min(matrix, axis=0)
                    oracle_costs = oracle.tolist()
                    oracle_mean = float(np.mean(oracle))
                    winner_counts = {
                        str(valid_individuals[index]): int(np.sum(winners == index))
                        for index in sorted(set(winners.tolist()))
                    }
                per_run.append({
                    'method': method,
                    'run': run,
                    'checkpoint': info['checkpoint'],
                    'k': k,
                    'valid_k': len(rows),
                    'suite': suite,
                    'size': size,
                    'oracle_mean_tour_length': oracle_mean,
                    'oracle_instance_tour_lengths': oracle_costs,
                    'winner_counts': winner_counts,
                })

    aggregate: dict[str, Any] = {}
    methods = sorted({row['method'] for row in per_run})
    for method in methods:
        by_suite = {}
        for suite in suites:
            by_size = []
            all_values = []
            for size in sizes:
                values = [
                    row['oracle_mean_tour_length'] for row in per_run
                    if row['method'] == method and row['suite'] == suite
                    and row['size'] == size and row['oracle_mean_tour_length'] is not None
                ]
                all_values.extend(values)
                by_size.append({
                    'size': size,
                    'mean': statistics.fmean(values) if values else None,
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
                    'runs': len(values),
                })
            by_suite[suite] = {
                'by_size': by_size,
                'overall_mean': statistics.fmean(all_values) if all_values else None,
            }
        aggregate[method] = by_suite
    return per_run, aggregate


def _write_report(summary: dict[str, Any], path: Path) -> None:
    lines = [
        '# TSP OOD final-population Oracle@K',
        '',
        '- Oracle is selected independently per test instance; lower tour length is better.',
        '- Values are mean +/- sample stdev over three runs.',
        '- This is an upper bound that assumes an oracle selector, not a deployable score.',
        '',
    ]
    for suite in summary['suites']:
        lines.extend([
            f'## {suite}',
            '',
            '| Method | ' + ' | '.join(f'n={size}' for size in summary['sizes']) + ' | Overall |',
            '|---|' + '|'.join('---:' for _ in summary['sizes']) + '|---:|',
        ])
        for method in sorted(summary['aggregate']):
            item = summary['aggregate'][method][suite]
            cells = []
            for row in item['by_size']:
                cells.append(
                    'FAILED' if row['mean'] is None
                    else f"{row['mean']:.6f} +/- {row['stdev']:.6f}"
                )
            lines.append(
                f"| {method} | " + ' | '.join(cells)
                + f" | {item['overall_mean']:.6f} |"
            )
        lines.append('')
    lines.extend(['## Run populations', ''])
    for item in summary['runs']:
        lines.append(
            f"- {item['method']} `{item['run']}`: K={item['k']}, `{item['checkpoint']}`"
        )
    lines.extend([
        '',
        '## Validation',
        '',
        f"- Behavior-equivalent optimization checks: {len(summary['optimisation_validations'])}",
        f"- All checks matched: {all(v['matched'] for v in summary['optimisation_validations'])}",
        f"- Failed individual/dataset evaluations: {summary['failed_evaluations']}",
        '',
    ])
    path.write_text('\n'.join(lines), encoding='utf-8')


def main() -> int:
    parser = argparse.ArgumentParser(description='Evaluate final-population Oracle@K on TSP OOD data.')
    parser.add_argument('--suites', nargs='+', default=list(DEFAULT_SUITES))
    parser.add_argument('--sizes', type=int, nargs='+', default=list(DEFAULT_SIZES))
    parser.add_argument('--max-workers', type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()

    suites = list(args.suites)
    sizes = list(args.sizes)
    dataset_paths = {
        f'{suite}/size_{size}': str((DATA_ROOT / suite / f'size_{size}.pkl').resolve())
        for suite in suites for size in sizes
    }
    for path in dataset_paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    tasks = []
    runs = []
    for method, relative_dirs in DEFAULT_RUNS.items():
        for relative_dir in relative_dirs:
            log_dir = REPO_ROOT / relative_dir
            checkpoint, records = _load_final_population(method, log_dir)
            programs = [record.get('program') or record.get('function') for record in records]
            if any(not program for program in programs):
                raise ValueError(f'Missing function in {checkpoint}')
            if len(set(programs)) != len(programs):
                raise ValueError(f'Duplicate program in {checkpoint}')
            runs.append({
                'method': method,
                'run': log_dir.name,
                'checkpoint': str(checkpoint),
                'k': len(programs),
            })
            for individual, program in enumerate(programs):
                for dataset_key, dataset_path in dataset_paths.items():
                    task_id = f'{method}|{log_dir.name}|{individual}|{dataset_key}'
                    tasks.append({
                        'task_id': task_id,
                        'method': method,
                        'run': log_dir.name,
                        'individual': individual,
                        'program': program,
                        'dataset_key': dataset_key,
                        'dataset_path': dataset_path,
                    })

    validations = _validate_optimisations(
        tasks,
        DATA_ROOT / 'diagonal' / 'size_20.pkl',
    )
    unique_individuals = sum(item['k'] for item in runs)
    print(f'[Oracle] Loaded {unique_individuals} individuals / {len(tasks)} dataset tasks '
          f'from {len(runs)} runs; '
          f'validated {len(validations)} optimized programs.')

    args.output_dir.mkdir(parents=True, exist_ok=True)
    partial_path = args.output_dir / 'partial_results.json'
    raw = []
    if partial_path.is_file():
        partial = json.loads(partial_path.read_text(encoding='utf-8'))
        if isinstance(partial, list):
            raw = partial
    completed_ids = {item['task_id'] for item in raw}
    pending_tasks = [task for task in tasks if task['task_id'] not in completed_ids]
    if completed_ids:
        print(f'[Oracle] Resuming {len(completed_ids)} completed tasks; '
              f'{len(pending_tasks)} remain.')

    with concurrent.futures.ProcessPoolExecutor(max_workers=max(1, args.max_workers)) as executor:
        futures = [executor.submit(_evaluate_individual, task) for task in pending_tasks]
        for newly_completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            raw.append(result)
            completed = len(completed_ids) + newly_completed
            print(
                f"[Oracle] {completed:02d}/{len(tasks)} {result['method']} "
                f"{result['run']} individual={result['individual']} "
                f"dataset={result['dataset_key']} error={result['error'] is not None} "
                f"elapsed={result['elapsed_seconds']:.1f}s",
                flush=True,
            )
            if newly_completed % 10 == 0:
                partial_path.write_text(json.dumps(raw), encoding='utf-8')
        partial_path.write_text(json.dumps(raw), encoding='utf-8')

    per_run, aggregate = _aggregate(raw, runs, suites, sizes)
    failed = sum(item['error'] is not None for item in raw)
    summary = {
        'metric': 'per-instance Oracle@K mean tour length',
        'suites': suites,
        'sizes': sizes,
        'runs': runs,
        'per_run': per_run,
        'aggregate': aggregate,
        'optimisation_validations': validations,
        'failed_evaluations': failed,
        'raw_individual_results': raw,
    }
    json_path = args.output_dir / 'results.json'
    report_path = args.output_dir / 'report.md'
    json_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    _write_report(summary, report_path)
    print(f'[Oracle] JSON: {json_path}')
    print(f'[Oracle] Report: {report_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
