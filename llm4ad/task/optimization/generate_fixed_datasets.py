"""Generate fixed train and test datasets for construct optimization tasks.

Each task gets one default training dataset and optional fixed-size training
datasets:

    <task_folder>/train_dataset.pkl
    <task_folder>/train_datasets/size_<N>.pkl

and one or more fixed test datasets at different problem sizes:

    <task_folder>/test_datasets/size_<N>.pkl

Run examples:
    python llm4ad/task/optimization/generate_fixed_datasets.py
    python llm4ad/task/optimization/generate_fixed_datasets.py --task tsp_construct
    python llm4ad/task/optimization/generate_fixed_datasets.py --task tsp_construct --split train --train-sizes 20 50 100 200
    python llm4ad/task/optimization/generate_fixed_datasets.py --task tsp_construct --split test --test-sizes 20 50 100 200
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TRAIN_DATASET_FILENAME = 'train_dataset.pkl'
TRAIN_DATASET_DIRNAME = 'train_datasets'
TEST_DATASET_DIRNAME = 'test_datasets'


TASK_SPECS = [
    {
        'name': 'tsp_construct',
        'import': 'llm4ad.task.optimization.tsp_construct.get_instance.GetData',
        'train_args': {'n_instance': 64, 'n_cities': 50},
        'test_n_instance': 64,
        'size_args': ('n_cities',),
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'cvrp_construct',
        'import': 'llm4ad.task.optimization.cvrp_construct.get_instance.GetData',
        'train_args': {'n_instance': 16, 'n_cities': 51, 'capacity': 40},
        'size_args': ('n_cities',),
        'size_offset': 1,
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'ovrp_construct',
        'import': 'llm4ad.task.optimization.ovrp_construct.get_instance.GetData',
        'train_args': {'n_instance': 16, 'n_cities': 51},
        'size_args': ('n_cities',),
        'size_offset': 1,
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'vrptw_construct',
        'import': 'llm4ad.task.optimization.vrptw_construct.get_instance.GetData',
        'train_args': {'n_instance': 16, 'n_cities': 50},
        'size_args': ('n_cities',),
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'bp_1d_construct',
        'import': 'llm4ad.task.optimization.bp_1d_construct.get_instance.GetData',
        'train_args': {'n_instance': 8, 'n_items': 500, 'bin_capacity': 100},
        'size_args': ('n_items',),
        'test_sizes': (100, 500, 1000, 2000),
    },
    {
        'name': 'bp_2d_construct',
        'import': 'llm4ad.task.optimization.bp_2d_construct.get_instance.GetData',
        'train_args': {'n_instance': 8, 'n_items': 100, 'bin_width': 100, 'bin_height': 100},
        'size_args': ('n_items',),
        'test_sizes': (50, 100, 200, 500),
    },
    {
        'name': 'knapsack_construct',
        'import': 'llm4ad.task.optimization.knapsack_construct.get_instance.GetData',
        'train_args': {'n_instance': 32, 'n_items': 50, 'knapsack_capacity': 100},
        'size_args': ('n_items',),
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'jssp_construct',
        'import': 'llm4ad.task.optimization.jssp_construct.get_instance.GetData',
        'train_args': {'n_instance': 16, 'n_jobs': 50, 'n_machines': 10},
        'size_args': ('n_jobs',),
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'qap_construct',
        'import': 'llm4ad.task.optimization.qap_construct.get_instance.GetData',
        'train_args': {'n_instance': 8, 'n_facilities': 20},
        'size_args': ('n_facilities',),
        'test_sizes': (10, 20, 50, 100),
    },
    {
        'name': 'cflp_construct',
        'import': 'llm4ad.task.optimization.cflp_construct.get_instance.GetData',
        'train_args': {
            'n_instance': 16, 'n_facilities': 50, 'n_customers': 50,
            'max_capacity': 100, 'max_demand': 20, 'max_cost': 50,
        },
        'size_args': ('n_facilities', 'n_customers'),
        'test_sizes': (20, 50, 100, 200),
    },
    {
        'name': 'set_cover_construct',
        'import': 'llm4ad.task.optimization.set_cover_construct.get_instance.GetData',
        'train_args': {
            'n_instance': 16, 'n_elements': 50, 'n_subsets': 50, 'max_subset_size': 8,
        },
        'size_args': ('n_elements', 'n_subsets'),
        'test_sizes': (20, 50, 100, 200),
    },
]


def import_class(dotted_path: str):
    """Import a class from a dotted module path like 'pkg.mod.ClassName'."""
    import importlib.util

    module_path, class_name = dotted_path.rsplit('.', 1)
    file_path = os.path.join(REPO_ROOT, *module_path.split('.')) + '.py'
    spec = importlib.util.spec_from_file_location(module_path, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot find {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def args_for_size(spec: dict, size: int, n_instance: int | None = None) -> dict:
    args = dict(spec['train_args'])
    offset = spec.get('size_offset', 0)
    for arg_name in spec['size_args']:
        args[arg_name] = size + offset
    if n_instance is not None:
        args['n_instance'] = n_instance
    return args


def generate_instances(spec: dict, init_args: dict):
    GetDataCls = import_class(spec['import'])
    return GetDataCls(**init_args).generate_instances()


def write_pickle(instances, out_path: str, overwrite: bool) -> bool:
    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] {out_path} already exists")
        return False

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(instances, f)

    n = len(instances) if hasattr(instances, '__len__') else '?'
    size_kb = os.path.getsize(out_path) / 1024
    print(f"[OK]   {n} instances -> {out_path} ({size_kb:.1f} KB)")
    return True


def generate_train(spec: dict,
                   overwrite: bool,
                   train_size: int | None,
                   n_train_instance: int | None) -> bool:
    task_dir = os.path.join(HERE, spec['name'])
    init_args = (
        args_for_size(spec, train_size, n_train_instance)
        if train_size is not None
        else dict(spec['train_args'])
    )
    if n_train_instance is not None:
        init_args['n_instance'] = n_train_instance
    out_path = os.path.join(task_dir, TRAIN_DATASET_FILENAME)
    instances = generate_instances(spec, init_args)
    return write_pickle(instances, out_path, overwrite)


def generate_sized_train(spec: dict,
                         size: int,
                         overwrite: bool,
                         n_train_instance: int | None) -> bool:
    task_dir = os.path.join(HERE, spec['name'])
    init_args = args_for_size(spec, size, n_train_instance)
    out_path = os.path.join(task_dir, TRAIN_DATASET_DIRNAME, f'size_{size}.pkl')
    instances = generate_instances(spec, init_args)
    return write_pickle(instances, out_path, overwrite)


def generate_test(spec: dict,
                  size: int,
                  overwrite: bool,
                  n_test_instance: int | None) -> bool:
    task_dir = os.path.join(HERE, spec['name'])
    init_args = args_for_size(
        spec,
        size,
        n_test_instance if n_test_instance is not None else spec.get('test_n_instance')
    )
    out_path = os.path.join(task_dir, TEST_DATASET_DIRNAME, f'size_{size}.pkl')
    instances = generate_instances(spec, init_args)
    return write_pickle(instances, out_path, overwrite)


def generate_one(spec: dict,
                 split: str,
                 overwrite: bool,
                 train_size: int | None,
                 train_sizes: list[int] | None,
                 test_sizes: list[int] | None,
                 n_train_instance: int | None,
                 n_test_instance: int | None) -> int:
    task_dir = os.path.join(HERE, spec['name'])
    if not os.path.isdir(task_dir):
        print(f"[SKIP] {spec['name']}: task folder not found at {task_dir}")
        return 0

    n_ok = 0
    if split in ('train', 'all'):
        if generate_train(spec, overwrite, train_size, n_train_instance):
            n_ok += 1
        sizes = train_sizes if train_sizes is not None else list(spec['test_sizes'])
        for size in sizes:
            if generate_sized_train(spec, size, overwrite, n_train_instance):
                n_ok += 1

    if split in ('test', 'all'):
        sizes = test_sizes if test_sizes is not None else list(spec['test_sizes'])
        for size in sizes:
            if generate_test(spec, size, overwrite, n_test_instance):
                n_ok += 1

    return n_ok


def main():
    parser = argparse.ArgumentParser(description='Generate fixed train/test benchmark datasets.')
    parser.add_argument('--task', type=str, default=None,
                        help='Only generate for this task (default: all).')
    parser.add_argument('--split', choices=('train', 'test', 'all'), default='all',
                        help='Which dataset split to generate.')
    parser.add_argument('--train-size', type=int, default=None,
                        help='Override the default training problem size.')
    parser.add_argument('--train-sizes', type=int, nargs='+', default=None,
                        help='Override fixed-size train dataset sizes.')
    parser.add_argument('--test-sizes', type=int, nargs='+', default=None,
                        help='Override default test problem sizes.')
    parser.add_argument('--n-train-instance', type=int, default=None,
                        help='Override number of training instances.')
    parser.add_argument('--n-test-instance', type=int, default=None,
                        help='Override number of test instances.')
    parser.add_argument('--no-overwrite', action='store_true',
                        help='Skip files that already exist.')
    args = parser.parse_args()

    specs = TASK_SPECS
    if args.task is not None:
        specs = [s for s in TASK_SPECS if s['name'] == args.task]
        if not specs:
            print(f"Unknown task: {args.task}")
            print("Available:", ', '.join(s['name'] for s in TASK_SPECS))
            sys.exit(1)

    print(f"Generating {args.split} dataset files for {len(specs)} task(s)...")
    n_ok = 0
    for spec in specs:
        try:
            n_ok += generate_one(
                spec=spec,
                split=args.split,
                overwrite=not args.no_overwrite,
                train_size=args.train_size,
                train_sizes=args.train_sizes,
                test_sizes=args.test_sizes,
                n_train_instance=args.n_train_instance,
                n_test_instance=args.n_test_instance,
            )
        except Exception as e:
            print(f"[FAIL] {spec['name']}: {type(e).__name__}: {e}")
            traceback.print_exc()

    print(f"\nDone: {n_ok} dataset file(s) generated.")


if __name__ == '__main__':
    main()
