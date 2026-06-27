from __future__ import annotations

import argparse
import contextlib
import io
import importlib
import inspect
import os
import sys
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PROFILER_BY_METHOD = {
    'AdvEoH': 'AdvEoHProfiler',
    'EoH': 'EoHProfiler',
    'FunSearch': 'FunSearchProfiler',
    'LHNS': 'LHNSProfiler',
    'MCTS_AHD': 'MAProfiler',
    'MEoH': 'MEoHProfiler',
    'MLES': 'MLESProfiler',
    'MOEAD': 'MOEADProfiler',
    'NSGA2': 'NSGA2Profiler',
    'PartEvo': 'PartEvoProfiler',
    'ReEvo': 'ReEvoProfiler',
}


def quiet_import_module(module_name: str):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return importlib.import_module(module_name)


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f'Cannot find config file: {path}')
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_key_value(raw: str) -> tuple[str, Any]:
    if '=' not in raw:
        raise argparse.ArgumentTypeError(f'Expected KEY=VALUE, got: {raw}')
    key, value = raw.split('=', 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f'Empty key in override: {raw}')
    try:
        parsed_value = yaml.safe_load(value)
    except yaml.YAMLError:
        parsed_value = value
    return key, parsed_value


def apply_overrides(config: dict[str, Any], overrides: list[tuple[str, Any]] | None) -> None:
    for key, value in overrides or []:
        config[key] = value


def resolve_attr(module_names: list[str], attr_name: str) -> type:
    errors = []
    for module_name in module_names:
        try:
            module = quiet_import_module(module_name)
        except Exception as exc:
            errors.append(f'{module_name}: {type(exc).__name__}: {exc}')
            continue
        attr = getattr(module, attr_name, None)
        if inspect.isclass(attr):
            return attr
    searched = ', '.join(module_names)
    details = '\n'.join(errors)
    raise ValueError(f'Cannot resolve class {attr_name!r}. Searched: {searched}\n{details}')


def resolve_resume_function(method_folder: str, method_name: str):
    resume_func_by_method = {
        'AdvEoH': 'resume_adveoh',
        'EoH': 'resume_eoh',
        'ReEvo': 'resume_reevo',
    }
    func_name = resume_func_by_method.get(method_name)
    if func_name is None:
        raise ValueError(f'Resume is not wired for method {method_name!r}.')
    module = quiet_import_module(f'llm4ad.method.{method_folder}.resume')
    resume_func = getattr(module, func_name, None)
    if not callable(resume_func):
        raise ValueError(f'Cannot resolve resume function {func_name!r}.')
    return resume_func


def filtered_kwargs(cls: type, params: dict[str, Any], blocked: set[str] | None = None) -> dict[str, Any]:
    blocked = blocked or set()
    signature = inspect.signature(cls.__init__)
    allowed = {
        name
        for name, param in signature.parameters.items()
        if name != 'self'
        and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {
        key: value
        for key, value in params.items()
        if key not in blocked and key in allowed
    }


def build_default_log_dir(problem_name: str, method_name: str) -> str:
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    unique = f'{timestamp}_pid{os.getpid()}_{uuid.uuid4().hex[:8]}'
    return str(REPO_ROOT / 'logs' / f'{unique}_{problem_name}_{method_name}')


def resolve_adveoh_adapter_classes(problem: str) -> tuple[type, type]:
    module = quiet_import_module(f'llm4ad.method.adveoh.adapters.{problem}')
    classes = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__
    ]
    heuristic_classes = [
        cls for cls in classes
        if cls.__name__.startswith('Adv')
        and cls.__name__.endswith('Evaluation')
        and not cls.__name__.endswith('InstanceGenEvaluation')
    ]
    instance_classes = [
        cls for cls in classes
        if cls.__name__.endswith('InstanceGenEvaluation')
    ]
    if not heuristic_classes or not instance_classes:
        raise ValueError(
            f'Cannot find AdvEoH adapter classes for problem {problem!r}. '
            f'Expected Adv...Evaluation and ...InstanceGenEvaluation in llm4ad.method.adveoh.adapters.{problem}.'
        )
    return heuristic_classes[0], instance_classes[0]


def add_common_overrides(args: argparse.Namespace, method_config: dict[str, Any]) -> None:
    common = {
        'max_sample_nums': args.max_sample_nums,
        'max_generations': args.max_generations,
        'pop_size': args.pop_size,
        'num_evaluators': args.num_evaluators,
    }
    for key, value in common.items():
        if value is not None:
            method_config[key] = value

    if args.num_samplers is not None:
        method_config['num_samplers'] = args.num_samplers

    if args.debug:
        method_config['debug_mode'] = True

    if 'num_samplers' not in method_config and 'num_evaluators' in method_config:
        method_config['num_samplers'] = method_config['num_evaluators']


def apply_train_size_override(args: argparse.Namespace, eval_config: dict[str, Any]) -> None:
    if args.train_size is None:
        return
    eval_config['load_from_file'] = True
    eval_config['dataset_split'] = 'train'
    eval_config['dataset_size'] = args.train_size


def sync_instance_problem_size(inst_eval_cls: type,
                               inst_eval_kwargs: dict[str, Any],
                               eval_config: dict[str, Any]) -> None:
    if 'problem_size' in inst_eval_kwargs:
        return
    if 'dataset_size' not in eval_config:
        return
    signature = inspect.signature(inst_eval_cls.__init__)
    if 'problem_size' in signature.parameters:
        inst_eval_kwargs['problem_size'] = eval_config['dataset_size']


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run LLM4AD from the command line and optionally evaluate the best heuristic on fixed test data.'
    )
    parser.add_argument('--method', default='mcts_ahd', help='Method folder under llm4ad/method.')
    parser.add_argument('--problem', default='tsp_construct', help='Problem folder under llm4ad/task/optimization.')

    parser.add_argument('--llm', default='HttpsApi', help='LLM class name.')
    parser.add_argument('--host', default='https://api.vectorengine.ai/v1', help='OpenAI-compatible base URL or host.')
    parser.add_argument('--model', default='gpt-5-nano', help='Model name.')
    parser.add_argument('--key', default=None, help='API key. Prefer --key-env for shell history safety.')
    parser.add_argument('--key-env', default='LLM_API_KEY', help='Environment variable containing the API key.')
    parser.add_argument('--llm-timeout', type=int, default=120, help='LLM API timeout in seconds.')
    parser.add_argument('--inst-host', default=None, help='AdvEoH instance-generator LLM host. Defaults to --host.')
    parser.add_argument('--inst-model', default=None, help='AdvEoH instance-generator model. Defaults to --model.')
    parser.add_argument('--inst-key', default=None, help='AdvEoH instance-generator API key. Defaults to --key.')
    parser.add_argument('--inst-key-env', default=None, help='AdvEoH instance-generator key env var. Defaults to --key-env.')
    parser.add_argument('--inst-llm-timeout', type=int, default=None, help='AdvEoH instance-generator timeout. Defaults to --llm-timeout.')

    parser.add_argument('--profiler', default=None, help='Profiler class name. Defaults to the profiler for the method.')
    parser.add_argument('--log-dir', default=None, help='Output log directory.')
    parser.add_argument('--log-style', choices=['simple', 'complex'], default='complex')

    parser.add_argument('--max-sample-nums', type=int, default=None)
    parser.add_argument('--max-generations', type=int, default=None)
    parser.add_argument('--pop-size', type=int, default=None)
    parser.add_argument(
        '--num-samplers',
        type=int,
        default=None,
        help=(
            'Number of sampler threads for methods that support it. '
            'Use method-specific sample budget parameters to change total generated samples.'
        ),
    )
    parser.add_argument('--num-evaluators', type=int, default=None)
    parser.add_argument('--debug', action='store_true')

    parser.add_argument(
        '--method-param',
        action='append',
        type=parse_key_value,
        help='Override a method parameter, for example --method-param alpha=0.7',
    )
    parser.add_argument(
        '--eval-param',
        action='append',
        type=parse_key_value,
        help='Override an evaluation parameter, for example --eval-param problem_size=100',
    )

    parser.add_argument('--dry-run', action='store_true', help='Print resolved config without calling the LLM.')
    parser.add_argument(
        '--train-size',
        type=int,
        default=None,
        help='Use train_datasets/size_<N>.pkl as the fixed training set for problem size N.',
    )
    parser.add_argument('--skip-fixed-test-eval', action='store_true', help='Do not run fixed test evaluation after training.')
    parser.add_argument('--eval-only-log-dir', default=None, help='Only evaluate the best sample in an existing log dir.')
    parser.add_argument('--fixed-test-n-instance', type=int, default=64,
                        help='Number of fixed test instances to evaluate per test size.')
    parser.add_argument('--fixed-test-timeout-seconds', type=float, default=240,
                        help='Timeout for each fixed test evaluation. Use <= 0 to keep the problem default.')
    parser.add_argument('--resume-log-dir', default=None, help='Resume a previous run from this log directory.')
    return parser


def main() -> int:
    args = make_parser().parse_args()
    fixed_eval_module = quiet_import_module('llm4ad.task.optimization.fixed_test_eval')
    evaluate_best_on_fixed_test_datasets = fixed_eval_module.evaluate_best_on_fixed_test_datasets

    method_yaml = REPO_ROOT / 'llm4ad' / 'method' / args.method / 'paras.yaml'
    problem_yaml = REPO_ROOT / 'llm4ad' / 'task' / 'optimization' / args.problem / 'paras.yaml'
    method_config = load_yaml(method_yaml)
    eval_config = load_yaml(problem_yaml)

    apply_overrides(method_config, args.method_param)
    apply_overrides(eval_config, args.eval_param)
    add_common_overrides(args, method_config)
    apply_train_size_override(args, eval_config)

    method_name = method_config['name']
    eval_name = eval_config['name']
    profiler_name = args.profiler or DEFAULT_PROFILER_BY_METHOD.get(method_name, 'ProfilerBase')
    if args.resume_log_dir and args.eval_only_log_dir:
        raise ValueError('--resume-log-dir cannot be used with --eval-only-log-dir.')
    if args.resume_log_dir and args.log_dir and os.path.abspath(args.resume_log_dir) != os.path.abspath(args.log_dir):
        raise ValueError('--resume-log-dir and --log-dir must match when both are provided.')
    log_dir = args.eval_only_log_dir or args.resume_log_dir or args.log_dir or build_default_log_dir(eval_name, method_name)

    if eval_config.get('load_from_file') is True:
        eval_config.setdefault('dataset_split', 'train')

    llm_cls = resolve_attr(
        ['llm4ad.tools.llm', 'llm4ad.tools.llm.llm_api_https'],
        args.llm,
    )
    method_cls = resolve_attr(
        [f'llm4ad.method.{args.method}', 'llm4ad.method'],
        method_name,
    )
    eval_cls = resolve_attr(
        [
            f'llm4ad.task.optimization.{args.problem}',
            f'llm4ad.task.optimization.{args.problem}.evaluation',
            'llm4ad.task',
        ],
        eval_name,
    )
    profiler_cls = resolve_attr(
        [
            f'llm4ad.method.{args.method}.profiler',
            f'llm4ad.method.{args.method}',
            'llm4ad.tools.profiler',
            'llm4ad.method',
        ],
        profiler_name,
    )

    method_kwargs = filtered_kwargs(method_cls, method_config, blocked={'name'})
    eval_kwargs = filtered_kwargs(eval_cls, eval_config, blocked={'name'})
    if args.resume_log_dir:
        method_kwargs['resume_mode'] = True

    print('[CLI] Method:', method_name, method_kwargs)
    print('[CLI] Problem:', eval_name, eval_kwargs)
    print('[CLI] LLM:', args.llm, {'host': args.host, 'model': args.model, 'timeout': args.llm_timeout})
    print('[CLI] Profiler:', profiler_name)
    print('[CLI] Log dir:', log_dir)
    if args.resume_log_dir:
        print('[CLI] Resume log dir:', args.resume_log_dir)

    if method_name == 'AdvEoH':
        heu_eval_cls, inst_eval_cls = resolve_adveoh_adapter_classes(args.problem)
        heu_eval_kwargs = filtered_kwargs(heu_eval_cls, eval_config, blocked={'name'})
        inst_eval_kwargs = filtered_kwargs(inst_eval_cls, eval_config, blocked={'name', 'load_from_file', 'dataset_split', 'dataset_size', 'dataset_file'})
        sync_instance_problem_size(inst_eval_cls, inst_eval_kwargs, eval_config)
        print('[CLI] AdvEoH heuristic evaluation:', heu_eval_cls.__name__, heu_eval_kwargs)
        print('[CLI] AdvEoH instance evaluation:', inst_eval_cls.__name__, inst_eval_kwargs)

    if args.dry_run:
        print('[CLI] Dry run only; no LLM call was made.')
        return 0

    if args.eval_only_log_dir:
        fixed_test_timeout = (
            args.fixed_test_timeout_seconds
            if args.fixed_test_timeout_seconds and args.fixed_test_timeout_seconds > 0
            else None
        )
        result = evaluate_best_on_fixed_test_datasets(
            eval_cls,
            eval_kwargs,
            args.eval_only_log_dir,
            n_test_instance=args.fixed_test_n_instance,
            timeout_seconds=fixed_test_timeout,
        )
        if result is None:
            print('[CLI] No fixed test result was produced.')
        return 0

    api_key = args.key or os.environ.get(args.key_env)
    if not api_key:
        raise RuntimeError(f'Missing API key. Set ${args.key_env} or pass --key.')

    if method_name == 'AdvEoH':
        inst_key_env = args.inst_key_env or args.key_env
        inst_api_key = args.inst_key or args.key or os.environ.get(inst_key_env)
        if not inst_api_key:
            raise RuntimeError(f'Missing AdvEoH instance-generator API key. Set ${inst_key_env} or pass --inst-key.')

        llm_heu = llm_cls(
            host=args.host,
            key=api_key,
            model=args.model,
            timeout=args.llm_timeout,
        )
        llm_inst = llm_cls(
            host=args.inst_host or args.host,
            key=inst_api_key,
            model=args.inst_model or args.model,
            timeout=args.inst_llm_timeout or args.llm_timeout,
        )
        heu_evaluation = heu_eval_cls(**heu_eval_kwargs)
        inst_evaluation = inst_eval_cls(**inst_eval_kwargs)
        profiler = profiler_cls(
            log_dir=log_dir,
            log_style=args.log_style,
            create_random_path=False,
        )
        method = method_cls(
            llm_heu=llm_heu,
            llm_inst=llm_inst,
            heu_evaluation=heu_evaluation,
            inst_evaluation=inst_evaluation,
            profiler=profiler,
            **method_kwargs,
        )
        if args.resume_log_dir:
            resume_func = resolve_resume_function(args.method, method_name)
            resume_func(method, args.resume_log_dir)
        method.run()
        if not args.skip_fixed_test_eval:
            fixed_test_timeout = (
                args.fixed_test_timeout_seconds
                if args.fixed_test_timeout_seconds and args.fixed_test_timeout_seconds > 0
                else None
            )
            result = evaluate_best_on_fixed_test_datasets(
                eval_cls,
                eval_kwargs,
                log_dir,
                n_test_instance=args.fixed_test_n_instance,
                timeout_seconds=fixed_test_timeout,
            )
            if result is None:
                print('[CLI] No fixed test result was produced.')
        print('[CLI] Done. Log dir:', log_dir)
        return 0

    llm = llm_cls(
        host=args.host,
        key=api_key,
        model=args.model,
        timeout=args.llm_timeout,
    )
    evaluation = eval_cls(**eval_kwargs)
    profiler = profiler_cls(
        log_dir=log_dir,
        log_style=args.log_style,
        create_random_path=False,
    )
    method = method_cls(
        llm=llm,
        evaluation=evaluation,
        profiler=profiler,
        **method_kwargs,
    )

    if args.resume_log_dir:
        resume_func = resolve_resume_function(args.method, method_name)
        resume_func(method, args.resume_log_dir)

    method.run()

    if not args.skip_fixed_test_eval:
        fixed_test_timeout = (
            args.fixed_test_timeout_seconds
            if args.fixed_test_timeout_seconds and args.fixed_test_timeout_seconds > 0
            else None
        )
        result = evaluate_best_on_fixed_test_datasets(
            eval_cls,
            eval_kwargs,
            log_dir,
            n_test_instance=args.fixed_test_n_instance,
            timeout_seconds=fixed_test_timeout,
        )
        if result is None:
            print('[CLI] No fixed test result was produced.')

    print('[CLI] Done. Log dir:', log_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
