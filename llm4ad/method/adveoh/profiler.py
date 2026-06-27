from __future__ import annotations

import json
import math
import os
from threading import Lock
from typing import Any, List, Dict, Optional

from .population import Population
from ...base import Function
from ...tools.profiler import TensorboardProfiler, ProfilerBase, WandBProfiler


class AdvEoHProfiler(ProfilerBase):
    """Profiler for AdvEoH.

    Heuristic and instance-generator populations have different evolution
    clocks. The role-specific register methods below log each population only
    when that side is actually evolved.
    """

    def __init__(self,
                 log_dir: Optional[str] = None,
                 *,
                 initial_num_samples: int = 0,
                 log_style: str = 'complex',
                 create_random_path: bool = True,
                 **kwargs):
        super().__init__(log_dir=log_dir,
                         initial_num_samples=initial_num_samples,
                         log_style=log_style,
                         create_random_path=create_random_path,
                         **kwargs)
        self._cur_gen = 0
        self._pop_lock = Lock()
        self._best_role_score = {
            'heuristic': float('-inf'),
            'instance': float('-inf'),
        }
        self._best_role_sample_order = {
            'heuristic': None,
            'instance': None,
        }
        self._token_usage_lock = Lock()
        self._token_usage_event_count = 0
        self._token_usage_total = self._empty_token_usage_total()
        self._token_usage_by_role = {
            'heuristic': self._empty_token_usage_total(),
            'instance': self._empty_token_usage_total(),
        }
        self._last_metrics_token_total = self._empty_token_usage_total()
        if self._log_dir:
            self._heu_ckpt_dir = os.path.join(self._log_dir, 'heuristic_pop')
            self._inst_ckpt_dir = os.path.join(self._log_dir, 'instance_pop')
            self._metrics_path = os.path.join(self._log_dir, 'generation_metrics.json')
            self._token_usage_path = os.path.join(self._log_dir, 'token_usage.json')
            os.makedirs(self._heu_ckpt_dir, exist_ok=True)
            os.makedirs(self._inst_ckpt_dir, exist_ok=True)

    @staticmethod
    def _empty_token_usage_total() -> Dict[str, int]:
        return {
            'requests': 0,
            'estimated_requests': 0,
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }

    @staticmethod
    def _normalise_token_usage(token_usage: Any) -> Dict[str, Any] | None:
        if token_usage is None:
            return None
        if not isinstance(token_usage, dict):
            return None

        normalised = dict(token_usage)
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            try:
                normalised[key] = int(normalised.get(key, 0) or 0)
            except (TypeError, ValueError):
                normalised[key] = 0
        normalised['estimated'] = bool(normalised.get('estimated', False))
        normalised.setdefault('source', 'unknown')
        return normalised

    @staticmethod
    def _add_token_usage_to_total(total: Dict[str, int], token_usage: Dict[str, Any]) -> None:
        total['requests'] += 1
        if token_usage.get('estimated'):
            total['estimated_requests'] += 1
        for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            total[key] += int(token_usage.get(key, 0) or 0)

    @staticmethod
    def _copy_token_total(total: Dict[str, int]) -> Dict[str, int]:
        return {key: int(value) for key, value in total.items()}

    def _token_usage_snapshot_unlocked(self) -> Dict[str, Any]:
        return {
            'total': self._copy_token_total(self._token_usage_total),
            'by_role': {
                role: self._copy_token_total(total)
                for role, total in self._token_usage_by_role.items()
            },
        }

    def _append_token_usage_event_unlocked(self, event: Dict[str, Any]) -> None:
        if not self._log_dir:
            return
        try:
            with open(self._token_usage_path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(event)
        with open(self._token_usage_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def register_token_usage(self, token_usage: Dict[str, Any] | None, *,
                             role: str = 'heuristic',
                             operator: str = 'Unknown',
                             accepted: bool = False,
                             sample_order: int | None = None,
                             reason: str | None = None) -> Dict[str, Any] | None:
        token_usage = self._normalise_token_usage(token_usage)
        if token_usage is None:
            return None
        role = role if role in self._token_usage_by_role else 'heuristic'

        with self._token_usage_lock:
            self._token_usage_event_count += 1
            self._add_token_usage_to_total(self._token_usage_total, token_usage)
            self._add_token_usage_to_total(self._token_usage_by_role[role], token_usage)
            event = {
                'event_index': self._token_usage_event_count,
                'sample_order': sample_order,
                'role': role,
                'operator': operator,
                'accepted': accepted,
                'reason': reason,
                'token_usage': token_usage,
                'cumulative': self._token_usage_snapshot_unlocked(),
            }
            self._append_token_usage_event_unlocked(event)
            return event

    def _token_usage_metrics_snapshot(self) -> Dict[str, Any]:
        with self._token_usage_lock:
            total = self._copy_token_total(self._token_usage_total)
            delta = {
                key: total[key] - self._last_metrics_token_total.get(key, 0)
                for key in total
            }
            self._last_metrics_token_total = self._copy_token_total(total)
            return {
                'token_usage_total': total,
                'token_usage_by_role': {
                    role: self._copy_token_total(role_total)
                    for role, role_total in self._token_usage_by_role.items()
                },
                'token_usage_since_last_metrics': delta,
            }

    def load_token_usage_log(self) -> None:
        if not self._log_dir or not hasattr(self, '_token_usage_path'):
            return
        try:
            with open(self._token_usage_path, 'r') as json_file:
                events = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            return
        if not isinstance(events, list):
            return

        with self._token_usage_lock:
            self._token_usage_event_count = 0
            self._token_usage_total = self._empty_token_usage_total()
            self._token_usage_by_role = {
                'heuristic': self._empty_token_usage_total(),
                'instance': self._empty_token_usage_total(),
            }
            for event in events:
                if not isinstance(event, dict):
                    continue
                token_usage = self._normalise_token_usage(event.get('token_usage'))
                if token_usage is None:
                    continue
                role = event.get('role', 'heuristic')
                role = role if role in self._token_usage_by_role else 'heuristic'
                self._token_usage_event_count = max(
                    self._token_usage_event_count,
                    int(event.get('event_index', 0) or 0),
                )
                self._add_token_usage_to_total(self._token_usage_total, token_usage)
                self._add_token_usage_to_total(self._token_usage_by_role[role], token_usage)
            self._last_metrics_token_total = self._copy_token_total(self._token_usage_total)

    def register_generation(self, heu_pop: Population, inst_pop: Population,
                            gen: int, heldout_score: float | None = None):
        """Backward-compatible combined snapshot using the H generation clock."""
        try:
            self._pop_lock.acquire()
            self._save_population(heu_pop, os.path.join(self._heu_ckpt_dir, f'pop_{gen}.json'))
            self._save_population(inst_pop, os.path.join(self._inst_ckpt_dir, f'pop_{gen}.json'))
            info = {
                'generation': gen,
                'heu_pop_size': len(heu_pop),
                'inst_pop_size': len(inst_pop),
                'heu_best_score': max(
                    (f.score for f in heu_pop.population if f.score is not None),
                    default=None,
                ),
                'inst_best_score': max(
                    (f.score for f in inst_pop.population if f.score is not None),
                    default=None,
                ),
                'heldout_score': heldout_score,
            }
            info.update(self._token_usage_metrics_snapshot())
            self._append_generation_metrics(info)
            print(f'[AdvEoH] Gen {gen}: heu_best={info["heu_best_score"]}, '
                  f'inst_best={info["inst_best_score"]}, '
                  f'heldout={info["heldout_score"]}, '
                  f'tokens+={info["token_usage_since_last_metrics"]["total_tokens"]}, '
                  f'tokens_total={info["token_usage_total"]["total_tokens"]}')
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    @staticmethod
    def _best_population_score(pop: Population):
        return max(
            (f.score for f in pop.population if f.score is not None),
            default=None,
        )

    def register_heuristic_generation(self, heu_pop: Population, gen: int,
                                      heldout_score: float | None = None):
        """Log one heuristic generation after H has evolved."""
        info = {
            'event': 'heuristic_generation',
            'generation': gen,
            'heu_generation': gen,
            'inst_generation': None,
            'heu_pop_size': len(heu_pop),
            'inst_pop_size': None,
            'heu_best_score': self._best_population_score(heu_pop),
            'inst_best_score': None,
            'heldout_score': heldout_score,
        }
        info.update(self._token_usage_metrics_snapshot())
        try:
            self._pop_lock.acquire()
            if self._log_dir:
                self._save_population(
                    heu_pop,
                    os.path.join(self._heu_ckpt_dir, f'pop_{gen}.json'),
                )
            self._append_generation_metrics(info)
            print(f'[AdvEoH] H Gen {gen}: heu_best={info["heu_best_score"]}, '
                  f'heldout={info["heldout_score"]}, '
                  f'tokens+={info["token_usage_since_last_metrics"]["total_tokens"]}, '
                  f'tokens_total={info["token_usage_total"]["total_tokens"]}')
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def register_instance_generation(self, inst_pop: Population, gen: int,
                                     trigger_heuristic_gen: int | None = None):
        """Log one instance-generator generation after G has evolved."""
        info = {
            'event': 'instance_generation',
            'generation': gen,
            'heu_generation': trigger_heuristic_gen,
            'inst_generation': gen,
            'heu_pop_size': None,
            'inst_pop_size': len(inst_pop),
            'heu_best_score': None,
            'inst_best_score': self._best_population_score(inst_pop),
            'heldout_score': None,
        }
        info.update(self._token_usage_metrics_snapshot())
        try:
            self._pop_lock.acquire()
            if self._log_dir:
                self._save_population(
                    inst_pop,
                    os.path.join(self._inst_ckpt_dir, f'pop_{gen}.json'),
                )
            self._append_generation_metrics(info)
            trigger = (
                f' after H Gen {trigger_heuristic_gen}'
                if trigger_heuristic_gen is not None else ''
            )
            print(f'[AdvEoH] G Gen {gen}{trigger}: '
                  f'inst_best={info["inst_best_score"]}, '
                  f'tokens+={info["token_usage_since_last_metrics"]["total_tokens"]}, '
                  f'tokens_total={info["token_usage_total"]["total_tokens"]}')
        finally:
            if self._pop_lock.locked():
                self._pop_lock.release()

    def _append_generation_metrics(self, info: Dict):
        if not self._log_dir:
            return
        try:
            with open(self._metrics_path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(info)
        with open(self._metrics_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def _save_population(self, pop: Population, path: str):
        funcs_json: List[Dict] = []
        for f in pop.population:
            funcs_json.append({
                'algorithm': getattr(f, 'algorithm', ''),
                'function': str(f),
                'score': f.score,
            })
        with open(path, 'w') as json_file:
            json.dump(funcs_json, json_file, indent=4)

    def _append_sample_json(self, filename: str, content: Dict):
        if not self._log_dir:
            return
        if not hasattr(self, '_samples_json_dir'):
            self._samples_json_dir = os.path.join(self._log_dir, 'samples')
        os.makedirs(self._samples_json_dir, exist_ok=True)
        path = os.path.join(self._samples_json_dir, filename)
        try:
            with open(path, 'r') as json_file:
                data = json.load(json_file)
        except (FileNotFoundError, json.JSONDecodeError):
            data = []
        data.append(content)
        with open(path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def _write_json(self, function: Function, program: str = '', *,
                    record_type: str = 'history', record_sep: int = 200,
                    role: str = 'heuristic'):
        if not self._log_dir:
            return
        sample_order = self._num_samples
        content = {
            'sample_order': sample_order,
            'role': role,
            'algorithm': getattr(function, 'algorithm', ''),
            'function': str(function),
            'score': function.score,
            'program': program,
        }
        token_usage = getattr(function, 'token_usage', None)
        if token_usage is not None:
            content['token_usage'] = token_usage
        token_usage_cumulative = getattr(function, 'token_usage_cumulative', None)
        if token_usage_cumulative is not None:
            content['token_usage_cumulative'] = token_usage_cumulative
        token_usage_event_index = getattr(function, 'token_usage_event_index', None)
        if token_usage_event_index is not None:
            content['token_usage_event_index'] = token_usage_event_index
        if record_type == 'history':
            lower_bound = ((sample_order - 1) // record_sep) * record_sep
            upper_bound = lower_bound + record_sep
            self._append_sample_json(f'samples_{lower_bound + 1}~{upper_bound}.json', content)
            return

        if role == 'instance':
            self._append_sample_json('samples_best_generator.json', content)
        else:
            self._append_sample_json('samples_best.json', content)
            self._append_sample_json('samples_best_heuristic.json', content)

    def record_best_heldout_heuristic(self, function: Function, program: str,
                                      heldout_score: float | None):
        """Append the final heldout-selected heuristic without counting a new sample."""
        if not self._log_dir:
            return
        content = {
            'sample_order': self._num_samples,
            'role': 'heuristic',
            'algorithm': getattr(function, 'algorithm', ''),
            'function': str(function),
            'score': heldout_score,
            'heldout_score': heldout_score,
            'fixed_score': heldout_score,
            'phase_score': getattr(function, 'score', None),
            'selection_metric': 'heldout_score',
            'program': program,
        }
        token_usage = getattr(function, 'token_usage', None)
        if token_usage is not None:
            content['token_usage'] = token_usage
        self._append_sample_json('samples_best.json', content)
        self._append_sample_json('samples_best_heuristic.json', content)

    @staticmethod
    def _score_is_valid(score) -> bool:
        try:
            return score is not None and math.isfinite(float(score))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _role_label(role: str) -> str:
        return 'generator' if role == 'instance' else role

    def _record_and_print_verbose(self, function: Function, program: str = '',
                                  *, role: str = 'heuristic'):
        function_str = str(function).strip('\n')
        score = function.score
        role = role if role in self._best_role_score else 'heuristic'
        role_label = self._role_label(role)

        if self._score_is_valid(score) and score > self._best_role_score[role]:
            self._best_role_score[role] = score
            self._best_role_sample_order[role] = self._num_samples
            self._write_json(function, record_type='best', program=program, role=role)

        if self._log_style == 'complex':
            print(f'================= Evaluated Function =================')
            print(f'{function_str}')
            print(f'------------------------------------------------------')
            print(f'Role         : {role_label}')
            print(f'Operator     : {function.operator}')
            print(f'Score        : {str(score)}')
            print(f'Sample time  : {str(function.sample_time)}')
            print(f'Evaluate time: {str(function.evaluate_time)}')
            token_usage = getattr(function, 'token_usage', None)
            if token_usage is not None:
                print(
                    'Token usage  : '
                    f"prompt={token_usage.get('prompt_tokens')}, "
                    f"completion={token_usage.get('completion_tokens')}, "
                    f"total={token_usage.get('total_tokens')}, "
                    f"source={token_usage.get('source')}"
                )
            print(f'Sample orders: {str(self._num_samples)}')
            print(f'------------------------------------------------------')
            print(f'Current best heuristic score: {self._best_role_score["heuristic"]}')
            print(f'Current best generator score: {self._best_role_score["instance"]}')
            print(f'======================================================\n')
        else:
            best = self._best_role_score[role]
            token_usage = getattr(function, 'token_usage', None)
            token_text = ''
            if token_usage is not None:
                token_text = f"     Tokens={token_usage.get('total_tokens')}"
            if score is None:
                print(f'Sample{self._num_samples} [{role_label}]: Score=None{token_text}    Cur_Best_{role_label}={best: .3f}')
            else:
                print(f'Sample{self._num_samples} [{role_label}]: Score={score: .3f}{token_text}     Cur_Best_{role_label}={best: .3f}')

        if score is not None:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

    def register_function(self, function: Function, program: str = '',
                          *, role: str = 'heuristic',
                          token_usage: Dict[str, Any] | None = None,
                          **kwargs):
        try:
            self._register_function_lock.acquire()
            self._num_samples += 1
            token_usage = token_usage if token_usage is not None else getattr(function, 'token_usage', None)
            if token_usage is not None:
                function.token_usage = token_usage
                event = self.register_token_usage(
                    token_usage,
                    role=role,
                    operator=getattr(function, 'operator', 'Unknown'),
                    accepted=True,
                    sample_order=self._num_samples,
                )
                if event is not None:
                    function.token_usage_event_index = event['event_index']
                    function.token_usage_cumulative = event['cumulative']['total']
            self._record_and_print_verbose(function, program=program, role=role)
            self._write_json(function, program, role=role)
        finally:
            self._register_function_lock.release()


class AdvEoHTensorboardProfiler(TensorboardProfiler, AdvEoHProfiler):

    def __init__(self,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples: int = 0,
                 log_style: str = 'complex',
                 create_random_path: bool = True,
                 **kwargs):
        AdvEoHProfiler.__init__(self, log_dir=log_dir,
                                initial_num_samples=initial_num_samples,
                                log_style=log_style,
                                create_random_path=create_random_path,
                                **kwargs)
        TensorboardProfiler.__init__(self, log_dir=log_dir,
                                     initial_num_samples=initial_num_samples,
                                     log_style=log_style,
                                     create_random_path=create_random_path,
                                     **kwargs)

    def finish(self):
        if self._log_dir:
            self._writer.close()

    def register_function(self, function: Function, program: str = '',
                          *, role: str = 'heuristic',
                          token_usage: Dict[str, Any] | None = None,
                          **kwargs):
        AdvEoHProfiler.register_function(
            self,
            function,
            program=program,
            role=role,
            token_usage=token_usage,
            **kwargs,
        )
        self._write_tensorboard()


class AdvEoHWandbProfiler(WandBProfiler, AdvEoHProfiler):

    def __init__(self,
                 wandb_project_name: str,
                 log_dir: str | None = None,
                 *,
                 initial_num_samples: int = 0,
                 log_style: str = 'complex',
                 create_random_path: bool = True,
                 **kwargs):
        AdvEoHProfiler.__init__(self, log_dir=log_dir,
                                initial_num_samples=initial_num_samples,
                                log_style=log_style,
                                create_random_path=create_random_path,
                                **kwargs)
        WandBProfiler.__init__(self, wandb_project_name=wandb_project_name,
                               log_dir=log_dir,
                               initial_num_samples=initial_num_samples,
                               log_style=log_style,
                               create_random_path=create_random_path,
                               **kwargs)
        self._pop_lock = Lock()
        if self._log_dir:
            self._heu_ckpt_dir = os.path.join(self._log_dir, 'heuristic_pop')
            self._inst_ckpt_dir = os.path.join(self._log_dir, 'instance_pop')
            os.makedirs(self._heu_ckpt_dir, exist_ok=True)
            os.makedirs(self._inst_ckpt_dir, exist_ok=True)

    def finish(self):
        WandBProfiler.finish(self)

    def register_function(self, function: Function, program: str = '',
                          *, role: str = 'heuristic',
                          token_usage: Dict[str, Any] | None = None,
                          **kwargs):
        AdvEoHProfiler.register_function(
            self,
            function,
            program=program,
            role=role,
            token_usage=token_usage,
            **kwargs,
        )
        self._write_wandb()
