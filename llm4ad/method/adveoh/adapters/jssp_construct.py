"""JSSP (Job Shop Scheduling, constructive) adapter for AdvEoH.

Exports:
  - AdvJSSPEvaluation
  - JSSPInstanceGenEvaluation
  - jssp_instance_template_program
  - jssp_instance_task_description

Instance shape: tuple (processing_times: List[List[int]] shape (n_jobs, n_machines),
                       n_jobs: int,
                       n_machines: int).
Each processing_times[j][m] in [10, 100]. Each job has exactly n_machines
operations (one per machine). Defaults: n_jobs=50, n_machines=10.
"""
from __future__ import annotations

from typing import Any, List

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.jssp_construct.evaluation import JSSPEvaluation

__all__ = [
    'AdvJSSPEvaluation', 'JSSPInstanceGenEvaluation',
    'jssp_instance_template_program', 'jssp_instance_task_description',
]


jssp_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard Job Shop Scheduling instance for a constructive heuristic.

    A harder instance is one on which a greedy operation-selection heuristic
    produces a HIGHER makespan than on random uniform processing times.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 3-tuple (processing_times, n_jobs, n_machines) where:
          - processing_times: list of lists of ints, shape exactly (50, 10).
                              Each value in [10, 100]. Row j = job j's processing
                              time on each of the 10 machines.
          - n_jobs: int, must be 50.
          - n_machines: int, must be 10.

    The evaluator REJECTS the instance (score = None) if:
      - result is not a 3-tuple,
      - processing_times is not shape (50, 10),
      - any value < 10 or > 100,
      - n_jobs != 50 or n_machines != 10.
    """
    rng = np.random.default_rng(seed)
    processing_times = rng.integers(10, 101, size=(50, 10)).tolist()
    n_jobs = 50
    n_machines = 10
    return processing_times, n_jobs, n_machines
'''

jssp_instance_task_description = (
    "You are designing a hard instance generator for JOB SHOP SCHEDULING. A "
    "constructive heuristic iteratively picks the next operation to schedule "
    "(minimizing makespan) given current machine availability and feasible "
    "operations. Your goal: produce 50 jobs x 10 machines of integer "
    "processing times (each in [10, 100]) that make greedy heuristics "
    "produce HIGHER makespans than on random uniform instances.\n\n"
    "STRATEGY HINT: do NOT just use uniform random processing times. Exploit "
    "the target heuristics below — e.g. bimodal distributions that create "
    "bottleneck machines, jobs with one very long operation that blocks "
    "others, or correlated patterns that mislead priority rules."
)


class AdvJSSPEvaluation(JSSPEvaluation):
    """Subclass of JSSPEvaluation that accepts an optional ``instances`` kwarg."""

    def evaluate_program(self, program_str: str, callable_func: callable,
                         instances: list | None = None, **kwargs) -> Any | None:
        if instances is None:
            return self.evaluate(callable_func)
        orig_datasets = self._datasets
        orig_n = self.n_instance
        try:
            self._datasets = list(instances)
            self.n_instance = len(self._datasets)
            return self.evaluate(callable_func)
        finally:
            self._datasets = orig_datasets
            self.n_instance = orig_n


class JSSPInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for JSSP."""

    def __init__(self,
                 n_jobs: int = 50,
                 n_machines: int = 10,
                 min_time: int = 10,
                 max_time: int = 100,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_jobs = n_jobs
        self.n_machines = n_machines
        self.min_time = min_time
        self.max_time = max_time
        super().__init__(
            template_program=jssp_instance_template_program,
            task_description=jssp_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            return None
        proc_times, n_jobs, n_machines = result
        try:
            proc_times = [[int(t) for t in row] for row in proc_times]
        except (TypeError, ValueError):
            return None
        if len(proc_times) != self.n_jobs:
            return None
        for row in proc_times:
            if len(row) != self.n_machines:
                return None
            for t in row:
                if t < self.min_time or t > self.max_time:
                    return None
        if int(n_jobs) != self.n_jobs or int(n_machines) != self.n_machines:
            return None
        return (proc_times, int(n_jobs), int(n_machines))
