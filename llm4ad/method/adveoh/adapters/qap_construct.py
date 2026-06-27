"""QAP (Quadratic Assignment, constructive) adapter for AdvEoH.

Exports:
  - AdvQAPEvaluation
  - QAPInstanceGenEvaluation
  - qap_instance_template_program
  - qap_instance_task_description

Instance shape: tuple (flow_matrix: np.ndarray (n, n) int,
                       distance_matrix: np.ndarray (n, n) int).
BOTH matrices must be SYMMETRIC with ZERO DIAGONAL, values in [1, 100] off-diag.
Default n_facilities=20.

NOTE: QAPEvaluation.evaluate_program calls evaluate_qap (not evaluate).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.qap_construct.evaluation import QAPEvaluation

__all__ = [
    'AdvQAPEvaluation', 'QAPInstanceGenEvaluation',
    'qap_instance_template_program', 'qap_instance_task_description',
]


qap_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard QAP instance for a constructive heuristic.

    A harder instance is one on which a greedy assignment heuristic produces
    a HIGHER total cost (sum of flow[i,j] * distance[pi[i], pi[j]]) than on
    random uniform matrices.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 2-tuple (flow_matrix, distance_matrix) where both:
          - np.ndarray int, shape exactly (20, 20),
          - SYMMETRIC (M == M.T),
          - ZERO DIAGONAL (M[i,i] = 0 for all i),
          - off-diagonal values in [1, 100].

    To enforce symmetry and zero diagonal:
        M = rng.integers(1, 101, size=(20, 20))
        M = (M + M.T) // 2
        np.fill_diagonal(M, 0)

    The evaluator REJECTS the instance (score = None) if:
      - result is not a 2-tuple,
      - either matrix is not shape (20, 20) int,
      - either matrix is not symmetric (within atol=1e-6),
      - either matrix has a non-zero diagonal,
      - any off-diagonal value < 1 or > 100.
    """
    rng = np.random.default_rng(seed)
    flow_matrix = rng.integers(1, 101, size=(20, 20))
    flow_matrix = (flow_matrix + flow_matrix.T) // 2
    np.fill_diagonal(flow_matrix, 0)
    distance_matrix = rng.integers(1, 101, size=(20, 20))
    distance_matrix = (distance_matrix + distance_matrix.T) // 2
    np.fill_diagonal(distance_matrix, 0)
    return flow_matrix, distance_matrix
'''

qap_instance_task_description = (
    "You are designing a hard instance generator for the QUADRATIC ASSIGNMENT "
    "problem. A heuristic assigns facilities to locations (one-to-one) to "
    "minimize total cost = sum of flow[i,j] * distance[pi[i], pi[j]]. Your "
    "goal: produce 20x20 symmetric integer matrices (zero diagonal, off-diag "
    "in [1, 100]) for both flow and distance that make greedy heuristics "
    "produce HIGHER total cost than on random uniform matrices.\n\n"
    "STRATEGY HINT: do NOT just use uniform random matrices. Exploit the "
    "target heuristics below — e.g. anti-correlated flow/distance structure "
    "(high flow between facilities that heuristics place far apart), "
    "concentrated flow on a few facility pairs, or distance matrices with "
    "clusters that mislead greedy assignment."
)


class AdvQAPEvaluation(QAPEvaluation):
    """Subclass of QAPEvaluation that accepts an optional ``instances`` kwarg.
    NOTE: QAPEvaluation.evaluate_program calls ``evaluate_qap``, not ``evaluate``.
    """

    def evaluate_program(self, program_str: str, callable_func: callable,
                         instances: list | None = None, **kwargs) -> Any | None:
        if instances is None:
            return self.evaluate_qap(callable_func)
        orig_datasets = self._datasets
        orig_n = self.n_instance
        try:
            self._datasets = list(instances)
            self.n_instance = len(self._datasets)
            return self.evaluate_qap(callable_func)
        finally:
            self._datasets = orig_datasets
            self.n_instance = orig_n


class QAPInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for QAP.

    Both matrices must be (n, n) int, symmetric, zero diagonal, off-diag in [1, 100].
    """

    def __init__(self,
                 n_facilities: int = 20,
                 min_value: int = 1,
                 max_value: int = 100,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_facilities = n_facilities
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(
            template_program=qap_instance_template_program,
            task_description=qap_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def _validate_matrix(self, M) -> np.ndarray | None:
        M = np.asarray(M, dtype=int)
        n = self.n_facilities
        if M.shape != (n, n):
            return None
        if not np.allclose(M, M.T, atol=1e-6):
            return None
        if np.any(np.diag(M) != 0):
            return None
        off_diag = M[~np.eye(n, dtype=bool)]
        if np.any(off_diag < self.min_value) or np.any(off_diag > self.max_value):
            return None
        return M

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
        flow, dist = result
        flow = self._validate_matrix(flow)
        dist = self._validate_matrix(dist)
        if flow is None or dist is None:
            return None
        return (flow, dist)
