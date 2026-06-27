"""CVRP (Capacitated VRP, constructive) adapter for AdvEoH.

Exports:
  - AdvCVRPEvaluation
  - CVRPInstanceGenEvaluation
  - cvrp_instance_template_program
  - cvrp_instance_task_description

Instance shape: tuple (coordinates: np.ndarray (n+1, 2) float in [0,1),
                       distance_matrix: np.ndarray (n+1, n+1) float symmetric,
                       demands: np.ndarray (n+1,) int, demands[0]=0, demands[1:] in [1,10],
                       capacity: int = 40).
Node 0 is the depot. ``problem_size`` = number of customers (the constructor
adds +1 for the depot, matching the original CVRPEvaluation behavior).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.cvrp_construct.evaluation import CVRPEvaluation

__all__ = [
    'AdvCVRPEvaluation', 'CVRPInstanceGenEvaluation',
    'cvrp_instance_template_program', 'cvrp_instance_task_description',
]


cvrp_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard CVRP instance for a constructive heuristic.

    A harder instance is one on which a greedy next-node selection heuristic
    produces a LONGER total route distance than on random uniform points.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 4-tuple (coordinates, distance_matrix, demands, capacity) where:
          - coordinates: np.ndarray, dtype float, shape exactly (51, 2),
                         values in [0, 1). Row 0 = depot, rows 1..50 = customers.
          - distance_matrix: np.ndarray, dtype float, shape (51, 51),
                             Euclidean pairwise, symmetric, ~0 diagonal.
          - demands: np.ndarray, dtype int, shape (51,).
                     demands[0] = 0 (depot), demands[1:] in [1, 10].
          - capacity: int, must be 40.

    The evaluator REJECTS the instance (score = None) if any constraint is
    violated (shape, depot demand=0, demand range, capacity value, symmetry).
    """
    rng = np.random.default_rng(seed)
    n = 51
    coordinates = rng.random((n, 2))
    distance_matrix = np.linalg.norm(coordinates[:, np.newaxis] - coordinates, axis=2)
    demands = np.zeros(n, dtype=int)
    demands[1:] = rng.integers(1, 11, size=n - 1)
    capacity = 40
    return coordinates, distance_matrix, demands, capacity
'''

cvrp_instance_task_description = (
    "You are designing a hard instance generator for the CAPACITATED VRP. "
    "A heuristic iteratively picks the next customer to serve (given current "
    "node, depot, unvisited set, remaining capacity, demands, distance matrix) "
    "to minimize total route distance. Vehicles start and end at the depot "
    "(node 0) and have capacity 40. Your goal: produce 50 customer locations "
    "in [0,1)^2 (plus the depot) with integer demands in [1,10] that make "
    "greedy heuristics produce LONGER total distances than on random instances.\n\n"
    "STRATEGY HINT: do NOT just place iid uniform points. Exploit the target "
    "heuristics below — e.g. cluster high-demand customers far from the depot, "
    "or arrange points so greedy capacity filling leaves inefficient return trips."
)


class AdvCVRPEvaluation(CVRPEvaluation):
    """Subclass of CVRPEvaluation that accepts an optional ``instances`` kwarg."""

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


class CVRPInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for CVRP.

    Output must be a 4-tuple (coordinates, distance_matrix, demands, capacity).
    """

    def __init__(self,
                 problem_size: int = 50,
                 capacity: int = 40,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.problem_size = problem_size + 1  # +1 for depot (matches CVRPEvaluation)
        self.capacity = capacity
        super().__init__(
            template_program=cvrp_instance_template_program,
            task_description=cvrp_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            return None
        coords, distmat, demands, cap = result
        coords = np.asarray(coords, dtype=float)
        distmat = np.asarray(distmat, dtype=float)
        demands = np.asarray(demands, dtype=int)
        n = self.problem_size
        if coords.shape != (n, 2):
            return None
        if distmat.shape != (n, n):
            return None
        if demands.shape != (n,):
            return None
        if int(cap) != self.capacity:
            return None
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return None
        if np.any(np.isnan(distmat)) or np.any(np.isinf(distmat)):
            return None
        if np.any(coords < 0) or np.any(coords >= 1):
            return None
        if not np.allclose(distmat, distmat.T, atol=1e-6):
            return None
        if demands[0] != 0:
            return None
        if np.any(demands[1:] < 1) or np.any(demands[1:] > 10):
            return None
        return (coords, distmat, demands, int(cap))
