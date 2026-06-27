"""VRPTW (VRP with Time Windows, constructive) adapter for AdvEoH.

Exports:
  - AdvVRPTWEvaluation
  - VRPTWInstanceGenEvaluation
  - vrptw_instance_template_program
  - vrptw_instance_task_description

Instance shape produced by the LLM: 4-tuple (coordinates (n+1, 2) float in [0,1),
                                          distance_matrix (n+1, n+1) symmetric,
                                          demands (n+1,) int, demands[0]=0, demands[1:] in [1,10],
                                          capacity: int = 40).

The adapter auto-derives ``serviceTime`` and ``time_windows`` from the
coordinates using the original ``vrptw_construct/get_instance.py`` formula
(max_time=4.6). This GUARANTEES feasibility (time windows remain reachable
from the depot) and lets the LLM focus on adversarial coordinates/demands
while still indirectly controlling time-window tightness through d0i.

The final instance passed to the heuristic evaluator is the full 6-tuple
(coords, distmat, demands, capacity, serviceTime, time_windows) matching the
original ``self._datasets`` element format.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.vrptw_construct.evaluation import VRPTWEvaluation

__all__ = [
    'AdvVRPTWEvaluation', 'VRPTWInstanceGenEvaluation',
    'vrptw_instance_template_program', 'vrptw_instance_task_description',
]


vrptw_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard VRP-with-Time-Windows instance for a constructive heuristic.

    NOTE: You only generate coordinates, distance matrix, demands, and capacity.
    The adapter will AUTOMATICALLY derive service times and feasible time
    windows from your coordinates (using the standard VRPTW formula with
    max_time=4.6). Tighter clusters / harder-to-reach customers will produce
    tighter time windows — exploit this for adversarial effect.

    A harder instance is one on which a greedy next-node selection heuristic
    produces a LONGER total route distance than on random uniform points.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 4-tuple (coordinates, distance_matrix, demands, capacity) where:
          - coordinates: np.ndarray float, shape exactly (51, 2), values in [0, 1).
                         Row 0 = depot, rows 1..50 = customers.
          - distance_matrix: np.ndarray float, shape (51, 51), symmetric Euclidean.
          - demands: np.ndarray int, shape (51,). demands[0] = 0, demands[1:] in [1, 10].
          - capacity: int, must be 40.

    The evaluator REJECTS the instance (score = None) if any constraint is
    violated (shape, demand depot=0, demand range, capacity value, symmetry).
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

vrptw_instance_task_description = (
    "You are designing a hard instance generator for the VRP WITH TIME "
    "WINDOWS. A heuristic iteratively picks the next customer to serve "
    "(given current node, depot, unvisited set, remaining capacity, current "
    "time, demands, distance matrix, and time windows) to minimize total "
    "route distance while respecting per-node [early, late] windows and "
    "vehicle capacity 40. You generate coordinates (50 customers + depot in "
    "[0,1)^2) and integer demands in [1,10]; the adapter auto-derives service "
    "times and feasible time windows from your coordinates (max_time=4.6). "
    "Your goal: produce point layouts that make greedy heuristics produce "
    "LONGER total distances than on random uniform instances.\n\n"
    "STRATEGY HINT: do NOT just place iid uniform points. Exploit the target "
    "heuristics below — e.g. customers clustered at the edge of reach (tight "
    "time windows), high-demand customers far from depot, or geometries that "
    "force multiple return trips to the depot before the time window closes."
)


class AdvVRPTWEvaluation(VRPTWEvaluation):
    """Subclass of VRPTWEvaluation that accepts an optional ``instances`` kwarg.
    ``instances`` is a list of full 6-tuples (coords, distmat, demands, capacity,
    serviceTime, time_windows) — i.e. the format produced by
    ``VRPTWInstanceGenEvaluation.validate_instance``.
    """

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


class VRPTWInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for VRPTW.

    LLM returns a 4-tuple (coords, distmat, demands, capacity). The adapter
    auto-derives serviceTime and time_windows using the original formula
    (max_time=4.6) and returns the full 6-tuple for the heuristic evaluator.
    """

    def __init__(self,
                 problem_size: int = 50,
                 capacity: int = 40,
                 max_time: float = 4.6,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.problem_size = problem_size  # number of customers (excluding depot)
        self.capacity = capacity
        self.max_time = max_time
        super().__init__(
            template_program=vrptw_instance_template_program,
            task_description=vrptw_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def _derive_time_windows(self, coordinates: np.ndarray,
                             distance_matrix: np.ndarray):
        """Replicates vrptw_construct/get_instance.py:21-49."""
        n_customers = self.problem_size
        rng = np.random.default_rng(0)  # deterministic so seeds don't compound
        node_serviceTime = rng.random(n_customers) * 0.05 + 0.15
        serviceTime = np.append(np.array([0.0]), node_serviceTime)
        node_lengthTW = rng.random(n_customers) * 0.05 + 0.15
        d0i = distance_matrix[0][1:]
        # ei in [1, (max_time - serviceTime - lengthTW)/d0i - 1]
        upper = (self.max_time - node_serviceTime - node_lengthTW) / d0i - 1
        ei = rng.random(n_customers) * (upper - 1) + 1
        node_earlyTW = ei * d0i
        node_lateTW = node_earlyTW + node_lengthTW
        time_windows_node = np.stack([node_earlyTW, node_lateTW], axis=1)
        time_windows = np.concatenate(
            [np.array([[0.0, self.max_time]]), time_windows_node], axis=0
        )
        return serviceTime, time_windows

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            return None
        coords, distmat, demands, cap = result
        coords = np.asarray(coords, dtype=float)
        distmat = np.asarray(distmat, dtype=float)
        demands = np.asarray(demands, dtype=int)
        n = self.problem_size + 1
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
        serviceTime, time_windows = self._derive_time_windows(coords, distmat)
        return (coords, distmat, demands, int(cap), serviceTime, time_windows)
