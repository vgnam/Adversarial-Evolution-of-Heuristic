"""TSP (constructive) adapter for AdvEoH.

Exports:
  - AdvTSPEvaluation           : heuristic eval with ``instances`` kwarg.
  - TSPInstanceGenEvaluation   : validates generate_instance(seed) output.
  - tsp_instance_template_program
  - tsp_instance_task_description

The generator only needs to return coordinates. The adapter derives the
distance matrix and returns the original ``tsp_construct/get_instance.py``
format ``(coordinates, distance_matrix)`` to the heuristic evaluator.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.tsp_construct.evaluation import TSPEvaluation

__all__ = [
    'AdvTSPEvaluation', 'TSPInstanceGenEvaluation',
    'tsp_instance_template_program', 'tsp_instance_task_description',
]


# =============================================================================
# Instance generator template
# =============================================================================

def _make_tsp_instance_template_program(problem_size: int) -> str:
    return f'''
import numpy as np

def generate_instance(seed: int) -> np.ndarray:
    """Generate a hard TSP instance for a constructive heuristic.

    A harder instance is one on which a greedy next-node selection heuristic
    produces a LONGER tour than on random uniform points.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        coordinates: np.ndarray, dtype float, shape exactly ({problem_size}, 2),
                     values in [0, 1). Each row is a city's (x, y).

    Return ONLY the coordinates. Do not return a distance matrix. The evaluator
    will compute the Euclidean distance matrix automatically from the final
    coordinates.

    The evaluator REJECTS the instance (score = None) if:
      - coordinates is not shape ({problem_size}, 2) with finite values in [0, 1),
      - any NaN / Inf.
    """
    rng = np.random.default_rng(seed)
    coordinates = rng.random(({problem_size}, 2))
    return coordinates
'''

tsp_instance_template_program = _make_tsp_instance_template_program(50)

tsp_instance_task_description = (
    "You are designing a hard instance generator for the CONSTRUCTIVE TRAVELING "
    "SALESMAN PROBLEM. A heuristic iteratively picks the next node to visit "
    "(given the current node, destination, unvisited set, and distance matrix) "
    "to minimize total tour length. Your goal: produce point configurations "
    "in the unit square [0,1)^2 (50 points) that make such greedy heuristics "
    "produce LONGER tours than they would on random uniform points.\n\n"
    "STRATEGY HINT: do NOT just place iid uniform points — that is the baseline. "
    "Exploit the structure of the target heuristics described below to craft "
    "adversarial layouts (e.g. clustered, on a grid with traps, fractal-like, "
    "or with many near-equal-distance neighbors to confuse greedy choices)."
)


def _make_tsp_instance_task_description(problem_size: int) -> str:
    if problem_size == 50:
        return tsp_instance_task_description
    return tsp_instance_task_description.replace(
        "(50 points)",
        f"({problem_size} points)",
    )


# =============================================================================
# AdvTSPEvaluation
# =============================================================================

class AdvTSPEvaluation(TSPEvaluation):
    """Subclass of TSPEvaluation that accepts an optional ``instances`` kwarg.
    When ``instances`` is None, falls back to the default dataset. Otherwise
    temporarily swaps ``self._datasets`` and ``self.n_instance`` to reuse the
    inherited ``evaluate()`` method. Safe because SecureEvaluator runs each
    evaluation in an isolated subprocess.
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


# =============================================================================
# TSPInstanceGenEvaluation
# =============================================================================

class TSPInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for TSP.

    Output should be coordinates. A legacy 2-tuple (coordinates,
    distance_matrix) is still accepted, but the supplied distance matrix is
    ignored. The validator recomputes the Euclidean distance matrix from
    coordinates to avoid rejecting generators for simple formatting mistakes.
    """

    def __init__(self,
                 problem_size: int = 50,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.problem_size = problem_size
        self._last_validation_error = ''
        super().__init__(
            template_program=_make_tsp_instance_template_program(problem_size),
            task_description=_make_tsp_instance_task_description(problem_size),
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def _invalid(self, reason: str) -> None:
        self._last_validation_error = reason
        return None

    def validate_instance(self, result) -> tuple | None:
        if isinstance(result, dict):
            coords = result.get('coordinates', result.get('coords', None))
        elif isinstance(result, (tuple, list)) and len(result) == 2:
            coords, _ = result
        else:
            coords = result

        try:
            coords = np.asarray(coords, dtype=float)
        except Exception as exc:
            return self._invalid(f'coordinates cannot be converted to float array: {type(exc).__name__}: {exc}')

        n = self.problem_size
        if coords.shape != (n, 2):
            return self._invalid(f'coordinates shape is {coords.shape}, expected ({n}, 2)')
        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return self._invalid('coordinates contain NaN or Inf')

        coords = np.clip(coords, 0.0, 1.0 - 1e-12)
        distmat = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        self._last_validation_error = ''
        return (coords, distmat)
