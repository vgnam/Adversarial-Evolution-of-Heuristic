"""Set Cover (constructive) adapter for AdvEoH.

Exports:
  - AdvSetCoverEvaluation
  - SetCoverInstanceGenEvaluation
  - set_cover_instance_template_program
  - set_cover_instance_task_description

Instance shape: tuple (universal_set: List[int] = [1, 2, ..., n_elements],
                       subsets: List[List[int]] of length n_subsets,
                       each subset has size in [1, max_subset_size], no internal duplicates).
Defaults: n_elements=50, n_subsets=50, max_subset_size=8.

FEASIBILITY: the adapter rejects instances where ``union(subsets) !=
universal_set`` (the heuristic cannot cover all elements). The original
generator does NOT enforce this; we add it.
"""
from __future__ import annotations

from typing import Any, List

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.set_cover_construct.evaluation import SCPEvaluation

__all__ = [
    'AdvSetCoverEvaluation', 'SetCoverInstanceGenEvaluation',
    'set_cover_instance_template_program', 'set_cover_instance_task_description',
]


set_cover_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard Set Cover instance for a constructive heuristic.

    A harder instance is one on which a greedy subset-selection heuristic
    uses MORE subsets than on random uniform instances.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 2-tuple (universal_set, subsets) where:
          - universal_set: list of ints = [1, 2, ..., 50] (in this exact order).
          - subsets: list of lists of ints, length exactly 50.
                     Each subset: a list of distinct ints from universal_set,
                     length in [1, 8].

    FEASIBILITY: the adapter REJECTS the instance if the union of all subsets
    is not equal to the universal_set (i.e. some elements are uncoverable).
    Make sure every element 1..50 appears in at least one subset.

    The adapter also REJECTS the instance if:
      - result is not a 2-tuple,
      - universal_set is not [1..50],
      - subsets is not a list of length 50,
      - any subset has duplicates, empty, or length > 8,
      - any subset contains an element outside [1, 50].
    """
    rng = np.random.default_rng(seed)
    universal_set = list(range(1, 51))
    subsets = []
    for _ in range(50):
        size = int(rng.integers(1, 9))
        subset = rng.choice(universal_set, size=size, replace=False).tolist()
        subsets.append(subset)
    return universal_set, subsets
'''

set_cover_instance_task_description = (
    "You are designing a hard instance generator for the SET COVER problem. "
    "A constructive heuristic iteratively picks the next subset to add "
    "(minimizing the number of subsets used) until the union covers the "
    "universal set {1, 2, ..., 50}. Your goal: produce 50 subsets (each "
    "containing 1-8 distinct elements from 1..50) that make greedy heuristics "
    "use MORE subsets than on random uniform instances.\n\n"
    "FEASIBILITY: every element 1..50 must appear in at least one subset, or "
    "the adapter rejects the instance. Exploit this margin — e.g. make each "
    "element appear in only 1-2 subsets so the heuristic has no redundancy.\n\n"
    "STRATEGY HINT: do NOT just use uniform random subsets. Exploit the target "
    "heuristics below — e.g. many tiny subsets (each covering 1 element) to "
    "force high count, overlapping subsets that mislead greedy coverage rules, "
    "or 'trap' subsets that look useful but cover already-covered elements."
)


class AdvSetCoverEvaluation(SCPEvaluation):
    """Subclass of SCPEvaluation that accepts an optional ``instances`` kwarg."""

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


class SetCoverInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for Set Cover.

    Enforces: shape, value ranges, no internal duplicates, and feasibility
    (union(subsets) == universal_set).
    """

    def __init__(self,
                 n_elements: int = 50,
                 n_subsets: int = 50,
                 max_subset_size: int = 8,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_elements = n_elements
        self.n_subsets = n_subsets
        self.max_subset_size = max_subset_size
        super().__init__(
            template_program=set_cover_instance_template_program,
            task_description=set_cover_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
        universal_set, subsets = result
        try:
            universal_set = [int(x) for x in universal_set]
            subsets = [[int(x) for x in s] for s in subsets]
        except (TypeError, ValueError):
            return None
        if universal_set != list(range(1, self.n_elements + 1)):
            return None
        if len(subsets) != self.n_subsets:
            return None
        union = set()
        for s in subsets:
            if len(s) < 1 or len(s) > self.max_subset_size:
                return None
            if len(set(s)) != len(s):
                return None
            for x in s:
                if x < 1 or x > self.n_elements:
                    return None
            union.update(s)
        if union != set(universal_set):
            return None
        return (universal_set, subsets)
