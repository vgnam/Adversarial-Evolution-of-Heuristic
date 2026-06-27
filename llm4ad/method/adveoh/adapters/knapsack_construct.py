"""Knapsack (0/1, constructive) adapter for AdvEoH.

Exports:
  - AdvKnapsackEvaluation
  - KnapsackInstanceGenEvaluation
  - knapsack_instance_template_program
  - knapsack_instance_task_description

Instance shape: tuple (item_weights: List[int] of length n_items,
                       item_values: List[int] of length n_items,
                       knapsack_capacity: int).
Original defaults: n_items=50, knapsack_capacity=100,
weights in [10, cap/2+10], values in [1, 100].

NOTE: knapsack is a maximization problem, but the original evaluate returns
``-average_value`` so that higher = better (consistent with the rest of LLM4AD).
AdvEoH's adversarial instance fitness = -(mean heu_score) thus flips again,
so harder instance = lower heu value = higher inst fitness. Correct.
"""
from __future__ import annotations

from typing import Any, List

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.knapsack_construct.evaluation import KnapsackEvaluation

__all__ = [
    'AdvKnapsackEvaluation', 'KnapsackInstanceGenEvaluation',
    'knapsack_instance_template_program', 'knapsack_instance_task_description',
]


knapsack_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard 0/1 knapsack instance for a constructive heuristic.

    A harder instance is one on which a greedy item-selection heuristic
    achieves LOWER total value than on random instances (i.e. the heuristic
    is misled into picking low value-density items or wasting capacity).

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 3-tuple (item_weights, item_values, knapsack_capacity) where:
          - item_weights: list of ints, length exactly 50, each in [1, 60].
          - item_values:  list of ints, length exactly 50, each in [1, 100].
          - knapsack_capacity: int, must be 100.

    The evaluator REJECTS the instance (score = None) if:
      - result is not a 3-tuple,
      - weights or values are not length-50 lists of ints,
      - any weight < 1 or > 60,
      - any value < 1 or > 100,
      - knapsack_capacity != 100.
    """
    rng = np.random.default_rng(seed)
    item_weights = rng.integers(10, 60, size=50).tolist()
    item_values = rng.integers(1, 101, size=50).tolist()
    knapsack_capacity = 100
    return item_weights, item_values, knapsack_capacity
'''

knapsack_instance_task_description = (
    "You are designing a hard instance generator for the 0/1 KNAPSACK problem. "
    "A constructive heuristic iteratively picks the next item to add "
    "(maximizing total value within capacity 100). Your goal: produce 50 "
    "integer item weights (each in [1, 60]) and 50 integer values (each in "
    "[1, 100]) that make greedy heuristics achieve LOWER total value than on "
    "random instances — i.e. mislead them into low value-density choices or "
    "capacity waste.\n\n"
    "STRATEGY HINT: do NOT just use uniform random weights/values. Exploit "
    "the target heuristics below — e.g. items where high value correlates "
    "with high weight (low density) to trap value-greedy heuristics, or "
    "near-capacity items that block better combinations, or anti-correlated "
    "weight/value patterns that exploit the heuristic's selection rule."
)


class AdvKnapsackEvaluation(KnapsackEvaluation):
    """Subclass of KnapsackEvaluation that accepts an optional ``instances`` kwarg."""

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


class KnapsackInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for 0/1 knapsack."""

    def __init__(self,
                 n_items: int = 50,
                 knapsack_capacity: int = 100,
                 max_weight: int = 60,
                 max_value: int = 100,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_items = n_items
        self.knapsack_capacity = knapsack_capacity
        self.max_weight = max_weight
        self.max_value = max_value
        super().__init__(
            template_program=knapsack_instance_template_program,
            task_description=knapsack_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            return None
        weights, values, cap = result
        try:
            weights = [int(w) for w in weights]
            values = [int(v) for v in values]
        except (TypeError, ValueError):
            return None
        if len(weights) != self.n_items or len(values) != self.n_items:
            return None
        if int(cap) != self.knapsack_capacity:
            return None
        if any(w < 1 or w > self.max_weight for w in weights):
            return None
        if any(v < 1 or v > self.max_value for v in values):
            return None
        return (weights, values, int(cap))
