"""1D Bin Packing (constructive) adapter for AdvEoH.

Exports:
  - AdvBP1DEvaluation
  - BP1DInstanceGenEvaluation
  - bp1d_instance_template_program
  - bp1d_instance_task_description

Instance shape: tuple (item_weights: List[int] of length n_items, each in [1, bin_capacity-1],
                       bin_capacity: int).
Original generator samples weights from a Beta(2,5) scaled to int in [10, 50] with
bin_capacity=100. We allow the LLM to control the distribution but enforce that
each item fits in a bin (weight < bin_capacity).
"""
from __future__ import annotations

from typing import Any, List

import numpy as np

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.bp_1d_construct.evaluation import BP1DEvaluation

__all__ = [
    'AdvBP1DEvaluation', 'BP1DInstanceGenEvaluation',
    'bp1d_instance_template_program', 'bp1d_instance_task_description',
]


bp1d_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard 1D bin packing instance for a constructive heuristic.

    A harder instance is one on which a greedy (item, bin) selection heuristic
    uses MORE bins than on random Beta-distributed items.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 2-tuple (item_weights, bin_capacity) where:
          - item_weights: list of ints, length exactly 500.
                          Each value must be an int in [1, 99] (strictly < bin_capacity).
          - bin_capacity: int, must be 100.

    The evaluator REJECTS the instance (score = None) if:
      - result is not a 2-tuple,
      - item_weights is not a list/array of length 500,
      - any weight < 1 or >= bin_capacity,
      - any non-integer value (use int(...) or np.int64),
      - bin_capacity != 100.
    """
    rng = np.random.default_rng(seed)
    # Baseline: Beta(2,5) scaled to [10, 50]. Replace with your adversarial distribution.
    item_weights = (50 - rng.beta(2, 5, size=500) * 40).astype(int).tolist()
    bin_capacity = 100
    return item_weights, bin_capacity
'''

bp1d_instance_task_description = (
    "You are designing a hard instance generator for 1D BIN PACKING. A "
    "constructive heuristic iteratively picks the next (item, bin) pair to "
    "minimize the number of bins used. Your goal: produce 500 integer item "
    "weights (each in [1, 99]) for bins of capacity 100 that make greedy "
    "heuristics use MORE bins than on the Beta-distributed baseline.\n\n"
    "STRATEGY HINT: do NOT just use the Beta(2,5) baseline. Exploit the target "
    "heuristics below — e.g. many items near capacity/2 (max ambiguity), "
    "bi-modal tiny+near-capacity items, or adversarial orderings that force "
    "wasteful early placements."
)


class AdvBP1DEvaluation(BP1DEvaluation):
    """Subclass of BP1DEvaluation that accepts an optional ``instances`` kwarg."""

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


class BP1DInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for 1D bin packing."""

    def __init__(self,
                 n_items: int = 500,
                 bin_capacity: int = 100,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_items = n_items
        self.bin_capacity = bin_capacity
        super().__init__(
            template_program=bp1d_instance_template_program,
            task_description=bp1d_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
        item_weights, bin_cap = result
        try:
            item_weights = [int(w) for w in item_weights]
        except (TypeError, ValueError):
            return None
        if len(item_weights) != self.n_items:
            return None
        if int(bin_cap) != self.bin_capacity:
            return None
        if any(w < 1 or w >= self.bin_capacity for w in item_weights):
            return None
        return (item_weights, int(bin_cap))
