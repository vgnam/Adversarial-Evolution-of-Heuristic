"""OBP (Online Bin Packing) adapter for AdvEoH.

Exports:
  - AdvOBPEvaluation      : heuristic eval that accepts dynamic ``instances``.
  - OBPInstanceGenEvaluation : validates output of generate_instance(seed).
  - obp_instance_template_program : template for the instance generator.
  - obp_instance_task_description  : task description for the instance LLM.

Instance shape: tuple (items: np.ndarray (128,) float in (0,100], capacity: int = 100).
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.online_bin_packing.evaluation import OBPEvaluation

__all__ = [
    'AdvOBPEvaluation', 'OBPInstanceGenEvaluation',
    'obp_instance_template_program', 'obp_instance_task_description',
]


# =============================================================================
# Instance generator template
# =============================================================================

obp_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard online bin packing instance.

    The instance will be used to challenge a priority-based packing heuristic.
    A harder instance is one on which good heuristics use MORE bins (worse
    packing efficiency) than on random instances.

    Args:
        seed: int, random seed for reproducibility. Use it to construct a
            np.random.Generator, e.g. `rng = np.random.default_rng(seed)`.

    Returns:
        A tuple (items, capacity) where:
          - items: np.ndarray of dtype float, shape exactly (128,).
                   Each value must satisfy 0 < value <= 100.
                   No NaN, no Inf, no negative values.
          - capacity: int, must be exactly 100.

    Example:
        rng = np.random.default_rng(seed)
        items = rng.uniform(1, 100, size=128)
        capacity = 100
        return items, capacity

    The evaluator will REJECT your instance (treat it as invalid, score = None)
    if any of these constraints is violated:
      - returning something other than a 2-tuple,
      - items not being array-like of length 128,
      - any item value <= 0 or > 100,
      - capacity != 100,
      - any NaN or Inf in items.
    """
    rng = np.random.default_rng(seed)
    items = rng.uniform(1, 100, size=128)
    capacity = 100
    return items, capacity
'''

obp_instance_task_description = (
    "You are designing a hard instance generator for the ONLINE BIN PACKING "
    "problem. A priority-based heuristic packs 128 items (arriving one-by-one) "
    "into bins of capacity 100; it picks the bin that maximizes a `priority(item, "
    "bins)` score. Your goal: produce item-size distributions that make such "
    "heuristics use as MANY bins as possible.\n\n"
    "STRATEGY HINT: do NOT just draw iid uniform items — that is the baseline "
    "every heuristic already handles well. Exploit the structure of the target "
    "heuristics described below to craft adversarial item distributions "
    "(e.g. clustered, bi-modal, or near-capacity values)."
)


# =============================================================================
# AdvOBPEvaluation  — heuristic eval that accepts dynamic instances
# =============================================================================

class AdvOBPEvaluation(OBPEvaluation):
    """Subclass of OBPEvaluation that accepts an optional ``instances`` kwarg.
    When ``instances`` is None, evaluation falls back to the default Weibull
    dataset (used for warm-up and held-out reporting). When ``instances`` is a
    list of (items, capacity) tuples, the heuristic is evaluated on those
    custom instances only.
    """

    def evaluate_program(self, program_str: str, callable_func: callable,
                         instances: list | None = None, **kwargs) -> Any | None:
        if instances is None:
            return self.evaluate(callable_func)
        return self._evaluate_on_instances(callable_func, instances)

    def _evaluate_on_instances(self, priority: callable,
                               instances: List[Tuple[np.ndarray, int]]) -> float:
        num_bins = []
        for items, capacity in instances:
            items = np.asarray(items, dtype=float)
            n = len(items)
            bins = np.array([capacity for _ in range(n)])
            _, bins_packed = self.online_binpack(tuple(items.tolist()), bins, priority)
            num_bins.append((bins_packed != capacity).sum())
        return -float(np.mean(num_bins))


# =============================================================================
# OBPInstanceGenEvaluation  — runs generate_instance(seed), validates output
# =============================================================================

class OBPInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for OBP.

    Output must be a 2-tuple (items, capacity) where:
      - items: array-like of length ``n_items`` (default 128), all in (0, capacity].
      - capacity: int (default 100).
    """

    def __init__(self,
                 n_items: int = 128,
                 capacity: int = 100,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_items = n_items
        self.capacity = capacity
        super().__init__(
            template_program=obp_instance_template_program,
            task_description=obp_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
        items, cap = result
        items = np.asarray(items, dtype=float)
        if items.shape != (self.n_items,):
            return None
        if int(cap) != self.capacity:
            return None
        if np.any(items <= 0) or np.any(items > self.capacity):
            return None
        if np.any(np.isnan(items)) or np.any(np.isinf(items)):
            return None
        return (items, int(cap))
