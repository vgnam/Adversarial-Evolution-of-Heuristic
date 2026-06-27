"""2D Bin Packing (constructive) adapter for AdvEoH.

Exports:
  - AdvBP2DEvaluation
  - BP2DInstanceGenEvaluation
  - bp2d_instance_template_program
  - bp2d_instance_task_description

Instance shape: tuple (item_dimensions: List[(w: int, h: int)] of length n_items,
                       bin_dimensions: (w: int, h: int)).
Each item must fit in a bin: 1 <= w < bin_w, 1 <= h < bin_h.
Original defaults: n_items=100, bin_width=100, bin_height=100.

NOTE: BP2DEvaluation.evaluate_program calls evaluate_2d (not evaluate), so the
swap-restore override targets evaluate_2d's data source ``self._datasets``.
"""
from __future__ import annotations

from typing import Any, List, Tuple

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.bp_2d_construct.evaluation import BP2DEvaluation

__all__ = [
    'AdvBP2DEvaluation', 'BP2DInstanceGenEvaluation',
    'bp2d_instance_template_program', 'bp2d_instance_task_description',
]


bp2d_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> tuple:
    """Generate a hard 2D bin packing instance for a constructive heuristic.

    A harder instance is one on which a greedy (item, bin) selection heuristic
    uses MORE bins than on random uniform rectangles.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A 2-tuple (item_dimensions, bin_dimensions) where:
          - item_dimensions: list of (w, h) int tuples, length exactly 100.
                             Each w in [1, 99], each h in [1, 99].
          - bin_dimensions: (w, h) = (100, 100).

    The evaluator REJECTS the instance (score = None) if:
      - result is not a 2-tuple,
      - item_dimensions is not length 100, or any item isn't a 2-tuple of ints,
      - any item w or h is < 1 or >= the corresponding bin dimension,
      - bin_dimensions != (100, 100).
    """
    rng = np.random.default_rng(seed)
    item_widths = rng.integers(10, 90, size=100)
    item_heights = rng.integers(10, 90, size=100)
    item_dimensions = [(int(w), int(h)) for w, h in zip(item_widths, item_heights)]
    bin_dimensions = (100, 100)
    return item_dimensions, bin_dimensions
'''

bp2d_instance_task_description = (
    "You are designing a hard instance generator for 2D BIN PACKING. A "
    "constructive heuristic iteratively picks the next (item, bin) pair to "
    "minimize the number of bins used. Each item is a rectangle (w, h) and "
    "bins are 100x100. Your goal: produce 100 integer rectangle dimensions "
    "(each w,h in [1, 99]) that make greedy heuristics use MORE bins than on "
    "random uniform rectangles.\n\n"
    "STRATEGY HINT: do NOT just use uniform random rectangles. Exploit the "
    "target heuristics below — e.g. items near half-bin in both dimensions "
    "(max wastage), L-shaped adversarial mixes of long-thin + square items, "
    "or items that force fragmentation of bin space."
)


class AdvBP2DEvaluation(BP2DEvaluation):
    """Subclass of BP2DEvaluation that accepts an optional ``instances`` kwarg.
    NOTE: BP2DEvaluation.evaluate_program calls ``evaluate_2d``, not ``evaluate``.
    """

    def evaluate_program(self, program_str: str, callable_func: callable,
                         instances: list | None = None, **kwargs) -> Any | None:
        if instances is None:
            return self.evaluate_2d(callable_func)
        orig_datasets = self._datasets
        orig_n = self.n_instance
        try:
            self._datasets = list(instances)
            self.n_instance = len(self._datasets)
            return self.evaluate_2d(callable_func)
        finally:
            self._datasets = orig_datasets
            self.n_instance = orig_n


class BP2DInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for 2D bin packing."""

    def __init__(self,
                 n_items: int = 100,
                 bin_width: int = 100,
                 bin_height: int = 100,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_items = n_items
        self.bin_width = bin_width
        self.bin_height = bin_height
        super().__init__(
            template_program=bp2d_instance_template_program,
            task_description=bp2d_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> tuple | None:
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
        item_dims, bin_dims = result
        try:
            item_dims = [(int(w), int(h)) for (w, h) in item_dims]
        except (TypeError, ValueError):
            return None
        if len(item_dims) != self.n_items:
            return None
        try:
            bin_w, bin_h = int(bin_dims[0]), int(bin_dims[1])
        except (TypeError, ValueError, IndexError):
            return None
        if bin_w != self.bin_width or bin_h != self.bin_height:
            return None
        for w, h in item_dims:
            if w < 1 or w >= self.bin_width:
                return None
            if h < 1 or h >= self.bin_height:
                return None
        return (item_dims, (bin_w, bin_h))
