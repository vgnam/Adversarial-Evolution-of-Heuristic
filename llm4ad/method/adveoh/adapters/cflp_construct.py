"""CFLP (Capacitated Facility Location, constructive) adapter for AdvEoH.

Exports:
  - AdvCFLPEvaluation
  - CFLPInstanceGenEvaluation
  - cflp_instance_template_program
  - cflp_instance_task_description

Instance shape: dict {"facility_capacities": List[int] of length n_facilities,
                      "customer_demands":  List[int] of length n_customers,
                      "assignment_costs":  List[List[int]] shape (n_facilities, n_customers)}.
Defaults: n_facilities=50, n_customers=50, values: caps in [5,100], demands in [5,20], costs in [5,50].

FEASIBILITY: the adapter rejects instances where ``sum(customer_demands) >
sum(facility_capacities)`` (heuristic cannot assign all customers). The
original generator does NOT enforce this; we add it to avoid wasted evaluations.
"""
from __future__ import annotations

from typing import Any, List

from ._base import InstanceGenEvaluationBase
from ....base import Evaluation
from ....task.optimization.cflp_construct.evaluation import CFLPEvaluation

__all__ = [
    'AdvCFLPEvaluation', 'CFLPInstanceGenEvaluation',
    'cflp_instance_template_program', 'cflp_instance_task_description',
]


cflp_instance_template_program = '''
import numpy as np

def generate_instance(seed: int) -> dict:
    """Generate a hard CFLP instance for a constructive heuristic.

    A harder instance is one on which a greedy customer-assignment heuristic
    produces a HIGHER total assignment cost than on random uniform instances.

    Args:
        seed: int, random seed. Use `rng = np.random.default_rng(seed)`.

    Returns:
        A dict with exactly these keys:
          - "facility_capacities": list of ints, length exactly 50, each in [5, 100].
          - "customer_demands":    list of ints, length exactly 50, each in [5, 20].
          - "assignment_costs":    list of lists of ints, shape (50, 50),
                                    each value in [5, 50].
                                    assignment_costs[i][j] = cost of assigning
                                    customer j to facility i.

    FEASIBILITY: the adapter REJECTS the instance if
        sum(customer_demands) > sum(facility_capacities)
    (i.e. there isn't enough total capacity to serve all customers). Make sure
    your generated capacities can cover the total demand.

    The evaluator also REJECTS the instance if:
      - result is not a dict with the three required keys,
      - any list has the wrong length or value out of range.
    """
    rng = np.random.default_rng(seed)
    facility_capacities = rng.integers(5, 101, size=50).tolist()
    customer_demands = rng.integers(5, 21, size=50).tolist()
    assignment_costs = rng.integers(5, 51, size=(50, 50)).tolist()
    return {
        "facility_capacities": facility_capacities,
        "customer_demands": customer_demands,
        "assignment_costs": assignment_costs,
    }
'''

cflp_instance_task_description = (
    "You are designing a hard instance generator for the CAPACITATED FACILITY "
    "LOCATION problem. A constructive heuristic iteratively assigns customers "
    "to facilities (given current assignments, remaining customers, remaining "
    "facility capacities, customer demands, and the assignment cost matrix) "
    "to minimize total cost. Your goal: produce 50 facilities (caps in "
    "[5,100]) and 50 customers (demands in [5,20]) with a 50x50 cost matrix "
    "(values in [5,50]) that make greedy heuristics produce HIGHER total "
    "costs than on random uniform instances.\n\n"
    "FEASIBILITY: total facility capacity MUST be >= total customer demand, "
    "or the adapter rejects the instance. Exploit this margin — e.g. make "
    "capacity just barely enough so the heuristic has no slack.\n\n"
    "STRATEGY HINT: do NOT just use uniform random costs. Exploit the target "
    "heuristics below — e.g. anti-correlated cost/capacity (cheap facilities "
    "have tiny capacity), high-cost clusters that trap greedy choices, or "
    "demand patterns that force capacity-exhaustion early."
)


class AdvCFLPEvaluation(CFLPEvaluation):
    """Subclass of CFLPEvaluation that accepts an optional ``instances`` kwarg.
    NOTE: CFLPEvaluation.evaluate_program calls ``evaluate_cflp``, not ``evaluate``.
    """

    def evaluate_program(self, program_str: str, callable_func: callable,
                         instances: list | None = None, **kwargs) -> Any | None:
        if instances is None:
            return self.evaluate_cflp(callable_func)
        orig_datasets = self._datasets
        orig_n = self.n_instance
        try:
            self._datasets = list(instances)
            self.n_instance = len(self._datasets)
            return self.evaluate_cflp(callable_func)
        finally:
            self._datasets = orig_datasets
            self.n_instance = orig_n


class CFLPInstanceGenEvaluation(InstanceGenEvaluationBase):
    """Validates output of generate_instance(seed) for CFLP.

    Enforces: shape, value ranges, and feasibility (sum(cap) >= sum(demand)).
    """

    def __init__(self,
                 n_facilities: int = 50,
                 n_customers: int = 50,
                 max_capacity: int = 100,
                 max_demand: int = 20,
                 max_cost: int = 50,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        self.n_facilities = n_facilities
        self.n_customers = n_customers
        self.max_capacity = max_capacity
        self.max_demand = max_demand
        self.max_cost = max_cost
        super().__init__(
            template_program=cflp_instance_template_program,
            task_description=cflp_instance_task_description,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def validate_instance(self, result) -> dict | None:
        if not isinstance(result, dict):
            return None
        try:
            caps = [int(c) for c in result["facility_capacities"]]
            dems = [int(d) for d in result["customer_demands"]]
            costs = [[int(c) for c in row] for row in result["assignment_costs"]]
        except (KeyError, TypeError, ValueError):
            return None
        if len(caps) != self.n_facilities:
            return None
        if len(dems) != self.n_customers:
            return None
        if len(costs) != self.n_facilities:
            return None
        for row in costs:
            if len(row) != self.n_customers:
                return None
        if any(c < 5 or c > self.max_capacity for c in caps):
            return None
        if any(d < 5 or d > self.max_demand for d in dems):
            return None
        for row in costs:
            for v in row:
                if v < 5 or v > self.max_cost:
                    return None
        if sum(dems) > sum(caps):
            return None
        return {
            "facility_capacities": caps,
            "customer_demands": dems,
            "assignment_costs": costs,
        }
