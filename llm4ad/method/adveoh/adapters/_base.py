"""Base class for instance generator evaluations.

Each per-task adapter subclasses ``InstanceGenEvaluationBase`` and implements
``validate_instance(result)`` to enforce the task's instance shape constraints.
The base class handles: subprocess-safe execution of ``generate_instance(seed)``,
exception handling, and returning ``None`` on failure (which AdvEoH treats as
an invalid generator).
"""
from __future__ import annotations

from typing import Any

from ....base import Evaluation


class InstanceGenEvaluationBase(Evaluation):
    """Base for instance generator evaluations.

    Subclasses MUST:
      - call ``super().__init__(template_program=..., task_description=..., timeout_seconds=...)``
      - override ``validate_instance(self, result)`` to check the output of
        ``generate_instance(seed)`` and return the validated instance data
        (in the same format as elements of the task's ``self._datasets``),
        or ``None`` if invalid.
    """

    def __init__(self,
                 template_program: str,
                 task_description: str,
                 timeout_seconds: int | float = 10,
                 **kwargs):
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
            **kwargs
        )

    def evaluate_program(self, program_str: str, callable_func: callable,
                         seed: int = 0, **kwargs) -> Any | None:
        """Run ``callable_func(seed)`` and validate the output.

        Returns the validated instance data (in task-native format) or None.
        NOTE: the returned value is NOT a fitness score — AdvEoH caches it on
        ``func.instances`` and later passes it to the heuristic evaluation as
        ``instances=[...]``.
        """
        try:
            result = callable_func(seed)
            validated = self.validate_instance(result)
            if validated is None:
                reason = getattr(self, '_last_validation_error', 'validate_instance returned None')
                print(f'[AdvEoH] Instance generator invalid at seed {seed}: {reason}')
            return validated
        except Exception as exc:
            print(f'[AdvEoH] Instance generator exception at seed {seed}: {type(exc).__name__}: {exc}')
            return None

    def validate_instance(self, result) -> Any | None:
        """Subclass must implement. Return validated instance data or None."""
        raise NotImplementedError("Subclass must implement validate_instance()")
