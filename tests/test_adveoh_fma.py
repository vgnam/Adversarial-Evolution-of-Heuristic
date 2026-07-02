from __future__ import annotations

from threading import Lock
from types import SimpleNamespace

import pytest

from llm4ad.method.adveoh.adveoh import AdvEoH


class _DrawSampleOnlyLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def draw_sample(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return '  concise failure description  '


def test_fma_llm_adapter_uses_portable_draw_sample_interface() -> None:
    llm = _DrawSampleOnlyLLM()
    method = AdvEoH.__new__(AdvEoH)
    method._sampler_heu = SimpleNamespace(llm=llm)

    result = method._fma_llm_call('user prompt', 'system prompt')

    assert result == 'concise failure description'
    assert llm.prompts == ['system prompt\n\nuser prompt']


def test_instance_scoring_uses_generator_gated_coverage() -> None:
    method = AdvEoH.__new__(AdvEoH)
    method._use_fma = True
    method._pending_failure_lock = Lock()
    method._pending_failure_analysis = []

    scores = {'h1': -2.0, 'h2': -3.0}
    method._evaluate_heuristic_function = (
        lambda heuristic, instances: scores[heuristic]
    )
    coverage_calls = []
    method._fma_try_mark_covered = lambda **kwargs: coverage_calls.append(kwargs)
    generator = object()

    result = method._score_instances_against_reference(
        instances=['instance'],
        reference_heuristics=['h1', 'h2'],
        _generation=7,
        _generator_func=generator,
    )

    assert result == pytest.approx(2.0)
    assert len(coverage_calls) == 2
    assert all(call['generator_func'] is generator for call in coverage_calls)
    assert len(method._pending_failure_analysis) == 2
