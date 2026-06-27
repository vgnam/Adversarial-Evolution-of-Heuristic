from __future__ import annotations

import re
from typing import Tuple, Any

from ...base import LLM, SampleTrimmer, Function, Program, TextFunctionProgramConverter


class AdvEoHSampler:
    """Sampler for AdvEoH. Identical to EoHSampler: calls LLM, extracts thought
    from the first ``{...}`` block, and parses the response into a Function.
    """

    def __init__(self, llm: LLM, template_program: str | Program):
        self.llm = llm
        self._template_program = template_program

    def get_thought_function_and_usage(self, prompt: str) -> Tuple[str, Function, dict[str, Any] | None]:
        response = self.llm.draw_sample(prompt)
        token_usage = (
            self.llm.get_last_token_usage()
            if hasattr(self.llm, 'get_last_token_usage') else None
        )
        thought = self.__class__.trim_thought_from_response(response)
        function = self._response_to_function(response)
        return thought, function, token_usage

    def get_thought_and_function(self, prompt: str) -> Tuple[str, Function]:
        thought, function, _token_usage = self.get_thought_function_and_usage(prompt)
        return thought, function

    def _response_to_function(self, response: str) -> Function | None:
        candidates = self.__class__._code_candidates(response)
        for candidate in candidates:
            code = SampleTrimmer.trim_preface_of_function(candidate)
            function = SampleTrimmer.sample_to_function(code, self._template_program)
            if function is not None:
                return function

            function = self.__class__._function_from_full_program(candidate, self._template_program)
            if function is not None:
                return function
        return None

    @classmethod
    def _code_candidates(cls, response: str) -> list[str]:
        if not response:
            return ['']

        candidates = []
        fenced = re.findall(r'```(?:python|py)?\s*(.*?)```', response, flags=re.DOTALL | re.IGNORECASE)
        candidates.extend(block.strip() for block in fenced if block.strip())

        cleaned = re.sub(r'```(?:python|py)?', '', response, flags=re.IGNORECASE)
        cleaned = cleaned.replace('```', '')
        candidates.append(cleaned)
        candidates.append(re.sub(r'^\s*\{.*?\}[ \t]*(?:\r?\n)?', '', cleaned, flags=re.DOTALL))

        seen = set()
        unique = []
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            unique.append(candidate)
        return unique

    @classmethod
    def _function_from_full_program(
            cls,
            candidate: str,
            template_program: str | Program,
    ) -> Function | None:
        program = TextFunctionProgramConverter.text_to_program(candidate)
        if program is None or not program.functions:
            return None

        template = (
            TextFunctionProgramConverter.text_to_program(template_program)
            if isinstance(template_program, str)
            else template_program
        )
        target_name = template.functions[0].name if template and template.functions else None
        for function in program.functions:
            if target_name is None or function.name == target_name:
                return function
        return None

    @classmethod
    def trim_thought_from_response(cls, response: str) -> str | None:
        try:
            pattern = r'\{.*?\}'
            bracketed_texts = re.findall(pattern, response)
            return bracketed_texts[0]
        except Exception:
            return None
