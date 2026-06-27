from __future__ import annotations

import copy
from typing import List, Optional

from ...base import Function


# =============================================================================
# HeuristicPrompt  (for the heuristic LLM — solves the task)
# =============================================================================

class HeuristicPrompt:
    """Prompts for the heuristic LLM. Mirrors EoHPrompt but adds an optional
    ``opponent_descriptions`` argument: a list of natural-language descriptions
    of the hardest instances (from the instance LLM's ``algorithm`` field).
    When provided, the prompt instructs the heuristic LLM to counter those
    specific instance characteristics.
    """

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def _opponent_section(cls, opponent_descriptions: Optional[List[str]], role: str = 'instances') -> str:
        if not opponent_descriptions:
            return ''
        header = (
            f'\nThe current hardest {role} describe themselves as:\n'
        )
        body = ''.join(f'  - "{d}"\n' for d in opponent_descriptions)
        footer = (
            f'\nGenerate a heuristic that performs well against THESE specific '
            f'{role[:-1] if role.endswith("s") else role} characteristics.\n'
        )
        return header + body + footer

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        opponent = cls._opponent_section(opponent_descriptions)
        prompt_content = f'''{task_prompt}{opponent}
1. First, describe your new algorithm and main steps in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This counters instances that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi = copy.deepcopy(indi)
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n'
        opponent = cls._opponent_section(opponent_descriptions)
        prompt_content = f'''{task_prompt}{opponent}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This counters instances that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi = copy.deepcopy(indi)
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} algorithm and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n'
        opponent = cls._opponent_section(opponent_descriptions)
        prompt_content = f'''{task_prompt}{opponent}
I have {len(indivs)} existing algorithms with their codes as follows:
{indivs_prompt}
Please help me create a new algorithm that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided algorithms.
2. Secondly, based on the backbone idea describe your new algorithm in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This counters instances that <reference to the descriptions above>."
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indi = copy.deepcopy(indi)
        indi.docstring = ''
        opponent = cls._opponent_section(opponent_descriptions)
        prompt_content = f'''{task_prompt}{opponent}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new algorithm that has a different form but can be a modified version of the algorithm provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This counters instances that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indi = copy.deepcopy(indi)
        indi.docstring = ''
        opponent = cls._opponent_section(opponent_descriptions)
        prompt_content = f'''{task_prompt}{opponent}
I have one algorithm with its code as follows. Algorithm description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main algorithm parameters and assist me in creating a new algorithm that has a different parameter settings of the score function provided.
1. First, describe your new algorithm and main steps in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This counters instances that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content


# =============================================================================
# InstancePrompt  (for the instance LLM — generates hard instances)
# =============================================================================

class InstancePrompt:
    """Prompts for the instance LLM. Same operator structure as HeuristicPrompt
    but the ``opponent_descriptions`` are descriptions of the strongest heuristics.
    The prompt instructs the instance LLM to generate instances that defeat those
    specific heuristics.
    """

    @classmethod
    def get_system_prompt(cls) -> str:
        return ''

    @classmethod
    def _format_section(cls) -> str:
        return (
            '\nSTRICT OUTPUT FORMAT (the evaluator will REJECT any violation and '
            'your generator will score None):\n'
            '  def generate_instance(seed: int) -> tuple:\n'
            '      ...\n'
            '      return <one task-native instance>\n'
            '  - Follow the task-specific return format and validation rules in the '
            'function docstring below exactly.\n'
            '  - Use the requested problem size from that docstring; do not change '
            'the size unless the template explicitly says so.\n'
            '  - All returned arrays/numbers must be finite and satisfy the task '
            'constraints; no NaN, no Inf, no degenerate invalid cases.\n'
            '  - use `seed` via np.random.default_rng(seed) so the instance is '
            'reproducible for different seeds\n'
            '  - If you generate coordinates and add noise, clip them back to '
            'the valid range before returning.\n'
            '  - Recompute any derived matrices from the final returned data; '
            'do not hand-edit symmetric matrices.\n'
            '  - If the template says the evaluator computes derived data '
            'automatically, return only the primitive instance data requested.\n'
        )

    @classmethod
    def _example_section(cls, template_function: Function) -> str:
        example = copy.deepcopy(template_function)
        if not example.body:
            return ''
        return (
            '\nVALID BASELINE GENERATOR EXAMPLE (copy this structure and return '
            'format; change the sampling logic to make harder instances):\n'
            f'{str(example)}'
        )

    @classmethod
    def _output_format_section(cls) -> str:
        return (
            '\nOUTPUT FORMAT: match the heuristic LLM format exactly.\n'
            '  1. Start with one strategy sentence inside braces: {My strategy: ...}\n'
            '  2. Then output one Python implementation of the requested function.\n'
            '  3. Do not use Markdown fences such as ```python.\n'
            '  4. Do not include imports, helper functions, classes, tests, or any '
            'text after the function.\n'
        )

    @classmethod
    def _opponent_section(cls, opponent_descriptions: Optional[List[str]]) -> str:
        if not opponent_descriptions:
            return ''
        header = '\nThe current strongest heuristics describe themselves as:\n'
        body = ''.join(f'  - "{d}"\n' for d in opponent_descriptions)
        footer = (
            '\nGenerate an instance generator that produces instances hard for '
            'THESE specific heuristics.\n'
        )
        return header + body + footer

    @classmethod
    def get_prompt_i1(cls, task_prompt: str, template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        opponent = cls._opponent_section(opponent_descriptions)
        fmt = cls._format_section()
        example = cls._example_section(template_function)
        out_fmt = cls._output_format_section()
        prompt_content = f'''{task_prompt}{opponent}{fmt}{example}{out_fmt}
1. First, describe your instance generation strategy in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This defeats heuristics that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e1(cls, task_prompt: str, indivs: List[Function], template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi = copy.deepcopy(indi)
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} generator and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n'
        opponent = cls._opponent_section(opponent_descriptions)
        fmt = cls._format_section()
        example = cls._example_section(template_function)
        out_fmt = cls._output_format_section()
        prompt_content = f'''{task_prompt}{opponent}{fmt}{example}{out_fmt}
I have {len(indivs)} existing instance generators with their codes as follows:
{indivs_prompt}
Please help me create a new instance generator that has a totally different form from the given ones.
1. First, describe your instance generation strategy in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This defeats heuristics that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_e2(cls, task_prompt: str, indivs: List[Function], template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        for indi in indivs:
            assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indivs_prompt = ''
        for i, indi in enumerate(indivs):
            indi = copy.deepcopy(indi)
            indi.docstring = ''
            indivs_prompt += f'No. {i + 1} generator and the corresponding code are:\n{indi.algorithm}\n{str(indi)}\n'
        opponent = cls._opponent_section(opponent_descriptions)
        fmt = cls._format_section()
        example = cls._example_section(template_function)
        out_fmt = cls._output_format_section()
        prompt_content = f'''{task_prompt}{opponent}{fmt}{example}{out_fmt}
I have {len(indivs)} existing instance generators with their codes as follows:
{indivs_prompt}
Please help me create a new instance generator that has a totally different form from the given ones but can be motivated from them.
1. Firstly, identify the common backbone idea in the provided generators.
2. Secondly, based on the backbone idea describe your new generator in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This defeats heuristics that <reference to the descriptions above>."
3. Thirdly, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m1(cls, task_prompt: str, indi: Function, template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indi = copy.deepcopy(indi)
        indi.docstring = ''
        opponent = cls._opponent_section(opponent_descriptions)
        fmt = cls._format_section()
        example = cls._example_section(template_function)
        out_fmt = cls._output_format_section()
        prompt_content = f'''{task_prompt}{opponent}{fmt}{example}{out_fmt}
I have one instance generator with its code as follows. Generator description:
{indi.algorithm}
Code:
{str(indi)}
Please assist me in creating a new instance generator that has a different form but can be a modified version of the generator provided.
1. First, describe your instance generation strategy in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This defeats heuristics that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content

    @classmethod
    def get_prompt_m2(cls, task_prompt: str, indi: Function, template_function: Function,
                      opponent_descriptions: Optional[List[str]] = None) -> str:
        assert hasattr(indi, 'algorithm')
        temp_func = copy.deepcopy(template_function)
        temp_func.body = ''
        indi = copy.deepcopy(indi)
        indi.docstring = ''
        opponent = cls._opponent_section(opponent_descriptions)
        fmt = cls._format_section()
        example = cls._example_section(template_function)
        out_fmt = cls._output_format_section()
        prompt_content = f'''{task_prompt}{opponent}{fmt}{example}{out_fmt}
I have one instance generator with its code as follows. Generator description:
{indi.algorithm}
Code:
{str(indi)}
Please identify the main parameters of the generator and assist me in creating a new instance generator that has a different parameter settings.
1. First, describe your instance generation strategy in one sentence. The description must be inside boxed {{}}.
   Format: "My strategy: <strategy>. This defeats heuristics that <reference to the descriptions above>."
2. Next, implement the following Python function:
{str(temp_func)}
Do not give additional explanations.'''
        return prompt_content
