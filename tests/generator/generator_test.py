from typing import Dict

import pytest
from src.generator import GPT3Generator


@pytest.mark.parametrize('instructions, examples, prompt', [('List the ingredients for this meal',
                                                             {'fries': 'potato, oil'},
                                                             'List the ingredients for this meal\n\nInput: fries\nOutput: potato, oil')]
)
def test_prompt(instructions: str, examples: Dict[str, str], prompt: str) -> None:
    generator = GPT3Generator('davinci', 10)
    generator.set_instructions(instructions)
    for k, v in examples.items():
        generator.add_example(k, v)
    assert generator.get_prompt() == prompt
