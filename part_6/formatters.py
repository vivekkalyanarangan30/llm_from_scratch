"""Prompt/response formatting utilities (6.1).
We keep a very simple template with clear separators.
"""
from dataclasses import dataclass

template = (
    "<s>\n"
    "### Instruction:\n{instruction}\n\n"
    "### Response:\n{response}</s>"
)

@dataclass
class Example:
    instruction: str
    response: str


def format_example(ex: Example) -> str:
    return template.format(instruction=ex.instruction.strip(), response=ex.response.strip())


def format_prompt_only(instruction: str) -> str:
    return template.format(instruction=instruction.strip(), response="")