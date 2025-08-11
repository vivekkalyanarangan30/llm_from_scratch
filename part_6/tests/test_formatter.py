from formatters import Example, format_example, format_prompt_only

def test_template_contains_markers():
    ex = Example("Say hi","Hello!")
    s = format_example(ex)
    assert "### Instruction:" in s and "### Response:" in s
    p = format_prompt_only("Explain transformers.")
    assert p.endswith("### Response:\n") or p.endswith("### Response:\n</s>")