from cleanlab_codex.utils.prompt import default_format_prompt


def test_default_format_prompt() -> None:
    """Test the default_format_prompt function correctly formats the template."""
    # Test with simple inputs
    query = "What is the capital of France?"
    context = "France is a country in Europe. The capital of France is Paris."
    expected = (
        "Using only information from the following Context, answer the following Query.\n\n"
        "Context:\n"
        "France is a country in Europe. The capital of France is Paris.\n\n"
        "Query: What is the capital of France?"
    )

    result = default_format_prompt(query, context)
    assert result == expected

    # Test with multi-line context
    query = "Explain quantum mechanics"
    context = "Quantum mechanics is a theory in physics.\nIt explains the behavior of matter at atomic scales.\nIt was developed in the early 20th century."
    expected = (
        "Using only information from the following Context, answer the following Query.\n\n"
        "Context:\n"
        "Quantum mechanics is a theory in physics.\nIt explains the behavior of matter at atomic scales.\nIt was developed in the early 20th century.\n\n"
        "Query: Explain quantum mechanics"
    )

    result = default_format_prompt(query, context)
    assert result == expected

    # Test with empty strings
    query = ""
    context = ""
    expected = (
        "Using only information from the following Context, answer the following Query.\n\n"
        "Context:\n"
        "\n\n"
        "Query: "
    )

    result = default_format_prompt(query, context)
    assert result == expected
