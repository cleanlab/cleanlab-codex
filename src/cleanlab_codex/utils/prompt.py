"""
Helper functions for processing prompts in RAG applications.
"""


def default_format_prompt(query: str, context: str) -> str:
    """Default function for formatting RAG prompts.

    Args:
        query: The user's question
        context: The context/documents to use for answering

    Returns:
        str: A formatted prompt combining the query and context
    """
    template = (
        "Using only information from the following Context, answer the following Query.\n\n"
        "Context:\n{context}\n\n"
        "Query: {query}"
    )
    return template.format(context=context, query=query)
