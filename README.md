# Cleanlab Codex - Closing the AI Knowledge Gap

[![Build Status](https://github.com/cleanlab/cleanlab-codex/actions/workflows/ci.yml/badge.svg)](https://github.com/cleanlab/cleanlab-codex/actions/workflows/ci.yml) [![PyPI - Version](https://img.shields.io/pypi/v/cleanlab-codex.svg)](https://pypi.org/project/cleanlab-codex) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cleanlab-codex.svg)](https://pypi.org/project/cleanlab-codex) [![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://help.cleanlab.ai/codex/api/)

Codex enables you to seamlessly leverage knowledge from Subject Matter Experts (SMEs) to improve your RAG/Agentic applications.

The `cleanlab-codex` library provides a simple interface to integrate Codex's capabilities into your RAG application.
See immediate impact with just a few lines of code!

## Demo

Install the package:

```console
pip install cleanlab-codex
```

Integrating Codex into your RAG application is as simple as:

```python
from cleanlab_codex import Validator
validator = Validator(codex_access_key=...) # optional configurations can improve accuracy/latency

# Your existing RAG code:
context = rag_retrieve_context(user_query)
prompt = rag_form_prompt(user_query, retrieved_context)
response = rag_generate_response(prompt)

# Detect bad responses and remediate with Cleanlab
results = validator.validate(query=query, context=context, response=response,
    form_prompt=rag_form_prompt)

final_response = (
    results["expert_answer"] # Codex's answer
    if results["is_bad_response"] and results["expert_answer"]
    else response # Your RAG system's initial response
)
```

<!-- TODO: add demo video -->
<!-- Video should show Codex added to a RAG system, question asked that requires knowledge from an outside expert, Codex used to ask an outside expert, and expert response returned to the user -->

## Why Codex?
- **Detect Knowledge Gaps and Hallucinations**: Codex identifies knowledge gaps and incorrect/untrustworthy responses in your AI application, to help you know which questions require expert input.
- **Save SME time**: Codex ensures that SMEs see the most critical knowledge gaps first.
- **Easy Integration**: Integrate Codex into any RAG/Agentic application with just a few lines of code.
- **Immediate Impact**: SME answers instantly improve your AI, without any additional Engineering/technical work.

## Documentation

Comprehensive documentation along with tutorials and examples can be found [here](https://help.cleanlab.ai/codex).

## License

`cleanlab-codex` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
