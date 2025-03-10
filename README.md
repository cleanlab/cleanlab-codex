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

Integrating Codex into your RAG application as a tool is as simple as:

```python
from cleanlab_codex import CodexTool

def rag(question, system_prompt, tools) -> str:
    """Your RAG/Agentic code here"""
    ...

# Initialize the Codex tool
codex_tool = CodexTool.from_access_key("your-access-key")

# Update your system prompt to include information on how to use the Codex tool
system_prompt = f"""Answer the user's Question based on the following Context. If the Context doesn't adequately address the Question, use the {codex_tool.tool_name} tool to ask an outside expert."""

# Convert the Codex tool to a framework-specific tool
framework_specific_codex_tool = codex_tool.to_<framework_name>_tool() # i.e. codex_tool.to_llamaindex_tool(), codex_tool.to_openai_tool(), etc.

# Pass the Codex tool to your RAG/Agentic framework
response = rag(question, system_prompt, [framework_specific_codex_tool])
```

(Note: Exact code will depend on the RAG/Agentic framework you are using. [Other integrations](https://help.cleanlab.ai/codex/concepts/integrations/) are available if you prefer to avoid Tool Calls.)
<!-- TODO: add demo video -->
<!-- Video should show Codex tool added to a RAG system, question asked that requires knowledge from an outside expert, Codex tool used to ask an outside expert, and expert response returned to the user -->

## Why Codex?
- **Detect Knowledge Gaps and Hallucinations**: Codex identifies knowledge gaps and incorrect/untrustworthy responses in your AI application, to help you know which questions require expert input.
- **Save SME time**: Codex ensures that SMEs see the most critical knowledge gaps first.
- **Easy Integration**: Integrate Codex into any RAG/Agentic application with just a few lines of code.
- **Immediate Impact**: SME answers instantly improve your AI, without any additional Engineering/technical work.

## Documentation

Comprehensive documentation along with tutorials and examples can be found [here](https://help.cleanlab.ai/codex).

## License

`cleanlab-codex` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
