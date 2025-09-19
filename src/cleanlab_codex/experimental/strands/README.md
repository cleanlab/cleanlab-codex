# Cleanlab Strands Integration

This module provides integration between Cleanlab validation and the Strands framework, allowing you to wrap any Strands model with Cleanlab's response validation capabilities and pass that model to a Strands Agent.

## Overview

The `CleanlabModel` class wraps existing Strands models to add real-time response validation, guardrails, and expert escalation through Cleanlab's platform. It maintains compatibility with the Strands framework while adding validation capabilities.

## Installation

### 1. Install cleanlab-codex

First, install the cleanlab-codex package:

```bash
pip install cleanlab-codex
```

### 2. Install Strands

Install the Strands framework:

```bash
pip install strands-agents
```

### 3. Install Additional Dependencies

Then install any additional dependencies for the experimental integration:

```bash
pip install -r requirements.txt
```

Alternatively, you can install all dependencies at once:

```bash
pip install cleanlab-codex strands-agents "strands-agents[openai]"
```

**Note**: Users will need to install `cleanlab-codex` separately to create the `Project` instance that gets passed to `CleanlabModel`, but the strands integration itself only requires the packages listed above.

## Basic Usage

```python
import uuid

from strands.agent.agent import Agent
from strands.models.openai import OpenAIModel
from strands.session.file_session_manager import FileSessionManager

from cleanlab_codex.experimental.strands import CleanlabModel
from cleanlab_codex import Project

# Initialize your Cleanlab project
project = Project.from_access_key("your_access_key_here")

SYSTEM_PROMPT = "You are a customer service agent. Be polite and concise in your responses."
FALLBACK_RESPONSE = "Sorry I am unsure. You can try rephrasing your request."

# Create base model
base_model = OpenAIModel(
    model_id="gpt-4o-mini",
)

### New code to add for Cleanlab API ###
cleanlab_model = CleanlabModel( # Wrap with Cleanlab validation
    underlying_model=base_model,
    cleanlab_project=project,
    fallback_response=FALLBACK_RESPONSE,
    # context_retrieval_tools=["web_search", "get_payment_schedule", "get_total_amount_owed"]  # Specify tool(s) that provide context here
)
### End of new code to add for Cleanlab API ###

# Create agent with validated model for normal conversation
agent = Agent(
    model=cleanlab_model,
    system_prompt=SYSTEM_PROMPT,
    # tools=[get_payment_schedule, get_total_amount_owed, web_search],  # Add tools in Strands format here
    session_manager=FileSessionManager(session_id=uuid.uuid4().hex),  # Persist chat history
)

### New code to add for Cleanlab API ###
cleanlab_model.set_agent_reference(agent)
### End of new code to add for Cleanlab API ###

# Use the agent normally - validation happens automatically
response = agent("What's my payment schedule?")
```

## Configuration Options

### CleanlabModel Parameters

- `underlying_model`: The base Strands model to wrap
- `cleanlab_project`: Your Cleanlab project instance
- `fallback_response`: Response to return when guardrails are triggered
- `context_retrieval_tools`: List of tool names that provide context for validation
- `skip_validating_tool_calls`: Skip validation when response contains tool calls (default: True)

## Important Notes

1. **Agent Reference**: Always call `cleanlab_model.set_agent_reference(agent)` after creating your agent for full functionality
2. **Tool Validation**: By default, responses containing tool calls skip validation. Set `skip_validating_tool_calls=False` to validate all responses, including Agent responses with tool calls.
3. **Session Management**: The Codex UI uses session IDs for tracking chat history for validation results when available.

## Dependencies

### Runtime Dependencies
- `cleanlab-tlm>=1.1.14`
- `strands-agents`

### Type Checking Only (Optional)
- `cleanlab-codex` (for Project type hints)
- `codex-sdk` (for ProjectValidateResponse type hints)
- `openai` (for OpenAI type hints)