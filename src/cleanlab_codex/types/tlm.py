"""Types for TLM."""

from typing import Literal, Optional

from codex.types.tlm_prompt_params import Options
from codex.types.tlm_prompt_response import TlmPromptResponse as _TlmPromptResponse
from codex.types.tlm_score_response import TlmScoreResponse as _TlmScoreResponse
from pydantic import BaseModel, Field

from cleanlab_codex.internal.utils import generate_class_docstring, generate_pydantic_model_docstring

TLMQualityPreset = Literal["best", "high", "medium", "low", "base"]


class TLMOptions(Options): ...


TLMOptions.__doc__ = f"""
Customization options for querying TLM. For details, see the [TLM documentation](/tlm/api/python/tlm/#class-tlmoptions).

{generate_class_docstring(Options, name=TLMOptions.__name__)}
"""


class TLMPromptResponse(_TlmPromptResponse): ...


TLMPromptResponse.__doc__ = f"""
The response from prompting TLM.

{generate_class_docstring(_TlmPromptResponse, name=TLMPromptResponse.__name__)}
"""


class TLMScoreResponse(_TlmScoreResponse): ...


TLMScoreResponse.__doc__ = f"""
The result of scoring a response with TLM.

{generate_class_docstring(_TlmScoreResponse, name=TLMScoreResponse.__name__)}
"""


class TLMConfig(BaseModel):
    """Advanced configuration options for TLM."""

    quality_preset: Optional[TLMQualityPreset] = Field(
        default=None,
        description="See [TLM documentation for more information](/tlm/api/python/tlm/#class-tlm). If not provided, the default preset used is 'low'.",
    )
    task: Optional[str] = Field(
        default=None,
        description="See [TLM documentation for more information](/tlm/api/python/tlm/#class-tlm). If not provided, the default task used is 'default'.",
    )
    options: Optional[TLMOptions] = Field(
        default=None,
        description="See [TLM documentation for more information](/tlm/api/python/tlm/#class-tlmoptions) (including defaults).",
    )


TLMConfig.__doc__ = f"""
{TLMConfig.__doc__}

{generate_pydantic_model_docstring(TLMConfig, name=TLMConfig.__name__)}
"""
