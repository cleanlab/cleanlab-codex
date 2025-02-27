"""Types for Codex TLM endpoint."""

from codex.types.tlm_score_response import TlmScoreResponse as _TlmScoreResponse

from cleanlab_codex.internal.utils import generate_class_docstring


class TlmScoreResponse(_TlmScoreResponse): ...


TlmScoreResponse.__doc__ = f"""
Type representing a TLM score response in a Codex project. This is the complete data structure returned from the Codex API, including system-generated fields like ID and timestamps.

{generate_class_docstring(_TlmScoreResponse, name=TlmScoreResponse.__name__)}
"""

__all__ = ["TlmScoreResponse"]
