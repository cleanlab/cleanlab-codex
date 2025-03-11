from __future__ import annotations

from typing import Any, Dict, List, Optional

from cleanlab_codex.internal.sdk_client import (
    MissingAuthKeyError,
    client_from_access_key,
    client_from_api_key,
)
from cleanlab_codex.types.tlm import (
    TLMConfig,
    TLMOptions,
    TLMPromptResponse,
    TLMQualityPreset,
    TLMScoreResponse,
)


class TLM:
    def __init__(
        self,
        quality_preset: Optional[TLMQualityPreset] = None,
        *,
        task: Optional[str] = None,
        options: Optional[TLMOptions] = None,
        codex_access_key: Optional[str] = None,
    ):
        try:
            self._sdk_client = client_from_access_key(key=codex_access_key)
        except MissingAuthKeyError:
            self._sdk_client = client_from_api_key()

        self._tlm_kwargs: Dict[str, Any] = {}
        if quality_preset:
            self._tlm_kwargs["quality_preset"] = quality_preset
        if task:
            self._tlm_kwargs["task"] = task
        if options:
            self._tlm_kwargs["options"] = options

    @classmethod
    def from_config(cls, config: TLMConfig, *, codex_access_key: Optional[str] = None) -> TLM:
        return cls(**config.model_dump(), codex_access_key=codex_access_key)

    def prompt(
        self,
        prompt: str,
        *,
        constrain_outputs: Optional[List[str]] = None,
    ) -> TLMPromptResponse:
        return TLMPromptResponse.model_validate(
            self._sdk_client.tlm.prompt(
                prompt=prompt,
                constrain_outputs=constrain_outputs,
                **self._tlm_kwargs,
            ).model_dump()
        )

    def get_trustworthiness_score(
        self,
        prompt: str,
        response: str,
        *,
        constrain_outputs: Optional[List[str]] = None,
    ) -> TLMScoreResponse:
        return TLMScoreResponse.model_validate(
            self._sdk_client.tlm.score(
                prompt=prompt,
                response=response,
                constrain_outputs=constrain_outputs,
                **self._tlm_kwargs,
            ).model_dump()
        )
