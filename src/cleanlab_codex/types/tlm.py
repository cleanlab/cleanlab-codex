from typing import Any, Dict, Protocol, Sequence, Union, runtime_checkable


@runtime_checkable
class TLM(Protocol):
    def get_trustworthiness_score(
        self,
        prompt: Union[str, Sequence[str]],
        response: Union[str, Sequence[str]],
        **kwargs: Any,
    ) -> Dict[str, Any]: ...

    def prompt(
        self,
        prompt: Union[str, Sequence[str]],
        /,
        **kwargs: Any,
    ) -> Dict[str, Any]: ...
