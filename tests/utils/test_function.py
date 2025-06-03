from typing_extensions import Annotated

from cleanlab_codex.utils.function import pydantic_model_from_function


def test_function_schema_with_annotated_params() -> None:
    def function_with_annotated_params(
        a: Annotated[str, "This is a string"],  # noqa: ARG001
    ) -> None: ...

    fn_schema = pydantic_model_from_function("test_function", function_with_annotated_params)
    assert fn_schema.model_json_schema()["title"] == "test_function"
    assert fn_schema.model_fields["a"].annotation is str
    assert fn_schema.model_fields["a"].description == "This is a string"
    assert fn_schema.model_fields["a"].is_required()


def test_function_schema_without_annotations() -> None:
    def function_without_annotations(a) -> None:  # type: ignore # noqa: ARG001
        ...

    fn_schema = pydantic_model_from_function("test_function", function_without_annotations)
    assert fn_schema.model_json_schema()["title"] == "test_function"
    assert fn_schema.model_fields["a"].is_required()
    assert fn_schema.model_fields["a"].description is None


def test_function_schema_with_default_param() -> None:
    def function_with_default_param(a: int = 1) -> None:  # noqa: ARG001
        ...

    fn_schema = pydantic_model_from_function("test_function", function_with_default_param)
    assert fn_schema.model_json_schema()["title"] == "test_function"
    assert fn_schema.model_fields["a"].annotation is int
    assert fn_schema.model_fields["a"].default == 1
    assert not fn_schema.model_fields["a"].is_required()
    assert fn_schema.model_fields["a"].description is None
