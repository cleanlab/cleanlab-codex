"""Types for Codex projects."""

from codex.types.project_create_params import Config

from cleanlab_codex.internal.utils import generate_class_docstring


class ProjectConfig(Config): ...


ProjectConfig.__doc__ = f"""
    Type representing options that can be configured for a Codex project.

    {generate_class_docstring(Config, name=ProjectConfig.__name__)}
    ---

    #### <kbd>property</kbd> max_distance

    Distance threshold used to determine if two questions in a project are similar.
    The metric used is cosine distance. Valid threshold values range from 0 (identical vectors) to 1 (orthogonal vectors).
    While cosine distance can extend to 2 (opposite vectors), we limit this value to 1 since finding matches that are less similar than "unrelated" (orthogonal)
    content would not improve results of the system querying the Codex project.
    """


__all__ = ["ProjectConfig"]
