"""Types for Codex projects."""

from codex.types.project_create_params import Config


class ProjectConfig(Config):
    """
    Type representing options that can be configured for a Codex project.

    ```python
    class ProjectConfig(TypedDict):
        max_distance: float = 0.1
    ```
    ---

    #### <kbd>property</kbd> max_distance

    Distance threshold used to determine if two questions are similar when querying existing Entries in a project.
    The metric used is cosine distance. Valid threshold values range from 0 (identical vectors) to 1 (orthogonal vectors).
    While cosine distance can extend to 2 (opposite vectors), we limit this value to 1 since finding matches that are less similar than "unrelated" (orthogonal)
    content would not improve results of the system querying the Codex project.
    """


__all__ = ["ProjectConfig"]
