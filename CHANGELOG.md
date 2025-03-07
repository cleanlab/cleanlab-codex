# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Extract scores and metadata from detection functions in `response_validation.py`.
- Normalize scores used by `is_fallback_response` function to be between 0 and 1.

## [1.0.2] - 2025-03-07

- Pass metadata in headers for query requests.

## [1.0.1] - 2025-02-26

- Updates to logic for `is_unhelpful_response` util method.

## [1.0.0] - 2025-02-18

- Initial release of the `cleanlab-codex` client library.

[Unreleased]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.2...HEAD
[1.0.2]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/cleanlab/cleanlab-codex/compare/267a93300f77c94e215d7697223931e7926cad9e...v1.0.0
