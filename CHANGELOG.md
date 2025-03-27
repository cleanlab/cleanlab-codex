# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.5] - 2025-03-27

- Add `Validator` API
- Deprecate `response_validation.py` module.

## [1.0.4] - 2025-03-14

- Pass analytics metadata in headers for all Codex API requests.

## [1.0.3] - 2025-03-11

- Update response validation methods for Codex as backup to use TLM through Codex API instead of requiring separate TLM API key.

## [1.0.2] - 2025-03-07

- Extract scores and metadata from detection functions in `response_validation.py`.
- Normalize scores used by `is_fallback_response` function to be between 0 and 1.
- Pass metadata in headers for query requests.

## [1.0.1] - 2025-02-26

- Updates to logic for `is_unhelpful_response` util method.

## [1.0.0] - 2025-02-18

- Initial release of the `cleanlab-codex` client library.

[Unreleased]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.5...HEAD
[1.0.5]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/cleanlab/cleanlab-codex/compare/267a93300f77c94e215d7697223931e7926cad9e...v1.0.0
