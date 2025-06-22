# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.21] 2025-06-22

- Support adding remediations to a project
- Docstring updates

## [1.0.20] 2025-06-17

- Remove Codex-as-a-tool
- Remove support for deprecated entries data model

## [1.0.19] 2025-06-4

- Expose `eval_scores` property for `Validator.validate()` and use Pydantic types from Codex backend

## [1.0.18] 2025-06-3

- Expose `options` and `quality_preset` properties for `Validator.validate()`

## [1.0.17] 2025-06-3

- Refactor `validate()` to use `/validate` endpoint from Codex backend and leverage this default logic
- deprecate `project.query()` and `project.add_entries()`

## [1.0.16] 2025-05-15

- Update `codex-sdk` dependency to `0.1.0a20`.

## [1.0.15] 2025-04-24

- Update default thresholds for response helpfulness to 0.23 in `Validator` API.
- Update `codex-sdk` dependency to `0.1.0a19`.

## [1.0.14] 2025-04-23

- Update `codex-sdk` dependency to `0.1.0-alpha.17`.
- Capture data for the number of times the validator API is called on a Codex project.

## [1.0.13] - 2025-04-22

- Update `cleanlab-tlm` dependancy to `~=1.1`.

## [1.0.12] - 2025-04-17

- Support adding metadata in `validate()` method in Validator API.

## [1.0.11] - 2025-04-16

- Update default thresholds for custom evals to 0.0 in `Validator` API.

## [1.0.10] - 2025-04-15

- Add async support to `Validator` API.

## [1.0.9] - 2025-04-10

- Refactor threshold validation in the `Validator` class to only check user-provided metrics.

## [1.0.8] - 2025-04-03

- Update `Project.query()` method with optional `metadata` property to log and store arbitrary metadata.
- Remove `response_validation.py` module.

## [1.0.7] - 2025-04-02

- Update `Project.query()` method based on API changes from question grouping feature.

## [1.0.6] - 2025-03-27

- Fix links to docs

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

[Unreleased]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.21...HEAD
[1.0.21]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.20...v1.0.21
[1.0.20]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.19...v1.0.20
[1.0.19]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.18...v1.0.19
[1.0.18]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.17...v1.0.18
[1.0.17]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.16...v1.0.17
[1.0.16]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.15...v1.0.16
[1.0.15]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.14...v1.0.15
[1.0.14]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.13...v1.0.14
[1.0.13]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.12...v1.0.13
[1.0.12]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.11...v1.0.12
[1.0.11]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.10...v1.0.11
[1.0.10]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.9...v1.0.10
[1.0.9]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.8...v1.0.9
[1.0.8]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.7...v1.0.8
[1.0.7]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.6...v1.0.7
[1.0.6]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.5...v1.0.6
[1.0.5]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.4...v1.0.5
[1.0.4]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.3...v1.0.4
[1.0.3]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.2...v1.0.3
[1.0.2]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.1...v1.0.2
[1.0.1]: https://github.com/cleanlab/cleanlab-codex/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/cleanlab/cleanlab-codex/compare/267a93300f77c94e215d7697223931e7926cad9e...v1.0.0
