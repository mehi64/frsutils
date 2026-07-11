# Support policy

`frsutils` is a research software project maintained on a best-effort basis.
Public questions and reproducible reports are preferred because they create a
searchable record that can help other users and reviewers.

## Usage questions

Open a **Question or usage help** issue for help with installation, public API
usage, model configuration, execution modes, or interpretation of documented
outputs. Include a minimal example, Python version, operating system, package
version, and the complete error message when applicable.

Before opening an issue, check:

- the [documentation](https://mehi64.github.io/frsutils/),
- the [public API guide](docs/user/public_api.md),
- existing issues and closed issues.

## Bug reports

Use the **Bug report** issue form. A useful report includes:

- a minimal reproducible example,
- expected and observed behavior,
- package and dependency versions,
- operating system and Python version,
- the full traceback,
- whether the problem occurs with dense, blockwise, NumPy, or CuPy execution.

Please do not include confidential datasets. Replace them with synthetic or
anonymized data that still reproduces the problem.

## Feature and research proposals

Use the **Feature or scientific enhancement** issue form. Explain the research
use case, mathematical definition, expected public API, alternatives considered,
and how correctness can be tested. Opening an issue before implementing a large
change reduces the risk of work that falls outside the package boundary.

## Security and private reports

Do not publish suspected security vulnerabilities or sensitive information in a
public issue. Send a private report to `meam64@gmail.com` with the affected
version, impact, reproduction steps, and any suggested mitigation.

## Response expectations

There is no guaranteed response time or service-level agreement. Clear,
reproducible reports within the documented project scope are the easiest to
review. Requests for custom analysis, private consulting, or support for
undocumented internal APIs may not be accepted.
