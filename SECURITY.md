# Security Policy

## Supported versions

CurveStudio is pre-alpha. Only the latest commit on `main` receives security fixes.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report privately through GitHub Security Advisories:
<https://github.com/amitwh/ConcreteInfo_CurveStudio/security/advisories/new>

Include the affected component, steps to reproduce, and the impact you expect. You will get an
acknowledgement as soon as the maintainer can respond; fixes are coordinated with you before any
public disclosure.

## Scope notes

- CurveStudio never evaluates user-supplied equations with a raw `eval`/`exec` of the user's
  text: model expressions pass an AST whitelist (a fixed set of arithmetic nodes and functions,
  no attribute access, subscripts, comprehensions or names beyond the model's own variables and
  parameters) and are then parsed by SymPy's `parse_expr` against a closed namespace of
  whitelisted functions and symbols. Fitting code never unpickles user-supplied content either.
  Any path that reaches `eval`/`exec`/`pickle` with user text is a vulnerability; please report it.
- Numerical-correctness bugs (a wrong fit reported as successful) are tracked as normal issues
  using the bug report template, not as security reports.
