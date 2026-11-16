# NIST StRD reference datasets

Vendored copies of datasets from the NIST/ITL Statistical Reference Datasets (StRD), used to
grade the CurveStudio fitting engine against certified results (see
`packages/core/src/curvestudio/testing/nist.py` and `tests/numerical/`).

- **Retrieved:** 16/11/2026 (IST) with `scripts/fetch_nist_strd.py`.
- **Source URLs:**
  - Nonlinear: `https://www.itl.nist.gov/div898/strd/nls/data/LINKS/DATA/<Name>.dat`
  - Linear: `https://www.itl.nist.gov/div898/strd/lls/data/LINKS/DATA/<Name>.dat`
  - Index pages: <https://www.itl.nist.gov/div898/strd/nls/nls_main.shtml> and
    <https://www.itl.nist.gov/div898/strd/lls/lls.shtml>
- **Licence:** NIST StRD data is a work of the US Federal Government and is in the public domain
  (17 U.S.C. 105). It is redistributed here unmodified; the rest of this repository is
  AGPL-3.0-or-later.
- **Do not edit these files.** They are the ground truth the engine is graded against; the
  fetch script writes the downloaded bytes untouched and the parser reads the original layout.
  Never regenerate or hand-correct a value. The only permissible difference from the NIST
  download is line-ending normalisation (CRLF to LF) applied by git on commit; the parser
  accepts both.

## Contents

| Directory | Datasets |
|-----------|----------|
| `nonlinear/` | Misra1a, Misra1b, Misra1c, Misra1d, DanWood, Chwirut2, Lanczos3, Gauss1, BoxBOD, Rat42, MGH09, Eckerle4 |
| `linear/` | Norris, Pontius, Wampler1, Wampler4, Filip |

`Norris.dat` and `Wampler1.dat` were fetched earlier (Task 2.3); the fetch script skips files
that already exist, so re-running it never changes committed data (`--force` replaces them).

To re-fetch: `uv run python scripts/fetch_nist_strd.py [--force]` (network access required;
NIST rejects the default Python user agent, so the script sends its own).
