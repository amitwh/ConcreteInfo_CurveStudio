"""CurveStudio reports: CSV, JSON, XLSX and PDF renderings of one :class:`FitBundle`.

Every format states what produced it (tool, versions, IST timestamp), carries the
statistical conventions and assumptions, marks undefined statistics as such
(empty or "not defined", never 0 or NaN), flags a fit that did not converge
prominently, and neutralises text that a spreadsheet would execute as a formula.
"""

from curvestudio_reporting.csv_report import (
    observations_csv,
    parameters_csv,
    safe_text,
    statistics_csv,
)
from curvestudio_reporting.json_report import fit_to_json
from curvestudio_reporting.pdf_report import fit_to_pdf
from curvestudio_reporting.xlsx_report import fit_to_xlsx

__all__ = [
    "fit_to_json",
    "fit_to_pdf",
    "fit_to_xlsx",
    "observations_csv",
    "parameters_csv",
    "safe_text",
    "statistics_csv",
]
