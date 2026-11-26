"""Katana Studio & Katana Report (1.0.6) — Executive Splicing Synthesis & Interactive Workbench."""
from katana_studio.report_builder import open_workspace, run_bulk, run_report_from_nextflow, run_single_cell

__version__ = "1.0.6"

__all__ = [
    "__version__",
    "run_bulk",
    "run_single_cell",
    "run_report_from_nextflow",
    "open_workspace",
]
