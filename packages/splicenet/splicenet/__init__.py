"""SpliceNet 1.0.4 — Pure-Python Excess-BioVar Signed TOM Co-Splicing Network."""
from splicenet.evaluation import enrich_module_genes, orthogonalize_host_depth, score_modules
from splicenet.network import (
    compute_signed_tom_from_bicor,
    compute_tukey_bicor_matrix,
    consensus_core_audit,
    fit_splicenet,
    pick_soft_threshold,
)
from splicenet.qc_and_selection import select_events, unit_robust_audit

__version__ = "1.0.4"

__all__ = [
    "__version__",
    "unit_robust_audit",
    "select_events",
    "compute_tukey_bicor_matrix",
    "compute_signed_tom_from_bicor",
    "fit_splicenet",
    "pick_soft_threshold",
    "consensus_core_audit",
    "score_modules",
    "orthogonalize_host_depth",
    "enrich_module_genes",
]
