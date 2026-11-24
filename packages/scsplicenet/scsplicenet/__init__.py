"""scSpliceNet 1.0.2 (Smart-seq3 / HT Smart-seq3 / Smart-seq3xpress)."""
from scsplicenet.evaluation import (
    enrich_sc_module_genes,
    evaluate_heldout_edges_and_modules,
    evaluate_module_reproducibility,
    gene_equal_usage_mse,
    orthogonalize_sc_host_expression,
    predict_sc_usage_inductive,
)
from scsplicenet.input_contract import ValidatedSCUsageInput, validate_sc_usage_input
from scsplicenet.network import (
    SCSpliceNetwork,
    compute_signed_tom,
    discover_sc_modules,
    empirical_bayes_fisher_shrink,
    fit_scsplicenet,
    precision_pairwise_correlation,
)
from scsplicenet.qc_and_selection import (
    attach_split_state_reliability_and_psix,
    audit_sc_events,
    estimate_sc_observation_weights,
    select_sc_events,
)

__version__ = "1.0.2"

__all__ = [
    "__version__",
    "ValidatedSCUsageInput",
    "validate_sc_usage_input",
    "audit_sc_events",
    "estimate_sc_observation_weights",
    "select_sc_events",
    "attach_split_state_reliability_and_psix",
    "SCSpliceNetwork",
    "precision_pairwise_correlation",
    "empirical_bayes_fisher_shrink",
    "compute_signed_tom",
    "discover_sc_modules",
    "fit_scsplicenet",
    "orthogonalize_sc_host_expression",
    "evaluate_heldout_edges_and_modules",
    "evaluate_module_reproducibility",
    "enrich_sc_module_genes",
    "gene_equal_usage_mse",
    "predict_sc_usage_inductive",
]
