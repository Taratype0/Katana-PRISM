"""SpliceManifold 1.0.1 — PS-DAS Qualification, Empirical-Bayes PSI Clustermap, PERMANOVA PCA & Precision-Kernel Event-PSI UMAP."""
from splicemanifold.manifold import (
    compute_dual_expression_vs_psi_umap,
    compute_splicing_manifold,
    permanova_r2,
    render_dual_psi_umap,
    render_psi_clustermap,
    render_psi_pca,
)
from splicemanifold.ps_das import (
    PS_DAS_POLICY_ID,
    load_rmats_study,
    qualify_ps_das_events,
)

__version__ = "1.0.1"

__all__ = [
    "__version__",
    "PS_DAS_POLICY_ID",
    "permanova_r2",
    "compute_splicing_manifold",
    "compute_dual_expression_vs_psi_umap",
    "render_psi_clustermap",
    "render_psi_pca",
    "render_dual_psi_umap",
    "qualify_ps_das_events",
    "load_rmats_study",
]
