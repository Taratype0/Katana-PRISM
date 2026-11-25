"""scSpliceTrace 1.0.5 — Smart-seq3 RBP discovery, mature-mRNA motif, Module-RBP bridge & GTF/UniProt ORF/NMD engine."""
from scsplicetrace.contrast_discovery import (
    SCSpliceTraceContrastResult,
    compute_sc_splicing_contrast,
    compute_sc_stratified_meta_contrast,
    discover_sc_rbp_programs,
)
from scsplicetrace.extensible_db import (
    export_katana_motif_pack,
    iupac_to_pwm,
    load_external_motif_entries,
    merge_motif_entries_into_library,
)
from scsplicetrace.fast_accel import HAS_NUMBA, configure_server_threads, fast_score_mean_log_odds
from scsplicetrace.library import CuratedMotifLibrary, load_bundled_human_rbp_library
from scsplicetrace.module_bridge import bridge_splicenet_modules_to_rbps
from scsplicetrace.protein_consequences import (
    load_bundled_uniprot_domains,
    load_gtf_coding_transcripts,
    trace_protein_and_nmd_consequences,
)
from scsplicetrace.rna_forms import IndexedFasta, extract_event_rna
from scsplicetrace.sequence_and_consequence import (
    build_sc_directional_motif_tensor,
    extract_sc_event_sequences,
    load_default_curated_library,
    trace_sc_mature_motif_consequences,
)
from scsplicetrace.target_and_psi_models import (
    estimate_sc_psi_eb_baseline,
    gene_equal_psi_mse,
    predict_sc_psi_precision,
    rank_rbp_target_events,
)

__all__ = [
    "__version__",
    "HAS_NUMBA",
    "configure_server_threads",
    "fast_score_mean_log_odds",
    "IndexedFasta",
    "CuratedMotifLibrary",
    "SCSpliceTraceContrastResult",
    "load_bundled_human_rbp_library",
    "load_default_curated_library",
    "load_external_motif_entries",
    "merge_motif_entries_into_library",
    "export_katana_motif_pack",
    "iupac_to_pwm",
    "extract_event_rna",
    "extract_sc_event_sequences",
    "build_sc_directional_motif_tensor",
    "trace_sc_mature_motif_consequences",
    "compute_sc_splicing_contrast",
    "compute_sc_stratified_meta_contrast",
    "discover_sc_rbp_programs",
    "rank_rbp_target_events",
    "estimate_sc_psi_eb_baseline",
    "predict_sc_psi_precision",
    "gene_equal_psi_mse",
    "load_bundled_uniprot_domains",
    "load_gtf_coding_transcripts",
    "trace_protein_and_nmd_consequences",
    "bridge_splicenet_modules_to_rbps",
]

try:
    from scsplicetrace.protein_viz import (
        classify_proteome_fate,
        plot_causal_alluvial_flow,
        plot_consequence_landscape,
        plot_dual_track_nmd_architecture,
        plot_lollipop_protein_splicing_map,
        plot_side_by_side_inframe_domains,
    )
    __all__.extend([
        "classify_proteome_fate",
        "plot_consequence_landscape",
        "plot_dual_track_nmd_architecture",
        "plot_side_by_side_inframe_domains",
        "plot_lollipop_protein_splicing_map",
        "plot_causal_alluvial_flow",
    ])
except ImportError:
    pass

__version__ = "1.0.5"
