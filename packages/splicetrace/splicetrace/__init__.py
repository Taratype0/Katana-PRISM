"""SpliceTrace 1.0.2 — High-speed bulk AS RBP discovery, mature-mRNA motif, Module-RBP bridge & GTF/UniProt ORF/NMD engine."""
from splicetrace.continuous_rescue import SpliceTraceRescueResult, run_bulk_continuous_rescue
from splicetrace.extensible_db import (
    export_katana_motif_pack,
    iupac_to_pwm,
    load_external_motif_entries,
    merge_motif_entries_into_library,
)
from splicetrace.fast_accel import HAS_NUMBA, configure_server_threads, fast_score_mean_log_odds
from splicetrace.library import (
    CuratedMotifLibrary,
    load_bundled_human_rbp_library,
    load_curated_human_rbp_library,
)
from splicetrace.matching import match_events
from splicetrace.module_bridge import bridge_splicenet_modules_to_rbps
from splicetrace.protein_consequences import (
    load_bundled_uniprot_domains,
    load_gtf_coding_transcripts,
    trace_protein_and_nmd_consequences,
)
from splicetrace.rmats_io import load_rmats_measurable_events
from splicetrace.rna_forms import IndexedFasta, extract_event_rna, trace_motif_instances
from splicetrace.scanner import run_splicetrace_scan

__all__ = [
    "__version__",
    "HAS_NUMBA",
    "configure_server_threads",
    "fast_score_mean_log_odds",
    "CuratedMotifLibrary",
    "SpliceTraceRescueResult",
    "IndexedFasta",
    "load_bundled_human_rbp_library",
    "load_curated_human_rbp_library",
    "load_external_motif_entries",
    "merge_motif_entries_into_library",
    "export_katana_motif_pack",
    "iupac_to_pwm",
    "load_rmats_measurable_events",
    "match_events",
    "extract_event_rna",
    "trace_motif_instances",
    "load_bundled_uniprot_domains",
    "load_gtf_coding_transcripts",
    "trace_protein_and_nmd_consequences",
    "bridge_splicenet_modules_to_rbps",
    "run_splicetrace_scan",
    "run_bulk_continuous_rescue",
]

try:
    from splicetrace.protein_viz import (
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

__version__ = "1.0.2"
