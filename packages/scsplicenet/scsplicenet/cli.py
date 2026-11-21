"""Standalone CLI for scsplicenet (1.0.1) — Smart-seq3 support-precision co-splicing network & modules."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from scsplicenet.evaluation import enrich_sc_module_genes
from scsplicenet.network import fit_scsplicenet
from scsplicenet.qc_and_selection import (
    attach_split_state_reliability_and_psix,
    audit_sc_events,
    select_sc_events,
)

__version__ = "1.0.1"
def write_nfcore_versions_yml(path: Path | str, process_name: str = "KATANA_SCSPLICENET") -> None:
    """Emit standard nf-core `versions.yml` recording tool and Python versions."""
    import platform
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    yml = (
        f'"{process_name}":\n'
        f'    scsplicenet: "1.0.1"\n'
        f'    python: "{platform.python_version()}"\n'
    )
    p.write_text(yml, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scsplicenet",
        description="scSpliceNet 1.0.1: Smart-seq3 Hybrid-Manifold ExcessBioVar & support-precision TOM co-splicing network.",
    )
    p.add_argument("--version", action="version", version=f"scsplicenet {__version__}")
    p.add_argument("--usage", type=Path, required=False, help="TSV(.gz): cell_id rows x event_id columns observed usage")
    p.add_argument("--metadata", type=Path, required=False, help="TSV: QC-passed cell metadata (cell_id, donor_id, cell_type, condition, qc_pass)")
    p.add_argument("--event-genes", type=Path, required=False, help="TSV: event_id, gene_id[, gene_symbol]")
    p.add_argument("--support", type=Path, default=None, help="Optional TSV(.gz): cell_id x event_id count denominators")
    p.add_argument("--host-expr", type=Path, default=None, help="Optional TSV(.gz): cell_id x gene_id log-expression for Hybrid Manifold selector")
    p.add_argument("--select-budget", type=int, default=0, help="If > 0, select top-K splicing events first using --selector")
    p.add_argument(
        "--selector",
        choices=["hybrid_manifold_excess_bio_var", "reliability_excess_bio_var", "excess_bio_var"],
        default="hybrid_manifold_excess_bio_var",
    )
    p.add_argument("--level", default="cell_state_distribution", choices=["between_donor", "cell_state_distribution", "within_state_cell"])
    p.add_argument("--top-fraction", type=float, default=0.05)
    p.add_argument("--min-module-size", type=int, default=10)
    p.add_argument("--repeats", type=int, default=8)
    p.add_argument("--threads", type=int, default=0, help="CPU thread hint for BLAS matrix kernels")
    p.add_argument("--seed", type=int, default=20261117)
    p.add_argument("--gmt", type=Path, default=None, help="Optional GMT pathway file for module enrichment")
    p.add_argument("-o", "--out", "--output-dir", dest="out", type=Path, required=False, help="Output directory")
    p.add_argument("--emit-versions-yml", type=Path, default=None)
    p.add_argument("--process-name", default="KATANA_SCSPLICENET")
    args = p.parse_args(argv)

    if args.emit_versions_yml is not None and args.out is None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
        return 0
    if args.usage is None or args.metadata is None or args.event_genes is None or args.out is None:
        p.error("--usage, --metadata, --event-genes, and --out are required.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    usage = pd.read_csv(args.usage, sep="\t", index_col=0)
    metadata = pd.read_csv(args.metadata, sep="\t")
    eg_df = pd.read_csv(args.event_genes, sep="\t")
    support = pd.read_csv(args.support, sep="\t", index_col=0) if args.support else None
    eg_ser = eg_df.set_index("event_id")["gene_id"].astype(str)

    if args.select_budget > 0 and usage.shape[1] > args.select_budget:
        from scsplicenet.input_contract import validate_sc_usage_input
        u_val, md_val, s_val, eg_val, _ = validate_sc_usage_input(
            usage, metadata, eg_df, support=support, require_qc_whitelist=("qc_pass" in metadata.columns)
        )
        host_df = pd.read_csv(args.host_expr, sep="\t", index_col=0) if args.host_expr else None
        eg_ser_val = eg_val.set_index("event_id")["gene_id"].astype(str)
        aud = audit_sc_events(u_val, eg_ser_val, support=s_val)
        aud = attach_split_state_reliability_and_psix(aud, u_val, md_val, support=s_val, host_log_expr=host_df, seed=args.seed)
        sel = select_sc_events(aud, budget=args.select_budget, max_per_gene=2, ranking_metric=args.selector)
        keep_ids = sel["event_id"].astype(str).tolist()
        if keep_ids:
            usage = u_val.loc[:, keep_ids]
            support = s_val.loc[:, keep_ids] if s_val is not None else None
            metadata = md_val
            eg_df = eg_val[eg_val["event_id"].astype(str).isin(set(keep_ids))].copy()
            sel.to_csv(out / "scsplicenet_selected_events.tsv", sep="\t", index=False)

    net = fit_scsplicenet(
        usage=usage, metadata=metadata, event_genes=eg_df, support=support,
        level=args.level, top_fraction=args.top_fraction, repeats=args.repeats,
        random_seed=args.seed, min_module_size=args.min_module_size, use_r_dynamic_cut=False,
    )
    net.edges.to_csv(out / "scsplicenet_edges.tsv", sep="\t", index=False)
    net.modules.to_csv(out / "scsplicenet_modules.tsv", sep="\t", index=False)
    net.event_audit.to_csv(out / "scsplicenet_event_audit.tsv", sep="\t", index=False)

    if args.gmt is not None:
        sym_map = eg_df.set_index("event_id")["gene_symbol"] if "gene_symbol" in eg_df.columns else None
        enr = enrich_sc_module_genes(net.modules, Path(args.gmt), gene_symbol_map=sym_map)
        enr.to_csv(out / "scsplicenet_module_enrichment.tsv", sep="\t", index=False)

    diag = {"version": __version__, **net.diagnostics}
    (out / "scsplicenet_summary.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    if args.emit_versions_yml is not None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
    print(json.dumps({
        "version": __version__,
        "status": diag["status"],
        "selected_edges": diag["selected_edges"],
        "discovered_modules": diag["discovered_modules"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
