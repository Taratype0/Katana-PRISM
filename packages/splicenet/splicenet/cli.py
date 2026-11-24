"""Standalone CLI for splicenet (1.0.2) — pure Python out-of-the-box, nf-core compatible."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import pandas as pd

from splicenet.evaluation import enrich_module_genes, score_modules
from splicenet.network import consensus_core_audit, fit_splicenet
from splicenet.qc_and_selection import select_events, unit_robust_audit

__version__ = "1.0.2"
def write_nfcore_versions_yml(path: Path | str, process_name: str = "KATANA_SPLICENET") -> None:
    """Emit standard nf-core `versions.yml` recording tool and Python versions."""
    import platform
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    yml = (
        f'"{process_name}":\n'
        f'    splicenet: "1.0.2"\n'
        f'    python: "{platform.python_version()}"\n'
    )
    p.write_text(yml, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="splicenet",
        description="SpliceNet 1.0.2: Excess-BioVar debiased signed TOM co-splicing network & module discovery.",
    )
    parser.add_argument("--version", action="version", version=f"splicenet {__version__}")
    parser.add_argument("--bundle", type=Path, default=None, help="Katana upstream bundle directory (auto-extracts PSI & support matrices)")
    parser.add_argument("--rmats-dir", type=Path, default=None, help="External rMATS directory (auto-extracts PSI & support matrices)")
    parser.add_argument("--rmats-mode", choices=["JCEC", "JC"], default="JCEC", help="rMATS count mode when using --bundle or --rmats-dir")
    parser.add_argument("--usage", type=Path, required=False, help="Relative usage / PSI matrix TSV (rows=events, cols=samples)")
    parser.add_argument("--samples", type=Path, required=False, help="Sample metadata TSV/CSV (sample_id, biological_sample_id)")
    parser.add_argument("--support", type=Path, default=None, help="Optional total read-support matrix TSV")
    parser.add_argument("--annotation", type=Path, default=None, help="Optional event-to-gene TSV (event_key, gene_id)")
    parser.add_argument("--gmt", type=Path, default=None, help="Optional MSigDB/GO .gmt pathway file")
    parser.add_argument("-o", "--output-dir", "--out", "--outdir", dest="output_dir", type=Path, required=False, help="Output directory")
    parser.add_argument("--threads", type=int, default=0, help="CPU thread hint for BLAS operations")
    parser.add_argument("--feature-cap", type=int, default=5000)
    parser.add_argument("--min-module-size", type=int, default=20)
    parser.add_argument("--merge-cut-height", type=float, default=0.25, help="Module eigengene merge cut height")
    parser.add_argument("--ranking-metric", choices=["excess_bio_variance", "variance"], default="excess_bio_variance")
    parser.add_argument("--power", type=int, default=0, help="Signed TOM soft-threshold power (0 = auto scale-free pick_soft_threshold)")
    parser.add_argument("--backend", choices=["python", "r-wgcna"], default="python")
    parser.add_argument("--subsamples", type=int, default=0)
    parser.add_argument("--emit-versions-yml", type=Path, default=None, help="Write nf-core compatible versions.yml")
    parser.add_argument("--process-name", default="SPLICENET")
    args = parser.parse_args(argv)

    if args.emit_versions_yml is not None and args.output_dir is None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
        return 0
    if args.output_dir is None or (args.bundle is None and args.rmats_dir is None and (args.usage is None or args.samples is None)):
        parser.error("Provide (--usage and --samples), or --bundle, or --rmats-dir, together with --output-dir.")

    t0 = time.monotonic()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    auto_gene_map = None
    if args.bundle is not None or args.rmats_dir is not None:
        from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats
        usage, meta, support, auto_gene_map = extract_matrices_from_bundle_or_rmats(
            bundle_dir=args.bundle, rmats_dir=args.rmats_dir, samples_path=args.samples, rmats_mode=args.rmats_mode
        )
    else:
        usage = pd.read_csv(args.usage, sep="\t", index_col=0)
        _s_sep = "," if str(args.samples).lower().endswith((".csv", ".csv.gz")) else "\t"
        meta = pd.read_csv(args.samples, sep=_s_sep, dtype={"sample_id": str, "biological_sample_id": str})
        support = pd.read_csv(args.support, sep="\t", index_col=0) if args.support is not None else None

    masked, audit = unit_robust_audit(usage, meta, support)
    selected, sel_audit = select_events(masked, audit, feature_cap=args.feature_cap, ranking_metric=args.ranking_metric)
    sel_audit.to_csv(out_dir / "event_selection_audit.tsv.gz", sep="\t", index_label="event_key")

    if len(selected) < 4:
        assignments = pd.DataFrame(columns=["event_key", "module", "block", "kME"])
        assignments.to_csv(out_dir / "module_assignments.tsv", sep="\t", index=False)
        pd.DataFrame(index=meta["sample_id"]).to_csv(out_dir / "module_scores.tsv", sep="\t", index_label="sample_id")
        pd.DataFrame(columns=["module", "events", "pc1_variance_fraction", "evaluable_samples"]).to_csv(out_dir / "module_summary.tsv", sep="\t", index=False)
        pd.DataFrame(columns=["event_key", "module", "center", "scale", "loading"]).to_csv(out_dir / "module_pc1_basis.tsv", sep="\t", index=False)
        receipt = {
            "version": __version__,
            "status": "insufficient_events_for_network",
            "backend": args.backend,
            "input_events": int(len(usage)),
            "qc_eligible_events": int(audit["eligible"].sum()) if not audit.empty else 0,
            "selected_events": int(len(selected)),
            "non_grey_modules": 0,
            "assigned_events": 0,
            "elapsed_seconds": round(time.monotonic() - t0, 2),
        }
        (out_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        if args.emit_versions_yml is not None:
            write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
        print(json.dumps(receipt))
        return 0

    eff_min_mod = min(args.min_module_size, max(4, len(selected) // 3))
    assignments = fit_splicenet(
        selected,
        out_dir / "wgcna_fit",
        power=args.power,
        min_module_size=eff_min_mod,
        merge_cut_height=args.merge_cut_height,
        backend=args.backend,
    )
    stability_metrics: dict[str, float] = {}
    if args.subsamples > 0:
        assignments, stability_metrics = consensus_core_audit(
            usage, meta, support, assignments,
            feature_cap=args.feature_cap, ranking_metric=args.ranking_metric,
            power=args.power,
            merge_cut_height=args.merge_cut_height,
            min_module_size=eff_min_mod,
            n_subsamples=args.subsamples,
            backend=args.backend,
        )

    assignments.to_csv(out_dir / "module_assignments.tsv", sep="\t", index=False)
    scores, _, mod_summary, basis = score_modules(selected, assignments)
    scores.to_csv(out_dir / "module_scores.tsv", sep="\t", index_label="sample_id")
    if mod_summary.empty:
        mod_summary = pd.DataFrame(columns=["module", "events", "pc1_variance_fraction", "evaluable_samples"])
    if basis.empty:
        basis = pd.DataFrame(columns=["event_key", "module", "center", "scale", "loading"])
    mod_summary.to_csv(out_dir / "module_summary.tsv", sep="\t", index=False)
    basis.to_csv(out_dir / "module_pc1_basis.tsv", sep="\t", index=False)

    gene_map = None
    if args.annotation is not None:
        ann_df = pd.read_csv(args.annotation, sep="\t", dtype=str).drop_duplicates(subset=["event_key"]).set_index("event_key")
        gene_map = ann_df["gene_id"]
    elif auto_gene_map is not None and not auto_gene_map.empty:
        gene_map = auto_gene_map

    if gene_map is not None:
        mod_genes = assignments[assignments["module"].ne(0)].copy()
        mod_genes["gene_id"] = mod_genes["event_key"].map(gene_map)
        mod_genes.to_csv(out_dir / "module_genes.tsv", sep="\t", index=False)
        if args.gmt is not None:
            enr = enrich_module_genes(assignments, gene_map, args.gmt)
            enr.to_csv(out_dir / "module_pathway_enrichment.tsv", sep="\t", index=False)

    receipt = {
        "version": __version__,
        "backend": args.backend,
        "input_events": int(len(usage)),
        "qc_eligible_events": int(audit["eligible"].sum()),
        "selected_events": int(len(selected)),
        "non_grey_modules": int(assignments.loc[assignments["module"].ne(0), "module"].nunique()),
        "assigned_events": int(assignments["module"].ne(0).sum()),
        "stability": stability_metrics,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    (out_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.emit_versions_yml is not None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
