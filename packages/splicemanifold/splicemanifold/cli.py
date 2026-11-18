"""Standalone CLI for splicemanifold (1.0.0) — Bulk PS-DAS + Manifold & Single-Cell Precision Event-PSI UMAP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
import pandas as pd

from splicemanifold.manifold import (
    compute_dual_expression_vs_psi_umap,
    compute_splicing_manifold,
    render_dual_psi_umap,
    render_psi_clustermap,
    render_psi_pca,
)
from splicemanifold.ps_das import PS_DAS_POLICY_ID, load_rmats_study, qualify_ps_das_events

__version__ = "1.0.0"


def write_nfcore_versions_yml(path: Path | str, process_name: str = "SPLICEMANIFOLD") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f'"{process_name}":\n    splicemanifold: "{__version__}"\n    python: "{platform.python_version()}"\n',
        encoding="utf-8",
    )


def _run_bulk(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    if args.psi_matrix is not None and args.samples is not None:
        psi_mat = pd.read_csv(args.psi_matrix, sep="\t", index_col=0)
        sup_mat = pd.read_csv(args.support_matrix, sep="\t", index_col=0) if args.support_matrix else None
        s_df = pd.read_csv(args.samples, sep="\t" if str(args.samples).endswith(".tsv") else ",")
        sample_meta = [
            {
                "sample_id": str(r.get("sample_id") or r.get("sample")),
                "biological_sample_id": str(r.get("biological_sample_id") or r.get("sample_id") or r.get("sample")),
                "condition": str(r.get("condition") or r.get("group") or "Group"),
            }
            for _, r in s_df.iterrows()
        ]
        event_meta = {}
        qualified = pd.DataFrame()
    else:
        events_raw, psi_mat, sup_mat, sample_meta, event_meta = load_rmats_study(
            bundle_dir=args.bundle,
            rmats_dir=args.rmats_dir,
            samples_file=args.samples,
            contrasts_file=args.contrasts,
            rmats_mode=args.rmats_mode,
        )
        qualified = qualify_ps_das_events(
            events_raw,
            fdr_cutoff=args.fdr_cutoff,
            dpsi_cutoff=args.dpsi_cutoff,
            min_effective_reads=args.min_effective_reads,
            max_jeffreys_se=args.max_jeffreys_se,
        )
        qualified.to_csv(out / "ps_das_events.tsv", sep="\t", index=False)
        psi_mat.to_csv(out / "psi_matrix.tsv", sep="\t")
        sup_mat.to_csv(out / "support_matrix.tsv", sep="\t")
        pd.DataFrame(sample_meta).to_csv(out / "samples_metadata.tsv", sep="\t", index=False)
        ann_rows = [{"event_key": eid, "gene_id": m["gene_id"], "gene_symbol": m["gene_symbol"], "event_type": m["event_type"]} for eid, m in event_meta.items()]
        pd.DataFrame(ann_rows).to_csv(out / "event_annotation.tsv", sep="\t", index=False)

    sup_ids = None
    if args.supervised and not qualified.empty:
        sup_ids = set(qualified.loc[qualified["passes_ps_das"], "event_id"].astype(str))

    landscape = compute_splicing_manifold(
        psi_mat,
        sample_meta,
        depth_matrix=sup_mat,
        event_meta=event_meta,
        supervised_event_ids=sup_ids,
        top_n=args.top_n,
        manifold_top_n=args.manifold_top_n,
        prior_kappa=args.prior_kappa,
    )
    if landscape.get("available"):
        fig_dir = out / "figures"
        render_psi_clustermap(landscape, fig_dir / "psi_clustermap.png", fig_dir / "psi_clustermap.svg", event_meta=event_meta)
        render_psi_pca(landscape, fig_dir / "psi_pca.png", fig_dir / "psi_pca.svg")
        pd.DataFrame(landscape["samples"]).to_csv(out / "sample_manifold_coordinates.tsv", sep="\t", index=False)

    receipt = {
        "version": __version__,
        "mode": "bulk",
        "policy": PS_DAS_POLICY_ID,
        "total_events_evaluated": int(len(qualified)),
        "ordinary_das_events": int(qualified["passes_ordinary_das"].sum()) if not qualified.empty else 0,
        "ps_das_events": int(qualified["passes_ps_das"].sum()) if not qualified.empty else 0,
        "nmd_asymmetry_rescued": int((qualified["passes_ps_das"] & qualified["nmd_asymmetry_rescued"]).sum()) if not qualified.empty else 0,
        "manifold_available": bool(landscape.get("available", False)),
        "permanova_r2": landscape.get("permanova_r2"),
        "permanova_p": landscape.get("permanova_p"),
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.emit_versions_yml:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
    print(json.dumps(receipt))
    return 0


def _run_single_cell(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    psi_df = pd.read_csv(args.psi, sep="\t", index_col=0)
    sup_df = pd.read_csv(args.support, sep="\t", index_col=0) if args.support else None
    if args.cells_as_rows:
        psi_df = psi_df.T
        if sup_df is not None:
            sup_df = sup_df.T

    md_df = pd.read_csv(args.metadata, sep="\t" if str(args.metadata).endswith(".tsv") else ",")
    sample_meta = []
    for _, r in md_df.iterrows():
        cid = str(r.get("cell_id") or r.get("sample_id"))
        sample_meta.append({
            "sample_id": cid,
            "condition": str(r.get("condition") or r.get("cell_type") or "State"),
            "cell_type": str(r.get("cell_type") or r.get("condition") or "Cell"),
            "donor_id": str(r.get("donor_id") or r.get("donor") or ""),
        })

    landscape = compute_splicing_manifold(
        psi_df,
        sample_meta,
        depth_matrix=sup_df,
        top_n=args.top_n,
        manifold_top_n=args.manifold_top_n,
        min_present_fraction=args.min_present_fraction,
        prior_kappa=args.prior_kappa,
        run_umap=True,
    )
    if not landscape.get("available"):
        raise RuntimeError(landscape.get("reason", "Manifold unavailable"))

    fig_dir = out / "figures"
    render_psi_clustermap(landscape, fig_dir / "sc_psi_clustermap.png", fig_dir / "sc_psi_clustermap.svg")
    render_psi_pca(landscape, fig_dir / "sc_psi_pca.png", fig_dir / "sc_psi_pca.svg")

    receipt: dict = {
        "version": __version__,
        "mode": "single-cell",
        "cells": len(landscape["samples"]),
        "manifold_events": landscape["manifold_events"],
        "permanova_r2": landscape["permanova_r2"],
        "permanova_p": landscape["permanova_p"],
    }

    if args.expression is not None:
        expr_df = pd.read_csv(args.expression, sep="\t", index_col=0)
        if not args.cells_as_rows:
            expr_df = expr_df.T
        dual = compute_dual_expression_vs_psi_umap(expr_df, landscape, k_neighbors=args.k_neighbors)
        render_dual_psi_umap(dual, fig_dir / "sc_dual_psi_umap.png", fig_dir / "sc_dual_psi_umap.svg")
        pd.DataFrame(dual["cells"]).to_csv(out / "sc_cell_manifold_coordinates.tsv", sep="\t", index=False)
        receipt.update({
            "mean_modality_divergence": dual["mean_divergence"],
            "median_donor_lisi": dual["median_donor_lisi"],
            "cell_type_silhouette_psi": dual["cell_type_silhouette_psi"],
            "cell_type_silhouette_expr": dual["cell_type_silhouette_expr"],
        })
    else:
        pd.DataFrame(landscape["samples"]).to_csv(out / "sc_cell_manifold_coordinates.tsv", sep="\t", index=False)

    receipt["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    (out / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.emit_versions_yml:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
    print(json.dumps(receipt))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="splicemanifold",
        description="SpliceManifold 1.0.0: PS-DAS qualification, Empirical-Bayes PSI Clustermap, PERMANOVA PCA & Precision-Kernel Event-PSI UMAP.",
    )
    p.add_argument("--version", action="version", version=f"splicemanifold {__version__}")
    p.add_argument("--emit-versions-yml", type=Path, default=None)
    p.add_argument("--process-name", default="SPLICEMANIFOLD")
    sub = p.add_subparsers(dest="subcommand")

    pb = sub.add_parser("bulk", help="Bulk RNA-seq PS-DAS qualification + Beyond-Jutils PSI Clustermap/PCA/UMAP")
    pb.add_argument("--bundle", type=Path, default=None, help="Katana upstream bundle directory")
    pb.add_argument("--rmats-dir", type=Path, default=None, help="External rMATS output directory")
    pb.add_argument("--psi-matrix", type=Path, default=None, help="Precomputed PSI matrix TSV (events x samples)")
    pb.add_argument("--support-matrix", type=Path, default=None, help="Optional junction depth matrix TSV")
    pb.add_argument("--samples", type=Path, default=None, help="Samplesheet CSV/TSV (sample_id, condition)")
    pb.add_argument("--contrasts", type=Path, default=None, help="Optional contrastsheet CSV/TSV")
    pb.add_argument("--rmats-mode", choices=["JCEC", "JC"], default="JCEC")
    pb.add_argument("--fdr-cutoff", type=float, default=0.05)
    pb.add_argument("--dpsi-cutoff", type=float, default=0.10)
    pb.add_argument("--min-effective-reads", type=float, default=10.0)
    pb.add_argument("--max-jeffreys-se", type=float, default=0.16)
    pb.add_argument("--top-n", type=int, default=150)
    pb.add_argument("--manifold-top-n", type=int, default=2000)
    pb.add_argument("--prior-kappa", type=float, default=10.0)
    pb.add_argument("--supervised", action="store_true", help="Restrict manifold to PS-DAS signature events")
    pb.add_argument("-o", "--outdir", "--out", dest="outdir", type=Path, required=True)
    pb.add_argument("--emit-versions-yml", type=Path, default=None)
    pb.add_argument("--process-name", default="SPLICEMANIFOLD")

    psc = sub.add_parser("single-cell", help="Smart-seq3 Precision-Kernel Event-PSI UMAP & Modality Divergence")
    psc.add_argument("--psi", type=Path, required=True, help="Single-cell PSI matrix TSV(.gz)")
    psc.add_argument("--support", type=Path, default=None, help="Optional junction read-support matrix TSV(.gz)")
    psc.add_argument("--metadata", type=Path, required=True, help="Cell metadata TSV/CSV (cell_id, cell_type, condition, donor_id)")
    psc.add_argument("--expression", type=Path, default=None, help="Optional host log-expression matrix TSV(.gz) for Procrustes dual UMAP & Delta_kNN")
    psc.add_argument("--cells-as-rows", action="store_true", help="Set if input matrices have cells as rows and events/genes as columns")
    psc.add_argument("--top-n", type=int, default=150)
    psc.add_argument("--manifold-top-n", type=int, default=2000)
    psc.add_argument("--min-present-fraction", type=float, default=0.25)
    psc.add_argument("--prior-kappa", type=float, default=10.0)
    psc.add_argument("--k-neighbors", type=int, default=12)
    psc.add_argument("-o", "--outdir", "--out", dest="outdir", type=Path, required=True)
    psc.add_argument("--emit-versions-yml", type=Path, default=None)
    psc.add_argument("--process-name", default="SPLICEMANIFOLD_SC")

    args = p.parse_args(argv)
    if args.subcommand is None:
        if args.emit_versions_yml is not None:
            write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
            return 0
        p.print_help()
        return 0
    if args.subcommand == "bulk":
        return _run_bulk(args)
    return _run_single_cell(args)


if __name__ == "__main__":
    raise SystemExit(main())
