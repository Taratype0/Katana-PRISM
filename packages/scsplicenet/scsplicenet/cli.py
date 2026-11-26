"""Standalone CLI for scsplicenet (1.0.7) — Smart-seq3 support-precision co-splicing network & modules."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pandas as pd

from scsplicenet.evaluation import enrich_sc_module_genes
from scsplicenet.network import fit_scsplicenet
from scsplicenet.qc_and_selection import (
    attach_split_state_reliability_and_psix,
    audit_sc_events,
    select_sc_events,
)

__version__ = "1.0.7"
def write_nfcore_versions_yml(path: Path | str, process_name: str = "KATANA_SCSPLICENET") -> None:
    """Emit standard nf-core `versions.yml` recording tool and Python versions."""
    import platform
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    yml = (
        f'"{process_name}":\n'
        f'    scsplicenet: "1.0.7"\n'
        f'    python: "{platform.python_version()}"\n'
    )
    p.write_text(yml, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="scsplicenet",
        description="scSpliceNet 1.0.7: Smart-seq3 Hybrid-Manifold ExcessBioVar & support-precision TOM co-splicing network.",
    )
    p.add_argument("--version", action="version", version=f"scsplicenet {__version__}")
    p.add_argument("--usage", type=Path, required=False, help="TSV(.gz): cell_id rows x event_id columns observed usage")
    p.add_argument("--metadata", type=Path, required=False, help="TSV/CSV: QC-passed cell metadata (cell_id, donor_id, cell_type, condition, qc_pass)")
    p.add_argument("--event-genes", type=Path, required=False, help="TSV: event_id, gene_id[, gene_symbol]")
    p.add_argument("--donor-col", default=None, help="Explicit donor/animal column name (auto-detects unambiguous donor aliases if omitted)")
    p.add_argument("--group-col", default=None, help="Optional group/condition column when restricting network to two contrast arms")
    p.add_argument("--numerator", default=None, help="Optional numerator condition label (requires --denominator)")
    p.add_argument("--denominator", default=None, help="Optional denominator condition label (requires --numerator)")
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

    if int(args.threads or 0) > 0:
        t_str = str(int(args.threads))
        for env_k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[env_k] = t_str
        try:
            from threadpoolctl import threadpool_limits
            threadpool_limits(limits=int(args.threads))
        except Exception:
            pass

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def _read_id_mat(path: Path) -> pd.DataFrame:
        df_m = pd.read_csv(path, sep="\t", dtype={0: str})
        if df_m.shape[1] > 0:
            df_m[df_m.columns[0]] = df_m[df_m.columns[0]].astype(str).str.strip()
            df_m = df_m.set_index(df_m.columns[0])
        df_m.index = df_m.index.astype(str)
        df_m.columns = df_m.columns.astype(str)
        return df_m

    usage = _read_id_mat(args.usage)
    md_sep = "," if str(args.metadata).lower().endswith((".csv", ".csv.gz")) else "\t"
    metadata = pd.read_csv(args.metadata, sep=md_sep, dtype=str, keep_default_na=False)
    if len(metadata.columns) != len({str(c).strip().casefold() for c in metadata.columns}):
        raise ValueError(f"Case-colliding or duplicate column headers in {args.metadata}: {list(metadata.columns)}")
    eg_df = pd.read_csv(args.event_genes, sep="\t", dtype={"event_id": str, "gene_id": str, "gene_symbol": str})
    eg_df["event_id"] = eg_df["event_id"].astype(str)
    eg_df["gene_id"] = eg_df["gene_id"].astype(str)
    support = _read_id_mat(args.support) if args.support else None
    if (args.numerator is None) != (args.denominator is None):
        raise ValueError("Specify both --numerator and --denominator (or neither) for contrast-scoped scSpliceNet.")
    if args.numerator is not None and args.denominator is not None:
        num_s = str(args.numerator).strip()
        den_s = str(args.denominator).strip()
        if not num_s or not den_s or num_s == den_s:
            raise ValueError(f"--numerator ({num_s!r}) and --denominator ({den_s!r}) must be distinct non-empty labels.")
        gcol = str(args.group_col or "condition").strip()
        if gcol not in metadata.columns:
            raise ValueError(f"--group-col {gcol!r} not found in metadata columns: {list(metadata.columns)}")
        cid_col = "cell_id" if "cell_id" in metadata.columns else metadata.columns[0]
        gvals = metadata[gcol].astype(str).str.strip()
        m_num = gvals.eq(num_s)
        m_den = gvals.eq(den_s)
        if int(m_num.sum()) == 0 or int(m_den.sum()) == 0:
            raise ValueError(f"Contrast {num_s!r} vs {den_s!r} has empty arm(s) in {gcol!r}: n_num={int(m_num.sum())}, n_den={int(m_den.sum())}.")
        metadata = metadata.loc[m_num | m_den].reset_index(drop=True)
        keep_cells = [c for c in metadata[cid_col].astype(str).str.strip().tolist() if c in usage.index]
        usage = usage.loc[keep_cells]
        if support is not None:
            support = support.loc[[c for c in keep_cells if c in support.index]]
    eg_ser = eg_df.set_index("event_id")["gene_id"].astype(str)

    if args.select_budget > 0 and usage.shape[1] > args.select_budget:
        from scsplicenet.input_contract import validate_sc_usage_input
        bundle = validate_sc_usage_input(
            usage, metadata, eg_df, support=support, donor_col=args.donor_col, require_qc_whitelist=("qc_pass" in metadata.columns)
        )
        host_df = _read_id_mat(args.host_expr) if args.host_expr else None
        n_c = int(bundle.usage.shape[0])
        min_c = max(3, min(15, int(n_c * 0.15)))
        min_sup = max(3.0, min(10.0, float(min_c)))
        aud = audit_sc_events(
            bundle.usage, bundle.event_genes, support=bundle.support, min_cells=min_c, min_total_support=min_sup
        )
        if int(aud["qc_eligible"].sum()) < min(80, int(args.select_budget)) and min_c > 3:
            aud = audit_sc_events(
                bundle.usage, bundle.event_genes, support=bundle.support, min_cells=3, min_total_support=3.0
            )
        aud = attach_split_state_reliability_and_psix(
            aud, bundle.usage, bundle.metadata, support=bundle.support, host_log_expr=host_df, seed=args.seed
        )
        sel = select_sc_events(aud, budget=args.select_budget, max_per_gene=2, ranking_metric=args.selector)
        keep_ids = sel["event_id"].astype(str).tolist()
        if keep_ids:
            usage = bundle.usage.loc[:, keep_ids]
            support = bundle.support.loc[:, keep_ids] if bundle.support is not None else None
            metadata = bundle.metadata
            eg_df = eg_df[eg_df["event_id"].astype(str).isin(set(keep_ids))].copy()
            sel.to_csv(out / "scsplicenet_selected_events.tsv", sep="\t", index=False)

    net = fit_scsplicenet(
        usage=usage, metadata=metadata, event_genes=eg_df, support=support,
        donor_col=args.donor_col,
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
