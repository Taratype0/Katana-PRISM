#!/usr/bin/env python3
"""Slice a cohort-level single-cell splicing matrix bundle (e.g., from `SC_MARVEL_QUANT`)
for a specific `(population_col, population_value, numerator, denominator)` contrast row in `--sc_contrasts`.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def main() -> int:
    ap = argparse.ArgumentParser(description="Slice single-cell splicing matrices for one population contrast.")
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--psi", type=Path, required=True)
    ap.add_argument("--support", type=Path, default=None)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--expression", type=Path, default=None)
    ap.add_argument("--population-col", default="ALL")
    ap.add_argument("--population-value", default="ALL")
    ap.add_argument("--group-col", default="condition")
    ap.add_argument("--numerator", required=True)
    ap.add_argument("--denominator", required=True)
    ap.add_argument("--min-cells", type=int, default=8)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(args.events, sep="\t", low_memory=False)
    md = pd.read_csv(args.metadata, sep="\t")
    psi = pd.read_csv(args.psi, sep="\t", index_col=0)
    psi.index = psi.index.astype(str)
    sup = pd.read_csv(args.support, sep="\t", index_col=0) if (args.support and args.support.is_file()) else None
    if sup is not None:
        sup.index = sup.index.astype(str)

    md["cell_id"] = md["cell_id"].astype(str).str.strip()
    md = md[md["cell_id"].isin(psi.index)].drop_duplicates("cell_id").reset_index(drop=True)

    pcol = str(args.population_col).strip()
    pval = str(args.population_value).strip()
    if pcol and pcol.upper() not in ("ALL", "*", "NONE") and pval.upper() not in ("ALL", "*"):
        if pcol == "not_doublet" and "flag_top2pct" in md.columns:
            md = md[~md["flag_top2pct"].astype(str).str.lower().isin(("true", "1"))].copy()
        elif ":" in pcol and ":" in pval:
            for k_c, v_c in zip(pcol.split(":"), pval.split(":")):
                if k_c in md.columns:
                    if v_c.startswith("!"):
                        md = md[~md[k_c].astype(str).str.lower().eq(v_c[1:].lower())].copy()
                    else:
                        md = md[md[k_c].astype(str).eq(v_c)].copy()
        elif pcol in md.columns:
            md = md[md[pcol].astype(str).eq(pval)].copy()
        else:
            raise ValueError(f"Population column {pcol!r} not found in cell metadata columns: {list(md.columns)}")

    if args.group_col in md.columns and args.group_col != "condition":
        md["condition"] = md[args.group_col].astype(str)

    cids = md["cell_id"].tolist()
    eids = ev["event_id"].astype(str).tolist()
    p_sub = psi.reindex(index=cids, columns=eids)
    s_sub = sup.reindex(index=cids, columns=eids).fillna(0.0) if sup is not None else pd.DataFrame(10.0, index=cids, columns=eids)

    arr = p_sub.to_numpy(dtype=float)
    n_obs = np.isfinite(arr).sum(axis=0)
    with np.errstate(invalid="ignore"):
        m_psi = np.nanmean(arr, axis=0)
    eff_min = max(3, min(int(args.min_cells), len(cids) // 5))
    keep = (n_obs >= eff_min) & np.isfinite(m_psi) & (m_psi >= 0.01) & (m_psi <= 0.99)
    if not keep.any():
        keep = n_obs >= 2
    if not keep.any():
        keep = np.zeros(len(ev), dtype=bool)
        keep[: min(10, len(ev))] = True

    ev_k = ev.loc[keep].reset_index(drop=True)
    kept_ids = ev_k["event_id"].astype(str).tolist()
    p_k = p_sub.loc[:, kept_ids]
    s_k = s_sub.loc[:, kept_ids]
    arr_k = p_k.to_numpy(dtype=float)

    m1 = md["condition"].astype(str).eq(str(args.numerator)).to_numpy()
    m2 = md["condition"].astype(str).eq(str(args.denominator)).to_numpy()
    dpsi = np.full(len(ev_k), np.nan, dtype=float)
    pvals = np.full(len(ev_k), np.nan, dtype=float)
    for i in range(len(ev_k)):
        v1 = arr_k[m1, i]; v1 = v1[np.isfinite(v1)]
        v2 = arr_k[m2, i]; v2 = v2[np.isfinite(v2)]
        if len(v1) >= 2 and len(v2) >= 2:
            dpsi[i] = float(np.mean(v1) - np.mean(v2))
            if len(v1) >= 3 and len(v2) >= 3 and np.std(np.concatenate([v1, v2])) > 0:
                _, pvals[i] = mannwhitneyu(v1, v2, alternative="two-sided")

    fdr = np.full(len(ev_k), np.nan, dtype=float)
    ok_p = np.isfinite(pvals)
    if ok_p.any():
        pv_sub = pvals[ok_p]
        n_p = len(pv_sub)
        o = np.argsort(pv_sub)
        rk = pv_sub[o] * n_p / np.arange(1, n_p + 1)
        q = np.minimum(1.0, np.minimum.accumulate(rk[::-1])[::-1])
        f_sub = np.empty_like(q)
        f_sub[o] = q
        fdr[ok_p] = f_sub

    ev_k["dpsi"] = np.round(dpsi, 5)
    ev_k["pvalue"] = pvals
    ev_k["fdr"] = fdr
    ev_k["comparison_id"] = f"{args.numerator}_vs_{args.denominator}"

    ev_k.to_csv(args.outdir / "sc_events.tsv", sep="\t", index=False)
    ev_k[["event_id", "gene_id", "gene_symbol"]].to_csv(args.outdir / "sc_event_genes.tsv", sep="\t", index=False)
    md.to_csv(args.outdir / "sc_cells.tsv", sep="\t", index=False)
    p_k.to_csv(args.outdir / "sc_psi.tsv.gz", sep="\t", index_label="cell_id", compression="gzip")
    s_k.to_csv(args.outdir / "sc_support.tsv.gz", sep="\t", index_label="cell_id", compression="gzip")

    if args.expression and args.expression.is_file() and args.expression.name != "NO_EXPRESSION":
        expr = pd.read_csv(args.expression, sep="\t", index_col=0)
        expr.index = expr.index.astype(str)
        expr.reindex(index=cids).fillna(0.0).to_csv(
            args.outdir / "sc_expression.tsv.gz", sep="\t", index_label="cell_id", compression="gzip"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
