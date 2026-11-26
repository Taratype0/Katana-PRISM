#!/usr/bin/env python3
"""Slice a cohort-level single-cell splicing matrix bundle (e.g., from `SC_MARVEL_QUANT`)
for a specific `(population_col, population_value, numerator, denominator)` contrast row in `--sc_contrasts`.
"""
from __future__ import annotations

import argparse
import json
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
    def _read_id_mat(path: Path) -> pd.DataFrame:
        df_m = pd.read_csv(path, sep="\t", dtype={0: str})
        if df_m.shape[1] > 0:
            df_m[df_m.columns[0]] = df_m[df_m.columns[0]].astype(str).str.strip()
            df_m = df_m.set_index(df_m.columns[0])
        df_m.index = df_m.index.astype(str)
        df_m.columns = df_m.columns.astype(str)
        return df_m

    ev = pd.read_csv(args.events, sep="\t", dtype={"event_id": str, "gene_id": str, "gene_symbol": str}, low_memory=False)
    md_sep = "," if str(args.metadata).lower().endswith((".csv", ".csv.gz")) else "\t"
    with (
        pd.io.common.get_handle(args.metadata, "r", encoding="utf-8-sig") as _mh
    ):
        _hdr_line = _mh.handle.readline().rstrip("\r\n")
        _raw_hdr = [c.strip() for c in _hdr_line.split(md_sep)] if _hdr_line else []
        if len(_raw_hdr) != len({c.casefold() for c in _raw_hdr}):
            raise ValueError(f"Duplicate or case-colliding column names in --metadata ({args.metadata}): {_raw_hdr}")
    md = pd.read_csv(args.metadata, sep=md_sep, dtype=str, keep_default_na=False)
    md.columns = [str(c).strip() for c in md.columns]
    psi = _read_id_mat(args.psi)
    if psi.index.has_duplicates:
        raise ValueError("Duplicate cell_id rows detected in --psi matrix.")
    if psi.columns.has_duplicates:
        raise ValueError("Duplicate event_id columns detected in --psi matrix.")
    sup = None
    if args.support is not None and args.support.name != "NO_SUPPORT":
        if not args.support.is_file():
            raise FileNotFoundError(f"Missing --support matrix: {args.support}")
        sup = _read_id_mat(args.support)
    if sup is not None and sup.index.has_duplicates:
        raise ValueError("Duplicate cell_id rows detected in --support matrix.")
    if sup is not None and sup.columns.has_duplicates:
        raise ValueError("Duplicate event_id columns detected in --support matrix.")

    md["cell_id"] = md["cell_id"].astype(str).str.strip()
    dup_cid = md["cell_id"].duplicated(keep=False)
    if dup_cid.any():
        bad_cids = md.loc[dup_cid, "cell_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Duplicate physical cell_id entries in metadata (e.g. {bad_cids}).")
    if "qc_pass" in md.columns:
        qc_bad = {"false", "0", "no", "f", "fail"}
        md = md[~md["qc_pass"].astype(str).str.strip().str.lower().isin(qc_bad)].copy()
        if md.empty:
            raise ValueError("Zero cells remain after qc_pass filtering.")
    md = md[md["cell_id"].isin(psi.index)].reset_index(drop=True)

    pcol = str(args.population_col).strip()
    pval = str(args.population_value).strip()
    pop_filters: list[list[str]] = []
    if pcol and pcol.upper() not in ("ALL", "*", "NONE"):
        if ":" in pcol or ":" in pval:
            p_cols = [x.strip() for x in pcol.split(":")]
            p_vals = [x.strip() for x in pval.split(":")]
            if len(p_cols) != len(p_vals) or any(not c for c in p_cols) or any(not v for v in p_vals):
                raise ValueError(
                    f"Malformed composite population filter --population-col={pcol!r}, --population-value={pval!r}."
                )
            for k_c, v_c in zip(p_cols, p_vals):
                if k_c.upper() in ("ALL", "*", "NONE"):
                    continue
                if k_c not in md.columns:
                    raise ValueError(
                        f"Population column {k_c!r} not found in cell metadata columns: {list(md.columns)}"
                    )
                pop_filters.append([k_c, v_c])
                if v_c.upper() in ("ALL", "*"):
                    continue
                if v_c.startswith("!"):
                    md = md[~md[k_c].astype(str).str.lower().eq(v_c[1:].lower())].copy()
                else:
                    md = md[md[k_c].astype(str).eq(v_c)].copy()
        elif pcol == "not_doublet" and "flag_top2pct" in md.columns:
            pop_filters.append([pcol, pval])
            md = md[~md["flag_top2pct"].astype(str).str.lower().isin(("true", "1"))].copy()
        elif pcol in md.columns:
            pop_filters.append([pcol, pval])
            if pval.upper() not in ("ALL", "*"):
                md = md[md[pcol].astype(str).eq(pval)].copy()
        else:
            raise ValueError(f"Population column {pcol!r} not found in cell metadata columns: {list(md.columns)}")

    if args.group_col not in md.columns:
        raise ValueError(f"Group column {args.group_col!r} not found in cell metadata columns: {list(md.columns)}")
    if args.group_col != "condition":
        md["condition"] = md[args.group_col].astype(str)
    if str(args.numerator).strip() == str(args.denominator).strip():
        raise ValueError(f"--numerator and --denominator must name two distinct groups (got {args.numerator!r}).")

    m1_full = md["condition"].astype(str).eq(str(args.numerator)).to_numpy()
    m2_full = md["condition"].astype(str).eq(str(args.denominator)).to_numpy()
    n_num_cells = int(m1_full.sum())
    n_den_cells = int(m2_full.sum())
    eff_min = max(1, int(args.min_cells))
    if n_num_cells == 0 or n_den_cells == 0:
        receipt_err = {
            "status": "missing_contrast_arm",
            "support_supplied": bool(sup is not None),
            "support_mode": "measured" if sup is not None else "missing",
            "population_col": pcol,
            "population_value": pval,
            "population_filters": pop_filters,
            "group_col": str(args.group_col),
            "numerator": str(args.numerator),
            "denominator": str(args.denominator),
            "numerator_cells": n_num_cells,
            "denominator_cells": n_den_cells,
            "min_cells_requested": int(args.min_cells),
            "passing_events": 0,
            "n_cells": int(n_num_cells + n_den_cells),
            "n_events": 0,
        }
        (args.outdir / "slice_receipt.json").write_text(json.dumps(receipt_err, indent=2), encoding="utf-8")
        raise ValueError(
            f"Contrast arm(s) missing after population filtering (numerator={args.numerator!r}: {n_num_cells} cells, "
            f"denominator={args.denominator!r}: {n_den_cells} cells)."
        )

    # Restrict strictly to the two requested contrast arms so off-contrast groups never inflate eligibility or leak into scSpliceNet
    md = md.loc[m1_full | m2_full].reset_index(drop=True)
    m1 = md["condition"].astype(str).eq(str(args.numerator)).to_numpy()
    m2 = md["condition"].astype(str).eq(str(args.denominator)).to_numpy()

    cids = md["cell_id"].tolist()
    eids = ev["event_id"].astype(str).tolist()
    p_sub = psi.reindex(index=cids, columns=eids)
    s_sub = sup.reindex(index=cids, columns=eids).fillna(0.0) if sup is not None else None

    arr = p_sub.to_numpy(dtype=float)
    if sup is not None:
        s_arr = s_sub.to_numpy(dtype=float)  # type: ignore[union-attr]
        obs_mask = np.isfinite(arr) & np.isfinite(s_arr) & (s_arr > 0.0)
        arr_masked = np.where(obs_mask, arr, np.nan)
    else:
        obs_mask = np.isfinite(arr)
        arr_masked = arr
    n_obs = obs_mask.sum(axis=0)
    n_obs_num = obs_mask[m1, :].sum(axis=0)
    n_obs_den = obs_mask[m2, :].sum(axis=0)
    with np.errstate(invalid="ignore"):
        m_psi = np.nanmean(arr_masked, axis=0)
    keep = (n_obs >= eff_min) & (n_obs_num >= 1) & (n_obs_den >= 1) & np.isfinite(m_psi) & (m_psi >= 0.01) & (m_psi <= 0.99)

    ev_k = ev.loc[keep].reset_index(drop=True)
    kept_ids = ev_k["event_id"].astype(str).tolist()
    p_k = p_sub.loc[:, kept_ids]
    s_k = s_sub.loc[:, kept_ids] if s_sub is not None else None
    arr_k = p_k.to_numpy(dtype=float)

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
    sup_out = args.outdir / "sc_support.tsv.gz"
    if s_k is not None:
        s_k.to_csv(sup_out, sep="\t", index_label="cell_id", compression="gzip")
    elif sup_out.exists():
        sup_out.unlink()

    receipt = {
        "status": "complete" if len(ev_k) > 0 else "zero_passing_events",
        "support_supplied": bool(s_k is not None),
        "support_mode": "measured" if s_k is not None else "missing",
        "population_col": pcol,
        "population_value": pval,
        "population_filters": pop_filters,
        "group_col": str(args.group_col),
        "numerator": str(args.numerator),
        "denominator": str(args.denominator),
        "numerator_cells": n_num_cells,
        "denominator_cells": n_den_cells,
        "min_cells_requested": int(args.min_cells),
        "passing_events": int(len(ev_k)),
        "n_cells": int(len(md)),
        "n_events": int(len(ev_k)),
    }
    (args.outdir / "slice_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    if args.expression and args.expression.is_file() and args.expression.name != "NO_EXPRESSION":
        expr = _read_id_mat(args.expression)
        expr.reindex(index=cids).fillna(0.0).to_csv(
            args.outdir / "sc_expression.tsv.gz", sep="\t", index_label="cell_id", compression="gzip"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
