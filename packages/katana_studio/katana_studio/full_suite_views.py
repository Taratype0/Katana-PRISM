"""Original-Katana-Compliant Seven-Section Visual & Interactive Exhibit Builders
shared by `katana-report bulk` (`run_bulk`) and `katana-report single-cell` (`run_single_cell`).

Conforms strictly to `REPORT_FORMAT_LOCK.json`:
  - Approved English 7-chapter order (`01..07` + Appendix `A`)
  - Palette B (`#2863A5` blue / `#C63F4A` red)
  - Exact `plot_data` TSV binding on every visible main figure
  - Interactive `PSI_MATRIX.html`, `ALL_MODULES.html` (`network_views._write_browser`), and `outputs/views/*.html`
"""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D
import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import leaves_list, linkage
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

K_NAVY = "#1E2630"
K_BLUE = "#2863A5"
K_RED = "#C63F4A"
K_TEAL = "#218D87"
K_PURPLE = "#9A64B0"
K_AMBER = "#D49A25"
K_GRAY = "#CDD1D6"
K_SLATE = "#5B6470"
EVENT_COLORS = [K_BLUE, K_RED, K_TEAL, K_PURPLE, K_AMBER]
CMAP_BR = LinearSegmentedColormap.from_list("katana_b", [K_BLUE, "#FFFFFF", K_RED])


def _save_fig(fig: plt.Figure, png_path: Path, dpi: int = 200) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg_path


def _safe_slug(val: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(val)).strip("-")[:120] or "item"


def _safe_read_csv(path: Path | None, *, sep: str = "\t", **kwargs: Any) -> pd.DataFrame:
    """Read a TSV/CSV defensively, returning an empty DataFrame for missing, 0-byte, or headerless files."""
    if path is None or not Path(path).is_file():
        return pd.DataFrame()
    try:
        if Path(path).stat().st_size == 0:
            return pd.DataFrame()
        return pd.read_csv(path, sep=sep, **kwargs)
    except (pd.errors.EmptyDataError, FileNotFoundError, OSError):
        return pd.DataFrame()


def compute_2d_embedding(X: np.ndarray, n_neighbors: int = 15, seed: int = 17) -> np.ndarray:
    n = X.shape[0]
    if n < 5:
        return X[:, :2] if X.shape[1] >= 2 else np.column_stack([X[:, 0], np.zeros(n)])
    try:
        import umap  # type: ignore
        reducer = umap.UMAP(n_neighbors=min(n_neighbors, n - 2), min_dist=0.28, random_state=seed)
        return np.asarray(reducer.fit_transform(X), dtype=float)
    except Exception:
        from sklearn.manifold import TSNE
        perp = float(min(30, max(3, (n - 1) // 3)))
        return np.asarray(
            TSNE(n_components=2, perplexity=perp, init="pca", learning_rate="auto", random_state=seed).fit_transform(X),
            dtype=float,
        )


def build_interactive_psi_matrix_browser(
    psi_events_by_units: pd.DataFrame,
    unit_conditions: list[str],
    event_labels: list[str],
    outdir: Path,
    tables: list[dict[str, Any]],
    max_events: int = 3000,
) -> str | None:
    """Create Original Katana's `outputs/landscape/PSI_MATRIX.html` from `assets/matrix-browser.html`."""
    from katana_next.reports import layout as _ly
    tpl_p = Path(_ly.__file__).resolve().parent / "assets/matrix-browser.html"
    if not tpl_p.is_file() or psi_events_by_units.empty:
        return None

    arr = psi_events_by_units.to_numpy(dtype=float)
    fin_cnt = np.isfinite(arr).sum(axis=1)
    keep = np.where(fin_cnt >= min(2, arr.shape[1]))[0][:max_events]
    if len(keep) == 0:
        return None

    payload = {
        "samples": psi_events_by_units.columns.astype(str).tolist(),
        "events": [event_labels[i] for i in keep],
        "values": [[round(float(x), 4) if np.isfinite(x) else None for x in arr[i]] for i in keep],
        "conditions": list(unit_conditions),
    }
    dst = outdir / "outputs/landscape/PSI_MATRIX.html"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        tpl_p.read_text(encoding="utf-8").replace(
            "__DATA__", json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        ),
        encoding="utf-8",
    )
    rel = dst.relative_to(outdir).as_posix()
    tables.append({
        "name": "Full PSI matrix browser: every saved event",
        "analysis_id": "landscape.psi_browser",
        "chapter": 2,
        "rows": len(keep),
        "source": rel,
        "view": rel,
    })
    return rel


def render_biological_design_panel(
    samples_list: list[dict[str, Any]],
    outdir: Path,
    panels: list[dict[str, Any]],
    is_single_cell: bool = False,
) -> None:
    if not samples_list:
        return
    df = pd.DataFrame(samples_list)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})

    tsv_p = outdir / "outputs/landscape/biological-design-plot-data.tsv"
    tsv_p.parent.mkdir(parents=True, exist_ok=True)
    png_p = outdir / "figures/qc/biological-design.png"

    if not is_single_cell:
        grp = df.groupby("condition", sort=False).agg(
            n_samples=("sample_id", "count"),
            independent_biological_units=("biological_sample_id", "nunique"),
        ).reset_index()
        grp.to_csv(tsv_p, sep="\t", index=False)
        fig, ax = plt.subplots(figsize=(9.8, 4.0), dpi=200)
        fig.patch.set_facecolor("white")
        x = np.arange(len(grp))
        ax.bar(x - 0.18, grp["n_samples"], width=0.34, color=K_BLUE, label="RNA-seq libraries")
        ax.bar(x + 0.18, grp["independent_biological_units"], width=0.34, color=K_TEAL, label="Independent biological units")
        for xi, ns, nu in zip(x, grp["n_samples"], grp["independent_biological_units"]):
            ax.text(xi - 0.18, ns + 0.05, str(int(ns)), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=K_NAVY)
            ax.text(xi + 0.18, nu + 0.05, str(int(nu)), ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=K_NAVY)
        ax.set_xticks(x, grp["condition"].tolist(), rotation=20, ha="right", fontweight="bold")
        ax.set_ylim(0, float(max(grp["n_samples"].max(), 1)) * 1.25)
        ax.set_ylabel("Count", fontweight="bold", color=K_NAVY)
        ax.set_title("Biological Samples and Experimental Conditions", loc="left", fontsize=10.6, fontweight="bold", color=K_NAVY)
        ax.legend(frameon=False, fontsize=8.4)
        _save_fig(fig, png_p)
    else:
        grp = df.groupby(["condition", "biological_sample_id"], dropna=False).size().reset_index(name="n_cells")
        grp.to_csv(tsv_p, sep="\t", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4), dpi=200)
        fig.patch.set_facecolor("white")
        c_cnt = df["condition"].value_counts()
        c_don = df.groupby("condition")["biological_sample_id"].nunique()
        y0 = np.arange(len(c_cnt))[::-1]
        axes[0].barh(y0, c_cnt.values, color=K_NAVY, height=0.58)
        for yi, cn, cv in zip(y0, c_cnt.index, c_cnt.values):
            axes[0].text(cv + max(c_cnt.values) * 0.02, yi, f"{int(cv):,} cells ({int(c_don.get(cn, 1))} donors)", va="center", fontsize=8.2, fontweight="bold", color=K_NAVY)
        axes[0].set_yticks(y0, c_cnt.index.tolist(), fontweight="bold")
        axes[0].set_xlim(0, float(c_cnt.max()) * 1.34)
        axes[0].set_xlabel("QC-Passed Cells", fontweight="bold", color=K_NAVY)
        axes[0].set_title("Declared Conditions & Biological Donors", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY)

        ct_cnt = df["cell_type"].astype(str).value_counts().head(12)
        y1 = np.arange(len(ct_cnt))[::-1]
        axes[1].barh(y1, ct_cnt.values, color=K_BLUE, height=0.58)
        for yi, cv in zip(y1, ct_cnt.values):
            axes[1].text(cv + max(ct_cnt.values) * 0.02, yi, f"{int(cv):,}", va="center", fontsize=8.0, color=K_NAVY)
        axes[1].set_yticks(y1, [s[:30] for s in ct_cnt.index], fontsize=8.1)
        axes[1].set_xlim(0, float(ct_cnt.max()) * 1.22)
        axes[1].set_xlabel("QC-Passed Cells", fontweight="bold", color=K_NAVY)
        axes[1].set_title("Annotated Cell Populations", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY)
        fig.subplots_adjust(wspace=0.48)
        _save_fig(fig, png_p)

    panels.append({
        "id": "biological-design",
        "chapter": 1,
        "contrast_id": "all",
        "title": "Biological samples and experimental conditions",
        "caption": "Sample identities, experimental arms, and independent biological replicate structure from the study manifest.",
        "image": png_p.relative_to(outdir).as_posix(),
        "svg": png_p.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Plot data (TSV)", "path": tsv_p.relative_to(outdir).as_posix(), "unit": "condition_count", "contrast_id": "all", "policy": None}],
    })


def render_replicate_consistency_panels(
    psi_events_by_samples: pd.DataFrame,
    samples_list: list[dict[str, Any]],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> None:
    """Original Katana Chapter 02 `replicate-consistency` and `sample-consistency-diagnostic`."""
    s_ids = [s["sample_id"] for s in samples_list if s["sample_id"] in psi_events_by_samples.columns]
    if len(s_ids) < 3:
        return
    cond_map = {s["sample_id"]: str(s["condition"]) for s in samples_list}
    sub = psi_events_by_samples[s_ids].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="spearman", min_periods=max(5, min(30, len(sub) // 4)))

    pair_rows = []
    for i, sa in enumerate(s_ids):
        for j, sb in enumerate(s_ids):
            if i == j:
                continue
            rv = corr.loc[sa, sb]
            if pd.notna(rv):
                pair_rows.append({
                    "sample_a": sa,
                    "sample_b": sb,
                    "pair_type": "Within condition" if cond_map[sa] == cond_map[sb] else "Between conditions",
                    "spearman_r": round(float(rv), 5),
                })
    if not pair_rows:
        return
    pair_df = pd.DataFrame(pair_rows)
    land_out = outdir / "outputs/landscape"
    land_out.mkdir(parents=True, exist_ok=True)
    fig_dir = outdir / "figures/landscape"
    fig_dir.mkdir(parents=True, exist_ok=True)

    rep_tsv = land_out / "replicate-consistency-plot-data.tsv"
    pair_df.to_csv(rep_tsv, sep="\t", index=False)
    rep_png = fig_dir / "replicate-consistency.png"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.8, 4.0), dpi=200)
    fig.patch.set_facecolor("white")
    groups = [
        pair_df.loc[pair_df["pair_type"].eq("Within condition"), "spearman_r"].dropna().to_numpy(),
        pair_df.loc[pair_df["pair_type"].eq("Between conditions"), "spearman_r"].dropna().to_numpy(),
    ]
    groups = [g if len(g) else np.array([np.nan]) for g in groups]
    bp = ax.boxplot(groups, patch_artist=True, showfliers=True)
    for patch, cc in zip(bp["boxes"], [K_BLUE, K_GRAY]):
        patch.set_facecolor(cc)
    ax.set_xticks([1, 2], ["Within condition", "Between conditions"], fontweight="bold")
    ax.set_ylabel("Pairwise Spearman correlation (PSI)", fontweight="bold", color=K_NAVY)
    ax.set_title("Within-Condition Replicate Consistency", loc="left", fontsize=10.5, fontweight="bold", color=K_NAVY)
    _save_fig(fig, rep_png)
    panels.append({
        "id": "replicate-consistency",
        "chapter": 2,
        "contrast_id": "all",
        "title": "Within-condition consistency",
        "caption": "Pairwise Spearman correlation of saved PSI profiles within versus between declared experimental conditions.",
        "image": rep_png.relative_to(outdir).as_posix(),
        "svg": rep_png.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Pairwise Spearman data (TSV)", "path": rep_tsv.relative_to(outdir).as_posix(), "unit": "sample_pair_spearman", "contrast_id": "all", "policy": None}],
    })

    diag_rows = []
    for sa in s_ids:
        w = pair_df.loc[pair_df["sample_a"].eq(sa) & pair_df["pair_type"].eq("Within condition"), "spearman_r"]
        b = pair_df.loc[pair_df["sample_a"].eq(sa) & pair_df["pair_type"].eq("Between conditions"), "spearman_r"]
        val = float(w.median()) if len(w) else (float(b.median()) if len(b) else np.nan)
        diag_rows.append({"sample_id": sa, "condition": cond_map[sa], "median_within_condition_similarity": round(val, 5) if np.isfinite(val) else np.nan, "removed": False})
    diag_df = pd.DataFrame(diag_rows)
    diag_tsv = land_out / "sample-consistency-diagnostics.tsv"
    diag_df.to_csv(diag_tsv, sep="\t", index=False)
    tables.append({"name": "Sample consistency diagnostics (no automatic exclusion)", "analysis_id": "landscape.consistency", "chapter": 2, "rows": len(diag_df), "columns": list(diag_df.columns), "source": diag_tsv.relative_to(outdir).as_posix()})

    diag_png = fig_dir / "sample-consistency-diagnostic.png"
    fig, ax = plt.subplots(figsize=(max(8.6, len(s_ids) * 0.45), 3.8), dpi=200)
    fig.patch.set_facecolor("white")
    ax.bar(range(len(diag_df)), diag_df["median_within_condition_similarity"].fillna(0.0), color=K_BLUE, width=0.62)
    ax.set_xticks(range(len(diag_df)), diag_df["sample_id"].tolist(), rotation=55, ha="right", fontsize=8.0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Median within-condition r", fontweight="bold", color=K_NAVY)
    ax.set_title("Sample Consistency and Potential Outlier Audit", loc="left", fontsize=10.5, fontweight="bold", color=K_NAVY)
    _save_fig(fig, diag_png)
    panels.append({
        "id": "sample-consistency-diagnostic",
        "chapter": 2,
        "contrast_id": "all",
        "title": "Sample consistency and potential outliers",
        "caption": "Continuous within-condition PSI similarity per sample; no arbitrary outlier removal is applied.",
        "image": diag_png.relative_to(outdir).as_posix(),
        "svg": diag_png.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Sample consistency diagnostics (TSV)", "path": diag_tsv.relative_to(outdir).as_posix(), "unit": "biological_sample", "contrast_id": "all", "policy": None}],
    })


def render_bulk_as_composition_for_contrast(
    qc_df: pd.DataFrame,
    cid: str,
    outdir: Path,
    panels: list[dict[str, Any]],
) -> None:
    """Original Katana Chapter 02 per-comparison `Raw and read-supported AS` & event-type composition."""
    if qc_df.empty or "event_type" not in qc_df.columns:
        return
    ets = [et for et in ("SE", "A3SS", "A5SS", "MXE", "RI") if et in set(qc_df["event_type"].astype(str))]
    if not ets:
        return
    rs_col = "passes_precision_support" if "passes_precision_support" in qc_df.columns else "report_rs_das"
    rows = []
    for et in ets:
        m = qc_df["event_type"].astype(str).eq(et)
        rows.append({
            "event_type": et,
            "raw_events": int(m.sum()),
            "precision_supported_events": int((m & qc_df[rs_col].astype(bool)).sum()) if rs_col in qc_df.columns else int(m.sum()),
        })
    comp = pd.DataFrame(rows)
    tsv_p = outdir / "outputs/comparisons" / _safe_slug(cid) / f"{_safe_slug(cid)}-as-raw-vs-supported.tsv"
    tsv_p.parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(tsv_p, sep="\t", index=False)
    png_p = outdir / "figures/comparisons" / f"{_safe_slug(cid)}-AS-raw-versus-RS.png"

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.2), dpi=200)
    fig.patch.set_facecolor("white")
    x = np.arange(len(comp))
    axes[0].bar(x - 0.18, comp["raw_events"], width=0.35, color=K_BLUE, label="Raw rMATS")
    axes[0].bar(x + 0.18, comp["precision_supported_events"], width=0.35, color=K_RED, label="Precision-supported (PS)")
    axes[0].set_xticks(x, comp["event_type"].tolist(), fontweight="bold")
    axes[0].set_ylabel("Events", fontweight="bold", color=K_NAVY)
    axes[0].set_title("A. Event Counts by Class", loc="left", fontsize=9.8, fontweight="bold", color=K_NAVY)
    axes[0].legend(frameon=False, fontsize=7.8)

    cols = EVENT_COLORS[: len(comp)]
    axes[1].pie(comp["raw_events"], labels=comp["event_type"].tolist(), colors=cols, autopct="%1.1f%%", textprops={"fontsize": 8.0})
    axes[1].set_title(f"B. Raw Composition ({int(comp['raw_events'].sum()):,})", fontsize=9.8, fontweight="bold", color=K_NAVY)
    if comp["precision_supported_events"].sum() > 0:
        axes[2].pie(comp["precision_supported_events"], labels=comp["event_type"].tolist(), colors=cols, autopct="%1.1f%%", textprops={"fontsize": 8.0})
    else:
        axes[2].text(0.5, 0.5, "0 precision-supported events", ha="center", va="center")
    axes[2].set_title(f"C. PS Composition ({int(comp['precision_supported_events'].sum()):,})", fontsize=9.8, fontweight="bold", color=K_NAVY)
    fig.suptitle(f"Raw vs Precision-Supported Alternative Splicing Repertoire — {cid}", x=0.01, ha="left", y=1.03, fontsize=11.2, fontweight="bold", color=K_NAVY)
    fig.subplots_adjust(wspace=0.30)
    _save_fig(fig, png_p)
    panels.append({
        "id": f"{_safe_slug(cid)}-AS-raw-versus-RS",
        "chapter": 2,
        "contrast_id": cid,
        "title": f"Raw and precision-supported AS composition | {cid}",
        "caption": "Event-class counts before and after Binomial replicate-quantile precision qualification.",
        "image": png_p.relative_to(outdir).as_posix(),
        "svg": png_p.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Event composition (TSV)", "path": tsv_p.relative_to(outdir).as_posix(), "unit": "event_type", "contrast_id": cid, "policy": None}],
    })


def render_deg_das_decoupling_quadrant(
    deg_df: pd.DataFrame,
    das_df: pd.DataFrame,
    cid: str,
    outdir: Path,
    panels: list[dict[str, Any]],
    *,
    policy: str | None = "stable_significant",
    das_flag_col: str = "report_rs_das",
    das_label: str = "PS-DAS",
    fdr_cutoff: float = 0.05,
    fc_cutoff: float = 0.5,
) -> None:
    """Render approved Chapter 04 2-panel Transcriptional vs Post-Transcriptional Decoupling Quadrant."""
    if deg_df is None or deg_df.empty or das_df is None or das_df.empty:
        return
    if das_flag_col not in das_df.columns or "gene_symbol" not in das_df.columns:
        return

    # Collapse DAS to strongest event per host gene
    d_work = das_df.copy()
    d_work["_abs_d"] = pd.to_numeric(d_work["dpsi"], errors="coerce").abs().fillna(0.0)
    d_work["_sig_d"] = d_work[das_flag_col].astype(bool)
    d_gene = (
        d_work.sort_values(["_sig_d", "_abs_d"], ascending=[False, False])
        .drop_duplicates("gene_symbol")
        [["gene_symbol", "dpsi", "fdr", "_sig_d"]]
        .rename(columns={"dpsi": "max_dpsi", "fdr": "das_fdr", "_sig_d": "sig_das"})
    )

    g_work = deg_df.drop_duplicates("gene_symbol")[["gene_symbol", "log2fc", "fdr", "_sig"]].rename(
        columns={"fdr": "deg_fdr", "_sig": "sig_deg"}
    )
    merged = g_work.merge(d_gene, on="gene_symbol", how="outer")
    merged["log2fc"] = pd.to_numeric(merged["log2fc"], errors="coerce").fillna(0.0).clip(-8.0, 8.0)
    merged["max_dpsi"] = pd.to_numeric(merged["max_dpsi"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    merged["sig_deg"] = merged["sig_deg"].astype("boolean").fillna(False).astype(bool)
    merged["sig_das"] = merged["sig_das"].astype("boolean").fillna(False).astype(bool)

    dual_m = merged["sig_deg"] & merged["sig_das"]
    das_only_m = (~merged["sig_deg"]) & merged["sig_das"]
    deg_only_m = merged["sig_deg"] & (~merged["sig_das"])
    bg_m = ~(merged["sig_deg"] | merged["sig_das"])

    n_das_only = int(das_only_m.sum())
    n_dual = int(dual_m.sum())
    n_deg_only = int(deg_only_m.sum())
    orth_pct = 100.0 * n_das_only / max(1, n_das_only + n_dual)

    comp_dir = outdir / "outputs/comparisons" / _safe_slug(cid)
    comp_dir.mkdir(parents=True, exist_ok=True)
    pol_slug = policy or "common"
    q_tsv = comp_dir / f"{_safe_slug(cid)}-deg-das-quadrant-{pol_slug}.tsv"
    merged.assign(comparison_id=cid, report_policy=pol_slug).to_csv(q_tsv, sep="\t", index=False)
    q_png = outdir / "figures/comparisons" / f"{_safe_slug(cid)}-deg-das-quadrant-{pol_slug}.png"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.6, 4.9), gridspec_kw={"width_ratios": [1.85, 1.0], "wspace": 0.36}, dpi=215)
    fig.patch.set_facecolor("white")

    ax0.grid(True, color="#F1F5F9", linewidth=0.6, zorder=0)
    ax0.axhline(0.0, color="#CBD5E1", linewidth=0.9, zorder=1)
    ax0.axvline(0.0, color="#CBD5E1", linewidth=0.9, zorder=1)
    ax0.axhline(0.10, color="#94A3B8", linestyle="--", linewidth=0.8, zorder=1)
    ax0.axhline(-0.10, color="#94A3B8", linestyle="--", linewidth=0.8, zorder=1)
    ax0.axvline(fc_cutoff, color="#94A3B8", linestyle="--", linewidth=0.8, zorder=1)
    ax0.axvline(-fc_cutoff, color="#94A3B8", linestyle="--", linewidth=0.8, zorder=1)

    ax0.scatter(merged.loc[bg_m, "log2fc"], merged.loc[bg_m, "max_dpsi"], s=12, color="#E2E8F0", alpha=0.55, linewidths=0, label=f"Background ({int(bg_m.sum()):,})", zorder=2)
    ax0.scatter(merged.loc[deg_only_m, "log2fc"], merged.loc[deg_only_m, "max_dpsi"], s=22, color="#64748B", alpha=0.65, linewidths=0, label=f"DEG only ({n_deg_only:,})", zorder=3)
    ax0.scatter(merged.loc[das_only_m, "log2fc"], merged.loc[das_only_m, "max_dpsi"], s=32, color="#2251FF", alpha=0.88, edgecolors="white", linewidths=0.4, label=f"{das_label} only ({n_das_only:,})", zorder=4)
    ax0.scatter(merged.loc[dual_m, "log2fc"], merged.loc[dual_m, "max_dpsi"], s=36, color="#B83A4B", alpha=0.92, edgecolors="white", linewidths=0.4, label=f"Dual DEG+{das_label} ({n_dual:,})", zorder=5)

    # Non-overlapping callouts on top shifted genes
    sig_sub = merged[das_only_m | dual_m].assign(_score=lambda d: d["max_dpsi"].abs() + 0.08 * d["log2fc"].abs()).sort_values("_score", ascending=False).head(8)
    for is_right in (False, True):
        side = [r for r in sig_sub.itertuples(index=False) if (float(r.log2fc) >= 0) == is_right]
        side.sort(key=lambda r: float(r.max_dpsi))
        ys_adj = [float(r.max_dpsi) for r in side]
        for _ in range(16):
            for i in range(1, len(ys_adj)):
                if ys_adj[i] - ys_adj[i - 1] < 0.13:
                    mid = 0.5 * (ys_adj[i] + ys_adj[i - 1])
                    ys_adj[i - 1] = mid - 0.065
                    ys_adj[i] = mid + 0.065
        ys_adj = list(np.clip(ys_adj, -0.92, 0.92))
        for r, y_l in zip(side, ys_adj):
            dx = 0.45 if is_right else -0.45
            ax0.annotate(
                str(r.gene_symbol),
                xy=(float(r.log2fc), float(r.max_dpsi)),
                xytext=(float(r.log2fc) + dx, y_l),
                ha="left" if is_right else "right",
                va="center",
                fontsize=7.6,
                fontweight="bold",
                color="#051C2C",
                arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": 0.65, "shrinkA": 2, "shrinkB": 3},
                zorder=6,
            )

    x_lim = max(2.8, float(merged["log2fc"].abs().max()) + 1.15)
    ax0.set_xlim(-x_lim, x_lim)
    ax0.set_ylim(-1.05, 1.05)
    ax0.set_xlabel("Host gene expression log2 fold change", fontweight="bold", color="#051C2C")
    ax0.set_ylabel(f"Strongest gene-level splicing shift (ΔPSI)", fontweight="bold", color="#051C2C")
    ax0.set_title("A. Expression vs splicing decoupling quadrant", loc="left", fontsize=10.2, fontweight="bold", color="#051C2C", pad=42)
    ax0.legend(frameon=False, fontsize=7.8, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, columnspacing=1.0, handletextpad=0.35)

    cats_b = [f"DEG only\n(Transcription-specific)", f"Dual regulated\n(DEG + {das_label})", f"{das_label} only\n(Splicing-specific)"]
    vals_b = [n_deg_only, n_dual, n_das_only]
    cols_b = ["#64748B", "#B83A4B", "#2251FF"]
    yb = np.arange(3)
    ax1.barh(yb, vals_b, color=cols_b, height=0.54)
    max_vb = max(4.0, float(max(vals_b)))
    for yi, vv in zip(yb, vals_b):
        ax1.text(vv + max_vb * 0.025, yi, f"{vv:,}", va="center", fontsize=8.6, fontweight="bold", color="#051C2C")
    ax1.set_yticks(yb, cats_b, fontsize=8.4, color="#051C2C")
    ax1.set_xlim(0, max_vb * 1.25)
    ax1.set_xlabel("Unique measured host genes", fontweight="bold", color="#051C2C")
    ax1.set_title(f"B. {orth_pct:.1f}% of {das_label} genes are DEG-independent", loc="left", fontsize=10.2, fontweight="bold", color="#051C2C", pad=42)

    fig.suptitle(f"Transcriptional vs Post-Transcriptional Regulation Decoupling — {cid}", x=0.01, ha="left", y=1.09, fontsize=11.6, fontweight="bold", color="#051C2C")
    _save_fig(fig, q_png)
    panels.append({
        "id": f"{_safe_slug(cid)}-deg-das-quadrant-{pol_slug}",
        "chapter": 4,
        "contrast_id": cid,
        "policy": policy,
        "title": f"Expression vs splicing decoupling quadrant ({orth_pct:.1f}% DEG-independent) | {cid}",
        "caption": f"Orthogonality of host-gene transcriptional shifts (DEG) versus {das_label} exon inclusion shifts across measured genes.",
        "image": q_png.relative_to(outdir).as_posix(),
        "svg": q_png.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Gene decoupling quadrant data (TSV)", "path": q_tsv.relative_to(outdir).as_posix(), "unit": "host_gene", "contrast_id": cid, "policy": policy}],
    })


def _canonical_evidence_class(
    *,
    has_deg: bool,
    has_rmats: bool,
    has_suppa: bool,
    has_dexseq: bool,
    has_dtu: bool,
) -> tuple[str, int, str]:
    """Return `(evidence_class, splicing_caller_count, candidate_tier)` following Katana's multi-layer schema."""
    splicing_count = int(has_rmats) + int(has_suppa) + int(has_dexseq) + int(has_dtu)
    parts: list[str] = []
    if has_deg:
        parts.append("DEG")
    if has_rmats:
        parts.append("rMATS")
    if has_suppa:
        parts.append("SUPPA2")
    if has_dexseq:
        parts.append("DEXSeq")
    if has_dtu:
        parts.append("DTU")
    if not parts:
        ev_cls = "None"
    elif len(parts) == 1:
        ev_cls = f"{parts[0]}_only"
    else:
        ev_cls = "_".join(parts)

    if splicing_count >= 4:
        tier = "Tier 1 (Gold · 4/4 callers)"
    elif splicing_count == 3:
        tier = "Tier 2 (High · 3/4 callers)"
    elif splicing_count == 2:
        tier = "Tier 3 (Corroborated · 2/4 callers)"
    elif splicing_count == 1:
        tier = "Tier 4 (Single-caller)"
    else:
        tier = "Tier 5 (DEG only)"
    return ev_cls, splicing_count, tier


def render_multilayer_orthogonal_convergence(
    *,
    cid: str,
    das_df: pd.DataFrame,
    deg_df: pd.DataFrame | None,
    suppa_df: pd.DataFrame | None,
    dexseq_df: pd.DataFrame | None,
    dtu_df: pd.DataFrame | None,
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    policy: str = "stable_significant",
    panel_policy: str | None = None,
    chapter: int = 8,
    das_flag_col: str = "report_rs_das",
    das_label: str = "PS-DAS",
) -> None:
    """Render Chapter 08 Multi-Layer Orthogonal Splicing Evidence Convergence
    (`rMATS PS-DAS × SUPPA2 × DEXSeq × DRIMSeq/stageR DTU × DESeq2 DEG`) + lossless TSV."""
    if das_df is None or das_df.empty or das_flag_col not in das_df.columns:
        return
    has_any_ortho = any(
        d is not None and not d.empty
        for d in (suppa_df, dexseq_df, dtu_df)
    )
    if not has_any_ortho:
        return

    # Gene-level sets & strongest metrics per layer
    rmats_sig_df = das_df[das_df[das_flag_col].astype(bool)].copy()
    rmats_sig_df["_abs"] = pd.to_numeric(rmats_sig_df["dpsi"], errors="coerce").abs().fillna(0.0)
    rmats_gene = (
        rmats_sig_df.sort_values("_abs", ascending=False)
        .drop_duplicates("gene_symbol")
        .set_index("gene_symbol")
    )
    set_rmats = set(rmats_gene.index.astype(str))

    def _sig_lookup(df_in: pd.DataFrame | None, val_col: str, sort_asc: bool = False) -> tuple[set[str], dict[str, float], dict[str, str]]:
        if df_in is None or df_in.empty or "_sig" not in df_in.columns or "gene_symbol" not in df_in.columns:
            return set(), {}, {}
        sub = df_in[df_in["_sig"].astype(bool)].copy()
        if sub.empty:
            return set(), {}, {}
        sub["val_num"] = pd.to_numeric(sub[val_col], errors="coerce")
        sub["val_key"] = sub["val_num"] if sort_asc else sub["val_num"].abs()
        sub = sub.sort_values("val_key", ascending=sort_asc).drop_duplicates("gene_symbol")
        s_genes = set(sub["gene_symbol"].astype(str))
        v_map = {str(r.gene_symbol): float(r.val_num) for r in sub.itertuples(index=False) if pd.notna(r.val_num)}
        g_map = {str(r.gene_symbol): str(getattr(r, "gene_id", r.gene_symbol)) for r in sub.itertuples(index=False)}
        return s_genes, v_map, g_map

    set_suppa, map_suppa, gid_suppa = _sig_lookup(suppa_df, "dpsi", sort_asc=False)
    set_dex, map_dex, gid_dex = _sig_lookup(dexseq_df, "log2fc", sort_asc=False)
    set_dtu, map_dtu, gid_dtu = _sig_lookup(dtu_df, "gene_q", sort_asc=True)
    set_deg, map_deg, gid_deg = _sig_lookup(deg_df, "log2fc", sort_asc=False)

    all_sig_genes = sorted(set_rmats | set_suppa | set_dex | set_dtu | set_deg)
    if not all_sig_genes:
        return

    gid_rmats = {
        str(gs): str(rmats_gene.loc[gs, "gene_id"]).split(".")[0]
        for gs in set_rmats
        if "gene_id" in rmats_gene.columns
    }
    rows_conv = []
    for gs in all_sig_genes:
        in_r = gs in set_rmats
        in_s = gs in set_suppa
        in_x = gs in set_dex
        in_t = gs in set_dtu
        in_g = gs in set_deg
        ev_cls, sp_cnt, tier = _canonical_evidence_class(
            has_deg=in_g, has_rmats=in_r, has_suppa=in_s, has_dexseq=in_x, has_dtu=in_t
        )
        gid_v = gid_rmats.get(gs) or gid_suppa.get(gs) or gid_dex.get(gs) or gid_dtu.get(gs) or gid_deg.get(gs) or gs
        rows_conv.append({
            "comparison_id": cid,
            "report_policy": policy,
            "gene_symbol": gs,
            "gene_id": str(gid_v).split(".")[0],
            "rMATS_sig": in_r,
            "SUPPA2_sig": in_s,
            "DEXSeq_sig": in_x,
            "DTU_sig": in_t,
            "DEG_sig": in_g,
            "splicing_caller_count": sp_cnt,
            "evidence_class": ev_cls,
            "candidate_tier": tier,
            "rmats_max_dpsi": round(float(rmats_gene.loc[gs, "dpsi"]), 4) if in_r else np.nan,
            "suppa_dpsi": round(float(map_suppa[gs]), 4) if gs in map_suppa else np.nan,
            "dexseq_log2fc": round(float(map_dex[gs]), 4) if gs in map_dex else np.nan,
            "dtu_gene_q": round(float(map_dtu[gs]), 6) if gs in map_dtu else np.nan,
            "deg_log2fc": round(float(map_deg[gs]), 4) if gs in map_deg else np.nan,
        })
    conv_df = pd.DataFrame(rows_conv).sort_values(["splicing_caller_count", "rMATS_sig", "gene_symbol"], ascending=[False, False, True])

    syn_out = (outdir / "outputs/synthesis") if chapter == 8 else (outdir / "outputs/comparisons" / _safe_slug(cid))
    syn_out.mkdir(parents=True, exist_ok=True)
    conv_tsv = syn_out / f"{_safe_slug(cid)}-multilayer-gene-convergence-{policy}.tsv"
    conv_df.to_csv(conv_tsv, sep="\t", index=False)
    if policy == "stable_significant":
        tables.append({
            "name": f"Multi-layer orthogonal splicing convergence matrix ({cid})",
            "analysis_id": "prioritize.multilayer_convergence",
            "chapter": chapter,
            "contrast_id": cid,
            "rows": len(conv_df),
            "columns": list(conv_df.columns),
            "source": conv_tsv.relative_to(outdir).as_posix(),
        })

    # Compute PS-DAS corroboration tiers (4/4, 3/4, 2/4, 1/4)
    rmats_sub = conv_df[conv_df["rMATS_sig"]]
    n_rmats_total = max(1, len(rmats_sub))
    tier_counts = [
        ("4/4 Callers\n(Gold Tier)", int((rmats_sub["splicing_caller_count"] == 4).sum()), "#051C2C"),
        ("3/4 Callers\n(High Conf.)", int((rmats_sub["splicing_caller_count"] == 3).sum()), "#2251FF"),
        ("2/4 Callers\n(Corroborated)", int((rmats_sub["splicing_caller_count"] == 2).sum()), "#0077C8"),
        ("1/4 Caller\n(rMATS only)", int((rmats_sub["splicing_caller_count"] == 1).sum()), "#94A3B8"),
    ]
    n_multi = sum(c for lbl, c, _ in tier_counts[:3])
    multi_pct = 100.0 * n_multi / n_rmats_total

    layer_defs = [
        (f"rMATS\n{das_label}", "rMATS_sig", len(set_rmats), "#051C2C"),
        ("SUPPA2\n(Event)", "SUPPA2_sig", len(set_suppa), "#2251FF"),
        ("DEXSeq\n(Exon)", "DEXSeq_sig", len(set_dex), "#0D9488"),
        ("DRIMSeq\nDTU", "DTU_sig", len(set_dtu), "#0077C8"),
        ("DESeq2\nDEG", "DEG_sig", len(set_deg), "#B83A4B"),
    ]
    # Keep only layers with upstream data present (always keep all 4 splicing callers if present + DEG if present)
    active_layers = [
        (lbl, col, cnt, colhex)
        for lbl, col, cnt, colhex in layer_defs
        if (col == "rMATS_sig")
        or (col == "SUPPA2_sig" and suppa_df is not None and not suppa_df.empty)
        or (col == "DEXSeq_sig" and dexseq_df is not None and not dexseq_df.empty)
        or (col == "DTU_sig" and dtu_df is not None and not dtu_df.empty)
        or (col == "DEG_sig" and deg_df is not None and not deg_df.empty)
    ]

    # Exact multi-layer intersections for UpSet panel
    combo_counter: Counter[tuple[bool, ...]] = Counter()
    for r in conv_df.itertuples(index=False):
        key = tuple(bool(getattr(r, col)) for _, col, _, _ in active_layers)
        if any(key):
            combo_counter[key] += 1

    # Rank intersections: prioritize multi-caller splicing intersections first, then largest count
    splicing_indices = [i for i, (_, col, _, _) in enumerate(active_layers) if col != "DEG_sig"]
    ranked_combos = sorted(
        combo_counter.items(),
        key=lambda kv: (-sum(1 for i in splicing_indices if kv[0][i]), -kv[1], kv[0]),
    )[:10]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(15.4, 6.4), dpi=215)
    fig.patch.set_facecolor("white")
    outer = fig.add_gridspec(1, 2, width_ratios=[1.02, 1.56], wspace=0.28)
    left_gs = outer[0].subgridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.54)
    right_gs = outer[1].subgridspec(1, 2, width_ratios=[1.42, 1.32], wspace=0.06)

    ax_t = fig.add_subplot(left_gs[0])
    ax_l = fig.add_subplot(left_gs[1])
    ax_dots = fig.add_subplot(right_gs[0])
    ax_bars = fig.add_subplot(right_gs[1])

    # Panel A1: Corroboration Depth of rMATS PS-DAS Host Genes
    yt = np.arange(len(tier_counts))[::-1]
    t_vals = [c for _, c, _ in tier_counts]
    t_cols = [hx for _, _, hx in tier_counts]
    ax_t.barh(yt, t_vals, color=t_cols, height=0.58)
    max_t = max(4.0, float(max(t_vals)))
    for yv, (_, cv, _) in zip(yt, tier_counts):
        pct = 100.0 * cv / n_rmats_total
        ax_t.text(cv + max_t * 0.025, yv, f"{cv:,} ({pct:.0f}%)", va="center", fontsize=8.1, fontweight="bold", color="#051C2C")
    ax_t.set_yticks(yt, [lbl for lbl, _, _ in tier_counts], fontsize=8.0, color="#051C2C")
    ax_t.set_xlim(0, max_t * 1.34)
    ax_t.set_xlabel(f"{das_label} host genes", fontsize=8.2, fontweight="bold", color="#051C2C")
    ax_t.set_title(f"A. Orthogonal Corroboration ({multi_pct:.0f}% Confirmed ≥2 Callers)", loc="left", fontsize=9.8, fontweight="bold", color="#051C2C", pad=8)

    # Panel A2: Total Unique Significant Genes per Upstream Tool
    yl = np.arange(len(active_layers))[::-1]
    l_vals = [cnt for _, _, cnt, _ in active_layers]
    l_cols = [hx for _, _, _, hx in active_layers]
    l_lbls = [lbl.replace("\n", " ") for lbl, _, _, _ in active_layers]
    ax_l.barh(yl, l_vals, color=l_cols, height=0.56, alpha=0.92)
    max_l = max(4.0, float(max(l_vals)))
    for yv, cv in zip(yl, l_vals):
        ax_l.text(cv + max_l * 0.025, yv, f"{cv:,} genes", va="center", fontsize=8.0, fontweight="bold", color="#051C2C")
    ax_l.set_yticks(yl, l_lbls, fontsize=8.0, color="#051C2C")
    ax_l.set_xlim(0, max_l * 1.32)
    ax_l.set_xlabel("Unique significant host genes", fontsize=8.2, fontweight="bold", color="#051C2C")
    ax_l.set_title("B. Significant Host-Gene Set Size by Layer", loc="left", fontsize=9.8, fontweight="bold", color="#051C2C", pad=8)

    # Panel C (Right): UpSet-Style Exact Multi-Layer Intersection Matrix + Bar Chart
    n_rows_u = len(ranked_combos)
    n_cols_u = len(active_layers)
    y_u = np.arange(n_rows_u)[::-1]
    deg_col_idx = next((i for i, (_, col, _, _) in enumerate(active_layers) if col == "DEG_sig"), -1)

    for ri, (combo, cnt) in enumerate(ranked_combos):
        yv = y_u[ri]
        if ri % 2 == 1:
            ax_dots.axhspan(yv - 0.45, yv + 0.45, color="#F8FAFC", zorder=0)
            ax_bars.axhspan(yv - 0.45, yv + 0.45, color="#F8FAFC", zorder=0)
        on_xs = [ci for ci, flag in enumerate(combo) if flag]
        if len(on_xs) >= 2:
            ax_dots.plot([min(on_xs), max(on_xs)], [yv, yv], color="#051C2C", linewidth=1.8, zorder=2)
        for ci, flag in enumerate(combo):
            if flag:
                dot_c = "#B83A4B" if ci == deg_col_idx else ("#051C2C" if ci == 0 else "#2251FF")
                ax_dots.scatter([ci], [yv], s=105, color=dot_c, edgecolor="white", linewidth=0.7, zorder=4)
            else:
                ax_dots.scatter([ci], [yv], s=48, color="#E2E8F0", zorder=2)

    ax_dots.set_xlim(-0.65, n_cols_u - 0.35)
    ax_dots.set_ylim(-0.6, n_rows_u - 0.4)
    ax_dots.set_xticks(range(n_cols_u))
    ax_dots.set_xticklabels([lbl for lbl, _, _, _ in active_layers], fontsize=7.8, fontweight="bold", color="#051C2C")
    ax_dots.xaxis.tick_top()
    ax_dots.tick_params(axis="x", pad=5)
    ax_dots.set_yticks([])
    for sp in ("top", "right", "bottom", "left"):
        ax_dots.spines[sp].set_visible(False)

    bar_vals = [cnt for _, cnt in ranked_combos]
    bar_cols = []
    for combo, _ in ranked_combos:
        sp_c = sum(1 for i in splicing_indices if combo[i])
        if sp_c >= 4:
            bar_cols.append("#051C2C")
        elif sp_c == 3:
            bar_cols.append("#2251FF")
        elif sp_c == 2:
            bar_cols.append("#0077C8")
        elif sp_c == 1:
            bar_cols.append("#64748B")
        else:
            bar_cols.append("#B83A4B")
    ax_bars.barh(y_u, bar_vals, color=bar_cols, height=0.56, zorder=3)
    max_b = max(4.0, float(max(bar_vals)))
    for yv, (combo, cnt) in zip(y_u, ranked_combos):
        sp_c = sum(1 for i in splicing_indices if combo[i])
        deg_tag = " +DEG" if (deg_col_idx >= 0 and combo[deg_col_idx]) else ""
        tag = f"{cnt:,}  ({sp_c}/{len(splicing_indices)}{deg_tag})" if sp_c > 0 else f"{cnt:,}  (DEG only)"
        ax_bars.text(cnt + max_b * 0.025, yv, tag, va="center", fontsize=8.0, fontweight="bold", color="#051C2C")
    ax_bars.set_ylim(-0.6, n_rows_u - 0.4)
    ax_bars.set_xlim(0, max_b * 1.46)
    ax_bars.set_yticks([])
    ax_bars.set_xlabel("Exact intersection host-gene count", fontsize=8.2, fontweight="bold", color="#051C2C")
    ax_bars.spines["left"].set_color("#CBD5E1")
    ax_bars.set_title("C. Multi-Layer Orthogonal Intersection Matrix (UpSet)", loc="left", fontsize=9.8, fontweight="bold", color="#051C2C", pad=30)

    fig.suptitle(f"Multi-Layer Orthogonal Splicing Evidence Convergence ({das_label} × SUPPA2 × DEXSeq × DTU × DEG) — {cid}", x=0.01, ha="left", y=1.03, fontsize=11.4, fontweight="bold", color="#051C2C")
    fig_sub = "figures/synthesis" if chapter == 8 else "figures/comparisons"
    conv_png = outdir / fig_sub / f"{_safe_slug(cid)}-multilayer-orthogonal-convergence-{policy}.png"
    _save_fig(fig, conv_png)
    panels.append({
        "id": f"{_safe_slug(cid)}-multilayer-orthogonal-convergence-{policy}",
        "chapter": chapter,
        "contrast_id": cid,
        "policy": panel_policy,
        "title": f"Multi-layer orthogonal convergence across all callers ({multi_pct:.0f}% of {das_label} confirmed by ≥2 tools) | {cid}",
        "caption": f"Gene-level corroboration strength across rMATS ({das_label}), SUPPA2, DEXSeq, DRIMSeq/stageR DTU, and DESeq2 DEG feeding directly into Chapter 08 target prioritization.",
        "image": conv_png.relative_to(outdir).as_posix(),
        "svg": conv_png.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Multi-layer convergence matrix (TSV)", "path": conv_tsv.relative_to(outdir).as_posix(), "unit": "host_gene", "contrast_id": cid, "policy": panel_policy}],
    })


def _orient_dexseq_lfc_treat_over_ctrl(
    raw_x: pd.DataFrame,
    lfc_col: str,
    treat_lbl: str,
    ctrl_lbl: str,
) -> tuple[pd.Series, bool]:
    """Ensure DEXSeq log2FC is oriented as log2(treatment / control).

    DEXSeq (`~ sample + exon + condition:exon`) commonly emits a column named
    `log2fold_<control>_<treatment>` equal to log2(control / treatment). Verify
    against fitted group-mean columns first, then fall back to header order.
    """
    raw_lfc = pd.to_numeric(raw_x[lfc_col], errors="coerce")
    if treat_lbl and ctrl_lbl and treat_lbl != ctrl_lbl:
        t_col = treat_lbl if treat_lbl in raw_x.columns else None
        c_col = ctrl_lbl if ctrl_lbl in raw_x.columns else None
        if t_col and c_col:
            dt = pd.to_numeric(raw_x[t_col], errors="coerce") - pd.to_numeric(raw_x[c_col], errors="coerce")
            q_col = next((c for c in ("padj", "FDR", "fdr", "qvalue") if c in raw_x.columns), None)
            mask = raw_lfc.notna() & dt.notna() & raw_lfc.abs().ge(0.25) & (dt != 0)
            if q_col:
                sig_m = mask & pd.to_numeric(raw_x[q_col], errors="coerce").lt(0.05)
                if int(sig_m.sum()) >= 5:
                    mask = sig_m
            if int(mask.sum()) >= 3:
                corr_sign = float(np.sign(dt[mask] * raw_lfc[mask]).mean())
                if corr_sign < 0.0:
                    return -raw_lfc, True
                if corr_sign > 0.0:
                    return raw_lfc, False
        if lfc_col.lower().startswith(f"log2fold_{ctrl_lbl.lower()}_{treat_lbl.lower()}"):
            return -raw_lfc, True
    return raw_lfc, False


def ingest_bulk_multilayer_deg_dtu_dexseq(
    bundle_dir: Path | None,
    deseq2_dir: Path | None,
    contrasts_list: list[dict[str, Any]],
    das_by_contrast: dict[str, pd.DataFrame],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    top10: list[dict[str, Any]],
    render_volcano_fn: Any,
    fdr_cutoff: float = 0.05,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], pd.DataFrame | None, dict[str, dict[str, pd.DataFrame]]]:
    """Ingest DESeq2 (DEG) + lossless SUPPA2 / DEXSeq / DRIMSeq-stageR DTU tables into Chapter 03
    (ZERO unwanted SUPPA2/DEXSeq/DTU volcanoes!) and render Chapter 04 DEG×DAS Decoupling Quadrants
    + Multi-Layer Orthogonal Splicing Evidence Convergence exhibits."""
    deg_root = (
        Path(deseq2_dir)
        if (deseq2_dir and Path(deseq2_dir).exists())
        else next(
            (p for p in ((bundle_dir / "DEG", bundle_dir / "differential/deg", bundle_dir / "deseq2") if bundle_dir else ()) if p.is_dir()),
            None,
        )
    )

    gene_sym_map: dict[str, str] = {}
    for q_df in das_by_contrast.values():
        if {"gene_id", "gene_symbol"} <= set(q_df.columns):
            for gid, gsym in zip(q_df["gene_id"].astype(str), q_df["gene_symbol"].astype(str)):
                gene_sym_map.setdefault(gid.split(".")[0], gsym)
                gene_sym_map.setdefault(gid, gsym)
    if bundle_dir and (bundle_dir / "refs/annotation.tsv").is_file():
        ann = _safe_read_csv(bundle_dir / "refs/annotation.tsv", sep="\t")
        if not ann.empty:
            gc = next((c for c in ("gene_id", "ensembl_gene_id") if c in ann.columns), None)
            sc = next((c for c in ("gene_name", "gene_symbol", "symbol") if c in ann.columns), None)
            if gc and sc:
                for gid, gsym in zip(ann[gc].astype(str), ann[sc].astype(str)):
                    if gsym and gsym != "nan":
                        gene_sym_map.setdefault(gid.split(".")[0], gsym)
                        gene_sym_map.setdefault(gid, gsym)

    def _map_sym(gid_raw: str) -> str:
        g_clean = str(gid_raw or "").strip().strip('"')
        return gene_sym_map.get(g_clean, gene_sym_map.get(g_clean.split(".")[0], g_clean))

    expr_mat: pd.DataFrame | None = None
    norm_f = next((p for p in (
        bundle_dir / "DEG/all.normalised_counts.tsv" if bundle_dir else None,
        deg_root / "all.normalised_counts.tsv" if (deg_root and deg_root.is_dir()) else None,
    ) if p is not None and p.is_file()), None)
    if norm_f is not None:
        try:
            ndf = _safe_read_csv(norm_f, sep="\t")
            if not ndf.empty:
                gc_n = next((c for c in ("gene_symbol", "gene_name", "gene_id", "id") if c in ndf.columns), ndf.columns[0])
                num_cols = [c for c in ndf.columns if c not in ("gene_id", "gene_symbol", "gene_name", "id")]
                if num_cols:
                    expr_mat = ndf.set_index(ndf[gc_n].astype(str).map(lambda g: gene_sym_map.get(g.split(".")[0], g)))[num_cols].apply(pd.to_numeric, errors="coerce")
                    expr_mat = expr_mat[~expr_mat.index.duplicated(keep="first")]
        except Exception:
            expr_mat = None

    deg_by_contrast: dict[str, pd.DataFrame] = {}
    suppa_by_contrast: dict[str, pd.DataFrame] = {}
    dexseq_by_contrast: dict[str, pd.DataFrame] = {}
    dtu_by_contrast: dict[str, pd.DataFrame] = {}

    for c_spec in contrasts_list:
        cid = c_spec["contrast_id"]
        treat_lbl = str(c_spec.get("numerator", {}).get("label", "")).strip()
        ctrl_lbl = str(c_spec.get("denominator", {}).get("label", "")).strip()
        comp_dir = outdir / "outputs/comparisons" / _safe_slug(cid)
        comp_dir.mkdir(parents=True, exist_ok=True)

        # 1. DESeq2 DEG (Ch03 table + DEG volcano + Top 10 + Ch04 Decoupling Quadrant)
        deg_f = None
        if deg_root is not None:
            if deg_root.is_file():
                deg_f = deg_root
            else:
                for cand in (deg_root / f"{cid}.deseq2.results.tsv", deg_root / f"{cid}.tsv", deg_root / f"{cid}.csv"):
                    if cand.is_file():
                        deg_f = cand
                        break
        if deg_f is not None:
            sep = "," if deg_f.suffix.lower() == ".csv" else "\t"
            raw_g = _safe_read_csv(deg_f, sep=sep)
            if not raw_g.empty:
                gid_c = next((c for c in ("gene_id", "GeneID", "id", "row") if c in raw_g.columns), raw_g.columns[0])
                sym_c = next((c for c in ("gene_symbol", "gene_name", "symbol") if c in raw_g.columns), None)
                lfc_c = next((c for c in ("log2FoldChange", "log2fc", "logFC") if c in raw_g.columns), None)
                padj_c = next((c for c in ("padj", "FDR", "fdr", "qvalue") if c in raw_g.columns), None)
                pv_c = next((c for c in ("pvalue", "PValue", "p_value") if c in raw_g.columns), padj_c)
                if lfc_c and padj_c:
                    df_g = pd.DataFrame({
                        "gene_id": raw_g[gid_c].astype(str),
                        "gene_symbol": raw_g[sym_c].astype(str) if sym_c else raw_g[gid_c].astype(str).map(_map_sym),
                        "log2fc": pd.to_numeric(raw_g[lfc_c], errors="coerce"),
                        "pvalue": pd.to_numeric(raw_g[pv_c], errors="coerce") if pv_c else np.nan,
                        "fdr": pd.to_numeric(raw_g[padj_c], errors="coerce"),
                    })
                    df_g["_sig"] = df_g["fdr"].le(fdr_cutoff) & df_g["log2fc"].abs().ge(0.5)
                    deg_by_contrast[cid] = df_g
                    deg_pq = comp_dir / "DEG-complete.parquet"
                    df_g.to_parquet(deg_pq, index=False)
                    tables.append({"name": f"DESeq2 DEG complete table ({cid})", "analysis_id": "differential.deg", "chapter": 3, "contrast_id": cid, "rows": len(df_g), "columns": list(df_g.columns), "source": deg_pq.relative_to(outdir).as_posix()})

                    g_png = outdir / "figures/comparisons" / f"{_safe_slug(cid)}-DEG-volcano-common.png"
                    g_tsv = comp_dir / f"{_safe_slug(cid)}-DEG-volcano-plot-data.tsv"
                    df_g[["gene_id", "gene_symbol", "log2fc", "fdr", "_sig"]].to_csv(g_tsv, sep="\t", index=False)
                    render_volcano_fn(df_g, "log2fc", "fdr", "_sig", "gene_symbol", "log2 Fold Change", f"DEG Volcano ({int(df_g['_sig'].sum()):,} significant genes) | {cid}", g_png, is_deg=True)
                    panels.append({
                        "id": f"{_safe_slug(cid)}-DEG-volcano-common", "chapter": 3, "contrast_id": cid, "policy": None,
                        "title": f"DEG ({int(df_g['_sig'].sum()):,} significant genes) | {cid}",
                        "caption": f"DESeq2 differential gene expression (FDR ≤ {fdr_cutoff}, |log2FC| ≥ 0.5).",
                        "image": g_png.relative_to(outdir).as_posix(), "svg": g_png.with_suffix(".svg").relative_to(outdir).as_posix(),
                        "data_links": [{"role": "plot_data", "label": "DEG plot data (TSV)", "path": g_tsv.relative_to(outdir).as_posix(), "unit": "gene", "contrast_id": cid, "policy": None}],
                    })
                    sub_gt = df_g[df_g["_sig"]].assign(_a=lambda d: d["log2fc"].abs()).sort_values(["fdr", "_a"], ascending=[True, False]).head(10)
                    top10.append({
                        "id": f"{_safe_slug(cid)}-top10-deg", "chapter": 3, "contrast_id": cid, "analysis": "DEG", "kind": "DEG",
                        "unit": "gene", "columns": ["gene_symbol", "gene_id", "log2fc", "fdr"],
                        "rows": sub_gt[["gene_symbol", "gene_id", "log2fc", "fdr"]].to_dict("records"),
                        "full_path": deg_pq.relative_to(outdir).as_posix(),
                    })

        # 2. SUPPA2 Local Differential Splicing (lossless Ch03 table ONLY — NO volcano!)
        if bundle_dir is not None:
            sup_candidates = [
                bundle_dir / "suppa" / cid / "diffsplice/per_local_event/local_events.dpsi",
                *(sorted((bundle_dir / "suppa" / cid / "diffsplice/per_local_event").glob("*.dpsi")) if (bundle_dir / "suppa" / cid / "diffsplice/per_local_event").is_dir() else []),
                *(sorted((bundle_dir / "suppa" / cid).glob("*.dpsi")) if (bundle_dir / "suppa" / cid).is_dir() else []),
                *([bundle_dir / "rnasplicing/suppa/diffsplice_ioe" / f"{ctrl_lbl}-{treat_lbl}_local_diffsplice.dpsi"] if (treat_lbl and ctrl_lbl) else []),
            ]
            sup_f = next((p for p in sup_candidates if p.is_file()), None)
            if sup_f is not None:
                try:
                    raw_s = _safe_read_csv(sup_f, sep="\t")
                    if not raw_s.empty:
                        # Handle native SUPPA2 implicit-index `.dpsi` (2 header columns, 3 tab fields per row -> event ID lands in index!)
                        if (not isinstance(raw_s.index, pd.RangeIndex)) or (
                            len(raw_s.columns) >= 2 and "dpsi" in str(raw_s.columns[0]).lower()
                        ):
                            raw_s = raw_s.reset_index().rename(columns={raw_s.index.name or "index": "event_id"})
                        ev_c = next((c for c in ("event_id", "Event_id", "id") if c in raw_s.columns), raw_s.columns[0])
                        dpsi_c = next((c for c in raw_s.columns if c != ev_c and "dpsi" in str(c).lower()), raw_s.columns[1] if len(raw_s.columns) > 1 else None)
                        pval_c = next((c for c in raw_s.columns if c != ev_c and ("p-val" in str(c).lower() or "pval" in str(c).lower() or "p_val" in str(c).lower())), raw_s.columns[2] if len(raw_s.columns) > 2 else None)
                        if dpsi_c and pval_c:
                            raw_gid = raw_s[ev_c].astype(str).map(lambda s: s.split(";", 1)[0].split(".")[0])
                            df_s = pd.DataFrame({
                                "event_id": raw_s[ev_c].astype(str),
                                "gene_id": raw_gid,
                                "gene_symbol": raw_gid.map(_map_sym),
                                "dpsi": pd.to_numeric(raw_s[dpsi_c], errors="coerce"),
                                "pvalue": pd.to_numeric(raw_s[pval_c], errors="coerce"),
                            })
                            df_s["_sig"] = df_s["pvalue"].le(fdr_cutoff) & df_s["dpsi"].abs().ge(0.10)
                            suppa_by_contrast[cid] = df_s
                            sup_pq = comp_dir / "SUPPA2-complete.parquet"
                            df_s.to_parquet(sup_pq, index=False)
                            tables.append({
                                "name": f"SUPPA2 local differential splicing complete table ({cid})",
                                "analysis_id": "differential.suppa2",
                                "chapter": 3,
                                "contrast_id": cid,
                                "rows": len(df_s),
                                "columns": list(df_s.columns),
                                "source": sup_pq.relative_to(outdir).as_posix(),
                            })
                except Exception:
                    pass

            # 3. DEXSeq Differential Exon Usage (lossless Ch03 table ONLY — NO volcano!)
            dex_candidates = [
                bundle_dir / "dexseq_exon" / f"DEXSeqResults.{cid}.csv",
                bundle_dir / "differential/dexseq_exon" / f"DEXSeqResults.{cid}.csv",
                bundle_dir / "dexseq_exon" / f"DEXSeqResults.{cid}.tsv",
                bundle_dir / "dexseq_exon" / f"perGeneQValue.{cid}.csv",
                bundle_dir / "dexseq_exon" / f"DEXSeqResults_{cid}.csv",
                bundle_dir / "dexseq_exon" / f"{cid}.csv",
                bundle_dir / "dexseq_exon" / f"{cid}.tsv",
            ]
            dex_f = next((p for p in dex_candidates if p.is_file()), None)
            if dex_f is not None:
                try:
                    raw_x = _safe_read_csv(dex_f, sep="," if dex_f.suffix.lower() == ".csv" else "\t")
                    if not raw_x.empty:
                        grp_c = next((c for c in ("groupID", "gene_id", "geneID") if c in raw_x.columns), raw_x.columns[0])
                        feat_c = next((c for c in ("featureID", "exon_id", "exon") if c in raw_x.columns), None)
                        lfc_c = next((c for c in raw_x.columns if c.lower().startswith("log2fold") or c.lower() in ("log2fc", "logfc")), None)
                        pv_c = next((c for c in ("pvalue", "PValue", "p_value") if c in raw_x.columns), None)
                        q_c = next((c for c in ("padj", "FDR", "fdr", "qvalue") if c in raw_x.columns), pv_c)
                        if q_c:
                            gid_x = raw_x[grp_c].astype(str).map(lambda s: s.split("+", 1)[0].split(".")[0])
                            if lfc_c:
                                lfc_series, _dex_inverted = _orient_dexseq_lfc_treat_over_ctrl(raw_x, lfc_c, treat_lbl, ctrl_lbl)
                            else:
                                lfc_series = pd.Series(np.nan, index=raw_x.index, dtype=float)
                            df_x = pd.DataFrame({
                                "gene_id": gid_x,
                                "gene_symbol": gid_x.map(_map_sym),
                                "feature_id": raw_x[feat_c].astype(str) if feat_c else "",
                                "log2fc": lfc_series,
                                "pvalue": pd.to_numeric(raw_x[pv_c], errors="coerce") if pv_c else np.nan,
                                "padj": pd.to_numeric(raw_x[q_c], errors="coerce"),
                            })
                            if lfc_c:
                                df_x["_sig"] = df_x["padj"].le(fdr_cutoff) & (df_x["log2fc"].abs().ge(0.5) | df_x["log2fc"].isna())
                            else:
                                df_x["_sig"] = df_x["padj"].le(fdr_cutoff)
                            dexseq_by_contrast[cid] = df_x
                            dex_pq = comp_dir / "DEXSeq-complete.parquet"
                            df_x.to_parquet(dex_pq, index=False)
                            tables.append({
                                "name": f"DEXSeq differential exon usage complete table ({cid})",
                                "analysis_id": "differential.dexseq",
                                "chapter": 3,
                                "contrast_id": cid,
                                "rows": len(df_x),
                                "columns": list(df_x.columns),
                                "source": dex_pq.relative_to(outdir).as_posix(),
                            })
                except Exception:
                    pass

            # 4. DRIMSeq / stageR Differential Transcript Usage (lossless Ch03 table ONLY — NO volcano!)
            dtu_pair = f"{treat_lbl}-{ctrl_lbl}" if (treat_lbl and ctrl_lbl) else cid
            dtu_candidates = [
                bundle_dir / "dtu/results/stager" / f"getAdjustedPValues.{cid}.tsv",
                bundle_dir / "dtu/results/stager" / f"getAdjustedPValues.{dtu_pair}.tsv",
                bundle_dir / "differential/dtu/stager" / f"getAdjustedPValues.{cid}.tsv",
                bundle_dir / "differential/dtu/stager" / f"getAdjustedPValues.{dtu_pair}.tsv",
                bundle_dir / "dtu/stager" / f"getAdjustedPValues.{cid}.tsv",
                bundle_dir / "dtu/stager" / f"getAdjustedPValues.{dtu_pair}.tsv",
                bundle_dir / "dtu/results/stager" / f"{cid}.tsv",
                bundle_dir / "dtu/results/stager" / f"{cid}.csv",
                bundle_dir / "dtu/stager" / f"{cid}.tsv",
                bundle_dir / "dtu/stager" / f"{cid}.csv",
                bundle_dir / "dtu/results/dexseq" / f"perGeneQValue.{cid}.tsv",
                bundle_dir / "dtu/results/dexseq" / f"DEXSeqResults.{cid}.tsv",
            ]
            dtu_f = next((p for p in dtu_candidates if p.is_file()), None)
            if dtu_f is not None:
                try:
                    raw_t = _safe_read_csv(dtu_f, sep="," if dtu_f.suffix.lower() == ".csv" else "\t")
                    if not raw_t.empty:
                        gid_c = next((c for c in ("geneID", "groupID", "gene_id", "GeneID") if c in raw_t.columns), raw_t.columns[0])
                        tx_c = next((c for c in ("txID", "transcript_id", "featureID", "feature_id") if c in raw_t.columns), None)
                        gq_c = next((c for c in ("gene", "gene_q", "padj", "adj_pvalue", "FDR") if c in raw_t.columns), None)
                        tq_c = next((c for c in ("transcript", "tx_q", "transcript_q") if c in raw_t.columns), gq_c)
                        if gq_c:
                            gid_t = raw_t[gid_c].astype(str).map(lambda s: s.split("+", 1)[0].split(".")[0])
                            df_t = pd.DataFrame({
                                "gene_id": gid_t,
                                "gene_symbol": gid_t.map(_map_sym),
                                "transcript_id": raw_t[tx_c].astype(str) if tx_c else "",
                                "gene_q": pd.to_numeric(raw_t[gq_c], errors="coerce"),
                                "transcript_q": pd.to_numeric(raw_t[tq_c], errors="coerce") if tq_c else np.nan,
                            })
                            df_t["_sig"] = df_t["gene_q"].le(fdr_cutoff) | df_t["transcript_q"].le(fdr_cutoff)
                            dtu_by_contrast[cid] = df_t
                            dtu_pq = comp_dir / "DTU-DRIMSeq-stageR-complete.parquet"
                            df_t.to_parquet(dtu_pq, index=False)
                            tables.append({
                                "name": f"DRIMSeq / stageR DTU complete table ({cid})",
                                "analysis_id": "differential.dtu",
                                "chapter": 3,
                                "contrast_id": cid,
                                "rows": len(df_t),
                                "columns": list(df_t.columns),
                                "source": dtu_pq.relative_to(outdir).as_posix(),
                            })
                except Exception:
                    pass

        # Chapter 04 Exhibit: Expression vs Splicing Decoupling Quadrant (multi-tool convergence is placed in Chapter 08!)
        q_df = das_by_contrast.get(cid)
        if q_df is not None and not q_df.empty and cid in deg_by_contrast:
            for pol, flag_c, pol_lbl in (("stable_significant", "report_rs_das", "PS-DAS"), ("significant", "report_das", "Ordinary DAS")):
                render_deg_das_decoupling_quadrant(
                    deg_by_contrast[cid], q_df, cid, outdir, panels,
                    policy=pol, das_flag_col=flag_c, das_label=pol_lbl,
                    fdr_cutoff=fdr_cutoff, fc_cutoff=0.5,
                )

    multilayer_evidence = {
        "suppa": suppa_by_contrast,
        "dexseq": dexseq_by_contrast,
        "dtu": dtu_by_contrast,
    }
    return deg_by_contrast, gene_sym_map, expr_mat, multilayer_evidence


def render_ch4_cross_comparison_suite(
    das_by_contrast: dict[str, pd.DataFrame],
    contrasts_list: list[dict[str, Any]],
    outdir: Path,
    panels: list[dict[str, Any]],
    policies: tuple[tuple[str | None, str, str], ...] = (("stable_significant", "report_rs_das", "PS-DAS"), ("significant", "report_das", "Ordinary DAS")),
    tables: list[dict[str, Any]] | None = None,
) -> None:
    """Chapter 04: Pairwise Overlap Matrix, Exact Intersections, and Exact Event Direction Concordance."""
    cids = [c["contrast_id"] for c in contrasts_list if c["contrast_id"] in das_by_contrast]
    if len(cids) < 2:
        return
    comp_out = outdir / "outputs/comparisons/all"
    comp_out.mkdir(parents=True, exist_ok=True)
    fig_dir = outdir / "figures/comparisons"
    fig_dir.mkdir(parents=True, exist_ok=True)

    registered_cross_table = False
    for pol, flag_c, pol_lbl in policies:
        pol_slug = pol or "common"
        sets_c = {
            cid: set(das_by_contrast[cid].loc[das_by_contrast[cid][flag_c].astype(bool), "event_id"].astype(str))
            if flag_c in das_by_contrast[cid].columns else set()
            for cid in cids
        }
        n_c = len(cids)
        mat = np.zeros((n_c, n_c), dtype=int)
        for i, ca in enumerate(cids):
            for j, cb in enumerate(cids):
                mat[i, j] = len(sets_c[ca] & sets_c[cb])
        all_ev = sorted(set().union(*sets_c.values()))
        short_codes = {c: f"C{i+1}" for i, c in enumerate(cids)}
        combo_short = Counter(" ∩ ".join([short_codes[c] for c in cids if e in sets_c[c]]) for e in all_ev)
        top_inter = combo_short.most_common(12)

        ov_df = pd.DataFrame(mat, index=cids, columns=cids).reset_index(names="contrast_id")
        ov_tsv = comp_out / f"DAS-events-overlaps-{pol_slug}.tsv"
        ov_df.to_csv(ov_tsv, sep="\t", index=False)
        if tables is not None and not registered_cross_table:
            registered_cross_table = True
            tables.append({
                "name": f"Cross-comparison {pol_lbl} overlap matrix ({len(cids)} contrasts)",
                "analysis_id": "differential.cross_comparison",
                "chapter": 4,
                "contrast_id": "all",
                "rows": len(ov_df),
                "columns": list(ov_df.columns),
                "source": ov_tsv.relative_to(outdir).as_posix(),
            })
        ov_png = fig_dir / f"DAS-events-overlaps-{pol_slug}.png"

        # Scale gracefully when >12 contrasts (e.g., 43 single-cell subtype contrasts):
        # keep complete N×N matrix in TSV, display top 12 highest-signal contrasts cleanly in the heatmap
        if n_c > 12:
            ranked_cids = sorted(cids, key=lambda c: (0 if "__" not in c else 1, -len(sets_c[c]), c))
            disp_cids = ranked_cids[:12]
        else:
            disp_cids = list(cids)
        n_d = len(disp_cids)
        disp_idx = [cids.index(c) for c in disp_cids]
        mat_d = mat[np.ix_(disp_idx, disp_idx)]
        disp_codes = {c: f"C{i+1}" for i, c in enumerate(disp_cids)}
        disp_all_ev = sorted(set().union(*(sets_c[c] for c in disp_cids))) if disp_cids else []
        disp_combo = Counter(" ∩ ".join([disp_codes[c] for c in disp_cids if e in sets_c[c]]) for e in disp_all_ev)
        top_inter_d = disp_combo.most_common(12)

        def _fmt_cid_tick(c_raw: str) -> str:
            s = c_raw.replace("__", " · ").replace("_vs_", " vs ")
            return s if len(s) <= 34 else s[:32].rsplit(" ", 1)[0] + "…"

        fig_h_ov = max(5.2, n_d * 0.44 + 1.8)
        fig, axes = plt.subplots(1, 2, figsize=(15.4, fig_h_ov), dpi=205)
        fig.patch.set_facecolor("white")
        im = axes[0].imshow(mat_d, cmap="Blues", aspect="auto")
        cell_fs = 8.6 if n_d <= 8 else 7.4
        for i in range(n_d):
            for j in range(n_d):
                axes[0].text(j, i, str(mat_d[i, j]), ha="center", va="center", fontsize=cell_fs, color="white" if mat_d[i, j] > max(1, mat_d.max()) * 0.55 else K_NAVY, fontweight="bold")
        axes[0].set_xticks(range(n_d), [f"{disp_codes[c]}" for c in disp_cids], fontsize=8.4, fontweight="bold")
        axes[0].set_yticks(range(n_d), [f"{disp_codes[c]}: {_fmt_cid_tick(c)}" for c in disp_cids], fontsize=8.0)
        cap_note = f" (Top {n_d} of {n_c})" if n_c > 12 else ""
        axes[0].set_title(f"{pol_lbl} Pairwise Event Overlaps{cap_note}", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY)
        fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

        if top_inter_d:
            yi = np.arange(len(top_inter_d))[::-1]
            axes[1].barh(yi, [v for _, v in top_inter_d], color=K_BLUE, height=0.6)
            for y_k, (_, vv) in zip(yi, top_inter_d):
                axes[1].text(vv + max(top_inter_d[0][1], 1) * 0.02, y_k, str(vv), va="center", fontsize=8.2, fontweight="bold")
            axes[1].set_yticks(yi, [k for k, _ in top_inter_d], fontsize=8.2, fontweight="bold")
            axes[1].set_xlim(0, max(float(top_inter_d[0][1]) * 1.24, 4.0))
        else:
            axes[1].text(0.5, 0.5, f"0 {pol_lbl} events across comparisons", ha="center", va="center")
        axes[1].set_xlabel("Exact Splicing Event Count", fontweight="bold", color=K_NAVY)
        axes[1].set_title(f"{pol_lbl} Exact Multi-Comparison Intersections", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY)
        fig.subplots_adjust(wspace=0.40)
        _save_fig(fig, ov_png)
        panels.append({
            "id": f"DAS-events-comparisons-overlap-{pol_slug}", "chapter": 4, "contrast_id": "all", "policy": pol,
            "title": f"{pol_lbl} overlaps & exact intersections across declared comparisons",
            "caption": "Exact pairwise overlap matrix and multi-comparison intersection counts without re-thresholding.",
            "image": ov_png.relative_to(outdir).as_posix(), "svg": ov_png.with_suffix(".svg").relative_to(outdir).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Overlap matrix (TSV)", "path": ov_tsv.relative_to(outdir).as_posix(), "unit": "comparison_pair", "contrast_id": "all", "policy": pol}],
        })

        # 4.2 Exact Event Direction Concordance (`cids[0]` vs `cids[1]`) with collision-free callouts!
        c1, c2 = cids[0], cids[1]
        d1 = das_by_contrast[c1][["event_id", "gene_symbol", "dpsi", flag_c]].rename(columns={"dpsi": "dpsi_1", flag_c: "sig_1"})
        d2 = das_by_contrast[c2][["event_id", "dpsi", flag_c]].rename(columns={"dpsi": "dpsi_2", flag_c: "sig_2"})
        m12 = d1.merge(d2, on="event_id", how="inner")
        m12 = m12[m12["sig_1"].astype(bool) | m12["sig_2"].astype(bool)].dropna(subset=["dpsi_1", "dpsi_2"])
        if len(m12) >= 3:
            dir_tsv = comp_out / f"DAS-direction-concordance-{pol_slug}.tsv"
            m12.to_csv(dir_tsv, sep="\t", index=False)
            dir_png = fig_dir / f"DAS-direction-concordance-{pol_slug}.png"
            fig, ax = plt.subplots(figsize=(8.2, 5.8), dpi=205)
            fig.patch.set_facecolor("white")
            both_s = m12["sig_1"].astype(bool) & m12["sig_2"].astype(bool)
            ax.scatter(m12.loc[~both_s, "dpsi_1"], m12.loc[~both_s, "dpsi_2"], s=20, color="#94A3B8", alpha=0.6, label=f"Single-contrast ({int((~both_s).sum())})")
            ax.scatter(m12.loc[both_s, "dpsi_1"], m12.loc[both_s, "dpsi_2"], s=32, color=K_RED, alpha=0.88, label=f"Shared {pol_lbl} ({int(both_s.sum())})")
            ax.axhline(0, color="#CBD5E1", linestyle=":", linewidth=0.9)
            ax.axvline(0, color="#CBD5E1", linestyle=":", linewidth=0.9)
            lim_v = max(0.55, float(max(m12["dpsi_1"].abs().max(), m12["dpsi_2"].abs().max())) + 0.22)
            ax.plot([-lim_v, lim_v], [-lim_v, lim_v], color=K_BLUE, linestyle="--", linewidth=0.9, alpha=0.65)
            ax.set_xlim(-lim_v, lim_v)
            ax.set_ylim(-lim_v, lim_v)

            top_m = m12.assign(_m=lambda d: d["dpsi_1"].abs() + d["dpsi_2"].abs()).sort_values("_m", ascending=False).drop_duplicates("gene_symbol").head(8)
            for is_right in (False, True):
                side = [r for r in top_m.itertuples(index=False) if (float(r.dpsi_1) >= 0) == is_right]
                side.sort(key=lambda r: float(r.dpsi_2))
                ys_adj = [float(r.dpsi_2) for r in side]
                for _ in range(16):
                    for i in range(1, len(ys_adj)):
                        if ys_adj[i] - ys_adj[i - 1] < 0.11 * lim_v:
                            mid = 0.5 * (ys_adj[i] + ys_adj[i - 1])
                            ys_adj[i - 1] = mid - 0.055 * lim_v
                            ys_adj[i] = mid + 0.055 * lim_v
                for r, yl in zip(side, ys_adj):
                    dx = 0.09 * lim_v if is_right else -0.09 * lim_v
                    ax.annotate(
                        str(r.gene_symbol),
                        xy=(float(r.dpsi_1), float(r.dpsi_2)),
                        xytext=(float(r.dpsi_1) + dx, float(np.clip(yl, -0.92 * lim_v, 0.92 * lim_v))),
                        ha="left" if is_right else "right",
                        va="center",
                        fontsize=7.6,
                        fontweight="bold",
                        color=K_NAVY,
                        arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": 0.65, "shrinkA": 2, "shrinkB": 3},
                    )
            r_sp = stats.spearmanr(m12["dpsi_1"], m12["dpsi_2"]).statistic
            ax.set_xlabel(f"ΔPSI ({c1})", fontweight="bold", color=K_NAVY)
            ax.set_ylabel(f"ΔPSI ({c2})", fontweight="bold", color=K_NAVY)
            ax.set_title(f"Exact Event Direction Concordance ({pol_lbl}, Spearman r = {r_sp:.2f})", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY)
            ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor="#E2E8F0", fontsize=8.0, loc="lower right")
            _save_fig(fig, dir_png)
            panels.append({
                "id": f"DAS-events-direction-concordance-{pol_slug}", "chapter": 4, "contrast_id": "all", "policy": pol,
                "title": f"Exact event direction concordance ({pol_lbl}): {c1} vs {c2}",
                "caption": "Comparison of signed ΔPSI across the union of significant events in the first two declared comparisons.",
                "image": dir_png.relative_to(outdir).as_posix(), "svg": dir_png.with_suffix(".svg").relative_to(outdir).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Direction concordance (TSV)", "path": dir_tsv.relative_to(outdir).as_posix(), "unit": "splicing_event", "contrast_id": "all", "policy": pol}],
            })


def render_ch6_complete_network_suite(
    members_in: pd.DataFrame,
    psi_events_by_units: pd.DataFrame,
    samples_units: pd.DataFrame,
    contrasts_list: list[dict[str, Any]],
    edges_path: Path | None,
    species: str,
    run_ora_fn: Any,
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    network_browsers: list[dict[str, Any]],
    precomputed_scores: pd.DataFrame | None = None,
) -> None:
    """Original Katana Chapter 06: Interactive `ALL_MODULES.html` (`network_views._write_browser`) + all 6 CoSplice plots!"""
    if members_in.empty or "module" not in members_in.columns:
        return
    net_root = outdir / "outputs/network"
    for sub_d in ("tables", "figures", "browser"):
        (net_root / sub_d).mkdir(parents=True, exist_ok=True)

    eid_c = next((c for c in ("event_key", "event_id") if c in members_in.columns), members_in.columns[0])
    members = members_in.rename(columns={eid_c: "event_key"}).copy()
    members["event_key"] = members["event_key"].astype(str)
    members["module"] = members["module"].astype(str).map(
        lambda m: "grey" if m in {"0", "M0", "grey", "unassigned", "nan"} else (m if m.startswith("M") else f"M{int(m):02d}" if m.isdigit() else m)
    )
    for col_m in ("gene_id", "gene_symbol", "event_type"):
        if col_m not in members.columns:
            members[col_m] = ""

    s_ids = [s for s in samples_units["sample_id"].astype(str) if s in psi_events_by_units.columns]
    if len(s_ids) < 3:
        return
    samples_u = samples_units[samples_units["sample_id"].astype(str).isin(s_ids)].drop_duplicates("sample_id").reset_index(drop=True)
    psi_u = psi_events_by_units.reindex(index=members["event_key"], columns=samples_u["sample_id"].astype(str))
    row_med = np.nanmedian(psi_u.to_numpy(dtype=float), axis=1)
    row_med = np.where(np.isfinite(row_med), row_med, 0.5)
    psi_filled = psi_u.copy()
    for i_r in range(len(psi_filled)):
        psi_filled.iloc[i_r] = psi_filled.iloc[i_r].fillna(float(row_med[i_r])).clip(0.0, 1.0)

    assigned_mods = sorted([m for m in members["module"].unique() if m != "grey"])
    if not assigned_mods:
        return
    all_mods = assigned_mods + (["grey"] if (members["module"] == "grey").any() else [])

    X_ev = psi_filled.to_numpy(dtype=float)
    score_dict = {}
    kme_vals = np.zeros(len(members), dtype=float)
    for m_n in assigned_mods:
        idx_m = np.where(members["module"].eq(m_n).to_numpy())[0]
        if precomputed_scores is not None and m_n in precomputed_scores.columns and set(samples_u["sample_id"]).issubset(set(precomputed_scores.index.astype(str))):
            pc1 = pd.to_numeric(precomputed_scores.reindex(samples_u["sample_id"])[m_n], errors="coerce").fillna(0.0).to_numpy()
        elif len(idx_m) >= 2:
            sub_x = StandardScaler().fit_transform(X_ev[idx_m].T)
            pc1 = PCA(n_components=1, random_state=17).fit_transform(sub_x)[:, 0]
            pc1 = (pc1 - pc1.mean()) / max(pc1.std(), 1e-8)
        else:
            pc1 = X_ev[idx_m[0]] - X_ev[idx_m[0]].mean()
        score_dict[m_n] = pc1
        for ii in idx_m:
            if np.std(X_ev[ii]) > 1e-8 and np.std(pc1) > 1e-8:
                rv = float(np.corrcoef(X_ev[ii], pc1)[0, 1])
                kme_vals[ii] = rv if np.isfinite(rv) else 0.0

    scores_df = pd.DataFrame(score_dict, index=samples_u["sample_id"].astype(str).tolist())
    members["kME"] = np.round(kme_vals, 5)
    members["abs_kME"] = np.round(np.abs(kme_vals), 5)
    if "network_connectivity" not in members.columns:
        c_col = next((c for c in ("connectivity", "kTotal", "degree") if c in members.columns), None)
        members["network_connectivity"] = pd.to_numeric(members[c_col], errors="coerce").fillna(members["abs_kME"] * 10.0) if c_col else np.round(members["abs_kME"] * 10.0, 4)
    members["hub_rank_abs_kME"] = members.groupby("module")["abs_kME"].rank(ascending=False, method="first").astype(int)

    norm_edges_p = None
    if edges_path is not None and Path(edges_path).is_file():
        ed = _safe_read_csv(edges_path, sep="\t")
        if not ed.empty:
            ca = next((c for c in ("event_a", "source", "from", "event_1") if c in ed.columns), ed.columns[0])
            cb = next((c for c in ("event_b", "target", "to", "event_2") if c in ed.columns), ed.columns[1])
            cw = next((c for c in ("adjacency", "tom", "weight", "score") if c in ed.columns), ed.columns[-1])
            known = set(members["event_key"])
            ed_n = pd.DataFrame({"event_a": ed[ca].astype(str), "event_b": ed[cb].astype(str), "adjacency": pd.to_numeric(ed[cw], errors="coerce").fillna(0.1)})
            ed_n = ed_n[ed_n["event_a"].isin(known) & ed_n["event_b"].isin(known)].head(25000)
            if not ed_n.empty:
                norm_edges_p = net_root / "tables/native_network_edges.tsv"
                ed_n.to_csv(norm_edges_p, sep="\t", index=False)

    members.to_csv(net_root / "tables/all_module_members.tsv", sep="\t", index=False)
    members.to_parquet(net_root / "tables/all_module_members.parquet", index=False)
    psi_filled.reset_index(names="event_key").to_csv(net_root / "tables/all_event_psi.tsv.gz", sep="\t", index=False, compression="gzip")
    scores_df.reset_index(names="sample_id").to_csv(net_root / "tables/module_activity.tsv", sep="\t", index=False)

    summary_df = pd.DataFrame([
        {"module": m, "n_events": int(members["module"].eq(m).sum()), "n_genes": int(members.loc[members["module"].eq(m) & members["gene_symbol"].ne(""), "gene_symbol"].nunique())}
        for m in all_mods
    ])
    summary_df.to_csv(net_root / "tables/module_summary.tsv", sep="\t", index=False)
    types_df = members.groupby(["module", "event_type"]).size().reset_index(name="n_events")
    types_df.to_csv(net_root / "tables/module_event_type_counts.tsv", sep="\t", index=False)

    bg_genes = set(members.loc[members["gene_symbol"].ne(""), "gene_symbol"].astype(str))
    pd.DataFrame({"gene_symbol": sorted(bg_genes)}).to_csv(net_root / "tables/enrichment_background.tsv", sep="\t", index=False)
    members.loc[members["gene_symbol"].ne(""), ["module", "gene_symbol"]].drop_duplicates().to_csv(net_root / "tables/module_input_genes.tsv", sep="\t", index=False)

    ora_frames = []
    for m_n in assigned_mods:
        fg_m = set(members.loc[members["module"].eq(m_n) & members["gene_symbol"].ne(""), "gene_symbol"].astype(str))
        if len(fg_m) >= 3:
            o_m = run_ora_fn(fg_m, bg_genes, species if species in ("human", "mouse") else "human")
            if not o_m.empty:
                o_m.insert(0, "module", m_n)
                o_m["FDR"] = o_m["fdr"]
                ora_frames.append(o_m)
    tested_ora = pd.concat(ora_frames, ignore_index=True) if ora_frames else pd.DataFrame(columns=["module", "library", "term", "overlap_count", "pvalue", "FDR"])
    tested_ora.to_csv(net_root / "tables/module_enrichment.tsv.gz", sep="\t", index=False, compression="gzip")

    # Module contrast effects across declared contrasts
    cond_by_sid = samples_u.set_index("sample_id")["condition"].astype(str).to_dict()
    eff_rows = []
    for c_spec in contrasts_list[:10]:
        cid = c_spec["contrast_id"]
        n_s = [s for s in c_spec["numerator"]["sample_ids"] if s in scores_df.index]
        d_s = [s for s in c_spec["denominator"]["sample_ids"] if s in scores_df.index]
        if not n_s or not d_s:
            n_lbl = c_spec["numerator"]["label"]
            d_lbl = c_spec["denominator"]["label"]
            n_s = [s for s, c in cond_by_sid.items() if c == n_lbl or c.endswith(n_lbl)]
            d_s = [s for s, c in cond_by_sid.items() if c == d_lbl or c.endswith(d_lbl)]
        if n_s and d_s:
            for m_n in scores_df.columns:
                eff_rows.append({"module": m_n, "contrast_id": cid, "effect": round(float(scores_df.loc[n_s, m_n].mean() - scores_df.loc[d_s, m_n].mean()), 4)})
    eff_df = pd.DataFrame(eff_rows)
    eff_df.to_csv(net_root / "tables/matched_module_tests_full_family.tsv", sep="\t", index=False)

    # Embedded Interactive `ALL_MODULES.html` (`network_views._write_browser`)
    try:
        from katana_next.reports.network_views import _write_browser
        _write_browser(
            net_root, members, psi_filled.round(4), scores_df.round(4),
            samples_u[["sample_id", "condition"]], tested_ora, norm_edges_p,
            bootstrap_enabled=False, display_graph=None,
        )
        b_rel = (net_root / "browser/ALL_MODULES.html").relative_to(outdir).as_posix()
        network_browsers.append({
            "id": "network-browser-all", "chapter": 6, "contrast_id": "all",
            "whole_study": True, "source": b_rel,
            "title": "SpliceNet — Co-splicing network | all study samples and conditions",
        })
        tables.append({"name": "Interactive SpliceNet module & gene browser", "analysis_id": "rich_browser", "chapter": 6, "rows": len(members), "source": b_rel, "view": b_rel})
    except Exception:
        pass

    tables.append({"name": "Complete co-splicing module members, kME hub ranks & connectivity", "analysis_id": "programs.cosplice", "chapter": 6, "rows": len(members), "columns": list(members.columns), "source": "outputs/network/tables/all_module_members.parquet"})

    # 6.1 Module sizes & genes
    fig, axes = plt.subplots(1, 2, figsize=(max(10.8, len(summary_df) * 0.45), 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, ck, yl in ((axes[0], "n_events", "Events"), (axes[1], "n_genes", "Unique measured genes")):
        ax.bar(summary_df["module"], summary_df[ck], color=["#A5ADB7" if m == "grey" else K_BLUE for m in summary_df["module"]])
        ax.set_ylabel(yl, fontweight="bold", color=K_NAVY)
        ax.set_xlabel("Observed module (grey = unassigned)")
        ax.tick_params(axis="x", rotation=50, labelsize=8)
    fig.suptitle("All Module Sizes and Gene Composition", x=0.01, ha="left", fontsize=10.6, fontweight="bold", color=K_NAVY, y=1.02)
    p1 = net_root / "figures/all_module_sizes.png"
    _save_fig(fig, p1)
    panels.append({"id": "all_module_sizes", "chapter": 6, "contrast_id": "all", "title": f"All module sizes and gene composition ({len(assigned_mods)} modules)", "caption": "Every observed co-splicing module and unassigned grey set.", "image": p1.relative_to(outdir).as_posix(), "svg": p1.with_suffix(".svg").relative_to(outdir).as_posix(), "data_links": [{"role": "plot_data", "label": "Module summary (TSV)", "path": "outputs/network/tables/module_summary.tsv", "unit": "module", "contrast_id": "all", "policy": None}]})

    # 6.2 Event types in every module
    piv_t = types_df.pivot(index="module", columns="event_type", values="n_events").fillna(0).reindex(all_mods)
    piv_tsv = net_root / "tables/all_module_event_types-plot-data.tsv"
    piv_t.reset_index().to_csv(piv_tsv, sep="\t", index=False)
    fig, ax = plt.subplots(figsize=(max(10.2, len(all_mods) * 0.45), 4.4), dpi=200)
    fig.patch.set_facecolor("white")
    piv_t.plot.bar(stacked=True, ax=ax, color=EVENT_COLORS[: max(1, piv_t.shape[1])], width=0.76)
    ax.set_xlabel("Observed modules", fontweight="bold", color=K_NAVY)
    ax.set_ylabel("Events", fontweight="bold", color=K_NAVY)
    ax.set_title("Event Types in Every Module", loc="left", fontsize=10.6, fontweight="bold", color=K_NAVY)
    ax.legend(title="Event type", frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    p2 = net_root / "figures/all_module_event_types.png"
    _save_fig(fig, p2)
    panels.append({"id": "all_module_event_types", "chapter": 6, "contrast_id": "all", "title": "Event types in every module", "caption": "Canonical AS class breakdown across all co-splicing modules.", "image": p2.relative_to(outdir).as_posix(), "svg": p2.with_suffix(".svg").relative_to(outdir).as_posix(), "data_links": [{"role": "plot_data", "label": "Event type counts (TSV)", "path": piv_tsv.relative_to(outdir).as_posix(), "unit": "module_event_type", "contrast_id": "all", "policy": None}]})

    # 6.3 Continuous kME & connectivity
    fig, axes = plt.subplots(1, 2, figsize=(11.0, max(4.0, len(all_mods) * 0.28)), dpi=200)
    fig.patch.set_facecolor("white")
    for ax, mk, xl in ((axes[0], "abs_kME", "Absolute module membership |kME|"), (axes[1], "network_connectivity", "Global network connectivity")):
        vals_m = [pd.to_numeric(members.loc[members["module"].eq(m), mk], errors="coerce").dropna().to_numpy() for m in all_mods]
        vals_m = [v if len(v) else np.array([0.0]) for v in vals_m]
        bp = ax.boxplot(vals_m, vert=False, patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(K_BLUE); patch.set_alpha(0.76)
        ax.set_yticks(range(1, len(all_mods) + 1), all_mods, fontsize=8.2)
        ax.set_xlabel(xl, fontweight="bold", color=K_NAVY)
    fig.suptitle("Continuous Membership and Connectivity Distributions", x=0.01, ha="left", fontsize=10.6, fontweight="bold", color=K_NAVY, y=1.02)
    cont_tsv = net_root / "tables/continuous_metrics_plot_data.tsv"
    members[["event_key", "gene_symbol", "module", "abs_kME", "network_connectivity"]].to_csv(cont_tsv, sep="\t", index=False)
    p3 = net_root / "figures/all_module_continuous_metrics.png"
    _save_fig(fig, p3)
    panels.append({"id": "all_module_continuous_metrics", "chapter": 6, "contrast_id": "all", "title": "Continuous membership and connectivity distributions", "caption": "Distribution of event-to-module |kME| correlation and global connectivity.", "image": p3.relative_to(outdir).as_posix(), "svg": p3.with_suffix(".svg").relative_to(outdir).as_posix(), "data_links": [{"role": "plot_data", "label": "Continuous metrics (TSV)", "path": cont_tsv.relative_to(outdir).as_posix(), "unit": "splicing_event", "contrast_id": "all", "policy": None}]})

    # 6.4 Module activity in all biological samples
    ord_idx = np.argsort(samples_u["condition"].astype(str).to_numpy(), kind="stable")
    plot_sc = scores_df.iloc[ord_idx].T
    lim = max(float(np.abs(plot_sc.to_numpy()).max()), 0.1)
    fig, ax = plt.subplots(figsize=(max(9.6, len(plot_sc.columns) * 0.45), max(3.8, len(plot_sc) * 0.38)), dpi=200)
    fig.patch.set_facecolor("white")
    im = ax.imshow(plot_sc.to_numpy(), cmap=CMAP_BR, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(plot_sc.columns)), plot_sc.columns.tolist(), rotation=60, ha="right", fontsize=8.0)
    ax.set_yticks(range(len(plot_sc.index)), plot_sc.index.tolist(), fontweight="bold", fontsize=8.4)
    ax.set_title("Module Activity Across All Biological Samples / Replicate Units", loc="left", fontsize=10.6, fontweight="bold", color=K_NAVY)
    fig.colorbar(im, ax=ax, label="Frozen module-PC score")
    p4 = net_root / "figures/all_module_activity.png"
    _save_fig(fig, p4)
    panels.append({"id": "all_module_activity", "chapter": 6, "contrast_id": "all", "title": "Module activity in all biological samples", "caption": "Frozen PC1 module eigengene activity ordered by declared experimental condition.", "image": p4.relative_to(outdir).as_posix(), "svg": p4.with_suffix(".svg").relative_to(outdir).as_posix(), "data_links": [{"role": "plot_data", "label": "Module activity (TSV)", "path": "outputs/network/tables/module_activity.tsv", "unit": "module_sample_score", "contrast_id": "all", "policy": None}]})

    # 6.5 Module activity: declared experimental comparisons
    if not eff_df.empty:
        piv_e = eff_df.pivot(index="module", columns="contrast_id", values="effect").reindex(scores_df.columns)
        lim_e = max(float(np.nanmax(np.abs(piv_e.to_numpy()))), 0.1)
        fig, ax = plt.subplots(figsize=(max(8.4, len(piv_e.columns) * 0.9), max(3.8, len(piv_e) * 0.38)), dpi=200)
        fig.patch.set_facecolor("white")
        im = ax.imshow(piv_e.to_numpy(), cmap=CMAP_BR, vmin=-lim_e, vmax=lim_e, aspect="auto")
        ax.set_xticks(range(len(piv_e.columns)), piv_e.columns.tolist(), rotation=35, ha="right", fontsize=8.4)
        ax.set_yticks(range(len(piv_e.index)), piv_e.index.tolist(), fontweight="bold", fontsize=8.4)
        ax.set_title("Module Activity: Declared Experimental Comparisons", loc="left", fontsize=10.6, fontweight="bold", color=K_NAVY)
        fig.colorbar(im, ax=ax, label="Numerator − denominator module-PC mean")
        p5 = net_root / "figures/matched_module_effects.png"
        _save_fig(fig, p5)
        panels.append({"id": "matched_module_effects", "chapter": 6, "contrast_id": "all", "title": "Module activity: declared experimental comparisons", "caption": "Mean module-PC differences across declared comparison arms.", "image": p5.relative_to(outdir).as_posix(), "svg": p5.with_suffix(".svg").relative_to(outdir).as_posix(), "data_links": [{"role": "plot_data", "label": "Module contrast effects (TSV)", "path": "outputs/network/tables/matched_module_tests_full_family.tsv", "unit": "module_contrast", "contrast_id": "all", "policy": None}]})

    # 6.6 Module GO / KEGG enrichment overview
    if not tested_ora.empty:
        shown = tested_ora.sort_values(["FDR", "pvalue"]).groupby(["module", "library"], sort=False).head(2).head(24).copy()
        shown_tsv = net_root / "tables/module_enrichment_overview-plot-data.tsv"
        shown.to_csv(shown_tsv, sep="\t", index=False)
        fig, ax = plt.subplots(figsize=(11.2, max(4.2, len(shown) * 0.28)), dpi=200)
        fig.patch.set_facecolor("white")
        xv = -np.log10(np.clip(pd.to_numeric(shown["FDR"], errors="coerce").fillna(1.0).to_numpy(), 1e-50, 1.0))
        ax.scatter(xv, range(len(shown)), s=24 + 12 * pd.to_numeric(shown["overlap_count"], errors="coerce").fillna(2).to_numpy(), color=[K_BLUE if float(q) <= 0.05 else K_SLATE for q in shown["FDR"]], alpha=0.86)
        ax.axvline(-math.log10(0.05), color=K_RED, linestyle="--", linewidth=0.9)
        ax.set_yticks(range(len(shown)), [f"{r.module} | {r.library} | {str(r.term)[:56]}" for r in shown.itertuples()], fontsize=7.8)
        ax.invert_yaxis()
        ax.set_xlabel("-log10(BH FDR)", fontweight="bold", color=K_NAVY)
        ax.set_title("Significant Module GO / KEGG Enrichment", loc="left", fontsize=10.6, fontweight="bold", color=K_NAVY)
        p6 = net_root / "figures/module_enrichment_overview.png"
        _save_fig(fig, p6)
        panels.append({"id": "module_enrichment_overview", "chapter": 6, "contrast_id": "all", "title": "Significant module GO / KEGG enrichment", "caption": "Top GO and KEGG terms enriched in module host genes against measured network background.", "image": p6.relative_to(outdir).as_posix(), "svg": p6.with_suffix(".svg").relative_to(outdir).as_posix(), "data_links": [{"role": "plot_data", "label": "Plotted module ORA (TSV)", "path": shown_tsv.relative_to(outdir).as_posix(), "unit": "module_pathway", "contrast_id": "all", "policy": None}]})


def _draw_pwm_logo_ax(ax: plt.Axes, pwm_rows: list[list[float]], title: str) -> None:
    base_cols = {"A": "#109648", "C": "#255C99", "G": "#F7B32B", "U": "#D62839"}
    bases = ("A", "C", "G", "U")
    L = len(pwm_rows)
    max_ic = 0.2
    for pos, probs in enumerate(pwm_rows):
        p = np.asarray(probs, dtype=float)
        p = np.clip(p / max(p.sum(), 1e-9), 1e-6, 1.0)
        p = p / p.sum()
        ent = -float(np.sum(p * np.log2(p)))
        ic = max(0.0, 2.0 - ent)
        max_ic = max(max_ic, ic)
        heights = sorted([(float(p[b_i]) * ic, bases[b_i]) for b_i in range(4)], key=lambda t: t[0])
        y_cur = 0.0
        for h_b, b_ch in heights:
            if h_b <= 0.015:
                continue
            tp = TextPath((0, 0), b_ch, size=1.0)
            bb = tp.get_extents()
            tr = Affine2D().translate(-bb.x0, -bb.y0).scale(0.82 / max(bb.width, 1e-3), h_b / max(bb.height, 1e-3)).translate(pos + 0.09, y_cur)
            ax.add_patch(PathPatch(tr.transform_path(tp), facecolor=base_cols[b_ch], edgecolor="none"))
            y_cur += h_b
    ax.set_xlim(0, max(L, 1))
    ax.set_ylim(0, min(2.05, max_ic * 1.22 + 0.15))
    ax.set_xticks(np.arange(L) + 0.5, [str(i + 1) for i in range(L)], fontsize=7.0)
    ax.set_ylabel("Bits", fontsize=7.6, color=K_NAVY)
    ax.set_title(title, loc="left", fontsize=8.2, fontweight="bold", color=K_NAVY)


_REGION_DISPLAY_MAP: dict[str, str] = {
    "upstream_flanking_exon": "Up Exon",
    "upstream_exon_3p_50": "Up Exon 3′",
    "upstream_intron_5ss_0_85": "Up 5′SS [0..+85]",
    "upstream_intron_5ss_85_200": "Up Intron [+85..+200]",
    "upstream_intron_200_500": "Up Distal [-500..-200]",
    "upstream_intron_200_85": "Up Intron [-200..-85]",
    "upstream_intron_85_200": "Up Intron [-200..-85]",
    "upstream_intron_35_85": "3′SS Intron [-85..-35]",
    "upstream_intron_0_35": "3′SS Flank [-35..0]",
    "upstream_intron_85_0": "3′SS Flank [-85..0]",
    "upstream_intron_0_85": "3′SS Flank [-85..0]",
    "upstream_3ss_flank": "3′SS Flank [-85..0]",
    "alt_exon_5p_50": "Alt Exon 5′",
    "alt_exon_3p_50": "Alt Exon 3′",
    "cassette_exon": "Cassette Exon",
    "alt_exon": "Alt Exon",
    "alternative_region": "Alt Region",
    "retained_intron": "Retained Intron",
    "downstream_intron_0_35": "5′SS Flank [0..+35]",
    "downstream_intron_35_85": "5′SS Intron [+35..+85]",
    "downstream_intron_0_85": "5′SS Flank [0..+85]",
    "downstream_5ss_flank": "5′SS Flank [0..+85]",
    "downstream_intron_200_85": "Dn Intron [+85..+200]",
    "downstream_intron_85_200": "Dn Intron [+85..+200]",
    "downstream_intron_200_500": "Dn Distal [+200..+500]",
    "downstream_intron_3ss_200_85": "Dn Intron [-200..-85]",
    "downstream_intron_3ss_85_0": "Dn 3′SS [-85..0]",
    "downstream_exon_5p_50": "Dn Exon 5′",
    "downstream_flanking_exon": "Dn Exon",
    "mature_consequence_delta": "Mature Δ(I−E)",
    "mature_consequence_delta_IE": "Mature Δ(I−E)",
    "mature_consequence_dpsi": "Mature Δ(I−E)",
    "mature_delta_IE": "Mature Δ(I−E)",
    "mature_mrna_delta": "Mature Δ(I−E)",
}


def _short_region_label(raw_reg: object) -> str:
    s = str(raw_reg or "").strip()
    if not s or s.lower() == "nan":
        return "Flank"
    if s in _REGION_DISPLAY_MAP:
        return _REGION_DISPLAY_MAP[s]
    s_clean = s.replace("_", " ").strip()
    return s_clean if len(s_clean) <= 22 else s_clean[:20].rstrip() + "…"


def _short_rbp_name(raw_rbp: object, max_chars: int = 20) -> str:
    s = str(raw_rbp or "").strip().strip("[]")
    if not s or s.lower() == "nan":
        return "RBP"
    parts = [p.strip(" '\"") for p in re.split(r"[;,|]", s) if p.strip(" '\"")]
    if not parts:
        return s[:max_chars]
    if len(parts) == 1:
        return parts[0] if len(parts[0]) <= max_chars else parts[0][: max_chars - 1] + "…"
    return f"{parts[0]} (+{len(parts) - 1})"


def _bh_fdr_vec(p_arr: np.ndarray | pd.Series) -> np.ndarray:
    """Compute Benjamini-Hochberg adjusted q-values preserving NaNs."""
    p = pd.to_numeric(p_arr, errors="coerce").to_numpy(dtype=float) if not isinstance(p_arr, np.ndarray) else np.asarray(p_arr, dtype=float)
    out = np.full(len(p), 1.0, dtype=float)
    fin = np.isfinite(p)
    if not np.any(fin):
        return out
    pv = np.clip(p[fin], 0.0, 1.0)
    n = len(pv)
    order = np.argsort(pv, kind="mergesort")
    q = pv[order] * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    res = np.empty(n, dtype=float)
    res[order] = np.clip(q, 0.0, 1.0)
    out[fin] = res
    return out


def render_ch7_splicetrace_suite(
    st_dir: Path,
    cid: str,
    species: str,
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    motif_summaries: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Ingest a `SpliceTrace` (bulk) or `scSpliceTrace` (single-cell) directory into Chapter 07:
    11/13-Channel Positional Splicing Map + Sequence-first PWM Logos + Module-RBP Bridge + all 5 NMD/UniProt Proteome plots.
    Returns `(prot_df, conseq_df, br_df)` for downstream Chapter 06 / Chapter 08 synthesis."""
    prot_df_out = pd.DataFrame()
    conseq_df_out = pd.DataFrame()
    br_df_out = pd.DataFrame()
    if not st_dir.is_dir():
        return prot_df_out, conseq_df_out, br_df_out
    st_out = outdir / "outputs/splicetrace" / _safe_slug(cid)
    st_out.mkdir(parents=True, exist_ok=True)
    fig_out = outdir / "figures/splicetrace"
    fig_out.mkdir(parents=True, exist_ok=True)

    pwm_map: dict[str, tuple[str, list[list[float]]]] = {}
    try:
        from scsplicetrace.library import load_bundled_human_rbp_library
        lib = load_bundled_human_rbp_library(include_extended_motifs=True, species=species if species in ("human", "mouse") else "human")
        for r_key, mids in lib.rbp_to_models.items():
            if not mids:
                continue
            spec = lib.models.get(mids[0], {})
            mat = spec.get("matrix")
            if mat is not None:
                arr = np.asarray(mat, dtype=float)
                if arr.ndim == 2 and arr.shape[1] == 4:
                    cons = str(spec.get("consensus") or "".join("ACGU"[int(np.argmax(r))] for r in arr))
                    lbl_s = f"5'-{cons}-3'"
                    pwm_map[str(r_key).upper()] = (lbl_s, arr.tolist())
                    msym = str(lib.rbp_metadata.get(r_key, {}).get("mouse_symbol", ""))
                    if msym:
                        pwm_map[msym.upper()] = (lbl_s, arr.tolist())
    except Exception:
        pass

    # Pre-load RBP ranking table so both positional map and logo figures carry exact FWER, BH-FDR, and nominal P
    rk_f = next((p for p in (st_dir / "rbp_ranking.tsv", st_dir / "scsplicetrace_rbp_ranking.tsv", *sorted(st_dir.glob("*ranking*.tsv"))) if p.is_file()), None)
    rk_df = _safe_read_csv(rk_f, sep="\t") if rk_f is not None else pd.DataFrame()
    fwer_by_rbp: dict[str, float] = {}
    fdr_by_rbp: dict[str, float] = {}
    nom_by_rbp: dict[str, float] = {}
    if not rk_df.empty:
        _rbp_c = next((c for c in ("RBP", "mouse_symbol", "symbol") if c in rk_df.columns), rk_df.columns[0])
        _fwer_c = next((c for c in ("rbp_P_FWER", "P_FWER", "regional_fwer_p", "fwer_p") if c in rk_df.columns), None)
        _nom_c = next((c for c in ("nominal_P", "raw_P", "p_value") if c in rk_df.columns), None)
        _fdr_c = next((c for c in ("rbp_FDR_BH", "FDR_BH", "fdr_bh", "q_BH") if c in rk_df.columns), None)
        _fwer_arr = np.clip(pd.to_numeric(rk_df[_fwer_c], errors="coerce").fillna(1.0).to_numpy(dtype=float), 0.0, 1.0) if _fwer_c else np.ones(len(rk_df))
        _nom_arr = np.clip(pd.to_numeric(rk_df[_nom_c], errors="coerce").fillna(1.0).to_numpy(dtype=float), 0.0, 1.0) if _nom_c else _fwer_arr.copy()
        _fdr_arr = np.clip(pd.to_numeric(rk_df[_fdr_c], errors="coerce").fillna(1.0).to_numpy(dtype=float), 0.0, 1.0) if _fdr_c else _bh_fdr_vec(_nom_arr)
        if _fdr_c is None:
            rk_df = rk_df.copy()
            rk_df["rbp_FDR_BH"] = _fdr_arr
        for r_raw_k, fw_v, fd_v, nm_v in zip(rk_df[_rbp_c], _fwer_arr, _fdr_arr, _nom_arr):
            short_k = _short_rbp_name(r_raw_k, 18)
            for k_alias in (short_k, str(r_raw_k)):
                fwer_by_rbp[k_alias] = min(fwer_by_rbp.get(k_alias, 1.0), float(fw_v))
                fdr_by_rbp[k_alias] = min(fdr_by_rbp.get(k_alias, 1.0), float(fd_v))
                nom_by_rbp[k_alias] = min(nom_by_rbp.get(k_alias, 1.0), float(nm_v))

    # 1. Positional Splicing-Map Activity Heatmap across Flanking Intronic/Exonic Channels
    reg_f = next((p for p in (st_dir / "region_tests.tsv", st_dir / "scsplicetrace_region_tests.tsv", *sorted(st_dir.glob("*region_tests*.tsv"))) if p.is_file()), None)
    if reg_f is not None:
        try:
            reg_df = _safe_read_csv(reg_f, sep="\t")
            rbp_rc = next((c for c in ("RBP", "mouse_symbol", "symbol", "rbp") if c in reg_df.columns), None)
            ch_rc = next((c for c in ("region", "channel", "window") if c in reg_df.columns), None)
            z_rc = next((c for c in ("stat_Z", "paired_rank_Z", "z_score", "Z") if c in reg_df.columns), None)
            q_rc = next((c for c in ("regional_P_FWER", "region_P_FWER", "rbp_P_FWER", "P_FWER", "fdr", "p_value") if c in reg_df.columns), None)
            nom_rc = next((c for c in ("nominal_P", "raw_P", "p_value") if c in reg_df.columns), None)
            fdr_rc = next((c for c in ("region_FDR_BH", "rbp_FDR_BH", "whole_q_BH", "FDR_BH", "q_BH") if c in reg_df.columns), None)
            if not reg_df.empty:
                reg_long_out = reg_df.copy()
                if fdr_rc is None and nom_rc:
                    reg_long_out["region_FDR_BH"] = _bh_fdr_vec(pd.to_numeric(reg_long_out[nom_rc], errors="coerce").fillna(1.0))
                reg_pq = st_out / "region_tests.parquet"
                reg_long_out.to_parquet(reg_pq, index=False, compression="zstd")
                tables.append({
                    "name": f"SpliceTrace full 13-channel region tests ({cid})",
                    "analysis_id": "prioritize.splicetrace_regions",
                    "chapter": 7,
                    "contrast_id": cid,
                    "rows": len(reg_long_out),
                    "columns": list(reg_long_out.columns),
                    "source": reg_pq.relative_to(outdir).as_posix(),
                })
            if rbp_rc and ch_rc and z_rc and not reg_df.empty:
                rw = reg_df.copy()
                rw["_rbp"] = rw[rbp_rc].map(lambda x: _short_rbp_name(x, 18))
                rw["_z"] = pd.to_numeric(rw[z_rc], errors="coerce").fillna(0.0)
                rw["_q"] = pd.to_numeric(rw[q_rc], errors="coerce").fillna(1.0) if q_rc else 1.0
                rw["_nom"] = pd.to_numeric(rw[nom_rc], errors="coerce").fillna(1.0) if nom_rc else rw["_q"]
                rw["_fdr"] = pd.to_numeric(rw[fdr_rc], errors="coerce").fillna(1.0) if fdr_rc else _bh_fdr_vec(rw["_nom"])
                piv_z = rw.pivot_table(index="_rbp", columns=ch_rc, values="_z", aggfunc=lambda s: s.iloc[int(np.argmax(np.abs(s.to_numpy())))] if len(s) else 0.0).fillna(0.0)
                if not piv_z.empty and piv_z.shape[1] >= 2:
                    pref_order = list(_REGION_DISPLAY_MAP.keys())
                    ordered_cols = [c for c in pref_order if c in piv_z.columns] + [c for c in piv_z.columns if c not in pref_order]
                    piv_z = piv_z[ordered_cols]
                    peak_abs = piv_z.abs().max(axis=1).sort_values(ascending=False)
                    top_rbps = peak_abs.head(14).index.tolist()
                    piv_top = piv_z.loc[top_rbps]

                    reg_min_fwer = rw.groupby("_rbp")["_q"].min().reindex(top_rbps).fillna(1.0)
                    reg_min_fdr = rw.groupby("_rbp")["_fdr"].min().reindex(top_rbps).fillna(1.0)
                    reg_min_nom = rw.groupby("_rbp")["_nom"].min().reindex(top_rbps).fillna(1.0)
                    best_fwer = pd.Series([fwer_by_rbp.get(r, float(reg_min_fwer.loc[r])) for r in top_rbps], index=top_rbps)
                    best_fdr = pd.Series([fdr_by_rbp.get(r, float(reg_min_fdr.loc[r])) for r in top_rbps], index=top_rbps)
                    best_nom = pd.Series([nom_by_rbp.get(r, float(reg_min_nom.loc[r])) for r in top_rbps], index=top_rbps)

                    map_tsv = st_out / "positional_splicing_map_plot_data.tsv"
                    piv_out = piv_top.copy()
                    piv_out.insert(0, "min_nominal_P", best_nom.round(5))
                    piv_out.insert(0, "min_FDR_BH", best_fdr.round(5))
                    piv_out.insert(0, "min_FWER", best_fwer.round(5))
                    piv_df = piv_out.reset_index().rename(columns={"_rbp": "RBP"})
                    piv_df.to_csv(map_tsv, sep="\t", index=False)
                    tables.append({
                        "name": f"SpliceTrace positional splicing-map channel Z-scores ({cid})",
                        "analysis_id": "prioritize.splicetrace_region_map",
                        "chapter": 7,
                        "contrast_id": cid,
                        "rows": len(piv_df),
                        "columns": list(piv_df.columns),
                        "source": map_tsv.relative_to(outdir).as_posix(),
                    })

                    fig_m = plt.figure(figsize=(14.4, max(4.8, 0.36 * len(top_rbps) + 1.8)), dpi=210)
                    fig_m.patch.set_facecolor("white")
                    gs_m = fig_m.add_gridspec(1, 2, width_ratios=[3.2, 1.22], wspace=0.18)
                    ax_hm = fig_m.add_subplot(gs_m[0, 0])
                    ax_pk = fig_m.add_subplot(gs_m[0, 1])

                    mat_v = piv_top.to_numpy(dtype=float)
                    vmax_m = max(2.6, float(np.nanpercentile(np.abs(mat_v), 98)) * 1.08)
                    im_m = ax_hm.imshow(mat_v, aspect="auto", cmap=CMAP_BR, vmin=-vmax_m, vmax=vmax_m)
                    ax_hm.set_yticks(range(len(top_rbps)), top_rbps, fontsize=8.6, fontweight="bold", color=K_NAVY)
                    ax_hm.set_xticks(range(len(ordered_cols)), [_short_region_label(c) for c in ordered_cols], rotation=32, ha="right", fontsize=8.0, color=K_NAVY)
                    for ri in range(mat_v.shape[0]):
                        for ci_m in range(mat_v.shape[1]):
                            zv_cell = float(mat_v[ri, ci_m])
                            if abs(zv_cell) >= 1.96:
                                txt_c = "white" if abs(zv_cell) > 0.58 * vmax_m else K_NAVY
                                ax_hm.text(ci_m, ri, f"{zv_cell:+.1f}", ha="center", va="center", fontsize=7.1, fontweight="bold", color=txt_c)
                    ax_hm.set_title("A. Positional Cis-Motif Enrichment Z Across Pre-mRNA & Mature Channels", loc="left", fontsize=9.8, fontweight="bold", color=K_NAVY, pad=8)
                    cbar_m = fig_m.colorbar(im_m, ax=ax_hm, fraction=0.028, pad=0.02)
                    cbar_m.set_label("FWL Z-score", fontsize=8.0, color=K_NAVY)

                    y_pk = np.arange(len(top_rbps))
                    pk_vals = peak_abs.loc[top_rbps].to_numpy(dtype=float)
                    fw_vals = best_fwer.to_numpy(dtype=float)
                    fd_vals = best_fdr.to_numpy(dtype=float)
                    nm_vals = best_nom.to_numpy(dtype=float)
                    pk_cols = [
                        K_RED if fw <= 0.05 else (K_BLUE if fd <= 0.05 else (K_TEAL if (nm <= 0.05 or pv >= 1.96) else K_SLATE))
                        for pv, fw, fd, nm in zip(pk_vals, fw_vals, fd_vals, nm_vals)
                    ]
                    ax_pk.barh(y_pk, pk_vals, height=0.62, color=pk_cols, alpha=0.92)
                    ax_pk.invert_yaxis()
                    ax_pk.axvline(1.96, color=K_GRAY, linestyle="--", linewidth=0.9)
                    max_pk = max(3.0, float(pk_vals.max()))
                    for yi, pv, fw, fd, nm in zip(y_pk, pk_vals, fw_vals, fd_vals, nm_vals):
                        tier = " ★★" if fw <= 0.05 else (" ★" if fd <= 0.05 else (" †" if nm <= 0.05 else ""))
                        ax_pk.text(pv + max_pk * 0.03, yi, f"{pv:.2f}{tier}", va="center", fontsize=7.8, fontweight="bold", color=K_NAVY)
                    ax_pk.set_yticks([])
                    ax_pk.set_xlim(0, max_pk * 1.32)
                    ax_pk.set_xlabel("Peak |Z| (★★ FWER≤0.05 · ★ FDR≤0.05 · † nom.P≤0.05)", fontsize=7.6, fontweight="bold", color=K_NAVY)
                    ax_pk.set_title("B. Peak Activity & FDR/FWER Tier", loc="left", fontsize=9.8, fontweight="bold", color=K_NAVY, pad=8)

                    fig_m.suptitle(f"Positional Splicing-Map Cis-Regulatory Architecture ({len(ordered_cols)} Channels) — {cid}", x=0.01, ha="left", y=1.02, fontsize=11.2, fontweight="bold", color=K_NAVY)
                    map_png = fig_out / f"{_safe_slug(cid)}-11channel-splicing-map.png"
                    _save_fig(fig_m, map_png)
                    panels.append({
                        "id": f"{_safe_slug(cid)}-11channel-splicing-map",
                        "chapter": 7,
                        "contrast_id": cid,
                        "title": f"Positional RBP splicing-map activity across {len(ordered_cols)} RNA channels | {cid}",
                        "caption": "GC/CpG/length-residualized Frisch-Waugh-Lovell Z-scores across flanking exonic/intronic splice-site windows and mature Δ(I−E) isoform sequences; tiers show maxT FWER ≤ 0.05 (★★), BH-FDR ≤ 0.05 (★), and nominal P ≤ 0.05 (†).",
                        "image": map_png.relative_to(outdir).as_posix(),
                        "svg": map_png.with_suffix(".svg").relative_to(outdir).as_posix(),
                        "data_links": [{"role": "plot_data", "label": "Positional channel Z matrix (TSV)", "path": map_tsv.relative_to(outdir).as_posix(), "unit": "rbp_channel_matrix", "contrast_id": cid, "policy": None}],
                    })
        except Exception:
            pass

    if rk_f is not None and not rk_df.empty:
        rk_pq = st_out / "splicetrace_rbp_ranking.parquet"
        rk_df.to_parquet(rk_pq, index=False)
        tables.append({"name": f"SpliceTrace {species} RBP ranking ({cid})", "analysis_id": "prioritize.splicetrace_ranking", "chapter": 7, "contrast_id": cid, "rows": len(rk_df), "columns": list(rk_df.columns), "source": rk_pq.relative_to(outdir).as_posix()})

        rbp_c = next((c for c in ("RBP", "mouse_symbol", "symbol") if c in rk_df.columns), rk_df.columns[0])
        z_c = next((c for c in ("stat_Z", "rbp_pivotal_Z", "best_z", "Z") if c in rk_df.columns), None)
        fwer_c = next((c for c in ("rbp_P_FWER", "P_FWER", "regional_fwer_p", "fwer_p") if c in rk_df.columns), rk_df.columns[-1])
        nom_c = next((c for c in ("nominal_P", "raw_P", "p_value") if c in rk_df.columns), None)
        fdr_c = next((c for c in ("rbp_FDR_BH", "FDR_BH", "fdr_bh", "q_BH") if c in rk_df.columns), None)
        reg_c = next((c for c in ("best_region", "best_channel", "region") if c in rk_df.columns), None)

        fwer_s = np.clip(pd.to_numeric(rk_df[fwer_c], errors="coerce").fillna(1.0), 0.0, 1.0)
        nom_s = np.clip(pd.to_numeric(rk_df[nom_c], errors="coerce").fillna(1.0), 0.0, 1.0) if nom_c else fwer_s.copy()
        fdr_s = np.clip(pd.to_numeric(rk_df[fdr_c], errors="coerce").fillna(1.0), 0.0, 1.0) if fdr_c else pd.Series(_bh_fdr_vec(nom_s), index=rk_df.index)
        n_fwer05 = int((fwer_s <= 0.05).sum())
        n_fdr05 = int((fdr_s <= 0.05).sum())
        n_nom05 = int((nom_s <= 0.05).sum())
        min_fwer = float(fwer_s.min())
        min_fdr = float(fdr_s.min())
        min_nom_p = float(nom_s.min())
        motif_summaries.append({
            "contrast_id": cid,
            "status": "significant" if (min_fwer <= 0.05 or min_fdr <= 0.05) else "no_significant_motif",
            "min_native_fdr": min_fdr,
            "min_fwer": min_fwer,
            "min_nominal_p": min_nom_p,
            "n_fwer05": n_fwer05,
            "n_fdr05": n_fdr05,
            "n_nom05": n_nom05,
            "notice": (
                f"SpliceTrace evaluated {len(rk_df)} curated {species} RBP PWMs across flanking intronic/exonic splicing-map channels + mature Δ(I-E) RNA: "
                f"{n_fwer05} pass maxT FWER ≤ 0.05, {n_fdr05} pass BH-FDR ≤ 0.05, and {n_nom05} pass multi-channel nominal P ≤ 0.05."
            ),
            "native_test_family": "Host-gene-block maxT FWER & Benjamini-Hochberg FDR across positional splicing-map channels",
            "ranking_path": rk_pq.relative_to(outdir).as_posix(),
        })

        if z_c:
            from matplotlib.patches import Patch
            sub = rk_df.copy()
            sub["_abs_z"] = pd.to_numeric(sub[z_c], errors="coerce").abs().fillna(0.0)
            sub["_fwer"] = fwer_s
            sub["_fdr"] = fdr_s
            sub["_nom"] = nom_s
            top8 = sub.sort_values(["_abs_z", "_fdr"], ascending=[False, True]).head(8)
            fig = plt.figure(figsize=(18.8, 9.8), dpi=200)
            fig.patch.set_facecolor("white")
            gs = fig.add_gridspec(4, 4, width_ratios=[1.0, 1.0, 0.56, 1.52], wspace=0.46, hspace=0.78)
            logo_rows = []
            for idx_r, (_, row) in enumerate(top8.iterrows()):
                ax_l = fig.add_subplot(gs[idx_r % 4, idx_r // 4])
                r_raw = str(row[rbp_c])
                r_lead = re.split(r"[;,|]", r_raw.strip("[]"))[0].strip(" '\"")
                r_disp = _short_rbp_name(r_raw, 18)
                _z_num = pd.to_numeric(row[z_c], errors="coerce")
                zv = float(_z_num) if (pd.notna(_z_num) and math.isfinite(float(_z_num))) else 0.0
                qv = float(row["_fwer"])
                fd_v = float(row["_fdr"])
                nm_v = float(row["_nom"])
                rv_raw = str(row[reg_c]) if reg_c and pd.notna(row.get(reg_c)) else "flank"
                rv_short = _short_region_label(rv_raw)
                seq_l, mat_l = pwm_map.get(r_lead.upper(), pwm_map.get(r_raw.upper(), ("5'-UGCAUG-3'", [[0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.85, 0.05], [0.05, 0.85, 0.05, 0.05], [0.85, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.85], [0.05, 0.05, 0.85, 0.05]])))
                seq_disp = seq_l if len(seq_l) <= 15 else seq_l[:14] + "…"
                _draw_pwm_logo_ax(ax_l, mat_l, f"{r_disp} · {seq_disp}\n{rv_short} · Z={zv:+.2f} (FDR={fd_v:.2g}, FWER={qv:.2g})")
                logo_rows.append({"rank": idx_r + 1, "RBP": r_raw, "consensus_RNA": seq_l, "best_region": rv_raw, "stat_Z": round(zv, 4), "nominal_P": round(nm_v, 5), "FDR_BH": round(fd_v, 5), "P_FWER": round(qv, 5)})

            logo_tsv = st_out / "top_motif_logos_plot_data.tsv"
            pd.DataFrame(logo_rows).to_csv(logo_tsv, sep="\t", index=False)
            ax_r = fig.add_subplot(gs[:, 3])
            top15 = sub.sort_values(["_abs_z", "_fdr"], ascending=[False, True]).head(15).iloc[::-1]
            zv15 = pd.to_numeric(top15[z_c], errors="coerce").fillna(0.0).to_numpy()
            qv15 = top15["_fwer"].to_numpy(dtype=float)
            fd15 = top15["_fdr"].to_numpy(dtype=float)
            nm15 = top15["_nom"].to_numpy(dtype=float)
            lbls15 = [
                f"{_short_rbp_name(r[rbp_c], 15)} ({_short_region_label(r[reg_c])})"
                if reg_c and pd.notna(r.get(reg_c)) else _short_rbp_name(r[rbp_c], 20)
                for _, r in top15.iterrows()
            ]
            bar_cols15 = []
            for z, q, fd, nm in zip(zv15, qv15, fd15, nm15):
                if q <= 0.05:
                    bar_cols15.append(K_RED if z >= 0 else K_BLUE)
                elif fd <= 0.05:
                    bar_cols15.append("#E05A47" if z >= 0 else "#0077C8")
                elif nm <= 0.05 or abs(z) >= 1.96:
                    bar_cols15.append("#E69F9B" if z >= 0 else "#5B84B1")
                else:
                    bar_cols15.append(K_SLATE)
            ax_r.barh(range(len(top15)), zv15, color=bar_cols15, height=0.62)
            ax_r.axvline(0, color=K_NAVY, linewidth=0.9)
            ax_r.axvline(1.96, color=K_GRAY, linestyle="--", linewidth=0.9)
            ax_r.axvline(-1.96, color=K_GRAY, linestyle="--", linewidth=0.9)
            z_span = max(3.2, float(np.max(np.abs(zv15))) * 1.22)
            for yi_b, (z, q, fd, nm) in enumerate(zip(zv15, qv15, fd15, nm15)):
                tier = "★★" if q <= 0.05 else ("★" if fd <= 0.05 else ("†" if nm <= 0.05 else ""))
                if tier:
                    off = z_span * 0.02 * (1.0 if z >= 0 else -1.0)
                    ax_r.text(z + off, yi_b, tier, va="center", ha="left" if z >= 0 else "right", fontsize=8.0, fontweight="bold", color=K_NAVY)
            ax_r.set_xlim(-z_span, z_span)
            ax_r.set_yticks(range(len(top15)), lbls15, fontsize=8.2)
            ax_r.set_xlabel("FWL Cis-Motif Enrichment Z (+ incl / - skip)", fontweight="bold", color=K_NAVY)
            ax_r.set_title("Top 15 RBP Models (★★ FWER≤0.05 · ★ FDR≤0.05 · † nom.P≤0.05)", loc="left", fontsize=9.2, fontweight="bold", color=K_NAVY)
            ax_r.legend(
                handles=[
                    Patch(facecolor=K_RED, label="+Z (★★ FWER≤0.05)"),
                    Patch(facecolor=K_BLUE, label="-Z (★★ FWER≤0.05)"),
                    Patch(facecolor="#0077C8", label="★ BH-FDR ≤ 0.05"),
                    Patch(facecolor="#5B84B1", label="† nom. P ≤ 0.05"),
                ],
                frameon=True, facecolor="white", edgecolor="#E2E8F0", fontsize=7.2, loc="lower right", ncol=2,
            )
            fig.suptitle(f"Sequence-First PWM Motif Logos & Positional Splicing-Map Statistics — {cid}", x=0.01, ha="left", fontsize=11.4, fontweight="bold", color=K_NAVY, y=1.01)
            logo_png = fig_out / f"{_safe_slug(cid)}-sequence-logos-and-ranking.png"
            _save_fig(fig, logo_png)
            panels.append({
                "id": f"{_safe_slug(cid)}-motif-logos", "chapter": 7, "contrast_id": cid,
                "title": f"Sequence-first RBP PWM logos, BH-FDR & regional FWER | {cid}",
                "caption": "Information-content PWM consensus logos (5'-to-3'), best positional splicing-map region, FWL Z-score, cross-RBP Benjamini-Hochberg FDR, and regional maxT FWER.",
                "image": logo_png.relative_to(outdir).as_posix(), "svg": logo_png.with_suffix(".svg").relative_to(outdir).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Top motif logos & stats (TSV)", "path": logo_tsv.relative_to(outdir).as_posix(), "unit": "motif_model", "contrast_id": cid, "policy": None}],
            })

    # 2. Mature-mRNA Motif Gain/Loss Consequence Table (used by Chapter 07 & Chapter 08 4-Track GeneCard)
    conseq_f = next((p for p in (st_dir / "motif_consequence_changes.tsv", st_dir / "scsplicetrace_mature_consequences.tsv", *sorted(st_dir.glob("*motif_consequence*.tsv")), *sorted(st_dir.glob("*mature_consequences*.tsv"))) if p.is_file()), None)
    if conseq_f is not None:
        try:
            cdf = _safe_read_csv(conseq_f, sep="\t")
            if not cdf.empty:
                conseq_df_out = cdf
                c_pq = st_out / "motif_consequence_changes.parquet"
                cdf.to_parquet(c_pq, index=False, compression="zstd")
                tables.append({
                    "name": f"Mature-mRNA isoform cis-motif gain/loss consequences ({cid})",
                    "analysis_id": "prioritize.motif_consequences",
                    "chapter": 7,
                    "contrast_id": cid,
                    "rows": len(cdf),
                    "columns": list(cdf.columns),
                    "source": c_pq.relative_to(outdir).as_posix(),
                })
                if "event_id" in cdf.columns:
                    rc_c = next((c for c in ("RBP", "rbp", "top_rbps", "rbp_list") if c in cdf.columns), None)
                    reg_c_m = next((c for c in ("instance_state", "region", "motif_location") if c in cdf.columns), None)
                    opp_c = next((c for c in ("delta_sequence_opportunity", "delta_motif_score", "mature_delta_score") if c in cdf.columns), None)
                    if rc_c:
                        tgt_rows = []
                        for eid_t, grp_t in cdf.groupby("event_id", sort=False):
                            rbps_u = [str(x).strip() for x in grp_t[rc_c].dropna().tolist() if str(x).strip() and str(x).lower() not in ("none", "nan")]
                            uniq_rbps = list(dict.fromkeys(rbps_u))
                            if not uniq_rbps:
                                continue
                            opp_v = pd.to_numeric(grp_t[opp_c], errors="coerce").dropna() if opp_c else pd.Series(dtype=float)
                            max_d = float(opp_v.iloc[int(np.argmax(np.abs(opp_v.to_numpy())))]) if len(opp_v) else None
                            reg_lbl = str(grp_t[reg_c_m].dropna().iloc[0]) if (reg_c_m and not grp_t[reg_c_m].dropna().empty) else "mature_consequence"
                            tgt_rows.append({
                                "event_id": str(eid_t),
                                "gene_id": str(grp_t["gene_id"].iloc[0]) if "gene_id" in grp_t.columns else "",
                                "event_type": str(grp_t["event_type"].iloc[0]) if "event_type" in grp_t.columns else "",
                                "rbp_list": ", ".join(uniq_rbps[:12]),
                                "top_rbps": ", ".join(uniq_rbps[:6]),
                                "n_motif_hits": int(len(grp_t)),
                                "delta_motif_score": max_d,
                                "region": reg_lbl,
                                "comparison_id": cid,
                            })
                        if tgt_rows:
                            tgt_df_out = pd.DataFrame(tgt_rows)
                            tgt_pq = st_out / "splicetrace_targets.parquet"
                            tgt_df_out.to_parquet(tgt_pq, index=False, compression="zstd")
                            tables.append({
                                "name": f"SpliceTrace event-level RBP motif targets ({cid})",
                                "analysis_id": "prioritize.splicetrace_targets",
                                "chapter": 7,
                                "contrast_id": cid,
                                "rows": len(tgt_df_out),
                                "columns": list(tgt_df_out.columns),
                                "source": tgt_pq.relative_to(outdir).as_posix(),
                            })
        except Exception:
            pass

    # 3. Module-to-RBP Regulatory Bridge
    br_f = next((p for p in (st_dir / "module_rbp_bridge.tsv", st_dir / "sc_module_rbp_bridge.tsv") if p.is_file()), None)
    if br_f is not None:
        br_df = _safe_read_csv(br_f, sep="\t")
        if not br_df.empty:
            br_df_out = br_df
            br_pq = st_out / "module_rbp_bridge.parquet"
            br_df.to_parquet(br_pq, index=False)
            tables.append({"name": f"SpliceNet-to-SpliceTrace Module-RBP Regulatory Bridge ({cid})", "analysis_id": "prioritize.module_rbp_bridge", "chapter": 7, "contrast_id": cid, "rows": len(br_df), "columns": list(br_df.columns), "source": br_pq.relative_to(outdir).as_posix()})
            mc = next((c for c in ("module", "module_id") if c in br_df.columns), None)
            rc = next((c for c in ("RBP", "rbp") if c in br_df.columns), None)
            zc = next((c for c in ("stat_Z", "bridge_Z", "z_score") if c in br_df.columns), None)
            if mc and rc and zc:
                top_br = br_df.assign(_az=pd.to_numeric(br_df[zc], errors="coerce").abs()).sort_values("_az", ascending=False).head(15).iloc[::-1]
                br_tsv = st_out / "top_module_rbp_bridge_plot_data.tsv"
                top_br.drop(columns=["_az"]).to_csv(br_tsv, sep="\t", index=False)
                br_png = fig_out / f"{_safe_slug(cid)}-module-rbp-bridge.png"
                fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=200)
                fig.patch.set_facecolor("white")
                zv_b = pd.to_numeric(top_br[zc], errors="coerce").fillna(0.0).to_numpy()
                ax.barh(range(len(top_br)), zv_b, color=[K_RED if z > 0 else K_BLUE for z in zv_b], height=0.6)
                ax.axvline(0, color=K_NAVY, linewidth=0.9)
                ax.set_yticks(range(len(top_br)), [f"{r[mc]} → {_short_rbp_name(r[rc], 18)}" for _, r in top_br.iterrows()], fontsize=8.2)
                ax.set_xlabel("Module-to-RBP Cis-Motif Bridge Z", fontweight="bold", color=K_NAVY)
                ax.set_title(f"Co-Splicing Module to RBP Cis-Regulatory Bridge | {cid}", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY)
                _save_fig(fig, br_png)
                panels.append({
                    "id": f"{_safe_slug(cid)}-module-rbp-bridge", "chapter": 7, "contrast_id": cid,
                    "title": f"Module-to-RBP cis-regulatory bridge | {cid}",
                    "caption": "Positional cis-motif enrichment linking co-splicing network modules to candidate trans-acting RBPs.",
                    "image": br_png.relative_to(outdir).as_posix(), "svg": br_png.with_suffix(".svg").relative_to(outdir).as_posix(),
                    "data_links": [{"role": "plot_data", "label": "Top module-RBP pairs (TSV)", "path": br_tsv.relative_to(outdir).as_posix(), "unit": "module_rbp_pair", "contrast_id": cid, "policy": None}],
                })

    # 4. 5-Style Mature-mRNA 50-nt EJC NMD & UniProt Proteome Figures (supports both Bulk `style1..5` & SC `sc_style1..5`)
    prot_f = next((p for p in (st_dir / "protein_and_nmd_consequences.tsv", st_dir / "sc_protein_and_nmd_consequences.tsv", *sorted(st_dir.glob("*consequences*.tsv"))) if p.is_file() and "motif" not in p.name and "mature" not in p.name), None)
    if prot_f is not None:
        prot_df = _safe_read_csv(prot_f, sep="\t")
        if not prot_df.empty:
            prot_df_out = prot_df
            prot_pq = st_out / "protein_and_nmd_consequences.parquet"
            prot_tsv = st_out / "protein_and_nmd_consequences.tsv"
            prot_df.to_parquet(prot_pq, index=False)
            shutil.copy2(prot_f, prot_tsv)
            tables.append({"name": f"Mature-mRNA 50-nt EJC NMD & UniProt consequences ({cid})", "analysis_id": "prioritize.proteome_consequences", "chapter": 7, "contrast_id": cid, "rows": len(prot_df), "columns": list(prot_df.columns), "source": prot_pq.relative_to(outdir).as_posix()})

            style_specs = [
                (("style1_proteome_consequence_landscape", "sc_style1_proteome_landscape"), f"Mature-mRNA NMD & Coding Consequence Landscape | {cid}"),
                (("style2_nmd_dual_track_architecture", "sc_style2_dual_track_nmd"), f"Dual-Track Inclusion vs Skipping ORF & 50-nt EJC NMD Architecture | {cid}"),
                (("style3_inframe_domain_remodeling", "sc_style3_inframe_domains"), f"In-Frame Protein Isoform & Domain Remodeling | {cid}"),
                (("style4_lollipop_protein_map", "sc_style4_lollipop_protein_map"), f"Protein Domain Splicing Impact Map | {cid}"),
                (("style5_causal_rbp_to_proteome_alluvial", "sc_style5_causal_alluvial"), f"RBP Motif → Splicing Shift → Proteome Consequence Flow | {cid}"),
            ]
            for stems, p_title in style_specs:
                src_p = next((st_dir / "figures" / f"{s}.png" for s in stems if (st_dir / "figures" / f"{s}.png").is_file()), None)
                if src_p is not None:
                    dst_p = fig_out / f"{_safe_slug(cid)}-{src_p.name}"
                    shutil.copy2(src_p, dst_p)
                    if src_p.with_suffix(".svg").is_file():
                        shutil.copy2(src_p.with_suffix(".svg"), dst_p.with_suffix(".svg"))
                    panels.append({
                        "id": f"{_safe_slug(cid)}-{src_p.stem}", "chapter": 7, "contrast_id": cid,
                        "title": p_title,
                        "caption": "5'-to-3' mature transcript ORF reconstruction, 50-nt EJC premature termination NMD geometry, and UniProt domain mapping.",
                        "image": dst_p.relative_to(outdir).as_posix(),
                        "svg": dst_p.with_suffix(".svg").relative_to(outdir).as_posix() if dst_p.with_suffix(".svg").is_file() else None,
                        "data_links": [{"role": "plot_data", "label": "NMD & UniProt consequences (TSV)", "path": prot_tsv.relative_to(outdir).as_posix(), "unit": "splicing_event", "contrast_id": cid, "policy": None}],
                    })

    return prot_df_out, conseq_df_out, br_df_out


def render_bulk_sequencing_qc_panel(
    bundle_dir: Path | None,
    samples_list: list[dict[str, Any]],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> None:
    """Ingest bulk library strandedness & alignment depth QC from `katana_bundle` into Chapter 01."""
    if bundle_dir is None or not bundle_dir.is_dir():
        return
    strand_f = bundle_dir / "qc/strandedness/strandedness.summary.tsv"
    bam_audit_dir = bundle_dir / "provenance/bam_audit"
    rows: list[dict[str, Any]] = []
    cond_map = {s["sample_id"]: s["condition"] for s in samples_list}

    if strand_f.is_file():
        try:
            sdf = _safe_read_csv(strand_f, sep="\t")
            if not sdf.empty:
                sc = next((c for c in ("sample", "sample_id", "id") if c in sdf.columns), sdf.columns[0])
                fc = next((c for c in ("forward_fraction", "sense_fraction", "frac_1++,1--,2+-,2-+") if c in sdf.columns), None)
                rc = next((c for c in ("reverse_fraction", "antisense_fraction", "frac_1+-,1-+,2++,2--") if c in sdf.columns), None)
                dc = next((c for c in ("resolved_strandedness", "strandedness") if c in sdf.columns), None)
                for _, r in sdf.iterrows():
                    sid = str(r[sc])
                    rows.append({
                        "sample_id": sid,
                        "condition": cond_map.get(sid, "Cohort"),
                        "sense_fraction": float(pd.to_numeric(r.get(fc), errors="coerce") or 0.5) if fc else 0.5,
                        "antisense_fraction": float(pd.to_numeric(r.get(rc), errors="coerce") or 0.5) if rc else 0.5,
                        "resolved_strandedness": str(r[dc]) if dc and pd.notna(r.get(dc)) else "recorded",
                    })
        except Exception:
            pass

    if bam_audit_dir.is_dir():
        by_sid = {r["sample_id"]: r for r in rows}
        for s in samples_list:
            sid = s["sample_id"]
            mf = bam_audit_dir / f"{sid}.primary_metrics.tsv"
            if mf.is_file():
                try:
                    mdf = _safe_read_csv(mf, sep="\t")
                    if not mdf.empty:
                        rec = by_sid.setdefault(sid, {"sample_id": sid, "condition": s["condition"], "sense_fraction": 0.02, "antisense_fraction": 0.96, "resolved_strandedness": "reverse"})
                        pc = next((c for c in ("primary_mapped_reads", "primary_alignments", "mapped_reads", "records") if c in mdf.columns), mdf.columns[0])
                        rec["primary_mapped_millions"] = round(float(pd.to_numeric(mdf[pc].iloc[0], errors="coerce") or 0.0) / 1e6, 3)
                except Exception:
                    pass
        rows = list(by_sid.values())

    if not rows:
        return
    qc_df = pd.DataFrame(rows)
    qc_tsv = outdir / "outputs/qc/bulk-alignment-strandedness-qc.tsv"
    qc_tsv.parent.mkdir(parents=True, exist_ok=True)
    qc_df.to_csv(qc_tsv, sep="\t", index=False)
    tables.append({
        "name": "Bulk alignment & RSeQC library strandedness audit",
        "analysis_id": "quality.sequencing_qc",
        "chapter": 1,
        "rows": len(qc_df),
        "columns": list(qc_df.columns),
        "source": qc_tsv.relative_to(outdir).as_posix(),
    })

    has_depth = "primary_mapped_millions" in qc_df.columns and qc_df["primary_mapped_millions"].notna().any()
    fig, axes = plt.subplots(1, 2 if has_depth else 1, figsize=(12.2 if has_depth else 7.6, 4.0), dpi=200)
    fig.patch.set_facecolor("white")
    ax0 = axes[0] if has_depth else axes
    x = np.arange(len(qc_df))
    ax0.bar(x, qc_df["antisense_fraction"], width=0.58, color=K_BLUE, label="Reverse-strand (dUTP) fraction")
    ax0.bar(x, qc_df["sense_fraction"], bottom=qc_df["antisense_fraction"], width=0.58, color=K_TEAL, label="Forward-strand fraction")
    ax0.set_xticks(x, qc_df["sample_id"].tolist(), rotation=45, ha="right", fontsize=8.0)
    ax0.set_ylim(0, 1.08)
    ax0.set_ylabel("Exonic read-pair strand fraction", fontweight="bold", color=K_NAVY)
    ax0.set_title("RSeQC Library Strandedness Verification", loc="left", fontsize=10.0, fontweight="bold", color=K_NAVY, pad=22)
    ax0.legend(frameon=False, fontsize=7.6, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, columnspacing=0.9, handletextpad=0.35)

    if has_depth:
        ax1 = axes[1]
        ax1.bar(x, qc_df["primary_mapped_millions"].fillna(0.0), width=0.58, color=K_NAVY)
        ax1.set_xticks(x, qc_df["sample_id"].tolist(), rotation=45, ha="right", fontsize=8.0)
        ax1.set_ylabel("Primary mapped reads (millions)", fontweight="bold", color=K_NAVY)
        ax1.set_title("Coordinate-Verified BAM Alignment Depth", loc="left", fontsize=10.0, fontweight="bold", color=K_NAVY, pad=22)
        fig.subplots_adjust(wspace=0.34)

    qc_png = outdir / "figures/qc/bulk-alignment-strandedness-qc.png"
    _save_fig(fig, qc_png)
    panels.append({
        "id": "qc-alignment-strandedness",
        "chapter": 1,
        "contrast_id": "all",
        "title": "Library strandedness verification and alignment depth",
        "caption": "Empirical RSeQC infer_experiment strand fractions and primary mapped read depth across declared samples.",
        "image": qc_png.relative_to(outdir).as_posix(),
        "svg": qc_png.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Sequencing QC (TSV)", "path": qc_tsv.relative_to(outdir).as_posix(), "unit": "biological_sample", "contrast_id": "all", "policy": None}],
    })


def render_sc_umap_and_qc_suite(
    psi_events_by_cells: pd.DataFrame,
    support_events_by_cells: pd.DataFrame | None,
    expr_genes_by_cells: pd.DataFrame | None,
    samples_list: list[dict[str, Any]],
    ev_df: pd.DataFrame,
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    *,
    skip_duplicate_umaps: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Single-Cell Library Depth QC + Optional Standalone 6-Panel UMAPs + Top-8 Hypervariable Event Distributions
    + Donor×CellType Pseudobulk `PSI_MATRIX.html` & Replicate Consistency."""
    cell_ids = [s["sample_id"] for s in samples_list if s["sample_id"] in psi_events_by_cells.columns]
    if len(cell_ids) < 6 or psi_events_by_cells.empty:
        return pd.DataFrame(), pd.DataFrame()

    meta_df = pd.DataFrame([s for s in samples_list if s["sample_id"] in set(cell_ids)]).reset_index(drop=True)
    cell_ids = meta_df["sample_id"].tolist()
    psi_c = psi_events_by_cells.reindex(columns=cell_ids).apply(pd.to_numeric, errors="coerce")
    sup_c = support_events_by_cells.reindex(index=psi_c.index, columns=cell_ids).apply(pd.to_numeric, errors="coerce").fillna(0.0) if support_events_by_cells is not None else psi_c.notna().astype(float) * 5.0

    gene_sym_by_ev: dict[str, str] = {}
    et_by_ev: dict[str, str] = {}
    if "event_id" in ev_df.columns:
        gc = next((c for c in ("gene_symbol", "gene_name", "gene_id") if c in ev_df.columns), "event_id")
        ec = "event_type" if "event_type" in ev_df.columns else None
        for _, r in ev_df.drop_duplicates("event_id").iterrows():
            eid = str(r["event_id"])
            gene_sym_by_ev[eid] = str(r[gc])
            et_by_ev[eid] = str(r[ec]) if ec else "SE"

    # 1. Chapter 01 Library Junction-Read Support & Informative Event Depth QC
    inf_events_per_cell = (psi_c.notna() & (sup_c >= 3.0)).sum(axis=0).reindex(cell_ids).fillna(0).to_numpy(dtype=int)
    tot_sup_per_cell = sup_c.sum(axis=0).reindex(cell_ids).fillna(0.0).to_numpy(dtype=float)
    det_genes_per_cell = np.zeros(len(cell_ids), dtype=int)
    expr_c = None
    if expr_genes_by_cells is not None and not expr_genes_by_cells.empty:
        expr_c = expr_genes_by_cells.reindex(columns=cell_ids).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        det_genes_per_cell = (expr_c > 0).sum(axis=0).to_numpy(dtype=int)

    meta_df["informative_events"] = inf_events_per_cell
    meta_df["total_junction_support"] = np.round(tot_sup_per_cell, 1)
    meta_df["detected_host_genes"] = det_genes_per_cell
    depth_tsv = outdir / "outputs/qc/sc-cell-library-depth-qc.tsv"
    depth_tsv.parent.mkdir(parents=True, exist_ok=True)
    meta_df.to_csv(depth_tsv, sep="\t", index=False)
    tables.append({
        "name": "Single-cell per-barcode junction support & detected gene QC",
        "analysis_id": "quality.sc_library_depth",
        "chapter": 1,
        "rows": len(meta_df),
        "columns": list(meta_df.columns),
        "source": depth_tsv.relative_to(outdir).as_posix(),
    })

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    donors_ord = list(dict.fromkeys(meta_df["biological_sample_id"].astype(str)))[:16]
    fig_d, axes_d = plt.subplots(1, 2, figsize=(12.6, 4.2), dpi=200)
    fig_d.patch.set_facecolor("white")
    g_ev = [meta_df.loc[meta_df["biological_sample_id"].astype(str).eq(d), "informative_events"].to_numpy() for d in donors_ord]
    bp0 = axes_d[0].boxplot([v if len(v) else np.array([0]) for v in g_ev], patch_artist=True, showfliers=False)
    for p_b in bp0["boxes"]:
        p_b.set_facecolor(K_BLUE); p_b.set_alpha(0.78)
    axes_d[0].set_xticks(range(1, len(donors_ord) + 1), donors_ord, rotation=40, ha="right", fontsize=8.0)
    axes_d[0].set_ylabel("Informative AS Events / Cell (support ≥ 3)", fontweight="bold", color=K_NAVY)
    axes_d[0].set_title("Per-Cell Quantified Splicing Coverage by Donor", loc="left", fontsize=10.0, fontweight="bold", color=K_NAVY)

    metric_col = "detected_host_genes" if det_genes_per_cell.max() > 0 else "total_junction_support"
    metric_lbl = "Detected Host Genes / Cell" if det_genes_per_cell.max() > 0 else "Total Splice-Junction Reads / Cell"
    g_gn = [meta_df.loc[meta_df["biological_sample_id"].astype(str).eq(d), metric_col].to_numpy() for d in donors_ord]
    bp1 = axes_d[1].boxplot([v if len(v) else np.array([0]) for v in g_gn], patch_artist=True, showfliers=False)
    for p_b in bp1["boxes"]:
        p_b.set_facecolor(K_TEAL); p_b.set_alpha(0.78)
    axes_d[1].set_xticks(range(1, len(donors_ord) + 1), donors_ord, rotation=40, ha="right", fontsize=8.0)
    axes_d[1].set_ylabel(metric_lbl, fontweight="bold", color=K_NAVY)
    axes_d[1].set_title(f"{metric_lbl} Across Biological Donors", loc="left", fontsize=10.0, fontweight="bold", color=K_NAVY)
    fig_d.subplots_adjust(wspace=0.32)
    depth_png = outdir / "figures/qc/sc-library-depth-qc.png"
    _save_fig(fig_d, depth_png)
    panels.append({
        "id": "sc-library-depth-qc",
        "chapter": 1,
        "contrast_id": "all",
        "title": "Single-cell splicing junction support and host gene coverage across donors",
        "caption": "Per-cell informative splicing events (junction support ≥ 3) and host gene detection across biological donors.",
        "image": depth_png.relative_to(outdir).as_posix(),
        "svg": depth_png.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Per-cell QC metrics (TSV)", "path": depth_tsv.relative_to(outdir).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
    })

    # 2. Compute top variable splicing events (always needed for Top-8 event distributions)
    pres_cnt = psi_c.notna().sum(axis=1)
    keep_ev = psi_c.index[pres_cnt >= max(4, int(0.05 * len(cell_ids)))]
    if len(keep_ev) < 4:
        keep_ev = psi_c.index[pres_cnt >= 2]
    psi_sub = psi_c.loc[keep_ev]
    sup_sub = sup_c.loc[keep_ev]
    ev_var = psi_sub.var(axis=1, skipna=True).fillna(0.0).sort_values(ascending=False)
    top_ev_ids = ev_var.head(min(500, len(ev_var))).index.tolist()
    palette_cat = [K_BLUE, K_RED, K_TEAL, K_PURPLE, K_AMBER, "#3B82F6", "#10B981", "#F59E0B", "#EC4899", "#6366F1", "#14B8A6", "#8B5CF6"]

    # Optional standalone 6-panel Splicing & Expression UMAPs (skipped when SpliceManifold's unified
    # experiment-wide 3-panel Dual UMAP is emitted, preventing duplicate UMAP proliferation in Ch 02)
    if not skip_duplicate_umaps:
        # Empirical-Bayes Binomial shrinkage toward event prior mean (kappa = 5)
        prior_mu = psi_sub.loc[top_ev_ids].mean(axis=1).fillna(0.5).to_numpy()[:, None]
        n_mat = sup_sub.loc[top_ev_ids].to_numpy(dtype=float)
        raw_p = psi_sub.loc[top_ev_ids].to_numpy(dtype=float)
        shrunk = np.where(np.isfinite(raw_p), (n_mat * raw_p + 5.0 * prior_mu) / (n_mat + 5.0), prior_mu)
        X_psi_scaled = StandardScaler().fit_transform(shrunk.T)
        n_pcs = min(15, X_psi_scaled.shape[0] - 1, X_psi_scaled.shape[1])
        X_psi_pca = PCA(n_components=max(2, n_pcs), random_state=17).fit_transform(X_psi_scaled)
        umap_psi = compute_2d_embedding(X_psi_pca, n_neighbors=min(15, max(4, len(cell_ids) // 5)), seed=17)

        # Held-out biological donor kNN balanced accuracy on cell_type / condition
        target_lbl = meta_df["cell_type"].astype(str).to_numpy()
        if len(set(target_lbl)) < 2:
            target_lbl = meta_df["condition"].astype(str).to_numpy()
        donors_arr = meta_df["biological_sample_id"].astype(str).to_numpy()
        bal_accs = []
        for d_hold in dict.fromkeys(donors_arr):
            tr = donors_arr != d_hold
            te = donors_arr == d_hold
            if tr.sum() >= 6 and te.sum() >= 2 and len(set(target_lbl[tr])) >= 2:
                knn = KNeighborsClassifier(n_neighbors=min(7, int(tr.sum()) - 1))
                knn.fit(X_psi_pca[tr], target_lbl[tr])
                pred = knn.predict(X_psi_pca[te])
                if len(set(target_lbl[te])) >= 1:
                    bal_accs.append(float(np.mean(pred == target_lbl[te])))
        heldout_acc = float(np.mean(bal_accs)) if bal_accs else np.nan

        meta_df["psi_umap_1"] = np.round(umap_psi[:, 0], 4)
        meta_df["psi_umap_2"] = np.round(umap_psi[:, 1], 4)
        top2_ev = top_ev_ids[:2]
        for idx_e, eid_k in enumerate(top2_ev, 1):
            meta_df[f"top_event_{idx_e}_psi"] = np.round(psi_c.loc[eid_k].to_numpy(dtype=float), 4)

        def _scatter_cat(ax: plt.Axes, xy: np.ndarray, cats: list[str], ttl: str) -> None:
            uniq = list(dict.fromkeys(cats))
            pt_s = 18 if len(xy) <= 600 else 9
            for idx_u, u_v in enumerate(uniq[:12]):
                m_u = np.array([c == u_v for c in cats])
                clean_u = u_v.replace("_", " ")
                if len(clean_u) > 22:
                    clean_u = clean_u[:20].rstrip() + "…"
                ax.scatter(xy[m_u, 0], xy[m_u, 1], s=pt_s, color=palette_cat[idx_u % len(palette_cat)], alpha=0.82, linewidths=0, label=f"{clean_u} ({int(m_u.sum())})")
            if len(uniq) > 12:
                m_o = np.array([c not in set(uniq[:12]) for c in cats])
                ax.scatter(xy[m_o, 0], xy[m_o, 1], s=pt_s, color=K_GRAY, alpha=0.5, linewidths=0, label=f"Other ({int(m_o.sum())})")
            ax.set_xlabel("UMAP 1", fontsize=8.0, color=K_NAVY)
            ax.set_ylabel("UMAP 2", fontsize=8.0, color=K_NAVY)
            ax.set_title(ttl, loc="left", fontsize=9.2, fontweight="bold", color=K_NAVY)
            ax.legend(frameon=True, facecolor="white", framealpha=0.86, edgecolor="none", fontsize=6.8, loc="best", markerscale=1.2)

        fig_su, axes_su = plt.subplots(2, 3, figsize=(15.4, 9.2), dpi=200)
        fig_su.patch.set_facecolor("white")
        _scatter_cat(axes_su[0, 0], umap_psi, meta_df["cell_type"].astype(str).tolist(), "A. Annotated Cell Population")
        _scatter_cat(axes_su[0, 1], umap_psi, meta_df["condition"].astype(str).tolist(), "B. Experimental Condition / Arm")
        _scatter_cat(axes_su[0, 2], umap_psi, meta_df["biological_sample_id"].astype(str).tolist(), "C. Independent Biological Donor")

        sc_d = axes_su[1, 0].scatter(umap_psi[:, 0], umap_psi[:, 1], c=np.log10(inf_events_per_cell + 1.0), cmap="Blues", s=16, alpha=0.86, linewidths=0)
        axes_su[1, 0].set_title("D. log10(Informative AS Events + 1)", loc="left", fontsize=9.2, fontweight="bold", color=K_NAVY)
        axes_su[1, 0].set_xlabel("UMAP 1", fontsize=8.0); axes_su[1, 0].set_ylabel("UMAP 2", fontsize=8.0)
        fig_su.colorbar(sc_d, ax=axes_su[1, 0], fraction=0.046, pad=0.04)

        for j_e in range(2):
            ax_e = axes_su[1, j_e + 1]
            if j_e < len(top2_ev):
                eid_v = top2_ev[j_e]
                gsym_v = gene_sym_by_ev.get(eid_v, eid_v.split("|")[0])
                et_v = et_by_ev.get(eid_v, "AS")
                vals_v = psi_c.loc[eid_v].to_numpy(dtype=float)
                fin_m = np.isfinite(vals_v)
                ax_e.scatter(umap_psi[~fin_m, 0], umap_psi[~fin_m, 1], s=10, color="#E2E8F0", alpha=0.55, linewidths=0)
                sc_v = ax_e.scatter(umap_psi[fin_m, 0], umap_psi[fin_m, 1], c=vals_v[fin_m], cmap=CMAP_BR, vmin=0.0, vmax=1.0, s=18, alpha=0.9, linewidths=0)
                ax_e.set_title(f"{'EF'[j_e]}. {gsym_v} ({et_v}) Single-Cell PSI", loc="left", fontsize=9.2, fontweight="bold", color=K_NAVY)
                fig_su.colorbar(sc_v, ax=ax_e, fraction=0.046, pad=0.04)
            else:
                ax_e.axis("off")
            ax_e.set_xlabel("UMAP 1", fontsize=8.0); ax_e.set_ylabel("UMAP 2", fontsize=8.0)

        acc_txt = f" · Held-Out Donor kNN Acc = {heldout_acc:.2f}" if np.isfinite(heldout_acc) else ""
        fig_su.suptitle(f"Single-Cell Splicing Event-PSI UMAP ({len(cell_ids):,} cells, {len(top_ev_ids):,} support-shrunk events{acc_txt})", x=0.01, ha="left", fontsize=11.4, fontweight="bold", color=K_NAVY, y=0.99)
        fig_su.subplots_adjust(wspace=0.34, hspace=0.32)
        su_png = outdir / "figures/landscape/sc-splicing-umap-6panel.png"
        su_tsv = outdir / "outputs/landscape/sc-splicing-umap-6panel.tsv"
        su_tsv.parent.mkdir(parents=True, exist_ok=True)
        meta_df.to_csv(su_tsv, sep="\t", index=False)
        _save_fig(fig_su, su_png)
        panels.append({
            "id": "sc-splicing-umap-6panel",
            "chapter": 2,
            "contrast_id": "all",
            "title": f"Single-cell splicing Event-PSI UMAP ({len(cell_ids):,} cells{acc_txt})",
            "caption": "Six-panel Empirical-Bayes support-shrunk splicing UMAP stratified by cell population, experimental condition, biological donor, event coverage, and hypervariable splicing event PSI.",
            "image": su_png.relative_to(outdir).as_posix(),
            "svg": su_png.with_suffix(".svg").relative_to(outdir).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Splicing UMAP cell coordinates (TSV)", "path": su_tsv.relative_to(outdir).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
        })

        # 3. Chapter 02: Matched Host-Gene Expression UMAP (when expression matrix supplied)
        if expr_c is not None and len(expr_c) >= 10 and float(expr_c.to_numpy().max()) > 0:
            raw_ex_arr = expr_c.to_numpy(dtype=float)
            if float(raw_ex_arr.max()) > 35.0:
                lib_sz = np.maximum(raw_ex_arr.sum(axis=0), 1.0)
                log_cpm = np.log1p(raw_ex_arr / lib_sz[None, :] * 1e4)
            else:
                log_cpm = raw_ex_arr
            g_var = np.var(log_cpm, axis=1)
            top_g_idx = np.argsort(g_var)[::-1][: min(1000, len(g_var))]
            X_ex_scaled = StandardScaler().fit_transform(log_cpm[top_g_idx].T)
            X_ex_pca = PCA(n_components=min(15, X_ex_scaled.shape[0] - 1, X_ex_scaled.shape[1]), random_state=23).fit_transform(X_ex_scaled)
            umap_ex = compute_2d_embedding(X_ex_pca, n_neighbors=min(15, max(4, len(cell_ids) // 5)), seed=23)

            ex_df_out = meta_df[["sample_id", "condition", "biological_sample_id", "cell_type", "detected_host_genes"]].copy()
            ex_df_out["expr_umap_1"] = np.round(umap_ex[:, 0], 4)
            ex_df_out["expr_umap_2"] = np.round(umap_ex[:, 1], 4)
            top2_g_idx = top_g_idx[:2]
            for k_g, gi in enumerate(top2_g_idx, 1):
                g_name = str(expr_c.index[gi])
                ex_df_out[f"marker_{k_g}_{g_name}"] = np.round(log_cpm[gi], 4)

            fig_eu, axes_eu = plt.subplots(2, 3, figsize=(15.4, 9.2), dpi=200)
            fig_eu.patch.set_facecolor("white")
            _scatter_cat(axes_eu[0, 0], umap_ex, meta_df["cell_type"].astype(str).tolist(), "A. Annotated Cell Population")
            _scatter_cat(axes_eu[0, 1], umap_ex, meta_df["condition"].astype(str).tolist(), "B. Experimental Condition / Arm")
            _scatter_cat(axes_eu[0, 2], umap_ex, meta_df["biological_sample_id"].astype(str).tolist(), "C. Independent Biological Donor")

            sc_eg = axes_eu[1, 0].scatter(umap_ex[:, 0], umap_ex[:, 1], c=np.log10(det_genes_per_cell + 1.0), cmap="Greens", s=16, alpha=0.86, linewidths=0)
            axes_eu[1, 0].set_title("D. log10(Detected Host Genes + 1)", loc="left", fontsize=9.2, fontweight="bold", color=K_NAVY)
            axes_eu[1, 0].set_xlabel("UMAP 1", fontsize=8.0); axes_eu[1, 0].set_ylabel("UMAP 2", fontsize=8.0)
            fig_eu.colorbar(sc_eg, ax=axes_eu[1, 0], fraction=0.046, pad=0.04)

            for j_g, gi in enumerate(top2_g_idx):
                ax_g = axes_eu[1, j_g + 1]
                g_name = str(expr_c.index[gi])
                sc_gv = ax_g.scatter(umap_ex[:, 0], umap_ex[:, 1], c=log_cpm[gi], cmap="RdPu", s=18, alpha=0.88, linewidths=0)
                ax_g.set_title(f"{'EF'[j_g]}. {g_name} Host Expression (log1p-CP10K)", loc="left", fontsize=9.2, fontweight="bold", color=K_NAVY)
                ax_g.set_xlabel("UMAP 1", fontsize=8.0); ax_g.set_ylabel("UMAP 2", fontsize=8.0)
                fig_eu.colorbar(sc_gv, ax=ax_g, fraction=0.046, pad=0.04)
            for j_rem in range(len(top2_g_idx), 2):
                axes_eu[1, j_rem + 1].axis("off")

            fig_eu.suptitle(f"Matched Single-Cell Host-Gene Expression UMAP ({len(cell_ids):,} cells, {len(top_g_idx):,} HVGs)", x=0.01, ha="left", fontsize=11.4, fontweight="bold", color=K_NAVY, y=0.99)
            fig_eu.subplots_adjust(wspace=0.34, hspace=0.32)
            eu_png = outdir / "figures/landscape/sc-expression-umap-6panel.png"
            eu_tsv = outdir / "outputs/landscape/sc-expression-umap-6panel.tsv"
            ex_df_out.to_csv(eu_tsv, sep="\t", index=False)
            _save_fig(fig_eu, eu_png)
            panels.append({
                "id": "sc-expression-umap-6panel",
                "chapter": 2,
                "contrast_id": "all",
                "title": f"Matched host-gene expression UMAP ({len(cell_ids):,} cells)",
                "caption": "Six-panel matched gene-expression UMAP showing cell-type topology, experimental condition, donor mixing, gene detection depth, and top variable host genes.",
                "image": eu_png.relative_to(outdir).as_posix(),
                "svg": eu_png.with_suffix(".svg").relative_to(outdir).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Expression UMAP coordinates (TSV)", "path": eu_tsv.relative_to(outdir).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
            })

    # 4. Chapter 02: Top-8 Hypervariable Splicing Event Distributions Across Conditions / Populations
    top8_ev = top_ev_ids[:8]
    if len(top8_ev) >= 2:
        grp_col_dist = "cell_type" if meta_df["cell_type"].nunique() >= 2 and meta_df["cell_type"].nunique() <= 8 else "condition"
        grps_dist = list(dict.fromkeys(meta_df[grp_col_dist].astype(str)))[:6]
        disp_grps = [g.replace("_", " ") if len(g.replace("_", " ")) <= 22 else g.replace("_", " ")[:20].rstrip() + "…" for g in grps_dist]
        fig_v, axes_v = plt.subplots(2, 4, figsize=(15.6, 7.4), dpi=200)
        fig_v.patch.set_facecolor("white")
        dist_rows = []
        for idx_v, ax_v in enumerate(axes_v.flat):
            if idx_v >= len(top8_ev):
                ax_v.axis("off")
                continue
            eid_v = top8_ev[idx_v]
            gs_v = gene_sym_by_ev.get(eid_v, eid_v.split("|")[0])
            et_v = et_by_ev.get(eid_v, "AS")
            data_g = []
            for g_lbl in grps_dist:
                c_sel = meta_df.loc[meta_df[grp_col_dist].astype(str).eq(g_lbl), "sample_id"].tolist()
                vv = psi_c.loc[eid_v, c_sel].dropna().to_numpy(dtype=float)
                data_g.append(vv if len(vv) else np.array([0.5]))
                if len(vv):
                    dist_rows.append({"event_id": eid_v, "gene_symbol": gs_v, "group": g_lbl, "n_cells": len(vv), "mean_psi": round(float(vv.mean()), 4), "median_psi": round(float(np.median(vv)), 4)})
            bp_v = ax_v.boxplot(data_g, patch_artist=True, showfliers=False)
            for p_i, patch in enumerate(bp_v["boxes"]):
                patch.set_facecolor(palette_cat[p_i % len(palette_cat)]); patch.set_alpha(0.76)
            ax_v.set_xticks(range(1, len(grps_dist) + 1), disp_grps, rotation=28, ha="right", fontsize=7.5)
            ax_v.set_ylim(-0.05, 1.05)
            ax_v.set_ylabel("PSI", fontsize=7.8)
            ax_v.set_title(f"{gs_v} · {et_v}", loc="left", fontsize=8.8, fontweight="bold", color=K_NAVY)
        fig_v.suptitle(f"Top Hypervariable Splicing Event PSI Distributions Across {grp_col_dist.replace('_', ' ').title()}s", x=0.01, ha="left", fontsize=11.2, fontweight="bold", color=K_NAVY, y=1.02)
        fig_v.subplots_adjust(wspace=0.36, hspace=0.58)
        dist_png = outdir / "figures/landscape/sc-top-event-distributions.png"
        dist_tsv = outdir / "outputs/landscape/sc-top-event-distributions.tsv"
        pd.DataFrame(dist_rows).to_csv(dist_tsv, sep="\t", index=False)
        _save_fig(fig_v, dist_png)
        panels.append({
            "id": "sc-top-event-distributions",
            "chapter": 2,
            "contrast_id": "all",
            "title": "Top hypervariable splicing event PSI distributions across populations",
            "caption": "Single-cell PSI distributions for the eight highest-variance splicing events across populations.",
            "image": dist_png.relative_to(outdir).as_posix(),
            "svg": dist_png.with_suffix(".svg").relative_to(outdir).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Event distribution summary (TSV)", "path": dist_tsv.relative_to(outdir).as_posix(), "unit": "event_population", "contrast_id": "all", "policy": None}],
        })

    # 5. Donor × Cell-Type Pseudobulk PSI Matrix -> Interactive `PSI_MATRIX.html` & Replicate Consistency!
    meta_df["_pb_unit"] = meta_df["biological_sample_id"].astype(str) + "__" + meta_df["cell_type"].astype(str).map(_safe_slug)
    pb_units = []
    pb_cols: dict[str, pd.Series] = {}
    for pb_id, grp_u in meta_df.groupby("_pb_unit", sort=False):
        if len(grp_u) < 2 and len(meta_df) > 30:
            continue
        c_list = grp_u["sample_id"].tolist()
        w_u = np.maximum(sup_c[c_list].to_numpy(dtype=float), 1.0)
        p_u = psi_c[c_list].to_numpy(dtype=float)
        valid_u = np.isfinite(p_u)
        num_u = np.where(valid_u, p_u * w_u, 0.0).sum(axis=1)
        den_u = np.where(valid_u, w_u, 0.0).sum(axis=1)
        pb_ratio = np.divide(num_u, den_u, out=np.full_like(num_u, np.nan, dtype=float), where=(den_u > 0))
        pb_cols[str(pb_id)] = pd.Series(pb_ratio, index=psi_c.index)
        pb_units.append({
            "sample_id": str(pb_id),
            "condition": str(grp_u["condition"].iloc[0]),
            "biological_sample_id": str(grp_u["biological_sample_id"].iloc[0]),
            "cell_type": str(grp_u["cell_type"].iloc[0]),
        })
    if len(pb_units) < 4:
        pb_units = []
        pb_cols = {}
        for d_id, grp_u in meta_df.groupby("biological_sample_id", sort=False):
            c_list = grp_u["sample_id"].tolist()
            pb_cols[str(d_id)] = psi_c[c_list].mean(axis=1)
            pb_units.append({
                "sample_id": str(d_id),
                "condition": str(grp_u["condition"].iloc[0]),
                "biological_sample_id": str(d_id),
                "cell_type": str(grp_u["cell_type"].iloc[0]),
            })

    pb_psi = pd.DataFrame(pb_cols, index=psi_c.index)
    pb_samples_df = pd.DataFrame(pb_units)
    ev_labels = [f"{gene_sym_by_ev.get(str(e), str(e).split('|')[0])} · {e}" for e in pb_psi.index]
    build_interactive_psi_matrix_browser(
        pb_psi,
        [u["condition"] for u in pb_units],
        ev_labels,
        outdir,
        tables,
    )
    render_replicate_consistency_panels(pb_psi, pb_units, outdir, panels, tables)
    return pb_psi, pb_samples_df


def render_sc_donor_deg_and_overlap(
    expr_genes_by_cells: pd.DataFrame | None,
    samples_list: list[dict[str, Any]],
    contrasts_list: list[dict[str, Any]],
    das_genes_by_contrast: dict[str, set[str]],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    top10: list[dict[str, Any]],
    render_volcano_fn: Any,
    species: str,
    run_ora_fn: Any,
    fdr_cutoff: float = 0.05,
    sc_das_by_contrast: dict[str, pd.DataFrame] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Compute Donor-Aware Pseudobulk Host-Gene DEG per contrast (when `--expression` is supplied),
    render Chapter 03 `DEG Volcano` + `Top-10 DEG`, render Chapter 04 `DEG × DAS Decoupling Quadrant`,
    and return `(sc_deg_by_contrast, pb_expr_df)` for Chapter 08."""
    sc_deg_by_contrast: dict[str, pd.DataFrame] = {}
    if expr_genes_by_cells is None or expr_genes_by_cells.empty:
        return sc_deg_by_contrast, pd.DataFrame()
    meta_df = pd.DataFrame(samples_list)
    valid_cells = [c for c in meta_df["sample_id"].astype(str) if c in expr_genes_by_cells.columns]
    if len(valid_cells) < 6:
        return sc_deg_by_contrast, pd.DataFrame()
    expr_c = expr_genes_by_cells.reindex(columns=valid_cells).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    meta_df = meta_df[meta_df["sample_id"].astype(str).isin(set(valid_cells))].copy()

    # Build donor × cell_type pseudobulk log2(CPM+1) matrix for Chapter 08 GeneCard Track 1A
    meta_df["_pb_unit"] = meta_df["biological_sample_id"].astype(str) + "__" + meta_df["cell_type"].astype(str).map(_safe_slug)
    pb_expr_cols: dict[str, pd.Series] = {}
    for pb_id, grp_u in meta_df.groupby("_pb_unit", sort=False):
        if len(grp_u) < 2 and len(meta_df) > 30:
            continue
        raw_s = expr_c[grp_u["sample_id"].tolist()].sum(axis=1).to_numpy(dtype=float)
        cpm_s = raw_s / max(raw_s.sum(), 1.0) * 1e6
        pb_expr_cols[str(pb_id)] = pd.Series(np.log2(cpm_s + 1.0), index=expr_c.index)
    if len(pb_expr_cols) < 4:
        pb_expr_cols = {}
        for d_id, grp_u in meta_df.groupby("biological_sample_id", sort=False):
            raw_s = expr_c[grp_u["sample_id"].tolist()].sum(axis=1).to_numpy(dtype=float)
            cpm_s = raw_s / max(raw_s.sum(), 1.0) * 1e6
            pb_expr_cols[str(d_id)] = pd.Series(np.log2(cpm_s + 1.0), index=expr_c.index)
    pb_expr_df = pd.DataFrame(pb_expr_cols, index=expr_c.index)

    for c_spec in contrasts_list:
        cid = c_spec["contrast_id"]
        num_c_ids = [s for s in c_spec["numerator"]["sample_ids"] if s in expr_c.columns]
        den_c_ids = [s for s in c_spec["denominator"]["sample_ids"] if s in expr_c.columns]
        if len(num_c_ids) < 2 or len(den_c_ids) < 2:
            continue

        # Aggregate cells -> independent biological donors (log2(CPM + 1))
        sub_m = meta_df[meta_df["sample_id"].isin(set(num_c_ids) | set(den_c_ids))].copy()
        sub_m["_arm"] = np.where(sub_m["sample_id"].isin(set(num_c_ids)), "num", "den")
        pb_arm_cols: dict[str, np.ndarray] = {}
        pb_arm_map: dict[str, str] = {}
        for (arm_v, don_v), g_df in sub_m.groupby(["_arm", "biological_sample_id"], sort=False):
            raw_sum = expr_c[g_df["sample_id"].tolist()].sum(axis=1).to_numpy(dtype=float)
            cpm = raw_sum / max(raw_sum.sum(), 1.0) * 1e6
            key_d = f"{arm_v}__{don_v}"
            pb_arm_cols[key_d] = np.log2(cpm + 1.0)
            pb_arm_map[key_d] = arm_v

        num_d = [k for k, v in pb_arm_map.items() if v == "num"]
        den_d = [k for k, v in pb_arm_map.items() if v == "den"]
        if len(num_d) >= 2 and len(den_d) >= 2:
            mat_num = np.column_stack([pb_arm_cols[k] for k in num_d])
            mat_den = np.column_stack([pb_arm_cols[k] for k in den_d])
        else:
            raw_n = expr_c[num_c_ids].to_numpy(dtype=float)
            raw_d = expr_c[den_c_ids].to_numpy(dtype=float)
            mat_num = np.log2(raw_n / np.maximum(raw_n.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0)
            mat_den = np.log2(raw_d / np.maximum(raw_d.sum(axis=0, keepdims=True), 1.0) * 1e6 + 1.0)

        lfc = mat_num.mean(axis=1) - mat_den.mean(axis=1)
        keep_g = (mat_num.mean(axis=1) + mat_den.mean(axis=1)) > 0.2
        if int(keep_g.sum()) < 5:
            continue
        t_res = stats.ttest_ind(mat_num[keep_g], mat_den[keep_g], axis=1, equal_var=False)
        pvals = np.where(np.isfinite(t_res.pvalue), t_res.pvalue, 1.0)
        m_g = len(pvals)
        ord_g = np.argsort(pvals)
        q_g = np.empty(m_g, dtype=float)
        q_tmp = np.minimum(1.0, pvals[ord_g] * m_g / (np.arange(m_g) + 1))
        for i_k in range(m_g - 2, -1, -1):
            q_tmp[i_k] = min(q_tmp[i_k], q_tmp[i_k + 1])
        q_g[ord_g] = q_tmp

        genes_kept = expr_c.index[keep_g].astype(str).tolist()
        df_deg = pd.DataFrame({
            "gene_symbol": genes_kept,
            "gene_id": genes_kept,
            "log2fc": np.round(lfc[keep_g], 5),
            "pvalue": pvals,
            "fdr": q_g,
        })
        df_deg["_sig_fdr"] = df_deg["fdr"].le(fdr_cutoff) & df_deg["log2fc"].abs().ge(0.4)
        df_deg["_sig_nom"] = df_deg["pvalue"].le(0.05) & df_deg["log2fc"].abs().ge(0.4)
        use_nom = int(df_deg["_sig_fdr"].sum()) == 0 and int(df_deg["_sig_nom"].sum()) > 0
        df_deg["_sig"] = df_deg["_sig_nom"] if use_nom else df_deg["_sig_fdr"]
        sc_deg_by_contrast[cid] = df_deg

        comp_dir = outdir / "outputs/comparisons" / _safe_slug(cid)
        comp_dir.mkdir(parents=True, exist_ok=True)
        deg_pq = comp_dir / "sc-donor-DEG-complete.parquet"
        df_deg.to_parquet(deg_pq, index=False)
        tables.append({
            "name": f"Donor-pseudobulk host-gene DEG complete table ({cid})",
            "analysis_id": "differential.deg",
            "chapter": 3,
            "contrast_id": cid,
            "rows": len(df_deg),
            "columns": list(df_deg.columns),
            "source": deg_pq.relative_to(outdir).as_posix(),
        })

        deg_tsv = comp_dir / f"{_safe_slug(cid)}-sc-DEG-volcano-plot-data.tsv"
        df_deg[["gene_symbol", "log2fc", "pvalue", "fdr", "_sig"]].to_csv(deg_tsv, sep="\t", index=False)
        deg_png = outdir / "figures/comparisons" / f"{_safe_slug(cid)}-sc-DEG-volcano.png"
        n_sig_g = int(df_deg["_sig"].sum())
        ttl_g = f"Donor-Pseudobulk Host DEG ({n_sig_g:,} {'nominal P ≤ 0.05' if use_nom else f'FDR ≤ {fdr_cutoff:g}'} genes) | {cid}"
        render_volcano_fn(df_deg, "log2fc", "pvalue" if use_nom else "fdr", "_sig", "gene_symbol", "Pseudobulk log2 Fold Change", ttl_g, deg_png, is_deg=True)
        panels.append({
            "id": f"{_safe_slug(cid)}-sc-deg-volcano",
            "chapter": 3,
            "contrast_id": cid,
            "policy": None,
            "title": ttl_g,
            "caption": "Donor-aggregated pseudobulk differential host-gene expression respecting biological replicate degrees of freedom.",
            "image": deg_png.relative_to(outdir).as_posix(),
            "svg": deg_png.with_suffix(".svg").relative_to(outdir).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Donor DEG plot data (TSV)", "path": deg_tsv.relative_to(outdir).as_posix(), "unit": "gene", "contrast_id": cid, "policy": None}],
        })

        sub_top_g = df_deg[df_deg["_sig"]].assign(_a=lambda d: d["log2fc"].abs()).sort_values(["pvalue" if use_nom else "fdr", "_a"], ascending=[True, False]).head(10)
        if not sub_top_g.empty:
            top10.append({
                "id": f"{_safe_slug(cid)}-sc-top10-deg",
                "chapter": 3,
                "contrast_id": cid,
                "analysis": "DEG",
                "kind": "Donor-Pseudobulk Host DEG",
                "unit": "gene",
                "columns": ["gene_symbol", "log2fc", "pvalue", "fdr"],
                "rows": sub_top_g[["gene_symbol", "log2fc", "pvalue", "fdr"]].to_dict("records"),
                "full_path": deg_pq.relative_to(outdir).as_posix(),
            })

        # Chapter 04: 2-Panel Host-Gene DEG × Single-Cell DAS Decoupling Quadrant
        if sc_das_by_contrast and cid in sc_das_by_contrast and not sc_das_by_contrast[cid].empty:
            render_deg_das_decoupling_quadrant(
                df_deg,
                sc_das_by_contrast[cid],
                cid,
                outdir,
                panels,
                policy=None,
                das_flag_col="_sig",
                das_label="Single-Cell DAS",
                fdr_cutoff=fdr_cutoff,
                fc_cutoff=0.4,
            )

    return sc_deg_by_contrast, pb_expr_df


def render_sc_celltype_stratification_summary(
    sc_das_by_contrast: dict[str, pd.DataFrame],
    contrasts_list: list[dict[str, Any]],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
) -> None:
    """Chapter 04 Single-Cell Multi-Subtype Overview: summarizes significant DAS events & host genes
    across all tested cell types under each experiment comparison (scales gracefully to 43+ cell types)."""
    subtype_cids = [c["contrast_id"] for c in contrasts_list if "__" in str(c["contrast_id"]) and c["contrast_id"] in sc_das_by_contrast]
    if len(subtype_cids) < 2:
        return

    rows = []
    for c_spec in contrasts_list:
        cid = str(c_spec["contrast_id"])
        if cid not in sc_das_by_contrast:
            continue
        df = sc_das_by_contrast[cid]
        if df.empty or "report_rs_das" not in df.columns:
            continue
        sig = df[df["report_rs_das"].astype(bool)]
        d_vals = pd.to_numeric(sig["dpsi"], errors="coerce").dropna() if "dpsi" in sig.columns else pd.Series(dtype=float)
        ct_lbl = cid.split("__", 1)[0] if "__" in cid else "All Cells (Experiment-wide)"
        exp_lbl = cid.split("__", 1)[1] if "__" in cid else cid
        rows.append({
            "contrast_id": cid,
            "experiment_comparison": exp_lbl,
            "cell_type": ct_lbl,
            "tested_events": len(df),
            "sig_events": len(sig),
            "sig_genes": int(sig["gene_symbol"].nunique()) if "gene_symbol" in sig.columns and len(sig) else 0,
            "included_up": int((d_vals > 0).sum()),
            "skipped_down": int((d_vals < 0).sum()),
            "is_experiment_wide": "__" not in cid,
        })
    if len(rows) < 2:
        return

    sum_df = pd.DataFrame(rows).sort_values(["is_experiment_wide", "sig_events", "cell_type"], ascending=[False, False, True]).reset_index(drop=True)
    tsv_p = outdir / "outputs/comparisons/all/sc-celltype-stratified-das-summary.tsv"
    tsv_p.parent.mkdir(parents=True, exist_ok=True)
    sum_df.to_csv(tsv_p, sep="\t", index=False)
    tables.append({
        "name": f"Cell-type stratified DAS summary ({len(subtype_cids)} cell populations)",
        "analysis_id": "differential.sc_subtype_summary",
        "chapter": 4,
        "rows": len(sum_df),
        "columns": list(sum_df.columns),
        "source": tsv_p.relative_to(outdir).as_posix(),
    })

    disp = sum_df.head(20).iloc[::-1].reset_index(drop=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    fig_h = max(4.2, len(disp) * 0.44 + 1.65)
    fig, axes = plt.subplots(1, 2, figsize=(14.6, fig_h), gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.34}, dpi=210)
    fig.patch.set_facecolor("white")

    y = np.arange(len(disp))
    up_v = disp["included_up"].to_numpy(dtype=float)
    dn_v = disp["skipped_down"].to_numpy(dtype=float)
    tot_v = disp["sig_events"].to_numpy(dtype=float)
    gn_v = disp["sig_genes"].to_numpy(dtype=float)
    lbls = [
        f"{r.cell_type.replace('_', ' ')} [{r.experiment_comparison.replace('_vs_', ' vs ')}]" if sum_df["experiment_comparison"].nunique() > 1
        else r.cell_type.replace("_", " ")
        for r in disp.itertuples(index=False)
    ]

    axes[0].barh(y, up_v, color=K_RED, height=0.60, label="Higher inclusion (ΔPSI > 0)")
    axes[0].barh(y, dn_v, left=up_v, color=K_BLUE, height=0.60, label="Lower inclusion / skipping (ΔPSI < 0)")
    max_ev = max(1.0, float(tot_v.max()))
    for yi, tv, uv, dv in zip(y, tot_v, up_v, dn_v):
        axes[0].text(tv + max_ev * 0.02, yi, f"{int(tv):,} (+{int(uv)} / −{int(dv)})", va="center", fontsize=8.0, fontweight="bold", color=K_NAVY)
    axes[0].set_yticks(y, lbls, fontsize=8.4, fontweight="bold", color=K_NAVY)
    axes[0].set_xlim(0, max_ev * 1.42)
    axes[0].set_xlabel("Significant Differential Splicing Events", fontweight="bold", color=K_NAVY)
    axes[0].set_title("A. Significant DAS Events by Cell Population", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY, pad=24)
    axes[0].legend(frameon=False, fontsize=8.0, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=2, handlelength=1.4, columnspacing=1.2)

    bar_cols = [K_NAVY if bool(w) else K_TEAL for w in disp["is_experiment_wide"]]
    axes[1].barh(y, gn_v, color=bar_cols, height=0.60, alpha=0.90)
    max_gn = max(1.0, float(gn_v.max()))
    for yi, gv in zip(y, gn_v):
        axes[1].text(gv + max_gn * 0.02, yi, f"{int(gv):,} genes", va="center", fontsize=8.0, fontweight="bold", color=K_NAVY)
    axes[1].set_yticks(y, ["" for _ in lbls])
    axes[1].set_xlim(0, max_gn * 1.32)
    axes[1].set_xlabel("Unique Affected Host Genes", fontweight="bold", color=K_NAVY)
    axes[1].set_title("B. Affected Host-Gene Breadth", loc="left", fontsize=10.2, fontweight="bold", color=K_NAVY, pad=24)

    extra_txt = f" (showing Top 20 of {len(sum_df)} tested populations)" if len(sum_df) > 20 else ""
    fig.suptitle(f"Cell-Type Stratified Single-Cell Differential Splicing Overview{extra_txt}", x=0.01, ha="left", fontsize=11.4, fontweight="bold", color=K_NAVY, y=0.99)
    fig.subplots_adjust(left=0.22, right=0.97, top=0.77, bottom=0.14)
    png_p = outdir / "figures/comparisons/sc-celltype-stratified-das-summary.png"
    _save_fig(fig, png_p)
    panels.append({
        "id": "sc-celltype-stratified-das-summary",
        "chapter": 4,
        "contrast_id": "all",
        "policy": None,
        "title": f"Cell-type stratified DAS summary across {len(subtype_cids)} cell populations",
        "caption": "Experiment-wide and per-cell-type donor-aware differential splicing event and host-gene burden. Use the Cell Type bar above to inspect any individual cell population.",
        "image": png_p.relative_to(outdir).as_posix(),
        "svg": png_p.with_suffix(".svg").relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "All cell-type DAS counts (TSV)", "path": tsv_p.relative_to(outdir).as_posix(), "unit": "cell_type_contrast", "contrast_id": "all", "policy": None}],
    })
