"""Core orchestration for standalone Python execution (`run_bulk`, `run_single_cell`) and Nextflow convergence.

Conforms strictly to `REPORT_FORMAT_LOCK.json` & approved Katana v2.7 Consulting Architecture:
  - Eight approved English chapters (`01` Design/QC, `02` AS Landscape, `03` Differential Molecular Results,
    `04` Cross-Comparison Overlap & Interaction, `05` GO/KEGG & Pathway Enrichment,
    `06` Katana-Splice-Network, `07` RBP and MotifTrace,
    `08` Integrative Evidence Synthesis & Target Prioritization) + Appendix `A`
  - Non-circular Multi-Community Spatial Halo `SPLICE_NETWORK_EXPLORER.html` in Chapter 06
  - Self-contained offline Interactive Candidate Gene Discovery, 5-Pillar Scorer & 4-Track GeneCard Workbench in Chapter 08
  - SHA-256 verified lossless Parquet/TSV outputs, interactive `PSI_MATRIX.html`, and `outputs/views/*.html` table browsers
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any
import webbrowser

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from katana_studio.candidate_workbench import build_chapter8_synthesis_and_workbench
from katana_studio.full_suite_views import (
    _safe_read_csv,
    _safe_slug,
    build_interactive_psi_matrix_browser,
    ingest_bulk_multilayer_deg_dtu_dexseq,
    render_biological_design_panel,
    render_bulk_as_composition_for_contrast,
    render_bulk_sequencing_qc_panel,
    render_ch4_cross_comparison_suite,
    render_ch7_splicetrace_suite,
    render_replicate_consistency_panels,
    render_sc_donor_deg_and_overlap,
    render_sc_celltype_stratification_summary,
    render_sc_umap_and_qc_suite,
)
from katana_studio.network_explorer import build_chapter6_splicenet_suite

NAVY = "#1E2630"
BLUE = "#2863A5"
CERULEAN = "#218D87"
CORAL = "#C63F4A"
SLATE = "#5B6470"
EVENTS = ("SE", "A3SS", "A5SS", "MXE", "RI")

_COORD_COLS: dict[str, tuple[str, ...]] = {
    "SE": ("exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "A3SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "A5SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "MXE": ("1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "RI": ("riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_tabular(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    sep = "," if (name.endswith(".csv") or name.endswith(".csv.gz")) else "\t"
    return pd.read_csv(path, sep=sep)


def _save_dual(fig: plt.Figure, png_path: Path) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg_path


def _clean_coord_str(val: object) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    return str(val).strip().strip('"').strip("'")


def _canonical_rmats_event_ids(df: pd.DataFrame, et: str) -> pd.Series:
    """Construct contrast-invariant 0-based coordinate keys identical to `SpliceNet` and `SpliceTrace`."""
    cols = _COORD_COLS.get(et, ())
    if cols and {"chr", "strand", *cols}.issubset(df.columns):
        return (
            et
            + ":"
            + df["chr"].map(_clean_coord_str)
            + ":"
            + df["strand"].map(_clean_coord_str)
            + ":"
            + df[list(cols)].apply(lambda c: c.map(_clean_coord_str)).agg(":".join, axis=1)
        )
    gid_c = next((c for c in ("GeneID", "gene_id") if c in df.columns), "ID")
    id_c = "ID" if "ID" in df.columns else df.columns[0]
    return df[gid_c].astype(str).str.strip('"') + "|" + et + "|" + df[id_c].astype(str)


def _run_ora_from_bundled_catalog(fg_symbols: set[str], bg_symbols: set[str], species: str) -> pd.DataFrame:
    """Exact Fisher/Hypergeometric + per-category BH FDR across GO (BP, CC, MF) and KEGG."""
    res_dir = Path(__file__).resolve().parents[1] / "katana_next" / "resources"
    gz_path = res_dir / ("pathways_mouse_go_kegg.json.gz" if species == "mouse" else "pathways_human_go_kegg.json.gz")
    if not gz_path.is_file():
        return pd.DataFrame(columns=["category", "term", "overlap_count", "term_size", "selected_genes", "background_genes", "fold_enrichment", "pvalue", "fdr", "hit_genes"])
    raw_cat = json.loads(gzip.open(gz_path, "rt", encoding="utf-8").read())
    catalog = raw_cat.get("categories", raw_cat)

    sym_case_map: dict[str, str] = {}
    for cmap in catalog.values():
        for g_list in cmap.values():
            for gx in g_list:
                s_gx = str(gx).strip()
                if s_gx:
                    sym_case_map.setdefault(s_gx.upper(), s_gx)
    for g in list(bg_symbols) + list(fg_symbols):
        sg = str(g).strip()
        if sg and not sg.upper().startswith("ENS") and sg.upper() != "NAN":
            sym_case_map[sg.upper()] = sg

    fg_u = {g.strip().upper() for g in fg_symbols if g and not str(g).upper().startswith("ENS") and str(g).upper() != "NAN"}
    bg_u = {g.strip().upper() for g in bg_symbols if g and not str(g).upper().startswith("ENS") and str(g).upper() != "NAN"} | fg_u
    cat_universe = set(sym_case_map.keys())
    if len(bg_u & cat_universe) < 500:
        bg_u = bg_u | cat_universe
    fg_in_bg = fg_u & bg_u
    M = len(bg_u)
    N = len(fg_in_bg)
    if N == 0 or M < 10:
        return pd.DataFrame(columns=["category", "term", "overlap_count", "term_size", "selected_genes", "background_genes", "fold_enrichment", "pvalue", "fdr", "hit_genes"])

    out_rows: list[dict[str, Any]] = []
    for cat in ("BP", "CC", "MF", "KEGG"):
        cat_rows = []
        n_eligible_tested = 0
        for term, raw_genes in catalog.get(cat, {}).items():
            t_set = {str(x).upper() for x in raw_genes} & bg_u
            n_t = len(t_set)
            if n_t < 5 or n_t > 2000:
                continue
            n_eligible_tested += 1
            hits = sorted(fg_in_bg & t_set)
            k = len(hits)
            if k < 2:
                continue
            pv = float(sp_stats.hypergeom.sf(k - 1, M, n_t, N))
            fe = (k / N) / (n_t / M)
            cat_rows.append({
                "category": cat,
                "ontology_category": cat,
                "library": "KEGG" if cat == "KEGG" else f"GO_{cat}",
                "term": term,
                "overlap_count": k,
                "overlap_genes": k,
                "term_size": n_t,
                "selected_genes": N,
                "background_genes": M,
                "gene_ratio": round(k / max(1, N), 5),
                "fold_enrichment": round(fe, 4),
                "pvalue": pv,
                "hit_genes": ",".join(sym_case_map.get(h, h) for h in hits[:50]),
            })
        if cat_rows:
            cat_rows.sort(key=lambda r: r["pvalue"])
            # Exact Benjamini-Hochberg denominator across ALL eligible tested terms in this ontology family
            # (never shrink m_t to the k>=2 subset, which would artificially inflate significance).
            m_t = max(n_eligible_tested, len(cat_rows))
            q_vals = [min(1.0, r["pvalue"] * m_t / (idx + 1)) for idx, r in enumerate(cat_rows)]
            for idx in range(len(cat_rows) - 2, -1, -1):
                q_vals[idx] = min(q_vals[idx], q_vals[idx + 1])
            for r, qv in zip(cat_rows, q_vals):
                r["fdr"] = qv
                r["minus_log10_fdr"] = -np.log10(max(qv, 1e-300))
            out_rows.extend(cat_rows)
    return pd.DataFrame(out_rows)


def _render_ora_faceted_figure(ora_df: pd.DataFrame, png_path: Path, title: str) -> Path | None:
    if ora_df.empty:
        return None
    from katana_next.reports.enrichment_views import render_editorial_ora_facets

    sig_df = ora_df[pd.to_numeric(ora_df["fdr"], errors="coerce").fillna(1.0) <= 0.05].copy()
    svg_path = png_path.with_suffix(".svg")
    render_editorial_ora_facets(
        sig_df,
        png_path,
        svg_path=svg_path,
        title=title,
        complete_df=ora_df,
    )
    return svg_path


def _render_volcano(
    df: pd.DataFrame,
    x_col: str,
    q_col: str,
    sig_col: str,
    label_col: str,
    xlabel: str,
    title: str,
    png_path: Path,
    *,
    is_deg: bool = False,
) -> Path:
    """Publication-grade Volcano Plot with top-outside legend and collision-repelled leader-line gene callouts."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 5.5), dpi=220)
    fig.patch.set_facecolor("white")
    xv = pd.to_numeric(df[x_col], errors="coerce").fillna(0.0).to_numpy()
    qv = np.clip(pd.to_numeric(df[q_col], errors="coerce").fillna(1.0).to_numpy(), 1e-300, 1.0)
    yv = -np.log10(qv)
    yv_disp = np.minimum(yv, np.nanpercentile(yv, 99.5) * 1.12 + 1.8) if len(yv) else yv
    sig = df[sig_col].astype(bool).to_numpy() if sig_col in df.columns else (qv <= 0.05)

    x_abs_max = max(float(np.nanmax(np.abs(xv))) if len(xv) else 0.5, 0.35 if not is_deg else 1.2)
    x_lim = x_abs_max * 1.34
    y_top = max(float(np.nanmax(yv_disp)) if len(yv_disp) else 2.0, 2.2) * 1.22

    pos_lbl = "Up-regulated" if is_deg else "Included (ΔPSI > 0)"
    neg_lbl = "Down-regulated" if is_deg else "Skipped (ΔPSI < 0)"

    ax.scatter(xv[~sig], yv_disp[~sig], s=13, color="#CBD5E1", alpha=0.52, linewidths=0, label=f"Background ({int((~sig).sum()):,})")
    up = sig & (xv > 0)
    dn = sig & (xv < 0)
    if up.any():
        ax.scatter(xv[up], yv_disp[up], s=25, color=CORAL, alpha=0.88, edgecolors="white", linewidths=0.35, label=f"{pos_lbl} ({int(up.sum()):,})", zorder=3)
    if dn.any():
        ax.scatter(xv[dn], yv_disp[dn], s=25, color=BLUE, alpha=0.88, edgecolors="white", linewidths=0.35, label=f"{neg_lbl} ({int(dn.sum()):,})", zorder=3)
    ax.axhline(-np.log10(0.05), color=SLATE, linestyle="--", linewidth=0.9, alpha=0.7)
    ax.axvline(0.0, color="#94A3B8", linestyle=":", linewidth=0.8)

    if sig.any() and label_col in df.columns:
        sig_indices = np.flatnonzero(sig)
        # Rank by significance then |effect|
        score_rank = yv_disp[sig_indices] + 0.35 * (np.abs(xv[sig_indices]) / max(x_abs_max, 1e-6))
        ordered_sig = sig_indices[np.argsort(-score_rank)]

        for side_sign, ha_align in ((-1.0, "right"), (1.0, "left")):
            picked: list[tuple[float, float, str]] = []
            seen_side: set[str] = set()
            for idx in ordered_sig:
                if (xv[idx] < 0 and side_sign > 0) or (xv[idx] >= 0 and side_sign < 0):
                    continue
                lbl = str(df.iloc[idx][label_col] or "").strip()
                if not lbl or lbl.lower() == "nan" or lbl in seen_side:
                    continue
                if len(lbl) > 16:
                    lbl = lbl[:15] + "…"
                seen_side.add(lbl)
                picked.append((float(xv[idx]), float(yv_disp[idx]), lbl))
                if len(picked) >= 5:
                    break
            if not picked:
                continue
            picked.sort(key=lambda t: t[1])
            ly = np.array([t[1] for t in picked], dtype=float)
            min_dy = 0.082 * y_top
            for _ in range(36):
                moved = False
                for i in range(1, len(ly)):
                    if ly[i] - ly[i - 1] < min_dy:
                        mid = 0.5 * (ly[i] + ly[i - 1])
                        ly[i - 1] = mid - 0.5 * min_dy
                        ly[i] = mid + 0.5 * min_dy
                        moved = True
                ly = np.clip(ly, 0.07 * y_top, 0.94 * y_top)
                if not moved:
                    break
            for (px, py, lbl), ty in zip(picked, ly):
                tx = px + side_sign * (0.085 * x_abs_max)
                tx = float(np.clip(tx, -x_lim * 0.94, x_lim * 0.94))
                ax.annotate(
                    lbl,
                    xy=(px, py),
                    xytext=(tx, float(ty)),
                    ha=ha_align,
                    va="center",
                    fontsize=7.9,
                    fontweight="bold",
                    color=NAVY,
                    bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.86},
                    arrowprops={"arrowstyle": "-", "color": "#64748B", "lw": 0.65, "shrinkA": 0, "shrinkB": 3},
                    zorder=5,
                )

    ax.set_xlim(-x_lim, x_lim)
    ax.set_ylim(-0.03 * y_top, y_top)
    ax.set_xlabel(xlabel, fontweight="bold", color=NAVY)
    ax.set_ylabel("-log10(FDR / P)", fontweight="bold", color=NAVY)
    ax.set_title(title, loc="left", fontsize=10.6, fontweight="bold", color=NAVY, pad=22)
    ax.legend(frameon=False, fontsize=8.0, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=3, columnspacing=1.0, handletextpad=0.35)
    return _save_dual(fig, png_path)


def _render_das_class_direction_bar(
    qc_df: pd.DataFrame,
    flag_col: str,
    lbl: str,
    cid: str,
    pol: str | None,
    out: Path,
    panels: list[dict[str, Any]],
) -> None:
    """Clean Chapter 03 summary of significant alternative splicing events by class and ΔPSI sign."""
    if qc_df.empty or flag_col not in qc_df.columns or "event_type" not in qc_df.columns:
        return
    sub = qc_df[qc_df[flag_col].astype(bool)].copy()
    if sub.empty:
        return
    ets = [et for et in EVENTS if et in set(qc_df["event_type"].astype(str))]
    if not ets:
        return
    rows = []
    for et in ets:
        m = sub["event_type"].astype(str).eq(et)
        dp = pd.to_numeric(sub.loc[m, "dpsi"], errors="coerce").fillna(0.0)
        rows.append({
            "event_type": et,
            "included_pos_dpsi": int((dp > 0).sum()),
            "skipped_neg_dpsi": int((dp < 0).sum()),
            "total_significant": int(m.sum()),
        })
    cdf = pd.DataFrame(rows)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", cid)
    pol_s = pol or "common"
    tsv_p = out / "outputs/comparisons" / slug / f"{slug}-das-class-direction-{pol_s}.tsv"
    tsv_p.parent.mkdir(parents=True, exist_ok=True)
    cdf.to_csv(tsv_p, sep="\t", index=False)
    png_p = out / "figures/comparisons" / f"{slug}-das-class-direction-{pol_s}.png"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.2, 3.6), dpi=220)
    fig.patch.set_facecolor("white")
    y = np.arange(len(cdf))[::-1]
    inc = cdf["included_pos_dpsi"].to_numpy(dtype=float)
    skp = cdf["skipped_neg_dpsi"].to_numpy(dtype=float)
    ax.barh(y, -skp, height=0.56, color=BLUE, alpha=0.9, label="Skipped (ΔPSI < 0)")
    ax.barh(y, inc, height=0.56, color=CORAL, alpha=0.9, label="Included (ΔPSI > 0)")
    ax.axvline(0, color=NAVY, linewidth=0.9)
    max_side = max(float(max(inc.max(), skp.max())), 3.0)
    for yi, iv, sv in zip(y, inc, skp):
        if iv > 0:
            ax.text(iv + max_side * 0.03, yi, f"+{int(iv)}", va="center", ha="left", fontsize=8.1, fontweight="bold", color=CORAL)
        if sv > 0:
            ax.text(-sv - max_side * 0.03, yi, f"−{int(sv)}", va="center", ha="right", fontsize=8.1, fontweight="bold", color=BLUE)
    ax.set_yticks(y, cdf["event_type"].tolist(), fontweight="bold", color=NAVY)
    ax.set_xlim(-max_side * 1.28, max_side * 1.28)
    xticks = ax.get_xticks()
    ax.set_xticks(xticks, [str(int(abs(t))) if t == int(t) else f"{abs(t):.1f}" for t in xticks])
    ax.set_xlabel("Significant Splicing Events (Skipped ← 0 → Included)", fontweight="bold", color=NAVY)
    ax.set_title(f"{lbl} Events by Splice Class & Direction | {cid}", loc="left", fontsize=10.2, fontweight="bold", color=NAVY, pad=28)
    ax.legend(frameon=False, fontsize=8.0, loc="lower right", bbox_to_anchor=(1.0, 1.02), ncol=2)
    _save_dual(fig, png_p)
    panels.append({
        "id": f"{slug}-das-class-direction-{pol_s}",
        "chapter": 3,
        "contrast_id": cid,
        "policy": pol,
        "title": f"{lbl} splice-class directional breakdown | {cid}",
        "caption": "Mirrored counts of exon-included (ΔPSI > 0) versus exon-skipped (ΔPSI < 0) significant events per canonical splice class.",
        "image": png_p.relative_to(out).as_posix(),
        "svg": png_p.with_suffix(".svg").relative_to(out).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Class directional counts (TSV)", "path": tsv_p.relative_to(out).as_posix(), "unit": "event_type", "contrast_id": cid, "policy": pol}],
    })


def _has_rmats_files(d: Path, rmats_mode: str = "JCEC") -> bool:
    modes = (rmats_mode.upper(), "JCEC", "JC")
    return d.is_dir() and any((d / f"{et}.MATS.{m}.txt").is_file() for et in EVENTS for m in modes)


def _discover_rmats_contrasts(rmats_arg: str | Path | dict[str, Path], declared_contrasts: list[dict[str, Any]], rmats_mode: str = "JCEC") -> dict[str, Path]:
    """Resolve rMATS directory mapping from single dir, parent dir of contrasts, or contrast=dir mapping."""
    if isinstance(rmats_arg, dict):
        return {str(k): Path(v) for k, v in rmats_arg.items()}
    s = str(rmats_arg)
    if "=" in s and not Path(s).exists():
        out = {}
        for part in s.split(","):
            k, v = part.split("=", 1)
            out[k.strip()] = Path(v.strip())
        return out
    base = Path(s)
    if _has_rmats_files(base, rmats_mode):
        cid = declared_contrasts[0]["contrast_id"] if declared_contrasts else base.name
        return {cid: base}
    out = {}
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        if _has_rmats_files(sub, rmats_mode):
            out[sub.name] = sub
        elif _has_rmats_files(sub / "rmats_post", rmats_mode):
            out[sub.name] = sub / "rmats_post"
    if not out:
        raise FileNotFoundError(f"No *.MATS.(JCEC|JC).txt files found under {base}")
    return out


def _resolve_splicetrace_dirs(
    splicetrace_dirs: list[str | Path] | dict[str, Path] | None,
    declared_contrasts: list[dict[str, Any]],
) -> dict[str, Path]:
    """Map precomputed per-contrast SpliceTrace directories (`cid=path` or `splicetrace_<cid>` or ordered dirs) to contrast IDs."""
    if not splicetrace_dirs:
        return {}
    if isinstance(splicetrace_dirs, dict):
        return {str(k): Path(v) for k, v in splicetrace_dirs.items() if Path(v).is_dir()}

    cids = [c["contrast_id"] for c in declared_contrasts]
    mapped: dict[str, Path] = {}
    unmatched: list[Path] = []
    for item in splicetrace_dirs:
        raw_s = str(item).strip()
        if not raw_s or raw_s == "NO_SPLICETRACE":
            continue
        if "=" in raw_s and not Path(raw_s).exists():
            k, v = raw_s.split("=", 1)
            pv = Path(v.strip())
            if pv.is_dir():
                mapped[k.strip()] = pv
            continue
        p = Path(raw_s)
        if not p.is_dir():
            continue
        matched_cid = None
        for cid in cids:
            if p.name == cid or p.name == f"splicetrace_{cid}" or p.name.endswith(f"_{cid}"):
                matched_cid = cid
                break
        if matched_cid:
            mapped[matched_cid] = p
        else:
            unmatched.append(p)

    if unmatched and len(cids) == 1 and cids[0] not in mapped:
        mapped[cids[0]] = unmatched[0]
    elif unmatched:
        rem_cids = [c for c in cids if c not in mapped]
        for cid, p in zip(rem_cids, unmatched):
            mapped[cid] = p
    return mapped


def _resolve_bundle_inputs(bundle_dir: Path) -> dict[str, Any]:
    """Auto-discover standard upstream Nextflow `katana_bundle` outputs for seamless convergence."""
    bundle = Path(bundle_dir).resolve()
    if not bundle.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle}")
    samples_f = next((p for p in (bundle / "pipeline_info/samplesheet.valid.csv", bundle / "inputs/samplesheet.csv", bundle / "samplesheet.csv", bundle / "samples.csv", bundle / "provenance/strandedness/samplesheet.requested.csv") if p.is_file()), None)
    contrasts_f = next((p for p in (bundle / "pipeline_info/contrastsheet.valid.csv", bundle / "inputs/contrastsheet.csv", bundle / "contrasts.csv", bundle / "contrastsheet.csv") if p.is_file()), None)
    rmats_dir = next((p for p in (bundle / "rmats", bundle / "splicing/rmats") if p.is_dir()), None)
    deg_dir = next((p for p in (bundle / "DEG", bundle / "deseq2") if p.is_dir()), None)
    multiqc_dir = next((p for p in (bundle / "qc/multiqc", bundle / "qc", bundle / "multiqc") if p.is_dir()), None)
    fasta_f = next((p for p in (bundle / "refs/genome.fasta", bundle / "refs/genome.fa") if p.is_file()), None)
    gtf_f = next((p for p in (bundle / "refs/annotation.gtf", bundle / "refs/genes.gtf") if p.is_file()), None)
    bam_dir = bundle / "bam" if (bundle / "bam").is_dir() else None
    if samples_f is None or rmats_dir is None:
        raise ValueError(f"Bundle {bundle} must contain samplesheet.csv and rmats/ directory")
    return {
        "samples": samples_f,
        "contrasts": contrasts_f,
        "rmats": rmats_dir,
        "deseq2": deg_dir,
        "multiqc": multiqc_dir,
        "fasta": fasta_f,
        "gtf": gtf_f,
        "bams": bam_dir,
    }


def _finalize_consulting_report_and_zip(
    outdir: Path,
    study_name: str,
    mode: str,
    samples_list: list[dict[str, Any]],
    contrasts_list: list[dict[str, Any]],
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    top10: list[dict[str, Any]],
    policy_data: dict[str, Any],
    motif_summaries: list[dict[str, Any]],
    species: str,
    network_browsers: list[dict[str, Any]] | None = None,
    enrichment_summaries: list[dict[str, Any]] | None = None,
    interactive_blocks: list[dict[str, Any]] | None = None,
    dummy: bool = False,
) -> dict[str, Any]:
    """Build SHA256-verified report-manifest.json, provenance.json, interactive table browsers, REPORT.html, and REPORT.zip."""
    from katana_next.reports import engine as rep_engine, layout

    is_sc_mode = str(mode).lower().startswith("single")
    rep_engine.CHAPTERS = {
        1: ("Design, inputs & QC", "Declared biological samples, comparisons, references and available QC."),
        2: ("SpliceManifold — AS landscape", "Empirical-Bayes depth-stabilized PSI manifold, event composition, sample/cell embedding and PSI patterns."),
        3: ("Differential molecular results", "Gene expression and precision-supported alternative splicing events; original measurement units retained."),
        4: ("Cross-comparison & interaction", "Transcriptional vs splicing decoupling, cell-type/multi-comparison overlaps and ΔPSI concordance."),
        5: ("GO, KEGG & pathway enrichment", "Comparison-specific foreground sets and measured backgrounds across GO (BP/CC/MF) and KEGG."),
        6: (
            "scSpliceNet — Co-splicing network" if is_sc_mode else "SpliceNet — Co-splicing network",
            "Multi-community spatial co-splicing network, module eigengene ΔME, hub genes and functional pathways.",
        ),
        7: (
            "scSpliceTrace — RBP & proteome" if is_sc_mode else "SpliceTrace — RBP & proteome",
            "Positional RBP splicing-map FWL Z-scores, PWM logos, module-RBP bridge and 50-nt EJC NMD / UniProt consequences.",
        ),
        8: ("Katana-Studio — Integrative synthesis", "Multi-tool orthogonal convergence, convergent target scorecard, interactive Candidate Gene Finder & BAM RPKM Katana Native Sashimi workbench."),
    }
    rep_engine.CHAPTER_NUMBERS = {identity: number for number, identity in enumerate(rep_engine.CHAPTERS, 1)}
    layout.CHAPTERS = rep_engine.CHAPTERS
    layout.CHAPTER_NUMBERS = rep_engine.CHAPTER_NUMBERS

    assets_src = Path(layout.__file__).resolve().parent / "assets"
    assets_dst = outdir / "assets"
    assets_dst.mkdir(parents=True, exist_ok=True)
    for asset_name in ("report.css", "report.js", "matrix-browser.html", "table-template.html"):
        src_a = assets_src / asset_name
        if src_a.is_file():
            shutil.copy2(src_a, assets_dst / asset_name)

    tbl_tpl_p = assets_src / "table-template.html"
    tbl_tpl = tbl_tpl_p.read_text(encoding="utf-8") if tbl_tpl_p.is_file() else ""
    views_dir = outdir / "outputs/views"
    views_dir.mkdir(parents=True, exist_ok=True)

    source_files: list[dict[str, Any]] = []
    seen_rel: set[str] = set()
    for p in panels:
        for lk in p.get("data_links", []):
            rel = str(lk["path"]).replace("\\", "/")
            lk["path"] = rel
            fp = outdir / rel
            if fp.is_file():
                dig = _sha256_file(fp)
                lk["source_sha256"] = dig
                lk["portable_sha256"] = dig
                lk["bytes"] = fp.stat().st_size
                if rel not in seen_rel:
                    source_files.append({"source_name": fp.name, "source_sha256": dig, "portable_sha256": dig, "bytes": fp.stat().st_size, "report_path": rel})
                    seen_rel.add(rel)

    for idx_t, t in enumerate(tables):
        rel = str(t["source"]).replace("\\", "/")
        t["source"] = rel
        t["view"] = t.get("view") or rel
        fp = outdir / rel
        if fp.is_file():
            dig = _sha256_file(fp)
            t.setdefault("metadata", {})["portable_sha256"] = dig
            if rel not in seen_rel:
                source_files.append({"source_name": fp.name, "source_sha256": dig, "portable_sha256": dig, "bytes": fp.stat().st_size, "report_path": rel})
                seen_rel.add(rel)
            if tbl_tpl and fp.suffix.lower() in {".parquet", ".tsv", ".csv"} and fp.stat().st_size <= 8 * 1024 * 1024:
                try:
                    df_v = _read_tabular(fp)
                    if len(df_v) <= 15000:
                        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{idx_t:02d}-{fp.stem}").strip("-")[:90]
                        view_p = views_dir / f"{safe_stem}.html"
                        rows_json = [
                            [None if pd.isna(v) else (round(float(v), 6) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)) for v in row]
                            for row in df_v.itertuples(index=False, name=None)
                        ]
                        payload_str = json.dumps({"columns": list(df_v.columns), "rows": rows_json}, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
                        html_v = (
                            tbl_tpl.replace("__TITLE__", str(t.get("name") or fp.stem))
                            .replace("__RAW__", "../../" + rel)
                            .replace("__DATA__", payload_str)
                        )
                        view_p.write_text(html_v, encoding="utf-8")
                        t["view"] = view_p.relative_to(outdir).as_posix()
                except Exception:
                    pass

    project_dict = {
        "project_id": re.sub(r"[^A-Za-z0-9_.-]+", "-", study_name.lower()).strip("-") or "katana-study",
        "name": study_name,
        "samples": samples_list,
        "contrasts": contrasts_list,
        "annotation": {"genome_build": "GRCm39 / mm39" if species == "mouse" else "GRCh38 / hg38", "species": species},
        "metadata": {"assay_mode": "single-cell" if mode == "single-cell" else "bulk", "species": species},
    }

    class _SuiteReportBuilder:
        def __init__(self) -> None:
            self.output = outdir
            self.source = {
                "project": project_dict,
                "mode": mode,
                "dummy": bool(dummy),
                "report_selection": {"applied": False},
                "artifacts": [],
            }
            self.analyses = {
                "landscape": True,
                "cross_comparison": len(contrasts_list) >= 1,
                "enrichment": any(p.get("chapter") == 5 for p in panels),
                "rbp_identity": bool(enrichment_summaries),
                "motiftrace": any(p.get("chapter") == 7 for p in panels) or bool(motif_summaries),
                "motif_binding": False,
                "mapp": False,
                "interaction": any(p.get("chapter") == 4 for p in panels),
                "network": any(p.get("chapter") == 6 for p in panels) or any(t.get("chapter") == 6 for t in tables) or bool(network_browsers),
                "sequencing_qc": True,
            }
            self.event_set = "stable_significant"
            self.panels = panels
            self.tables = tables
            self.top10 = top10
            self.policy_data = policy_data
            self.enrichment_summaries = list(enrichment_summaries or [])
            self.motif_summaries = motif_summaries
            self.network_browsers = list(network_browsers or [])
            self.interactive_blocks = list(interactive_blocks or [])
            self.audit = source_files
            self.exclusions: list[dict[str, Any]] = []
            self.derived_audits: list[dict[str, Any]] = []
            self.create_archive = True
            n_units = len({s.get("biological_sample_id", s.get("sample_id")) for s in samples_list})
            n_conds = len({s.get("condition") for s in samples_list})
            self.chapter_callouts: dict[int, str] = {
                1: f"Audited {len(samples_list):,} {'cells/nuclei' if mode == 'single-cell' else 'RNA-seq libraries'} across {n_units:,} independent biological units and {n_conds} conditions ({len(contrasts_list)} declared contrast{'s' if len(contrasts_list) != 1 else ''}).",
                2: "Empirical-Bayes Binomial support-shrunk PSI landscape separating biological splicing variation from sampling noise.",
                3: "Differential host-gene transcription (DEG), precision-supported alternative splicing (DAS), and lossless orthogonal SUPPA2 / DEXSeq / DRIMSeq-stageR DTU tables.",
                4: "Host transcription vs post-transcriptional splicing decoupling, pairwise cross-comparison overlaps, and ΔPSI directional concordance.",
                5: "Exact hypergeometric enrichment across GO (BP, CC, MF) and KEGG pathways with per-category Benjamini-Hochberg FDR.",
                6: "Non-circular Multi-Community Spatial Halo co-splicing network with signed sample-PSI correlations, hub |kME|, and contrast ΔME.",
                7: "Positional splicing-map cis-motif Z-scores, information-content PWM logos, module-RBP bridge, and 50-nt EJC NMD / UniProt consequences.",
                8: "Multi-tool orthogonal convergence (rMATS × SUPPA2 × DEXSeq × DTU × DEG), convergent target scorecard, interactive Candidate Gene Finder, and BAM RPKM Katana Native Sashimi 4-Track Workbench.",
            }

        def figure_data_coverage(self) -> dict[str, Any]:
            main_p = [p for p in self.panels if not p.get("output_only")]
            with_d = [p for p in main_p if any(d.get("role") == "plot_data" for d in p.get("data_links", []))]
            return {
                "schema": "katana.figure_data_links/v1",
                "main_figures": len(main_p),
                "main_figures_with_plot_data": len(with_d),
                "missing_plot_data": [{"id": p["id"], "title": p["title"]} for p in main_p if p not in with_d],
            }

    manifest = layout.render(_SuiteReportBuilder())
    return {
        "status": "completed",
        "study_name": study_name,
        "mode": mode,
        "species": species,
        "dummy": bool(dummy),
        "outdir": str(outdir),
        "report_html": str(outdir / "REPORT.html"),
        "report_zip": str(outdir / "REPORT.zip"),
        "manifest_json": str(outdir / "report-manifest.json"),
        "panels_count": len(panels),
        "tables_count": len(tables),
        "missing_plot_data": len(manifest.get("figure_data_links", {}).get("missing_plot_data", [])),
        "generated_utc": manifest.get("generated_utc", datetime.now(timezone.utc).isoformat()),
    }


def run_bulk(
    *,
    outdir: str | Path,
    rmats: str | Path | dict[str, Path] | None = None,
    samples: str | Path | pd.DataFrame | None = None,
    contrasts: str | Path | pd.DataFrame | None = None,
    deseq2: str | Path | dict[str, Path] | None = None,
    multiqc: str | Path | None = None,
    fasta: str | Path | None = None,
    gtf: str | Path | None = None,
    bams: str | Path | None = None,
    bundle: str | Path | None = None,
    splicemanifold_dir: str | Path | None = None,
    splicenet_dir: str | Path | None = None,
    splicetrace_dirs: list[str | Path] | dict[str, Path] | None = None,
    rmats_mode: str = "JCEC",
    study_name: str = "Katana Bulk Splicing Study",
    species: str = "auto",
    fdr_cutoff: float = 0.05,
    dpsi_cutoff: float = 0.10,
    manifold_top_n: int = 2000,
    heatmap_top_n: int = 150,
    dummy: bool = False,
) -> dict[str, Any]:
    """Execute or assemble the complete Original-Katana-compliant Bulk Suite (Ch 01..07 + Appendix A)."""
    from katana_next.science.bulk import apply_stable_event_qc, parse_number_list, significance_columns
    from katana_next.science.psi_landscape import (
        compute_beyond_jutils_landscape,
        render_editorial_psi_clustermap,
        render_editorial_psi_pca,
    )

    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    bundle_path = Path(bundle).resolve() if bundle is not None else None

    if bundle_path is not None:
        resolved = _resolve_bundle_inputs(bundle_path)
        samples = samples or resolved["samples"]
        contrasts = contrasts or resolved["contrasts"]
        rmats = rmats or resolved["rmats"]
        deseq2 = deseq2 or resolved["deseq2"]
        multiqc = multiqc or resolved["multiqc"]
        fasta = fasta or resolved["fasta"]
        gtf = gtf or resolved["gtf"]
        bams = bams or resolved["bams"]

    if samples is None or rmats is None:
        raise ValueError("Provide either `--bundle <dir>` or both `--rmats` and `--samples`.")

    s_df = samples.copy() if isinstance(samples, pd.DataFrame) else _read_tabular(Path(samples))
    sid_col = next((c for c in ("sample_id", "sample", "id") if c in s_df.columns), s_df.columns[0])
    cond_col = next((c for c in ("condition", "group") if c in s_df.columns), s_df.columns[1])
    bio_col = next((c for c in ("biological_sample_id", "subject", "donor_id") if c in s_df.columns), None)
    samples_list = [
        {
            "sample_id": str(r[sid_col]).strip(),
            "condition": str(r[cond_col]).strip(),
            "biological_sample_id": str(r[bio_col]).strip() if bio_col and pd.notna(r[bio_col]) and str(r[bio_col]).strip() else str(r[sid_col]).strip(),
            "cell_type": str(r[cond_col]).strip(),
        }
        for _, r in s_df.iterrows()
    ]
    cond_to_sids: dict[str, list[str]] = {}
    for s in samples_list:
        cond_to_sids.setdefault(s["condition"], []).append(s["sample_id"])

    if contrasts is not None:
        c_df = contrasts.copy() if isinstance(contrasts, pd.DataFrame) else _read_tabular(Path(contrasts))
        cid_c = next((c for c in ("contrast_id", "contrast", "comparison_id") if c in c_df.columns), c_df.columns[0])
        num_c = next((c for c in ("treatment", "numerator", "group_a") if c in c_df.columns), c_df.columns[1])
        den_c = next((c for c in ("control", "denominator", "group_b") if c in c_df.columns), c_df.columns[2])
        contrasts_list = [
            {
                "contrast_id": str(r[cid_c]).strip(),
                "label": f"{str(r[num_c]).strip()} vs {str(r[den_c]).strip()}",
                "numerator": {"label": str(r[num_c]).strip(), "sample_ids": cond_to_sids.get(str(r[num_c]).strip(), [])},
                "denominator": {"label": str(r[den_c]).strip(), "sample_ids": cond_to_sids.get(str(r[den_c]).strip(), [])},
            }
            for _, r in c_df.iterrows()
        ]
    else:
        conds = list(cond_to_sids.keys())
        if len(conds) < 2:
            raise ValueError("At least 2 conditions in samples table are required when --contrasts is omitted.")
        contrasts_list = [{
            "contrast_id": f"{conds[0]}_vs_{conds[1]}",
            "label": f"{conds[0]} vs {conds[1]}",
            "numerator": {"label": conds[0], "sample_ids": cond_to_sids[conds[0]]},
            "denominator": {"label": conds[1], "sample_ids": cond_to_sids[conds[1]]},
        }]

    rmats_map = _discover_rmats_contrasts(rmats, contrasts_list, rmats_mode=rmats_mode)
    panels: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    top10: list[dict[str, Any]] = []
    network_browsers: list[dict[str, Any]] = []
    policy_data: dict[str, Any] = {"stable_significant": {}, "significant": {}}
    motif_summaries: list[dict[str, Any]] = []

    # Chapter 01: Biological design + Alignment/Strandedness QC + MultiQC 5'-to-3' coverage
    render_biological_design_panel(samples_list, out, panels, is_single_cell=False)
    render_bulk_sequencing_qc_panel(bundle_path, samples_list, out, panels, tables)

    if multiqc is not None and Path(multiqc).exists():
        mqc_path = Path(multiqc)
        from katana_next.reports.multiqc_coverage import parse_multiqc_5to3_coverage, render_editorial_5to3_figure
        cov = parse_multiqc_5to3_coverage(mqc_path, {s["sample_id"]: s["condition"] for s in samples_list})
        if cov is not None and "curves" in cov:
            cov_png = out / "figures/qc/multiqc-5to3-coverage.png"
            cov_tsv = out / "outputs/qc/multiqc-5to3-coverage-plot-data.tsv"
            cov_png.parent.mkdir(parents=True, exist_ok=True)
            cov_tsv.parent.mkdir(parents=True, exist_ok=True)
            cov["curves"].to_csv(cov_tsv, sep="\t", index=False)
            render_editorial_5to3_figure(cov, cov_png, cov_png.with_suffix(".svg"), title=f"Transcript 5′ → 3′ Gene-Body Coverage — {study_name}")
            panels.append({
                "id": "qc-multiqc-5to3",
                "chapter": 1,
                "contrast_id": "all",
                "title": f"5′ → 3′ transcript coverage uniformity ({cov.get('provenance_source', 'MultiQC')})",
                "caption": "Directly reused from MultiQC / Qualimap saved raw curves without slow BAM re-scanning.",
                "image": cov_png.relative_to(out).as_posix(),
                "svg": cov_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Coverage curves (TSV)", "path": cov_tsv.relative_to(out).as_posix(), "unit": "percentile_bin", "contrast_id": "all", "policy": None}],
            })
        mqc_html = next((p for p in (mqc_path / "multiqc_report.html", *sorted(mqc_path.rglob("multiqc_report.html"))) if p.is_file()), None) if mqc_path.is_dir() else None
        if mqc_html is not None:
            dst_mqc = out / "outputs/qc/multiqc_report.html"
            dst_mqc.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(mqc_html, dst_mqc)
            tables.append({
                "name": "Native MultiQC sequencing & alignment report",
                "analysis_id": "quality.multiqc_html",
                "chapter": 1,
                "rows": len(samples_list),
                "source": dst_mqc.relative_to(out).as_posix(),
                "view": dst_mqc.relative_to(out).as_posix(),
            })

    # 1. Load rMATS across all contrasts (`JCEC` preferred with fallback to `JC`) & qualify PS-DAS
    psi_sample_accumulator: dict[str, dict[str, float]] = {}
    event_meta: dict[str, dict[str, str]] = {}
    das_by_contrast: dict[str, pd.DataFrame] = {}
    pref_modes = (rmats_mode.upper(), "JCEC", "JC")

    # Authoritative rMATS sample order & arm orientation from bundle manifest.json (if present)
    manifest_rmats_order: dict[str, list[str]] = {}
    manifest_comp_arms: dict[str, tuple[str, str]] = {}
    if bundle_path is not None and (bundle_path / "manifest.json").is_file():
        try:
            bm = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
            if isinstance(bm.get("rmats_sample_order"), dict):
                manifest_rmats_order = {str(k): [str(x) for x in v] for k, v in bm["rmats_sample_order"].items() if isinstance(v, list)}
            for comp_m in bm.get("comparisons", []):
                if isinstance(comp_m, dict) and comp_m.get("comparison_id") and comp_m.get("rmats_group1") and comp_m.get("rmats_group2"):
                    manifest_comp_arms[str(comp_m["comparison_id"])] = (str(comp_m["rmats_group1"]), str(comp_m["rmats_group2"]))
        except Exception:
            pass

    for c_spec in contrasts_list:
        cid = c_spec["contrast_id"]
        rdir = rmats_map.get(cid) or next(iter(rmats_map.values()))
        num_lbl = str(c_spec["numerator"]["label"])
        den_lbl = str(c_spec["denominator"]["label"])
        g1_lbl, g2_lbl = manifest_comp_arms.get(cid, (num_lbl, den_lbl))
        rmats_inverted = (g1_lbl == den_lbl and g2_lbl == num_lbl)
        g1_sids = manifest_rmats_order.get(g1_lbl) or (c_spec["denominator"]["sample_ids"] if rmats_inverted else c_spec["numerator"]["sample_ids"])
        g2_sids = manifest_rmats_order.get(g2_lbl) or (c_spec["numerator"]["sample_ids"] if rmats_inverted else c_spec["denominator"]["sample_ids"])
        num_sids = g2_sids if rmats_inverted else g1_sids
        den_sids = g1_sids if rmats_inverted else g2_sids
        frames = []
        for et in EVENTS:
            f_et = next((rdir / f"{et}.MATS.{m}.txt" for m in pref_modes if (rdir / f"{et}.MATS.{m}.txt").is_file()), None)
            if f_et is None:
                continue
            raw = pd.read_csv(f_et, sep="\t")
            if raw.empty:
                continue
            raw["event_type"] = et
            gid_c = next((c for c in ("GeneID", "gene_id") if c in raw.columns), "ID")
            sym_c = next((c for c in ("geneSymbol", "gene_symbol") if c in raw.columns), gid_c)
            raw["gene_id"] = raw[gid_c].astype(str).str.strip('"')
            raw["gene_symbol"] = raw[sym_c].fillna(raw["gene_id"]).astype(str).str.strip('"')
            raw["event_id"] = _canonical_rmats_event_ids(raw, et)
            raw["comparison_id"] = cid
            raw["reverse_direction"] = rmats_inverted
            dpsi_raw = pd.to_numeric(raw["IncLevelDifference"], errors="coerce")
            raw["dpsi"] = -dpsi_raw if rmats_inverted else dpsi_raw
            raw["abs_dpsi"] = raw["dpsi"].abs()
            raw["fdr"] = pd.to_numeric(raw["FDR"], errors="coerce")
            raw["pvalue"] = pd.to_numeric(raw["PValue"], errors="coerce")
            raw["numerator_expected_samples"] = len(num_sids)
            raw["denominator_expected_samples"] = len(den_sids)
            frames.append(raw)
        if not frames:
            continue
        cat_df = pd.concat(frames, ignore_index=True)
        if species == "auto":
            species = "mouse" if cat_df["gene_id"].astype(str).str.startswith("ENSMUSG").any() else "human"

        qc_df = apply_stable_event_qc(cat_df, {})
        qc_df = significance_columns(qc_df, {"fdr": fdr_cutoff, "min_abs_dpsi": dpsi_cutoff})
        qc_df["report_das"] = qc_df["significant"].astype(bool)
        qc_df["report_rs_das"] = qc_df["stable_significant"].astype(bool)
        das_by_contrast[cid] = qc_df

        # Chapter 02 per-contrast Raw vs Precision-Supported AS composition
        render_bulk_as_composition_for_contrast(qc_df, cid, out, panels)

        for _, r in qc_df.iterrows():
            eid = str(r["event_id"])
            event_meta.setdefault(eid, {"gene_id": str(r["gene_id"]), "gene_symbol": str(r["gene_symbol"]), "event_type": str(r["event_type"])})
            p1 = parse_number_list(r.get("IncLevel1"))
            p2 = parse_number_list(r.get("IncLevel2"))
            rec = psi_sample_accumulator.setdefault(eid, {})
            for s_name, pv in zip(g1_sids, p1):
                if np.isfinite(pv):
                    rec[s_name] = float(pv)
            for s_name, pv in zip(g2_sids, p2):
                if np.isfinite(pv):
                    rec[s_name] = float(pv)

        comp_dir = out / "outputs/comparisons" / cid
        comp_dir.mkdir(parents=True, exist_ok=True)
        das_pq = comp_dir / "DAS-complete-original-and-report-flags.parquet"
        qc_df.to_parquet(das_pq, index=False)
        tables.append({
            "name": f"DAS complete table ({cid})",
            "analysis_id": "differential.as",
            "chapter": 3,
            "rows": len(qc_df),
            "columns": list(qc_df.columns),
            "source": das_pq.relative_to(out).as_posix(),
        })

        policy_data["stable_significant"][cid] = {
            "available": True,
            "events": int(qc_df["report_rs_das"].sum()),
            "genes": int(qc_df.loc[qc_df["report_rs_das"], "gene_symbol"].nunique()),
            "raw_events": len(qc_df),
            "read_supported_events": int(qc_df["passes_precision_support"].sum()),
        }
        policy_data["significant"][cid] = {
            "available": True,
            "events": int(qc_df["report_das"].sum()),
            "genes": int(qc_df.loc[qc_df["report_das"], "gene_symbol"].nunique()),
            "raw_events": len(qc_df),
        }

        for pol, col_flag, lbl in (("stable_significant", "report_rs_das", "PS-DAS"), ("significant", "report_das", "Ordinary DAS")):
            v_png = out / "figures/comparisons" / f"{cid}-AS-volcano-{pol}.png"
            _render_volcano(qc_df, "dpsi", "fdr", col_flag, "gene_symbol", "ΔPSI (Inclusion difference)", f"{lbl} Splicing Volcano — {cid}", v_png, is_deg=False)
            v_tsv = comp_dir / f"{cid}-AS-volcano-{pol}-plot-data.tsv"
            qc_df[["event_id", "gene_symbol", "event_type", "dpsi", "fdr", col_flag]].to_csv(v_tsv, sep="\t", index=False)
            panels.append({
                "id": f"{cid}-as-volcano-{pol}",
                "chapter": 3,
                "contrast_id": cid,
                "policy": pol,
                "title": f"{lbl} volcano ({int(qc_df[col_flag].sum()):,} significant events) | {cid}",
                "caption": f"FDR ≤ {fdr_cutoff}, |ΔPSI| ≥ {dpsi_cutoff}. Original caller P/FDR preserved.",
                "image": v_png.relative_to(out).as_posix(),
                "svg": v_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Plot data (TSV)", "path": v_tsv.relative_to(out).as_posix(), "unit": "splicing_event", "contrast_id": cid, "policy": pol}],
            })
            _render_das_class_direction_bar(qc_df, col_flag, lbl, cid, pol, out, panels)
            sub_sig = qc_df[qc_df[col_flag]].sort_values(["fdr", "abs_dpsi"], ascending=[True, False]).head(10)
            top10.append({
                "id": f"{cid}-top10-das-{pol}",
                "chapter": 3,
                "contrast_id": cid,
                "analysis": "DAS",
                "kind": f"DAS ({lbl})",
                "policy": pol,
                "unit": "splicing_event",
                "columns": ["gene_symbol", "event_id", "event_type", "dpsi", "fdr"],
                "rows": sub_sig[["gene_symbol", "event_id", "event_type", "dpsi", "fdr"]].to_dict("records"),
                "full_path": das_pq.relative_to(out).as_posix(),
            })

        # Chapter 05 GO + KEGG ORA on PS-DAS & Ordinary DAS
        for pol, col_flag, lbl in (("stable_significant", "report_rs_das", "PS-DAS"), ("significant", "report_das", "Ordinary DAS")):
            fg_s = set(qc_df.loc[qc_df[col_flag], "gene_symbol"].dropna().astype(str))
            if len(fg_s) < 2 and pol == "stable_significant":
                fg_s = set(qc_df.loc[qc_df["passes_precision_support"] & (qc_df["abs_dpsi"] >= dpsi_cutoff), "gene_symbol"].dropna().astype(str))
            bg_s = set(qc_df["gene_symbol"].dropna().astype(str))
            ora_df = _run_ora_from_bundled_catalog(fg_s, bg_s, species if species != "auto" else "human")
            if not ora_df.empty:
                ora_tsv = out / "outputs/enrichment" / f"{cid}-{pol}-go-kegg-ora.tsv"
                ora_tsv.parent.mkdir(parents=True, exist_ok=True)
                ora_df.to_csv(ora_tsv, sep="\t", index=False)
                ora_png = out / "figures/enrichment" / f"{cid}-{pol}-go-kegg-ora.png"
                _render_ora_faceted_figure(ora_df, ora_png, f"GO (BP / CC / MF) & KEGG Pathway Enrichment ({lbl}) — {cid}")
                panels.append({
                    "id": f"{cid}-ora-{pol}",
                    "chapter": 5,
                    "contrast_id": cid,
                    "policy": pol,
                    "title": f"GO & KEGG functional enrichment ({lbl} host genes) | {cid}",
                    "caption": f"Exact hypergeometric test with per-category Benjamini-Hochberg FDR across {len(bg_s):,} measured splicing background genes.",
                    "image": ora_png.relative_to(out).as_posix(),
                    "svg": ora_png.with_suffix(".svg").relative_to(out).as_posix(),
                    "data_links": [{"role": "plot_data", "label": "Complete ORA table", "path": ora_tsv.relative_to(out).as_posix(), "unit": "pathway_term", "contrast_id": cid, "policy": pol}],
                })
                if pol == "stable_significant":
                    tables.append({
                        "name": f"GO & KEGG ORA complete table ({cid})",
                        "analysis_id": "prioritize.enrichment",
                        "chapter": 5,
                        "rows": len(ora_df),
                        "columns": list(ora_df.columns),
                        "source": ora_tsv.relative_to(out).as_posix(),
                    })

    if species == "auto":
        species = "human"

    # Chapter 03 DEG + Lossless Orthogonal SUPPA2/DEXSeq/DTU Tables (ZERO unwanted volcanoes)
    # + Chapter 04 DEG×DAS Decoupling Quadrant & Multi-Layer Orthogonal Convergence UpSet
    deg_by_contrast, gene_sym_map, expr_mat, multilayer_evidence = ingest_bulk_multilayer_deg_dtu_dexseq(
        bundle_dir=bundle_path,
        deseq2_dir=Path(deseq2) if isinstance(deseq2, (str, Path)) else None,
        contrasts_list=contrasts_list,
        das_by_contrast=das_by_contrast,
        outdir=out,
        panels=panels,
        tables=tables,
        top10=top10,
        render_volcano_fn=_render_volcano,
        fdr_cutoff=fdr_cutoff,
    )
    for eid_k, em in event_meta.items():
        gid_k = str(em.get("gene_id", "")).split(".")[0]
        gs_k = str(em.get("gene_symbol", ""))
        if gid_k and gs_k and not gs_k.startswith("ENS"):
            gene_sym_map.setdefault(gid_k, gs_k)

    # Chapter 04 Cross-Comparison Overlaps, Exact Multi-Comparison Intersections & Direction Concordance
    render_ch4_cross_comparison_suite(das_by_contrast, contrasts_list, out, panels)

    # Chapter 02: Sample-level PSI matrix + Interactive `PSI_MATRIX.html` + Replicate Consistency + Clustermap/PCA
    all_sids = [s["sample_id"] for s in samples_list]
    psi_mat = pd.DataFrame.from_dict(psi_sample_accumulator, orient="index").reindex(columns=all_sids)
    psi_mat.index.name = "event_id"
    res_dir = out / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)
    psi_pq = res_dir / "sample_level_psi.parquet"
    psi_mat.reset_index().to_parquet(psi_pq, index=False)
    tables.append({
        "name": "Sample-level PSI matrix",
        "analysis_id": "landscape.bulk",
        "chapter": 2,
        "rows": len(psi_mat),
        "columns": ["event_id", *all_sids],
        "source": psi_pq.relative_to(out).as_posix(),
    })

    ev_labels = [f"{event_meta.get(str(e), {}).get('gene_symbol', str(e).split(':')[0])} · {e}" for e in psi_mat.index]
    build_interactive_psi_matrix_browser(psi_mat, [s["condition"] for s in samples_list], ev_labels, out, tables)
    render_replicate_consistency_panels(psi_mat, samples_list, out, panels, tables)

    sm_bulk_dir = Path(splicemanifold_dir) if (splicemanifold_dir and str(splicemanifold_dir) != "NO_SPLICEMANIFOLD" and Path(splicemanifold_dir).is_dir()) else None
    ingested_bulk_sm = False
    if sm_bulk_dir is not None and (sm_bulk_dir / "sample_manifold_coordinates.tsv").is_file():
        sm_rec = json.loads((sm_bulk_dir / "run_receipt.json").read_text(encoding="utf-8")) if (sm_bulk_dir / "run_receipt.json").is_file() else {}
        pca_tsv_dst = out / "outputs/landscape/psi-pca-coordinates.tsv"
        pca_tsv_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sm_bulk_dir / "sample_manifold_coordinates.tsv", pca_tsv_dst)
        for src_name, dst_name, pid, ttl, cap, dpath, dunit in [
            ("psi_pca.png", "psi-pca.png", "psi-pca", f"Global splicing PCA (PERMANOVA R² = {(sm_rec.get('permanova_r2') or 0.0):.3f}, P = {(sm_rec.get('permanova_p') or 1.0):.3g})", "Empirical-Bayes depth-stabilized splicing manifold across measurable cohort events.", pca_tsv_dst, "biological_sample"),
            ("psi_clustermap.png", "psi-heatmap.png", "psi-heatmap", "Depth-stabilized PSI Z-score clustermap", "Separated Sample Group vs Splicing Subtype legends with representative gene callouts.", psi_pq, "splicing_event"),
        ]:
            sp = sm_bulk_dir / "figures" / src_name
            if sp.is_file():
                dp = out / "outputs/landscape" / dst_name
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dp)
                if sp.with_suffix(".svg").is_file():
                    shutil.copy2(sp.with_suffix(".svg"), dp.with_suffix(".svg"))
                panels.append({
                    "id": pid, "chapter": 2, "contrast_id": "all", "title": ttl, "caption": cap,
                    "image": dp.relative_to(out).as_posix(),
                    "svg": dp.with_suffix(".svg").relative_to(out).as_posix() if dp.with_suffix(".svg").is_file() else None,
                    "data_links": [{"role": "plot_data", "label": "Plot data", "path": dpath.relative_to(out).as_posix(), "unit": dunit, "contrast_id": "all", "policy": None}],
                })
                ingested_bulk_sm = True

    if not ingested_bulk_sm and len(psi_mat) >= 8 and len(all_sids) >= 3:
        land = compute_beyond_jutils_landscape(
            psi_mat,
            samples_list,
            event_meta=event_meta,
            top_n=heatmap_top_n,
            manifold_top_n=manifold_top_n,
            min_present_fraction=0.65,
            prior_kappa=10.0,
            z_clip=2.5,
            run_umap=len(all_sids) >= 4,
            umap_neighbors=min(8, max(2, len(all_sids) - 1)),
        )
        if land.get("available"):
            hm_png = out / "outputs/landscape/psi-heatmap.png"
            pca_png = out / "outputs/landscape/psi-pca.png"
            render_editorial_psi_clustermap(land, hm_png, hm_png.with_suffix(".svg"), title=f"Depth-Stabilized Splicing Z-Score Clustermap — {study_name}", event_meta=event_meta)
            render_editorial_psi_pca(land, pca_png, pca_png.with_suffix(".svg"), title=f"Global Splicing PCA ({land['manifold_events']:,} Events) & PERMANOVA — {study_name}")
            pca_tsv = out / "outputs/landscape/psi-pca-coordinates.tsv"
            pd.DataFrame(land["samples"]).to_csv(pca_tsv, sep="\t", index=False)
            panels.append({
                "id": "psi-pca", "chapter": 2, "contrast_id": "all",
                "title": f"Global splicing PCA ({land['manifold_events']:,} events · PC1 {land['pca_variance_pct'][0]:.1f}%, PERMANOVA P = {land['permanova_p']:.3g})",
                "caption": "Empirical-Bayes depth-stabilized splicing manifold across measurable cohort events.",
                "image": pca_png.relative_to(out).as_posix(), "svg": pca_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "PCA coordinates (TSV)", "path": pca_tsv.relative_to(out).as_posix(), "unit": "biological_sample", "contrast_id": "all", "policy": None}],
            })
            panels.append({
                "id": "psi-heatmap", "chapter": 2, "contrast_id": "all",
                "title": f"Depth-stabilized PSI Z-score clustermap (top {land['heatmap_events']} rows; sample dendrogram across {land['manifold_events']:,} events)",
                "caption": "Separated Sample Group vs Splicing Subtype legends with right-axis representative gene callouts.",
                "image": hm_png.relative_to(out).as_posix(), "svg": hm_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Sample PSI matrix", "path": psi_pq.relative_to(out).as_posix(), "unit": "splicing_event", "contrast_id": "all", "policy": None}],
            })

    # Discover or fit `SpliceNet` modules first so `SpliceTrace` can also evaluate Module-to-RBP bridges
    sn_in_dir = Path(splicenet_dir) if (splicenet_dir and str(splicenet_dir) != "NO_SPLICENET" and Path(splicenet_dir).is_dir()) else None
    sn_members_df = pd.DataFrame()
    sn_scores_df = None
    sn_edges_df = None
    sn_ora_existing = None
    sn_mod_tsv_for_bridge: Path | None = None

    if sn_in_dir is not None and (sn_in_dir / "module_assignments.tsv").is_file():
        _sn_m = _safe_read_csv(sn_in_dir / "module_assignments.tsv", sep="\t")
        if not _sn_m.empty:
            sn_members_df = _sn_m
            sn_mod_tsv_for_bridge = sn_in_dir / "module_assignments.tsv"
        if (sn_in_dir / "module_scores.tsv").is_file():
            _sn_s = _safe_read_csv(sn_in_dir / "module_scores.tsv", sep="\t", index_col=0)
            sn_scores_df = _sn_s if not _sn_s.empty else None
        sn_edges_path = next((p for p in (sn_in_dir / "wgcna_fit/network_edges.tsv", sn_in_dir / "network_edges.tsv") if p.is_file()), None)
        if sn_edges_path is not None:
            _sn_e = _safe_read_csv(sn_edges_path, sep="\t")
            sn_edges_df = _sn_e if not _sn_e.empty else None
        sn_ora_path = next((p for p in (sn_in_dir / "module_ora.tsv", sn_in_dir / "module_pathway_enrichment.tsv") if p.is_file()), None)
        if sn_ora_path is not None:
            _sn_o = _safe_read_csv(sn_ora_path, sep="\t")
            sn_ora_existing = _sn_o if not _sn_o.empty else None
    elif len(psi_mat) >= 24 and len(all_sids) >= 4:
        try:
            from splicenet.evaluation import score_modules
            from splicenet.network import fit_splicenet
            from splicenet.qc_and_selection import select_events, unit_robust_audit

            meta_sn = pd.DataFrame(samples_list)[["sample_id", "biological_sample_id", "condition"]]
            masked, audit_sn = unit_robust_audit(psi_mat, meta_sn, None)
            min_mod = max(5, min(20, len(psi_mat) // 10))
            selected_sn, _ = select_events(masked, audit_sn, feature_cap=min(5000, len(psi_mat)), ranking_metric="variance")
            if len(selected_sn) >= max(10, min_mod * 2):
                sn_fit_dir = out / "outputs/splicenet_tables"
                sn_fit_dir.mkdir(parents=True, exist_ok=True)
                sn_members_df = fit_splicenet(selected_sn, sn_fit_dir / "wgcna_fit", power=0, min_module_size=min_mod, backend="python")
                sn_scores_df, _, _, _ = score_modules(selected_sn, sn_members_df)
                sn_mod_tsv_for_bridge = sn_fit_dir / "module_assignments.tsv"
                sn_members_df.to_csv(sn_mod_tsv_for_bridge, sep="\t", index=False)
                sn_edges_path = next((p for p in ((sn_fit_dir / "wgcna_fit/network_edges.tsv"),) if p.is_file()), None)
                if sn_edges_path is not None:
                    _sn_ef = _safe_read_csv(sn_edges_path, sep="\t")
                    sn_edges_df = _sn_ef if not _sn_ef.empty else None
        except Exception:
            pass

    # Chapter 07: Ingest precomputed Nextflow `SpliceTrace` dirs OR run standalone `SpliceTrace`
    st_map = _resolve_splicetrace_dirs(splicetrace_dirs, contrasts_list)
    norm_counts_f = bundle_path / "DEG/all.normalised_counts.tsv" if (bundle_path and (bundle_path / "DEG/all.normalised_counts.tsv").is_file()) else None
    prot_by_contrast: dict[str, pd.DataFrame] = {}
    conseq_by_contrast: dict[str, pd.DataFrame] = {}
    bridge_frames: list[pd.DataFrame] = []

    for c_spec in contrasts_list:
        cid = c_spec["contrast_id"]
        st_dir_c = st_map.get(cid)
        if st_dir_c is None and fasta is not None and gtf is not None and Path(fasta).is_file() and Path(gtf).is_file():
            from splicetrace.cli import main as splicetrace_cli_main
            rdir = rmats_map.get(cid) or next(iter(rmats_map.values()))
            st_run_dir = out / "outputs/splicetrace_runs" / cid
            st_run_dir.mkdir(parents=True, exist_ok=True)
            cli_args = [
                "--rmats", str(rdir),
                "--fasta", str(fasta),
                "--gtf", str(gtf),
                "--species", species,
                "--contrast-id", cid,
                "--output-dir", str(st_run_dir),
            ]
            if sn_mod_tsv_for_bridge is not None and sn_mod_tsv_for_bridge.is_file():
                cli_args.extend(["--splicenet-modules", str(sn_mod_tsv_for_bridge)])
            if norm_counts_f is not None:
                cli_args.extend(["--rbp-expression", str(norm_counts_f)])
            try:
                splicetrace_cli_main(cli_args)
                st_dir_c = st_run_dir
            except Exception:
                st_dir_c = None

        if st_dir_c is not None and st_dir_c.is_dir():
            p_df_c, csq_df_c, br_df_c = render_ch7_splicetrace_suite(
                st_dir=st_dir_c,
                cid=cid,
                species=species,
                outdir=out,
                panels=panels,
                tables=tables,
                motif_summaries=motif_summaries,
            )
            if not p_df_c.empty:
                prot_by_contrast[cid] = p_df_c
            if not csq_df_c.empty:
                conseq_by_contrast[cid] = csq_df_c
            if not br_df_c.empty:
                bridge_frames.append(br_df_c)

    bridge_df_all = pd.concat(bridge_frames, ignore_index=True).drop_duplicates() if bridge_frames else None

    # Chapter 06: Non-Circular Multi-Community Spatial Halo `SPLICE_NETWORK_EXPLORER.html` + Static Exhibits
    splicenet_meta: dict[str, Any] = {}
    if not sn_members_df.empty:
        ek_c = next((c for c in ("event_key", "event_id") if c in sn_members_df.columns), sn_members_df.columns[0])
        sn_members_df["gene_id"] = sn_members_df[ek_c].map(lambda k: event_meta.get(str(k), {}).get("gene_id", str(k).split(":")[0]))
        sn_members_df["gene_symbol"] = sn_members_df[ek_c].map(lambda k: event_meta.get(str(k), {}).get("gene_symbol", str(k).split(":")[0]))
        sn_members_df["event_type"] = sn_members_df[ek_c].map(lambda k: event_meta.get(str(k), {}).get("event_type", str(k).split(":")[0] if ":" in str(k) else "SE"))
        splicenet_meta = build_chapter6_splicenet_suite(
            mod_df_raw=sn_members_df,
            scores_df_raw=sn_scores_df,
            samples_list=samples_list,
            contrasts_spec=contrasts_list,
            psi_events_by_units=psi_mat,
            gene_sym_map=gene_sym_map,
            outdir=out,
            panels=panels,
            tables=tables,
            network_browsers=network_browsers,
            species=species,
            is_single_cell=False,
            bridge_df=bridge_df_all,
            edges_df=sn_edges_df,
            existing_pathways_df=sn_ora_existing,
        )

    # Chapter 08: Convergent 7-Pillar Scorecard + Self-Contained Interactive Candidate Gene Finder & Katana Native Sashimi 4-Track Workbench
    interactive_blocks: list[dict[str, Any]] = []
    build_chapter8_synthesis_and_workbench(
        contrasts_spec=contrasts_list,
        das_by_contrast=das_by_contrast,
        deg_by_contrast=deg_by_contrast,
        prot_by_contrast=prot_by_contrast,
        conseq_by_contrast=conseq_by_contrast,
        splicenet_meta=splicenet_meta,
        bridge_df=bridge_df_all,
        psi_events_by_units=psi_mat,
        expr_genes_by_units=expr_mat,
        samples_list=samples_list,
        gene_sym_map=gene_sym_map,
        outdir=out,
        panels=panels,
        tables=tables,
        top10_list=top10,
        interactive_blocks=interactive_blocks,
        is_single_cell=False,
        multilayer_evidence=multilayer_evidence,
        gtf_path=Path(gtf) if gtf else None,
        bams_dir=Path(bams) if bams else None,
    )

    return _finalize_consulting_report_and_zip(
        outdir=out,
        study_name=study_name,
        mode="bulk",
        samples_list=samples_list,
        contrasts_list=contrasts_list,
        panels=panels,
        tables=tables,
        top10=top10,
        policy_data=policy_data,
        motif_summaries=motif_summaries,
        species=species,
        network_browsers=network_browsers,
        interactive_blocks=interactive_blocks,
        dummy=dummy,
    )


def _render_sc_cohort_qc(samples_list: list[dict[str, Any]], png_path: Path, tsv_path: Path) -> Path | None:
    if not samples_list:
        return None
    df_s = pd.DataFrame(samples_list)
    grp = df_s.groupby(["condition", "biological_sample_id"], dropna=False).size().reset_index(name="n_cells")
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    grp.to_csv(tsv_path, sep="\t", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), dpi=220)
    fig.patch.set_facecolor("white")

    cond_counts = df_s["condition"].value_counts()
    cond_donors = df_s.groupby("condition")["biological_sample_id"].nunique()
    y0 = np.arange(len(cond_counts))[::-1]
    axes[0].barh(y0, cond_counts.values, color=NAVY, height=0.56, alpha=0.9)
    for yi, c_name, cnt in zip(y0, cond_counts.index, cond_counts.values):
        nd = int(cond_donors.get(c_name, 1))
        axes[0].text(cnt + max(cond_counts.values) * 0.02, yi, f"{int(cnt):,} cells ({nd} donors)", va="center", fontsize=8.3, fontweight="bold", color=NAVY)
    axes[0].set_yticks(y0, cond_counts.index.tolist(), fontweight="bold", color=NAVY)
    axes[0].set_xlim(0, float(cond_counts.max()) * 1.34)
    axes[0].set_xlabel("QC-Passed Cells / Nuclei", fontweight="bold", color=NAVY)
    axes[0].set_title("Cohort Composition & Donor Replication", loc="left", fontsize=10.2, fontweight="bold", color=NAVY)

    ct_counts = df_s["cell_type"].value_counts().head(12)
    y1 = np.arange(len(ct_counts))[::-1]
    axes[1].barh(y1, ct_counts.values, color=BLUE, height=0.58, alpha=0.9)
    for yi, cnt in zip(y1, ct_counts.values):
        axes[1].text(cnt + max(ct_counts.values) * 0.02, yi, f"{int(cnt):,}", va="center", fontsize=8.1, color=NAVY)
    axes[1].set_yticks(y1, [str(x)[:32] for x in ct_counts.index], fontsize=8.2, color=NAVY)
    axes[1].set_xlim(0, float(ct_counts.max()) * 1.22)
    axes[1].set_xlabel("QC-Passed Cells / Nuclei", fontweight="bold", color=NAVY)
    axes[1].set_title("Top Annotated Cell Populations", loc="left", fontsize=10.2, fontweight="bold", color=NAVY)
    fig.subplots_adjust(wspace=0.48)
    return _save_dual(fig, png_path)


def _render_sc_event_composition(ev_df: pd.DataFrame, png_path: Path, tsv_path: Path) -> Path | None:
    if ev_df.empty or "event_type" not in ev_df.columns:
        return None
    vc = ev_df["event_type"].astype(str).value_counts()
    comp_df = pd.DataFrame({"event_type": vc.index.tolist(), "event_count": vc.values.tolist()})
    comp_df["fraction"] = np.round(comp_df["event_count"] / max(1, comp_df["event_count"].sum()), 4)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    comp_df.to_csv(tsv_path, sep="\t", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=220)
    fig.patch.set_facecolor("white")
    et_cols = {"SE": BLUE, "A3SS": CERULEAN, "A5SS": "#9A64B0", "RI": "#D49A25", "MXE": CORAL}
    y = np.arange(len(comp_df))[::-1]
    colors = [et_cols.get(str(et), SLATE) for et in comp_df["event_type"]]
    ax.barh(y, comp_df["event_count"], color=colors, height=0.58, alpha=0.9)
    for yi, cnt, frac in zip(y, comp_df["event_count"], comp_df["fraction"]):
        ax.text(cnt + max(comp_df["event_count"]) * 0.015, yi, f"{int(cnt):,} ({frac * 100:.1f}%)", va="center", fontsize=8.5, fontweight="bold", color=NAVY)
    ax.set_yticks(y, comp_df["event_type"].tolist(), fontweight="bold", color=NAVY)
    ax.set_xlim(0, float(comp_df["event_count"].max()) * 1.24)
    ax.set_xlabel("Quantified Alternative Splicing Events", fontweight="bold", color=NAVY)
    ax.set_title(f"Single-Cell Alternative Splicing Event Repertoire (Total = {int(comp_df['event_count'].sum()):,})", loc="left", fontsize=10.6, fontweight="bold", color=NAVY)
    return _save_dual(fig, png_path)


def _orient_matrix_events_by_cells(df_in: pd.DataFrame, sample_id_set: set[str], was_dataframe: bool) -> pd.DataFrame:
    df = df_in.copy()
    if not was_dataframe and len(df.columns) > 0 and str(df.columns[0]) not in sample_id_set:
        df = df.set_index(df.columns[0])
    if len(sample_id_set & set(df.columns.astype(str))) == 0 and len(sample_id_set & set(df.index.astype(str))) > 0:
        df = df.T
    return df


def run_single_cell(
    *,
    outdir: str | Path,
    events: str | Path | pd.DataFrame,
    cells: str | Path | pd.DataFrame,
    psi_matrix: str | Path | pd.DataFrame | None = None,
    support_matrix: str | Path | pd.DataFrame | None = None,
    expression: str | Path | pd.DataFrame | None = None,
    gtf: str | Path | None = None,
    bams: str | Path | None = None,
    sc_bam: str | Path | None = None,
    sc_bc_tag: str = "BC",
    sc_barcode_shares: str | Path | None = None,
    splicemanifold_dir: str | Path | None = None,
    scsplicenet_dir: str | Path | None = None,
    scsplicetrace_dir: str | Path | None = None,
    subtypes_dirs: list[str | Path] | None = None,
    study_name: str = "Katana Single-Cell Splicing Study",
    species: str = "human",
    fdr_cutoff: float = 0.05,
    dpsi_cutoff: float = 0.10,
    dummy: bool = False,
) -> dict[str, Any]:
    """Execute and assemble the complete Original-Katana-compliant Single-Cell Suite (Ch 01..07 + Appendix A)."""
    from katana_next.science.psi_landscape import (
        compute_beyond_jutils_landscape,
        compute_dual_expression_vs_psi_umap,
        render_editorial_dual_psi_umap,
        render_editorial_psi_clustermap,
        render_editorial_psi_pca,
    )

    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    ev_df = events.copy() if isinstance(events, pd.DataFrame) else _read_tabular(Path(events))
    cell_df = cells.copy() if isinstance(cells, pd.DataFrame) else _read_tabular(Path(cells))

    cid_c = next((c for c in ("cell_id", "sample_id", "barcode") if c in cell_df.columns), cell_df.columns[0])
    cond_c = next((c for c in ("condition", "state", "activation_state", "group") if c in cell_df.columns), cell_df.columns[1])
    donor_c = next((c for c in ("donor_id", "mouse_id", "biological_sample_id", "subject") if c in cell_df.columns), None)
    ct_c = next((c for c in ("atlas_subclass", "cell_type", "cluster", "subset") if c in cell_df.columns), cond_c)

    samples_list = [
        {
            "sample_id": str(r[cid_c]).strip(),
            "condition": str(r[cond_c]).strip(),
            "biological_sample_id": str(r[donor_c]).strip() if donor_c and pd.notna(r[donor_c]) and str(r[donor_c]).strip() else str(r[cid_c]).strip(),
            "donor_id": str(r[donor_c]).strip() if donor_c and pd.notna(r[donor_c]) and str(r[donor_c]).strip() else "D1",
            "cell_type": str(r[ct_c]).strip(),
            "activation_state": str(r[cond_c]).strip(),
        }
        for _, r in cell_df.iterrows()
    ]
    sample_id_set = {s["sample_id"] for s in samples_list}

    # Merge primary donor-aware event contrast statistics from scSpliceTrace if available
    st_dir = Path(scsplicetrace_dir) if (scsplicetrace_dir and str(scsplicetrace_dir) != "NO_SCSPLICETRACE" and Path(scsplicetrace_dir).is_dir()) else None
    st_dirs_by_contrast: dict[str, Path] = {}

    def _merge_st_events(base_ev: pd.DataFrame, trace_dir: Path, forced_cid: str | None = None) -> tuple[pd.DataFrame, str | None]:
        st_cmp = forced_cid
        st_sum_f = trace_dir / "scsplicetrace_summary.json"
        if st_cmp is None and st_sum_f.is_file():
            try:
                st_j = json.loads(st_sum_f.read_text(encoding="utf-8"))
                ng = st_j.get("numerator_group") or st_j.get("numerator")
                dg = st_j.get("denominator_group") or st_j.get("denominator")
                if ng and dg:
                    st_cmp = f"{ng}_vs_{dg}"
            except Exception:
                pass
        if (trace_dir / "scsplicetrace_event_contrasts.tsv").is_file() and "event_id" in base_ev.columns:
            st_ev = _safe_read_csv(trace_dir / "scsplicetrace_event_contrasts.tsv", sep="\t")
            if not st_ev.empty and "event_id" in st_ev.columns:
                m_cols = [c for c in ("event_id", "raw_dpsi", "z_shift", "p_value", "fdr", "eligible") if c in st_ev.columns]
                ev_m = base_ev.merge(st_ev[m_cols], on="event_id", how="left", suffixes=("_prelim", ""))
                if "raw_dpsi" in ev_m.columns:
                    ev_m["dpsi"] = ev_m["raw_dpsi"].combine_first(ev_m.get("dpsi", pd.Series(dtype=float)))
                if "p_value" in ev_m.columns:
                    ev_m["pvalue"] = ev_m["p_value"].combine_first(ev_m.get("pvalue", pd.Series(dtype=float)))
                if "fdr_prelim" in ev_m.columns and "fdr" in ev_m.columns:
                    ev_m["fdr"] = ev_m["fdr"].combine_first(ev_m["fdr_prelim"])
                if st_cmp:
                    ev_m["comparison_id"] = st_cmp
                return ev_m, st_cmp
        if st_cmp:
            base_ev = base_ev.copy()
            base_ev["comparison_id"] = st_cmp
        return base_ev, st_cmp

    if st_dir is not None:
        ev_df, primary_cid = _merge_st_events(ev_df, st_dir)
        if primary_cid:
            st_dirs_by_contrast[primary_cid] = st_dir

    cmp_col = next((c for c in ("comparison_id", "contrast_id", "contrast") if c in ev_df.columns), None)
    if cmp_col is None or ev_df[cmp_col].astype(str).eq("Cohort_All").all():
        cond_vals = list(dict.fromkeys(s["condition"] for s in samples_list))
        default_cid = f"{cond_vals[0]}_vs_{cond_vals[1]}" if len(cond_vals) >= 2 else "StateA_vs_StateB"
        ev_df["comparison_id"] = default_cid
        cmp_col = "comparison_id"
    if st_dir is not None and not st_dirs_by_contrast:
        first_cid = str(ev_df[cmp_col].astype(str).iloc[0])
        st_dirs_by_contrast[first_cid] = st_dir

    # Optional multi-subtype / multi-contrast ingestion (`--subtypes-dirs` from SC_SUBTYPE_DOWNSTREAM)
    subtype_cell_sets: dict[str, tuple[list[str], list[str], str, str]] = {}
    sn_dirs_by_contrast: dict[str, Path] = {}
    sub_pop_cells: dict[str, set[str]] = {}
    if subtypes_dirs:
        base_has_stats = (
            st_dir is not None
            or any(
                c in ev_df.columns and pd.to_numeric(ev_df[c], errors="coerce").notna().any()
                for c in ("pvalue", "p_value", "fdr", "FDR")
            )
        )
        sub_frames: list[pd.DataFrame] = [ev_df] if base_has_stats else []
        for sd_raw in subtypes_dirs:
            if not sd_raw or str(sd_raw) == "NO_SUBTYPES":
                continue
            sd = Path(sd_raw)
            if not sd.is_dir():
                continue
            sub_cid = sd.name
            sub_ev_f = sd / "sc_matrices/sc_events.tsv"
            sub_cells_f = sd / "sc_matrices/sc_cells.tsv"
            sub_st_d = sd / "scsplicetrace"
            sub_sn_d = sd / "scsplicenet"
            if (sub_sn_d / "scsplicenet_modules.tsv").is_file():
                sn_dirs_by_contrast[sub_cid] = sub_sn_d
            if sub_ev_f.is_file():
                s_ev = _safe_read_csv(sub_ev_f, sep="\t")
                if not s_ev.empty:
                    if sub_st_d.is_dir():
                        s_ev, _ = _merge_st_events(s_ev, sub_st_d, forced_cid=sub_cid)
                        st_dirs_by_contrast[sub_cid] = sub_st_d
                    else:
                        s_ev["comparison_id"] = sub_cid
                    s_ev[cmp_col] = sub_cid
                    sub_frames.append(s_ev)
            if sub_cells_f.is_file():
                sc_sub = _safe_read_csv(sub_cells_f, sep="\t")
                if not sc_sub.empty:
                    sc_id_c = next((c for c in ("cell_id", "sample_id") if c in sc_sub.columns), sc_sub.columns[0])
                    sc_g_c = next((c for c in ("condition", "group") if c in sc_sub.columns), sc_sub.columns[1])
                    sub_pop_cells[sub_cid] = set(sc_sub[sc_id_c].astype(str).str.strip())
                    parts_s = sub_cid.split("__")[-1].split("_vs_", 1) if "_vs_" in sub_cid else ["Group1", "Group2"]
                    n_ids = sc_sub.loc[sc_sub[sc_g_c].astype(str).eq(parts_s[0]), sc_id_c].astype(str).tolist()
                    d_ids = sc_sub.loc[sc_sub[sc_g_c].astype(str).eq(parts_s[1]), sc_id_c].astype(str).tolist()
                    subtype_cell_sets[sub_cid] = (n_ids, d_ids, parts_s[0], parts_s[1])
        if sub_frames:
            ev_df = pd.concat(sub_frames, ignore_index=True)

    contrasts_list = []
    for cid_v in list(dict.fromkeys(ev_df[cmp_col].astype(str))):
        if cid_v in subtype_cell_sets:
            n_ids, d_ids, n_lbl, d_lbl = subtype_cell_sets[cid_v]
        else:
            tail = cid_v.split("__")[-1]
            parts = tail.split("_vs_", 1) if "_vs_" in tail else [tail, "Baseline"]
            n_lbl, d_lbl = parts[0], parts[1]
            pop_prefix = cid_v.split("__")[0] if "__" in cid_v else None
            n_ids = [s["sample_id"] for s in samples_list if s["condition"] == n_lbl and (pop_prefix is None or s["cell_type"] == pop_prefix)] or [s["sample_id"] for s in samples_list if s["condition"] == n_lbl] or [samples_list[0]["sample_id"]]
            d_ids = [s["sample_id"] for s in samples_list if s["condition"] == d_lbl and (pop_prefix is None or s["cell_type"] == pop_prefix)] or [s["sample_id"] for s in samples_list if s["condition"] == d_lbl] or [samples_list[-1]["sample_id"]]
        contrasts_list.append({
            "contrast_id": cid_v,
            "label": cid_v.replace("__", " · ").replace("_vs_", " vs "),
            "numerator": {"label": n_lbl, "sample_ids": n_ids},
            "denominator": {"label": d_lbl, "sample_ids": d_ids},
        })

    # Orient optional single-cell matrices to `(features × cells)`
    p_df = _orient_matrix_events_by_cells(psi_matrix if isinstance(psi_matrix, pd.DataFrame) else _read_tabular(Path(psi_matrix)), sample_id_set, isinstance(psi_matrix, pd.DataFrame)) if psi_matrix is not None else None
    sup_df = _orient_matrix_events_by_cells(support_matrix if isinstance(support_matrix, pd.DataFrame) else _read_tabular(Path(support_matrix)), sample_id_set, isinstance(support_matrix, pd.DataFrame)) if support_matrix is not None else None
    expr_df = _orient_matrix_events_by_cells(expression if isinstance(expression, pd.DataFrame) else _read_tabular(Path(expression)), sample_id_set, isinstance(expression, pd.DataFrame)) if expression is not None else None

    panels: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    top10: list[dict[str, Any]] = []
    network_browsers: list[dict[str, Any]] = []
    motif_summaries: list[dict[str, Any]] = []
    policy_data: dict[str, Any] = {"stable_significant": {}, "significant": {}}

    # Chapter 01 Exhibit 1.1: Cohort Design, Donor Replication & Cell Population QC
    qc_png = out / "figures/qc/sc-biological-design.png"
    qc_tsv = out / "outputs/qc/sc-donor-condition-counts.tsv"
    if _render_sc_cohort_qc(samples_list, qc_png, qc_tsv) is not None:
        panels.append({
            "id": "biological-design",
            "chapter": 1,
            "contrast_id": "all",
            "title": f"Single-cell cohort design & biological donor distribution ({len(samples_list):,} cells)",
            "caption": "QC-passed cells stratified by experimental condition, independent biological donor/replicate, and annotated cell population.",
            "image": qc_png.relative_to(out).as_posix(),
            "svg": qc_png.with_suffix(".svg").relative_to(out).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Donor & condition cell counts (TSV)", "path": qc_tsv.relative_to(out).as_posix(), "unit": "biological_donor", "contrast_id": "all", "policy": None}],
        })

    # Chapter 02 Exhibit 2.1: Event-Type Repertoire Composition
    comp_png = out / "figures/landscape/sc-event-composition.png"
    comp_tsv = out / "outputs/landscape/sc-event-composition.tsv"
    if _render_sc_event_composition(ev_df.drop_duplicates("event_id") if "event_id" in ev_df.columns else ev_df, comp_png, comp_tsv) is not None:
        panels.append({
            "id": "sc-event-composition",
            "chapter": 2,
            "contrast_id": "all",
            "title": f"Single-cell alternative splicing repertoire ({ev_df['event_id'].nunique() if 'event_id' in ev_df.columns else len(ev_df):,} events)",
            "caption": "Distribution of canonical alternative splicing classes quantified across QC-passed single cells.",
            "image": comp_png.relative_to(out).as_posix(),
            "svg": comp_png.with_suffix(".svg").relative_to(out).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Event class counts (TSV)", "path": comp_tsv.relative_to(out).as_posix(), "unit": "event_type", "contrast_id": "all", "policy": None}],
        })

    # Chapter 01 Library Depth QC + Chapter 02 Top-Event Distributions + `PSI_MATRIX.html` + Consistency
    # (Skip redundant 6-panel UMAPs so SpliceManifold's unified experiment-wide 3-panel Dual UMAP is the single clean UMAP exhibit!)
    pb_psi, pb_samples = pd.DataFrame(), pd.DataFrame()
    if p_df is not None and not p_df.empty:
        pb_psi, pb_samples = render_sc_umap_and_qc_suite(
            psi_events_by_cells=p_df,
            support_events_by_cells=sup_df,
            expr_genes_by_cells=expr_df,
            samples_list=samples_list,
            ev_df=ev_df,
            outdir=out,
            panels=panels,
            tables=tables,
            skip_duplicate_umaps=True,
        )

    # Chapter 02: SpliceManifold Precomputed OR On-the-Fly Landscape (Dual UMAP Δ_kNN, PERMANOVA PCA, Clustermap)
    sm_dir = Path(splicemanifold_dir) if (splicemanifold_dir and str(splicemanifold_dir) != "NO_SPLICEMANIFOLD" and Path(splicemanifold_dir).is_dir()) else None
    ingested_sm = False
    if sm_dir is not None and (sm_dir / "sc_cell_manifold_coordinates.tsv").is_file():
        sm_coord_dst = out / "outputs/landscape/sc_cell_manifold_coordinates.tsv"
        sm_coord_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sm_dir / "sc_cell_manifold_coordinates.tsv", sm_coord_dst)
        sm_rec = json.loads((sm_dir / "run_receipt.json").read_text(encoding="utf-8")) if (sm_dir / "run_receipt.json").is_file() else {}
        for stem, pid, ptitle, pcap in [
            ("sc_dual_psi_umap", "sc-dual-psi-umap", f"Procrustes Precision-Kernel Event-PSI UMAP vs Expression UMAP (Donor LISI = {sm_rec.get('median_donor_lisi', 0.0):.2f})", "Three-panel single-cell manifold comparing gene expression UMAP, Binomial Fisher-precision weighted Event-PSI UMAP, and per-cell modality divergence Δ_kNN."),
            ("sc_psi_pca", "sc-psi-pca", f"Empirical-Bayes Shrunk PSI PCA (PERMANOVA R² = {sm_rec.get('permanova_r2', 0.0):.4f}, P = {sm_rec.get('permanova_p', 1.0):.4g})", "Support-shrunk single-cell splicing manifold PCA with 95% condition covariance ellipses."),
            ("sc_psi_clustermap", "sc-psi-clustermap", "Empirical-Bayes Precision-Shrunk Single-Cell PSI Clustermap", "Top ExcessBioVar alternative splicing events clustered across cells after Binomial support shrinkage."),
        ]:
            src_png = sm_dir / "figures" / f"{stem}.png"
            if src_png.is_file():
                dst_png = out / "figures/landscape" / f"{stem}.png"
                dst_png.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_png, dst_png)
                if src_png.with_suffix(".svg").is_file():
                    shutil.copy2(src_png.with_suffix(".svg"), dst_png.with_suffix(".svg"))
                panels.append({
                    "id": pid, "chapter": 2, "contrast_id": "all", "title": ptitle, "caption": pcap,
                    "image": dst_png.relative_to(out).as_posix(),
                    "svg": dst_png.with_suffix(".svg").relative_to(out).as_posix() if dst_png.with_suffix(".svg").is_file() else None,
                    "data_links": [{"role": "plot_data", "label": "Manifold cell coordinates (TSV)", "path": sm_coord_dst.relative_to(out).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
                })
                ingested_sm = True

    if not ingested_sm and p_df is not None and not p_df.empty:
        ev_meta_fb: dict[str, dict[str, str]] = {}
        g_fb_c = next((c for c in ("gene_symbol", "gene_name", "gene_id") if c in ev_df.columns), ev_df.columns[0])
        if "event_id" in ev_df.columns:
            for _, r_m in ev_df.drop_duplicates("event_id").iterrows():
                ev_meta_fb[str(r_m["event_id"])] = {
                    "gene_symbol": str(r_m.get(g_fb_c, "")),
                    "gene_id": str(r_m.get("gene_id", r_m.get(g_fb_c, ""))),
                    "event_type": str(r_m.get("event_type", "SE")),
                }
        min_pres = 0.05 if len(samples_list) > 80 else 0.20
        psi_land = compute_beyond_jutils_landscape(
            p_df,
            samples_list,
            event_meta=ev_meta_fb,
            depth_matrix=sup_df,
            min_present_fraction=min_pres,
            run_umap=True,
            umap_neighbors=min(12, max(3, len(samples_list) - 2)),
        )
        if psi_land.get("available"):
            s_ids_sc = [s["sample_id"] for s in samples_list]
            pca_png = out / "figures/landscape/sc-psi-pca.png"
            cm_png = out / "figures/landscape/sc-psi-clustermap.png"
            pca_tsv = out / "outputs/landscape/sc-psi-pca-cells.tsv"
            pca_tsv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(psi_land["samples"]).to_csv(pca_tsv, sep="\t", index=False)
            render_editorial_psi_pca(psi_land, pca_png, pca_png.with_suffix(".svg"))
            render_editorial_psi_clustermap(psi_land, cm_png, cm_png.with_suffix(".svg"), event_meta=ev_meta_fb)
            panels.append({
                "id": "sc-psi-pca", "chapter": 2, "contrast_id": "all",
                "title": f"Empirical-Bayes Shrunk PSI PCA (PERMANOVA R² = {psi_land['permanova_r2']:.4f}, P = {psi_land['permanova_p']:.4g})",
                "caption": "Support-shrunk single-cell splicing manifold PCA with 95% condition covariance ellipses.",
                "image": pca_png.relative_to(out).as_posix(), "svg": pca_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Cell PCA coordinates (TSV)", "path": pca_tsv.relative_to(out).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
            })
            panels.append({
                "id": "sc-psi-clustermap", "chapter": 2, "contrast_id": "all",
                "title": "Empirical-Bayes Precision-Shrunk Single-Cell PSI Clustermap",
                "caption": "Top ExcessBioVar alternative splicing events clustered across cells after Binomial support shrinkage.",
                "image": cm_png.relative_to(out).as_posix(), "svg": cm_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Cell PCA coordinates (TSV)", "path": pca_tsv.relative_to(out).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
            })
            expr_cells_rows = expr_df.T if (expr_df is not None and not expr_df.empty) else p_df.T.fillna(0.5)
            dual = compute_dual_expression_vs_psi_umap(expr_cells_rows, psi_land, k_neighbors=min(10, max(3, len(s_ids_sc) // 3)))
            if dual.get("cells"):
                umap_png = out / "figures/landscape/sc-dual-psi-umap.png"
                umap_png.parent.mkdir(parents=True, exist_ok=True)
                render_editorial_dual_psi_umap(dual, umap_png, umap_png.with_suffix(".svg"))
                umap_tsv = out / "outputs/landscape/sc-dual-psi-umap-cells.tsv"
                pd.DataFrame(dual["cells"]).to_csv(umap_tsv, sep="\t", index=False)
                panels.append({
                    "id": "sc-dual-psi-umap", "chapter": 2, "contrast_id": "all",
                    "title": f"Procrustes Precision-Kernel Event-PSI UMAP vs Expression UMAP (Donor LISI = {dual['median_donor_lisi']:.2f})",
                    "caption": "Three-panel single-cell manifold comparing gene expression UMAP, Binomial Fisher-precision weighted Event-PSI UMAP, and per-cell modality divergence Δ_kNN.",
                    "image": umap_png.relative_to(out).as_posix(), "svg": umap_png.with_suffix(".svg").relative_to(out).as_posix(),
                    "data_links": [{"role": "plot_data", "label": "Cell coordinates & Δ_kNN (TSV)", "path": umap_tsv.relative_to(out).as_posix(), "unit": "single_cell", "contrast_id": "all", "policy": None}],
                })

    # Chapter 03: Single-Cell Differential Splicing (`DAS`) + Donor-Pseudobulk `DEG` + Chapter 04 Overlap + Chapter 05 ORA
    res_dir = out / "outputs/results"
    res_dir.mkdir(parents=True, exist_ok=True)
    ev_pq = res_dir / "sc_differential_splicing.parquet"
    ev_df.to_parquet(ev_pq, index=False)
    tables.append({
        "name": "Single-cell donor-aware differential splicing events",
        "analysis_id": "differential.sc_as",
        "chapter": 3,
        "rows": len(ev_df),
        "columns": list(ev_df.columns),
        "source": ev_pq.relative_to(out).as_posix(),
    })

    q_c = next((c for c in ("fdr", "FDR", "q_value", "padj") if c in ev_df.columns), None)
    p_c = next((c for c in ("pvalue", "p_value", "PValue") if c in ev_df.columns), None)
    d_c = next((c for c in ("dpsi", "delta_psi", "raw_dpsi", "effect") if c in ev_df.columns), None)
    g_c = next((c for c in ("gene_symbol", "gene_name", "gene_id") if c in ev_df.columns), ev_df.columns[0])

    sc_das_by_contrast: dict[str, pd.DataFrame] = {}
    sc_das_genes: dict[str, set[str]] = {}
    gene_sym_map_sc: dict[str, str] = {}
    if {"gene_id", g_c} <= set(ev_df.columns):
        for gid_v, gsym_v in zip(ev_df["gene_id"].astype(str), ev_df[g_c].astype(str)):
            if gsym_v and gsym_v != "nan":
                gene_sym_map_sc.setdefault(gid_v.split(".")[0], gsym_v)
                gene_sym_map_sc.setdefault(gid_v, gsym_v)

    for c_spec in contrasts_list:
        cid = c_spec["contrast_id"]
        sub = ev_df[ev_df[cmp_col].astype(str) == cid].copy()
        if d_c and (q_c or p_c):
            eff_col = q_c if (q_c and pd.to_numeric(sub[q_c], errors="coerce").notna().any()) else p_c
            sub["_sig_fdr"] = (pd.to_numeric(sub[q_c], errors="coerce").fillna(1.0) <= fdr_cutoff) & (pd.to_numeric(sub[d_c], errors="coerce").abs().fillna(0.0) >= dpsi_cutoff) if q_c else False
            sub["_sig_nom"] = (pd.to_numeric(sub[p_c], errors="coerce").fillna(1.0) <= 0.05) & (pd.to_numeric(sub[d_c], errors="coerce").abs().fillna(0.0) >= dpsi_cutoff) if p_c else sub["_sig_fdr"]
            n_fdr_sig = int(sub["_sig_fdr"].sum())
            n_nom_sig = int(sub["_sig_nom"].sum())

            use_nom_axis = (n_fdr_sig == 0) and (p_c is not None) and (n_nom_sig > 0)
            plot_q_col = p_c if use_nom_axis else eff_col
            sub["_sig"] = sub["_sig_nom"] if use_nom_axis else sub["_sig_fdr"]
            sub["gene_symbol"] = sub[g_c].astype(str)
            sub["dpsi"] = pd.to_numeric(sub[d_c], errors="coerce")
            sub["fdr"] = pd.to_numeric(sub[q_c], errors="coerce") if q_c else pd.to_numeric(sub[plot_q_col], errors="coerce")
            sub["report_rs_das"] = sub["_sig"].astype(bool)
            sc_das_by_contrast[cid] = sub
            sc_das_genes[cid] = set(sub.loc[sub["_sig"], "gene_symbol"].dropna().astype(str))

            v_png = out / "figures/comparisons" / f"{cid}-sc-as-volcano.png"
            v_title = (
                f"Donor-Aware DAS Volcano ({n_fdr_sig:,} FDR ≤ {fdr_cutoff:g} events) | {cid}"
                if not use_nom_axis else
                f"Donor-Aware DAS Volcano ({n_nom_sig:,} nominal P ≤ 0.05, |ΔPSI| ≥ {dpsi_cutoff:g}; 0 BH-FDR) | {cid}"
            )
            _render_volcano(sub, d_c, plot_q_col, "_sig", g_c, "Single-cell ΔPSI", v_title, v_png, is_deg=False)
            v_tsv = out / "outputs/comparisons" / cid / f"{cid}-sc-volcano-plot-data.tsv"
            v_tsv.parent.mkdir(parents=True, exist_ok=True)
            keep_v_cols = list(dict.fromkeys([c for c in ("event_id", "event_type", g_c, d_c, p_c, q_c, "_sig") if c and c in sub.columns]))
            sub[keep_v_cols].to_csv(v_tsv, sep="\t", index=False)
            panels.append({
                "id": f"{cid}-sc-volcano",
                "chapter": 3,
                "contrast_id": cid,
                "policy": None,
                "title": v_title,
                "caption": "Donor-aware single-cell differential splicing preserving biological replicate degrees of freedom.",
                "image": v_png.relative_to(out).as_posix(),
                "svg": v_png.with_suffix(".svg").relative_to(out).as_posix(),
                "data_links": [{"role": "plot_data", "label": "Volcano data (TSV)", "path": v_tsv.relative_to(out).as_posix(), "unit": "splicing_event", "contrast_id": cid, "policy": None}],
            })
            _render_das_class_direction_bar(sub, "_sig", "Single-Cell DAS", cid, None, out, panels)
            policy_data["stable_significant"][cid] = {"available": True, "events": n_fdr_sig, "genes": int(sub.loc[sub["_sig_fdr"], g_c].nunique()) if n_fdr_sig else 0}

            cand_top = sub[sub["_sig"]].copy()
            if not cand_top.empty:
                sort_p = p_c if (p_c and p_c in cand_top.columns) else plot_q_col
                cand_top["_abs_d"] = pd.to_numeric(cand_top[d_c], errors="coerce").abs().fillna(0.0)
                cand_top = cand_top.sort_values([sort_p, "_abs_d"], ascending=[True, False]).head(10)
                t_cols = [c for c in ("gene_symbol", "event_type", "event_id", "dpsi", "pvalue", "fdr") if c in cand_top.columns]
                top10.append({
                    "id": f"{cid}-sc-top10",
                    "chapter": 3,
                    "contrast_id": cid,
                    "analysis": "DAS",
                    "kind": "Single-Cell Differential Splicing (FDR ≤ 0.05)" if n_fdr_sig > 0 else "Single-Cell Candidate DAS Events (Nominal P ≤ 0.05)",
                    "unit": "splicing_event",
                    "columns": t_cols,
                    "rows": cand_top[t_cols].to_dict(orient="records"),
                    "full_path": ev_pq.relative_to(out).as_posix(),
                })

            fg_sc = sc_das_genes[cid]
            if fg_sc:
                ora_sc = _run_ora_from_bundled_catalog(fg_sc, set(sub[g_c].dropna().astype(str)), species)
                if not ora_sc.empty:
                    ora_png = out / "figures/enrichment" / f"{cid}-sc-go-kegg-ora.png"
                    ora_tsv = out / "outputs/enrichment" / f"{cid}-sc-go-kegg-ora.tsv"
                    ora_tsv.parent.mkdir(parents=True, exist_ok=True)
                    ora_sc.to_csv(ora_tsv, sep="\t", index=False)
                    _render_ora_faceted_figure(ora_sc, ora_png, f"Single-Cell GO & KEGG Enrichment — {cid}")
                    panels.append({
                        "id": f"{cid}-sc-ora",
                        "chapter": 5,
                        "contrast_id": cid,
                        "policy": None,
                        "title": f"Single-cell GO & KEGG pathway enrichment | {cid}",
                        "caption": "Exact hypergeometric enrichment across GO (BP, CC, MF) and KEGG pathways.",
                        "image": ora_png.relative_to(out).as_posix(),
                        "svg": ora_png.with_suffix(".svg").relative_to(out).as_posix(),
                        "data_links": [{"role": "plot_data", "label": "Complete ORA table", "path": ora_tsv.relative_to(out).as_posix(), "unit": "pathway_term", "contrast_id": cid, "policy": None}],
                    })
                    tables.append({
                        "name": f"Single-cell GO & KEGG ORA complete table ({cid})",
                        "analysis_id": "prioritize.enrichment",
                        "chapter": 5,
                        "rows": len(ora_sc),
                        "columns": list(ora_sc.columns),
                        "source": ora_tsv.relative_to(out).as_posix(),
                    })

    # Donor-Pseudobulk DEG (Chapter 03) + Host DEG × Single-Cell DAS Decoupling Quadrant (Chapter 04)
    sc_deg_by_contrast, pb_expr_df = render_sc_donor_deg_and_overlap(
        expr_genes_by_cells=expr_df,
        samples_list=samples_list,
        contrasts_list=contrasts_list,
        das_genes_by_contrast=sc_das_genes,
        outdir=out,
        panels=panels,
        tables=tables,
        top10=top10,
        render_volcano_fn=_render_volcano,
        species=species,
        run_ora_fn=_run_ora_from_bundled_catalog,
        fdr_cutoff=fdr_cutoff,
        sc_das_by_contrast=sc_das_by_contrast,
    )

    # Chapter 04: Cell-Type Stratified DAS Summary + Cross-Subtype Overlaps & Direction Concordance (when ≥2 contrasts exist)
    if len(sc_das_by_contrast) >= 2:
        render_sc_celltype_stratification_summary(sc_das_by_contrast, contrasts_list, out, panels, tables)
        render_ch4_cross_comparison_suite(
            sc_das_by_contrast,
            contrasts_list,
            out,
            panels,
            policies=(("common", "report_rs_das", "Single-Cell DAS"),),
        )

    # Chapter 07: scSpliceTrace Positional Splicing Map + PWM Logos + Module-RBP Bridge + 5 NMD/UniProt Plots!
    sc_prot_by_contrast: dict[str, pd.DataFrame] = {}
    sc_conseq_by_contrast: dict[str, pd.DataFrame] = {}
    sc_bridge_by_contrast: dict[str, pd.DataFrame] = {}
    sc_bridge_frames: list[pd.DataFrame] = []
    for c_spec in contrasts_list:
        cid_t = c_spec["contrast_id"]
        trace_d = st_dirs_by_contrast.get(cid_t)
        if trace_d is not None and trace_d.is_dir():
            p_df_sc, csq_df_sc, br_df_sc = render_ch7_splicetrace_suite(
                st_dir=trace_d,
                cid=cid_t,
                species=species,
                outdir=out,
                panels=panels,
                tables=tables,
                motif_summaries=motif_summaries,
            )
            if not p_df_sc.empty:
                sc_prot_by_contrast[cid_t] = p_df_sc
            if not csq_df_sc.empty:
                sc_conseq_by_contrast[cid_t] = csq_df_sc
            if not br_df_sc.empty:
                sc_bridge_by_contrast[cid_t] = br_df_sc
                sc_bridge_frames.append(br_df_sc)

    sc_bridge_df = pd.concat(sc_bridge_frames, ignore_index=True).drop_duplicates() if sc_bridge_frames else None

    # Chapter 06: scSpliceNet Non-Circular Multi-Community Spatial Halo `SPLICE_NETWORK_EXPLORER.html` + Static Exhibits!
    sc_splicenet_meta: dict[str, Any] = {"event_to_module": {}, "event_to_kme": {}, "gene_to_module": {}, "gene_to_kme": {}}
    net_units_psi = pb_psi if not pb_psi.empty else (p_df if p_df is not None else pd.DataFrame())
    net_units_list = pb_samples.to_dict("records") if not pb_samples.empty else samples_list

    # Map each biological_sample_id / pseudobulk ID back to its constituent raw cell IDs
    pb_unit_cells: dict[str, set[str]] = {}
    for _s in samples_list:
        _bio = str(_s["biological_sample_id"])
        _cts = _safe_slug(str(_s["cell_type"]))
        pb_unit_cells.setdefault(f"{_bio}__{_cts}", set()).add(str(_s["sample_id"]))
        pb_unit_cells.setdefault(_bio, set()).add(str(_s["sample_id"]))
        pb_unit_cells.setdefault(str(_s["sample_id"]), set()).add(str(_s["sample_id"]))

    # Map contrast sample_ids onto pseudobulk units when pseudobulk units are active
    net_contrasts_spec = []
    if not pb_samples.empty:
        for c_sp in contrasts_list:
            n_lbl = c_sp["numerator"]["label"]
            d_lbl = c_sp["denominator"]["label"]
            cid_cur = c_sp["contrast_id"]
            pop_cells = sub_pop_cells.get(cid_cur)
            if pop_cells:
                nu_ids = [u["sample_id"] for u in net_units_list if u["condition"] == n_lbl and (pb_unit_cells.get(str(u["sample_id"]), set()) & pop_cells)]
                du_ids = [u["sample_id"] for u in net_units_list if u["condition"] == d_lbl and (pb_unit_cells.get(str(u["sample_id"]), set()) & pop_cells)]
            else:
                pop_p = cid_cur.split("__")[0] if "__" in cid_cur else None
                nu_ids = [u["sample_id"] for u in net_units_list if u["condition"] == n_lbl and (pop_p is None or u.get("cell_type") == pop_p)]
                du_ids = [u["sample_id"] for u in net_units_list if u["condition"] == d_lbl and (pop_p is None or u.get("cell_type") == pop_p)]
            nu_ids = nu_ids or [u["sample_id"] for u in net_units_list if u["condition"] == n_lbl]
            du_ids = du_ids or [u["sample_id"] for u in net_units_list if u["condition"] == d_lbl]
            net_contrasts_spec.append({
                "contrast_id": cid_cur,
                "label": c_sp["label"],
                "numerator": {"label": n_lbl, "sample_ids": nu_ids},
                "denominator": {"label": d_lbl, "sample_ids": du_ids},
            })
    else:
        net_contrasts_spec = contrasts_list

    ev_meta_sc: dict[str, dict[str, str]] = {}
    if "event_id" in ev_df.columns:
        for _, r_e in ev_df.drop_duplicates("event_id").iterrows():
            ev_meta_sc[str(r_e["event_id"])] = {
                "gene_symbol": str(r_e.get(g_c, "")),
                "gene_id": str(r_e.get("gene_id", r_e.get(g_c, ""))),
                "event_type": str(r_e.get("event_type", "SE")),
            }

    def _load_sc_net_dir(sn_path: Path) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
        m_df = _safe_read_csv(sn_path / "scsplicenet_modules.tsv", sep="\t")
        if m_df.empty or "module" not in m_df.columns:
            return pd.DataFrame(), None, None
        has_assigned = m_df["module"].astype(str).map(lambda m: m not in ("", "0", "M0", "M00", "grey", "unassigned", "Unassigned", "nan", "None")).any()
        if not has_assigned:
            return pd.DataFrame(), None, None
        ek_sc = next((c for c in ("event_key", "event_id") if c in m_df.columns), m_df.columns[0])
        if "gene_symbol" not in m_df.columns or m_df["gene_symbol"].isna().all():
            m_df["gene_symbol"] = m_df[ek_sc].map(lambda k: ev_meta_sc.get(str(k), {}).get("gene_symbol", str(k).split("|")[0]))
        if "gene_id" not in m_df.columns:
            m_df["gene_id"] = m_df[ek_sc].map(lambda k: ev_meta_sc.get(str(k), {}).get("gene_id", str(k).split("|")[0]))
        if "event_type" not in m_df.columns:
            m_df["event_type"] = m_df[ek_sc].map(lambda k: ev_meta_sc.get(str(k), {}).get("event_type", "SE"))
        ep = next((p for p in (sn_path / "scsplicenet_edges.tsv", sn_path / "network_edges.tsv") if p.is_file()), None)
        ed = _safe_read_csv(ep, sep="\t") if ep is not None else pd.DataFrame()
        sp = next((p for p in (sn_path / "scsplicenet_donor_scores.tsv", sn_path / "module_scores.tsv") if p.is_file()), None)
        sc = _safe_read_csv(sp, sep="\t", index_col=0) if sp is not None else pd.DataFrame()
        return m_df, (ed if not ed.empty else None), (sc if not sc.empty else None)

    def _merge_sn_meta(dst: dict[str, Any], src_meta: dict[str, Any]) -> None:
        for k in ("event_to_module", "event_to_kme", "gene_to_module", "gene_to_kme"):
            for mk, mv in (src_meta.get(k) or {}).items():
                dst[k].setdefault(mk, mv)

    covered_cids: set[str] = set()
    root_network_taken = False
    if sn_dirs_by_contrast and not net_units_psi.empty:
        pop_groups: dict[str, list[dict[str, Any]]] = {}
        for c_sp in net_contrasts_spec:
            cid_cur = str(c_sp["contrast_id"])
            pkey = cid_cur.split("__", 1)[0] if "__" in cid_cur else "__GLOBAL__"
            pop_groups.setdefault(pkey, []).append(c_sp)

        for pop_key, pop_c_specs in pop_groups.items():
            pop_cids = [str(c["contrast_id"]) for c in pop_c_specs]
            sn_pick = next((sn_dirs_by_contrast[c] for c in pop_cids if c in sn_dirs_by_contrast), None)
            if sn_pick is None:
                continue
            p_mod_df, p_edges_df, p_scores_df = _load_sc_net_dir(sn_pick)
            if p_mod_df.empty:
                continue

            # Slice units strictly to this population's biological units (falling back to cell-level slice when <4 pseudobulk units exist)
            pop_cells = set().union(*(sub_pop_cells.get(c, set()) for c in pop_cids))
            if pop_cells and not pb_samples.empty:
                matched_pb = [u for u in net_units_list if len(pb_unit_cells.get(str(u["sample_id"]), set()) & pop_cells) > 0]
            else:
                matched_pb = [u for u in net_units_list if u.get("cell_type") == pop_key] if pop_key != "__GLOBAL__" else []

            if len(matched_pb) >= 4:
                pop_units_list = matched_pb
                pop_cols = [u["sample_id"] for u in matched_pb if u["sample_id"] in net_units_psi.columns]
                pop_units_psi = net_units_psi[pop_cols]
                pop_contrasts_use = pop_c_specs
            elif p_df is not None and len(pop_cells & set(p_df.columns)) >= 4:
                pop_units_list = [s for s in samples_list if s["sample_id"] in pop_cells and s["sample_id"] in p_df.columns]
                pop_units_psi = p_df[[s["sample_id"] for s in pop_units_list]]
                pop_cid_set = set(pop_cids)
                pop_contrasts_use = [c for c in contrasts_list if c["contrast_id"] in pop_cid_set]
            else:
                pop_units_list = net_units_list
                pop_units_psi = net_units_psi
                pop_contrasts_use = pop_c_specs

            use_root_dir = (not root_network_taken) and (pop_key.lower().startswith("all") or pop_key == "__GLOBAL__")
            if use_root_dir:
                root_network_taken = True
            pop_br_list = [sc_bridge_by_contrast[c] for c in pop_cids if c in sc_bridge_by_contrast]
            pop_bridge_df = pd.concat(pop_br_list, ignore_index=True).drop_duplicates() if pop_br_list else sc_bridge_df
            pop_meta = build_chapter6_splicenet_suite(
                mod_df_raw=p_mod_df,
                scores_df_raw=p_scores_df,
                samples_list=pop_units_list,
                contrasts_spec=pop_contrasts_use,
                psi_events_by_units=pop_units_psi,
                gene_sym_map=gene_sym_map_sc,
                outdir=out,
                panels=panels,
                tables=tables,
                network_browsers=network_browsers,
                species=species,
                is_single_cell=True,
                bridge_df=pop_bridge_df,
                edges_df=p_edges_df,
                scope_slug=None if use_root_dir else _safe_slug(pop_key),
                scope_title=f"scSpliceNet — Population co-splicing network | {pop_key}",
                scope_note=f"Population-specific scSpliceNet co-splicing network fitted on {pop_key} cells. Switching Cell Type in the Scope Bar displays this population's modules, hub genes, and contrast ΔME.",
                comparison_ids=pop_cids,
                whole_study=bool(use_root_dir),
            )
            _merge_sn_meta(sc_splicenet_meta, pop_meta)
            covered_cids.update(pop_cids)

    sn_dir = Path(scsplicenet_dir) if (scsplicenet_dir and str(scsplicenet_dir) != "NO_SCSPLICENET" and Path(scsplicenet_dir).is_dir()) else None
    if sn_dir is not None and (sn_dir / "scsplicenet_modules.tsv").is_file() and not net_units_psi.empty:
        mod_df, edges_sc_df, scores_sc_df = _load_sc_net_dir(sn_dir)
        if not mod_df.empty:
            fallback_cids = [str(c["contrast_id"]) for c in net_contrasts_spec if str(c["contrast_id"]) not in covered_cids]
            if not covered_cids:
                glob_meta = build_chapter6_splicenet_suite(
                    mod_df_raw=mod_df,
                    scores_df_raw=scores_sc_df,
                    samples_list=net_units_list,
                    contrasts_spec=net_contrasts_spec,
                    psi_events_by_units=net_units_psi,
                    gene_sym_map=gene_sym_map_sc,
                    outdir=out,
                    panels=panels,
                    tables=tables,
                    network_browsers=network_browsers,
                    species=species,
                    is_single_cell=True,
                    bridge_df=sc_bridge_df,
                    edges_df=edges_sc_df,
                )
                _merge_sn_meta(sc_splicenet_meta, glob_meta)
            elif fallback_cids:
                fb_specs = [c for c in net_contrasts_spec if str(c["contrast_id"]) in set(fallback_cids)]
                glob_meta = build_chapter6_splicenet_suite(
                    mod_df_raw=mod_df,
                    scores_df_raw=scores_sc_df,
                    samples_list=net_units_list,
                    contrasts_spec=fb_specs,
                    psi_events_by_units=net_units_psi,
                    gene_sym_map=gene_sym_map_sc,
                    outdir=out,
                    panels=panels,
                    tables=tables,
                    network_browsers=network_browsers,
                    species=species,
                    is_single_cell=True,
                    bridge_df=sc_bridge_df,
                    edges_df=edges_sc_df,
                    scope_slug="cohort-fallback" if root_network_taken else None,
                    scope_title="scSpliceNet — Cohort-wide co-splicing network (shared fallback)",
                    scope_note="Shared cohort-wide scSpliceNet co-splicing network shown for sparse subpopulations that fell below the minimum cell/event threshold for population-specific network fitting.",
                    comparison_ids=fallback_cids,
                    whole_study=True,
                )
                _merge_sn_meta(sc_splicenet_meta, glob_meta)
            else:
                ek_col_g = next((c for c in ("event_id", "event_key", "ID") if c in mod_df.columns), mod_df.columns[0])
                for r_g in mod_df.itertuples(index=False):
                    m_g = str(getattr(r_g, "module", ""))
                    if m_g not in ("", "0", "M0", "M00", "grey", "unassigned", "Unassigned", "nan", "None"):
                        ek_v = str(getattr(r_g, ek_col_g))
                        gs_v = str(getattr(r_g, "gene_symbol", ek_v))
                        sc_splicenet_meta["event_to_module"].setdefault(ek_v, m_g)
                        sc_splicenet_meta["gene_to_module"].setdefault(gs_v, m_g)

    # Chapter 08: Convergent Target Scorecard + Self-Contained Interactive Candidate Gene Finder & Katana Native Sashimi 4-Track Workbench!
    interactive_blocks_sc: list[dict[str, Any]] = []
    expr_units_for_ch8 = pb_expr_df if not pb_expr_df.empty else expr_df
    build_chapter8_synthesis_and_workbench(
        contrasts_spec=net_contrasts_spec,
        das_by_contrast=sc_das_by_contrast,
        deg_by_contrast=sc_deg_by_contrast,
        prot_by_contrast=sc_prot_by_contrast,
        conseq_by_contrast=sc_conseq_by_contrast,
        splicenet_meta=sc_splicenet_meta,
        bridge_df=sc_bridge_df,
        psi_events_by_units=net_units_psi,
        expr_genes_by_units=expr_units_for_ch8,
        samples_list=net_units_list,
        gene_sym_map=gene_sym_map_sc,
        outdir=out,
        panels=panels,
        tables=tables,
        top10_list=top10,
        interactive_blocks=interactive_blocks_sc,
        is_single_cell=True,
        gtf_path=Path(gtf) if gtf else None,
        bams_dir=Path(bams) if bams else None,
        sc_psi_cells=p_df,
        sc_support_cells=sup_df,
        sc_cells_meta=samples_list,
        sc_contrasts_cell_spec=contrasts_list,
        sc_bam_path=Path(sc_bam) if (sc_bam and str(sc_bam) != "NO_SC_BAM") else None,
        sc_bc_tag=sc_bc_tag or "BC",
        sc_barcode_shares=Path(sc_barcode_shares) if (sc_barcode_shares and str(sc_barcode_shares) != "NO_BARCODE_SHARES") else None,
    )

    return _finalize_consulting_report_and_zip(
        outdir=out,
        study_name=study_name,
        mode="single-cell",
        samples_list=samples_list,
        contrasts_list=contrasts_list,
        panels=panels,
        tables=tables,
        top10=top10,
        policy_data=policy_data,
        motif_summaries=motif_summaries,
        species=species,
        network_browsers=network_browsers,
        interactive_blocks=interactive_blocks_sc,
        dummy=dummy,
    )


def open_workspace(
    report_path: str | Path | None = None,
    *,
    host: str = "127.0.0.1",
    port: int = 18775,
    open_browser: bool = True,
) -> None:
    """Launch the local interactive workspace and optionally pre-import a `REPORT.zip` or report directory."""
    import uvicorn
    from katana_next.api.app import PlatformRuntime, create_app

    app_root = Path(__file__).resolve().parents[1]
    runtime = PlatformRuntime(application_root=app_root)
    if report_path is not None:
        rp = Path(report_path).expanduser().resolve()
        try:
            from uuid import uuid4
            from katana_next.api.local_files import workspace_home
            from katana_next.services.report_archive import report_source
            from katana_next.services.report_import import import_report

            ws_home = workspace_home()
            src_dir = report_source(rp, ws_home)
            ws_dest = ws_home / f"report-{rp.stem}-{uuid4().hex[:8]}"
            ctx = import_report(src_dir, ws_dest, runtime.projects)
            runtime.jobs.bind_project(ctx.manifest.project_id, ctx.manifest.output_root)
        except Exception as exc:
            print(f"[katana view] Note: could not pre-import {rp}: {exc}")

    app = create_app(runtime)
    url = f"http://{host}:{port}"
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=host, port=port)


def run_report_from_nextflow(
    *,
    outdir: str | Path,
    bundle: str | Path,
    fasta: str | Path | None = None,
    gtf: str | Path | None = None,
    study_name: str = "Katana Study",
    species: str = "auto",
    fdr_cutoff: float = 0.05,
    dpsi_cutoff: float = 0.10,
    rmats_mode: str = "JCEC",
    splicemanifold_dir: str | Path | None = None,
    splicenet_dir: str | Path | None = None,
    splicetrace_dirs: list[str | Path] | dict[str, Path] | None = None,
    dummy: bool = False,
) -> dict[str, Any]:
    """Assemble Katana Report (`REPORT.html` + `REPORT.zip`) in Nextflow, reusing upstream SpliceManifold, SpliceNet & SpliceTrace outputs."""
    return run_bulk(
        outdir=outdir,
        bundle=bundle,
        fasta=fasta,
        gtf=gtf,
        study_name=study_name,
        species=species,
        fdr_cutoff=fdr_cutoff,
        dpsi_cutoff=dpsi_cutoff,
        rmats_mode=rmats_mode,
        splicemanifold_dir=splicemanifold_dir,
        splicenet_dir=splicenet_dir,
        splicetrace_dirs=splicetrace_dirs,
        dummy=dummy,
    )
