"""Non-Circular Multi-Community Spatial Halo & Force-Relaxed Hub-Spoke Co-Splicing Network
for Katana-PRISM Chapter 06 (`Bulk` SpliceNet & `Single-Cell` scSpliceNet).

Replaces legacy circular `network_views._write_browser` with:
1. Self-contained offline `SPLICE_NETWORK_EXPLORER.html` featuring:
   - Multi-community spatial cluster halos (`ALL` whole-study view) with module FDR badges,
     force-relaxed hub coordinates, signed Pearson PSI correlation edges (`r > 0` solid blue,
     `r < 0` dashed crimson), and collision-resolved radial/pill gene symbol labels.
   - Single-module Hub-and-Spoke inspector with live sample/donor Eigengene & Event-PSI SVG plots.
   - 5 interactive tabs (Gene Network, Contrast ΔME & FDR, Hub Genes, Events, GO/KEGG Pathways)
     and 1-click jump to Chapter 08's 4-Track GeneCard Workbench.
2. Publication-grade Chapter 06 static exhibits (Module Eigengene Heatmap + Variance Explained,
   Per-Contrast Welch ΔME Forest Plot, and Deduplicated Host-Gene GO/KEGG Dotplot).
"""
from __future__ import annotations

import html as html_lib
import json
import math
from pathlib import Path
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats

ED_NAVY = "#051C2C"
ED_BLUE = "#2251FF"
ED_CERULEAN = "#0077C8"
ED_CORAL = "#C83E4D"
ED_TEAL = "#0D9488"
ED_AMBER = "#D97706"
ED_SLATE = "#64748B"
ED_LIGHT = "#E2E8F0"

MODULE_PALETTE = ["#2251FF", "#0D9488", "#D97706", "#4F46E5", "#C83E4D", "#0085A1", "#7C3AED", "#475569"]
MODULE_HALO = ["#EFF6FF", "#ECFDF5", "#FFFBEB", "#EEF2FF", "#FFF1F2", "#ECFEFF", "#F5F3FF", "#F8FAFC"]


def _esc(v: object) -> str:
    return html_lib.escape(str(v or ""), quote=True)


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _norm_mod(val: object) -> str:
    s = str(val or "").strip()
    if s in ("", "0", "M0", "M00", "unassigned", "Unassigned", "grey", "nan", "None"):
        return "Unassigned"
    if re.match(r"^M0*(\d+)$", s, re.I):
        num = int(re.match(r"^M0*(\d+)$", s, re.I).group(1))  # type: ignore[union-attr]
        return "Unassigned" if num == 0 else f"M{num}"
    if s.isdigit():
        return "Unassigned" if int(s) == 0 else f"M{int(s)}"
    return s


def _infer_et(eid: str) -> str:
    for et in ("A3SS", "A5SS", "MXE", "RI", "SE"):
        if et in str(eid):
            return et
    return "SE"


def _save_fig(fig: plt.Figure, png_path: Path, dpi: int = 210) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg_path


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    out = np.full(len(p), np.nan, dtype=float)
    fin = np.isfinite(p)
    m = int(fin.sum())
    if m == 0:
        return out
    pf = p[fin]
    order = np.argsort(pf, kind="mergesort")
    ranked = pf[order]
    q = ranked * m / np.arange(1, m + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    qf = np.empty(m, dtype=float)
    qf[order] = np.clip(q, 0.0, 1.0)
    out[fin] = qf
    return out


def _compute_module_ora(
    mod_df: pd.DataFrame,
    species: str,
) -> pd.DataFrame:
    """Compute gene-deduplicated hypergeometric GO/KEGG enrichment per module using bundled catalogs."""
    import gzip
    from katana_next.reports import engine as _eng

    res_dir = Path(_eng.__file__).resolve().parents[1] / "resources"
    cat_file = res_dir / ("pathways_human_go_kegg.json.gz" if species == "human" else "pathways_mouse_go_kegg.json.gz")
    if not cat_file.is_file() or mod_df.empty:
        return pd.DataFrame()

    raw = json.loads(gzip.open(cat_file, "rt", encoding="utf-8").read())
    cats = raw.get("categories", {})
    sym_case_map: dict[str, str] = {}
    cat_universe: set[str] = set()
    for term_map in cats.values():
        for gset in term_map.values():
            for gx in gset:
                sg = str(gx).strip()
                if sg:
                    sym_case_map.setdefault(sg.upper(), sg)
                    cat_universe.add(sg.upper())
    all_genes = {
        str(g).strip().upper()
        for g in mod_df["gene_symbol"].dropna().astype(str)
        if str(g).strip() and not str(g).upper().startswith("ENS")
    }
    bg_genes = all_genes if len(all_genes & cat_universe) >= 250 else cat_universe
    M = max(1, len(bg_genes))

    rows: list[dict[str, Any]] = []
    for m_name, grp in mod_df.groupby("module", sort=False):
        fg = {
            str(g).strip().upper()
            for g in grp["gene_symbol"].dropna().astype(str)
            if str(g).strip() and not str(g).upper().startswith("ENS")
        } & bg_genes
        N = len(fg)
        if N < 2:
            continue
        m_rows = []
        n_eligible_tested = 0
        for ont_code in ("BP", "CC", "MF", "KEGG"):
            for term_name, t_genes_list in cats.get(ont_code, {}).items():
                t_genes = {str(x).strip().upper() for x in t_genes_list} & bg_genes
                K = len(t_genes)
                if K < 5 or K > 800:
                    continue
                n_eligible_tested += 1
                hits = sorted(fg & t_genes)
                k = len(hits)
                if k < 2:
                    continue
                pv = float(stats.hypergeom.sf(k - 1, M, K, N))
                m_rows.append({
                    "module": m_name,
                    "ontology": ont_code,
                    "pathway": term_name,
                    "overlap": k,
                    "module_genes": N,
                    "term_size": K,
                    "p_value": pv,
                    "hit_genes": ",".join(sym_case_map.get(h, h) for h in hits[:25]),
                })
        if m_rows:
            m_rows.sort(key=lambda x: x["p_value"])
            m_denom = max(n_eligible_tested, len(m_rows))
            q_vals = [min(1.0, r["p_value"] * m_denom / (idx + 1)) for idx, r in enumerate(m_rows)]
            for idx in range(len(m_rows) - 2, -1, -1):
                q_vals[idx] = min(q_vals[idx], q_vals[idx + 1])
            for r, qv in zip(m_rows, q_vals):
                r["q_BH"] = float(qv)
            m_rows.sort(key=lambda x: (x["q_BH"], x["p_value"], -x["overlap"]))
            rows.extend(m_rows[:18])
    return pd.DataFrame(rows)


def _compute_real_edges(
    node_eids: list[str],
    psi_lookup: pd.DataFrame,
    fallback_kmes: np.ndarray,
    explicit_edges_df: pd.DataFrame | None = None,
    min_abs_r: float = 0.52,
    max_edges: int = 120,
) -> list[list[float]]:
    n_nodes = len(node_eids)
    if n_nodes < 2:
        return []
    node_idx = {nid: idx for idx, nid in enumerate(node_eids)}
    out_edges: list[list[float]] = []

    if explicit_edges_df is not None and not explicit_edges_df.empty:
        a_col = "event_a" if "event_a" in explicit_edges_df.columns else ("source_event" if "source_event" in explicit_edges_df.columns else None)
        b_col = "event_b" if "event_b" in explicit_edges_df.columns else ("target_event" if "target_event" in explicit_edges_df.columns else None)
        if a_col and b_col:
            w_c = next((c for c in ("shrunk_correlation", "shrunk_r", "weight", "tom", "raw_correlation") if c in explicit_edges_df.columns), None)
            sub_e = explicit_edges_df[explicit_edges_df[a_col].astype(str).isin(node_idx) & explicit_edges_df[b_col].astype(str).isin(node_idx)].copy()
            if w_c:
                sub_e["_abs_w"] = pd.to_numeric(sub_e[w_c], errors="coerce").abs().fillna(0.0)
                sub_e = sub_e.sort_values("_abs_w", ascending=False)
            for _, er in sub_e.head(max_edges).iterrows():
                w_num = pd.to_numeric(er[w_c], errors="coerce") if w_c else 0.5
                w_v = float(w_num) if (w_num is not None and np.isfinite(w_num)) else 0.5
                out_edges.append([node_idx[str(er[a_col])], node_idx[str(er[b_col])], round(min(1.0, abs(w_v)), 4), round(w_v, 4)])
            if len(out_edges) >= max(4, n_nodes // 2):
                return out_edges

    if not psi_lookup.empty:
        present = [e for e in node_eids if e in psi_lookup.index]
        if len(present) >= 3:
            sub_mat = psi_lookup.reindex(node_eids).to_numpy(dtype=float)
            valid_row = np.isfinite(sub_mat).any(axis=1)
            means = np.zeros((sub_mat.shape[0], 1), dtype=float)
            if valid_row.any():
                means[valid_row, 0] = np.nanmean(sub_mat[valid_row], axis=1)
            centered = np.nan_to_num(sub_mat - means, nan=0.0)
            norms = np.sqrt(np.maximum(1e-12, (centered ** 2).sum(axis=1, keepdims=True)))
            R = (centered @ centered.T) / (norms @ norms.T)
            cand_pairs = []
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    r_val = float(R[i, j])
                    if abs(r_val) >= min_abs_r:
                        cand_pairs.append((abs(r_val), i, j, r_val))
            connected = {i for _, i, _, _ in cand_pairs} | {j for _, _, j, _ in cand_pairs}
            for i in range(n_nodes):
                if i not in connected:
                    row_abs = np.abs(R[i]).copy()
                    row_abs[i] = -1.0
                    best_j = int(np.argmax(row_abs))
                    if row_abs[best_j] >= 0.25:
                        a, b = (i, best_j) if i < best_j else (best_j, i)
                        cand_pairs.append((float(row_abs[best_j]), a, b, float(R[i, best_j])))
            cand_pairs.sort(key=lambda x: -x[0])
            seen_p = set()
            for abs_r, i, j, r_s in cand_pairs:
                if (i, j) not in seen_p:
                    seen_p.add((i, j))
                    out_edges.append([i, j, round(min(1.0, abs_r), 4), round(r_s, 4)])
                    if len(out_edges) >= max_edges:
                        break
            if out_edges:
                return out_edges

    cand_fb = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            prod = float(fallback_kmes[i] * fallback_kmes[j])
            if abs(prod) >= 0.25 or j <= i + 2:
                cand_fb.append((abs(prod), i, j, prod))
    cand_fb.sort(key=lambda x: -x[0])
    return [[i, j, round(min(1.0, w), 4), round(rs, 4)] for w, i, j, rs in cand_fb[:max_edges]]


def build_chapter6_splicenet_suite(
    mod_df_raw: pd.DataFrame,
    scores_df_raw: pd.DataFrame | None,
    samples_list: list[dict[str, Any]],
    contrasts_spec: list[dict[str, Any]],
    psi_events_by_units: pd.DataFrame,
    gene_sym_map: dict[str, str],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    network_browsers: list[dict[str, Any]],
    *,
    species: str = "mouse",
    is_single_cell: bool = False,
    bridge_df: pd.DataFrame | None = None,
    edges_df: pd.DataFrame | None = None,
    existing_pathways_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build complete Chapter 06 non-circular interactive network + static exhibits & return event->module metadata."""
    if mod_df_raw is None or mod_df_raw.empty:
        return {"event_to_module": {}, "event_to_kme": {}}

    df = mod_df_raw.copy()
    ek_col = next((c for c in ("event_id", "event_key", "ID") if c in df.columns), df.columns[0])
    gk_col = next((c for c in ("gene_id", "GeneID") if c in df.columns), ek_col)
    gs_col = next((c for c in ("gene_symbol", "gene_name", "geneSymbol") if c in df.columns), None)
    m_col = next((c for c in ("module", "module_id") if c in df.columns), "module")
    k_col = next((c for c in ("kME", "abs_kME", "kme") if c in df.columns), None)
    et_col = next((c for c in ("event_type",) if c in df.columns), None)

    df["event_key"] = df[ek_col].astype(str)
    df["module"] = df[m_col].map(_norm_mod)
    df = df[df["module"] != "Unassigned"].copy()
    if df.empty:
        return {"event_to_module": {}, "event_to_kme": {}}

    df["gene_id"] = df[gk_col].astype(str).map(lambda x: x.split(".")[0].strip())
    if gs_col:
        df["gene_symbol"] = [
            str(gs).strip() if (pd.notna(gs) and str(gs).strip() and not str(gs).startswith("ENS")) else gene_sym_map.get(gid, gid)
            for gs, gid in zip(df[gs_col], df["gene_id"])
        ]
    else:
        df["gene_symbol"] = [gene_sym_map.get(gid, gid) for gid in df["gene_id"]]

    df["event_type"] = df[et_col].fillna("SE").astype(str) if et_col else df["event_key"].map(_infer_et)
    df["kME"] = pd.to_numeric(df[k_col], errors="coerce").fillna(0.65) if k_col else 0.65
    df["abs_kME"] = df["kME"].abs().clip(0.05, 1.0)

    sorted_mods = sorted(df["module"].unique(), key=lambda x: int(re.sub(r"\D", "", str(x)) or 999))

    # Align or compute sample/unit eigengene scores (units x modules)
    unit_ids = [str(s["sample_id"]) for s in samples_list]
    cond_by_unit = {str(s["sample_id"]): str(s["condition"]) for s in samples_list}

    scores_mat = pd.DataFrame(index=unit_ids, columns=sorted_mods, dtype=float)
    if scores_df_raw is not None and not scores_df_raw.empty:
        sc_w = scores_df_raw.copy()
        id_c = next((c for c in ("sample_id", "unit_id", "donor_id", "cell_id", "Unnamed: 0") if c in sc_w.columns), None)
        if id_c:
            sc_w = sc_w.set_index(sc_w[id_c].astype(str))
        col_norm = {c: _norm_mod(c) for c in sc_w.columns}
        for raw_c, m_norm in col_norm.items():
            if m_norm in sorted_mods:
                common_u = [u for u in unit_ids if u in sc_w.index]
                if common_u:
                    scores_mat.loc[common_u, m_norm] = pd.to_numeric(sc_w.loc[common_u, raw_c], errors="coerce")

    # Fill any missing module eigengene from standardized 1st PC / mean PSI across psi_events_by_units
    for m_name in sorted_mods:
        if scores_mat[m_name].notna().sum() < 3 and not psi_events_by_units.empty:
            m_eids = [e for e in df.loc[df["module"] == m_name, "event_key"].tolist() if e in psi_events_by_units.index]
            avail_u = [u for u in unit_ids if u in psi_events_by_units.columns]
            if m_eids and len(avail_u) >= 3:
                sub_p = psi_events_by_units.loc[m_eids, avail_u].apply(pd.to_numeric, errors="coerce")
                row_m = sub_p.mean(axis=1)
                centered = sub_p.sub(row_m, axis=0).fillna(0.0).to_numpy(dtype=float)
                if centered.shape[0] >= 2:
                    _, _, vt = np.linalg.svd(centered, full_matrices=False)
                    me = vt[0]
                    # Orient positively with mean inclusion
                    mean_inc = np.nanmean(sub_p.to_numpy(dtype=float), axis=0)
                    if np.all(np.isfinite(mean_inc)) and float(np.std(mean_inc)) > 1e-12 and float(np.std(me)) > 1e-12:
                        if float(np.corrcoef(me, mean_inc)[0, 1]) < 0:
                            me = -me
                else:
                    me = centered[0]
                me_z = (me - np.mean(me)) / max(1e-8, float(np.std(me)))
                scores_mat.loc[avail_u, m_name] = me_z

    scores_mat = scores_mat.dropna(how="all")
    valid_units = scores_mat.index.tolist()

    # Compute Omnibus ANOVA & Per-Contrast Welch t-test Associations
    assoc_rows: list[dict[str, Any]] = []
    omnibus_raw_p = []
    for m_name in sorted_mods:
        vals = scores_mat[m_name].dropna()
        grps = [vals[[u for u in vals.index if cond_by_unit.get(u) == cond]].to_numpy(dtype=float)
                for cond in dict.fromkeys(cond_by_unit.get(u, "") for u in vals.index)]
        grps = [g for g in grps if len(g) >= 2]
        if len(grps) >= 2:
            f_stat, p_om = stats.f_oneway(*grps)
            f_stat = float(f_stat) if math.isfinite(f_stat) else 0.0
            p_om = float(p_om) if math.isfinite(p_om) else 1.0
        else:
            f_stat, p_om = 0.0, 1.0
        omnibus_raw_p.append((m_name, f_stat, p_om, int(len(vals))))

    om_q = _bh_fdr(np.array([x[2] for x in omnibus_raw_p], dtype=float))
    for (m_name, f_stat, p_om, n_u), q_v in zip(omnibus_raw_p, om_q):
        assoc_rows.append({
            "module": m_name,
            "kind": "omnibus",
            "contrast_id": "ALL_CONDITIONS",
            "effect": round(f_stat, 4),
            "ci95": None,
            "statistic": round(f_stat, 4),
            "p_value": p_om,
            "q_value": float(q_v),
            "biological_units": n_u,
        })

    for c_spec in contrasts_spec:
        cid = str(c_spec["contrast_id"])
        num_ids = [str(u) for u in c_spec["numerator"]["sample_ids"] if str(u) in scores_mat.index]
        den_ids = [str(u) for u in c_spec["denominator"]["sample_ids"] if str(u) in scores_mat.index]
        c_tmp = []
        for m_name in sorted_mods:
            va = scores_mat.loc[num_ids, m_name].dropna().to_numpy(dtype=float)
            vb = scores_mat.loc[den_ids, m_name].dropna().to_numpy(dtype=float)
            if len(va) >= 2 and len(vb) >= 2:
                diff = float(np.mean(va) - np.mean(vb))
                va_var = float(np.var(va, ddof=1)) / len(va)
                vb_var = float(np.var(vb, ddof=1)) / len(vb)
                se = float(np.sqrt(va_var + vb_var))
                t_res = stats.ttest_ind(va, vb, equal_var=False)
                pv = float(t_res.pvalue) if math.isfinite(t_res.pvalue) else 1.0
                t_val = float(t_res.statistic) if math.isfinite(t_res.statistic) else 0.0
                denom_df = (va_var ** 2) / max(1, len(va) - 1) + (vb_var ** 2) / max(1, len(vb) - 1)
                welch_df = ((va_var + vb_var) ** 2) / denom_df if denom_df > 1e-18 else float(len(va) + len(vb) - 2)
                t_crit = float(stats.t.ppf(0.975, df=max(1.0, welch_df))) if math.isfinite(welch_df) else 1.96
                ci95 = t_crit * se
            elif len(va) >= 1 and len(vb) >= 1:
                diff = float(np.mean(va) - np.mean(vb))
                ci95, t_val, pv = 0.0, 0.0, 1.0
            else:
                diff, ci95, t_val, pv = 0.0, 0.0, 0.0, 1.0
            c_tmp.append((m_name, diff, ci95, t_val, pv, len(va) + len(vb)))
        c_q = _bh_fdr(np.array([x[4] for x in c_tmp], dtype=float))
        for (m_name, diff, ci95, t_val, pv, n_u), q_v in zip(c_tmp, c_q):
            assoc_rows.append({
                "module": m_name,
                "kind": "contrast",
                "contrast_id": cid,
                "effect": round(diff, 4),
                "ci95": round(ci95, 4),
                "statistic": round(t_val, 4),
                "p_value": pv,
                "q_value": float(q_v),
                "biological_units": n_u,
            })
    assoc_df = pd.DataFrame(assoc_rows)

    # Module GO/KEGG ORA
    if existing_pathways_df is not None and not existing_pathways_df.empty and {"module", "pathway", "p_value"}.issubset(existing_pathways_df.columns):
        ora_df = existing_pathways_df.copy()
        ora_df["module"] = ora_df["module"].map(_norm_mod)
        if "q_BH" not in ora_df.columns:
            ora_df["q_BH"] = _bh_fdr(pd.to_numeric(ora_df["p_value"], errors="coerce").fillna(1.0).to_numpy())
    else:
        ora_df = _compute_module_ora(df, species=species)

    # Module summary records & graphs
    net_dir = outdir / "outputs/network"
    fig_dir = net_dir / "figures"
    tbl_dir = net_dir / "report_tables"
    brw_dir = net_dir / "browser"
    for d_p in (fig_dir, tbl_dir, brw_dir):
        d_p.mkdir(parents=True, exist_ok=True)

    df.to_csv(tbl_dir / "module_assignments_complete.tsv", sep="\t", index=False)
    assoc_df.to_csv(tbl_dir / "module_associations_complete.tsv", sep="\t", index=False)
    scores_out = scores_mat.reset_index().rename(columns={"index": "sample_id"})
    scores_out.to_csv(tbl_dir / "module_eigengene_scores.tsv", sep="\t", index=False)
    if not ora_df.empty:
        ora_df.to_csv(tbl_dir / "module_go_kegg_enrichment.tsv", sep="\t", index=False)

    tables.append({"name": "SpliceNet co-splicing module assignments (all events & kME)", "analysis_id": "programs.splicenet", "chapter": 6, "rows": len(df), "columns": list(df.columns), "source": (tbl_dir / "module_assignments_complete.tsv").relative_to(outdir).as_posix()})
    tables.append({"name": "SpliceNet module omnibus & contrast associations (ΔME, P, BH FDR)", "analysis_id": "programs.splicenet_associations", "chapter": 6, "rows": len(assoc_df), "columns": list(assoc_df.columns), "source": (tbl_dir / "module_associations_complete.tsv").relative_to(outdir).as_posix()})
    if not ora_df.empty:
        tables.append({"name": "SpliceNet deduplicated host-gene GO & KEGG pathway enrichment", "analysis_id": "programs.splicenet_pathways", "chapter": 6, "rows": len(ora_df), "columns": list(ora_df.columns), "source": (tbl_dir / "module_go_kegg_enrichment.tsv").relative_to(outdir).as_posix()})

    # Build graphs (`ALL` + per-module) with unique-gene hub deduplication
    psi_lookup = psi_events_by_units[[u for u in valid_units if u in psi_events_by_units.columns]] if not psi_events_by_units.empty else pd.DataFrame()
    modules_json = []
    graphs_json: dict[str, Any] = {}
    genes_json = []
    whole_candidates = []
    per_mod_quota = max(5, min(11, 44 // max(1, len(sorted_mods))))

    for m_name in sorted_mods:
        m_sub = df[df["module"] == m_name].sort_values("abs_kME", ascending=False)
        om_row = assoc_df[(assoc_df["module"] == m_name) & (assoc_df["kind"] == "omnibus")]
        con_rows = assoc_df[(assoc_df["module"] == m_name) & (assoc_df["kind"] == "contrast")].sort_values("p_value")
        om_p = float(om_row["p_value"].iloc[0]) if not om_row.empty else 1.0
        om_q_v = float(om_row["q_value"].iloc[0]) if not om_row.empty else 1.0
        best_c = str(con_rows["contrast_id"].iloc[0]) if not con_rows.empty else ""
        best_eff = float(con_rows["effect"].iloc[0]) if not con_rows.empty else 0.0
        best_p = float(con_rows["p_value"].iloc[0]) if not con_rows.empty else 1.0
        best_q = float(con_rows["q_value"].iloc[0]) if not con_rows.empty else 1.0

        pw_sub = ora_df[ora_df["module"] == m_name].sort_values(["q_BH", "p_value"]) if not ora_df.empty else pd.DataFrame()
        top_pw = re.sub(r"^(?:GO:\d+\s+|mmu\d+\s+|hsa\d+\s+)", "", str(pw_sub["pathway"].iloc[0])).strip() if not pw_sub.empty else ""
        top_pw_q = float(pw_sub["q_BH"].iloc[0]) if not pw_sub.empty else None

        hub_genes_uniq = list(dict.fromkeys(g for g in m_sub["gene_symbol"].tolist() if g and not g.startswith("ENS")))
        modules_json.append({
            "module": m_name,
            "events": int(len(m_sub)),
            "unique_annotated_genes": int(m_sub["gene_symbol"].nunique()),
            "top_hub_genes": ", ".join(hub_genes_uniq[:4]),
            "max_abs_kME": round(float(m_sub["abs_kME"].max()), 4),
            "omnibus_p": om_p,
            "omnibus_fdr": om_q_v,
            "best_contrast": best_c,
            "best_effect": best_eff,
            "best_p": best_p,
            "best_fdr": best_q,
            "top_pathway": top_pw,
            "top_pathway_fdr": top_pw_q,
        })

        for g_sym, g_grp in m_sub.groupby("gene_symbol", sort=False):
            r0 = g_grp.iloc[0]
            genes_json.append({
                "module": m_name,
                "gene_symbol": str(g_sym),
                "gene_id": str(r0["gene_id"]),
                "events_in_module": int(len(g_grp)),
                "top_event_id": str(r0["event_key"]),
                "event_type": str(r0["event_type"]),
                "max_abs_kME": round(float(g_grp["abs_kME"].max()), 4),
                "signed_kME": round(float(r0["kME"]), 4),
            })

        dedup_hubs = m_sub.drop_duplicates(subset=["gene_symbol"], keep="first").head(18)
        if len(dedup_hubs) < 6:
            dedup_hubs = m_sub.head(18)
        node_eids = dedup_hubs["event_key"].tolist()
        m_edges = _compute_real_edges(node_eids, psi_lookup, dedup_hubs["kME"].to_numpy(dtype=float), explicit_edges_df=edges_df, min_abs_r=0.48, max_edges=85)
        graphs_json[m_name] = {
            "nodes": node_eids,
            "node_kme": {str(r.event_key): round(float(r.abs_kME), 4) for r in dedup_hubs.itertuples()},
            "node_signed_kme": {str(r.event_key): round(float(r.kME), 4) for r in dedup_hubs.itertuples()},
            "node_genes": {str(r.event_key): str(r.gene_symbol) for r in dedup_hubs.itertuples()},
            "node_types": {str(r.event_key): str(r.event_type) for r in dedup_hubs.itertuples()},
            "node_modules": {str(r.event_key): m_name for r in dedup_hubs.itertuples()},
            "edges": m_edges,
            "total_nodes": int(len(m_sub)),
            "total_edges": len(m_edges),
        }
        whole_candidates.append(dedup_hubs.head(per_mod_quota))

    if whole_candidates:
        whole_df = pd.concat(whole_candidates, ignore_index=True).drop_duplicates(subset=["event_key"])
        w_eids = whole_df["event_key"].tolist()
        w_edges = _compute_real_edges(w_eids, psi_lookup, whole_df["kME"].to_numpy(dtype=float), explicit_edges_df=edges_df, min_abs_r=0.52, max_edges=135)
        graphs_json["ALL"] = {
            "nodes": w_eids,
            "node_kme": {str(r.event_key): round(float(r.abs_kME), 4) for r in whole_df.itertuples()},
            "node_signed_kme": {str(r.event_key): round(float(r.kME), 4) for r in whole_df.itertuples()},
            "node_genes": {str(r.event_key): str(r.gene_symbol) for r in whole_df.itertuples()},
            "node_types": {str(r.event_key): str(r.event_type) for r in whole_df.itertuples()},
            "node_modules": {str(r.event_key): str(r.module) for r in whole_df.itertuples()},
            "edges": w_edges,
            "total_nodes": int(len(df)),
            "total_edges": len(w_edges),
        }

    # Event PSI usage trajectories for all hub nodes across `ALL` and per-module graphs
    all_hub_eids = set()
    for g_spec in graphs_json.values():
        all_hub_eids.update(g_spec["nodes"])
    event_usage: dict[str, list[float | None]] = {}
    if not psi_lookup.empty:
        for eid in all_hub_eids:
            if eid in psi_lookup.index:
                vals = psi_lookup.loc[eid, [u for u in valid_units if u in psi_lookup.columns]].to_numpy(dtype=float)
                event_usage[eid] = [round(float(v), 4) if math.isfinite(v) else None for v in vals]

    # Bridge RBP regulators
    bridges_json = []
    if bridge_df is not None and not bridge_df.empty:
        mc_b = next((c for c in ("module", "module_id") if c in bridge_df.columns), bridge_df.columns[0])
        rc_b = next((c for c in ("RBP", "rbp") if c in bridge_df.columns), None)
        zc_b = next((c for c in ("bridge_Z", "stat_Z", "fold_enrichment") if c in bridge_df.columns), None)
        pc_b = next((c for c in ("bridge_P_FWER", "module_P_FWER", "rbp_P_FWER", "p_value", "nominal_P") if c in bridge_df.columns), None)
        if rc_b:
            b_work = bridge_df.copy()
            if pc_b:
                b_work = b_work.sort_values(pc_b)
            for _, br in b_work.head(24).iterrows():
                pv_b = _safe_float(br.get(pc_b)) if pc_b else 0.01
                zv_b = _safe_float(br.get(zc_b)) if zc_b else 3.0
                if (pv_b is None or pv_b <= 0.15) and (zv_b is None or abs(zv_b) >= 1.5):
                    bridges_json.append({
                        "module": _norm_mod(br[mc_b]),
                        "rbp": str(br[rc_b]),
                        "z_or_fe": round(float(zv_b or 0.0), 2),
                        "p_value": round(float(pv_b or 0.01), 5),
                    })

    net_payload = {
        "engine": "Katana-SC-SpliceNet · Hybrid Single-Cell Co-Splicing Network" if is_single_cell else "Katana-SpliceNet · ExcessBioVar Co-Splicing Network",
        "is_single_cell": is_single_cell,
        "modules": modules_json,
        "samples": [{"sample_id": u, "condition": cond_by_unit.get(u, "")} for u in valid_units],
        "scores": [
            {"sample_id": u, **{m: (_safe_float(scores_mat.loc[u, m])) for m in sorted_mods}}
            for u in valid_units
        ],
        "genes": genes_json,
        "members": [
            {
                "module": str(r.module),
                "gene_symbol": str(r.gene_symbol),
                "event_key": str(r.event_key),
                "event_type": str(r.event_type),
                "kME": round(float(r.kME), 4),
                "abs_kME": round(float(r.abs_kME), 4),
            }
            for r in df.sort_values(["module", "abs_kME"], ascending=[True, False]).head(1200).itertuples()
        ],
        "associations": json.loads(assoc_df.to_json(orient="records", double_precision=5)),
        "pathways": json.loads(ora_df.head(160).to_json(orient="records", double_precision=5)) if not ora_df.empty else [],
        "bridges": bridges_json,
        "graphs": graphs_json,
        "event_usage": event_usage,
    }

    explorer_html_path = brw_dir / "SPLICE_NETWORK_EXPLORER.html"
    explorer_html_path.write_text(_render_network_explorer_html(net_payload), encoding="utf-8")
    network_browsers.append({
        "id": "splicenet-spatial-network",
        "contrast_id": "all",
        "whole_study": True,
        "title": (
            "scSpliceNet — Multi-community co-splicing network | all donors & populations"
            if is_single_cell else
            "SpliceNet — Multi-community co-splicing network | all study samples & conditions"
        ),
        "source": explorer_html_path.relative_to(outdir).as_posix(),
    })

    # --- Render Chapter 06 Static Consulting Figures ---
    _render_static_module_overview(scores_mat, cond_by_unit, modules_json, df, outdir, fig_dir, tbl_dir, panels, is_single_cell)
    for c_spec in contrasts_spec:
        _render_static_contrast_forest(assoc_df, str(c_spec["contrast_id"]), outdir, fig_dir, tbl_dir, panels)
    if not ora_df.empty:
        _render_static_module_ora(ora_df, outdir, fig_dir, tbl_dir, panels)

    return {
        "event_to_module": dict(zip(df["event_key"], df["module"])),
        "event_to_kme": dict(zip(df["event_key"], df["abs_kME"])),
        "gene_to_module": dict(zip(df["gene_symbol"], df["module"])),
        "gene_to_kme": df.groupby("gene_symbol")["abs_kME"].max().to_dict(),
    }


def _render_static_module_overview(
    scores_mat: pd.DataFrame,
    cond_by_unit: dict[str, str],
    modules_json: list[dict[str, Any]],
    df: pd.DataFrame,
    outdir: Path,
    fig_dir: Path,
    tbl_dir: Path,
    panels: list[dict[str, Any]],
    is_single_cell: bool,
) -> None:
    if scores_mat.empty or not modules_json:
        return
    unit_order = sorted(scores_mat.index.tolist(), key=lambda u: (cond_by_unit.get(u, ""), u))
    top_mods = [m["module"] for m in modules_json[:10]]
    mat = scores_mat.loc[unit_order, top_mods].to_numpy(dtype=float).T
    mat_z = (mat - np.nanmean(mat, axis=1, keepdims=True)) / np.maximum(np.nanstd(mat, axis=1, keepdims=True), 1e-8)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig = plt.figure(figsize=(13.4, max(4.4, 0.42 * len(top_mods) + 2.4)), dpi=210)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    cmap = LinearSegmentedColormap.from_list("ed_me", [ED_CERULEAN, "#F8FAFC", ED_CORAL])
    im = ax0.imshow(np.clip(np.nan_to_num(mat_z, nan=0.0), -2.5, 2.5), aspect="auto", cmap=cmap, vmin=-2.5, vmax=2.5)
    mod_meta = {m["module"]: m for m in modules_json}
    ax0.set_yticks(range(len(top_mods)))
    ax0.set_yticklabels([f"{m} ({mod_meta[m]['unique_annotated_genes']}g / {mod_meta[m]['events']}ev)" for m in top_mods], fontsize=8.6, fontweight="bold", color=ED_NAVY)
    if len(unit_order) <= 24:
        ax0.set_xticks(range(len(unit_order)))
        ax0.set_xticklabels([f"{u} ({cond_by_unit.get(u, '')})" for u in unit_order], rotation=45, ha="right", fontsize=7.6, color=ED_NAVY)
    else:
        ax0.set_xticks([])
        ax0.set_xlabel(f"{len(unit_order)} biological replicate / donor-celltype units ordered by condition", fontsize=8.5, color=ED_NAVY)
    ax0.set_title("A. Module Eigengene Activity Across Conditions", loc="left", fontsize=10.4, fontweight="bold", color=ED_NAVY)
    cb = fig.colorbar(im, ax=ax0, shrink=0.84, pad=0.025)
    cb.set_label("Standardized Module Eigengene (Z)", fontsize=8.0, color=ED_SLATE)

    rev_mods = list(reversed(modules_json[:10]))
    y_pos = np.arange(len(rev_mods))
    ev_cnts = np.array([m["events"] for m in rev_mods], dtype=float)
    cols_b = [ED_BLUE if (m.get("omnibus_fdr") or 1.0) <= 0.05 or (m.get("best_fdr") or 1.0) <= 0.05 else "#94A3B8" for m in rev_mods]
    ax1.barh(y_pos, ev_cnts, color=cols_b, height=0.60, alpha=0.92)
    max_ev = max(1.0, float(np.max(ev_cnts)))
    for yi, m in enumerate(rev_mods):
        q_show = min(float(m.get("omnibus_fdr") or 1.0), float(m.get("best_fdr") or 1.0))
        sig_tag = f"FDR={q_show:.2g} ★" if q_show <= 0.05 else f"FDR={q_show:.2g}"
        ax1.text(m["events"] + max_ev * 0.025, yi, f"{m['events']} ev · {m['unique_annotated_genes']} genes ({sig_tag})", va="center", fontsize=7.8, color=ED_NAVY, fontweight="bold" if q_show <= 0.05 else "normal")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m["module"] for m in rev_mods], fontsize=8.8, fontweight="bold", color=ED_NAVY)
    ax1.set_xlim(0, max_ev * 1.56)
    ax1.set_xlabel("Assigned Co-Splicing Events (Blue = BH FDR ≤ 0.05)", fontsize=8.8, color=ED_NAVY)
    ax1.set_title("B. Module Size & Condition Significance", loc="left", fontsize=10.4, fontweight="bold", color=ED_NAVY)

    png_p = fig_dir / "splicenet_module_overview.png"
    svg_p = _save_fig(fig, png_p)
    plot_tsv = tbl_dir / "splicenet_module_overview_plot_data.tsv"
    pd.DataFrame(modules_json).to_csv(plot_tsv, sep="\t", index=False)

    panels.append({
        "id": "splicenet-module-overview",
        "chapter": 6,
        "contrast_id": "all",
        "title": ("Katana-SC-SpliceNet" if is_single_cell else "Katana-SpliceNet") + " module eigengene structure & condition significance",
        "caption": "Left: standardized module eigengene scores across biological replicate units. Right: assigned splicing events, unique host genes, and BH FDR significance.",
        "image": png_p.relative_to(outdir).as_posix(),
        "svg": svg_p.relative_to(outdir).as_posix(),
        "metadata": {"analysis_id": "programs.splicenet"},
        "data_links": [{"role": "plot_data", "label": "Module summary & FDR data (TSV)", "path": plot_tsv.relative_to(outdir).as_posix(), "unit": "splicing_module", "contrast_id": "all", "policy": None}],
    })


def _render_static_contrast_forest(
    assoc_df: pd.DataFrame,
    cid: str,
    outdir: Path,
    fig_dir: Path,
    tbl_dir: Path,
    panels: list[dict[str, Any]],
) -> None:
    sub = assoc_df[(assoc_df["kind"] == "contrast") & (assoc_df["contrast_id"] == cid)].copy()
    if sub.empty:
        return
    sub = sub.sort_values(["q_value", "p_value"]).head(12).iloc[::-1].reset_index(drop=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", cid)
    plot_tsv = tbl_dir / f"{slug}_module_forest_plot_data.tsv"
    sub.assign(comparison_id=cid).to_csv(plot_tsv, sep="\t", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11.6, max(3.4, 0.88 * len(sub) + 1.4)), dpi=210)
    fig.patch.set_facecolor("white")
    ax.axvline(0.0, color=ED_SLATE, linestyle="--", linewidth=1.0, alpha=0.7)

    max_extent = 1.8
    for row in sub.itertuples(index=False):
        max_extent = max(max_extent, abs(float(row.effect or 0.0)) + float(row.ci95 or 0.0) + 0.85)

    for yi, row in sub.iterrows():
        diff = float(row["effect"] or 0.0)
        ci95 = float(row["ci95"] or 0.0)
        qv = float(row["q_value"] if pd.notna(row["q_value"]) else 1.0)
        pv = float(row["p_value"] if pd.notna(row["p_value"]) else 1.0)
        is_sig = qv <= 0.05
        col = (ED_CORAL if diff >= 0 else ED_BLUE) if is_sig else "#94A3B8"
        ax.errorbar(diff, yi, xerr=ci95, fmt="o", color=col, ecolor=col, elinewidth=2.1 if is_sig else 1.3, capsize=3.5, markersize=6.8 if is_sig else 5.2, zorder=3)
        star = " ★" if is_sig else ""
        x_txt = float(np.clip(diff, -max_extent * 0.62, max_extent * 0.62))
        ax.text(
            x_txt,
            yi + 0.24,
            f"ΔME={diff:+.2f} (P={pv:.2g}, FDR={qv:.2g}){star}",
            va="bottom",
            ha="center",
            fontsize=8.0,
            color=ED_NAVY if is_sig else ED_SLATE,
            fontweight="bold" if is_sig else "normal",
            bbox={"boxstyle": "round,pad=0.14", "fc": "white", "ec": "none", "alpha": 0.88},
            zorder=4,
        )

    ax.set_yticks(range(len(sub)))
    ax.set_yticklabels(sub["module"].tolist(), fontsize=9.2, fontweight="bold", color=ED_NAVY)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-0.48, len(sub) - 0.28)
    ax.set_xlabel(f"Welch Module Eigengene Difference ({cid}, ±95% CI)", fontsize=9.0, fontweight="bold", color=ED_NAVY)
    ax.set_title(f"Module Eigengene Contrast Response | {cid}", loc="left", fontsize=10.6, fontweight="bold", color=ED_NAVY)

    png_p = fig_dir / f"{slug}_module_contrast_forest.png"
    svg_p = _save_fig(fig, png_p)
    panels.append({
        "id": f"{slug}-splicenet-forest",
        "chapter": 6,
        "contrast_id": cid,
        "title": f"Module eigengene Welch contrast response (±95% CI & BH FDR) | {cid}",
        "caption": "Module-level directional splicing shifts across independent biological units; ★ marks BH FDR ≤ 0.05.",
        "image": png_p.relative_to(outdir).as_posix(),
        "svg": svg_p.relative_to(outdir).as_posix(),
        "metadata": {"analysis_id": "programs.splicenet"},
        "data_links": [{"role": "plot_data", "label": "Contrast ΔME forest data (TSV)", "path": plot_tsv.relative_to(outdir).as_posix(), "unit": "splicing_module", "contrast_id": cid, "policy": None}],
    })


def _render_static_module_ora(
    ora_df: pd.DataFrame,
    outdir: Path,
    fig_dir: Path,
    tbl_dir: Path,
    panels: list[dict[str, Any]],
) -> None:
    top = ora_df.sort_values(["q_BH", "p_value"]).groupby("module", as_index=False).head(4).sort_values(["q_BH", "p_value"]).head(12).iloc[::-1].copy()
    if top.empty:
        return
    plot_tsv = tbl_dir / "module_ora_top_plot_data.tsv"
    top.to_csv(plot_tsv, sep="\t", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11.8, max(3.6, 0.40 * len(top) + 1.4)), dpi=210)
    fig.patch.set_facecolor("white")

    neg_p = -np.log10(np.clip(top["p_value"].to_numpy(dtype=float), 1e-30, 1.0))
    q_vals = top["q_BH"].to_numpy(dtype=float)
    ov = top["overlap"].to_numpy(dtype=float)
    y_pos = np.arange(len(top))
    cols = [ED_BLUE if q <= 0.05 else "#64748B" for q in q_vals]
    sizes = np.clip(ov * 14.0 + 42.0, 50.0, 260.0)

    ax.hlines(y_pos, 0, neg_p, color="#CBD5E1", linewidth=1.8, zorder=1)
    ax.scatter(neg_p, y_pos, s=sizes, c=cols, alpha=0.92, edgecolors="white", linewidths=0.8, zorder=3)
    labels = []
    for r in top.itertuples(index=False):
        clean_t = re.sub(r"^(?:GO:\d+\s+|mmu\d+\s+|hsa\d+\s+)", "", str(r.pathway)).strip()
        if len(clean_t) > 52:
            clean_t = clean_t[:50].rsplit(" ", 1)[0] + "…"
        labels.append(f"[{r.module} · {r.ontology}] {clean_t}")
    max_x = max(2.2, float(np.max(neg_p)))
    for yi, (xv, qv, ov_i) in enumerate(zip(neg_p, q_vals, ov)):
        ax.text(xv + max_x * 0.02, yi, f"n={int(ov_i)}, FDR={qv:.2g}", va="center", fontsize=7.8, color=ED_NAVY if qv <= 0.05 else ED_SLATE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.4, color=ED_NAVY)
    ax.set_xlim(0, max_x * 1.30)
    ax.set_xlabel("−log10(Hypergeometric P) [Gene-Deduplicated Host Background · Blue = BH FDR ≤ 0.05]", fontsize=8.8, fontweight="bold", color=ED_NAVY)
    ax.set_title("Deduplicated Host-Gene GO & KEGG Pathway Enrichment Across Co-Splicing Modules", loc="left", fontsize=10.6, fontweight="bold", color=ED_NAVY)

    png_p = fig_dir / "splicenet_module_ora_dotplot.png"
    svg_p = _save_fig(fig, png_p)
    panels.append({
        "id": "splicenet-module-ora",
        "chapter": 6,
        "contrast_id": "all",
        "title": "Deduplicated host-gene GO & KEGG enrichment across co-splicing modules",
        "caption": "Each host gene is counted at most once per module (eliminating multi-exon gene inflation); blue lollipops satisfy BH FDR ≤ 0.05.",
        "image": png_p.relative_to(outdir).as_posix(),
        "svg": svg_p.relative_to(outdir).as_posix(),
        "metadata": {"analysis_id": "programs.splicenet"},
        "data_links": [{"role": "plot_data", "label": "Plotted module ORA terms (TSV)", "path": plot_tsv.relative_to(outdir).as_posix(), "unit": "module_pathway", "contrast_id": "all", "policy": None}],
    })


def _render_network_explorer_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Katana-SpliceNet · Interactive Multi-Community Co-Splicing Gene Network</title>
<style>
  :root {{
    --navy: #051c2c;
    --ink: #1e293b;
    --blue: #2251ff;
    --coral: #c83e4d;
    --teal: #0d9488;
    --slate: #64748b;
    --light: #e2e8f0;
    --surface: #f8fafc;
    --font-sans: "IBM Plex Sans", "Inter", "Segoe UI", -apple-system, Arial, sans-serif;
    --font-serif: "Newsreader", Georgia, serif;
    --font-mono: "IBM Plex Mono", Consolas, monospace;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    padding: 16px 18px 20px;
    background: #ffffff;
    color: var(--ink);
    font: 13px/1.5 var(--font-sans);
  }}
  .eyebrow {{
    font-size: 10.5px;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--blue);
    text-transform: uppercase;
  }}
  .head-row {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 12px;
    flex-wrap: wrap;
    margin-bottom: 8px;
  }}
  h2 {{
    margin: 2px 0 0;
    font: 600 20px/1.2 var(--font-serif);
    color: var(--navy);
  }}
  .stat-pill {{
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    font-size: 11.5px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 4px;
  }}
  .hint {{
    margin: 0 0 12px;
    color: #475569;
    font-size: 12.2px;
  }}
  .mod-cards {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(225px, 1fr));
    gap: 9px;
    margin-bottom: 12px;
  }}
  .mod-card {{
    text-align: left;
    padding: 10px 12px;
    border-radius: 5px;
    border: 1px solid var(--light);
    background: #ffffff;
    cursor: pointer;
    font: inherit;
    transition: border-color 0.12s, background 0.12s;
  }}
  .mod-card:hover {{ background: var(--surface); }}
  .mod-card.active {{ background: #eff6ff; border: 2px solid var(--blue); }}
  .mod-card-top {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 6px;
    margin-bottom: 3px;
  }}
  .badge {{
    font-size: 10px;
    font-weight: 700;
    padding: 2px 6px;
    border-radius: 3px;
    white-space: nowrap;
  }}
  .badge.sig {{ background: #dbeafe; color: #1e40af; }}
  .badge.nom {{ background: #fef3c7; color: #92400e; }}
  .badge.ns {{ background: #f1f5f9; color: #64748b; }}
  .controls-bar {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    align-items: center;
    padding: 9px 12px;
    background: var(--surface);
    border: 1px solid var(--light);
    border-radius: 4px;
    margin-bottom: 10px;
  }}
  .controls-bar label {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 11.5px;
    font-weight: 600;
    color: var(--navy);
  }}
  .controls-bar select, .controls-bar input[type="search"] {{
    padding: 4px 8px;
    border: 1px solid #cbd5e1;
    border-radius: 3px;
    font: inherit;
    background: #ffffff;
  }}
  .bridge-strip {{
    padding: 7px 11px;
    background: #f8fafc;
    border-left: 3px solid var(--blue);
    font-size: 11.5px;
    margin-bottom: 10px;
  }}
  .rbp-tag {{
    display: inline-block;
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    color: #1e40af;
    font-size: 10.8px;
    font-weight: 600;
    padding: 1px 7px;
    border-radius: 3px;
    margin: 2px 5px 2px 0;
  }}
  .tabs {{
    display: flex;
    gap: 4px;
    border-bottom: 2px solid var(--navy);
    margin-bottom: 12px;
    flex-wrap: wrap;
  }}
  .tab-btn {{
    border: 1px solid var(--light);
    border-bottom: none;
    background: var(--surface);
    color: #475569;
    padding: 7px 13px;
    font: 600 11.8px/1 var(--font-sans);
    cursor: pointer;
    border-radius: 4px 4px 0 0;
  }}
  .tab-btn.active {{
    background: var(--navy);
    color: #ffffff;
    border-color: var(--navy);
  }}
  .net-layout {{
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 14px;
  }}
  .net-layout.split {{
    grid-template-columns: minmax(0, 1.58fr) minmax(248px, 1fr);
  }}
  .canvas-card {{
    border: 1px solid var(--light);
    border-top: 2px solid var(--navy);
    border-radius: 3px;
    padding: 12px 14px;
    background: #ffffff;
  }}
  .canvas-head {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 6px;
  }}
  .canvas-head h3 {{
    margin: 0;
    font: 600 16px/1.25 var(--font-serif);
    color: var(--navy);
  }}
  .legend-row {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-size: 11.2px;
    color: #334155;
    margin-bottom: 8px;
  }}
  .svg-box {{
    border: 1px solid #f1f5f9;
    border-radius: 4px;
    overflow: hidden;
    background: #ffffff;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 11.8px;
  }}
  th, td {{
    text-align: left;
    padding: 7px 10px 7px 0;
    border-bottom: 1px solid var(--light);
  }}
  th {{
    border-top: 1.5px solid var(--navy);
    border-bottom: 1px solid #94a3b8;
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--navy);
  }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .link-btn {{
    border: 1px solid #bfdbfe;
    background: #eff6ff;
    color: var(--blue);
    font-size: 11px;
    font-weight: 600;
    padding: 3px 8px;
    border-radius: 3px;
    cursor: pointer;
  }}
  .link-btn:hover {{ background: var(--blue); color: #ffffff; }}
  .mod-card {{ overflow-wrap: anywhere; }}
  @media (max-width: 700px) {{
    .net-layout.split {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>
<div id="app"></div>
<script id="net-data" type="application/json">{data_json}</script>
<script>
(() => {{
  const DATA = JSON.parse(document.getElementById('net-data').textContent);
  const PALETTE = {json.dumps(MODULE_PALETTE)};
  const HALO = {json.dumps(MODULE_HALO)};

  const modColor = {{}};
  const modHalo = {{}};
  DATA.modules.forEach((m, i) => {{
    modColor[m.module] = PALETTE[i % PALETTE.length];
    modHalo[m.module] = HALO[i % HALO.length];
  }});

  let state = {{
    scope: DATA.graphs.ALL ? 'ALL' : (DATA.modules[0]?.module || 'M1'),
    tab: 'network',
    minKme: 0.0,
    zoom: 1.0,
    selectedEvent: '',
    search: '',
  }};

  function fmtP(v) {{
    if (v == null || Number.isNaN(Number(v))) return 'NA';
    const n = Number(v);
    if (n <= 0) return '< 1e-4';
    if (n < 0.001) return n.toExponential(2);
    return n.toFixed(4);
  }}

  function esc(s) {{
    return String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }}

  function notifyHeight() {{
    const measure = () => {{
      const appEl = document.getElementById('app');
      const rectH = appEl ? Math.ceil(appEl.getBoundingClientRect().height) : 0;
      const h = rectH > 120 ? (rectH + 34) : (document.body.scrollHeight || 560);
      try {{ parent.postMessage({{ type: 'katana-iframe-height', height: h }}, '*'); }} catch (_) {{}}
    }};
    requestAnimationFrame(measure);
    setTimeout(measure, 90);
  }}

  window.openGeneCard = (gene, eventId) => {{
    try {{ parent.postMessage({{ type: 'katana-open-genecard', gene, eventId: eventId || '' }}, '*'); }} catch (_) {{}}
  }};

  function computeSpatialLayout(graph, isAllMode) {{
    const activeNodes = (graph?.nodes || []).filter(id => (graph.node_kme?.[id] ?? 1) >= state.minKme);
    const W = isAllMode ? 1080 : 680;
    let H = isAllMode ? 600 : 450;
    const cxMain = W / 2;
    let cyMain = isAllMode ? 305 : 230;
    const pos = new Map();
    const halos = [];

    if (isAllMode) {{
      const mods = Array.from(new Set(activeNodes.map(id => graph.node_modules?.[id] || 'M1')));
      const nM = Math.max(1, mods.length);
      const Rx = nM === 1 ? 0 : (nM === 2 ? 255 : (nM <= 4 ? 250 : 275));
      const Ry = nM === 1 ? 0 : (nM === 2 ? 0 : (nM <= 4 ? 152 : 176));
      const rHalo = nM === 1 ? 160 : (nM === 2 ? 148 : (nM <= 4 ? 114 : 92));
      H = nM <= 2 ? 410 : (nM <= 4 ? 580 : 630);
      cyMain = H / 2 + 6;

      mods.forEach((mName, mIdx) => {{
        const theta = nM === 2 ? (mIdx === 0 ? Math.PI : 0) : (-Math.PI / 2 + (mIdx * 2 * Math.PI) / nM);
        const ccx = cxMain + Rx * Math.cos(theta);
        const ccy = cyMain + Ry * Math.sin(theta);
        const mRec = DATA.modules.find(x => x.module === mName) || {{}};
        const fdr = Math.min(Number(mRec.omnibus_fdr ?? 1), Number(mRec.best_fdr ?? 1));
        const pv = Math.min(Number(mRec.omnibus_p ?? 1), Number(mRec.best_p ?? 1));
        const sigText = fdr <= 0.05 ? `FDR=${{fmtP(fdr)}} ★` : (pv <= 0.05 ? `P=${{fmtP(pv)}} · FDR=${{fmtP(fdr)}}` : `FDR=${{fmtP(fdr)}} (NS)`);
        halos.push({{ mod: mName, cx: ccx, cy: ccy, r: rHalo, sigText, isSig: fdr <= 0.05 }});

        const mNodes = activeNodes.filter(id => (graph.node_modules?.[id] || 'M1') === mName);
        const coreK = mNodes.length > 6 ? Math.min(3, Math.ceil(mNodes.length * 0.25)) : 0;
        const core = mNodes.slice(0, coreK);
        const outer = mNodes.slice(coreK);

        core.forEach((id, ci) => {{
          const a = -Math.PI / 2 + (ci * 2 * Math.PI) / Math.max(1, core.length) + 0.28;
          const rr = core.length === 1 ? 0 : rHalo * 0.31;
          pos.set(id, {{ x: ccx + rr * Math.cos(a), y: ccy + rr * Math.sin(a), cx: ccx, cy: ccy, mod: mName, isCore: true, rank: ci }});
        }});
        outer.forEach((id, oi) => {{
          const a = -Math.PI / 2 + (oi * 2 * Math.PI) / Math.max(1, outer.length);
          const stagger = (oi % 2 === 0 ? 0.74 : 0.62) * rHalo;
          pos.set(id, {{ x: ccx + stagger * Math.cos(a), y: ccy + stagger * Math.sin(a), cx: ccx, cy: ccy, mod: mName, isCore: coreK === 0 && oi < 2, rank: coreK + oi }});
        }});
      }});
    }} else {{
      const mName = state.scope;
      const nTot = activeNodes.length;
      const coreK = Math.min(5, Math.max(2, Math.floor(nTot * 0.28)));
      const core = activeNodes.slice(0, coreK);
      const outer = activeNodes.slice(coreK);
      halos.push({{ mod: mName, cx: cxMain, cy: cyMain, r: 184, sigText: '', isSig: true }});

      core.forEach((id, i) => {{
        const a = -Math.PI / 2 + (i * 2 * Math.PI) / Math.max(1, core.length) + 0.2;
        const rr = outer.length ? 64 : 118;
        pos.set(id, {{ x: cxMain + rr * Math.cos(a), y: cyMain + rr * Math.sin(a), cx: cxMain, cy: cyMain, mod: mName, isCore: true, rank: i }});
      }});
      outer.forEach((id, i) => {{
        const a = -Math.PI / 2 + ((i + 0.35) * 2 * Math.PI) / Math.max(1, outer.length);
        const rx = (i % 2 === 0 ? 176 : 142);
        const ry = (i % 2 === 0 ? 144 : 118);
        pos.set(id, {{ x: cxMain + rx * Math.cos(a), y: cyMain + ry * Math.sin(a), cx: cxMain, cy: cyMain, mod: mName, isCore: false, rank: coreK + i }});
      }});
    }}

    // Compute radial outward label positions + vertical collision resolution per left/right hemisphere of each cluster!
    const byMod = new Map();
    activeNodes.forEach(id => {{
      const p = pos.get(id);
      if (!p) return;
      const dx = p.x - p.cx;
      const dy = p.y - p.cy;
      const side = Math.abs(dx) < 8 && Math.abs(dy) < 8 ? 'start' : (dx >= 0 ? 'start' : 'end');
      p.anchor = side;
      p.lx = p.x + (side === 'start' ? 9 : -9);
      p.ly = p.y + 3.5;
      const key = `${{p.mod}}:${{side}}`;
      if (!byMod.has(key)) byMod.set(key, []);
      byMod.get(key).push(id);
    }});

    byMod.forEach(ids => {{
      ids.sort((a, b) => pos.get(a).ly - pos.get(b).ly);
      const minGap = 14.5;
      for (let iter = 0; iter < 12; iter++) {{
        for (let i = 1; i < ids.length; i++) {{
          const pa = pos.get(ids[i - 1]);
          const pb = pos.get(ids[i]);
          if (pb.ly - pa.ly < minGap) {{
            const mid = 0.5 * (pa.ly + pb.ly);
            pa.ly = mid - 0.5 * minGap;
            pb.ly = mid + 0.5 * minGap;
          }}
        }}
      }}
    }});

    // Tight vertical bounding-box fit: eliminates dead empty space below halos while guaranteeing zero top/bottom clipping
    let minY = Infinity, maxY = -Infinity;
    halos.forEach(h => {{
      minY = Math.min(minY, h.cy - h.r - (isAllMode ? 32 : 14));
      maxY = Math.max(maxY, h.cy + h.r + 14);
    }});
    activeNodes.forEach(id => {{
      const p = pos.get(id);
      if (!p) return;
      minY = Math.min(minY, p.y - 12, p.ly - 12);
      maxY = Math.max(maxY, p.y + 12, p.ly + 10);
    }});
    if (Number.isFinite(minY) && Number.isFinite(maxY)) {{
      const padT = 18, padB = 18;
      const dyShift = padT - minY;
      halos.forEach(h => {{ h.cy += dyShift; }});
      activeNodes.forEach(id => {{
        const p = pos.get(id);
        if (!p) return;
        p.y += dyShift; p.cy += dyShift; p.ly += dyShift;
      }});
      H = Math.max(300, Math.ceil((maxY - minY) + padT + padB));
    }}

    return {{ W, H, activeNodes, pos, halos }};
  }}

  function renderStripSvg(title, unit, items, zeroToOne) {{
    if (!items.length) return `<p class="hint">Per-unit trajectory not recorded.</p>`;
    const conds = Array.from(new Set(items.map(d => d.condition)));
    const vals = items.map(d => d.value).filter(v => v != null && !Number.isNaN(v));
    if (!vals.length) return `<p class="hint">No finite measurements for selected feature.</p>`;
    const vMin = zeroToOne ? 0 : Math.min(...vals) - 0.25;
    const vMax = zeroToOne ? 1 : Math.max(...vals) + 0.25;
    const W = 380, H = 175, pL = 44, pR = 12, pT = 22, pB = 42;
    const pw = W - pL - pR, ph = H - pT - pB;
    const yMap = v => pT + (1 - (v - vMin) / Math.max(1e-6, vMax - vMin)) * ph;

    let svg = `<svg viewBox="0 0 ${{W}} ${{H}}" style="width:100%;height:auto;display:block">`;
    svg += `<text x="0" y="13" font-size="11" font-weight="700" fill="#051C2C">${{esc(title)}}</text>`;
    svg += `<rect x="${{pL}}" y="${{pT}}" width="${{pw}}" height="${{ph}}" fill="#F8FAFC" stroke="#E2E8F0"/>`;
    [0, 0.5, 1].forEach(f => {{
      const v = vMin + f * (vMax - vMin);
      const y = yMap(v);
      svg += `<line x1="${{pL}}" y1="${{y}}" x2="${{pL + pw}}" y2="${{y}}" stroke="#E2E8F0"/>`;
      svg += `<text x="${{pL - 5}}" y="${{y + 3}}" text-anchor="end" font-size="8.5" fill="#64748B">${{v.toFixed(2)}}</text>`;
    }});
    conds.forEach((c, ci) => {{
      const cx = pL + ((ci + 0.5) / Math.max(1, conds.length)) * pw;
      const g = items.filter(d => d.condition === c && d.value != null);
      const col = PALETTE[ci % PALETTE.length];
      if (g.length) {{
        const mean = g.reduce((a, b) => a + b.value, 0) / g.length;
        svg += `<line x1="${{cx - 16}}" y1="${{yMap(mean)}}" x2="${{cx + 16}}" y2="${{yMap(mean)}}" stroke="${{col}}" stroke-width="2.2"/>`;
        g.slice(0, 48).forEach((pt, pi) => {{
          const jit = ((pi % 5) - 2) * 3.4;
          svg += `<circle cx="${{cx + jit}}" cy="${{yMap(pt.value)}}" r="3.6" fill="${{col}}" stroke="#fff" stroke-width="0.7"><title>${{esc(pt.sample)}} (${{esc(c)}}): ${{pt.value.toFixed(3)}}</title></circle>`;
        }});
      }}
      const partsC = c.replace(/__/g, ' ').split('_');
      if (partsC.length >= 2 && c.length > 10) {{
        svg += `<text x="${{cx}}" y="${{H - 18}}" text-anchor="middle" font-size="8.0" font-weight="600" fill="#051C2C">${{esc(partsC[0].slice(0, 11))}}</text>`;
        svg += `<text x="${{cx}}" y="${{H - 8}}" text-anchor="middle" font-size="7.8" fill="#475569">${{esc(partsC.slice(1).join(' ').slice(0, 11))}}</text>`;
      }} else {{
        const shortC = c.length > 14 ? c.slice(0, 13) + '…' : c;
        svg += `<text x="${{cx}}" y="${{H - 12}}" text-anchor="middle" font-size="8.4" font-weight="600" fill="#051C2C">${{esc(shortC)}}</text>`;
      }}
    }});
    svg += `</svg>`;
    return svg;
  }}

  function downloadTsv(filename, cols, rows) {{
    const lines = [cols.join('\\t')];
    rows.forEach(r => lines.push(cols.map(c => String(r[c] ?? '')).join('\\t')));
    const blob = new Blob([lines.join('\\n')], {{ type: 'text/tab-separated-values' }});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }}

  function render() {{
    const isAll = state.scope === 'ALL' && Boolean(DATA.graphs.ALL);
    const graph = DATA.graphs[isAll ? 'ALL' : state.scope] || DATA.graphs[DATA.modules[0]?.module];
    const layout = computeSpatialLayout(graph, isAll);
    const selKey = layout.activeNodes.includes(state.selectedEvent) ? state.selectedEvent : (layout.activeNodes[0] || '');
    const selMod = graph?.node_modules?.[selKey] || (isAll ? (DATA.modules[0]?.module || 'M1') : state.scope);
    const selGene = graph?.node_genes?.[selKey] || selKey;

    let cardsHtml = '';
    if (DATA.graphs.ALL) {{
      cardsHtml += `<button type="button" class="mod-card ${{isAll ? 'active' : ''}}" data-scope="ALL">
        <div class="mod-card-top"><strong style="color:#051C2C">Whole Network (All Modules)</strong><span class="badge sig">Multi-Cluster</span></div>
        <div style="font-size:11.5px;color:#334155">${{DATA.modules.length}} modules · ${{DATA.graphs.ALL.nodes.length}} labeled hub genes · ${{DATA.graphs.ALL.edges.length}} PSI edges</div>
        <small style="color:#64748B">Spatial community halos, module FDR badges &amp; signed correlations</small>
      </button>`;
    }}
    DATA.modules.forEach(m => {{
      const fdr = Math.min(Number(m.omnibus_fdr ?? 1), Number(m.best_fdr ?? 1));
      const pv = Math.min(Number(m.omnibus_p ?? 1), Number(m.best_p ?? 1));
      const cls = fdr <= 0.05 ? 'sig' : (pv <= 0.05 ? 'nom' : 'ns');
      const bTxt = fdr <= 0.05 ? `FDR = ${{fmtP(fdr)}} ★` : (pv <= 0.05 ? `P = ${{fmtP(pv)}}` : `FDR = ${{fmtP(fdr)}} (NS)`);
      const act = !isAll && state.scope === m.module;
      cardsHtml += `<button type="button" class="mod-card ${{act ? 'active' : ''}}" data-scope="${{esc(m.module)}}" style="border-left:4px solid ${{modColor[m.module]}}">
        <div class="mod-card-top"><strong style="color:#051C2C">${{esc(m.module)}} · ${{m.unique_annotated_genes}} genes (${{m.events}} ev)</strong><span class="badge ${{cls}}">${{bTxt}}</span></div>
        ${{m.best_contrast ? `<div style="font-size:11.2px;color:#1E293B"><strong>Peak:</strong> ${{esc(m.best_contrast.replace(/__/g, ' · ').replace(/_vs_/g, ' vs '))}} (ΔME=${{m.best_effect >= 0 ? '+' : ''}}${{Number(m.best_effect).toFixed(2)}})</div>` : ''}}
        ${{m.top_pathway ? `<div style="font-size:11px;color:#0F766E"><strong>GO/KEGG:</strong> ${{esc(m.top_pathway.slice(0, 40))}}</div>` : ''}}
        <small style="color:#64748B"><strong>Hubs:</strong> ${{esc(m.top_hub_genes)}}</small>
      </button>`;
    }});

    const bridges = (DATA.bridges || []).filter(b => isAll || b.module === state.scope);
    const bridgeHtml = bridges.length ? `<div class="bridge-strip"><strong>Enriched Upstream Cis-Motif RBP Regulators (${{isAll ? 'Study-Wide' : esc(state.scope)}}): </strong>
      ${{bridges.slice(0, 8).map(b => `<span class="rbp-tag">${{isAll ? esc(b.module) + ' ← ' : ''}}${{esc(b.rbp)}} (Z/FE=${{b.z_or_fe}}, P=${{fmtP(b.p_value)}})</span>`).join('')}}
    </div>` : '';

    let bodyHtml = '';
    if (state.tab === 'network') {{
      const activeSet = new Set(layout.activeNodes);
      let svg = `<svg viewBox="0 0 ${{layout.W}} ${{layout.H}}" style="width:100%;height:auto;display:block;transform:scale(${{state.zoom}});transform-origin:center center">`;

      // 1. Community Halos & FDR Headers
      layout.halos.forEach(h => {{
        const cHex = modColor[h.mod] || '#2251FF';
        const hHex = modHalo[h.mod] || '#EFF6FF';
        svg += `<circle cx="${{h.cx}}" cy="${{h.cy}}" r="${{h.r}}" fill="${{hHex}}" stroke="${{cHex}}" stroke-opacity="0.28" stroke-width="1.4" stroke-dasharray="5,4"/>`;
        if (isAll) {{
          const lbl = `${{h.mod}} (${{h.sigText}})`;
          const bw = Math.max(124, lbl.length * 6.5 + 16);
          svg += `<rect x="${{h.cx - bw / 2}}" y="${{h.cy - h.r - 22}}" width="${{bw}}" height="19" rx="9" fill="#FFFFFF" stroke="${{cHex}}" stroke-opacity="0.45"/>`;
          svg += `<text x="${{h.cx}}" y="${{h.cy - h.r - 9}}" text-anchor="middle" font-size="11" font-weight="700" fill="${{cHex}}">${{esc(lbl)}}</text>`;
        }}
      }});

      // 2. Signed Sample-PSI Correlation Edges
      (graph?.edges || []).forEach(e => {{
        const idA = graph.nodes[e[0]], idB = graph.nodes[e[1]];
        if (!activeSet.has(idA) || !activeSet.has(idB)) return;
        const pA = layout.pos.get(idA), pB = layout.pos.get(idB);
        if (!pA || !pB) return;
        const signedR = e[3] ?? e[2] ?? 0.5;
        const absW = Math.abs(e[2] ?? signedR);
        const neg = signedR < 0;
        const touchSel = idA === selKey || idB === selKey;
        svg += `<line x1="${{pA.x}}" y1="${{pA.y}}" x2="${{pB.x}}" y2="${{pB.y}}" stroke="${{neg ? '#C83E4D' : '#2251FF'}}" stroke-opacity="${{touchSel ? 0.85 : Math.min(0.56, 0.16 + absW * 0.44)}}" stroke-width="${{(touchSel ? 2.2 : 0.9 + absW * 1.6).toFixed(2)}}" ${{neg ? 'stroke-dasharray="4,2.5"' : ''}}/>`;
      }});

      // 3. Hub Nodes & Collision-Resolved Outward Gene Symbol Pills
      layout.activeNodes.forEach(id => {{
        const p = layout.pos.get(id);
        if (!p) return;
        const gSym = graph.node_genes?.[id] || id;
        const kme = graph.node_kme?.[id] ?? 0.65;
        const isSel = id === selKey;
        const cHex = isSel ? '#C83E4D' : (modColor[p.mod] || '#2251FF');
        const rad = (p.isCore ? 6.4 : 4.6) + kme * 2.8;
        const tw = gSym.length * 6.1 + 8;
        const rx = p.anchor === 'start' ? p.lx - 3 : p.lx - tw + 3;

        if (Math.abs(p.ly - (p.y + 3.5)) > 4.5) {{
          svg += `<line x1="${{p.x}}" y1="${{p.y}}" x2="${{p.lx}}" y2="${{p.ly - 3}}" stroke="#94A3B8" stroke-width="0.7"/>`;
        }}
        svg += `<g class="net-node" data-eid="${{esc(id)}}" style="cursor:pointer">`;
        svg += `<circle cx="${{p.x}}" cy="${{p.y}}" r="${{rad.toFixed(1)}}" fill="${{cHex}}" stroke="#FFFFFF" stroke-width="${{isSel ? 2.2 : 1.3}}"><title>${{esc(gSym)}} (${{esc(p.mod)}}) · |kME|=${{kme.toFixed(3)}} · ${{esc(id)}}</title></circle>`;
        svg += `<rect x="${{rx.toFixed(1)}}" y="${{(p.ly - 10).toFixed(1)}}" width="${{tw.toFixed(1)}}" height="13.5" rx="2.5" fill="#FFFFFF" fill-opacity="0.86"/>`;
        svg += `<text x="${{p.lx.toFixed(1)}}" y="${{p.ly.toFixed(1)}}" text-anchor="${{p.anchor}}" font-size="${{p.isCore || isSel ? 10.6 : 9.6}}" font-weight="${{p.isCore || isSel ? 700 : 600}}" fill="${{isSel ? '#C83E4D' : '#051C2C'}}">${{esc(gSym)}}</text>`;
        svg += `</g>`;
      }});
      svg += `</svg>`;

      const meItems = DATA.scores.map(s => ({{
        sample: s.sample_id,
        condition: DATA.samples.find(x => x.sample_id === s.sample_id)?.condition || '',
        value: s[selMod],
      }}));
      const psiArr = DATA.event_usage[selKey] || [];
      const psiItems = DATA.samples.map((s, idx) => ({{
        sample: s.sample_id,
        condition: s.condition,
        value: psiArr[idx] ?? null,
      }}));

      bodyHtml = `<div class="net-layout split">
        <div class="canvas-card">
          <div class="canvas-head">
            <h3>${{isAll ? `Whole-Study Co-Splicing Gene Network (${{layout.activeNodes.length}} Labeled Hub Genes Across ${{DATA.modules.length}} Modules)` : `Module ${{esc(state.scope)}} Hub-and-Spoke Co-Splicing Network (${{layout.activeNodes.length}} Hub Genes)`}}</h3>
          </div>
          <div class="legend-row">
            <span><strong style="color:#2251FF">━━</strong> Positive co-splicing (r &gt; 0)</span>
            <span><strong style="color:#C83E4D">╌╌</strong> Inverse / mutually-exclusive (r &lt; 0)</span>
            <span>Node size = Hub centrality |kME| · Click any node to inspect trajectories</span>
          </div>
          <div class="svg-box">${{svg}}</div>
        </div>
        <div class="canvas-card">
          <div class="canvas-head">
            <h3>Selected Hub: ${{esc(selGene)}} <small style="color:#64748B;font-family:var(--font-sans)">(${{esc(selMod)}} · |kME|=${{(graph?.node_kme?.[selKey] ?? 0).toFixed(3)}})</small></h3>
            <button type="button" class="link-btn" onclick="openGeneCard('${{esc(selGene)}}','${{esc(selKey)}}')">4-Track GeneCard →</button>
          </div>
          <p class="hint" style="margin-bottom:8px">Event ID: <code style="font-size:11px">${{esc(selKey)}}</code></p>
          ${{renderStripSvg(`Module ${{selMod}} Eigengene Trajectory`, 'Eigengene Z', meItems, false)}}
          <div style="height:10px"></div>
          ${{renderStripSvg(`${{selGene}} (${{graph?.node_types?.[selKey] || 'AS'}}) Inclusion PSI`, 'PSI', psiItems, true)}}
        </div>
      </div>`;
    }} else if (state.tab === 'associations') {{
      const rows = DATA.associations.filter(a => isAll || a.module === state.scope);
      bodyHtml = `<div class="canvas-card">
        <div class="canvas-head"><h3>Module Omnibus &amp; Pairwise Contrast Significance</h3>
          <button type="button" class="link-btn" id="dl-assoc">Export TSV</button></div>
        <table><thead><tr><th>Module</th><th>Test Kind</th><th>Comparison</th><th class="num">Effect (F / ΔME)</th><th class="num">P-Value</th><th class="num">BH FDR</th></tr></thead>
        <tbody>${{rows.map(r => `<tr><td><strong>${{esc(r.module)}}</strong></td><td>${{esc(r.kind)}}</td><td>${{esc(r.contrast_id)}}</td><td class="num">${{Number(r.effect ?? 0).toFixed(3)}}</td><td class="num">${{fmtP(r.p_value)}}</td><td class="num"><strong>${{fmtP(r.q_value)}}</strong></td></tr>`).join('')}}</tbody></table>
      </div>`;
    }} else if (state.tab === 'genes') {{
      const q = state.search.toLowerCase();
      const rows = DATA.genes.filter(g => (isAll || g.module === state.scope) && (!q || g.gene_symbol.toLowerCase().includes(q) || g.gene_id.toLowerCase().includes(q)));
      bodyHtml = `<div class="canvas-card">
        <div class="canvas-head"><h3>Module Hub Genes (${{rows.length}})</h3>
          <div style="display:flex;gap:8px"><input type="search" id="tbl-search" placeholder="Filter gene symbol…" value="${{esc(state.search)}}"><button type="button" class="link-btn" id="dl-genes">Export TSV</button></div></div>
        <table><thead><tr><th>Module</th><th>Gene Symbol</th><th>Splice Class</th><th class="num">Events</th><th class="num">Max |kME|</th><th>Inspect</th></tr></thead>
        <tbody>${{rows.slice(0, 80).map(r => `<tr><td><strong>${{esc(r.module)}}</strong></td><td><strong>${{esc(r.gene_symbol)}}</strong></td><td>${{esc(r.event_type)}}</td><td class="num">${{r.events_in_module}}</td><td class="num">${{Number(r.max_abs_kME).toFixed(3)}}</td><td><button type="button" class="link-btn" onclick="openGeneCard('${{esc(r.gene_symbol)}}','${{esc(r.top_event_id)}}')">GeneCard →</button></td></tr>`).join('')}}</tbody></table>
      </div>`;
    }} else if (state.tab === 'events') {{
      const q = state.search.toLowerCase();
      const rows = DATA.members.filter(m => (isAll || m.module === state.scope) && (!q || m.gene_symbol.toLowerCase().includes(q) || m.event_key.toLowerCase().includes(q)));
      bodyHtml = `<div class="canvas-card">
        <div class="canvas-head"><h3>Assigned Splicing Events (${{rows.length}})</h3>
          <div style="display:flex;gap:8px"><input type="search" id="tbl-search" placeholder="Filter gene or coordinate…" value="${{esc(state.search)}}"><button type="button" class="link-btn" id="dl-events">Export TSV</button></div></div>
        <table><thead><tr><th>Module</th><th>Gene</th><th>Type</th><th>Canonical Event Coordinate</th><th class="num">Signed kME</th><th>Inspect</th></tr></thead>
        <tbody>${{rows.slice(0, 80).map(r => `<tr><td><strong>${{esc(r.module)}}</strong></td><td><strong>${{esc(r.gene_symbol)}}</strong></td><td>${{esc(r.event_type)}}</td><td><code style="font-size:11px">${{esc(r.event_key)}}</code></td><td class="num">${{Number(r.kME).toFixed(3)}}</td><td><button type="button" class="link-btn" onclick="openGeneCard('${{esc(r.gene_symbol)}}','${{esc(r.event_key)}}')">GeneCard →</button></td></tr>`).join('')}}</tbody></table>
      </div>`;
    }} else if (state.tab === 'pathways') {{
      const rows = DATA.pathways.filter(p => isAll || p.module === state.scope);
      bodyHtml = `<div class="canvas-card">
        <div class="canvas-head"><h3>Deduplicated Host-Gene GO &amp; KEGG Pathways (${{rows.length}})</h3>
          <button type="button" class="link-btn" id="dl-pathways">Export TSV</button></div>
        <table><thead><tr><th>Module</th><th>Ontology</th><th>Pathway / Term</th><th class="num">Overlap</th><th class="num">P-Value</th><th class="num">BH FDR</th><th>Hit Genes</th></tr></thead>
        <tbody>${{rows.length ? rows.slice(0, 60).map(r => `<tr><td><strong>${{esc(r.module)}}</strong></td><td>${{esc(r.ontology || 'GO')}}</td><td>${{esc(r.pathway)}}</td><td class="num">${{r.overlap}}</td><td class="num">${{fmtP(r.p_value)}}</td><td class="num"><strong>${{fmtP(r.q_BH)}}</strong></td><td><small>${{esc(r.hit_genes || '')}}</small></td></tr>`).join('') : '<tr><td colspan="7">No pathways met overlap criteria for this module.</td></tr>'}}</tbody></table>
      </div>`;
    }}

    document.getElementById('app').innerHTML = `
      <div class="head-row">
        <div>
          <div class="eyebrow">${{esc(DATA.engine)}}</div>
          <h2>Multi-Community Co-Splicing Gene Network &amp; Module Statistical Significance</h2>
        </div>
        <span class="stat-pill">${{DATA.modules.length}} Co-Splicing Modules · ${{DATA.members.length}} Assigned Events</span>
      </div>
      <p class="hint">Groups splicing events whose exon inclusion (PSI) co-varies across biological samples/donors into unsupervised modules (<code>M1..M${{DATA.modules.length}}</code>), displaying spatial community halos, real sample-PSI correlations, and BH-adjusted contrast significance.</p>
      <div class="mod-cards">${{cardsHtml}}</div>
      <div class="controls-bar">
        <label>Network Scope
          <select id="scope-sel">
            ${{DATA.graphs.ALL ? `<option value="ALL" ${{isAll ? 'selected' : ''}}>All Modules · Whole Co-Splicing Network (${{DATA.modules.length}} modules)</option>` : ''}}
            ${{DATA.modules.map(m => `<option value="${{esc(m.module)}}" ${{!isAll && state.scope === m.module ? 'selected' : ''}}>${{esc(m.module)}} · ${{m.unique_annotated_genes}} genes (${{m.events}} events)</option>`).join('')}}
          </select>
        </label>
        <label>Hub |kME| ≥ <strong>${{state.minKme.toFixed(2)}}</strong>
          <input type="range" id="kme-range" min="0" max="0.80" step="0.05" value="${{state.minKme}}">
        </label>
        <label>Zoom <strong>${{Math.round(state.zoom * 100)}}%</strong>
          <input type="range" id="zoom-range" min="0.8" max="1.6" step="0.1" value="${{state.zoom}}">
        </label>
      </div>
      ${{bridgeHtml}}
      <div class="tabs">
        ${{[
          ['network', 'Interactive Gene Network'],
          ['associations', 'Contrast Significance (ΔME & FDR)'],
          ['genes', 'Module Hub Genes (kME)'],
          ['events', 'Assigned Splicing Events'],
          ['pathways', 'GO & KEGG Pathways'],
        ].map(([id, lbl]) => `<button type="button" class="tab-btn ${{state.tab === id ? 'active' : ''}}" data-tab="${{id}}">${{lbl}}</button>`).join('')}}
      </div>
      ${{bodyHtml}}
    `;

    document.querySelectorAll('.mod-card').forEach(btn => btn.addEventListener('click', () => {{
      state.scope = btn.dataset.scope; state.selectedEvent = ''; render();
    }}));
    document.getElementById('scope-sel')?.addEventListener('change', e => {{
      state.scope = e.target.value; state.selectedEvent = ''; render();
    }});
    document.getElementById('kme-range')?.addEventListener('input', e => {{
      state.minKme = Number(e.target.value); render();
    }});
    document.getElementById('zoom-range')?.addEventListener('input', e => {{
      state.zoom = Number(e.target.value); render();
    }});
    document.querySelectorAll('.tab-btn').forEach(btn => btn.addEventListener('click', () => {{
      state.tab = btn.dataset.tab; render();
    }}));
    document.querySelectorAll('.net-node').forEach(g => g.addEventListener('click', () => {{
      state.selectedEvent = g.dataset.eid; render();
    }}));
    document.getElementById('tbl-search')?.addEventListener('input', e => {{
      state.search = e.target.value; render();
      const el = document.getElementById('tbl-search');
      if (el) {{ el.focus(); el.setSelectionRange(el.value.length, el.value.length); }}
    }});
    document.getElementById('dl-assoc')?.addEventListener('click', () => downloadTsv('SpliceNet_Associations.tsv', ['module','kind','contrast_id','effect','p_value','q_value'], DATA.associations));
    document.getElementById('dl-genes')?.addEventListener('click', () => downloadTsv('SpliceNet_Hub_Genes.tsv', ['module','gene_symbol','gene_id','event_type','events_in_module','max_abs_kME'], DATA.genes));
    document.getElementById('dl-events')?.addEventListener('click', () => downloadTsv('SpliceNet_Events.tsv', ['module','gene_symbol','event_type','event_key','kME'], DATA.members));
    document.getElementById('dl-pathways')?.addEventListener('click', () => downloadTsv('SpliceNet_Pathways.tsv', ['module','ontology','pathway','overlap','p_value','q_BH','hit_genes'], DATA.pathways));

    notifyHeight();
  }}

  addEventListener('resize', notifyHeight);
  render();
}})();
</script>
</body>
</html>"""
