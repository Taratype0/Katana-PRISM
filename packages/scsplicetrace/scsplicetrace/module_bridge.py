"""Cross-Package Regulatory Bridge: Connect SpliceNet Co-Splicing Modules -> SpliceTrace 12-Channel Cis-Regulatory RBP Motifs.

Lives inside `splicetrace` (Bulk) and `scsplicetrace` (Single-Cell) because PWM sequence scanning
requires `--fasta` and the bundled 1.0 RBP library (`621` PWMs), keeping `splicenet` pure matrix-only.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

_BRIDGE_COORD_COLS = {
    "SE": ("exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "A3SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "A5SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "MXE": ("1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "RI": ("riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
}

_BRIDGE_REGION_NAMES = (
    "upstream_intron_0_35",
    "upstream_intron_35_85",
    "upstream_intron_85_200",
    "upstream_intron_200_500",
    "downstream_intron_0_35",
    "downstream_intron_35_85",
    "downstream_intron_85_200",
    "downstream_intron_200_500",
    "upstream_exon_3p_50",
    "alt_exon_5p_50",
    "alt_exon_3p_50",
    "downstream_exon_5p_50",
)


def _normalize_bridge_event_id(raw_id: object) -> str:
    s = str(raw_id or "").strip()
    if "|" not in s or "=" not in s:
        return s
    parts = s.split("|")
    kind = parts[0].strip().upper()
    kv = {}
    for tok in parts[1:]:
        if "=" in tok:
            k, v = tok.split("=", 1)
            kv[k.strip()] = v.strip()
    cols = _BRIDGE_COORD_COLS.get(kind)
    if cols and "chr" in kv and "strand" in kv and all(c in kv for c in cols):
        return f"{kind}:{kv['chr']}:{kv['strand']}:" + ":".join(kv[c] for c in cols)
    return s


def _seq_gc(seq: str) -> float:
    s = seq.upper().replace("U", "T")
    if not s:
        return 0.5
    tot = s.count("A") + s.count("C") + s.count("G") + s.count("T")
    return (s.count("G") + s.count("C")) / tot if tot > 0 else 0.5


def _fwl_resid(Y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w_root = np.sqrt(np.maximum(weights, 1e-12))
    Xw = X * w_root[:, None]
    if Y.ndim == 1:
        beta, _, _, _ = np.linalg.lstsq(Xw, Y * w_root, rcond=None)
        return Y - X @ beta
    beta, _, _, _ = np.linalg.lstsq(Xw, Y * w_root[:, None], rcond=None)
    return Y - X @ beta


def parse_rbp_expression_reference(
    rbp_expr_df: pd.DataFrame | None,
    *,
    module_scores_df: pd.DataFrame | None = None,
    candidate_rbps: set[str] | None = None,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame | None]:
    """Parse either a gene summary table (`gene_symbol`/`RBP` + `mean_expr`/`tpm` [+ `stat_z`/`log2fc`])
    or a sample x gene / gene x sample expression matrix into `(mean_expr_map, expr_z_map, sample_by_gene_df)`.
    """
    if rbp_expr_df is None or rbp_expr_df.empty:
        return {}, {}, None
    df = rbp_expr_df.copy()
    cols_lower = {str(c).lower(): c for c in df.columns}
    sym_col = next((cols_lower[k] for k in ("rbp", "gene_symbol", "gene_name", "symbol", "gene") if k in cols_lower), None)
    expr_col = next((cols_lower[k] for k in ("mean_expr", "mean_tpm", "tpm", "cpm", "fpkm", "basemean", "expression") if k in cols_lower), None)
    z_col = next((cols_lower[k] for k in ("rbp_expr_z", "stat_z", "z", "t", "stat", "log2fc", "log2foldchange") if k in cols_lower), None)

    if sym_col is not None and (expr_col is not None or z_col is not None):
        syms = df[sym_col].astype(str).str.strip().str.upper()
        mean_map: dict[str, float] = {}
        z_map: dict[str, float] = {}
        if expr_col is not None:
            vals = pd.to_numeric(df[expr_col], errors="coerce")
            for s, v in zip(syms, vals):
                if s and np.isfinite(v):
                    mean_map[s] = max(mean_map.get(s, -np.inf), float(v))
        if z_col is not None:
            zvals = pd.to_numeric(df[z_col], errors="coerce")
            for s, zv in zip(syms, zvals):
                if s and np.isfinite(zv):
                    if s not in z_map or abs(float(zv)) > abs(z_map[s]):
                        z_map[s] = float(zv)
        return mean_map, z_map, None

    # Matrix mode: either index = samples & columns = genes, or index/first-col = genes & columns = samples
    mat = df.copy()
    if sym_col is not None:
        mat = mat.set_index(sym_col)
    elif not pd.api.types.is_numeric_dtype(mat.iloc[:, 0]):
        mat = mat.set_index(mat.columns[0])
    num_mat = mat.apply(pd.to_numeric, errors="coerce")
    num_mat = num_mat.dropna(how="all", axis=0).dropna(how="all", axis=1)
    if num_mat.empty:
        return {}, {}, None

    if module_scores_df is not None and not module_scores_df.empty:
        sc_idx = set(module_scores_df.index.astype(str))
        if len(set(num_mat.columns.astype(str)) & sc_idx) > len(set(num_mat.index.astype(str)) & sc_idx):
            num_mat = num_mat.T
    elif candidate_rbps:
        col_hits = len(set(num_mat.columns.astype(str).str.strip().str.upper()) & candidate_rbps)
        row_hits = len(set(num_mat.index.astype(str).str.strip().str.upper()) & candidate_rbps)
        if row_hits > col_hits:
            num_mat = num_mat.T
        elif col_hits == 0 and row_hits == 0 and num_mat.shape[0] > num_mat.shape[1] * 3:
            num_mat = num_mat.T
    elif num_mat.shape[0] > num_mat.shape[1] * 3:
        # Likely genes x samples -> transpose to samples x genes
        num_mat = num_mat.T

    num_mat.columns = num_mat.columns.astype(str).str.strip().str.upper()
    if num_mat.columns.duplicated().any():
        num_mat = num_mat.T.groupby(level=0).max().T
    means = num_mat.mean(axis=0, skipna=True)
    mean_map = {str(k): float(v) for k, v in means.items() if np.isfinite(v)}
    return mean_map, {}, num_mat


def annotate_rbp_expression_disambiguation(
    ranking_df: pd.DataFrame,
    rbp_expr_df: pd.DataFrame | None,
    *,
    fwer_col: str = "rbp_P_FWER",
    z_col: str = "stat_Z",
    group_col: str | None = None,
    module_scores_df: pd.DataFrame | None = None,
    min_rbp_expr: float = 1.0,
) -> pd.DataFrame:
    """Attach non-circular cohort expression gating & paralog tie-breaking (`rbp_mean_expr`, `rbp_expressed`,
    `expressed_sequence_rank`) while leaving pure sequence `P_FWER` and `sequence_rank` untouched.
    """
    if ranking_df.empty or rbp_expr_df is None or rbp_expr_df.empty:
        return ranking_df
    cand_rbps = set(ranking_df["RBP"].astype(str).str.strip().str.upper())
    mean_map, z_map, samp_by_gene = parse_rbp_expression_reference(
        rbp_expr_df,
        module_scores_df=module_scores_df,
        candidate_rbps=cand_rbps,
    )
    if not mean_map and not z_map and samp_by_gene is None:
        return ranking_df

    out = ranking_df.copy()
    rbp_upper = out["RBP"].astype(str).str.strip().str.upper()
    out["rbp_mean_expr"] = [mean_map.get(u, np.nan) for u in rbp_upper]
    # Auto-adapt expression threshold for log1p-normalized single-cell matrices while preserving bulk TPM/CPM gating
    eff_min_expr = float(min_rbp_expr)
    pos_vals = np.array([v for v in mean_map.values() if np.isfinite(v) and v > 0.0], dtype=float)
    if eff_min_expr == 1.0 and len(pos_vals) >= 20 and float(np.quantile(pos_vals, 0.99)) <= 10.0 and float(np.max(pos_vals)) <= 15.0:
        eff_min_expr = 0.05
    out["rbp_expr_threshold"] = eff_min_expr
    # Only enforce expression gating when at least one RBP symbol actually maps into the provided expression table
    # (prevents Ensembl-ID-only matrices from falsely flagging 100% of RBPs as unexpressed)
    if mean_map and any(u in mean_map for u in rbp_upper):
        out["rbp_expressed"] = [bool(mean_map.get(u, 0.0) >= eff_min_expr) for u in rbp_upper]
    else:
        out["rbp_expressed"] = True

    co_signal = np.array([abs(z_map.get(u, 0.0)) for u in rbp_upper], dtype=float)
    if group_col is not None and group_col in out.columns and samp_by_gene is not None and module_scores_df is not None:
        ms = module_scores_df.copy()
        if not pd.api.types.is_numeric_dtype(ms.iloc[:, 0]) and ms.index.dtype == "int64":
            ms = ms.set_index(ms.columns[0])
        ms.index = ms.index.astype(str)
        common_s = [s for s in ms.index if s in set(samp_by_gene.index.astype(str))]
        if len(common_s) >= 4:
            sub_expr = samp_by_gene.loc[common_s]
            eig_corrs = np.zeros(len(out), dtype=float)
            for idx_r, (m_name, u_sym) in enumerate(zip(out[group_col].astype(str), rbp_upper)):
                if m_name in ms.columns and u_sym in sub_expr.columns:
                    xv = pd.to_numeric(ms.loc[common_s, m_name], errors="coerce").to_numpy(dtype=float)
                    yv = pd.to_numeric(sub_expr[u_sym], errors="coerce").to_numpy(dtype=float)
                    ok = np.isfinite(xv) & np.isfinite(yv)
                    if ok.sum() >= 4 and np.std(xv[ok]) > 1e-8 and np.std(yv[ok]) > 1e-8:
                        eig_corrs[idx_r] = float(np.corrcoef(xv[ok], yv[ok])[0, 1])
            out["eigengene_expr_r"] = np.round(eig_corrs, 4)
            co_signal = np.maximum(co_signal, np.abs(eig_corrs) * 4.0)
    elif z_map:
        out["rbp_expr_z"] = [round(z_map.get(u, np.nan), 4) if u in z_map else np.nan for u in rbp_upper]

    # Paralog tie-breaker: prioritize expressed RBPs, then sequence FWER tier (rounded to 0.25 log10 units), then sequence |Z| + 0.15*co_signal
    fwer_vals = np.clip(pd.to_numeric(out[fwer_col], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-6, 1.0)
    abs_z = np.abs(pd.to_numeric(out[z_col], errors="coerce").fillna(0.0).to_numpy(dtype=float))
    fwer_tier = np.round(-np.log10(fwer_vals) * 4.0) / 4.0
    tie_score = abs_z + 0.15 * np.minimum(co_signal, 6.0)
    out["_exp_flag"] = out["rbp_expressed"].astype(int)
    out["_fwer_tier"] = fwer_tier
    out["_tie_score"] = tie_score

    if group_col is not None and group_col in out.columns:
        out["expressed_sequence_rank"] = 0
        for _, grp_idx in out.groupby(group_col, sort=False).groups.items():
            sub_o = out.loc[grp_idx].sort_values(
                by=["_exp_flag", "_fwer_tier", "_tie_score", fwer_col],
                ascending=[False, False, False, True],
                kind="mergesort",
            )
            out.loc[sub_o.index, "expressed_sequence_rank"] = np.arange(1, len(sub_o) + 1, dtype=int)
    else:
        sub_o = out.sort_values(
            by=["_exp_flag", "_fwer_tier", "_tie_score", fwer_col],
            ascending=[False, False, False, True],
            kind="mergesort",
        )
        out.loc[sub_o.index, "expressed_sequence_rank"] = np.arange(1, len(sub_o) + 1, dtype=int)

    return out.drop(columns=["_exp_flag", "_fwer_tier", "_tie_score"])


def bridge_splicenet_modules_to_rbps(
    modules_df: pd.DataFrame,
    events_df: pd.DataFrame,
    fasta_path: Path | str,
    library: object,
    *,
    extract_rna_fn,
    score_odds_fn,
    distal_intron_nt: int = 500,
    rbp_expr_df: pd.DataFrame | None = None,
    module_scores_df: pd.DataFrame | None = None,
    min_rbp_expr: float = 1.0,
    min_module_events: int = 8,
    n_permutations: int = 999,
    seed: int = 20261118,
) -> pd.DataFrame:
    """Deconvolve trans-acting RBP cis-regulators across SpliceNet co-splicing modules.

    Contrasts each non-grey module M_k against the cohort's complementary SpliceNet universe
    (preserving exact gene-block exchangeability among regulated splicing events without constitutive-exon
    population shift!) across 12 spatial cis-regulatory channels (`0-35`, `35-85`, `85-200`, `200-500 nt`
    upstream/downstream introns + `50-nt` flanking/cassette exon edges) with FWL sequence-composition
    orthogonalization (`exon_gc`, `flank_gc`, `dist_gc`, `cpg_rate`, `log_len`, `event_type`), one-sided
    positive cis-binding enrichment (`Z > 0`), and gene-block max-T FWER calibration.
    """
    id_col = "event_id" if "event_id" in modules_df.columns else "event_key"
    if id_col not in modules_df.columns or "module" not in modules_df.columns:
        raise ValueError("modules_df must contain ('event_id' or 'event_key') and 'module'")

    mod = modules_df[[c for c in (id_col, "module", "kME") if c in modules_df.columns]].copy()
    mod = mod.rename(columns={id_col: "event_id"})
    mod["event_id"] = mod["event_id"].map(_normalize_bridge_event_id)

    def _norm_mod_label(v: object) -> str:
        s = str(v).strip()
        if s.lower() in ("0", "m0", "unassigned", "grey", "gray", "none", "nan", ""):
            return "M0"
        if s.isdigit():
            return f"M{int(s)}"
        return s

    mod["mod_label"] = mod["module"].map(_norm_mod_label)
    if "kME" not in mod.columns:
        mod["kME"] = np.where(mod["mod_label"].ne("M0"), 1.0, 0.0)

    ev = events_df.copy()
    ev["_orig_event_id"] = ev["event_id"].astype(str)
    ev["event_id"] = ev["event_id"].map(_normalize_bridge_event_id)
    ev = ev.drop_duplicates("event_id").reset_index(drop=True)

    merged = ev.merge(mod, on="event_id", how="inner").reset_index(drop=True)
    non_grey = sorted(
        [m for m in merged["mod_label"].unique() if m != "M0"],
        key=lambda x: (int(x[1:]) if x.startswith("M") and x[1:].isdigit() else 999, x),
    )
    if not non_grey or len(merged) < 12:
        return pd.DataFrame(columns=["module", "module_events", "module_genes", "RBP", "best_region", "kme_direction", "stat_Z", "nominal_P", "module_P_FWER"])

    # Only fall back to external background if modules_df had < 60 background/complementary events (e.g. single tiny module passed)
    if len(merged) < 80:
        mod_genes_active = set(merged["gene_id"].astype(str))
        extra_bg = ev[(~ev["event_id"].isin(set(merged["event_id"]))) & (~ev["gene_id"].astype(str).isin(mod_genes_active))].copy()
        if not extra_bg.empty:
            extra_bg = extra_bg.sample(n=min(len(extra_bg), 400), random_state=int(seed)).copy()
            extra_bg["module"] = 0
            extra_bg["mod_label"] = "M0"
            extra_bg["kME"] = 0.0
            merged = pd.concat([merged, extra_bg], ignore_index=True)

    if len(merged) < 20:
        return pd.DataFrame(columns=["module", "module_events", "module_genes", "RBP", "best_region", "kme_direction", "stat_Z", "nominal_P", "module_P_FWER"])

    if "comparison_id" not in merged.columns:
        merged["comparison_id"] = "splicenet_bridge"
    try:
        import importlib as _imp
        _ef_fn = _imp.import_module(f"{__name__.split('.')[0]}.rna_forms").event_forms
        _valid_m = []
        for _rec in merged.to_dict("records"):
            try:
                _ef_fn(_rec)
                _valid_m.append(True)
            except Exception:
                _valid_m.append(False)
        merged = merged.loc[_valid_m].reset_index(drop=True)
    except Exception:
        pass
    if len(merged) < 20:
        return pd.DataFrame(columns=["module", "module_events", "module_genes", "RBP", "best_region", "kme_direction", "stat_Z", "nominal_P", "module_P_FWER"])
    rna_df = extract_rna_fn(
        merged,
        Path(fasta_path),
        intron_nt=200,
        distal_intron_nt=int(distal_intron_nt),
        acceptor_core_nt=3,
        feature_only=False,
        compute_sha256=False,
    )

    models = library.models
    m_ids = list(models.keys())
    mid_idx = {mid: i for i, mid in enumerate(m_ids)}
    rbps = sorted(library.rbp_to_models.keys())
    rbp_cols = {r: [mid_idx[m] for m in library.rbp_to_models[r] if m in mid_idx] for r in rbps}

    ev_index = {str(eid): i for i, eid in enumerate(merged["event_id"])}
    etype_map = {str(eid): str(et) for eid, et in zip(merged["event_id"], merged["event_type"])}
    n_ev = len(merged)
    n_reg = len(_BRIDGE_REGION_NAMES)
    model_tensor = np.zeros((n_ev, n_reg, len(m_ids)), dtype=np.float64)

    mat_sub = rna_df[rna_df["layer"].eq("mature_local_form") & rna_df["region"].eq("inclusion")]
    if not mat_sub.empty:
        ex_seqs, ex_targets = [], []
        has_junc = "junctions_json" in mat_sub.columns
        cols_iter = ["event_id", "sequence", "junctions_json"] if has_junc else ["event_id", "sequence"]
        for rec in mat_sub[cols_iter].itertuples(index=False):
            ei = ev_index.get(str(rec.event_id))
            if ei is None:
                continue
            seq_inc = str(rec.sequence)
            joins = json.loads(str(rec.junctions_json)) if (has_junc and rec.junctions_json) else []
            if len(joins) >= 2:
                j1, j2 = int(joins[0]), int(joins[-1])
                for piece, sl_i in (
                    (seq_inc[max(0, j1 - 50) : j1], 8),
                    (seq_inc[j1 : min(j2, j1 + 50)], 9),
                    (seq_inc[max(j1, j2 - 50) : j2], 10),
                    (seq_inc[j2 : min(len(seq_inc), j2 + 50)], 11),
                ):
                    if len(piece) >= 6:
                        ex_seqs.append(piece)
                        ex_targets.append((ei, sl_i))
            elif len(joins) == 1:
                j1 = int(joins[0])
                for piece, sl_i in ((seq_inc[max(0, j1 - 50) : j1], 8), (seq_inc[j1 : min(len(seq_inc), j1 + 50)], 11)):
                    if len(piece) >= 6:
                        ex_seqs.append(piece)
                        ex_targets.append((ei, sl_i))
        if ex_seqs:
            ex_sc = score_odds_fn(ex_seqs, models)
            for idx_e, (ei, sl_i) in enumerate(ex_targets):
                model_tensor[ei, sl_i, :] = ex_sc[idx_e]

    reg_sub = rna_df[rna_df["layer"].eq("regulatory_pre_mRNA")]
    gc_num = np.zeros(n_ev, dtype=np.float64)
    gc_den = np.zeros(n_ev, dtype=np.float64)
    dist_gc_num = np.zeros(n_ev, dtype=np.float64)
    dist_gc_den = np.zeros(n_ev, dtype=np.float64)
    if not reg_sub.empty:
        sub_seqs, sub_targets = [], []
        for rec in reg_sub[["event_id", "region", "sequence"]].itertuples(index=False):
            ei = ev_index.get(str(rec.event_id))
            if ei is None:
                continue
            base_r = str(rec.region)
            seq_s = str(rec.sequence).upper()
            if not seq_s:
                continue
            is_distal = base_r.endswith("_distal")
            stem_r = base_r[:-7] if is_distal else base_r
            tot_nt = max(1, seq_s.count("A") + seq_s.count("C") + seq_s.count("G") + seq_s.count("T"))
            gc_cnt = seq_s.count("G") + seq_s.count("C")
            if is_distal:
                dist_gc_num[ei] += gc_cnt
                dist_gc_den[ei] += tot_nt
            else:
                gc_num[ei] += gc_cnt
                gc_den[ei] += tot_nt

            kind = etype_map.get(str(rec.event_id), "SE")
            route = None
            if kind == "SE":
                if stem_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif stem_r == "inclusion_junction_2_donor_intron":
                    route = (False, 1.0)
            elif kind == "RI":
                if stem_r == "exclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif stem_r == "exclusion_junction_1_donor_intron":
                    route = (False, 1.0)
            elif kind == "A3SS":
                if stem_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif stem_r == "exclusion_junction_1_acceptor_intron":
                    route = (True, -1.0)
                elif stem_r == "inclusion_junction_1_donor_intron":
                    route = (False, 1.0)
            elif kind == "A5SS":
                if stem_r == "inclusion_junction_1_donor_intron":
                    route = (False, 1.0)
                elif stem_r == "exclusion_junction_1_donor_intron":
                    route = (False, -1.0)
                elif stem_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
            elif kind == "MXE":
                if stem_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif stem_r == "exclusion_junction_1_acceptor_intron":
                    route = (True, -1.0)
                elif stem_r == "inclusion_junction_2_donor_intron":
                    route = (False, 1.0)
                elif stem_r == "exclusion_junction_2_donor_intron":
                    route = (False, -1.0)
            if route is None:
                continue
            is_up, sgn = route
            if is_distal:
                if len(seq_s) >= 12:
                    sub_seqs.append(seq_s)
                    sub_targets.append((ei, 3 if is_up else 7, sgn))
            else:
                L = len(seq_s)
                for b_i, (d0, d1) in enumerate(((0, 35), (35, 85), (85, 200))):
                    piece = seq_s[max(0, L - d1) : max(0, L - d0)] if is_up else seq_s[min(L, d0) : min(L, d1)]
                    if len(piece) >= 6:
                        sub_seqs.append(piece)
                        sub_targets.append((ei, (0 if is_up else 4) + b_i, sgn))
        if sub_seqs:
            b_sc = score_odds_fn(sub_seqs, models)
            for idx_s, (ei, slot, sgn) in enumerate(sub_targets):
                model_tensor[ei, slot, :] += sgn * b_sc[idx_s]

    rms_m = np.sqrt(np.mean(model_tensor * model_tensor, axis=0, keepdims=True))
    norm_m = np.divide(model_tensor, rms_m, out=np.zeros_like(model_tensor), where=rms_m > 1e-12)
    F_rbp = np.zeros((n_ev, n_reg, len(rbps)), dtype=np.float64)
    for r_i, r_name in enumerate(rbps):
        cols_r = rbp_cols[r_name]
        if cols_r:
            F_rbp[:, :, r_i] = np.mean(norm_m[:, :, cols_r], axis=2)

    aff = rna_df[rna_df["layer"].eq("affected_sequence")]
    exon_gc_map, aff_len_map = {}, {}
    for eid, grp in aff.groupby("event_id", sort=False):
        joined = "".join(grp["sequence"].astype(str))
        exon_gc_map[str(eid)] = _seq_gc(joined)
        aff_len_map[str(eid)] = float(len(joined))

    cpg_num = np.zeros(n_ev, dtype=np.float64)
    cpg_den = np.zeros(n_ev, dtype=np.float64)
    for rec in rna_df[["event_id", "sequence"]].itertuples(index=False):
        ei = ev_index.get(str(rec.event_id))
        if ei is not None:
            s = str(rec.sequence).upper()
            cpg_num[ei] += s.count("CG")
            cpg_den[ei] += max(len(s) - 1, 1)
    cpg_rate = cpg_num / np.maximum(cpg_den, 1.0)
    flank_gc = np.divide(gc_num, gc_den, out=np.full(n_ev, 0.5), where=gc_den > 0)
    dist_gc = np.divide(dist_gc_num, dist_gc_den, out=flank_gc, where=dist_gc_den > 0)
    exon_gc = np.array([exon_gc_map.get(str(x), 0.5) for x in merged["event_id"]], dtype=np.float64)
    log_len = np.array([float(np.log1p(aff_len_map.get(str(x), 100.0))) for x in merged["event_id"]], dtype=np.float64)
    etypes = merged["event_type"].astype(str).to_numpy()
    uniq_et = sorted(set(etypes))

    nuis_cols = [
        np.ones(n_ev, dtype=np.float64),
        exon_gc,
        flank_gc,
    ]
    if np.std(dist_gc - flank_gc) > 1e-6:
        nuis_cols.append(dist_gc)
    nuis_cols.extend([
        cpg_rate,
        log_len,
        *[(etypes == et).astype(np.float64) for et in uniq_et[1:]],
    ])
    X_nuis = np.column_stack(nuis_cols)

    genes = merged["gene_id"].astype(str).to_numpy()
    uniq_g, g_codes = np.unique(genes, return_inverse=True)
    n_genes = len(uniq_g)
    w = 1.0 / np.bincount(g_codes)[g_codes].astype(np.float64)
    w /= w.sum()

    F_flat = F_rbp.reshape(n_ev, n_reg * len(rbps))
    F_res = _fwl_resid(F_flat, X_nuis, w)
    sd0 = np.sqrt(np.maximum(np.sum((F_res * F_res) * w[:, None], axis=0), 1e-12))
    F_win = _fwl_resid(np.clip(F_res / sd0[None, :], -3.5, 3.5), X_nuis, w)
    F_g = np.zeros((n_genes, n_reg * len(rbps)), dtype=np.float64)
    np.add.at(F_g, g_codes, w[:, None] * F_win)
    F_gc = F_g - F_g.mean(axis=0, keepdims=True)
    sum_F2 = np.sum(F_gc * F_gc, axis=0)

    kme_vals = np.nan_to_num(merged["kME"].to_numpy(dtype=np.float64), nan=0.0)

    # Shared host-gene permutation matrix across all modules simultaneously for pivotal Westfall-Young minP study_P_FWER
    b_study = max(int(n_permutations), 4999)
    rng_study = np.random.default_rng(int(seed) + 7919)
    perm_shared = np.vstack([rng_study.permutation(n_genes) for _ in range(b_study)]) if n_genes >= 10 else None
    per_mod_cache: list[tuple[str, int, int, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    study_mod_maxima: list[np.ndarray] = []

    out_rows: list[dict] = []
    for mod_idx, m_lbl in enumerate(non_grey):
        in_m = merged["mod_label"].to_numpy() == m_lbl
        n_m_ev = int(in_m.sum())
        if n_m_ev < int(min_module_events):
            continue
        n_m_genes = int(len(set(genes[in_m])))
        if n_genes < 10:
            continue

        rng_m = np.random.default_rng(int(seed) + 1009 * mod_idx)
        perm_g = np.vstack([rng_m.permutation(n_genes) for _ in range(int(n_permutations))])

        arms = []
        if int((in_m & (kme_vals >= 0)).sum()) >= 6:
            arms.append(("kME_pos", np.where(in_m & (kme_vals >= 0), np.maximum(kme_vals, 0.25), 0.0)))
        if int((in_m & (kme_vals < 0)).sum()) >= 6:
            arms.append(("kME_neg", np.where(in_m & (kme_vals < 0), np.maximum(-kme_vals, 0.25), 0.0)))
        if not arms:
            arms.append(("kME_all", np.where(in_m, np.maximum(np.abs(kme_vals), 0.25), 0.0)))

        arm_Z_list, arm_score_list, arm_null_list, arm_study_null_list, arm_names = [], [], [], [], []
        for a_name, y_raw in arms:
            y_res = _fwl_resid(y_raw, X_nuis, w)
            y_g = np.zeros(n_genes, dtype=np.float64)
            np.add.at(y_g, g_codes, w * y_res)
            yc = y_g - y_g.mean()
            sd_perm = np.sqrt(np.sum(yc * yc) * sum_F2 / max(n_genes - 1, 1))
            valid_c = sd_perm > 1e-12
            Z_obs = np.divide(yc @ F_gc, sd_perm, out=np.zeros_like(sd_perm), where=valid_c)
            Y_perm = yc[perm_g]
            null_Z = np.divide(
                (Y_perm.astype(np.float32) @ F_gc.astype(np.float32)).astype(np.float64),
                sd_perm[None, :],
                out=np.zeros((int(n_permutations), F_gc.shape[1]), dtype=np.float64),
                where=valid_c[None, :],
            )
            Y_perm_sh = yc[perm_shared]
            study_null_Z = np.divide(
                (Y_perm_sh.astype(np.float32) @ F_gc.astype(np.float32)).astype(np.float64),
                sd_perm[None, :],
                out=np.zeros((b_study, F_gc.shape[1]), dtype=np.float64),
                where=valid_c[None, :],
            )
            Z_2d = Z_obs.reshape(n_reg, len(rbps))
            null_3d = null_Z.reshape(int(n_permutations), n_reg, len(rbps))
            study_null_3d = study_null_Z.reshape(b_study, n_reg, len(rbps))
            # One-sided positive cis-binding enrichment across all 12 spatial cis-regulatory channels!
            sc_2d = np.maximum(Z_2d, 0.0)
            null_sc_3d = np.maximum(null_3d, 0.0)
            study_null_sc_3d = np.maximum(study_null_3d, 0.0)
            arm_names.append(a_name)
            arm_Z_list.append(Z_2d)
            arm_score_list.append(sc_2d)
            arm_null_list.append(null_sc_3d)
            arm_study_null_list.append(study_null_sc_3d)

        score_stack = np.stack(arm_score_list, axis=0)
        Z_stack = np.stack(arm_Z_list, axis=0)
        null_stack = np.stack(arm_null_list, axis=1)
        study_null_stack = np.stack(arm_study_null_list, axis=1)

        rbp_max = np.max(score_stack, axis=(0, 1))
        null_rbp_max = np.max(null_stack, axis=(1, 2))
        glob_max = np.max(null_rbp_max, axis=1)
        study_mod_max = np.max(study_null_stack, axis=(1, 2, 3))
        study_mod_maxima.append(study_mod_max)

        nom_p = (1.0 + np.sum(null_rbp_max >= rbp_max[None, :] - 1e-6, axis=0)) / (n_permutations + 1.0)
        fwer_p = (1.0 + np.sum(glob_max[:, None] >= rbp_max[None, :] - 1e-6, axis=0)) / (n_permutations + 1.0)
        per_mod_cache.append((m_lbl, n_m_ev, n_m_genes, arm_names, score_stack, Z_stack, rbp_max, nom_p, fwer_p, study_mod_max))

    if study_mod_maxima:
        # Westfall-Young pivotal minP across modules so modules of heterogeneous size (e.g. 30 vs 1,178 events) are uniform on [0, 1]
        M_null = np.stack(study_mod_maxima, axis=1)  # (b_study, n_modules)
        P_null_mod = np.empty_like(M_null)
        for k_m in range(M_null.shape[1]):
            col_s = np.sort(M_null[:, k_m])
            ge_cnt = b_study - np.searchsorted(col_s, M_null[:, k_m] - 1e-9, side="left")
            # Leave-one-out count (ge_cnt - 1) puts P_null_mod on the exact same {1/(B+1)..1} grid as p_obs_mod
            P_null_mod[:, k_m] = (1.0 + np.maximum(ge_cnt - 1, 0)) / (b_study + 1.0)
        min_p_null = np.min(P_null_mod, axis=1)  # (b_study,)

        for m_lbl, n_m_ev, n_m_genes, arm_names, score_stack, Z_stack, rbp_max, nom_p, fwer_p, s_mod_max in per_mod_cache:
            p_obs_mod = (1.0 + np.sum(s_mod_max[:, None] >= rbp_max[None, :] - 1e-6, axis=0)) / (b_study + 1.0)
            study_fwer_p = (1.0 + np.sum(min_p_null[:, None] <= p_obs_mod[None, :] + 1e-12, axis=0)) / (b_study + 1.0)
            n_m_r = len(nom_p)
            o_nom = np.argsort(nom_p, kind="mergesort")
            q_nom = np.minimum(1.0, np.minimum.accumulate((nom_p[o_nom] * n_m_r / np.arange(1, n_m_r + 1))[::-1])[::-1])
            mod_fdr_bh = np.empty(n_m_r, dtype=float)
            mod_fdr_bh[o_nom] = np.clip(q_nom, 0.0, 1.0)
            order = np.argsort(-rbp_max)
            for rank_i, r_i in enumerate(order, 1):
                flat_best = int(np.argmax(score_stack[:, :, r_i])) if float(np.max(score_stack[:, :, r_i])) > 0.0 else int(np.argmax(Z_stack[:, :, r_i]))
                best_a, best_reg = divmod(flat_best, n_reg)
                out_rows.append({
                    "module": m_lbl,
                    "module_events": n_m_ev,
                    "module_genes": n_m_genes,
                    "RBP": rbps[r_i],
                    "best_region": _BRIDGE_REGION_NAMES[best_reg],
                    "kme_direction": arm_names[best_a],
                    "stat_Z": round(float(Z_stack[best_a, best_reg, r_i]), 4),
                    "nominal_P": round(float(nom_p[r_i]), 5),
                    "module_FDR_BH": round(float(mod_fdr_bh[r_i]), 5),
                    "module_P_FWER": round(float(fwer_p[r_i]), 5),
                    "study_P_FWER": round(float(study_fwer_p[r_i]), 5),
                    "sequence_rank": int(rank_i),
                })

    out_df = pd.DataFrame(out_rows)
    if rbp_expr_df is not None and not out_df.empty:
        out_df = annotate_rbp_expression_disambiguation(
            out_df,
            rbp_expr_df,
            fwer_col="module_P_FWER",
            z_col="stat_Z",
            group_col="module",
            module_scores_df=module_scores_df,
            min_rbp_expr=min_rbp_expr,
        )
    return out_df
