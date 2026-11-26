"""Self-contained SpliceTrace 1.0 small-sample 2v2 rescue engine (11-channel PPT + ESE/ESS + CpG FWL)."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

from splicetrace.library import CuratedMotifLibrary, load_bundled_human_rbp_library
from splicetrace.module_bridge import annotate_rbp_expression_disambiguation
from splicetrace.rna_forms import (
    compute_sequence_gc,
    event_forms,
    extract_event_rna,
    trace_motif_instances,
)
from splicetrace.fast_accel import (
    configure_server_threads,
    fast_gene_block_maxt_permutations,
    fast_gene_block_rademacher_maxt,
    fast_score_mean_log_odds,
)


@dataclass(frozen=True)
class SpliceTraceRescueResult:
    rbp_ranking: pd.DataFrame
    region_tests: pd.DataFrame
    event_contrasts: pd.DataFrame
    mature_consequences: pd.DataFrame
    diagnostics: dict


def _score_mean_log_odds(sequences: list[str], models: dict[str, dict]) -> np.ndarray:
    return fast_score_mean_log_odds(sequences, models)


def _pwm_ic_weight(matrix_obj: object) -> float:
    mat = np.clip(np.asarray(matrix_obj, dtype=float), 1e-6, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    ic_per_pos = 2.0 + np.sum(mat * np.log2(mat), axis=1)
    return float(max(0.25, np.sum(np.maximum(ic_per_pos, 0.0)) ** 1.5))


def build_directional_11channel_tensor(
    events_df: pd.DataFrame,
    rna_df: pd.DataFrame,
    library: CuratedMotifLibrary,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    models = library.models
    m_ids = list(models.keys())
    mid_to_idx = {mid: idx for idx, mid in enumerate(m_ids)}
    ic_weights = np.array([_pwm_ic_weight(models[m]["matrix"]) for m in m_ids], dtype=float)
    rbps = sorted(library.rbp_to_models.keys())
    rbp_to_mcols = {r: [mid_to_idx[m] for m in library.rbp_to_models[r] if m in mid_to_idx] for r in rbps}

    event_ids = events_df["event_id"].astype(str).to_numpy()
    ev_index = {eid: i for i, eid in enumerate(event_ids)}
    n_ev = len(event_ids)
    region_names = [
        "mature_consequence_delta",
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
    ]
    model_tensor = np.zeros((n_ev, len(region_names), len(m_ids)), dtype=float)

    mature = rna_df[rna_df["layer"].eq("mature_local_form") & rna_df["region"].isin(["inclusion", "exclusion"])].copy()
    if not mature.empty:
        m_scores = _score_mean_log_odds(mature["sequence"].astype(str).tolist(), models)
        inc_mat = np.zeros((n_ev, len(m_ids)), dtype=float)
        exc_mat = np.zeros((n_ev, len(m_ids)), dtype=float)
        ex_seqs, ex_targets = [], []
        for row_i, rec in enumerate(mature[["event_id", "region", "sequence", "junctions_json"]].itertuples(index=False)):
            ei = ev_index.get(str(rec.event_id))
            if ei is None:
                continue
            if rec.region == "inclusion":
                inc_mat[ei] = m_scores[row_i]
                seq_inc = str(rec.sequence)
                joins = json.loads(str(rec.junctions_json)) if rec.junctions_json else []
                if len(joins) >= 2:
                    j1, j2 = int(joins[0]), int(joins[-1])
                    for piece, sl_i in (
                        (seq_inc[max(0, j1 - 50) : j1], 9),
                        (seq_inc[j1 : min(j2, j1 + 50)], 10),
                        (seq_inc[max(j1, j2 - 50) : j2], 11),
                        (seq_inc[j2 : min(len(seq_inc), j2 + 50)], 12),
                    ):
                        if len(piece) >= 6:
                            ex_seqs.append(piece)
                            ex_targets.append((ei, sl_i))
                elif len(joins) == 1:
                    j1 = int(joins[0])
                    for piece, sl_i in ((seq_inc[max(0, j1 - 50) : j1], 9), (seq_inc[j1 : min(len(seq_inc), j1 + 50)], 12)):
                        if len(piece) >= 6:
                            ex_seqs.append(piece)
                            ex_targets.append((ei, sl_i))
            elif rec.region == "exclusion":
                exc_mat[ei] = m_scores[row_i]
        model_tensor[:, 0, :] = inc_mat - exc_mat
        if ex_seqs:
            ex_scores = _score_mean_log_odds(ex_seqs, models)
            for idx_e, (ei, sl_i) in enumerate(ex_targets):
                model_tensor[ei, sl_i, :] = ex_scores[idx_e]

    reg_df = rna_df[rna_df["layer"].eq("regulatory_pre_mRNA")].copy()
    if not reg_df.empty:
        etype_map = {str(eid): str(et) for eid, et in zip(events_df["event_id"], events_df["event_type"])}
        sub_seqs, sub_targets = [], []
        for rec in reg_df[["event_id", "region", "sequence"]].itertuples(index=False):
            ei = ev_index.get(str(rec.event_id))
            if ei is None:
                continue
            base_r = str(rec.region)
            seq = str(rec.sequence)
            if not seq:
                continue
            is_distal = base_r.endswith("_distal")
            stem_r = base_r[:-7] if is_distal else base_r
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
                if len(seq) >= 12:
                    sub_seqs.append(seq)
                    sub_targets.append((ei, 4 if is_up else 8, sgn))
            else:
                L = len(seq)
                for b_i, (d0, d1) in enumerate(((0, 35), (35, 85), (85, 200))):
                    piece = seq[max(0, L - d1) : max(0, L - d0)] if is_up else seq[min(L, d0) : min(L, d1)]
                    if len(piece) >= 6:
                        sub_seqs.append(piece)
                        sub_targets.append((ei, (1 if is_up else 5) + b_i, sgn))
        if sub_seqs:
            b_scores = _score_mean_log_odds(sub_seqs, models)
            for idx_s, (ei, slot, sgn) in enumerate(sub_targets):
                model_tensor[ei, slot, :] += sgn * b_scores[idx_s]

    rms_m = np.sqrt(np.mean(model_tensor * model_tensor, axis=0, keepdims=True))
    norm_m = np.divide(model_tensor, rms_m, out=np.zeros_like(model_tensor), where=rms_m > 1e-12)
    rbp_tensor = np.zeros((n_ev, len(region_names), len(rbps)), dtype=float)
    for r_i, r_name in enumerate(rbps):
        cols = rbp_to_mcols[r_name]
        if cols:
            w_ic = ic_weights[cols]
            w_ic = w_ic / w_ic.sum()
            rbp_tensor[:, :, r_i] = np.tensordot(norm_m[:, :, cols], w_ic, axes=([2], [0]))
    return rbp_tensor, np.asarray(rbps, dtype=str), region_names


def _fwl_residuals(Y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    w_root = np.sqrt(np.maximum(weights, 1e-12))
    Xw = X * w_root[:, None]
    if Y.ndim == 1:
        beta, _, _, _ = np.linalg.lstsq(Xw, Y * w_root, rcond=None)
        return Y - X @ beta
    beta, _, _, _ = np.linalg.lstsq(Xw, Y * w_root[:, None], rcond=None)
    return Y - X @ beta


def run_bulk_continuous_rescue(
    all_measurable_events: pd.DataFrame,
    fasta_path: Path | str,
    library: CuratedMotifLibrary | None = None,
    *,
    distal_intron_nt: int = 500,
    include_exon_gc: bool = False,
    use_fwl: bool = True,
    rbp_expr_df: pd.DataFrame | None = None,
    min_rbp_expr: float = 1.0,
    max_events_blind_depth: int = 3500,
    n_permutations: int = 4999,
    calibration: str = "gene_block_permutation",
    seed: int = 20261117,
    threads: int | None = None,
) -> SpliceTraceRescueResult:
    """Run SpliceTrace 1.0 multi-core calibrated rescue engine."""
    if calibration not in ("gene_block_permutation", "rademacher_hc"):
        raise ValueError(f"Unsupported calibration mode: {calibration!r}")
    configure_server_threads(threads)
    lib = load_bundled_human_rbp_library() if library is None else library
    df = all_measurable_events.copy()
    if "gene_symbol" not in df.columns:
        df["gene_symbol"] = df["gene_id"].astype(str)
    if "comparison_id" not in df.columns:
        df["comparison_id"] = "rescue_contrast"
    if "depth" not in df.columns:
        df["depth"] = 50.0

    def _strict_num_vec(
        ser: pd.Series, col_name: str, *, lo: float | None = None, hi: float | None = None, allow_na: bool = True,
    ) -> np.ndarray:
        out = np.full(len(ser), np.nan, dtype=float)
        blank_toks = {"", "na", "nan", "none", "null"}
        for r_i, raw in enumerate(ser.tolist()):
            if raw is None:
                if not allow_na:
                    raise ValueError(f"Missing required numeric value in '{col_name}' at row {r_i}.")
                continue
            if isinstance(raw, (int, float, np.number)) and not isinstance(raw, (bool, np.bool_)):
                fv = float(raw)
                if math.isnan(fv):
                    if not allow_na:
                        raise ValueError(f"Missing required numeric value in '{col_name}' at row {r_i}.")
                    continue
            else:
                if isinstance(raw, (bool, np.bool_)):
                    raise ValueError(f"Invalid boolean token {raw!r} in numeric column '{col_name}' (row {r_i}).")
                s = str(raw).strip()
                if s.lower() in blank_toks:
                    if not allow_na:
                        raise ValueError(f"Missing required numeric value in '{col_name}' at row {r_i}.")
                    continue
                try:
                    fv = float(s)
                except ValueError as exc:
                    raise ValueError(f"Non-numeric token {s!r} in column '{col_name}' (continuous rescue row {r_i}).") from exc
            if (not math.isfinite(fv)) or (lo is not None and fv < lo - 1e-9) or (hi is not None and fv > hi + 1e-9):
                rng_lbl = f"[{lo}, {hi}]" if (lo is not None and hi is not None) else "finite"
                raise ValueError(f"Out-of-range or non-finite {col_name} values in continuous rescue input; expected {rng_lbl}.")
            out[r_i] = fv
        return out

    import math

    # Strict domain & token validation before probability/PSI clipping
    df["dpsi"] = _strict_num_vec(df["dpsi"], "dpsi", lo=-1.0, hi=1.0, allow_na=False)
    df["baseline_PSI"] = _strict_num_vec(df["baseline_PSI"], "baseline_PSI", lo=0.0, hi=1.0, allow_na=True)
    df["depth"] = _strict_num_vec(df["depth"], "depth", lo=0.0, hi=None, allow_na=True)
    fdr_col_name = "fdr" if "fdr" in df.columns else ("FDR" if "FDR" in df.columns else None)
    if fdr_col_name is not None:
        df[fdr_col_name] = _strict_num_vec(df[fdr_col_name], fdr_col_name, lo=0.0, hi=1.0, allow_na=True)
    pv_col_name = next((c for c in ("PValue", "pvalue", "p_value") if c in df.columns), None)
    if pv_col_name is not None:
        df[pv_col_name] = _strict_num_vec(df[pv_col_name], pv_col_name, lo=0.0, hi=1.0, allow_na=True)

    p_ctrl = np.clip(df["baseline_PSI"].to_numpy(dtype=float), 0.0, 1.0)
    p_kd = np.clip(p_ctrl + df["dpsi"].to_numpy(dtype=float), 0.0, 1.0)
    non_const = np.isfinite(p_ctrl) & np.isfinite(p_kd) & (0.5 * (p_ctrl + p_kd) >= 0.03) & (0.5 * (p_ctrl + p_kd) <= 0.97)
    if int(non_const.sum()) >= 120:
        df = df.loc[non_const].reset_index(drop=True)

    fdr_all = pd.to_numeric(df[fdr_col_name] if fdr_col_name is not None else pd.Series(1.0, index=df.index), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    if pv_col_name is not None:
        pv_num = pd.to_numeric(df[pv_col_name], errors="coerce").to_numpy(dtype=float)
        pv_fin = np.isfinite(pv_num)
        pval_all = np.where(pv_fin, np.clip(pv_num, 1e-12, 1.0), np.nan)
    else:
        pv_fin = np.ones(len(df), dtype=bool)
        pval_all = np.clip(fdr_all, 1e-12, 1.0)
    dpsi_v = df["dpsi"].to_numpy(dtype=float)
    df["log_depth"] = np.log1p(np.maximum(df["depth"].to_numpy(dtype=float), 1.0))

    is_fg = pv_fin & (fdr_all <= 0.10) & (np.abs(dpsi_v) >= 0.05)
    if int(is_fg.sum()) < 35:
        is_fg = pv_fin & (pval_all <= 0.015) & (np.abs(dpsi_v) >= 0.05)
    is_bg = (fdr_all > 0.25) & (np.abs(dpsi_v) < 0.03)

    fg_df = df.loc[is_fg].copy()
    bg_pool = df.loc[is_bg & (~df["gene_id"].isin(set(fg_df["gene_id"])))].copy()
    has_prob_evidence = (
        bool(np.any(pv_fin & (pval_all < 1.0 - 1e-12)))
        if pv_col_name is not None
        else bool(np.any(np.isfinite(fdr_all) & (fdr_all < 1.0 - 1e-12)))
    )
    run_status = "complete"
    inference_valid = True
    if len(fg_df) >= 10 and len(bg_pool) >= 40:
        bg_parts = []
        for et, fg_et in fg_df.groupby("event_type"):
            bp_et = bg_pool.loc[bg_pool["event_type"].eq(et)].copy()
            if bp_et.empty:
                continue
            med_d, med_p = float(fg_et["log_depth"].median()), float(fg_et["baseline_PSI"].median())
            bp_et["_dist"] = ((bp_et["log_depth"] - med_d) / 0.85) ** 2 + ((bp_et["baseline_PSI"] - med_p) / 0.22) ** 2
            n_take = min(len(bp_et), max(250, 5 * len(fg_et)))
            bg_parts.append(bp_et.sort_values(["_dist", "event_id"], kind="mergesort").head(n_take).drop(columns=["_dist"]))
        bg_df = pd.concat(bg_parts, ignore_index=True) if bg_parts else bg_pool.head(800)
        panel_raw = pd.concat([fg_df, bg_df], ignore_index=True)
        panel_raw["_is_fg"] = [True] * len(fg_df) + [False] * len(bg_df)
    else:
        df["_fg_cand"] = is_fg
        df["_h"] = df["event_id"].astype(str).map(lambda s: hashlib.md5(s.encode()).hexdigest())
        panel_raw = df.sort_values(["depth", "_h"], ascending=[False, True]).head(int(max_events_blind_depth)).drop(columns=["_h"]).copy()
        if not has_prob_evidence:
            panel_raw["_is_fg"] = False
            run_status = "insufficient_probability_evidence"
            inference_valid = False
        elif not bool(panel_raw["_fg_cand"].any()):
            panel_raw["_is_fg"] = False
            run_status = "zero_foreground_events"
            inference_valid = False
        else:
            panel_raw["_is_fg"] = panel_raw["_fg_cand"].astype(bool)
            run_status = "complete"
            inference_valid = True
        panel_raw = panel_raw.drop(columns=["_fg_cand"])

    valid_mask = []
    for rec in panel_raw.to_dict("records"):
        try:
            event_forms(rec)
            valid_mask.append(True)
        except Exception:
            valid_mask.append(False)
    cont_df = panel_raw.loc[valid_mask].reset_index(drop=True)
    if cont_df.empty:
        return SpliceTraceRescueResult(
            rbp_ranking=pd.DataFrame(columns=["RBP", "best_region", "direction", "stat_Z", "nominal_P", "rbp_FDR_BH", "rbp_P_FWER", "sequence_rank"]),
            region_tests=pd.DataFrame(columns=["RBP", "region", "layer", "stat_Z", "nominal_P", "region_FDR_BH", "region_P_FWER"]),
            event_contrasts=cont_df,
            mature_consequences=pd.DataFrame(),
            diagnostics={"status": "no_valid_events", "n_events_total": 0, "n_events_fg": 0},
        )

    rna_df = extract_event_rna(
        cont_df,
        Path(fasta_path),
        intron_nt=200,
        distal_intron_nt=int(distal_intron_nt),
        acceptor_core_nt=3,
        feature_only=False,
        compute_sha256=False,
    )
    rbp_tensor, rbps, region_names = build_directional_11channel_tensor(cont_df, rna_df, lib)

    aff = rna_df[rna_df["layer"].eq("affected_sequence")]
    exon_gc_map, aff_len_map = {}, {}
    for eid, grp in aff.groupby("event_id", sort=False):
        joined = "".join(grp["sequence"].astype(str))
        exon_gc_map[str(eid)] = compute_sequence_gc(joined)[0]
        aff_len_map[str(eid)] = float(len(joined))

    ev_i = {str(e): i for i, e in enumerate(cont_df["event_id"])}
    n_c = len(cont_df)
    cpg_num = np.zeros(n_c, dtype=float)
    cpg_den = np.zeros(n_c, dtype=float)
    prox_gc_num = np.zeros(n_c, dtype=float)
    prox_gc_den = np.zeros(n_c, dtype=float)
    dist_gc_num = np.zeros(n_c, dtype=float)
    dist_gc_den = np.zeros(n_c, dtype=float)
    seen_reg_blks: set[tuple[int, str]] = set()
    has_blks = "blocks_json" in rna_df.columns
    r_cols = ["event_id", "layer", "region", "sequence", "blocks_json"] if has_blks else ["event_id", "layer", "region", "sequence"]
    for rec in rna_df[r_cols].itertuples(index=False):
        idx = ev_i.get(str(rec.event_id))
        if idx is None:
            continue
        s = str(rec.sequence).upper().replace("U", "T")
        cpg_num[idx] += s.count("CG")
        cpg_den[idx] += max(len(s) - 1, 1)
        if str(rec.layer) == "regulatory_pre_mRNA" and s:
            b_key = str(rec.blocks_json) if has_blks else str(rec.region)
            if (idx, b_key) in seen_reg_blks:
                continue
            seen_reg_blks.add((idx, b_key))
            tot_nt = s.count("A") + s.count("C") + s.count("G") + s.count("T")
            if tot_nt > 0:
                gc_cnt = s.count("G") + s.count("C")
                if str(rec.region).endswith("_distal"):
                    dist_gc_num[idx] += gc_cnt
                    dist_gc_den[idx] += tot_nt
                else:
                    prox_gc_num[idx] += gc_cnt
                    prox_gc_den[idx] += tot_nt
    cpg_rate = cpg_num / np.maximum(cpg_den, 1.0)

    exon_gc = np.array([exon_gc_map.get(str(x), 0.5) for x in cont_df["event_id"]], dtype=float)
    exon_gc = np.where(np.isfinite(exon_gc), exon_gc, 0.5)
    # Single intronic-GC nuisance covariate: distal intronic GC when available, else proximal intronic GC.
    prox_gc = np.divide(prox_gc_num, prox_gc_den, out=np.full(n_c, 0.5), where=prox_gc_den > 0)
    intronic_gc = np.divide(dist_gc_num, dist_gc_den, out=prox_gc.copy(), where=dist_gc_den > 0)
    log_len = np.array([float(np.log1p(aff_len_map.get(str(x), 100.0))) for x in cont_df["event_id"]], dtype=float)
    b_psi = np.clip(cont_df["baseline_PSI"].to_numpy(dtype=float), 0.02, 0.98)
    etypes = cont_df["event_type"].astype(str).to_numpy()
    uniq_et = sorted(set(etypes))

    nuis_cols = [np.ones(n_c, dtype=float)]
    if include_exon_gc:
        nuis_cols.append(exon_gc)
    nuis_cols.extend([
        intronic_gc,
        cpg_rate,
        log_len,
        b_psi,
        b_psi * (1.0 - b_psi),
        cont_df["log_depth"].to_numpy(dtype=float),
        *[(etypes == et).astype(float) for et in uniq_et[1:]],
    ])
    X_nuis = np.column_stack(nuis_cols)

    dpsi_c = cont_df["dpsi"].to_numpy(dtype=float)
    pv_c_ser = cont_df[pv_col_name] if pv_col_name is not None else (cont_df[fdr_col_name] if fdr_col_name is not None else pd.Series(1.0, index=cont_df.index))
    pv_c = np.clip(pd.to_numeric(pv_c_ser, errors="coerce").fillna(1.0).to_numpy(dtype=float), 1e-12, 1.0)
    z_raw = np.sign(dpsi_c) * np.clip(norm.isf(0.5 * pv_c), 0.0, 8.0)
    y_sparse = np.where(cont_df["_is_fg"].to_numpy(dtype=bool), z_raw, 0.0)
    cont_df["z_shift"] = y_sparse

    genes = cont_df["gene_id"].astype(str).to_numpy()
    ug, g_codes = np.unique(genes, return_inverse=True)
    G = len(ug)
    w = 1.0 / np.bincount(g_codes)[g_codes].astype(float)
    w /= w.sum()

    n_reg, n_rbp = len(region_names), len(rbps)
    F_flat = rbp_tensor.reshape(len(cont_df), n_reg * n_rbp)
    X_proj = X_nuis if use_fwl else np.ones((n_c, 1), dtype=float)
    F_res = _fwl_residuals(F_flat, X_proj, w)
    f_sd0 = np.sqrt(np.maximum(np.sum((F_res * F_res) * w[:, None], axis=0), 1e-12))
    F_win = _fwl_residuals(np.clip(F_res / f_sd0[None, :], -3.5, 3.5), X_proj, w)
    y_res = _fwl_residuals(y_sparse, X_proj, w)

    F_g = np.zeros((G, n_reg * n_rbp), dtype=float)
    np.add.at(F_g, g_codes, w[:, None] * F_win)
    y_g = np.zeros(G, dtype=float)
    np.add.at(y_g, g_codes, w * y_res)

    F_gc = F_g - F_g.mean(axis=0, keepdims=True)
    yc = y_g - y_g.mean()
    sd_perm = np.sqrt(np.sum(yc * yc) * np.sum(F_gc * F_gc, axis=0) / max(G - 1, 1))
    valid_c = sd_perm > 1e-12
    if calibration == "rademacher_hc":
        Z_obs_2d, rbp_nom_p, rbp_fwer_p, reg_nom_p, reg_fwer_p = fast_gene_block_rademacher_maxt(
            yc, F_gc, valid_c, n_reg, n_rbp, n_permutations=int(n_permutations), seed=int(seed)
        )
    else:
        Z_obs_2d, rbp_nom_p, rbp_fwer_p, reg_nom_p, reg_fwer_p = fast_gene_block_maxt_permutations(
            yc, F_gc, sd_perm, valid_c, n_reg, n_rbp, n_permutations=int(n_permutations), seed=int(seed)
        )
    def _bh_vec(p_arr: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p_arr, dtype=float), 0.0, 1.0)
        n = len(p)
        if n == 0:
            return p
        order = np.argsort(p, kind="mergesort")
        q = p[order] * n / np.arange(1, n + 1, dtype=float)
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = np.empty(n, dtype=float)
        out[order] = np.clip(q, 0.0, 1.0)
        return out

    reg_fdr_2d = _bh_vec(reg_nom_p.ravel()).reshape(n_reg, n_rbp)
    rbp_fdr_bh = _bh_vec(rbp_nom_p)
    reg_rows = [
        {
            "RBP": str(rbps[m_i]),
            "region": str(region_names[r_i]),
            "stat_Z": float(Z_obs_2d[r_i, m_i]),
            "nominal_P": float(reg_nom_p[r_i, m_i]),
            "region_FDR_BH": float(reg_fdr_2d[r_i, m_i]),
            "region_P_FWER": float(reg_fwer_p[r_i, m_i]),
        }
        for r_i in range(n_reg)
        for m_i in range(n_rbp)
    ]
    rbp_rows = []
    for m_i, r_name in enumerate(rbps):
        best_r = int(np.argmax(np.abs(Z_obs_2d[:, m_i])))
        zv = float(Z_obs_2d[best_r, m_i])
        rbp_rows.append({
            "RBP": str(r_name),
            "best_region": str(region_names[best_r]),
            "direction": "inclusion_promoted" if zv >= 0 else "skipping_promoted",
            "stat_Z": zv,
            "abs_Z": abs(zv),
            "nominal_P": float(rbp_nom_p[m_i]),
            "rbp_FDR_BH": float(rbp_fdr_bh[m_i]),
            "rbp_P_FWER": float(rbp_fwer_p[m_i]),
            "sequence_evidence_score": float(-norm.logsf(abs(zv)) / np.log(10.0)),
        })
    rbp_df = pd.DataFrame(rbp_rows).sort_values(
        by=["rbp_P_FWER", "rbp_FDR_BH", "sequence_evidence_score", "RBP"], ascending=[True, True, False, True], kind="mergesort"
    ).reset_index(drop=True)
    rbp_df["sequence_rank"] = np.arange(1, len(rbp_df) + 1, dtype=int)
    if rbp_expr_df is not None:
        rbp_df = annotate_rbp_expression_disambiguation(
            rbp_df,
            rbp_expr_df,
            fwer_col="rbp_P_FWER",
            z_col="stat_Z",
            min_rbp_expr=min_rbp_expr,
        )

    sig_rbps = rbp_df.loc[(rbp_df["rbp_P_FWER"] <= 0.05) | (rbp_df["rbp_FDR_BH"] <= 0.05), "RBP"].tolist()[:10]
    if not sig_rbps:
        sig_rbps = rbp_df.loc[rbp_df["nominal_P"] <= 0.01, "RBP"].tolist()[:5]
    conseq_df = pd.DataFrame()
    if sig_rbps:
        sig_m = {m: lib.models[m] for r in sig_rbps for m in lib.rbp_to_models.get(r, ()) if m in lib.models}
        fg_only = cont_df.loc[cont_df["_is_fg"]].head(200)
        _, conseq_df = trace_motif_instances(rna_df[rna_df["event_id"].isin(set(fg_only["event_id"]))], fg_only, sig_m)

    return SpliceTraceRescueResult(
        rbp_ranking=rbp_df,
        region_tests=pd.DataFrame(reg_rows),
        event_contrasts=cont_df,
        mature_consequences=conseq_df,
        diagnostics={
            "status": run_status,
            "inference_valid": inference_valid,
            "inference_mode": "case_control_cpg_rademacher_hc_maxT" if calibration == "rademacher_hc" else "case_control_cpg_gene_block_maxT",
            "calibration": calibration,
            "evaluated_events": len(cont_df),
            "foreground_events": int(cont_df["_is_fg"].sum()),
            "eligible_genes": int(G),
            "significant_rbps_fwer05": int((rbp_df["rbp_P_FWER"] <= 0.05).sum()) if inference_valid else 0,
            "significant_rbps_fdr05": int((rbp_df["rbp_FDR_BH"] <= 0.05).sum()) if inference_valid else 0,
            "significant_rbps_nom05": int((rbp_df["nominal_P"] <= 0.05).sum()) if inference_valid else 0,
        },
    )
