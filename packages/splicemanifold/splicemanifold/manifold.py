"""Publication-grade Splicing Landscape Algorithms & Executive Renderers:

1. Beyond-Jutils Precision-Shrunk PSI Z-Score Clustermap & PERMANOVA Confidence-Ellipse PCA:
   - Addresses the 4 methodological flaws of Jutils (Yang et al., Bioinformatics 2021):
     (a) Avoids complete-case `dropna()` truncation via coverage-filtered Empirical-Bayes prior shrinkage
         `PSI_tilde_{e,s} = (N_{e,s} * PSI_{e,s} + kappa * mu_e) / (N_{e,s} + kappa)`.
     (b) Selects features unsupervised via `ExcessBioVar = max(0, Var_obs(PSI) - E[PSI*(1-PSI)/(N+1)])`
         so low-depth binomial count noise is never mistaken for biological variance.
     (c) Computes precision-stabilized row Z-scores (`[-z_clip, +z_clip]`) with Ward hierarchical clustering,
         condition + coverage top strips, and event-type left strip.
     (d) Evaluates condition separation via permutation PERMANOVA pseudo-F / R^2 and 95% covariance ellipses
         without supervised pre-filtering leakage (while also offering an explicit Supervised Contrast Signature mode).

2. Precision-Kernel Event-PSI UMAP & Cell-Type Modality Divergence (Single-Cell / Large Cohort):
   - Computes pairwise cell/sample splicing distance exclusively across co-measured high-ExcessBioVar events
     weighted by joint Binomial Fisher precision `w_{ij,e} = N_{i,e} * N_{j,e} / (N_{i,e} + N_{j,e} + kappa)`
     (eliminating MARVEL / naive mean-imputation dropout blobs).
   - Aligns Expression UMAP and Precision Event-PSI UMAP via Procrustes rotation and computes per-cell
     Cross-Modality Neighborhood Divergence `Delta_kNN(i) = 1 - |N_k^expr(i) cap N_k^psi(i)| / k`,
     Cell-Type Silhouette, and Donor Inverse Simpson Index (LISI) batch-mixing score.
"""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2

EDITORIAL_NAVY = "#051C2C"
EDITORIAL_BLUE = "#2251FF"
EDITORIAL_CERULEAN = "#0077C8"
EDITORIAL_CORAL = "#B83A4B"
EDITORIAL_TEAL = "#0D9488"
EDITORIAL_AMBER = "#D97706"
EDITORIAL_VIOLET = "#4F46E5"
EDITORIAL_SLATE = "#64748B"

CONDITION_PALETTE = [
    "#051C2C", "#2251FF", "#C83E4D", "#0085A1",
    "#D97706", "#4F46E5", "#0F766E", "#475569",
]
# Deliberately distinct editorial jewel/slate palette so Event Subtypes NEVER share hex colors with Sample Conditions
EVENT_TYPE_COLORS = {
    "SE": "#64748B",
    "RI": "#8B5CF6",
    "A3SS": "#10B981",
    "A5SS": "#F59E0B",
    "MXE": "#EC4899",
    "AS": "#94A3B8",
}


def infer_event_type(eid: str) -> str:
    s = str(eid or "")
    for et in ("A3SS", "A5SS", "MXE", "RI", "SE"):
        if et in s:
            return et
    return "SE"


_BLANK_DONOR_TOKENS = {"", "nan", "none", "na", "null", "n/a"}


def _fmt_permanova_p(val: object) -> str:
    if val is None:
        return "P = N/A (descriptive)"
    try:
        fv = float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "P = N/A (descriptive)"
    if not math.isfinite(fv):
        return "P = N/A (descriptive)"
    return f"P = {fv:.3g}"


def _resolve_unit_donor(m: dict[str, Any], *, fallback_to_sample_id: bool = True) -> str:
    sid = str(m.get("sample_id", "")).strip()
    for k in ("donor_id", "biological_sample_id", "subject"):
        raw = m.get(k)
        if raw is None:
            continue
        s_val = str(raw).strip()
        if s_val and s_val.lower() not in _BLANK_DONOR_TOKENS:
            return s_val
    return sid if fallback_to_sample_id else ""


def _detect_paired_strata(sample_meta_ordered: list[dict[str, Any]]) -> tuple[list[str], int, bool]:
    """Identify paired/repeated-measures subjects across >=2 conditions (1 library per subject x condition)."""
    explicit_vals = [_resolve_unit_donor(m, fallback_to_sample_id=False) for m in sample_meta_ordered]
    if any(explicit_vals) and not all(explicit_vals):
        return [_resolve_unit_donor(m) for m in sample_meta_ordered], 0, False

    strata: list[str] = []
    donor_conds: dict[str, list[str]] = {}
    donor_sids: dict[str, list[str]] = {}
    cell_counts: dict[tuple[str, str], int] = {}
    for m in sample_meta_ordered:
        sid = str(m.get("sample_id", "")).strip()
        did = _resolve_unit_donor(m)
        cond = str(m.get("condition", "Unknown")).strip()
        strata.append(did)
        donor_conds.setdefault(did, []).append(cond)
        donor_sids.setdefault(did, []).append(sid)
        cell_counts[(did, cond)] = cell_counts.get((did, cond), 0) + 1

    paired_donors = [
        d for d, cs in donor_conds.items()
        if len(set(cs)) >= 2 and any(sd != d for sd in donor_sids[d])
    ]
    max_per_cell = max(cell_counts.values()) if cell_counts else 1
    if len(paired_donors) < 2 or max_per_cell > 1:
        return strata, 0, False

    paired_set = set(paired_donors)
    cond_sets = {frozenset(donor_conds[d]) for d in paired_donors}
    all_paired = all(d in paired_set for d in strata)
    is_fully_crossed = all_paired and len(cond_sets) == 1 and len(next(iter(cond_sets))) >= 2
    return strata, len(paired_donors), is_fully_crossed


def permanova_subject_blocked(
    dist_mat: np.ndarray,
    groups: list[str],
    strata: list[str],
    n_perm: int = 999,
    seed: int = 42,
) -> dict[str, Any]:
    """Blocked PERMANOVA controlling for subject baseline via within-subject restricted permutations."""
    n = len(groups)
    default_res: dict[str, Any] = {
        "subject_r2": 0.0,
        "within_subject_r2": 0.0,
        "p_within_subject": 1.0,
        "n_perms": 0,
        "exact": False,
    }
    if n < 4 or len(strata) != n or len(set(groups)) < 2:
        return default_res
    D2 = np.asarray(dist_mat, dtype=float) ** 2
    if D2.shape != (n, n) or not np.all(np.isfinite(D2)):
        return default_res
    np.fill_diagonal(D2, 0.0)
    H = np.eye(n) - np.ones((n, n)) / n
    G = -0.5 * (H @ D2 @ H)
    ss_total = float(np.trace(G))
    if ss_total <= 1e-12:
        return default_res

    uniq_strata = list(dict.fromkeys(str(s) for s in strata))
    s_to_i = {s: i for i, s in enumerate(uniq_strata)}
    Z_s = np.zeros((n, len(uniq_strata)), dtype=float)
    for r_i, s_v in enumerate(strata):
        Z_s[r_i, s_to_i[str(s_v)]] = 1.0
    X_s = H @ Z_s
    u_s, sv_s, _ = np.linalg.svd(X_s, full_matrices=False)
    Q_s = u_s[:, sv_s > 1e-10]
    df_s = int(Q_s.shape[1])
    ss_subject = max(0.0, float(np.sum(Q_s * (G @ Q_s)))) if df_s > 0 else 0.0
    ss_resid_after_subj = max(0.0, ss_total - ss_subject)
    subj_r2 = min(1.0, max(0.0, ss_subject / max(1e-12, ss_total)))
    if ss_resid_after_subj <= 1e-12:
        return {
            "subject_r2": round(subj_r2, 4),
            "within_subject_r2": 0.0,
            "p_within_subject": 1.0,
            "n_perms": 0,
            "exact": False,
        }

    uniq_g = list(dict.fromkeys(str(g) for g in groups))
    g_to_i = {g: i for i, g in enumerate(uniq_g)}
    k_g = len(uniq_g)
    obs_labels = [str(g) for g in groups]

    def _within_stat(labels: list[str]) -> tuple[float, float]:
        Z_c = np.zeros((n, k_g), dtype=float)
        for r_i, lb in enumerate(labels):
            Z_c[r_i, g_to_i[lb]] = 1.0
        X_c = H @ Z_c
        X_cs = X_c - Q_s @ (Q_s.T @ X_c) if df_s > 0 else X_c
        u_c, sv_c, _ = np.linalg.svd(X_cs, full_matrices=False)
        Q_cs = u_c[:, sv_c > 1e-10]
        df_c = int(Q_cs.shape[1])
        if df_c == 0:
            return 0.0, 0.0
        ss_cs = max(0.0, float(np.sum(Q_cs * (G @ Q_cs))))
        ws_r2 = min(1.0, ss_cs / max(1e-12, ss_resid_after_subj))
        df_resid = n - 1 - df_s - df_c
        if df_resid <= 0:
            return ws_r2, ss_cs
        ss_err = max(1e-12, ss_resid_after_subj - ss_cs)
        pseudo_f = (ss_cs / df_c) / (ss_err / df_resid)
        return ws_r2, pseudo_f

    ws_r2_obs, f_obs = _within_stat(obs_labels)

    block_indices: list[list[int]] = []
    block_perm_choices: list[list[tuple[str, ...]]] = []
    total_perms = 1
    exact_feasible = True
    for st in uniq_strata:
        idxs = [i for i, sv in enumerate(strata) if str(sv) == st]
        block_indices.append(idxs)
        if not exact_feasible:
            continue
        blk_lbls = [obs_labels[i] for i in idxs]
        if len(set(blk_lbls)) <= 1:
            block_perm_choices.append([tuple(blk_lbls)])
            continue
        if len(idxs) > 6 or total_perms * math.factorial(len(idxs)) > 16384:
            exact_feasible = False
            continue
        uniq_p = sorted(set(itertools.permutations(blk_lbls)))
        total_perms *= len(uniq_p)
        if total_perms > 2048:
            exact_feasible = False
        else:
            block_perm_choices.append(uniq_p)

    if exact_feasible and total_perms >= 2:
        ge_count = 0
        for combo in itertools.product(*block_perm_choices):
            perm_lbls = list(obs_labels)
            for idxs, p_tup in zip(block_indices, combo):
                for pos, lb in zip(idxs, p_tup):
                    perm_lbls[pos] = lb
            _, f_p = _within_stat(perm_lbls)
            if f_p >= f_obs - 1e-12:
                ge_count += 1
        return {
            "subject_r2": round(subj_r2, 4),
            "within_subject_r2": round(ws_r2_obs, 4),
            "p_within_subject": round(ge_count / total_perms, 4),
            "n_perms": int(total_perms),
            "exact": True,
        }

    rng = np.random.default_rng(seed)
    ge_count = 1
    for _ in range(n_perm):
        perm_lbls = list(obs_labels)
        for idxs in block_indices:
            if len(idxs) > 1:
                shuffled = rng.permutation([obs_labels[i] for i in idxs])
                for pos, lb in zip(idxs, shuffled):
                    perm_lbls[pos] = str(lb)
        _, f_p = _within_stat(perm_lbls)
        if f_p >= f_obs - 1e-12:
            ge_count += 1
    return {
        "subject_r2": round(subj_r2, 4),
        "within_subject_r2": round(ws_r2_obs, 4),
        "p_within_subject": round(ge_count / (n_perm + 1), 4),
        "n_perms": int(n_perm + 1),
        "exact": False,
    }


def permanova_r2(
    dist_mat: np.ndarray,
    groups: list[str],
    n_perm: int = 499,
    seed: int = 42,
    *,
    units: list[str] | None = None,
    return_details: bool = False,
) -> tuple[Any, ...]:
    """Non-parametric Anderson (2001) PERMANOVA R^2 with biological-unit-aware permutation."""
    n = len(groups)
    if n < 4 or len(set(groups)) < 2:
        return (0.0, None, "insufficient_samples", False) if return_details else (0.0, 1.0)
    D2 = np.asarray(dist_mat, dtype=float) ** 2
    if D2.shape != (n, n) or not np.all(np.isfinite(D2)):
        return (0.0, None, "invalid_distance_matrix", False) if return_details else (0.0, 1.0)
    np.fill_diagonal(D2, 0.0)
    ss_t = float(D2.sum()) / (2.0 * n)
    if ss_t <= 1e-12:
        return (0.0, None, "zero_total_dispersion", False) if return_details else (0.0, 1.0)

    uniq = sorted(set(str(g) for g in groups))
    g_arr = np.asarray([str(g) for g in groups], dtype=object)

    def _ss_within(labels: np.ndarray) -> float:
        ss_w = 0.0
        for u in uniq:
            idx = np.where(labels == u)[0]
            if len(idx) > 1:
                sub_d2 = D2[np.ix_(idx, idx)]
                ss_w += float(sub_d2.sum()) / (2.0 * len(idx))
        return ss_w

    ss_w_obs = _ss_within(g_arr)
    ss_b_obs = max(0.0, ss_t - ss_w_obs)
    r2_obs = ss_b_obs / max(1e-12, ss_t)
    r2_round = round(r2_obs, 4)

    clean_units: list[str] | None = None
    if units is not None and len(units) == n:
        clean_units = [("" if str(u).strip().lower() in _BLANK_DONOR_TOKENS else str(u).strip()) for u in units]

    if clean_units is not None and any(clean_units) and not all(clean_units):
        return (r2_round, None, "incomplete_biological_units", False) if return_details else (r2_round, math.nan)

    if clean_units is not None and all(clean_units) and len(set(clean_units)) < n:
        unit_idx: dict[str, np.ndarray] = {}
        unit_conds: dict[str, set[str]] = {}
        for u in dict.fromkeys(clean_units):
            idxs = np.array([i for i, cu in enumerate(clean_units) if cu == u], dtype=int)
            unit_idx[u] = idxs
            unit_conds[u] = set(g_arr[idxs].tolist())
        if all(len(cs) == 1 for cs in unit_conds.values()):
            u_names = list(unit_idx.keys())
            u_labels = [next(iter(unit_conds[u])) for u in u_names]
            if len(u_names) < 4 or len(set(u_labels)) < 2:
                return (r2_round, None, "insufficient_biological_units", False) if return_details else (r2_round, math.nan)
            cond_counts: dict[str, int] = {}
            for lbl in u_labels:
                cond_counts[lbl] = cond_counts.get(lbl, 0) + 1
            if min(cond_counts.values()) < 2:
                return (r2_round, None, "insufficient_biological_units", False) if return_details else (r2_round, math.nan)
            total_comb = math.factorial(len(u_labels))
            for cnt in cond_counts.values():
                total_comb //= math.factorial(cnt)
            if len(cond_counts) == 2 and total_comb <= 2000:
                g0, g1 = sorted(cond_counts.keys())
                n_g0 = cond_counts[g0]
                ge_count = 0
                for comb in itertools.combinations(range(len(u_names)), n_g0):
                    comb_set = set(comb)
                    perm = np.empty(n, dtype=object)
                    for ui, u in enumerate(u_names):
                        perm[unit_idx[u]] = g0 if ui in comb_set else g1
                    ss_b_p = max(0.0, ss_t - _ss_within(perm))
                    if ss_b_p >= ss_b_obs - 1e-12:
                        ge_count += 1
                p_val = round(ge_count / total_comb, 6)
            else:
                rng = np.random.default_rng(seed)
                u_lbl_arr = np.asarray(u_labels, dtype=object)
                ge_count = 1
                for _ in range(n_perm):
                    perm_u = rng.permutation(u_lbl_arr)
                    perm = np.empty(n, dtype=object)
                    for ui, u in enumerate(u_names):
                        perm[unit_idx[u]] = perm_u[ui]
                    ss_b_p = max(0.0, ss_t - _ss_within(perm))
                    if ss_b_p >= ss_b_obs - 1e-12:
                        ge_count += 1
                p_val = round(ge_count / (n_perm + 1), 4)
            return (r2_round, p_val, "between_biological_unit", True) if return_details else (r2_round, p_val)
        multi_cond_units = [u for u, cs in unit_conds.items() if len(cs) >= 2]
        if len(multi_cond_units) >= 2 and len(multi_cond_units) == len(unit_conds):
            cond_frozensets = {frozenset(cs) for cs in unit_conds.values()}
            if len(cond_frozensets) != 1 or next(iter(cond_frozensets)) != frozenset(uniq):
                return (r2_round, None, "split_plot_within_subject_only", False) if return_details else (r2_round, math.nan)
            # Verify balanced technical-replicate block sizes per (unit, condition) so nested library clones move as whole subsamples
            unit_cond_blocks: dict[str, dict[str, np.ndarray]] = {}
            for u in multi_cond_units:
                idxs = unit_idx[u]
                by_c = {c: idxs[g_arr[idxs] == c] for c in uniq}
                counts_u = {len(v) for v in by_c.values()}
                if len(counts_u) != 1 or next(iter(counts_u)) < 1:
                    return (r2_round, None, "unsupported_unbalanced_repeated_measures", False) if return_details else (r2_round, math.nan)
                unit_cond_blocks[u] = by_c
            if len(uniq) == 2 and (2 ** len(multi_cond_units)) <= 2048:
                c0, c1 = uniq[0], uniq[1]
                total_flips = 2 ** len(multi_cond_units)
                ge_count = 0
                for flips in itertools.product((False, True), repeat=len(multi_cond_units)):
                    perm = np.empty(n, dtype=object)
                    for u, flip in zip(multi_cond_units, flips):
                        perm[unit_cond_blocks[u][c0]] = c1 if flip else c0
                        perm[unit_cond_blocks[u][c1]] = c0 if flip else c1
                    ss_b_p = max(0.0, ss_t - _ss_within(perm))
                    if ss_b_p >= ss_b_obs - 1e-12:
                        ge_count += 1
                p_val = round(ge_count / total_flips, 6)
            else:
                rng = np.random.default_rng(seed)
                ge_count = 1
                for _ in range(n_perm):
                    perm = np.empty(n, dtype=object)
                    for u in multi_cond_units:
                        shuffled_c = rng.permutation(uniq)
                        for orig_c, new_c in zip(uniq, shuffled_c):
                            perm[unit_cond_blocks[u][orig_c]] = new_c
                    ss_b_p = max(0.0, ss_t - _ss_within(perm))
                    if ss_b_p >= ss_b_obs - 1e-12:
                        ge_count += 1
                p_val = round(ge_count / (n_perm + 1), 4)
            return (r2_round, p_val, "within_biological_unit", True) if return_details else (r2_round, p_val)
        return (r2_round, None, "unsupported_mixed_unit_design", False) if return_details else (r2_round, math.nan)

    rng = np.random.default_rng(seed)
    ge_count = 1
    for _ in range(n_perm):
        perm = rng.permutation(g_arr)
        ss_b_p = max(0.0, ss_t - _ss_within(perm))
        if ss_b_p >= ss_b_obs - 1e-12:
            ge_count += 1
    p_val = round(ge_count / (n_perm + 1), 4)
    return (r2_round, p_val, "independent_samples", True) if return_details else (r2_round, p_val)


def compute_splicing_manifold(
    psi_matrix: pd.DataFrame,
    sample_meta: list[dict[str, Any]],
    *,
    depth_matrix: pd.DataFrame | None = None,
    event_meta: dict[str, dict[str, Any]] | None = None,
    supervised_event_ids: set[str] | None = None,
    top_n: int = 150,
    manifold_top_n: int = 2000,
    min_present_fraction: float = 0.65,
    prior_kappa: float = 10.0,
    z_clip: float = 2.5,
    metric: str = "euclidean",
    event_types: list[str] | None = None,
    run_umap: bool = True,
    umap_neighbors: int = 12,
    umap_min_dist: float = 0.18,
    allow_coverage_fallback: bool = False,
) -> dict[str, Any]:
    """Compute Empirical-Bayes precision-shrunk PSI Z-scores, hierarchical clustermap, PCA, and Precision-Kernel UMAP."""
    s_ids = [str(s["sample_id"]) for s in sample_meta if str(s["sample_id"]) in psi_matrix.columns]
    if len(s_ids) < 3 or psi_matrix.empty:
        return {"available": False, "reason": "At least 3 measured samples/cells are required for landscape analysis."}

    meta_by_id = {str(s["sample_id"]): s for s in sample_meta}
    P_raw = psi_matrix[s_ids].apply(pd.to_numeric, errors="coerce")
    P_raw.index = psi_matrix.index.astype(str)

    if event_types:
        allowed_et = {e.upper() for e in event_types}
        et_series = pd.Series([
            str((event_meta or {}).get(eid, {}).get("event_type") or infer_event_type(eid)).upper()
            for eid in P_raw.index
        ], index=P_raw.index)
        P_raw = P_raw.loc[et_series.isin(allowed_et)]

    # Align junction read depth matrix N_{e,s} BEFORE coverage filtering so zero-support or empty depth tables never inflate coverage
    if depth_matrix is not None:
        d_aligned = depth_matrix.copy()
        d_aligned.index = d_aligned.index.astype(str)
        d_aligned.columns = d_aligned.columns.astype(str)
        N_raw_all = np.nan_to_num(
            d_aligned.reindex(index=P_raw.index, columns=s_ids).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float),
            nan=0.0,
        )
        P_raw_np = P_raw.to_numpy(dtype=float)
        usable_raw = np.isfinite(P_raw_np) & np.isfinite(N_raw_all) & (N_raw_all > 0.0)
    else:
        N_raw_all = np.where(P_raw.notna().to_numpy(), 25.0, 0.0)
        usable_raw = np.isfinite(P_raw.to_numpy(dtype=float))

    present_frac = usable_raw.mean(axis=1)
    keep_mask = present_frac >= float(min_present_fraction)
    eff_min_present = float(min_present_fraction)
    coverage_fallback_applied = False
    if int(keep_mask.sum()) < (5 if allow_coverage_fallback else 3):
        if allow_coverage_fallback:
            keep_mask = present_frac >= 0.25
            eff_min_present = 0.25
            coverage_fallback_applied = True
        else:
            return {
                "available": False,
                "reason": (
                    f"Only {int(keep_mask.sum())} splicing events met min_present_fraction={eff_min_present:.2f} "
                    "with positive read support."
                ),
                "effective_min_present_fraction": eff_min_present,
                "coverage_fallback_applied": False,
            }
    if int(keep_mask.sum()) < 3:
        return {
            "available": False,
            "reason": "Insufficient multi-sample splicing events after coverage filtering.",
            "effective_min_present_fraction": eff_min_present,
            "coverage_fallback_applied": coverage_fallback_applied,
        }
    P_obs = P_raw.loc[keep_mask].copy()
    N_mat = N_raw_all[keep_mask, :]

    P_np = P_obs.to_numpy(dtype=float)
    obs_mask = np.isfinite(P_np) & np.isfinite(N_mat) & (N_mat > 0)

    # Weighted event prior mean mu_e
    w_sum = np.where(obs_mask, N_mat, 0.0).sum(axis=1, keepdims=True)
    mu_num = np.where(obs_mask, N_mat * np.nan_to_num(P_np, nan=0.0), 0.0).sum(axis=1, keepdims=True)
    mu_e = np.divide(mu_num, w_sum, out=np.full((P_np.shape[0], 1), 0.5, dtype=float), where=w_sum > 0.0)
    mu_e = np.clip(mu_e, 0.01, 0.99)

    # Empirical-Bayes precision-shrunk PSI_tilde
    P_shrunk = np.where(
        obs_mask,
        (N_mat * np.nan_to_num(P_np, nan=0.0) + prior_kappa * mu_e) / (N_mat + prior_kappa),
        mu_e,
    )

    # ExcessBioVar = max(0, Var_obs(ddof=1) - E[mu*(1-mu)/(N+1)])
    n_obs_row = obs_mask.sum(axis=1)
    P_masked = np.where(obs_mask, P_np, np.nan)
    obs_var = np.zeros(P_np.shape[0], dtype=float)
    ge2 = n_obs_row >= 2
    if np.any(ge2):
        obs_var[ge2] = np.nanvar(P_masked[ge2], axis=1, ddof=1)
    binom_noise = np.zeros(P_np.shape[0], dtype=float)
    ge1 = n_obs_row >= 1
    if np.any(ge1):
        binom_noise[ge1] = np.nanmean(
            np.where(obs_mask[ge1], (mu_e[ge1] * (1.0 - mu_e[ge1])) / np.maximum(1.0, N_mat[ge1] + 1.0), np.nan),
            axis=1,
        )
    excess_biovar = np.maximum(0.0, np.nan_to_num(obs_var - binom_noise, nan=0.0))

    eids_all = P_obs.index.to_numpy()
    man_cap = max(int(top_n), int(manifold_top_n))
    if supervised_event_ids:
        sup_mask = np.isin(eids_all, list(supervised_event_ids))
        if sup_mask.sum() >= 5:
            cand_idx = np.where(sup_mask)[0]
            man_idx = cand_idx[np.argsort(-excess_biovar[cand_idx])[: max(5, man_cap)]]
            hm_idx = man_idx[: max(5, int(top_n))]
            selection_mode = "supervised_ps_das"
        else:
            man_idx = np.argsort(-excess_biovar)[: max(5, man_cap)]
            hm_idx = man_idx[: max(5, int(top_n))]
            selection_mode = "unsupervised_excess_biovar"
    else:
        man_idx = np.argsort(-excess_biovar)[: max(5, man_cap)]
        hm_idx = man_idx[: max(5, int(top_n))]
        selection_mode = "unsupervised_excess_biovar"

    # Global Manifold (PCA, PERMANOVA & Precision-Kernel UMAP) uses `man_idx` (e.g. 2,000 HVEs or all PS-DAS events!)
    P_man = P_shrunk[man_idx, :]
    N_man = N_mat[man_idx, :]
    mask_man = obs_mask[man_idx, :]
    n_s = len(s_ids)
    if int(mask_man.sum()) == 0:
        return {
            "available": False,
            "reason": "Selected manifold events have zero positive-support observations.",
            "effective_min_present_fraction": eff_min_present,
            "coverage_fallback_applied": coverage_fallback_applied,
        }
    m_int = mask_man.astype(int)
    shared_mat = m_int.T @ m_int
    tri_i, tri_j = np.triu_indices(n_s, k=1)
    pair_shared = shared_mat[tri_i, tri_j] if len(tri_i) > 0 else np.array([0], dtype=int)
    min_shared = int(pair_shared.min()) if len(pair_shared) > 0 else 0
    med_shared = float(np.median(pair_shared)) if len(pair_shared) > 0 else 0.0
    zero_shared_pairs = int((pair_shared == 0).sum())
    if len(pair_shared) > 0 and int(pair_shared.max()) == 0:
        return {
            "available": False,
            "reason": "No sample pairs share any co-measured positive-support splicing events.",
            "effective_min_present_fraction": eff_min_present,
            "coverage_fallback_applied": coverage_fallback_applied,
            "min_shared_events_per_pair": min_shared,
            "median_shared_events_per_pair": med_shared,
            "zero_shared_pairs": zero_shared_pairs,
        }

    # Heatmap Display uses `hm_idx` (top `top_n` events for readable row resolution)
    P_sel = P_shrunk[hm_idx, :]
    N_sel = N_mat[hm_idx, :]
    mask_sel = obs_mask[hm_idx, :]
    sel_eids = eids_all[hm_idx].tolist()

    # Row-wise precision-stabilized Z-score for Heatmap & Sample Dendrogram
    man_mean = P_man.mean(axis=1, keepdims=True)
    man_sd = np.maximum(0.025, P_man.std(axis=1, keepdims=True))
    Z_man = np.clip((P_man - man_mean) / man_sd, -float(z_clip), float(z_clip))

    row_mean = P_sel.mean(axis=1, keepdims=True)
    row_sd = np.maximum(0.025, P_sel.std(axis=1, keepdims=True))
    Z_mat = np.clip((P_sel - row_mean) / row_sd, -float(z_clip), float(z_clip))

    # Hierarchical clustering of samples (columns on global manifold) and heatmap events (rows)
    dist_metric = metric if metric in ("euclidean", "cosine", "correlation") else "euclidean"
    try:
        col_dist_vec = pdist(Z_man.T, metric=dist_metric)
        col_dist_vec = np.nan_to_num(col_dist_vec, nan=0.0)
        col_link = linkage(col_dist_vec, method="ward" if dist_metric == "euclidean" else "average")
        col_order = leaves_list(col_link).tolist()
    except Exception:
        col_link = None
        col_order = list(range(len(s_ids)))

    try:
        row_dist_vec = pdist(Z_mat, metric=dist_metric)
        row_dist_vec = np.nan_to_num(row_dist_vec, nan=0.0)
        row_link = linkage(row_dist_vec, method="ward" if dist_metric == "euclidean" else "average")
        row_order = leaves_list(row_link).tolist()
    except Exception:
        row_link = None
        row_order = list(range(len(sel_eids)))

    # PCA on centered shrunk global PSI (samples x manifold_events)
    X_pca = P_man.T - P_man.T.mean(axis=0, keepdims=True)
    u_s, s_vals, _ = np.linalg.svd(X_pca, full_matrices=False)
    coords_pca = u_s * s_vals
    var_tot = float((s_vals ** 2).sum()) or 1.0
    var_exp = [round(float((sv ** 2) / var_tot * 100.0), 2) for sv in s_vals[:5]]
    while len(var_exp) < 3:
        var_exp.append(0.0)

    # Precision-Kernel Pairwise Distance Matrix D_PSI on global manifold events
    D_prec = np.zeros((n_s, n_s), dtype=float)
    W_eff = np.where(mask_man, N_man, 0.0)  # (E_man, S)
    for i in range(n_s):
        wi = W_eff[:, i : i + 1]
        wj = W_eff[:, i:]
        joint_w = (wi * wj) / np.maximum(1.0, wi + wj + prior_kappa)
        diff2 = (P_man[:, i : i + 1] - P_man[:, i:]) ** 2
        denom = joint_w.sum(axis=0)
        d_row = np.sqrt(np.divide((joint_w * diff2).sum(axis=0), np.maximum(1e-8, denom)))
        fallback = np.sqrt(np.mean((P_man[:, i : i + 1] - P_man[:, i:]) ** 2, axis=0))
        d_final = np.where(denom > 1e-6, d_row, fallback)
        D_prec[i, i:] = d_final
        D_prec[i:, i] = d_final
    np.fill_diagonal(D_prec, 0.0)

    conditions = [str(meta_by_id[sid].get("condition", "Unknown")) for sid in s_ids]
    ordered_meta = [meta_by_id[sid] for sid in s_ids]
    units_raw = [_resolve_unit_donor(meta_by_id[sid], fallback_to_sample_id=False) for sid in s_ids]
    units_arg = units_raw if any(units_raw) else None
    strata, n_paired_subjects, is_fully_crossed_paired = _detect_paired_strata(ordered_meta)

    permanova_strata = "free"
    permanova_subject_r2: float | None = None
    permanova_within_subject_r2: float | None = None
    permanova_p_within_subject: float | None = None
    permanova_exact_perms: int | None = None

    perm_r2_raw, perm_p_raw, perm_scope_raw, perm_valid_raw = permanova_r2(
        D_prec, conditions, units=units_arg, return_details=True
    )
    perm_r2 = perm_r2_raw
    perm_p = perm_p_raw
    perm_scope = perm_scope_raw
    perm_valid = bool(perm_valid_raw)
    perm_p_unpaired = perm_p_raw

    if n_paired_subjects >= 2:
        blk = permanova_subject_blocked(D_prec, conditions, strata)
        permanova_subject_r2 = blk["subject_r2"]
        permanova_within_subject_r2 = blk["within_subject_r2"]
        permanova_p_within_subject = blk["p_within_subject"]
        permanova_exact_perms = blk["n_perms"]
        if is_fully_crossed_paired:
            permanova_strata = "subject_paired"
            perm_p = float(blk["p_within_subject"])
            perm_valid = True
        else:
            permanova_strata = "split_plot_paired"

    if zero_shared_pairs > 0:
        perm_p = None
        permanova_p_within_subject = None
        perm_scope = "non_inferential_zero_shared_pairs"
        perm_valid = False
        perm_warning = f"PERMANOVA P suppressed ({zero_shared_pairs} sample pair(s) share zero positive-support events)."
    elif selection_mode == "supervised_ps_das":
        perm_p = None
        permanova_p_within_subject = None
        perm_scope = "descriptive_only_post_selection_supervised"
        perm_valid = False
        perm_warning = "Supervised PS-DAS selection conditions on group labels; inferential PERMANOVA P is suppressed."
    else:
        perm_warning = "" if perm_valid else f"PERMANOVA P suppressed ({perm_scope})."

    # Precision-Kernel UMAP (if requested and n_samples >= 6); initialize with 2D PCA fallback
    umap_coords = np.zeros((n_s, 2), dtype=float)
    if coords_pca.shape[1] >= 1:
        umap_coords[:, 0] = coords_pca[:, 0]
    if coords_pca.shape[1] >= 2:
        umap_coords[:, 1] = coords_pca[:, 1]
    umap_method = "PCA_2D_fallback"
    if run_umap and n_s >= 6:
        try:
            import umap
            nn = max(3, min(int(umap_neighbors), n_s - 1))
            reducer = umap.UMAP(
                n_neighbors=nn,
                min_dist=float(umap_min_dist),
                metric="precomputed",
                random_state=42,
            )
            cand_umap = np.asarray(reducer.fit_transform(D_prec), dtype=float)
            if cand_umap.shape == (n_s, 2) and np.all(np.isfinite(cand_umap)):
                umap_coords = cand_umap
                umap_method = f"Precision-Kernel UMAP (n_neighbors={nn}, min_dist={umap_min_dist})"
        except Exception:
            pass

    # Sample records
    sample_out = []
    mean_cov_per_sample = mask_man.mean(axis=0)
    for idx, sid in enumerate(s_ids):
        m = meta_by_id[sid]
        sample_out.append({
            "sample_id": sid,
            "condition": str(m.get("condition", "")),
            "donor_id": _resolve_unit_donor(m, fallback_to_sample_id=False),
            "cell_type": str(m.get("cell_type", "")),
            "observed_event_fraction": round(float(mean_cov_per_sample[idx]), 4),
            "PC1": round(float(coords_pca[idx, 0]), 4),
            "PC2": round(float(coords_pca[idx, 1]) if coords_pca.shape[1] > 1 else 0.0, 4),
            "PC3": round(float(coords_pca[idx, 2]) if coords_pca.shape[1] > 2 else 0.0, 4),
            "psi_UMAP1": round(float(umap_coords[idx, 0]), 4),
            "psi_UMAP2": round(float(umap_coords[idx, 1]), 4),
        })

    # Ordered heatmap payload (capped to 160 rows in JSON for instant SVG rendering; full Z_mat used for PNG)
    ordered_samples = [s_ids[i] for i in col_order]
    ordered_conditions = [conditions[i] for i in col_order]
    ordered_rows = []
    for r_i in row_order[:160]:
        eid = sel_eids[r_i]
        em = (event_meta or {}).get(eid, {})
        ordered_rows.append({
            "event_id": eid,
            "gene_symbol": str(em.get("gene_symbol") or eid.split(":")[0]),
            "event_type": str(em.get("event_type") or infer_event_type(eid)),
            "excess_biovar": round(float(excess_biovar[hm_idx[r_i]]), 5),
            "z_values": [round(float(Z_mat[r_i, c_i]), 3) for c_i in col_order],
            "psi_values": [round(float(P_sel[r_i, c_i]), 3) for c_i in col_order],
        })

    return {
        "available": True,
        "display_available": True,
        "inference_available": bool(perm_valid),
        "selection_mode": selection_mode,
        "metric": dist_metric,
        "prior_kappa": prior_kappa,
        "z_clip": z_clip,
        "umap_method": umap_method,
        "total_candidate_events": int(len(P_obs)),
        "selected_events": int(len(man_idx)),
        "manifold_events": int(len(man_idx)),
        "heatmap_events": int(len(sel_eids)),
        "permanova_r2": perm_r2,
        "permanova_p": perm_p,
        "permanova_p_unpaired": perm_p_unpaired,
        "paired_subjects": int(n_paired_subjects),
        "permanova_strata": permanova_strata,
        "permanova_subject_r2": permanova_subject_r2,
        "permanova_within_subject_r2": permanova_within_subject_r2,
        "permanova_p_within_subject": permanova_p_within_subject,
        "permanova_exact_perms": permanova_exact_perms,
        "permanova_scope": perm_scope,
        "permanova_inference_valid": perm_valid,
        "permanova_warning": perm_warning,
        "effective_min_present_fraction": eff_min_present,
        "coverage_fallback_applied": coverage_fallback_applied,
        "min_shared_events_per_pair": min_shared,
        "median_shared_events_per_pair": med_shared,
        "zero_shared_pairs": zero_shared_pairs,
        "pca_variance_pct": var_exp,
        "samples": sample_out,
        "ordered_samples": ordered_samples,
        "ordered_conditions": ordered_conditions,
        "heatmap_rows": ordered_rows,
        "_internal": {
            "Z_ordered": Z_mat[np.ix_(row_order, col_order)],
            "col_link": col_link,
            "col_order": col_order,
            "row_order": row_order,
            "sel_eids": [sel_eids[i] for i in row_order],
            "man_eids": eids_all[man_idx].tolist(),
            "obs_frac_ordered": mean_cov_per_sample[col_order],
            "D_prec": D_prec,
        },
    }


def compute_dual_expression_vs_psi_umap(
    expr_df: pd.DataFrame,
    psi_landscape: dict[str, Any],
    k_neighbors: int = 12,
) -> dict[str, Any]:
    """Align Gene-Expression UMAP with Precision-Kernel Event-PSI UMAP via Procrustes & compute per-cell Divergence."""
    from sklearn.metrics import silhouette_score

    samples = psi_landscape["samples"]
    s_ids = [s["sample_id"] for s in samples]
    n_s = len(s_ids)
    expr_num = (
        expr_df.drop(columns=[c for c in ("row_id", "Unnamed: 0") if c in expr_df.columns])
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=1, how="all")
    )
    expr_num.index = expr_num.index.astype(str)
    X_e = expr_num.reindex(s_ids).fillna(0.0).to_numpy(dtype=float)
    X_e = X_e - X_e.mean(axis=0, keepdims=True)

    def _pca_2d(mat: np.ndarray) -> np.ndarray:
        out = np.zeros((mat.shape[0], 2), dtype=float)
        if mat.shape[1] == 0:
            return out
        u, sv, _ = np.linalg.svd(mat, full_matrices=False)
        proj = u * sv
        k_d = min(2, proj.shape[1])
        out[:, :k_d] = proj[:, :k_d]
        return out

    expr_umap = _pca_2d(X_e)
    if n_s >= 6 and X_e.shape[1] >= 2:
        try:
            import umap
            nn = max(4, min(int(k_neighbors), n_s - 1))
            row_norms = np.linalg.norm(X_e, axis=1)
            use_metric = "cosine" if np.all(row_norms > 1e-12) else "euclidean"
            cand_e = np.asarray(
                umap.UMAP(n_neighbors=nn, min_dist=0.22, metric=use_metric, random_state=42).fit_transform(X_e),
                dtype=float,
            )
            if cand_e.shape == (n_s, 2) and np.all(np.isfinite(cand_e)):
                expr_umap = cand_e
        except Exception:
            pass

    psi_umap = np.array([[s["psi_UMAP1"], s["psi_UMAP2"]] for s in samples], dtype=float)

    # Isotropic Frobenius standardization & Procrustes alignment of PSI UMAP onto Expression UMAP
    e_cen = expr_umap - expr_umap.mean(axis=0, keepdims=True)
    p_cen = psi_umap - psi_umap.mean(axis=0, keepdims=True)
    e_scale = max(1e-6, float(np.linalg.norm(e_cen)) / math.sqrt(max(1, n_s)))
    p_scale = max(1e-6, float(np.linalg.norm(p_cen)) / math.sqrt(max(1, n_s)))
    e_std = e_cen / e_scale
    p_std = p_cen / p_scale
    R_proc, _ = orthogonal_procrustes(p_std, e_std)
    p_aligned = p_std @ R_proc

    # Cross-modality kNN Divergence Score Delta_kNN(i) = 1 - |N_k^expr(i) cap N_k^psi(i)| / k
    k = max(1, min(int(k_neighbors), n_s - 1))
    D_e = squareform(pdist(X_e, metric="euclidean"))
    D_p = np.asarray(psi_landscape["_internal"]["D_prec"], dtype=float)
    D_e_knn = D_e.copy()
    D_p_knn = D_p.copy()
    np.fill_diagonal(D_e_knn, np.inf)
    np.fill_diagonal(D_p_knn, np.inf)

    divergences = []
    donor_lisi = []
    donors = [s.get("donor_id", "") for s in samples]
    for i in range(n_s):
        nn_e = set(np.argsort(D_e_knn[i])[:k].tolist())
        nn_p = set(np.argsort(D_p_knn[i])[:k].tolist())
        div = 1.0 - (len(nn_e & nn_p) / float(k))
        divergences.append(round(div, 4))
        # Local Inverse Simpson Index (LISI) on PSI neighborhood for Donor batch mixing (ideal for 2 equal donors = 2.0)
        d_neigh = [donors[j] for j in nn_p if donors[j]]
        if d_neigh:
            probs = pd.Series(d_neigh).value_counts(normalize=True).to_numpy()
            donor_lisi.append(round(float(1.0 / np.sum(probs ** 2)), 3))
        else:
            donor_lisi.append(1.0)

    cell_types = [s.get("cell_type") or s.get("condition") or "Cell" for s in samples]
    n_ct = len(set(cell_types))
    sil_ok = 2 <= n_ct <= (n_s - 1)
    sil_psi = round(float(silhouette_score(D_p, cell_types, metric="precomputed")), 4) if sil_ok else 0.0
    sil_expr = round(float(silhouette_score(D_e, cell_types, metric="precomputed")), 4) if sil_ok else 0.0

    enriched_cells = []
    for idx, s in enumerate(samples):
        enriched_cells.append({
            **s,
            "expr_UMAP1": round(float(e_std[idx, 0]), 4),
            "expr_UMAP2": round(float(e_std[idx, 1]), 4),
            "psi_UMAP1": round(float(p_aligned[idx, 0]), 4),
            "psi_UMAP2": round(float(p_aligned[idx, 1]), 4),
            "modality_divergence": divergences[idx],
            "donor_lisi": donor_lisi[idx],
        })

    return {
        "cells": enriched_cells,
        "k_neighbors": k,
        "mean_divergence": round(float(np.mean(divergences)), 4),
        "median_donor_lisi": round(float(np.median(donor_lisi)), 3),
        "cell_type_silhouette_psi": sil_psi,
        "cell_type_silhouette_expr": sil_expr,
    }


def render_psi_clustermap(
    landscape: dict[str, Any],
    png_path: str | Path,
    svg_path: str | Path | None = None,
    title: str = "Depth-Stabilized Splicing Z-Score Clustermap",
    event_meta: dict[str, dict[str, Any]] | None = None,
    max_display_rows: int = 40,
) -> None:
    """Render publication-grade PSI Z-Score Clustermap with exact 1-to-1 row-to-gene alignment,
    left row dendrogram, top sample dendrogram, distinct Condition vs Subtype legends, and explicit row/column axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Patch

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    internal = landscape["_internal"]
    Z_full = np.asarray(internal["Z_ordered"], dtype=float)
    col_link = internal["col_link"]
    sel_eids_full = list(internal["sel_eids"])
    obs_frac = np.asarray(internal["obs_frac_ordered"], dtype=float)
    samples = list(landscape["ordered_samples"])
    conds = list(landscape["ordered_conditions"])
    z_clip = float(landscape.get("z_clip", 2.5))
    show_coverage_strip = bool(len(obs_frac) > 0 and float(np.min(obs_frac)) < 0.97)

    # Select top `max_display_rows` highest-variance rows (preferring unique genes first so EVERY row is 1-to-1 labeled!)
    n_total_hm = len(sel_eids_full)
    row_var_map = {str(r["event_id"]): float(r.get("excess_biovar", 0.0)) for r in landscape.get("heatmap_rows", [])}
    if n_total_hm > max_display_rows:
        ranked_idx = sorted(
            range(n_total_hm),
            key=lambda i: (-row_var_map.get(sel_eids_full[i], float(np.nanstd(Z_full[i]))), i),
        )
        chosen: list[int] = []
        seen_g: set[str] = set()
        # Pass 1: prioritize unique host genes
        for idx_r in ranked_idx:
            if len(chosen) >= max_display_rows:
                break
            g_sym = str((event_meta or {}).get(sel_eids_full[idx_r], {}).get("gene_symbol", "")).strip()
            if g_sym and g_sym not in seen_g:
                seen_g.add(g_sym)
                chosen.append(idx_r)
        # Pass 2: fill remaining slots up to max_display_rows
        for idx_r in ranked_idx:
            if len(chosen) >= max_display_rows:
                break
            if idx_r not in chosen:
                chosen.append(idx_r)
        Z_sub = Z_full[chosen, :]
        eids_sub = [sel_eids_full[i] for i in chosen]
    else:
        Z_sub = Z_full
        eids_sub = sel_eids_full

    # Re-cluster displayed rows so the left row dendrogram matches displayed rows 1-to-1
    row_link = None
    if len(eids_sub) >= 3:
        try:
            rd = np.nan_to_num(pdist(Z_sub, metric="euclidean"), nan=0.0)
            row_link = linkage(rd, method="ward")
            r_ord = leaves_list(row_link).tolist()
            Z = Z_sub[r_ord, :]
            sel_eids = [eids_sub[i] for i in r_ord]
        except Exception:
            Z = Z_sub
            sel_eids = eids_sub
            row_link = None
    else:
        Z = Z_sub
        sel_eids = eids_sub

    n_disp = len(sel_eids)
    uniq_conds = list(dict.fromkeys(conds))
    cond_color = {c: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, c in enumerate(uniq_conds)}

    fig_h = max(8.2, min(12.2, 3.0 + 0.195 * n_disp))
    fig = plt.figure(figsize=(14.0, fig_h), dpi=220)
    fig.patch.set_facecolor("white")
    if show_coverage_strip:
        gs = fig.add_gridspec(
            4, 5,
            height_ratios=[0.85, 0.24, 0.18, 6.8],
            width_ratios=[0.82, 0.22, 6.9, 1.95, 0.24],
            hspace=0.035, wspace=0.04,
        )
        hm_row_idx = 3
    else:
        gs = fig.add_gridspec(
            3, 5,
            height_ratios=[0.88, 0.26, 6.9],
            width_ratios=[0.82, 0.22, 6.9, 1.95, 0.24],
            hspace=0.035, wspace=0.04,
        )
        hm_row_idx = 2

    # 1. Top Sample Dendrogram (over columns)
    ax_dend = fig.add_subplot(gs[0, 2])
    if col_link is not None and len(samples) >= 3:
        dendrogram(col_link, ax=ax_dend, color_threshold=0, above_threshold_color=EDITORIAL_NAVY, no_labels=True)
    ax_dend.axis("off")

    # 2. Sample Group Color Bar
    ax_cond = fig.add_subplot(gs[1, 2])
    cond_rgb = np.array([[matplotlib.colors.to_rgb(cond_color[c]) for c in conds]])
    ax_cond.imshow(cond_rgb, aspect="auto", interpolation="nearest")
    ax_cond.set_xticks([])
    ax_cond.set_yticks([0], ["Sample Group"], fontsize=8.0, fontweight="bold", color=EDITORIAL_NAVY)
    for sp in ax_cond.spines.values():
        sp.set_visible(False)

    # 3. Optional Cell Completeness Bar (only shown in sparse single-cell cohorts where <97% complete)
    if show_coverage_strip:
        ax_cov = fig.add_subplot(gs[2, 2])
        cov_cmap = LinearSegmentedColormap.from_list("cov_editorial", ["#E2E8F0", "#93C5FD", "#051C2C"])
        ax_cov.imshow(obs_frac[None, :], aspect="auto", cmap=cov_cmap, vmin=0.35, vmax=1.0, interpolation="nearest")
        ax_cov.set_xticks([])
        ax_cov.set_yticks([0], ["Detected %"], fontsize=7.6, color=EDITORIAL_SLATE)
        for sp in ax_cov.spines.values():
            sp.set_visible(False)

    # 4. Left Row Dendrogram (exact 1-to-1 alignment with displayed rows!)
    ax_rdend = fig.add_subplot(gs[hm_row_idx, 0])
    if row_link is not None and n_disp >= 3:
        dendrogram(row_link, ax=ax_rdend, orientation="left", color_threshold=0, above_threshold_color=EDITORIAL_NAVY, no_labels=True)
        ax_rdend.invert_yaxis()
    ax_rdend.axis("off")

    # 5. Left Event-Subtype Strip (aligned 1-to-1 with every row!)
    ax_et = fig.add_subplot(gs[hm_row_idx, 1])
    etypes = [str((event_meta or {}).get(e, {}).get("event_type") or infer_event_type(e)) for e in sel_eids]
    et_rgb = np.array([[matplotlib.colors.to_rgb(EVENT_TYPE_COLORS.get(et, "#94A3B8"))] for et in etypes])
    ax_et.imshow(et_rgb, aspect="auto", interpolation="nearest")
    if n_disp >= 2:
        ax_et.hlines(np.arange(0.5, n_disp - 0.5), -0.5, 0.5, colors="white", linewidths=0.35, alpha=0.65)
    ax_et.set_xticks([0], ["Type"], fontsize=7.6, rotation=90, fontweight="bold", color=EDITORIAL_SLATE)
    ax_et.set_yticks([])
    for sp in ax_et.spines.values():
        sp.set_visible(False)

    # 6. Main Diverging PSI Z-Score Heatmap — 100% 1-to-1 Row-to-Gene Alignment!
    ax_hm = fig.add_subplot(gs[hm_row_idx, 2])
    div_cmap = LinearSegmentedColormap.from_list("editorial_z", [EDITORIAL_CERULEAN, "#E0F2FE", "#FFFFFF", "#FECDD3", EDITORIAL_CORAL])
    im = ax_hm.imshow(Z, aspect="auto", cmap=div_cmap, vmin=-z_clip, vmax=z_clip, interpolation="nearest")
    if n_disp >= 2:
        ax_hm.hlines(np.arange(0.5, n_disp - 0.5), -0.5, len(samples) - 0.5, colors="white", linewidths=0.40, alpha=0.55)
    if len(samples) <= 32:
        ax_hm.vlines(np.arange(0.5, len(samples) - 0.5), -0.5, n_disp - 0.5, colors="white", linewidths=0.40, alpha=0.45)
        ax_hm.set_xticks(np.arange(len(samples)))
        ax_hm.set_xticklabels(samples, rotation=60, ha="right", fontsize=8.1, color=EDITORIAL_NAVY)
        ax_hm.set_xlabel(f"Columns: {len(samples)} Biological Samples (Ward-clustered across {landscape.get('manifold_events', landscape['selected_events']):,} events)", fontsize=8.5, fontweight="bold", color=EDITORIAL_NAVY, labelpad=6)
    else:
        ax_hm.set_xticks([])
        ax_hm.set_xlabel(f"Columns: {len(samples):,} Single Cells ordered by Ward linkage across {landscape.get('manifold_events', landscape['selected_events']):,} events", fontsize=8.6, fontweight="bold", color=EDITORIAL_NAVY, labelpad=6)

    # EVERY displayed row gets its exact 1-to-1 Gene (EventType) label!
    row_lbls = [
        f"{(event_meta or {}).get(sel_eids[i], {}).get('gene_symbol', sel_eids[i][:14])} ({etypes[i]})"
        for i in range(n_disp)
    ]
    lbl_fs = 7.7 if n_disp <= 40 else 6.8
    ax_hm.set_yticks(np.arange(n_disp), row_lbls, fontsize=lbl_fs, color=EDITORIAL_NAVY)
    ax_hm.yaxis.tick_right()
    ax_hm.tick_params(axis="y", length=3.0, pad=3.0)
    ax_hm.text(
        1.01,
        1.012,
        f"Rows: Top {n_disp} Events\n(1-to-1 row aligned)",
        transform=ax_hm.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.6,
        fontweight="bold",
        color=EDITORIAL_NAVY,
    )

    # 7. Colorbar (placed in column 4 after dedicated column-3 gene-label gutter)
    ax_cb = fig.add_subplot(gs[hm_row_idx, 4])
    cb = fig.colorbar(im, cax=ax_cb)
    cb.set_label(f"Row-Standardized Shrunk PSI Z-Score (±{z_clip:.1f} SD)", fontsize=8.3, fontweight="bold", color=EDITORIAL_NAVY)
    cb.ax.tick_params(labelsize=7.6)
    cb.outline.set_edgecolor("#CBD5E1")

    # 8. TWO SEPARATE, CLEARLY TITLED LEGENDS
    cond_patches = [Patch(facecolor=cond_color[c], label=c) for c in uniq_conds[:8]]
    present_ets = [et for et in ("SE", "RI", "A3SS", "A5SS", "MXE") if et in set(etypes)]
    et_patches = [Patch(facecolor=EVENT_TYPE_COLORS[et], label=et) for et in present_ets]

    leg1 = fig.legend(
        handles=cond_patches,
        title="Sample Group (Top Column Bar)",
        title_fontsize=8.0,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.925),
        ncol=min(6, max(1, len(cond_patches))),
        frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=7.8,
    )
    leg1.get_title().set_fontweight("bold")
    fig.add_artist(leg1)

    if et_patches:
        leg2 = fig.legend(
            handles=et_patches,
            title="Splicing Class (Left Row Strip)",
            title_fontsize=8.0,
            loc="upper right",
            bbox_to_anchor=(0.94, 0.925),
            ncol=len(et_patches),
            frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=7.8,
        )
        leg2.get_title().set_fontweight("bold")

    man_n = landscape.get("manifold_events", landscape["selected_events"])
    mode_lbl = "unsupervised biological variance" if "unsupervised" in landscape["selection_mode"] else "PS-DAS signature"
    r2_pct = float(landscape.get("permanova_r2") or 0.0) * 100.0
    p_txt = _fmt_permanova_p(landscape.get("permanova_p"))
    fig.text(0.04, 0.978, title, fontsize=11.8, fontweight="bold", color=EDITORIAL_NAVY, ha="left", va="top")
    fig.text(
        0.04, 0.949,
        f"Rows = Top {n_disp} variable splicing events (1-to-1 aligned) · Columns = {len(samples):,} {'samples' if len(samples) <= 32 else 'cells'} · PERMANOVA R² = {r2_pct:.1f}%, {p_txt} (across {man_n:,} {mode_lbl} events)",
        fontsize=8.2, color=EDITORIAL_SLATE, ha="left", va="top",
    )
    fig.subplots_adjust(left=0.04, right=0.89, top=0.84, bottom=0.12)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    if svg_path:
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _draw_confidence_ellipse(ax, xs: np.ndarray, ys: np.ndarray, color: str, level: float = 0.90) -> None:
    from matplotlib.patches import Ellipse
    if len(xs) < 3:
        return
    cov = np.cov(xs, ys)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    if vals[0] <= 1e-10:
        return
    scale = math.sqrt(chi2.ppf(level, df=2))
    width, height = 2 * scale * np.sqrt(np.maximum(1e-10, vals))
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    ell = Ellipse(
        xy=(float(np.mean(xs)), float(np.mean(ys))),
        width=float(width), height=float(height), angle=float(angle),
        facecolor=color, alpha=0.09, edgecolor=color, linewidth=1.1, linestyle="--", zorder=1,
    )
    ax.add_patch(ell)


def render_psi_pca(
    landscape: dict[str, Any],
    png_path: str | Path,
    svg_path: str | Path | None = None,
    title: str = "Precision-Shrunk Splicing Principal Components & Condition Separation",
) -> None:
    """Render publication 2-panel PSI PCA (PC1 vs PC2 with 90% confidence ellipses + PERMANOVA & Scree spectrum)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.0, "axes.spines.top": False, "axes.spines.right": False})
    samples = landscape["samples"]
    conds = list(dict.fromkeys(s["condition"] for s in samples))
    cond_color = {c: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, c in enumerate(conds)}
    markers = ["o", "s", "^", "D", "P", "X", "v", "<"]

    fig, (ax_pca, ax_scree) = plt.subplots(
        1, 2, figsize=(13.0, 5.1),
        gridspec_kw={"width_ratios": [2.25, 1.0], "wspace": 0.36}, dpi=220,
    )
    fig.patch.set_facecolor("white")

    ax_pca.axhline(0, color="#E2E8F0", linewidth=0.8, zorder=0)
    ax_pca.axvline(0, color="#E2E8F0", linewidth=0.8, zorder=0)
    ax_pca.grid(True, color="#F1F5F9", linewidth=0.6, zorder=0)

    all_x = np.array([float(s["PC1"]) for s in samples], dtype=float)
    all_y = np.array([float(s["PC2"]) for s in samples], dtype=float)
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    x_span = max(0.5, x_max - x_min)
    y_span = max(0.5, y_max - y_min)
    ax_pca.set_xlim(x_min - 0.30 * x_span, x_max + 0.32 * x_span)
    ax_pca.set_ylim(y_min - 0.22 * y_span, y_max + 0.24 * y_span)

    for idx, c in enumerate(conds):
        pts = [s for s in samples if s["condition"] == c]
        xs = np.array([p["PC1"] for p in pts], dtype=float)
        ys = np.array([p["PC2"] for p in pts], dtype=float)
        col = cond_color[c]
        mk = markers[idx % len(markers)]
        _draw_confidence_ellipse(ax_pca, xs, ys, col, level=0.90)
        ax_pca.scatter(xs, ys, s=78, color=col, marker=mk, edgecolors="white", linewidths=0.9, label=f"{c} (n={len(pts)})", zorder=3)

    if len(samples) <= 24:
        cond_rank = {c: idx for idx, c in enumerate(conds)}
        donor_pts: dict[str, list[dict[str, Any]]] = {}
        for s in samples:
            did = str(s.get("donor_id") or "").strip()
            if did and did.lower() not in _BLANK_DONOR_TOKENS:
                donor_pts.setdefault(did, []).append(s)
        paired_trajs = [
            pts for did, pts in donor_pts.items()
            if len({p["condition"] for p in pts}) >= 2 and any(str(p.get("sample_id", "")).strip() != did for p in pts)
        ]
        if len(paired_trajs) >= 2:
            for ti, pts in enumerate(paired_trajs):
                pts_ord = sorted(pts, key=lambda item: cond_rank.get(item["condition"], 999))
                ax_pca.plot(
                    [float(p["PC1"]) for p in pts_ord],
                    [float(p["PC2"]) for p in pts_ord],
                    color="#64748B",
                    linestyle="--",
                    linewidth=1.15,
                    alpha=0.58,
                    label=f"Paired subject (n={len(paired_trajs)})" if ti == 0 else None,
                    zorder=2,
                )

    if len(samples) <= 24:
        x_mid = 0.5 * (x_min + x_max)
        min_dy = 0.078 * (y_span * 1.46)
        for is_right in (False, True):
            side_pts = sorted(
                [p for p in samples if (float(p["PC1"]) >= x_mid) == is_right],
                key=lambda item: float(item["PC2"]),
            )
            if not side_pts:
                continue
            ys_adj = [float(p["PC2"]) for p in side_pts]
            for _ in range(18):
                for i in range(1, len(ys_adj)):
                    if ys_adj[i] - ys_adj[i - 1] < min_dy:
                        mid_v = 0.5 * (ys_adj[i] + ys_adj[i - 1])
                        ys_adj[i - 1] = mid_v - 0.5 * min_dy
                        ys_adj[i] = mid_v + 0.5 * min_dy
            y_lo, y_hi = y_min - 0.16 * y_span, y_max + 0.18 * y_span
            if ys_adj[-1] > y_hi:
                shift = ys_adj[-1] - y_hi
                ys_adj = [v - shift for v in ys_adj]
            if ys_adj[0] < y_lo:
                shift = y_lo - ys_adj[0]
                ys_adj = [v + shift for v in ys_adj]
            for p, y_lbl in zip(side_pts, ys_adj):
                px, py = float(p["PC1"]), float(p["PC2"])
                place_left_of_pt = px > (x_max - 0.18 * x_span) if is_right else px > (x_min + 0.22 * x_span)
                dx = -0.055 * x_span if place_left_of_pt else 0.055 * x_span
                ha = "right" if place_left_of_pt else "left"
                ax_pca.annotate(
                    p["sample_id"],
                    xy=(px, py),
                    xytext=(px + dx, y_lbl),
                    ha=ha,
                    va="center",
                    fontsize=7.4,
                    color=EDITORIAL_NAVY,
                    fontweight="bold",
                    arrowprops={"arrowstyle": "-", "color": "#94A3B8", "lw": 0.65, "shrinkA": 2, "shrinkB": 3},
                    zorder=5,
                )

    vexp = landscape["pca_variance_pct"]
    ax_pca.set_xlabel(f"PC1 ({vexp[0]:.1f}% splicing variance)", fontweight="bold", color=EDITORIAL_NAVY)
    ax_pca.set_ylabel(f"PC2 ({vexp[1]:.1f}% splicing variance)", fontweight="bold", color=EDITORIAL_NAVY)
    strata_mode = str(landscape.get("permanova_strata") or "free")
    glob_r2_pct = float(landscape.get("permanova_r2") or 0.0) * 100.0
    if strata_mode == "subject_paired" and landscape.get("permanova_within_subject_r2") is not None and landscape.get("permanova_p_within_subject") is not None:
        pca_title = (
            f"A. Sample splicing manifold (Within-subject R² = {float(landscape['permanova_within_subject_r2'])*100:.1f}%, "
            f"Paired {_fmt_permanova_p(landscape.get('permanova_p_within_subject'))}; Global R² = {glob_r2_pct:.1f}%)"
        )
    elif strata_mode == "split_plot_paired" and landscape.get("permanova_within_subject_r2") is not None:
        pca_title = (
            f"A. Sample splicing manifold (Within-subject R² = {float(landscape['permanova_within_subject_r2'])*100:.1f}%, "
            f"Paired {_fmt_permanova_p(landscape.get('permanova_p_within_subject'))}; Global R² = {glob_r2_pct:.1f}%)"
        )
    else:
        pca_title = f"A. Sample splicing manifold (PERMANOVA R² = {glob_r2_pct:.1f}%, {_fmt_permanova_p(landscape.get('permanova_p'))})"
    ax_pca.set_title(
        pca_title,
        loc="left", fontsize=9.6 if strata_mode != "free" else 10.2, fontweight="bold", color=EDITORIAL_NAVY,
    )
    ax_pca.legend(frameon=True, facecolor="#FFFFFF", framealpha=0.88, edgecolor="#E2E8F0", fontsize=8.0, loc="upper left")

    # Panel B: Variance spectrum + Beyond-Jutils QC summary
    pcs = [f"PC{i+1}" for i in range(len(vexp))]
    cum = np.cumsum(vexp)
    ax_scree.bar(np.arange(len(vexp)), vexp, width=0.52, color=EDITORIAL_NAVY, label="Individual PC %")
    ax_scree.plot(np.arange(len(vexp)), cum, color=EDITORIAL_BLUE, marker="o", linewidth=1.8, label="Cumulative %")
    for xi, v in enumerate(vexp):
        ax_scree.text(xi, v + max(vexp) * 0.04, f"{v:.1f}%", ha="center", fontsize=7.8, color=EDITORIAL_NAVY)
    ax_scree.set_xticks(np.arange(len(vexp)), pcs)
    ax_scree.set_ylabel("Explained variance (%)")
    ax_scree.set_ylim(0, max(cum[-1] * 1.18, 25))
    ax_scree.set_title("B. Variance spectrum", loc="left", fontsize=10.2, fontweight="bold", color=EDITORIAL_NAVY)
    ax_scree.legend(frameon=False, fontsize=8.0, loc="upper left")

    fig.suptitle(title, x=0.01, ha="left", fontsize=12.0, fontweight="bold", color=EDITORIAL_NAVY, y=1.02)
    fig.subplots_adjust(wspace=0.36)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    if svg_path:
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_dual_psi_umap(
    dual: dict[str, Any],
    png_path: str | Path,
    svg_path: str | Path | None = None,
    title: str = "Precision-Kernel Event-PSI UMAP & Cell-Type Modality Divergence",
) -> None:
    """Render 3-panel Single-Cell / Cohort UMAP:
    Panel A: Gene Expression UMAP (colored by Cell Type, shaped by Activation State)
    Panel B: Precision-Kernel Event-PSI UMAP (colored by Cell Type, shaped by Activation State)
    Panel C: Cross-Modality Splicing-vs-Expression Divergence Score Delta_kNN per cell
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D

    png_path = Path(png_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.8, "axes.spines.top": False, "axes.spines.right": False})

    cells = dual["cells"]
    ct_counts: dict[str, int] = {}
    for c in cells:
        k = str(c.get("cell_type") or c.get("condition") or "Cells")
        ct_counts[k] = ct_counts.get(k, 0) + 1
    ctypes = sorted(ct_counts.keys(), key=lambda x: (-ct_counts[x], x))
    ct_col = {ct: CONDITION_PALETTE[i % len(CONDITION_PALETTE)] for i, ct in enumerate(ctypes)}
    conds = sorted(set(str(c.get("condition", "")) for c in cells))
    mk_map = {cond: ["o", "^", "s", "D", "P", "X"][i % 6] for i, cond in enumerate(conds)}
    pt_s = 34 if len(cells) <= 400 else (18 if len(cells) <= 1500 else 10)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(16.4, 5.4),
        gridspec_kw={"wspace": 0.26}, dpi=220,
    )
    fig.patch.set_facecolor("white")

    for ct in ctypes:
        for cond in conds:
            sub = [c for c in cells if str(c.get("cell_type") or c.get("condition") or "Cells") == ct and str(c.get("condition", "")) == cond]
            if not sub:
                continue
            ax1.scatter(
                [c["expr_UMAP1"] for c in sub], [c["expr_UMAP2"] for c in sub],
                s=pt_s, color=ct_col[ct], marker=mk_map[cond], alpha=0.86, edgecolors="white", linewidths=0.35,
            )
            ax2.scatter(
                [c["psi_UMAP1"] for c in sub], [c["psi_UMAP2"] for c in sub],
                s=pt_s, color=ct_col[ct], marker=mk_map[cond], alpha=0.86, edgecolors="white", linewidths=0.35,
            )

    ax1.set_title(f"A. Expression UMAP (Sil = {dual['cell_type_silhouette_expr']:.2f})", loc="left", fontsize=10.0, fontweight="bold", color=EDITORIAL_NAVY)
    ax1.set_xlabel("Expression UMAP 1", fontweight="bold", color=EDITORIAL_NAVY)
    ax1.set_ylabel("Expression UMAP 2", fontweight="bold", color=EDITORIAL_NAVY)

    ax2.set_title(f"B. Precision Event-PSI UMAP (Sil = {dual['cell_type_silhouette_psi']:.2f})", loc="left", fontsize=10.0, fontweight="bold", color=EDITORIAL_NAVY)
    ax2.set_xlabel("Precision PSI UMAP 1 (Procrustes-aligned)", fontweight="bold", color=EDITORIAL_NAVY)
    ax2.set_ylabel("Precision PSI UMAP 2", fontweight="bold", color=EDITORIAL_NAVY)

    div_cmap = LinearSegmentedColormap.from_list("div_editorial", ["#E2E8F0", "#60A5FA", "#2251FF", EDITORIAL_CORAL])
    divs = np.array([c["modality_divergence"] for c in cells], dtype=float)
    sc3 = ax3.scatter(
        [c["psi_UMAP1"] for c in cells], [c["psi_UMAP2"] for c in cells],
        s=pt_s + 2, c=divs, cmap=div_cmap, vmin=0.0, vmax=1.0, edgecolors="white", linewidths=0.35,
    )
    cb = fig.colorbar(sc3, ax=ax3, fraction=0.046, pad=0.03)
    cb.set_label("Cross-modality divergence Δ_kNN", fontsize=8.2, fontweight="bold", color=EDITORIAL_NAVY)
    ax3.set_title(f"C. Modality Divergence (Donor LISI = {dual['median_donor_lisi']:.2f})", loc="left", fontsize=10.0, fontweight="bold", color=EDITORIAL_NAVY)
    ax3.set_xlabel("Precision PSI UMAP 1", fontweight="bold", color=EDITORIAL_NAVY)
    ax3.set_ylabel("Precision PSI UMAP 2", fontweight="bold", color=EDITORIAL_NAVY)

    # Separated, zero-overlap external legends above panels (Color = Cell Type; Shape = Condition)
    shown_ct = ctypes[:10]
    ct_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=ct_col[ct], markeredgecolor="white", markersize=7.0, label=f"{ct[:22]} (n={ct_counts[ct]})")
        for ct in shown_ct
    ]
    if len(ctypes) > 10:
        ct_handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="#94A3B8", markersize=6.5, label=f"+{len(ctypes) - 10} other populations"))
    cond_handles = [
        Line2D([0], [0], marker=mk_map[cd], color="none", markerfacecolor="#334155", markeredgecolor="white", markersize=7.2, label=cd)
        for cd in conds[:6]
    ]
    leg_ct = fig.legend(
        handles=ct_handles,
        title="Color: Cell Population",
        title_fontsize=8.0,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.935),
        ncol=min(5, max(1, len(ct_handles))),
        frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=7.6,
    )
    leg_ct.get_title().set_fontweight("bold")
    fig.add_artist(leg_ct)
    if len(cond_handles) >= 1 and set(conds) != set(ctypes):
        leg_cd = fig.legend(
            handles=cond_handles,
            title="Shape: Condition",
            title_fontsize=8.0,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.935),
            ncol=min(4, len(cond_handles)),
            frameon=True, facecolor="#F8FAFC", edgecolor="#E2E8F0", fontsize=7.6,
        )
        leg_cd.get_title().set_fontweight("bold")

    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=12.0, fontweight="bold", color=EDITORIAL_NAVY)
    top_margin = 0.77 if len(shown_ct) <= 5 else 0.72
    fig.subplots_adjust(left=0.05, right=0.96, top=top_margin, bottom=0.11, wspace=0.25)
    fig.savefig(png_path, dpi=220, bbox_inches="tight", facecolor="white")
    if svg_path:
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
