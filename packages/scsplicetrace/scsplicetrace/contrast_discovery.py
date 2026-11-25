"""Task 1: Threshold-free single-cell splicing contrast & dual donor/gene-block calibrated RBP discovery."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, t as student_t


def _bh_fdr_mask(p_vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    fdr = np.full(len(p_vals), np.nan, dtype=float)
    ok = mask & np.isfinite(p_vals)
    if not np.any(ok):
        return fdr
    pv = np.clip(p_vals[ok], 0.0, 1.0)
    n = len(pv)
    order = np.argsort(pv)
    q = pv[order] * n / np.arange(1, n + 1, dtype=float)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(q, 0.0, 1.0)
    fdr[ok] = out
    return fdr


_MISSING_DONOR_TOKENS = {"", "nan", "none", "na", "null", "n/a"}


def _valid_donor_mask(series: pd.Series) -> np.ndarray:
    """Return boolean mask identifying valid, non-empty biological donor identifiers."""
    vals = series.tolist()
    return np.array(
        [(not pd.isna(v)) and (str(v).strip().lower() not in _MISSING_DONOR_TOKENS) for v in vals],
        dtype=bool,
    )


@dataclass(frozen=True)
class SCSpliceTraceContrastResult:
    """Output of scSpliceTrace 1.0 de novo RBP & mature-mRNA motif discovery."""

    rbp_ranking: pd.DataFrame
    region_tests: pd.DataFrame
    event_contrasts: pd.DataFrame
    mature_consequences: pd.DataFrame
    diagnostics: dict


def compute_sc_splicing_contrast(
    usage: pd.DataFrame,
    metadata: pd.DataFrame,
    event_genes: pd.Series,
    *,
    support: pd.DataFrame | None = None,
    group_col: str = "condition",
    numerator_group: str,
    denominator_group: str,
    donor_col: str = "donor_id",
    min_cells_per_group: int = 8,
    min_support_per_group: float = 10.0,
    min_paired_donors: int = 3,
    min_unpaired_donors_per_group: int = 2,
    n_cell_blocks_if_few_donors: int = 8,
    design_mode: str = "auto",
    seed: int = 20261117,
) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    """Compute precision-weighted continuous splicing shifts (`dpsi`, `z_shift`) without hard FDR filtering.

    Unified dispatch (`design_mode='auto'` shared across CLI and Python API; valid override modes:
    `'paired'`, `'unpaired'`, `'stratified_meta'`):
    - Drops cells carrying missing/blank donor IDs when `donor_col` is present (never fabricates a `'nan'` donor).
    - `>= 5` paired donors -> `paired_donor_population` (within-donor paired deltas + sign-flip-invariant
      paired total cell/support eligibility + donor-delta matrix for donor sign-flip maxT).
    - `3–4` paired donors -> `paired_donor_small_n` (within-donor paired deltas + sign-flip-invariant paired
      eligibility + gene-block Rademacher maxT).
    - `2` paired donors -> `stratified_paired_2donor` (strictly within-donor estimation + inverse-variance
      meta-analysis across the 2 paired donors; zero cross-donor cell pooling; excludes unpaired donors).
    - `0` paired donors with `>= min_unpaired_donors_per_group` per arm -> `unpaired_donor_pseudobulk`
      (equal-donor-weighted means and Welch SE).
    - `1` paired donor with `>= min_unpaired_donors_per_group` disjoint donors per arm ->
      `partially_paired_disjoint_unpaired` (excludes overlapping donor before equal-donor Welch inference).
    - Exactly `1` biological donor across the entire comparison (or no donor column) ->
      `single_donor_gene_block` (within-cohort cell-block contrast, e.g., single-donor Perturb-seq).
    - Any other multi-donor configuration (e.g. `1` shared + `1` disjoint per arm) ->
      `insufficient_multi_donor_design` (`eligible_events = 0`; never pools cells across heterogeneous donors).
    """
    if design_mode not in {"auto", "paired", "unpaired", "stratified_meta"}:
        raise ValueError(f"Unsupported design_mode: {design_mode!r}")
    md = metadata.copy()
    if "qc_pass" in md.columns:
        md = md.loc[md["qc_pass"].astype(str).str.lower().isin(("true", "1", "yes", "pass"))].copy()
    cells_excluded_missing_donor_id = 0
    if donor_col not in md.columns and "donor" in md.columns:
        md[donor_col] = md["donor"]
    if donor_col in md.columns:
        valid_dn = _valid_donor_mask(md[donor_col])
        cells_excluded_missing_donor_id = int((~valid_dn).sum())
        md = md.loc[valid_dn].copy()
        if md.empty:
            raise ValueError("Zero cells remain after excluding missing or blank donor IDs")
        md[donor_col] = md[donor_col].astype(str).str.strip()
    else:
        md[donor_col] = "single_cohort"
    md["cell_id"] = md["cell_id"].astype(str)
    md = md.loc[md[group_col].astype(str).isin([str(numerator_group), str(denominator_group)])].copy()
    if md.empty:
        raise ValueError("Zero cells match numerator_group / denominator_group")

    u_df = usage.copy()
    u_df.index = u_df.index.astype(str)
    valid_cells = set(u_df.index)
    if support is not None:
        s_df = support.copy()
        s_df.index = s_df.index.astype(str)
        valid_cells &= set(s_df.index)
    else:
        s_df = None
    md = md.loc[md["cell_id"].isin(valid_cells)].copy()
    if md.empty:
        raise ValueError("Zero cells overlap between metadata and usage/support matrices")

    u = u_df.loc[md["cell_id"]].to_numpy(dtype=float)
    finite = np.isfinite(u)
    if s_df is not None:
        t = np.where(finite, np.maximum(s_df.loc[md["cell_id"], u_df.columns].to_numpy(dtype=float), 0.0), 0.0)
    else:
        t = finite.astype(float)

    g_vec = md[group_col].astype(str).to_numpy()
    d_vec = md[donor_col].astype(str).to_numpy()
    is_num = g_vec == str(numerator_group)
    is_den = g_vec == str(denominator_group)

    def _pooled_stats(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f_sub = finite[mask]
        t_sub = np.where(f_sub, t[mask], 0.0)
        u_sub = np.where(f_sub, u[mask], 0.0)
        w_sub = np.divide(t_sub, 1.0 + 0.02 * t_sub, out=np.zeros_like(t_sub), where=(t_sub > 0) & f_sub)
        w_tot = w_sub.sum(axis=0)
        w_sq = (w_sub * w_sub).sum(axis=0)
        t_tot = t_sub.sum(axis=0)
        n_c = f_sub.sum(axis=0)
        psi_hat = np.divide((w_sub * u_sub).sum(axis=0), w_tot, out=np.full(u.shape[1], np.nan), where=w_tot > 0)
        # Exact weighted binomial variance + Kish ESS biological dispersion floor
        p_clip = np.clip(np.nan_to_num(psi_hat, nan=0.5), 0.02, 0.98)
        n_eff = np.divide(w_tot * w_tot, w_sq, out=np.zeros_like(w_tot), where=w_sq > 0)
        w2_over_t = np.divide(w_sub * w_sub, np.maximum(t_sub, 1.0), out=np.zeros_like(t_sub), where=(t_sub > 0) & f_sub).sum(axis=0)
        bin_term = np.divide(w2_over_t, w_tot * w_tot, out=np.full(u.shape[1], np.nan), where=w_tot > 0)
        bio_term = np.divide(0.02, n_eff, out=np.zeros_like(w_tot), where=n_eff > 0)
        se2_bin = np.where(w_tot > 0, p_clip * (1.0 - p_clip) * (bin_term + bio_term), np.nan)
        return psi_hat, se2_bin, n_c, t_tot

    psi_num, se2_num, nc_num, sup_num = _pooled_stats(is_num)
    psi_den, se2_den, nc_den, sup_den = _pooled_stats(is_den)

    # Estimate replicate/block empirical variance respecting independent biological donor structure
    donors_num = sorted(set(d_vec[is_num]))
    donors_den = sorted(set(d_vec[is_den]))
    all_donors = sorted(set(donors_num) | set(donors_den))
    paired_donors = sorted(set(donors_num) & set(donors_den))

    donor_delta_df: pd.DataFrame | None = None
    donor_mean_d: np.ndarray | None = None
    donor_base_d: np.ndarray | None = None
    donor_arm1_d: np.ndarray | None = None
    donor_arm0_d: np.ndarray | None = None
    paired_nc_tot: np.ndarray | None = None
    paired_sup_tot: np.ndarray | None = None
    unpaired_nc_num: np.ndarray | None = None
    unpaired_nc_den: np.ndarray | None = None
    unpaired_sup_num: np.ndarray | None = None
    unpaired_sup_den: np.ndarray | None = None
    donor_within_var = np.full(u.shape[1], np.nan, dtype=float)
    df_event = np.full(u.shape[1], np.nan, dtype=float)
    n_paired_ev = np.zeros(u.shape[1], dtype=int)
    n_donors_num_ev = np.zeros(u.shape[1], dtype=int)
    n_donors_den_ev = np.zeros(u.shape[1], dtype=int)
    min_pd = int(max(3, min_paired_donors))
    min_ud = int(max(2, min_unpaired_donors_per_group))
    min_sup_donor = 3.0 if s_df is not None else 2.0
    disjoint_num = sorted(set(donors_num) - set(paired_donors))
    disjoint_den = sorted(set(donors_den) - set(paired_donors))
    extra_info: dict[str, int] = {"cells_excluded_missing_donor_id": cells_excluded_missing_donor_id}

    def _run_paired_donors(p_list: list[str], *, make_delta_df: bool) -> None:
        nonlocal donor_delta_df, donor_mean_d, donor_base_d, donor_arm1_d, donor_arm0_d
        nonlocal paired_nc_tot, paired_sup_tot
        nonlocal donor_within_var, df_event, n_paired_ev, n_donors_num_ev, n_donors_den_ev, donor_eligible, var_between
        d_deltas, d_bases, d_p1s, d_p0s, d_vars = [], [], [], [], []
        num_cov_list, den_cov_list = [], []
        p_cell_tot = np.zeros(u.shape[1], dtype=float)
        p_sup_tot_acc = np.zeros(u.shape[1], dtype=float)
        for d in p_list:
            p1, v1, c1, s1 = _pooled_stats(is_num & (d_vec == d))
            p0, v0, c0, s0 = _pooled_stats(is_den & (d_vec == d))
            ok1 = (c1 >= 2) & (s1 >= min_sup_donor) & np.isfinite(p1)
            ok0 = (c0 >= 2) & (s0 >= min_sup_donor) & np.isfinite(p0)
            num_cov_list.append(ok1)
            den_cov_list.append(ok0)
            ok_d = ok1 & ok0
            # Sum cells & support strictly across valid paired donors so eligibility is 100% invariant to arm swaps
            p_cell_tot += np.where(ok_d, c1 + c0, 0.0)
            p_sup_tot_acc += np.where(ok_d, s1 + s0, 0.0)
            d_deltas.append(np.where(ok_d, p1 - p0, np.nan))
            # Arm-symmetric donor midpoint so FWL nuisance baseline is invariant under donor sign flips
            d_bases.append(np.where(ok_d, 0.5 * (p1 + p0), np.nan))
            d_p1s.append(np.where(ok_d, p1, np.nan))
            d_p0s.append(np.where(ok_d, p0, np.nan))
            d_vars.append(np.where(ok_d, v1 + v0, np.nan))
        paired_nc_tot = p_cell_tot
        paired_sup_tot = p_sup_tot_acc
        d_mat = np.stack(d_deltas, axis=0)
        n_d = np.isfinite(d_mat).sum(axis=0)
        n_paired_ev = n_d.astype(int)
        n_donors_num_ev = np.stack(num_cov_list, axis=0).sum(axis=0).astype(int)
        n_donors_den_ev = np.stack(den_cov_list, axis=0).sum(axis=0).astype(int)
        donor_eligible = n_d >= min_pd
        mean_d = np.divide(np.nansum(d_mat, axis=0), np.maximum(n_d, 1), out=np.full(u.shape[1], np.nan), where=donor_eligible)
        donor_mean_d = mean_d
        donor_base_d = np.divide(np.nansum(np.stack(d_bases, axis=0), axis=0), np.maximum(n_d, 1), out=np.full(u.shape[1], np.nan), where=donor_eligible)
        donor_arm1_d = np.divide(np.nansum(np.stack(d_p1s, axis=0), axis=0), np.maximum(n_d, 1), out=np.full(u.shape[1], np.nan), where=donor_eligible)
        donor_arm0_d = np.divide(np.nansum(np.stack(d_p0s, axis=0), axis=0), np.maximum(n_d, 1), out=np.full(u.shape[1], np.nan), where=donor_eligible)
        var_between = np.divide(
            np.nansum((d_mat - mean_d[None, :]) ** 2, axis=0),
            np.maximum(n_d - 1, 1) * np.maximum(n_d, 1),
            out=np.full(u.shape[1], np.nan),
            where=donor_eligible,
        )
        donor_within_var = np.divide(
            np.nansum(np.stack(d_vars, axis=0), axis=0),
            np.maximum(n_d, 1) ** 2,
            out=np.full(u.shape[1], np.nan),
            where=donor_eligible,
        )
        df_event = np.where(donor_eligible, np.maximum(n_d - 1, 1).astype(float), np.nan)
        if make_delta_df:
            donor_delta_df = pd.DataFrame(d_mat, index=p_list, columns=u_df.columns.astype(str))

    def _run_unpaired_donors(d_num_list: list[str], d_den_list: list[str]) -> None:
        nonlocal donor_mean_d, donor_base_d, donor_arm1_d, donor_arm0_d
        nonlocal unpaired_nc_num, unpaired_nc_den, unpaired_sup_num, unpaired_sup_den
        nonlocal donor_within_var, df_event, n_donors_num_ev, n_donors_den_ev, donor_eligible, var_between
        p_num_cols, v_num_cols = [], []
        c_num_acc = np.zeros(u.shape[1], dtype=float)
        s_num_acc = np.zeros(u.shape[1], dtype=float)
        for d in d_num_list:
            p1, v1, c1, s1 = _pooled_stats(is_num & (d_vec == d))
            ok1 = (c1 >= 2) & (s1 >= min_sup_donor) & np.isfinite(p1)
            c_num_acc += c1
            s_num_acc += s1
            p_num_cols.append(np.where(ok1, p1, np.nan))
            v_num_cols.append(np.where(ok1, v1, np.nan))
        p_den_cols, v_den_cols = [], []
        c_den_acc = np.zeros(u.shape[1], dtype=float)
        s_den_acc = np.zeros(u.shape[1], dtype=float)
        for d in d_den_list:
            p0, v0, c0, s0 = _pooled_stats(is_den & (d_vec == d))
            ok0 = (c0 >= 2) & (s0 >= min_sup_donor) & np.isfinite(p0)
            c_den_acc += c0
            s_den_acc += s0
            p_den_cols.append(np.where(ok0, p0, np.nan))
            v_den_cols.append(np.where(ok0, v0, np.nan))
        unpaired_nc_num = c_num_acc
        unpaired_nc_den = c_den_acc
        unpaired_sup_num = s_num_acc
        unpaired_sup_den = s_den_acc
        p_num_d = np.stack(p_num_cols, axis=0)
        v_num_d = np.stack(v_num_cols, axis=0)
        p_den_d = np.stack(p_den_cols, axis=0)
        v_den_d = np.stack(v_den_cols, axis=0)
        n1_d = np.isfinite(p_num_d).sum(axis=0)
        n0_d = np.isfinite(p_den_d).sum(axis=0)
        n_donors_num_ev = n1_d.astype(int)
        n_donors_den_ev = n0_d.astype(int)
        m1_d = np.divide(np.nansum(p_num_d, axis=0), np.maximum(n1_d, 1), out=np.full(u.shape[1], np.nan), where=n1_d >= 1)
        m0_d = np.divide(np.nansum(p_den_d, axis=0), np.maximum(n0_d, 1), out=np.full(u.shape[1], np.nan), where=n0_d >= 1)
        v1_d = np.divide(np.nansum((p_num_d - m1_d[None, :]) ** 2, axis=0), np.maximum(n1_d - 1, 1), out=np.zeros(u.shape[1]), where=n1_d >= 2)
        v0_d = np.divide(np.nansum((p_den_d - m0_d[None, :]) ** 2, axis=0), np.maximum(n0_d - 1, 1), out=np.zeros(u.shape[1]), where=n0_d >= 2)
        se1_sq = np.divide(v1_d, np.maximum(n1_d, 1), out=np.zeros(u.shape[1]), where=n1_d >= 2)
        se0_sq = np.divide(v0_d, np.maximum(n0_d, 1), out=np.zeros(u.shape[1]), where=n0_d >= 2)
        both_d = (n1_d >= min_ud) & (n0_d >= min_ud)
        donor_eligible = both_d
        # Strictly equal-donor-weighted numerator, symmetric midpoint baseline, and Welch between-donor SE
        donor_arm1_d = np.where(both_d, m1_d, np.nan)
        donor_arm0_d = np.where(both_d, m0_d, np.nan)
        donor_mean_d = np.where(both_d, m1_d - m0_d, np.nan)
        donor_base_d = np.where(both_d, 0.5 * (m1_d + m0_d), np.nan)
        var_between = np.where(both_d, se1_sq + se0_sq, np.nan)
        wv1 = np.divide(np.nansum(v_num_d, axis=0), np.maximum(n1_d, 1) ** 2, out=np.full(u.shape[1], np.nan), where=n1_d >= 1)
        wv0 = np.divide(np.nansum(v_den_d, axis=0), np.maximum(n0_d, 1) ** 2, out=np.full(u.shape[1], np.nan), where=n0_d >= 1)
        donor_within_var = np.where(both_d, wv1 + wv0, np.nan)
        denom_df = (se1_sq ** 2) / np.maximum(n1_d - 1, 1) + (se0_sq ** 2) / np.maximum(n0_d - 1, 1)
        df_event = np.where(
            both_d & (denom_df > 1e-18),
            ((se1_sq + se0_sq) ** 2) / np.maximum(denom_df, 1e-18),
            np.where(both_d, np.maximum(n1_d + n0_d - 2, 1).astype(float), np.nan),
        )

    def _dispatch_stratified_2donor() -> tuple[pd.DataFrame, None, dict]:
        md_p = md.loc[md[donor_col].isin(paired_donors)].copy()
        mcd_cells = min(4, max(2, int(min_cells_per_group)))
        mcd_sup = min(6.0, float(min_support_per_group))
        meta_df, _, meta_info = compute_sc_stratified_meta_contrast(
            u_df, md_p, event_genes, support=s_df,
            group_col=group_col, numerator_group=numerator_group, denominator_group=denominator_group,
            donor_col=donor_col, min_cells_per_donor_group=mcd_cells, min_support_per_donor_group=mcd_sup,
            design_mode="auto", seed=seed,
        )
        el_m = meta_df["eligible"].to_numpy(dtype=bool)
        d_cnt = meta_df["contributing_donors"].to_numpy(dtype=int)
        inf_u = "stratified_paired_2donor" if len(paired_donors) == 2 else "stratified_multi_donor_meta"
        out_df = pd.DataFrame({
            "event_id": meta_df["event_id"],
            "gene_id": meta_df["gene_id"],
            "eligible": el_m,
            "event_inference_unit": np.where(el_m, inf_u, "insufficient_cells_or_support"),
            "n_paired_donors": d_cnt,
            "n_donors_numerator": d_cnt,
            "n_donors_denominator": d_cnt,
            "psi_numerator": meta_df["psi_numerator"],
            "psi_denominator": meta_df["psi_denominator"],
            "baseline_PSI": meta_df["baseline_PSI"],
            "raw_dpsi": meta_df["raw_dpsi"],
            "dpsi": meta_df["dpsi"],
            "se_dpsi": meta_df["se_dpsi"],
            "z_shift": meta_df["z_shift"],
            "p_value": meta_df["p_value"],
            "fdr": meta_df["fdr"],
            "cells_numerator": meta_df["cells_numerator"],
            "cells_denominator": meta_df["cells_denominator"],
            "support_numerator": meta_df["support_numerator"],
            "support_denominator": meta_df["support_denominator"],
            "contributing_donors": meta_df["contributing_donors"],
            "donor_sign_agreement": meta_df["donor_sign_agreement"],
        })
        out_info = {
            "inference_unit": inf_u,
            "paired_donors": len(paired_donors),
            "donors_numerator": len(donors_num),
            "donors_denominator": len(donors_den),
            "min_paired_donors": len(paired_donors),
            "min_unpaired_donors_per_group": 0,
            "unpaired_donors_excluded": len(set(donors_num) ^ set(donors_den)),
            "events_excluded_donor_insufficiency": int((~el_m).sum()),
            "eligible_events": int(el_m.sum()),
            "eligible_genes": int(out_df.loc[el_m, "gene_id"].astype(str).nunique()),
            "donors_evaluated": meta_info.get("donors_evaluated", paired_donors),
            **extra_info,
        }
        return out_df, None, out_info

    # Deterministic unified dispatch shared by CLI and Python API
    u_cand_num = donors_num if len(paired_donors) == 0 else disjoint_num
    u_cand_den = donors_den if len(paired_donors) == 0 else disjoint_den
    can_unpaired = len(u_cand_num) >= min_ud and len(u_cand_den) >= min_ud

    if design_mode == "stratified_meta":
        if len(paired_donors) >= 2:
            return _dispatch_stratified_2donor()
        use_branch = "single_donor" if len(all_donors) == 1 else "insufficient_multi_donor"
    elif design_mode == "paired":
        if len(paired_donors) >= 5:
            use_branch = "paired_pop"
        elif len(paired_donors) >= min_pd:
            use_branch = "paired_small"
        elif len(paired_donors) == 2:
            return _dispatch_stratified_2donor()
        elif len(all_donors) == 1:
            use_branch = "single_donor"
        else:
            use_branch = "insufficient_multi_donor"
    elif design_mode == "unpaired":
        if can_unpaired:
            use_branch = "unpaired_pure" if len(paired_donors) == 0 else "unpaired_disjoint"
        elif len(all_donors) == 1:
            use_branch = "single_donor"
        else:
            use_branch = "insufficient_multi_donor"
    else:  # design_mode == "auto"
        if len(paired_donors) >= 5:
            use_branch = "paired_pop"
        elif len(paired_donors) >= min_pd:
            use_branch = "paired_small"
        elif len(paired_donors) == 2:
            return _dispatch_stratified_2donor()
        elif len(paired_donors) == 0 and len(donors_num) >= min_ud and len(donors_den) >= min_ud:
            use_branch = "unpaired_pure"
        elif len(paired_donors) == 1 and len(disjoint_num) >= min_ud and len(disjoint_den) >= min_ud:
            use_branch = "unpaired_disjoint"
        elif len(all_donors) == 1:
            use_branch = "single_donor"
        else:
            use_branch = "insufficient_multi_donor"

    if use_branch == "paired_pop":
        inference_unit = "paired_donor_population"
        _run_paired_donors(paired_donors, make_delta_df=True)
        extra_info["unpaired_donors_excluded"] = len(set(donors_num) ^ set(donors_den))
    elif use_branch == "paired_small":
        inference_unit = "paired_donor_small_n"
        _run_paired_donors(paired_donors, make_delta_df=False)
        extra_info["unpaired_donors_excluded"] = len(set(donors_num) ^ set(donors_den))
    elif use_branch == "unpaired_pure":
        inference_unit = "unpaired_donor_pseudobulk"
        _run_unpaired_donors(donors_num, donors_den)
    elif use_branch == "unpaired_disjoint":
        inference_unit = "partially_paired_disjoint_unpaired"
        _run_unpaired_donors(disjoint_num, disjoint_den)
        extra_info["overlap_donors_excluded_from_unpaired"] = len(paired_donors)
    elif use_branch == "insufficient_multi_donor":
        inference_unit = "insufficient_multi_donor_design"
        donor_eligible = np.zeros(u.shape[1], dtype=bool)
        var_between = np.full(u.shape[1], np.nan, dtype=float)
    else:
        inference_unit = "single_donor_gene_block"
        donor_eligible = np.ones(u.shape[1], dtype=bool)
        if len(donors_num) > 0:
            p_num_c = np.stack([_pooled_stats(is_num & (d_vec == d))[2] > 0 for d in donors_num], axis=0)
            n_donors_num_ev = p_num_c.sum(axis=0).astype(int)
        if len(donors_den) > 0:
            p_den_c = np.stack([_pooled_stats(is_den & (d_vec == d))[2] > 0 for d in donors_den], axis=0)
            n_donors_den_ev = p_den_c.sum(axis=0).astype(int)
        if len(paired_donors) > 0:
            p_cov = np.stack([
                (_pooled_stats(is_num & (d_vec == d))[2] >= 2)
                & (_pooled_stats(is_den & (d_vec == d))[2] >= 2)
                for d in paired_donors
            ], axis=0)
            n_paired_ev = p_cov.sum(axis=0).astype(int)
        rng = np.random.default_rng(int(seed))
        K = int(max(4, n_cell_blocks_if_few_donors))
        num_idx = rng.permutation(np.flatnonzero(is_num))
        den_idx = rng.permutation(np.flatnonzero(is_den))
        num_splits = np.array_split(num_idx, K)
        den_splits = np.array_split(den_idx, K)
        b_deltas = []
        for k in range(K):
            m1 = np.zeros(len(md), dtype=bool)
            m0 = np.zeros(len(md), dtype=bool)
            m1[num_splits[k]] = True
            m0[den_splits[k]] = True
            p1, _, _, s1 = _pooled_stats(m1)
            p0, _, _, s0 = _pooled_stats(m0)
            ok_b = (s1 >= 1.0) & (s0 >= 1.0) & np.isfinite(p1) & np.isfinite(p0)
            b_deltas.append(np.where(ok_b, p1 - p0, np.nan))
        b_mat = np.stack(b_deltas, axis=0)
        n_b = np.isfinite(b_mat).sum(axis=0)
        bm = np.divide(np.nansum(b_mat, axis=0), np.maximum(n_b, 1), out=np.full(u.shape[1], np.nan), where=n_b >= 2)
        var_between = np.divide(
            np.nansum((b_mat - bm[None, :]) ** 2, axis=0),
            np.maximum(n_b - 1, 1) * np.maximum(n_b, 1),
            out=np.zeros(u.shape[1], dtype=float),
            where=n_b >= 2,
        )

    if inference_unit in ("paired_donor_population", "paired_donor_small_n") and paired_nc_tot is not None:
        # Strictly sign-flip-invariant pooled paired-donor cell and read-support floors
        pooled_ok = (
            (paired_nc_tot >= 2 * int(min_cells_per_group))
            & (paired_sup_tot >= 2.0 * float(min_support_per_group))
        )
    elif inference_unit == "partially_paired_disjoint_unpaired" and unpaired_nc_num is not None:
        pooled_ok = (
            (unpaired_nc_num >= int(min_cells_per_group))
            & (unpaired_nc_den >= int(min_cells_per_group))
            & (unpaired_sup_num >= float(min_support_per_group))
            & (unpaired_sup_den >= float(min_support_per_group))
        )
    else:
        pooled_ok = (
            (nc_num >= int(min_cells_per_group))
            & (nc_den >= int(min_cells_per_group))
            & (sup_num >= float(min_support_per_group))
            & (sup_den >= float(min_support_per_group))
            & np.isfinite(psi_num)
            & np.isfinite(psi_den)
        )
    eligible = pooled_ok & donor_eligible
    # Strictly use donor-level means/midpoints and within-donor variance across ALL multi-donor branches (zero cell-pooled leakage)
    if inference_unit == "insufficient_multi_donor_design":
        raw_dpsi = np.full(u.shape[1], np.nan, dtype=float)
        baseline_psi = np.full(u.shape[1], np.nan, dtype=float)
        psi_num_out = np.full(u.shape[1], np.nan, dtype=float)
        psi_den_out = np.full(u.shape[1], np.nan, dtype=float)
        within_var_use = np.full(u.shape[1], 0.05, dtype=float)
    elif donor_mean_d is not None and donor_base_d is not None:
        raw_dpsi = np.where(eligible & np.isfinite(donor_mean_d), donor_mean_d, np.nan)
        baseline_psi = np.where(eligible & np.isfinite(donor_base_d), donor_base_d, np.nan)
        eligible = eligible & np.isfinite(raw_dpsi)
        psi_num_out = np.where(eligible & np.isfinite(donor_arm1_d), donor_arm1_d, psi_num)
        psi_den_out = np.where(eligible & np.isfinite(donor_arm0_d), donor_arm0_d, psi_den)
        within_var_use = np.where(eligible & np.isfinite(donor_within_var), donor_within_var, 0.05)
    else:
        raw_dpsi = np.where(eligible, psi_num - psi_den, np.nan)
        baseline_psi = np.where(eligible, 0.5 * (psi_num + psi_den), np.nan)
        psi_num_out = psi_num
        psi_den_out = psi_den
        within_var_use = np.nan_to_num(se2_num + se2_den, nan=0.05)
    se2_tot = np.where(eligible, np.maximum(within_var_use, np.nan_to_num(var_between, nan=0.0)) + 0.0016, np.nan)
    se_tot = np.sqrt(se2_tot)
    z_shift = np.where(eligible, raw_dpsi / se_tot, np.nan)
    # Two-sided donor-aware event P-value & BH FDR strictly across donor-eligible events
    p_event = np.full(u.shape[1], np.nan, dtype=float)
    use_t = eligible & np.isfinite(df_event) & (df_event >= 1.0)
    use_z = eligible & (~use_t)
    if np.any(use_t):
        p_event[use_t] = np.clip(2.0 * student_t.sf(np.abs(z_shift[use_t]), df=df_event[use_t]), 1e-300, 1.0)
    if np.any(use_z):
        p_event[use_z] = np.clip(2.0 * norm.sf(np.abs(z_shift[use_z])), 1e-300, 1.0)
    fdr_event = _bh_fdr_mask(p_event, eligible)
    # Empirical-Bayes shrunken dPSI for downstream locus tracing
    tau2_dpsi = max(float(np.nanmean(np.maximum(raw_dpsi[eligible] ** 2 - se2_tot[eligible], 0.0))) if np.any(eligible) else 0.005, 0.001)
    shrunk_dpsi = np.where(eligible, raw_dpsi * (tau2_dpsi / (tau2_dpsi + se2_tot)), np.nan)

    insufficient_label = (
        "insufficient_paired_donors"
        if inference_unit in ("paired_donor_population", "paired_donor_small_n")
        else ("insufficient_multi_donor_design" if inference_unit == "insufficient_multi_donor_design" else "insufficient_group_donors")
    )
    diag_cell_ok = (
        pooled_ok
        | (
            (inference_unit in ("paired_donor_population", "paired_donor_small_n"))
            & ((nc_num + nc_den) >= 2 * int(min_cells_per_group))
            & ((sup_num + sup_den) >= 2.0 * float(min_support_per_group))
        )
    )
    ev_inf_unit = np.where(
        eligible,
        inference_unit,
        np.where(
            inference_unit == "insufficient_multi_donor_design",
            "insufficient_multi_donor_design",
            np.where(diag_cell_ok & (~donor_eligible), insufficient_label, "insufficient_cells_or_support"),
        ),
    )

    genes = event_genes.reindex(u_df.columns).astype(str).to_numpy()
    contrast_df = pd.DataFrame({
        "event_id": u_df.columns.astype(str),
        "gene_id": genes,
        "eligible": eligible,
        "event_inference_unit": ev_inf_unit,
        "n_paired_donors": n_paired_ev,
        "n_donors_numerator": n_donors_num_ev,
        "n_donors_denominator": n_donors_den_ev,
        "psi_numerator": psi_num_out,
        "psi_denominator": psi_den_out,
        "baseline_PSI": baseline_psi,
        "raw_dpsi": raw_dpsi,
        "dpsi": shrunk_dpsi,
        "se_dpsi": np.where(eligible, se_tot, np.nan),
        "z_shift": z_shift,
        "p_value": p_event,
        "fdr": fdr_event,
        "cells_numerator": nc_num.astype(int),
        "cells_denominator": nc_den.astype(int),
        "support_numerator": sup_num,
        "support_denominator": sup_den,
    })
    info = {
        "inference_unit": inference_unit,
        "paired_donors": len(paired_donors),
        "donors_numerator": len(donors_num),
        "donors_denominator": len(donors_den),
        "min_paired_donors": min_pd if inference_unit in ("paired_donor_population", "paired_donor_small_n") else 0,
        "min_unpaired_donors_per_group": min_ud if inference_unit in ("unpaired_donor_pseudobulk", "partially_paired_disjoint_unpaired") else 0,
        "events_excluded_donor_insufficiency": int((pooled_ok & (~donor_eligible)).sum()),
        "eligible_events": int(eligible.sum()),
        "eligible_genes": int(len(set(genes[eligible]))),
        **extra_info,
    }
    return contrast_df, donor_delta_df, info


def _build_nuisance_design(
    cov_df: pd.DataFrame,
    baseline_psi: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """Build Freedman-Lane nuisance design [1, exon_GC, flank_GC, (dist_GC), (cpg_rate), log_length, baseline_PSI, psi*(1-psi), event_type]."""
    sub = cov_df.loc[eligible]
    b_psi = np.nan_to_num(baseline_psi[eligible], nan=0.5)
    fgc = sub["flank_GC"].to_numpy(dtype=float)
    cols = [
        np.ones(len(sub), dtype=float),
        sub["exon_GC"].to_numpy(dtype=float),
        fgc,
    ]
    if "dist_GC" in sub.columns:
        dgc = sub["dist_GC"].to_numpy(dtype=float)
        if np.isfinite(dgc).all() and np.std(dgc - fgc) > 1e-6:
            cols.append(dgc)
    if "cpg_rate" in sub.columns:
        cpg = sub["cpg_rate"].to_numpy(dtype=float)
        if np.isfinite(cpg).all() and np.std(cpg) > 1e-8:
            cols.append(cpg)
    cols.extend([
        sub["log_length"].to_numpy(dtype=float),
        b_psi,
        b_psi * (1.0 - b_psi),
    ])
    etypes = sub["event_type"].astype(str).to_numpy()
    uniq_et = sorted(set(etypes))
    for et in uniq_et[1:]:
        cols.append((etypes == et).astype(float))
    X = np.column_stack(cols)
    # Guard against NaN/Inf in external sequence covariate tables by column-median / zero imputation
    for col_i in range(1, X.shape[1]):
        col = X[:, col_i]
        bad = ~np.isfinite(col)
        if np.any(bad):
            med = float(np.median(col[~bad])) if np.any(~bad) else 0.0
            X[bad, col_i] = med
    return X


def _weighted_fwl_residuals(Y: np.ndarray, X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted Frisch-Waugh-Lovell residualization for 1D or 2D Y."""
    w_root = np.sqrt(np.maximum(weights, 1e-12))
    Xw = X * w_root[:, None]
    if Y.ndim == 1:
        yw = Y * w_root
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
        return Y - X @ beta
    Yw = Y * w_root[:, None]
    beta, _, _, _ = np.linalg.lstsq(Xw, Yw, rcond=None)
    return Y - X @ beta


def discover_sc_rbp_programs(
    contrast_df: pd.DataFrame,
    rbp_tensor: np.ndarray,
    rbps: np.ndarray,
    region_names: list[str],
    covariates_df: pd.DataFrame,
    *,
    donor_delta_df: pd.DataFrame | None = None,
    mature_consequences_df: pd.DataFrame | None = None,
    rbp_expr_z: pd.Series | None = None,
    n_permutations: int = 4999,
    seed: int = 20261117,
    deconfound_correlated_motifs: bool = True,
) -> SCSpliceTraceContrastResult:
    """Discover candidate RBPs and region-specific motif mechanisms from single-cell splicing contrasts.

    - Dual statistical calibration:
      1. When `donor_delta_df` has >= 5 effective paired donors (`paired_donor_population`), computes exact or
         randomized variance-moderated donor sign-flip max-T FWER + BH-FDR (valid under an independent
         sign-symmetric donor-effect null centered at zero, where `D` equals the effective number of contributing
         paired donors per tested cohort) with gene-block cross-event Z tie-breaking and optional ridge
         de-spillover of correlated bystander motifs.
      2. When < 5 donors exist (`paired_donor_small_n`, `unpaired_donor_pseudobulk`, `single_donor_gene_block`),
         runs 5-covariate Freedman-Lane orthogonalized Self-Normalized Rademacher Gene-Block Westfall-Young
         Max-T FWER + BH-FDR across `{0..B}` draws.
    """
    elig = contrast_df["eligible"].to_numpy(dtype=bool) & np.isfinite(contrast_df["z_shift"].to_numpy(dtype=float))
    n_elig = int(elig.sum())
    if n_elig < 25:
        empty_rank = pd.DataFrame(columns=["RBP", "best_region", "direction", "stat_Z", "nominal_P", "rbp_FDR_BH", "rbp_P_FWER"])
        empty_reg = pd.DataFrame(columns=["RBP", "region", "stat_Z", "nominal_P", "region_FDR_BH", "region_P_FWER"])
        empty_conseq = pd.DataFrame(columns=["event_id", "gene_symbol", "RBP", "hit_RNA", "consequence_direction"])
        return SCSpliceTraceContrastResult(
            rbp_ranking=empty_rank,
            region_tests=empty_reg,
            event_contrasts=contrast_df,
            mature_consequences=mature_consequences_df if (mature_consequences_df is not None and len(mature_consequences_df.columns) > 0) else empty_conseq,
            diagnostics={"status": "insufficient_eligible_events", "eligible_events": n_elig},
        )

    ev_sub = contrast_df.loc[elig].reset_index(drop=True)
    genes_sub = ev_sub["gene_id"].astype(str).to_numpy()
    uniq_genes, gene_codes = np.unique(genes_sub, return_inverse=True)
    n_genes = len(uniq_genes)
    gene_mult = np.bincount(gene_codes)[gene_codes].astype(float)
    ev_weights = 1.0 / gene_mult
    ev_weights = ev_weights / ev_weights.sum()

    cov_aligned = covariates_df.set_index("event_id").reindex(contrast_df["event_id"].astype(str)).reset_index()
    X_nuis = _build_nuisance_design(cov_aligned, contrast_df["baseline_PSI"].to_numpy(dtype=float), elig)

    # Winsorize extreme z_shift lightly at +-8 to prevent a single ultra-deep outlier gene from dominating
    z_raw = np.clip(ev_sub["z_shift"].to_numpy(dtype=float), -8.0, 8.0)
    z_res = _weighted_fwl_residuals(z_raw, X_nuis, ev_weights)

    n_reg = len(region_names)
    n_rbp = len(rbps)
    F_sub = rbp_tensor[elig]  # n_elig x n_reg x n_rbp
    F_flat = F_sub.reshape(n_elig, n_reg * n_rbp)
    F_res = _weighted_fwl_residuals(F_flat, X_nuis, ev_weights)
    # Standardize residualized motif features with gene-equal weights
    f_var = np.sum((F_res * F_res) * ev_weights[:, None], axis=0)
    f_sd = np.sqrt(np.maximum(f_var, 1e-12))
    F_std = np.where(f_var[None, :] > 1e-11, F_res / f_sd[None, :], 0.0)

    # Gene-block self-normalized scores S_{g, c} for c in (n_reg * n_rbp)
    contrib = (ev_weights * z_res)[:, None] * F_std  # n_elig x C
    S_gene = np.zeros((n_genes, n_reg * n_rbp), dtype=float)
    np.add.at(S_gene, gene_codes, contrib)
    denom_gene = np.sqrt(np.sum(S_gene * S_gene, axis=0))
    valid_col = (f_var > 1e-11) & (denom_gene > 1e-12)
    Z_gene_obs = np.divide(S_gene.sum(axis=0), denom_gene, out=np.zeros(n_reg * n_rbp), where=valid_col)

    use_donor_pop = donor_delta_df is not None and len(donor_delta_df) >= 5
    rng = np.random.default_rng(int(seed))

    if use_donor_pop:
        inference_mode = "paired_donor_moderated_maxT"
        d_sub = donor_delta_df.reindex(columns=contrast_df.loc[elig, "event_id"].astype(str)).to_numpy(dtype=float)  # D x n_elig
        n_donors = d_sub.shape[0]
        # Compute nuisance-adjusted slope per donor x (region, RBP)
        d_filled = np.where(np.isfinite(d_sub), d_sub, 0.0)
        d_res = _weighted_fwl_residuals(d_filled.T, X_nuis, ev_weights).T  # D x n_elig

        if deconfound_correlated_motifs and n_reg * n_rbp > 1:
            # Ridge-decorrelated donor slopes suppress collinear bystander spillover while preserving sign symmetry!
            G_cov = (F_std.T * ev_weights) @ F_std
            ridge_lam = 0.35 * float(np.median(np.diag(G_cov)[valid_col])) if np.any(valid_col) else 0.35
            rhs_d = (d_res * ev_weights[None, :]) @ F_std  # D x C
            B_donor = np.linalg.solve(G_cov + max(ridge_lam, 0.05) * np.eye(G_cov.shape[0]), rhs_d.T).T
            B_donor[:, ~valid_col] = 0.0
        else:
            B_donor = (d_res * ev_weights[None, :]) @ F_std

        if n_donors <= 14:
            signs = np.array(list(product([-1.0, 1.0], repeat=n_donors)), dtype=float)
            exact_signs = True
        else:
            signs = rng.choice([-1.0, 1.0], size=(int(n_permutations), n_donors))
            exact_signs = False

        # Variance-moderated donor t-denominator (invariant to donor sign flips b_d -> +-b_d!)
        # Note: sum_d b_d^2 IS strictly invariant under arbitrary sign flips s_d in {-1, +1}!
        ss_b = np.sum(B_donor * B_donor, axis=0)
        pos_ss = ss_b[valid_col & (ss_b > 0)]
        s0_sq = float(np.quantile(pos_ss, 0.25)) * 0.15 if len(pos_ss) else 1e-8
        mod_denom = np.sqrt(ss_b + s0_sq)
        Z_obs = np.divide(B_donor.sum(axis=0), mod_denom, out=np.zeros(n_reg * n_rbp), where=valid_col & (mod_denom > 0))
        null_Z = np.divide(signs @ B_donor, mod_denom[None, :], out=np.zeros((len(signs), n_reg * n_rbp)), where=valid_col[None, :] & (mod_denom[None, :] > 0))

        # Per-RBP max(|Z|) across its regions for exact subset-compatible familywise calibration
        Z_obs_2d = Z_obs.reshape(n_reg, n_rbp)
        null_3d = null_Z.reshape(len(signs), n_reg, n_rbp)
        rbp_obs_max = np.max(np.abs(Z_obs_2d), axis=0)
        rbp_null_max = np.max(np.abs(null_3d), axis=1)  # n_signs x n_rbp
        global_null_max = np.max(rbp_null_max, axis=1) if n_rbp > 0 else np.zeros(len(signs))

        corr_add = 0.0 if exact_signs else 1.0
        denom_p = float(len(signs)) + corr_add
        nom_p = np.ones(n_reg * n_rbp, dtype=float)
        reg_fwer = np.ones(n_reg * n_rbp, dtype=float)
        for c in np.flatnonzero(valid_col):
            thr = abs(Z_obs[c]) - 1e-12
            nom_p[c] = (corr_add + float(np.sum(np.abs(null_Z[:, c]) >= thr))) / denom_p
            reg_fwer[c] = (corr_add + float(np.sum(global_null_max >= thr))) / denom_p

        rbp_nom_p = np.ones(n_rbp, dtype=float)
        rbp_fwer_p = np.ones(n_rbp, dtype=float)
        for r_i in range(n_rbp):
            thr_r = rbp_obs_max[r_i] - 1e-12
            if thr_r > 0:
                rbp_nom_p[r_i] = (corr_add + float(np.sum(rbp_null_max[:, r_i] >= thr_r))) / denom_p
                rbp_fwer_p[r_i] = (corr_add + float(np.sum(global_null_max >= thr_r))) / denom_p

    else:
        inference_mode = "gene_block_rademacher_maxT"
        B = int(n_permutations)
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(B, n_genes))
        S_norm = np.divide(S_gene, denom_gene[None, :], out=np.zeros_like(S_gene), where=valid_col[None, :])
        Z_obs = Z_gene_obs.copy()
        null_Z = signs @ np.ascontiguousarray(S_norm, dtype=np.float64)

        Z_obs_2d = Z_obs.reshape(n_reg, n_rbp)
        null_3d = null_Z.reshape(B, n_reg, n_rbp)
        rbp_obs_max = np.max(np.abs(Z_obs_2d), axis=0)
        rbp_null_max = np.max(np.abs(null_3d), axis=1)
        global_null_max = np.max(rbp_null_max, axis=1) if n_rbp > 0 else np.zeros(B)

        nom_p = (1.0 + np.sum(np.abs(null_Z) >= np.abs(Z_obs)[None, :] - 1e-12, axis=0)) / (B + 1.0)
        reg_fwer = (1.0 + np.sum(global_null_max[:, None] >= np.abs(Z_obs)[None, :] - 1e-12, axis=0)) / (B + 1.0)
        nom_p[~valid_col] = 1.0
        reg_fwer[~valid_col] = 1.0

        rbp_nom_p = (1.0 + np.sum(rbp_null_max >= rbp_obs_max[None, :] - 1e-12, axis=0)) / (B + 1.0)
        rbp_fwer_p = (1.0 + np.sum(global_null_max[:, None] >= rbp_obs_max[None, :] - 1e-12, axis=0)) / (B + 1.0)

    # Assemble region-level and RBP-level tables (with BH-FDR alongside strict Max-T FWER)
    Z_gene_2d = Z_gene_obs.reshape(n_reg, n_rbp)
    Z_obs_2d = Z_obs.reshape(n_reg, n_rbp)
    nom_2d = nom_p.reshape(n_reg, n_rbp)
    fwer_2d = reg_fwer.reshape(n_reg, n_rbp)
    reg_fdr_bh = np.where(valid_col, _bh_fdr_mask(nom_p, valid_col), 1.0)
    reg_fdr_2d = reg_fdr_bh.reshape(n_reg, n_rbp)
    rbp_fdr_bh = _bh_fdr_mask(rbp_nom_p, np.ones(n_rbp, dtype=bool))
    rbp_fdr_bh = np.where(np.isfinite(rbp_fdr_bh), rbp_fdr_bh, 1.0)

    reg_rows: list[dict] = []
    for r_idx, reg_name in enumerate(region_names):
        for m_idx, rbp_name in enumerate(rbps):
            zv = float(Z_obs_2d[r_idx, m_idx])
            reg_rows.append({
                "RBP": str(rbp_name),
                "region": str(reg_name),
                "layer": "affected_sequence" if reg_name == "mature_consequence_delta" else "regulatory_pre_mRNA",
                "direction": "positive" if zv >= 0 else "negative",
                "stat_Z": zv,
                "gene_block_Z": float(Z_gene_2d[r_idx, m_idx]),
                "nominal_P": float(nom_2d[r_idx, m_idx]),
                "region_FDR_BH": float(reg_fdr_2d[r_idx, m_idx]),
                "region_P_FWER": float(fwer_2d[r_idx, m_idx]),
            })
    reg_df = pd.DataFrame(reg_rows)

    rbp_rows: list[dict] = []
    for m_idx, rbp_name in enumerate(rbps):
        best_r = int(np.argmax(np.abs(Z_obs_2d[:, m_idx])))
        zv = float(Z_obs_2d[best_r, m_idx])
        gzv = float(Z_gene_2d[best_r, m_idx])
        mature_z = float(Z_obs_2d[0, m_idx]) if region_names[0] == "mature_consequence_delta" else np.nan
        mature_p = float(fwer_2d[0, m_idx]) if region_names[0] == "mature_consequence_delta" else np.nan
        # Continuous tie-free evidence score: analytical tail survival + gene-block Z tie-breaker
        seq_score = float(-norm.logsf(abs(zv)) / np.log(10.0) + 0.02 * abs(gzv))
        expr_z = float(rbp_expr_z.get(str(rbp_name), np.nan)) if rbp_expr_z is not None else np.nan
        concordant_score = seq_score + (0.35 * min(abs(expr_z), 8.0) if np.isfinite(expr_z) else 0.0)
        rbp_rows.append({
            "RBP": str(rbp_name),
            "best_region": str(region_names[best_r]),
            "best_layer": "affected_sequence" if region_names[best_r] == "mature_consequence_delta" else "regulatory_pre_mRNA",
            "direction": "inclusion_promoted" if zv >= 0 else "skipping_promoted",
            "stat_Z": zv,
            "abs_Z": abs(zv),
            "gene_block_Z": gzv,
            "mature_consequence_Z": mature_z,
            "mature_consequence_P_FWER": mature_p,
            "nominal_P": float(rbp_nom_p[m_idx]),
            "rbp_FDR_BH": float(rbp_fdr_bh[m_idx]),
            "rbp_P_FWER": float(rbp_fwer_p[m_idx]),
            "sequence_evidence_score": seq_score,
            "rbp_expr_z": expr_z,
            "concordant_activity_score": concordant_score,
        })

    rbp_df = pd.DataFrame(rbp_rows).sort_values(
        by=["rbp_P_FWER", "rbp_FDR_BH", "sequence_evidence_score", "RBP"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    rbp_df["sequence_rank"] = np.arange(1, len(rbp_df) + 1, dtype=int)

    empty_conseq = pd.DataFrame(columns=["event_id", "gene_symbol", "RBP", "hit_RNA", "consequence_direction"])
    return SCSpliceTraceContrastResult(
        rbp_ranking=rbp_df,
        region_tests=reg_df,
        event_contrasts=contrast_df,
        mature_consequences=mature_consequences_df if (mature_consequences_df is not None and len(mature_consequences_df.columns) > 0) else empty_conseq,
        diagnostics={
            "status": "complete",
            "inference_mode": inference_mode,
            "eligible_events": n_elig,
            "eligible_genes": n_genes,
            "evaluated_rbps": int(n_rbp),
            "significant_rbps_fwer05": int((rbp_df["rbp_P_FWER"] <= 0.05).sum()),
            "significant_rbps_fdr05": int((rbp_df["rbp_FDR_BH"] <= 0.05).sum()),
            "significant_rbps_nom05": int((rbp_df["nominal_P"] <= 0.05).sum()),
        },
    )


def compute_sc_stratified_meta_contrast(
    usage: pd.DataFrame,
    metadata: pd.DataFrame,
    event_genes: pd.Series,
    *,
    support: pd.DataFrame | None = None,
    group_col: str = "condition",
    numerator_group: str,
    denominator_group: str,
    donor_col: str = "donor_id",
    min_cells_per_donor_group: int = 4,
    min_support_per_donor_group: float = 6.0,
    design_mode: str = "auto",
    seed: int = 20261117,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    """2-4 Donor Stratified Fixed-Effect Meta-Analysis Contrast Engine.

    Eliminates between-donor baseline batch offsets by estimating precision-weighted `(dpsi_d, se_d)`
    STRICTLY WITHIN each donor first, then combining across donors via inverse-variance meta-analysis:
        w_{de} = 1 / se_{de}^2,   dpsi_meta_e = sum_d(w_{de} dpsi_{de}) / sum_d(w_{de}),
        z_meta_e = sum_d(w_{de} dpsi_{de}) / sqrt(sum_d(w_{de})).
    Also returns the per-donor contrast dictionary so downstream discovery can enforce directional sign concordance!
    """
    md = metadata.copy()
    if "qc_pass" in md.columns:
        md = md.loc[md["qc_pass"].astype(str).str.lower().isin(("true", "1", "yes", "pass"))].copy()
    if donor_col not in md.columns and "donor" in md.columns:
        md[donor_col] = md["donor"]
    if donor_col in md.columns:
        valid_dn = _valid_donor_mask(md[donor_col])
        md = md.loc[valid_dn].copy()
        if md.empty:
            raise ValueError("Zero cells remain after excluding missing or blank donor IDs")
        md[donor_col] = md[donor_col].astype(str).str.strip()
    donors = sorted(md[donor_col].astype(str).unique())

    per_donor: dict[str, pd.DataFrame] = {}
    for d_idx, d_id in enumerate(donors):
        md_d = md.loc[md[donor_col].astype(str).eq(d_id)].copy()
        if not ({str(numerator_group), str(denominator_group)} <= set(md_d[group_col].astype(str))):
            continue
        c_d, _, _ = compute_sc_splicing_contrast(
            usage,
            md_d,
            event_genes,
            support=support,
            group_col=group_col,
            numerator_group=numerator_group,
            denominator_group=denominator_group,
            donor_col=donor_col,
            min_cells_per_group=min_cells_per_donor_group,
            min_support_per_group=min_support_per_donor_group,
            design_mode=design_mode,
            seed=int(seed) + 31 * d_idx,
        )
        per_donor[d_id] = c_d

    if not per_donor:
        raise ValueError("Zero donors have both numerator and denominator cells")

    ev_ids = usage.columns.astype(str).to_numpy()
    genes = event_genes.reindex(usage.columns).astype(str).to_numpy()
    n_ev = len(ev_ids)
    w_sum = np.zeros(n_ev, dtype=float)
    wd_sum = np.zeros(n_ev, dtype=float)
    wb_sum = np.zeros(n_ev, dtype=float)
    wp1_sum = np.zeros(n_ev, dtype=float)
    wp0_sum = np.zeros(n_ev, dtype=float)
    d_cnt = np.zeros(n_ev, dtype=int)
    c_num_tot = np.zeros(n_ev, dtype=int)
    c_den_tot = np.zeros(n_ev, dtype=int)
    s_num_tot = np.zeros(n_ev, dtype=float)
    s_den_tot = np.zeros(n_ev, dtype=float)

    pos_d = np.zeros(n_ev, dtype=int)
    neg_d = np.zeros(n_ev, dtype=int)
    for c_d in per_donor.values():
        el = c_d["eligible"].to_numpy(dtype=bool)
        se = c_d["se_dpsi"].to_numpy(dtype=float)
        dp = c_d["raw_dpsi"].to_numpy(dtype=float)
        bp = c_d["baseline_PSI"].to_numpy(dtype=float)
        p1 = c_d["psi_numerator"].to_numpy(dtype=float)
        p0 = c_d["psi_denominator"].to_numpy(dtype=float)
        ok = el & np.isfinite(se) & (se > 0) & np.isfinite(dp)
        w = np.where(ok, 1.0 / (se * se), 0.0)
        w_sum += w
        wd_sum += np.where(ok, w * dp, 0.0)
        wb_sum += np.where(ok & np.isfinite(bp), w * bp, 0.0)
        wp1_sum += np.where(ok & np.isfinite(p1), w * p1, 0.0)
        wp0_sum += np.where(ok & np.isfinite(p0), w * p0, 0.0)
        d_cnt += ok.astype(int)
        pos_d += (ok & (dp > 0)).astype(int)
        neg_d += (ok & (dp < 0)).astype(int)
        c_num_tot += c_d["cells_numerator"].to_numpy(dtype=int)
        c_den_tot += c_d["cells_denominator"].to_numpy(dtype=int)
        s_num_tot += c_d["support_numerator"].to_numpy(dtype=float)
        s_den_tot += c_d["support_denominator"].to_numpy(dtype=float)

    elig_meta = w_sum > 0.0
    sign_agree = np.divide(np.maximum(pos_d, neg_d), np.maximum(d_cnt, 1), out=np.zeros(n_ev, dtype=float), where=d_cnt > 0)
    raw_meta = np.divide(wd_sum, w_sum, out=np.full(n_ev, np.nan), where=elig_meta)
    base_meta = np.divide(wb_sum, w_sum, out=np.full(n_ev, 0.5), where=elig_meta)
    psi1_meta = np.divide(wp1_sum, w_sum, out=np.full(n_ev, np.nan), where=elig_meta)
    psi0_meta = np.divide(wp0_sum, w_sum, out=np.full(n_ev, np.nan), where=elig_meta)
    se_meta = np.divide(1.0, np.sqrt(np.maximum(w_sum, 1e-12)), out=np.full(n_ev, np.nan), where=elig_meta)
    z_meta = np.divide(wd_sum, np.sqrt(np.maximum(w_sum, 1e-12)), out=np.full(n_ev, np.nan), where=elig_meta)
    p_meta = np.where(elig_meta & np.isfinite(z_meta), np.clip(2.0 * norm.sf(np.abs(z_meta)), 1e-300, 1.0), np.nan)
    fdr_meta = _bh_fdr_mask(p_meta, elig_meta)
    se2_m = np.where(elig_meta, se_meta * se_meta, 0.05)
    tau2_m = max(float(np.nanmean(np.maximum(raw_meta[elig_meta] ** 2 - se2_m[elig_meta], 0.0))) if np.any(elig_meta) else 0.005, 0.001)
    shrunk_meta = np.where(elig_meta, raw_meta * (tau2_m / (tau2_m + se2_m)), np.nan)

    meta_df = pd.DataFrame({
        "event_id": ev_ids,
        "gene_id": genes,
        "eligible": elig_meta,
        "contributing_donors": d_cnt,
        "donor_sign_agreement": sign_agree,
        "psi_numerator": psi1_meta,
        "psi_denominator": psi0_meta,
        "baseline_PSI": base_meta,
        "raw_dpsi": raw_meta,
        "dpsi": shrunk_meta,
        "se_dpsi": se_meta,
        "z_shift": z_meta,
        "p_value": p_meta,
        "fdr": fdr_meta,
        "cells_numerator": c_num_tot,
        "cells_denominator": c_den_tot,
        "support_numerator": s_num_tot,
        "support_denominator": s_den_tot,
    })
    return meta_df, per_donor, {
        "inference_unit": "stratified_multi_donor_meta",
        "n_donors_meta": len(per_donor),
        "donors_evaluated": list(per_donor.keys()),
        "eligible_events": int(elig_meta.sum()),
        "multi_donor_events": int((d_cnt >= 2).sum()),
    }
