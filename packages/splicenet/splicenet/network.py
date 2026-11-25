"""Pure-Python exact Tukey bicor + single-block Signed TOM + dynamic module discovery (plus optional R WGCNA)."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import average, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score


def compute_tukey_bicor_matrix(
    data_events_by_samples: np.ndarray,
    *,
    max_p_outliers: float = 0.10,
) -> np.ndarray:
    """Vectorized Langfelder & Horvath (2012) subgroup-safe Tukey biweight midcorrelation (`bicor`).

    Protects genuine biological replicate subgroups (`>=2` replicates in `N>=12` cohorts) by anchoring
    the tail reference quantile at `u_ref = 0.65`, while single-sample technical spikes (`|u| >= 1`)
    receive zero weight.
    """
    X = np.asarray(data_events_by_samples, dtype=float)
    n_ev, n_samp = X.shape
    med = np.median(X, axis=1, keepdims=True)
    diff = X - med
    mad = np.median(np.abs(diff), axis=1, keepdims=True)
    p_tail = min(0.25, max(max_p_outliers / 2.0, 1.25 / max(n_samp, 4)))
    q_lo = np.quantile(X, p_tail, axis=1, keepdims=True)
    q_hi = np.quantile(X, 1.0 - p_tail, axis=1, keepdims=True)

    scale_neg = np.maximum(9.0 * mad, (med - q_lo) / 0.65)
    scale_pos = np.maximum(9.0 * mad, (q_hi - med) / 0.65)
    denom = np.where(diff < 0.0, scale_neg, scale_pos)
    u = np.divide(diff, denom, out=np.zeros_like(diff), where=denom > 1e-12)
    w = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)
    xw = diff * w
    nrm = np.linalg.norm(xw, axis=1, keepdims=True)
    z = np.divide(xw, nrm, out=np.zeros_like(xw), where=nrm > 1e-12)
    bc = np.clip(z @ z.T, -1.0, 1.0)
    np.fill_diagonal(bc, 1.0)
    return bc




def pick_soft_threshold(bc: np.ndarray, candidate_powers: tuple[int, ...] = (6, 8, 10, 12, 14), r2_cutoff: float = 0.80) -> int:
    """Select scale-free topology signed soft-threshold power (Langfelder & Horvath criterion)."""
    s_base = np.clip((1.0 + bc) * 0.5, 0.0, 1.0)
    np.fill_diagonal(s_base, 0.0)
    best_p, best_r2 = candidate_powers[0], -1.0
    for p in candidate_powers:
        k = np.sum(np.power(s_base, p), axis=1)
        k_pos = k[k > 1e-9]
        if len(k_pos) < 30:
            continue
        hist, edges = np.histogram(np.log10(k_pos), bins=12)
        nz = hist > 0
        if int(nz.sum()) < 4:
            continue
        x = 0.5 * (edges[:-1][nz] + edges[1:][nz])
        # Jacobian-corrected density p(k) = hist / (N * delta_k) so slope directly estimates -gamma
        bin_w = np.maximum(np.power(10.0, edges[1:][nz]) - np.power(10.0, edges[:-1][nz]), 1e-12)
        y = np.log10((hist[nz] / float(len(k_pos))) / bin_w)
        if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
            continue
        slope, _ = np.polyfit(x, y, 1)
        if not np.isfinite(slope) or slope >= -1e-9:
            continue
        r2 = float(np.corrcoef(x, y)[0, 1] ** 2)
        if not np.isfinite(r2):
            continue
        if r2 >= r2_cutoff:
            return int(p)
        if r2 > best_r2:
            best_r2, best_p = r2, int(p)
    return int(best_p)


def compute_signed_tom_from_bicor(bc: np.ndarray, power: int = 6) -> np.ndarray:
    adj = np.power(np.clip((1.0 + bc) * 0.5, 0.0, 1.0), int(power))
    np.fill_diagonal(adj, 0.0)
    shared = adj @ adj
    k = adj.sum(axis=1)
    denom = np.minimum(k[:, None], k[None, :]) + 1.0 - adj
    tom = np.divide(shared + adj, denom, out=np.zeros_like(adj), where=denom > 1e-12)
    tom = np.clip(tom, 0.0, 1.0)
    np.fill_diagonal(tom, 1.0)
    return tom


def _modularity_score_sn(W: np.ndarray, labels: np.ndarray) -> float:
    tot = float(W.sum())
    if tot <= 1e-12:
        return 0.0
    k = W.sum(axis=1)
    q_val = 0.0
    for c in set(int(x) for x in labels if int(x) > 0):
        idx = labels == c
        e_in = float(W[np.ix_(idx, idx)].sum())
        k_c = float(k[idx].sum())
        q_val += (e_in / tot) - (k_c / tot) ** 2
    return q_val


def _is_balanced_multimod_sn(labels: np.ndarray, n: int, min_module_size: int) -> bool:
    pos = labels[labels > 0]
    if len(pos) == 0:
        return False
    vc = pd.Series(pos).value_counts()
    valid = vc[vc >= int(min_module_size)]
    if len(valid) >= 2 and int(valid.max()) <= max(2 * int(min_module_size), int(0.75 * n)):
        return True
    return len(valid) >= 1 and n < 3 * int(min_module_size)


def _cut_dendrogram_into_modules(dissim: np.ndarray, min_module_size: int = 20, deep_split: int = 2) -> np.ndarray:
    n = dissim.shape[0]
    if n < min_module_size:
        return np.zeros(n, dtype=int)
    d_clean = np.clip(0.5 * (dissim + dissim.T), 0.0, 1.0)
    np.fill_diagonal(d_clean, 0.0)
    Z = average(squareform(d_clean, checks=False))
    cand_lbl: np.ndarray | None = None
    try:
        from dynamicTreeCut import cutreeHybrid
        res = cutreeHybrid(Z, d_clean, minClusterSize=int(min_module_size), deepSplit=int(deep_split), pamRespectsDendro=False, verbose=0)
        labels = np.asarray(res["labels"] if isinstance(res, dict) else res, dtype=int)
        if len(labels) == n:
            cand_lbl = labels
    except Exception:
        cand_lbl = None

    if cand_lbl is not None and _is_balanced_multimod_sn(cand_lbl, n, min_module_size):
        return cand_lbl

    W = np.clip(1.0 - d_clean, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    try:
        import networkx as nx

        triu_i, triu_j = np.triu_indices(n, k=1)
        weights = W[triu_i, triu_j]
        pos_w = weights[weights > 1e-9]
        if len(pos_w) >= max(4, n):
            thr = float(np.quantile(pos_w, 0.70))
            keep_mask = weights > thr
            if int(keep_mask.sum()) < 2 * n:
                keep_mask = weights > 1e-9
            G = nx.Graph()
            G.add_nodes_from(range(n))
            for a, b_i, w_v in zip(triu_i[keep_mask], triu_j[keep_mask], weights[keep_mask]):
                G.add_edge(int(a), int(b_i), weight=float(w_v))
            comms = nx.community.louvain_communities(G, weight="weight", resolution=1.0, seed=20261117)
            big_comms = sorted([c for c in comms if len(c) >= int(min_module_size)], key=len, reverse=True)
            if len(big_comms) >= 2 and len(big_comms[0]) <= max(2 * int(min_module_size), int(0.75 * n)):
                mapped = np.zeros(n, dtype=int)
                for new_id, comm in enumerate(big_comms, start=1):
                    mapped[list(comm)] = new_id
                return mapped
    except Exception:
        pass

    best_labels = np.zeros(n, dtype=int)
    best_score = -1.0
    for q in np.linspace(0.15, 0.96, 28):
        h = float(np.quantile(Z[:, 2], q))
        raw = fcluster(Z, t=h, criterion="distance")
        vc = pd.Series(raw).value_counts()
        valid = vc[(vc >= int(min_module_size)) & (vc <= max(int(min_module_size), int(0.75 * n)))].index.tolist()
        if not valid:
            continue
        mapped = np.zeros(n, dtype=int)
        for new_id, cid in enumerate(valid, start=1):
            mapped[raw == cid] = new_id
        q_mod = max(_modularity_score_sn(W, mapped), 0.01)
        cov = float(np.mean(mapped > 0))
        sc = q_mod * float(np.sqrt(max(cov, 0.05))) * float(np.log1p(min(len(valid), 12)))
        if sc > best_score:
            best_score = sc
            best_labels = mapped
    if int(best_labels.max()) >= 2 or cand_lbl is None:
        return best_labels
    return cand_lbl


def _merge_close_modules_and_kme(
    X: np.ndarray,
    raw_labels: np.ndarray,
    merge_cut_height: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute module eigengenes (PC1), merge modules with `1 - cor(ME) < merge_cut_height`, and return `(module, kME)`."""
    xm = X.mean(axis=1, keepdims=True)
    xsd = np.maximum(X.std(axis=1, ddof=1, keepdims=True), 1e-8)
    Xz = (X - xm) / xsd

    def _eigengenes(lbls: np.ndarray) -> dict[int, np.ndarray]:
        mes: dict[int, np.ndarray] = {}
        for m in sorted(set(lbls) - {0}):
            sub = Xz[lbls == m]
            u, s, _ = np.linalg.svd(sub.T, full_matrices=False)
            me = u[:, 0] * s[0]
            sub_mean = sub.mean(axis=0)
            if float(np.std(me)) > 1e-12 and float(np.std(sub_mean)) > 1e-12:
                r_ori = float(np.corrcoef(me, sub_mean)[0, 1])
                if np.isfinite(r_ori) and r_ori < 0:
                    me = -me
            mes[int(m)] = me
        return mes

    labels = raw_labels.copy()
    while True:
        mes = _eigengenes(labels)
        m_ids = sorted(mes.keys())
        if len(m_ids) < 2:
            break
        me_mat = np.vstack([mes[m] for m in m_ids])
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = np.corrcoef(me_mat)
        diss = 1.0 - corr
        diss = np.where(np.isfinite(diss), diss, np.inf)
        np.fill_diagonal(diss, np.inf)
        i, j = np.unravel_index(int(np.argmin(diss)), diss.shape)
        if np.isfinite(diss[i, j]) and diss[i, j] < float(merge_cut_height):
            keep_m, drop_m = min(m_ids[i], m_ids[j]), max(m_ids[i], m_ids[j])
            labels[labels == drop_m] = keep_m
        else:
            break

    # Renumber modules 1..K ordered by descending size (exact WGCNA convention)
    vc = pd.Series(labels[labels > 0]).value_counts()
    remap = {int(old_m): new_m for new_m, old_m in enumerate(vc.index, start=1)}
    final_labels = np.array([remap.get(int(m), 0) for m in labels], dtype=int)
    final_mes = _eigengenes(final_labels)

    kme = np.full(len(X), np.nan, dtype=float)
    for idx in range(len(X)):
        m = int(final_labels[idx])
        if m != 0 and m in final_mes:
            me = final_mes[m]
            me_c = me - me.mean()
            xi_c = Xz[idx] - Xz[idx].mean()
            den = float(np.linalg.norm(xi_c) * np.linalg.norm(me_c))
            kme[idx] = float(xi_c @ me_c / den) if den > 1e-12 else 0.0
    return final_labels, kme


def fit_splicenet(
    selected_usage: pd.DataFrame,
    out_dir: Path | None = None,
    *,
    power: int = 6,
    min_module_size: int = 20,
    merge_cut_height: float = 0.25,
    deep_split: int = 2,
    single_block: bool = True,
    max_p_outliers: float = 0.10,
    seed: int = 31,
    return_matrices: bool = False,
    backend: str = "python",
    rscript: Path | str | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Fit single-block signed bicor TOM co-splicing network in pure Python (default) or via R WGCNA (`backend='r-wgcna'`)."""
    means = selected_usage.mean(axis=1)
    complete = selected_usage.T.fillna(means).T

    X = complete.to_numpy(dtype=float)
    if backend not in {"python", "r-wgcna"}:
        raise ValueError(f"Unsupported backend: {backend!r}")
    if backend == "r-wgcna":
        r_bin = str(rscript) if rscript else shutil.which("Rscript")
        if not r_bin:
            raise RuntimeError(
                "Requested backend='r-wgcna' but Rscript was not found on PATH. "
                "Install R with WGCNA or specify backend='python'."
            )
        eff_r_power = int(power)
        if eff_r_power <= 0:
            bc_tmp = compute_tukey_bicor_matrix(X, max_p_outliers=max_p_outliers)
            eff_r_power = pick_soft_threshold(bc_tmp)
        max_block = int(len(complete)) if single_block else 2000
        r_body = f"""
        set.seed({int(seed)})
        suppressPackageStartupMessages(library(WGCNA))
        args <- commandArgs(trailingOnly=TRUE)
        dat <- t(as.matrix(read.delim(args[1], row.names=1, check.names=FALSE)))
        net <- blockwiseModules(dat, power={eff_r_power}, corType='bicor', networkType='signed', TOMType='signed',
          minModuleSize={int(min_module_size)}, mergeCutHeight={float(merge_cut_height)}, numericLabels=TRUE,
          pamRespectsDendro=FALSE, deepSplit={int(deep_split)}, maxBlockSize={max_block}L, randomSeed={int(seed)},
          verbose=0, maxPOutliers={float(max_p_outliers)})
        km <- signedKME(dat, net$MEs, corFnc='cor', corOptions="use = 'p'")
        own <- rep(NA_real_, ncol(dat))
        for(i in seq_along(own)) if(net$colors[i] != 0) {{ cname <- paste0('kME', net$colors[i]); if(cname %in% colnames(km)) own[i] <- km[i, cname] }}
        write.table(data.frame(event_key=colnames(dat), module=as.integer(net$colors), block=as.integer(net$blocks), kME=own), args[2], sep='\t', quote=FALSE, row.names=FALSE)
        """
        with tempfile.TemporaryDirectory(prefix="kcs_r_") as tmp:
            tp = Path(tmp)
            complete.to_csv(tp / "in.tsv", sep="\t", index_label="event_key")
            (tp / "run.R").write_text(r_body, encoding="utf-8")
            proc = subprocess.run([r_bin, "--vanilla", str(tp / "run.R"), str(tp / "in.tsv"), str(tp / "out.tsv")], capture_output=True, text=True)
            if proc.returncode != 0 or not (tp / "out.tsv").is_file():
                err_msg = (proc.stderr or proc.stdout or "unknown R error").strip()[:400]
                raise RuntimeError(f"R WGCNA backend failed (exit {proc.returncode}): {err_msg}")
            df_r = pd.read_csv(tp / "out.tsv", sep="\t", dtype={"event_key": str})
            df_r["module"] = df_r["module"].astype(int)
            if out_dir is not None:
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                df_r.to_csv(Path(out_dir) / "assignments.tsv", sep="\t", index=False)
            if return_matrices:
                bc_r = compute_tukey_bicor_matrix(X, max_p_outliers=max_p_outliers)
                tom_r = compute_signed_tom_from_bicor(bc_r, power=eff_r_power)
                return df_r, bc_r, tom_r
            return df_r

    # Pure-Python exact execution (works offline everywhere without R!)
    bc = compute_tukey_bicor_matrix(X, max_p_outliers=max_p_outliers)
    eff_power = pick_soft_threshold(bc) if int(power) <= 0 else int(power)
    tom = compute_signed_tom_from_bicor(bc, power=eff_power)
    raw_mods = _cut_dendrogram_into_modules(1.0 - tom, min_module_size=min_module_size, deep_split=deep_split)
    final_mods, kme = _merge_close_modules_and_kme(X, raw_mods, merge_cut_height=merge_cut_height)
    df = pd.DataFrame({
        "event_key": complete.index.astype(str),
        "module": final_mods.astype(int),
        "block": 1,
        "kME": kme,
    })
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out_dir) / "assignments.tsv", sep="\t", index=False)
    if return_matrices:
        return df, bc, tom
    return df


def consensus_core_audit(
    raw_usage: pd.DataFrame,
    meta: pd.DataFrame,
    read_support: pd.DataFrame | None,
    base_assignments: pd.DataFrame,
    *,
    feature_cap: int = 500,
    ranking_metric: str = "excess_bio_variance",
    n_subsamples: int = 20,
    donor_keep_fraction: float = 0.85,
    power: int = 6,
    merge_cut_height: float = 0.25,
    min_module_size: int = 20,
    max_p_outliers: float = 0.10,
    backend: str = "python",
    seed: int = 20261116,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run end-to-end donor-block subsampling refits (repeating QC + selection + TOM) to score module stability."""
    from splicenet.qc_and_selection import select_events, unit_robust_audit

    rng = np.random.default_rng(seed)
    donors = np.array(sorted(meta["biological_sample_id"].astype(str).unique()))
    if len(donors) < 4:
        enriched = base_assignments.copy()
        enriched["coassignment_support"] = np.nan
        enriched["membership_tier"] = np.where(enriched["module"].eq(0), "unassigned_grey", "insufficient_donors_for_subsampling")
        return enriched, {
            "n_subsamples": 0,
            "median_subsample_ARI": np.nan,
            "median_coassignment_support": np.nan,
            "core_hub_events": 0,
        }
    n_keep = min(len(donors) - 1, max(3, int(round(donor_keep_fraction * len(donors)))))

    base = base_assignments.set_index("event_key").copy()
    keys = base.index.tolist()
    n_ev = len(keys)
    coassign_sum = np.zeros(n_ev, dtype=float)
    coassign_den = np.zeros(n_ev, dtype=float)
    aris_shared: list[float] = []

    for _ in range(n_subsamples):
        kept = set(rng.choice(donors, size=n_keep, replace=False).tolist())
        sub_meta = meta[meta["biological_sample_id"].astype(str).isin(kept)].reset_index(drop=True)
        cols = sub_meta["sample_id"].tolist()
        sup_sub = read_support.loc[:, cols] if read_support is not None else None
        masked, aud = unit_robust_audit(
            raw_usage.loc[:, cols], sub_meta, sup_sub,
            require_positive_excess_variance=(ranking_metric == "excess_bio_variance"),
        )
        sel, _ = select_events(masked, aud, feature_cap=feature_cap, ranking_metric=ranking_metric)
        if len(sel) < min_module_size:
            continue
        sub_assign = fit_splicenet(
            sel,
            power=power,
            min_module_size=min_module_size,
            merge_cut_height=merge_cut_height,
            max_p_outliers=max_p_outliers,
            backend=backend,
        )
        sub_map = sub_assign.set_index("event_key")["module"]
        shared = [i for i, k in enumerate(keys) if k in sub_map.index]
        if len(shared) >= min(20, min_module_size):
            b_vec = base["module"].to_numpy()[shared]
            s_vec = sub_map.loc[[keys[i] for i in shared]].to_numpy()
            aris_shared.append(float(adjusted_rand_score(b_vec, s_vec)))
            for pos_idx, glob_idx in enumerate(shared):
                mod_b = b_vec[pos_idx]
                if mod_b == 0:
                    continue
                mates = [p for p in range(len(shared)) if b_vec[p] == mod_b and p != pos_idx]
                if mates:
                    coassign_sum[glob_idx] += float(np.mean([s_vec[p] == s_vec[pos_idx] and s_vec[pos_idx] != 0 for p in mates]))
                    coassign_den[glob_idx] += 1.0

    support_vec = np.divide(coassign_sum, coassign_den, out=np.full(n_ev, np.nan), where=coassign_den > 0)
    enriched = base_assignments.copy()
    enriched["coassignment_support"] = support_vec
    enriched["membership_tier"] = np.where(
        enriched["module"].eq(0), "unassigned_grey",
        np.where((enriched["coassignment_support"] >= 0.75) & (enriched["kME"].abs() >= 0.60), "core_hub",
                 np.where(enriched["coassignment_support"] >= 0.50, "stable_member", "peripheral_boundary")),
    )
    return enriched, {
        "n_subsamples": int(n_subsamples),
        "median_subsample_ARI": float(np.median(aris_shared)) if aris_shared else np.nan,
        "median_coassignment_support": float(np.nanmedian(support_vec)) if np.isfinite(support_vec).any() else np.nan,
        "core_hub_events": int(enriched["membership_tier"].eq("core_hub").sum()),
    }
