"""Cross-Package Regulatory Bridge: Connect scSpliceNet Co-Splicing Modules -> scSpliceTrace 11-Channel RBP Motifs.

Lives inside `splicetrace` (Bulk) and `scsplicetrace` (Single-Cell) because PWM sequence scanning
requires `--fasta` and the bundled 1.0 RBP library (`621` PWMs), keeping `scsplicenet` pure matrix-only.
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
    "mature_consequence_delta",
    "upstream_intron_0_35",
    "upstream_intron_35_85",
    "upstream_intron_85_200",
    "downstream_intron_0_35",
    "downstream_intron_35_85",
    "downstream_intron_85_200",
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


def bridge_splicenet_modules_to_rbps(
    modules_df: pd.DataFrame,
    events_df: pd.DataFrame,
    fasta_path: Path | str,
    library: object,
    *,
    extract_rna_fn,
    score_odds_fn,
    min_module_events: int = 8,
    n_permutations: int = 999,
    seed: int = 20261118,
) -> pd.DataFrame:
    """Test every non-grey scSpliceNet co-splicing module against all curated RBP motifs across the
    11-channel spatial splicing map with GC/CpG/length FWL orthogonalization, one-sided cis-binding
    enrichment, and gene-block permutation max-T FWER calibration.
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

    merged_net = ev.merge(mod, on="event_id", how="inner")
    non_grey = sorted(
        [m for m in merged_net["mod_label"].unique() if m != "M0"],
        key=lambda x: (int(x[1:]) if x.startswith("M") and x[1:].isdigit() else 999, x),
    )
    if not non_grey or len(merged_net) < 12:
        return pd.DataFrame(columns=["module", "module_events", "module_genes", "RBP", "best_region", "stat_Z", "nominal_P", "module_P_FWER"])

    mod_genes_active = set(merged_net.loc[merged_net["mod_label"].ne("M0"), "gene_id"].astype(str))
    extra_bg = ev[(~ev["event_id"].isin(set(merged_net["event_id"]))) & (~ev["gene_id"].astype(str).isin(mod_genes_active))].copy()
    if not extra_bg.empty:
        cap_bg = min(len(extra_bg), max(1200, len(merged_net)))
        extra_bg = extra_bg.sample(n=cap_bg, random_state=int(seed)).copy()
        extra_bg["module"] = 0
        extra_bg["mod_label"] = "M0"
        extra_bg["kME"] = 0.0
        merged = pd.concat([merged_net, extra_bg], ignore_index=True)
    else:
        merged = merged_net.reset_index(drop=True)

    if len(merged) < 20:
        return pd.DataFrame(columns=["module", "module_events", "module_genes", "RBP", "best_region", "stat_Z", "nominal_P", "module_P_FWER"])

    if "comparison_id" not in merged.columns:
        merged["comparison_id"] = "splicenet_bridge"
    rna_df = extract_rna_fn(merged, Path(fasta_path), acceptor_core_nt=3, feature_only=False, compute_sha256=False)

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

    mat_sub = rna_df[rna_df["layer"].eq("mature_local_form") & rna_df["region"].isin(["inclusion", "exclusion"])]
    if not mat_sub.empty:
        ms = score_odds_fn(mat_sub["sequence"].astype(str).tolist(), models)
        inc_m = np.zeros((n_ev, len(m_ids)), dtype=np.float64)
        exc_m = np.zeros((n_ev, len(m_ids)), dtype=np.float64)
        ex_seqs, ex_targets = [], []
        has_junc = "junctions_json" in mat_sub.columns
        cols_iter = ["event_id", "region", "sequence", "junctions_json"] if has_junc else ["event_id", "region", "sequence"]
        for ri, rec in enumerate(mat_sub[cols_iter].itertuples(index=False)):
            ei = ev_index.get(str(rec.event_id))
            if ei is None:
                continue
            if rec.region == "inclusion":
                inc_m[ei] = ms[ri]
                seq_inc = str(rec.sequence)
                joins = json.loads(str(rec.junctions_json)) if (has_junc and rec.junctions_json) else []
                if len(joins) >= 2:
                    j1, j2 = int(joins[0]), int(joins[-1])
                    for piece, sl_i in (
                        (seq_inc[max(0, j1 - 50) : j1], 7),
                        (seq_inc[j1 : min(j2, j1 + 50)], 8),
                        (seq_inc[max(j1, j2 - 50) : j2], 9),
                        (seq_inc[j2 : min(len(seq_inc), j2 + 50)], 10),
                    ):
                        if len(piece) >= 6:
                            ex_seqs.append(piece)
                            ex_targets.append((ei, sl_i))
                elif len(joins) == 1:
                    j1 = int(joins[0])
                    for piece, sl_i in ((seq_inc[max(0, j1 - 50) : j1], 7), (seq_inc[j1 : min(len(seq_inc), j1 + 50)], 10)):
                        if len(piece) >= 6:
                            ex_seqs.append(piece)
                            ex_targets.append((ei, sl_i))
            else:
                exc_m[ei] = ms[ri]
        model_tensor[:, 0, :] = inc_m - exc_m
        if ex_seqs:
            ex_sc = score_odds_fn(ex_seqs, models)
            for idx_e, (ei, sl_i) in enumerate(ex_targets):
                model_tensor[ei, sl_i, :] = ex_sc[idx_e]

    reg_sub = rna_df[rna_df["layer"].eq("regulatory_pre_mRNA")]
    gc_num = np.zeros(n_ev, dtype=np.float64)
    gc_den = np.zeros(n_ev, dtype=np.float64)
    if not reg_sub.empty:
        sub_seqs, sub_targets = [], []
        for rec in reg_sub[["event_id", "region", "sequence"]].itertuples(index=False):
            ei = ev_index.get(str(rec.event_id))
            if ei is None:
                continue
            base_r = str(rec.region)
            seq_s = str(rec.sequence).upper()
            if not seq_s or base_r.endswith("_distal"):
                continue
            gc_num[ei] += seq_s.count("G") + seq_s.count("C")
            gc_den[ei] += max(1, seq_s.count("A") + seq_s.count("C") + seq_s.count("G") + seq_s.count("T"))
            kind = etype_map.get(str(rec.event_id), "SE")
            route = None
            if kind == "SE":
                if base_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif base_r == "inclusion_junction_2_donor_intron":
                    route = (False, 1.0)
            elif kind == "RI":
                if base_r == "exclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif base_r == "exclusion_junction_1_donor_intron":
                    route = (False, 1.0)
            elif kind == "A3SS":
                if base_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif base_r == "exclusion_junction_1_acceptor_intron":
                    route = (True, -1.0)
                elif base_r == "inclusion_junction_1_donor_intron":
                    route = (False, 1.0)
            elif kind == "A5SS":
                if base_r == "inclusion_junction_1_donor_intron":
                    route = (False, 1.0)
                elif base_r == "exclusion_junction_1_donor_intron":
                    route = (False, -1.0)
                elif base_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
            elif kind == "MXE":
                if base_r == "inclusion_junction_1_acceptor_intron":
                    route = (True, 1.0)
                elif base_r == "exclusion_junction_1_acceptor_intron":
                    route = (True, -1.0)
                elif base_r == "inclusion_junction_2_donor_intron":
                    route = (False, 1.0)
                elif base_r == "exclusion_junction_2_donor_intron":
                    route = (False, -1.0)
            if route is None:
                continue
            is_up, sgn = route
            L = len(seq_s)
            for b_i, (d0, d1) in enumerate(((0, 35), (35, 85), (85, 200))):
                piece = seq_s[max(0, L - d1) : max(0, L - d0)] if is_up else seq_s[min(L, d0) : min(L, d1)]
                if len(piece) >= 6:
                    sub_seqs.append(piece)
                    sub_targets.append((ei, (1 if is_up else 4) + b_i, sgn))
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
    exon_gc = np.array([exon_gc_map.get(str(x), 0.5) for x in merged["event_id"]], dtype=np.float64)
    log_len = np.array([float(np.log1p(aff_len_map.get(str(x), 100.0))) for x in merged["event_id"]], dtype=np.float64)
    etypes = merged["event_type"].astype(str).to_numpy()
    uniq_et = sorted(set(etypes))

    X_nuis = np.column_stack([
        np.ones(n_ev, dtype=np.float64),
        exon_gc,
        flank_gc,
        cpg_rate,
        log_len,
        *[(etypes == et).astype(np.float64) for et in uniq_et[1:]],
    ])

    genes = merged["gene_id"].astype(str).to_numpy()
    uniq_g, g_codes = np.unique(genes, return_inverse=True)
    n_genes = len(uniq_g)
    w = 1.0 / np.bincount(g_codes)[g_codes].astype(np.float64)
    w /= w.sum()

    F_flat = F_rbp.reshape(n_ev, n_reg * len(rbps))
    F_res = _fwl_resid(F_flat, X_nuis, w)
    f_sd0 = np.sqrt(np.maximum(np.sum((F_res * F_res) * w[:, None], axis=0), 1e-12))
    F_win = _fwl_resid(np.clip(F_res / f_sd0[None, :], -3.5, 3.5), X_nuis, w)
    F_g = np.zeros((n_genes, n_reg * len(rbps)), dtype=np.float64)
    np.add.at(F_g, g_codes, w[:, None] * F_win)
    F_gc = F_g - F_g.mean(axis=0, keepdims=True)
    sum_F2 = np.sum(F_gc * F_gc, axis=0)

    rng = np.random.default_rng(int(seed))
    perm_idx = np.vstack([rng.permutation(n_genes) for _ in range(int(n_permutations))])

    out_rows: list[dict] = []
    for m_lbl in non_grey:
        in_m = merged["mod_label"].to_numpy() == m_lbl
        n_m_ev = int(in_m.sum())
        if n_m_ev < int(min_module_events):
            continue
        n_m_genes = int(len(set(genes[in_m])))
        kme_vals = np.nan_to_num(merged["kME"].to_numpy(dtype=np.float64), nan=0.0)
        y_raw = np.where(in_m, np.maximum(np.abs(kme_vals), 0.25), 0.0)
        is_m0 = merged["mod_label"].to_numpy() == "M0"
        active_mask = in_m | is_m0 if int(is_m0.sum()) >= 100 else np.ones(n_ev, dtype=bool)
        if active_mask.all():
            y_res = _fwl_resid(y_raw, X_nuis, w)
            y_g = np.zeros(n_genes, dtype=np.float64)
            np.add.at(y_g, g_codes, w * y_res)
            yc = y_g - y_g.mean()
            F_gc_m = F_gc
            sum_F2_m = sum_F2
        else:
            sub_w = np.where(active_mask, w, 0.0)
            sub_w /= max(sub_w.sum(), 1e-12)
            F_res_m = _fwl_resid(F_flat, X_nuis, sub_w)
            sd_m = np.sqrt(np.maximum(np.sum((F_res_m * F_res_m) * sub_w[:, None], axis=0), 1e-12))
            F_win_m = _fwl_resid(np.clip(F_res_m / sd_m[None, :], -3.5, 3.5), X_nuis, sub_w)
            y_res_m = _fwl_resid(y_raw, X_nuis, sub_w)
            F_g_m = np.zeros((n_genes, n_reg * len(rbps)), dtype=np.float64)
            np.add.at(F_g_m, g_codes, sub_w[:, None] * F_win_m)
            y_g_m = np.zeros(n_genes, dtype=np.float64)
            np.add.at(y_g_m, g_codes, sub_w * y_res_m)
            g_act = np.bincount(g_codes, weights=active_mask.astype(float), minlength=n_genes) > 0
            n_ga = max(int(g_act.sum()), 2)
            F_gc_m = np.where(g_act[:, None], F_g_m - F_g_m[g_act].mean(axis=0, keepdims=True), 0.0)
            yc = np.where(g_act, y_g_m - y_g_m[g_act].mean(), 0.0)
            sum_F2_m = np.sum(F_gc_m * F_gc_m, axis=0) * ((n_genes - 1.0) / max(n_ga - 1.0, 1.0))

        sd_perm = np.sqrt(np.sum(yc * yc) * sum_F2_m / max(n_genes - 1, 1))
        valid_c = sd_perm > 1e-12
        Z_obs = np.divide(yc @ F_gc_m, sd_perm, out=np.zeros_like(sd_perm), where=valid_c)
        Y_perm = yc[perm_idx]
        null_Z = np.divide(
            (Y_perm.astype(np.float32) @ F_gc_m.astype(np.float32)).astype(np.float64),
            sd_perm[None, :],
            out=np.zeros((int(n_permutations), F_gc_m.shape[1]), dtype=np.float64),
            where=valid_c[None, :],
        )

        Z_2d = Z_obs.reshape(n_reg, len(rbps))
        null_3d = null_Z.reshape(int(n_permutations), n_reg, len(rbps))
        score_2d = np.vstack([np.abs(Z_2d[0:1, :]), np.maximum(Z_2d[1:, :], 0.0)])
        null_score_3d = np.concatenate([np.abs(null_3d[:, 0:1, :]), np.maximum(null_3d[:, 1:, :], 0.0)], axis=1)

        rbp_max = np.max(score_2d, axis=0)
        null_rbp_max = np.max(null_score_3d, axis=1)
        glob_max = np.max(null_rbp_max, axis=1)

        nom_p = (1.0 + np.sum(null_rbp_max >= rbp_max[None, :] - 1e-6, axis=0)) / (n_permutations + 1.0)
        fwer_p = (1.0 + np.sum(glob_max[:, None] >= rbp_max[None, :] - 1e-6, axis=0)) / (n_permutations + 1.0)

        order = np.argsort(-rbp_max)[:15]
        for r_i in order:
            best_reg = int(np.argmax(score_2d[:, r_i]))
            out_rows.append({
                "module": m_lbl,
                "module_events": n_m_ev,
                "module_genes": n_m_genes,
                "RBP": rbps[r_i],
                "best_region": _BRIDGE_REGION_NAMES[best_reg],
                "stat_Z": round(float(Z_2d[best_reg, r_i]), 4),
                "nominal_P": round(float(nom_p[r_i]), 5),
                "module_P_FWER": round(float(fwer_p[r_i]), 5),
            })

    return pd.DataFrame(out_rows)
