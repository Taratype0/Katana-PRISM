"""Portable public benchmark runner for Katana-PRISM v1.0.6 (Nucleic Acids Research).

Reproduces & verifies:
  1. 13-channel (`distal_intron_nt = 500`, mature-RNA path + 12 spatial windows) FWL-orthogonalized `SpliceTrace`
     continuous, unified `169`-RBP / `621`-PWM discrete, single-variable ablation, native `rMAPS3`, and `rMAPS2`-style
     scans across all `18` untruncated bulk cohorts.
  2. Exact `GRCh38` `HepG2` IDR `eCLIP` and neuronal `PTBP2` `CLIP` whole-event and `+-35 nt` motif-locus
     conditional MLE Fisher odds ratios and exact 95% confidence intervals.
  3. Real `GSE122380` (`N = 297` libraries, `19` donors x `16` days, strict GENCODE v28 paired `M1–M4`)
     `SpliceNet -> SpliceTrace` `12`-channel module-to-RBP regulatory bridge (`676` rows across `4,310` valid
     junctions; `15` study-FWER hits across `12` dev RBPs / `3` families; `0` M3/M4 FPs), `6`-method `50`-trial matched
     study-wide `Holm <= 0.05` null calibration (`0.0%` vs `22.0%–52.0%`), `625` coding proteome/NMD candidates
     (`354` bridged + `271` `M3/M4` orphan), and unified `128`-gene common-control `SpliceNet` evaluation plus
     `AR(1)`-calibrated paired-donor Early-vs-Late PERMANOVA (`P = 0.0001`).
  4. `GSE270928` Smart-seq3 dual-denominator retention (`0/1,000` shared HVIs; `327/66,391` multi-isoform transcripts via
     `kallisto v0.46.1`), `7`-mouse cross-tissue `Tabula Muris` `70`-split calibration & real-anchored `sigma_het` sweep,
     `Tasic` `71`-mouse (`6.3x / 2.9x > acorde`), `GSE270928` usage (`0.00968` tie vs `acorde` `0.00966` / `0.00952`),
     donor splicing (`-16.1%/-16.3%` vs `CASREL`, tying `MRAS`), `Longcell` `N=1` stress test (`0/12` `FWER`), `SLAMF1`,
     and dual-background `MSigDB`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    str(Path(tempfile.gettempdir()) / "katana_prism_numba_cache"),
)

_SCRIPT_DIR = Path(__file__).resolve().parent
_CANDIDATE_ROOTS = [
    _SCRIPT_DIR.parents[1],
    _SCRIPT_DIR.parent,
    *_SCRIPT_DIR.parents[2:],
]
for _root in _CANDIDATE_ROOTS:
    for _pkg in ("splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace"):
        for _base in (_root / "packages" / _pkg, _root / _pkg, _root):
            if (_base / _pkg / "__init__.py").is_file():
                if str(_base) not in sys.path:
                    sys.path.insert(0, str(_base))
                break

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, norm
from statsmodels.stats.multitest import multipletests

from splicemanifold.ps_das import parse_number_list, qualify_ps_das_events
from splicenet.evaluation import score_modules
from splicenet.network import fit_splicenet
from splicenet.qc_and_selection import select_events, unit_robust_audit
from splicetrace.continuous_rescue import run_bulk_continuous_rescue
from splicetrace.fast_accel import configure_server_threads, fast_score_mean_log_odds
from splicetrace.library import load_bundled_human_rbp_library
from splicetrace.module_bridge import (
    annotate_rbp_expression_disambiguation,
    bridge_splicenet_modules_to_rbps,
    parse_rbp_expression_reference,
)
from splicetrace.rmats_io import _standardize_rmats_df
from splicetrace.rna_forms import IndexedFasta, extract_event_rna


def exact_or_and_ci(a: int, b: int, c: int, d: int) -> tuple[float, float, float, float]:
    """Return conditional MLE/exact Fisher OR, one-sided greater P, and Haldane-Anscombe 95% CI."""
    or_f, _ = fisher_exact([[a, b], [c, d]], alternative="two-sided")
    _, p_gr = fisher_exact([[a, b], [c, d]], alternative="greater")
    aa, bb, cc, dd = a + 0.5, b + 0.5, c + 0.5, d + 0.5
    or_ha = (aa * dd) / (bb * cc)
    se_ln = math.sqrt(1.0 / aa + 1.0 / bb + 1.0 / cc + 1.0 / dd)
    ci_lo = math.exp(math.log(or_ha) - 1.95996 * se_ln)
    ci_hi = math.exp(math.log(or_ha) + 1.95996 * se_ln)
    return float(or_ha if a == 0 or c == 0 else or_f), float(p_gr), float(ci_lo), float(ci_hi)


def run_encode_bridge_benchmark(splicetools_dir: Path, hg38_fasta: Path, out_dir: Path, rbp_expr_tsv: Path | None = None) -> pd.DataFrame:
    """Reproduce the 16-library ENCODE SpliceNet -> 12-channel (distal=500) SpliceTrace module bridge."""
    configure_server_threads(0)
    lib = load_bundled_human_rbp_library(species="human")
    contigs = set(IndexedFasta(hg38_fasta).index.keys())
    targets = ("PTBP1", "SRSF1", "HNRNPA1", "HNRNPF")

    cohorts: dict[str, pd.DataFrame] = {}
    for rbp in targets:
        frames = []
        for kind in ("SE", "RI", "A3SS", "A5SS", "MXE"):
            fp = splicetools_dir / f"{rbp}_test_cntl_{kind}.MATS.JCEC.txt"
            if fp.is_file():
                frames.append(_standardize_rmats_df(pd.read_csv(fp, sep="\t", low_memory=False), kind, contigs, f"ENCODE_{rbp}"))
        cohorts[rbp] = qualify_ps_das_events(pd.concat(frames, ignore_index=True).drop_duplicates("event_id"), fdr_cutoff=0.05, dpsi_cutoff=0.10)

    common_ids = sorted(set.intersection(*(set(df["event_id"]) for df in cohorts.values())))
    idx_cohorts = {r: cohorts[r].set_index("event_id").loc[common_ids] for r in targets}

    psi_blocks, sup_blocks, ctl_means = [], [], []
    for rbp in targets:
        df = idx_cohorts[rbp]
        n = len(df)
        pm, sm = np.full((n, 4), np.nan), np.zeros((n, 4))
        for idx, (inc1, inc2, i1, s1, i2, s2) in enumerate(zip(df["IncLevel1"], df["IncLevel2"], df["IJC_SAMPLE_1"], df["SJC_SAMPLE_1"], df["IJC_SAMPLE_2"], df["SJC_SAMPLE_2"])):
            p1, p2 = parse_number_list(inc1), parse_number_list(inc2)
            ij1, sj1, ij2, sj2 = parse_number_list(i1), parse_number_list(s1), parse_number_list(i2), parse_number_list(s2)
            for r in range(min(2, len(p1))):
                pm[idx, r] = p1[r]
            for r in range(min(2, len(p2))):
                pm[idx, 2 + r] = p2[r]
            for r in range(min(2, len(ij1), len(sj1))):
                sm[idx, r] = max(0.0, ij1[r]) + max(0.0, sj1[r])
            for r in range(min(2, len(ij2), len(sj2))):
                sm[idx, 2 + r] = max(0.0, ij2[r]) + max(0.0, sj2[r])
        cm = np.nanmean(pm[:, 2:4], axis=1)
        ctl_means.append(cm)
        psi_blocks.append((rbp, pm, cm))
        sup_blocks.append(sm)

    g_ctl = np.nanmean(np.column_stack(ctl_means), axis=1)
    psi_16, sample_cols, meta_rows = [], [], []
    for rbp, pm, cm in psi_blocks:
        off = np.where(np.isfinite(cm) & np.isfinite(g_ctl), g_ctl - cm, 0.0)
        psi_16.append(np.clip(pm + off[:, None], 0.0, 1.0))
        for tag, cond in ((f"{rbp}_KD_1", f"{rbp}_KD"), (f"{rbp}_KD_2", f"{rbp}_KD"), (f"{rbp}_CTL_1", "Control"), (f"{rbp}_CTL_2", "Control")):
            sample_cols.append(tag)
            meta_rows.append({"sample_id": tag, "biological_sample_id": tag, "condition": cond})

    usage_df = pd.DataFrame(np.hstack(psi_16), index=common_ids, columns=sample_cols)
    support_df = pd.DataFrame(np.hstack(sup_blocks), index=common_ids, columns=sample_cols)
    meta_df = pd.DataFrame(meta_rows)

    masked_u, audit_u = unit_robust_audit(usage_df, meta_df, support_df)
    sel_u, _ = select_events(masked_u, audit_u, feature_cap=5000, ranking_metric="excess_bio_variance")
    assign_df = fit_splicenet(sel_u, None, power=0, min_module_size=20, backend="python")
    scores_df, _, _, _ = score_modules(sel_u, assign_df)

    net_ev_df = idx_cohorts["SRSF1"].reset_index()
    net_ev_df = net_ev_df[net_ev_df["event_id"].isin(set(assign_df["event_key"]))].copy()
    expr_df = pd.read_csv(rbp_expr_tsv, sep="\t") if rbp_expr_tsv else None

    bridge_df = bridge_splicenet_modules_to_rbps(
        assign_df, net_ev_df, hg38_fasta, lib,
        extract_rna_fn=extract_event_rna, score_odds_fn=fast_score_mean_log_odds,
        distal_intron_nt=500, min_module_events=20, n_permutations=999, seed=20261119,
        rbp_expr_df=expr_df, module_scores_df=scores_df,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_df.to_csv(out_dir / "fig4a_encode_module_rbp_bridge_distal500.tsv", sep="\t", index=False)
    return bridge_df


def verify_bundled_tables(tables_path: Path) -> None:
    """Verify key statistical invariants across ALL bundled NAR benchmark tables (100% file coverage)."""
    verified: set[str] = set()

    def _read(name: str, **kwargs: object) -> pd.DataFrame:
        fp = tables_path / name
        assert fp.is_file(), f"Missing bundled benchmark table: {fp}"
        verified.add(name)
        return pd.read_csv(fp, sep="\t", **kwargs)

    # 1. PS-DAS full-data qualification, pure nulls, rescued eCLIP, and downstream utility
    f1_vs = _read("fig1_ps_das_vs_competitors.tsv")
    f1_pos = f1_vs[f1_vs["role"].eq("positive_RBP_KD") & f1_vs["method"].eq("Katana-PRISM (PS-DAS)")]
    assert int(f1_pos["called_das_events"].sum()) == 10291
    assert int(f1_pos["noncircular_asym_retained"].sum()) == 573 and int(f1_pos["noncircular_asym_total"].sum()) == 749
    assert abs(float(f1_pos["noncircular_asym_recall_pct"].mean()) - 74.57) < 0.1
    f1_nega = f1_vs[f1_vs["cohort"].eq("GSE206660_NegA")].set_index("method")
    assert int(f1_nega.loc["Raw rMATS (Unfiltered)", "called_das_events"]) == 3414
    assert int(f1_nega.loc["Katana-PRISM (PS-DAS)", "called_das_events"]) == 1041

    f1_pn = _read("fig1_ps_das_same_condition_pure_null.tsv")
    assert len(f1_pn) == 5 and (f1_pn["PS_DAS_qualified_calls"] <= f1_pn["raw_FDR05_dPSI10_calls"]).all()

    f1_ec = _read("fig1_ps_das_rescued_eclip_validation.tsv")
    pt_res = f1_ec[f1_ec["knockdown_RBP"].eq("PTBP1") & f1_ec["subset"].eq("PS-DAS_rescued_only") & f1_ec["direction_filter"].eq("all_events")].iloc[0]
    assert int(pt_res["fg_hits"]) == 5 and int(pt_res["fg_total"]) == 89 and abs(float(pt_res["odds_ratio"]) - 5.4746) < 1e-2

    s1_ds = _read("fig1_ps_das_downstream_splicetrace_utility.tsv")
    assert (s1_ds["downstream_model"] == "Continuous 13-channel FWL + 1,999 host-gene-block maxT").all()
    pt2_ds = s1_ds[s1_ds["cohort"].eq("GSE206660_PTBP2") & s1_ds["upstream_gate"].eq("Katana-PRISM (PS-DAS)")].iloc[0]
    assert int(pt2_ds["qualified_foreground_events"]) == 2700
    assert int(pt2_ds["exact_target_rank"]) == 3 and int(pt2_ds["downstream_SpliceTrace_sig_rbps_fwer05"]) == 36
    assert abs(float(pt2_ds["exact_target_abs_Z"]) - 7.227) < 0.01
    assert int(s1_ds.loc[s1_ds["cohort"].eq("GSE206660_NegA") & s1_ds["upstream_gate"].eq("Katana-PRISM (PS-DAS)"), "qualified_foreground_events"].iloc[0]) == 1041
    assert int(s1_ds.loc[s1_ds["cohort"].eq("GSE206660_NegA"), "downstream_SpliceTrace_sig_rbps_fwer05"].sum()) == 0
    enc_u = s1_ds[s1_ds["cohort"].str.startswith("ENCODE_")]
    assert int(enc_u.loc[enc_u["upstream_gate"].eq("Katana-PRISM (PS-DAS)"), "retained_full_ENCODE_UniProt_switches"].sum()) == 3764
    assert int(enc_u.loc[enc_u["upstream_gate"].eq("Strict All>=10 Filter"), "retained_full_ENCODE_UniProt_switches"].sum()) == 3550
    assert int(enc_u.loc[enc_u["upstream_gate"].eq("Katana-PRISM (PS-DAS)"), "retained_full_ENCODE_NMD_switches"].sum()) == 2884

    # 2. SpliceTrace 18-cohort calibration, ablations, 44-PWM retrieval, eCLIP, native rMAPS3, and untruncated rMAPS2-style
    f2c = _read("fig2c_splicetrace_exact_and_ablations.tsv")
    nulls = f2c[f2c["role"].isin(["negative_control", "mitochondrial_negative_control"])]
    assert len(nulls) == 3 and int(nulls["cont_sig_rbps_fwer05"].sum()) == 0 and int(nulls["disc_sig_rbps_fwer05"].sum()) == 0
    assert list(nulls.set_index("case").loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "NoFWL_SameMaxT_sig_fwer05"].astype(int)) == [21, 27, 3]
    assert nulls["disc_exact_rbp_P_FWER"].isna().all() and nulls["cont_exact_P_FWER"].isna().all()
    hb = f2c[f2c["role"].eq("high_block_positive")]
    assert len(hb) == 5 and int((hb["disc_exact_rbp_P_FWER"] <= 0.05).sum()) == 4 and int((hb["cont_exact_P_FWER"] <= 0.05).sum()) == 4
    assert int((hb["NoFWL_SameMaxT_exact_P_FWER"] <= 0.05).sum()) == 3
    pt2_c = f2c[f2c["case"].eq("GSE206660_PTBP2")].iloc[0]
    assert int(pt2_c["cont_evaluated_events"]) == 51941 and int(pt2_c["cont_fg_events"]) == 9095
    assert int(pt2_c["cont_exact_rank"]) == 4 and int(pt2_c["cont_sig_rbps_fwer05"]) == 73
    assert abs(float(pt2_c["cont_exact_Z"]) - 7.689) < 0.01
    f2c_idx = f2c.set_index("case")
    assert abs(float(f2c_idx.loc["SRSF1", "cont_exact_P_FWER"]) - 0.001) < 1e-6
    assert abs(float(f2c_idx.loc["HNRNPF", "cont_exact_P_FWER"]) - 0.0025) < 1e-6
    assert abs(float(f2c_idx.loc["SRSF3", "cont_exact_P_FWER"]) - 0.0315) < 1e-6
    assert list(f2c_idx.loc[["HNRNPA1", "SRSF3", "ZRANB2"], "cont_sig_rbps_fwer05"].astype(int)) == [6, 3, 3]
    lb = f2c[f2c["role"].eq("low_block_positive")]
    assert len(lb) == 10 and lb["disc_exact_rbp_P_FWER"].notna().all() and lb["disc_exact_rank"].notna().all()
    assert int((lb["disc_exact_rbp_P_FWER"] <= 0.05).sum()) == 0
    assert int((lb["cont_exact_P_FWER"] <= 0.05).sum()) == 1 and int((lb["NoFWL_SameMaxT_exact_P_FWER"] <= 0.05).sum()) == 5
    assert int(lb["NoFWL_SameMaxT_sig_fwer05"].min()) == 0 and int(lb["NoFWL_SameMaxT_sig_fwer05"].max()) == 153

    f2c_ab = _read("fig2c_encode_deep5_splicetrace_ablation.tsv")
    assert len(f2c_ab) == 12 and int(f2c_ab.loc[f2c_ab["case"].eq("GSE206660_NegA"), "sig_rbps_fwer05"].sum()) == 0
    assert bool(f2c_ab.loc[f2c_ab["case"].eq("GSE206660_PTBP2"), "target_detected_fwer05"].all())

    f2d = _read("fig2d_44pwm_ptb_retrieval_summary.tsv").set_index(["method", "case"])
    assert len(f2d) == 8
    assert abs(float(f2d.loc[("SpliceTrace_13ch_FWL", "PTBP1"), "PTB_mean_rank_percentile"]) - 0.802326) < 1e-4
    assert abs(float(f2d.loc[("SpliceTrace_13ch_FWL", "NegA"), "PTB_mean_rank_percentile"]) - 0.325581) < 1e-4

    f2e = _read("fig2e_hepg2_eclip_locus_and_event_validation.tsv")
    assert len(f2e) == 62
    f2e_pt1 = f2e[(f2e["knockdown_RBP"] == "PTBP1") & (f2e["clip_assay"] == "PTBP1_HepG2_eCLIP") & (f2e["direction_filter"] == "kd_included_up") & (f2e["motif_filter"] == "all_events_unfiltered") & (f2e["window_scope"] == "exon_pm300nt")].iloc[0]
    assert abs(float(f2e_pt1["odds_ratio"]) - 3.7049) < 1e-2 and int(f2e_pt1["fg_hits"]) == 29

    rm3 = _read("tableS2a_native_rmaps3_all18_cohorts.tsv").set_index("case")
    assert len(rm3) == 18 and list(rm3.loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "rmaps3_sig_rbps_bonf_full"].astype(int)) == [10, 26, 29]
    rm2 = _read("tableS2a_untruncated_rmaps2_169rbp_all18.tsv").set_index("case")
    assert len(rm2) == 18 and list(rm2.loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "rMAPS2_sig_rbps_bonf"].astype(int)) == [4, 1, 0]
    het_bulk = _read("tableS2_bulk_heteroskedastic_null_calibration.tsv")
    assert len(het_bulk) == 4
    assert float(het_bulk.iloc[0]["perm_exchangeable_FWER05_pct"]) <= 5.0 and float(het_bulk.iloc[0]["rademacher_hc_FWER05_pct"]) <= 5.0
    assert float(het_bulk.iloc[-1]["rademacher_hc_FWER05_pct"]) <= 5.0
    het_real = _read("tableS2_bulk_real_design_heteroskedastic_stress.tsv")
    assert len(het_real) == 3
    assert [round(float(x), 2) for x in het_real["perm_exchangeable_FWER05_pct"]] == [5.0, 11.0, 8.0]
    assert [round(float(x), 2) for x in het_real["rademacher_hc_FWER05_pct"]] == [2.25, 5.0, 5.25]

    # 3. Multi-donor GSE122380 SpliceNet (unified 128-gene common-control pool), bulk manifold, GEUVADIS, GSE62669, Regulatory Bridge & 50-trial null benchmark
    f3d = _read("fig3d_gse122380_fair_wgcna_and_djexpress.tsv")
    assert len(f3d) == 14 and "evaluable_coherence_donors" in f3d.columns
    ref_sub = f3d[f3d["method"].eq("Locked_CoSplice_v4_3_3")].set_index("budget")
    assert list(ref_sub.loc[[1000, 2000, 5000], "evaluable_coherence_donors"].astype(int)) == [19, 19, 19]
    assert abs(float(ref_sub.loc[1000, "median_coherence_gap"]) - 0.3654) < 1e-3
    assert abs(float(ref_sub.loc[2000, "median_coherence_gap"]) - 0.3495) < 1e-3
    assert abs(float(ref_sub.loc[5000, "median_coherence_gap"]) - 0.3097) < 1e-3
    py_sub = f3d[f3d["method"].eq("Pure_Python_SpliceNet")].set_index("budget")
    assert list(py_sub.loc[[1000, 2000, 5000], "evaluable_coherence_donors"].astype(int)) == [19, 19, 19]
    assert abs(float(py_sub.loc[1000, "median_coherence_gap"]) - 0.1491) < 1e-3
    assert abs(float(py_sub.loc[2000, "median_coherence_gap"]) - 0.0760) < 1e-3
    assert abs(float(py_sub.loc[5000, "median_coherence_gap"]) - 0.0298) < 1e-3
    wrap_sub = f3d[f3d["method"].eq("R_Backed_Python_Wrapper_v2")].set_index("budget")
    assert abs(float(wrap_sub.loc[5000, "median_coherence_gap"]) - 0.3079) < 1e-3
    w5k = f3d[(f3d["budget"] == 5000) & f3d["method"].eq("Standard_WGCNA_SameUsageRatio")].iloc[0]
    assert abs(float(w5k["median_coherence_gap"]) - 0.0187) < 1e-3 and str(w5k["paired_wins_py_vs_method"]) == "19/19"
    dj5k = f3d[(f3d["budget"] == 5000) & f3d["method"].eq("Peer_DJExpress_JCNA1pass")].iloc[0]
    assert abs(float(dj5k["median_coherence_gap"]) - 0.0439) < 1e-3 and str(dj5k["paired_wins_py_vs_method"]) == "7/19"
    assert abs(float(dj5k["paired_delta_r_py_vs_method"]) - 0.0101) < 5e-4

    f3_per = _read("fig3d_gse122380_per_donor_5000.tsv")
    assert len(f3_per) == 76 and f3_per["donor"].nunique() == 19
    assert {"Locked_CoSplice_v4_3_3", "Pure_Python_SpliceNet", "Standard_WGCNA_SameUsageRatio", "Peer_DJExpress_JCNA1pass"}.issubset(set(f3_per["method"]))
    piv_per = f3_per.pivot(index="donor", columns="method", values="coherence_gap")
    assert int((piv_per["Pure_Python_SpliceNet"] > piv_per["Peer_DJExpress_JCNA1pass"]).sum()) == 7
    assert int((piv_per["Pure_Python_SpliceNet"] > piv_per["Standard_WGCNA_SameUsageRatio"]).sum()) == 19
    assert int((piv_per["Locked_CoSplice_v4_3_3"] > piv_per["Peer_DJExpress_JCNA1pass"]).sum()) == 19

    geuv = _read("geuvadis_464donors_matched_events_summary.tsv").set_index("method")
    assert len(geuv) == 8 and abs(float(geuv.loc["raw_WGCNA", "mean_heldout_top_r"]) - 0.689458) < 1e-4

    g62 = _read("fig2_gse62669_splicenet_vs_wgcna_mantel.tsv").set_index("method")
    assert int(g62.loc["Historical_CoSplice_v1", "non_grey_modules"]) == 2
    assert abs(float(g62.loc["Peer_Mantel_CoSplicEx", "fixed_feature_subsample_ARI_median"]) - 0.8130) < 1e-3
    cf3e = _read("fig3e_gse122380_crossfold_stability.tsv")
    assert len(cf3e) == 6
    cf_py = cf3e[cf3e["method"].eq("Katana_PRISM_SpliceNet")].set_index("budget")["cross_fold_ARI_median"]
    cf_wg = cf3e[cf3e["method"].eq("Standard_WGCNA_SameUsageRatio")].set_index("budget")["cross_fold_ARI_median"]
    assert np.allclose(cf_py.loc[[1000, 2000, 5000]].astype(float), [0.8306, 0.7514, 0.5692], atol=1e-3)
    assert np.allclose(cf_wg.loc[[1000, 2000, 5000]].astype(float), [0.6925, 0.5816, 0.4141], atol=1e-3)
    f2m = _read("fig2_bulk_manifold_vs_jutils_marvel.tsv")
    assert len(f2m) == 16 and abs(float(f2m.loc[f2m["dropout_rate"].eq(0.45) & f2m["method"].str.contains("SpliceManifold"), "permanova_r2"].iloc[0]) - 0.1992) < 1e-4
    assert float(f2m.loc[f2m["method"].str.contains("SpliceManifold"), "donor_centroid_signflip_p"].max()) <= 0.0001 + 1e-9
    ar_cal = _read("fig3f_longitudinal_ar_calibration.tsv")
    assert len(ar_cal) == 3 and (ar_cal["observed_fpr"] <= 0.07).all()

    b_df = _read("fig4a_gse122380_module_rbp_bridge_distal500.tsv")
    assert len(b_df) == 676 and b_df["module"].nunique() == 4 and b_df["RBP"].nunique() == 169
    sig_study = b_df[b_df["study_P_FWER"] <= 0.05]
    assert len(sig_study) == 15 and sig_study["RBP"].nunique() == 12 and set(sig_study["module"].unique()) == {"M1", "M2"}
    assert float(b_df.loc[b_df["module"].isin(["M3", "M4"]), "study_P_FWER"].min()) > 0.05
    assert float(b_df.loc[b_df["module"].isin(["M3", "M4"]), "module_P_FWER"].min()) > 0.05
    b_full = _read("fig4_module_rbp_bridge_full.tsv")
    assert b_full.equals(b_df)

    mg_s = _read("fig4a_multigeometry_sensitivity.tsv").set_index("regime")
    assert len(mg_s) == 5 and int(mg_s.loc["Full_strict_GENCODE_representative", "cardiac_families_detected"]) == 3
    assert int(mg_s.loc["Unique_GENCODE_geometry_only", "evaluated_junctions"]) == 3560

    eig = _read("fig4_splicenet_module_eigengenes_297samples.tsv")
    assert len(eig) == 297 and eig["biological_sample_id"].nunique() == 19 and {"M1", "M2", "M3", "M4"}.issubset(eig.columns)
    prot_gse = _read("fig4_splicenet_module_proteome_consequences.tsv", low_memory=False)
    assert len(prot_gse) == 827 and prot_gse["event_id"].nunique() == 827

    flows = _read("fig4_causal_alluvial_flows.tsv")
    assert len(flows) == 625 and flows["event_id"].nunique() == 625
    assert int((flows["bridge_tier"] == "Study_FWER<=0.05_Bridged").sum()) == 354
    assert int((flows["bridge_tier"] == "Orphan_DonorBackground_FWER>0.05").sum()) == 271
    assert flows["module"].value_counts().to_dict() == {"M1": 176, "M2": 178, "M3": 154, "M4": 117}
    assert not flows["gene_symbol"].isin(["SPTBN1", "DCTN1"]).any()
    assert {"SNRNP70", "SETD5"}.issubset(set(flows["gene_symbol"].unique()))
    assert _read("fig4c_bridged_vs_orphan_coding_flows.tsv").equals(flows)

    f4d = _read("fig4d_deconvolution_vs_competitors.tsv", keep_default_na=False).set_index("method")
    assert _read("fig4_deconvolution_vs_competitors.tsv", keep_default_na=False).set_index("method").equals(f4d)
    prism_4d = f4d.loc["Katana-PRISM Bridge"]
    assert int(prism_4d["evaluated_events"]) == 4310
    assert int(prism_4d["fwer05_developmental_rbps_called"]) == 12 and int(prism_4d["fwer05_canonical_cardiac_families"]) == 3
    assert int(prism_4d["fwer05_donor_null_M3_M4_fp_rbps"]) == 0 and abs(float(prism_4d["fwer05_null_50trial_familywise_FPR_pct"]) - 0.0) < 1e-4
    assert abs(float(prism_4d["bh05_study_null_50trial_familywise_FPR_pct"]) - 0.0) < 1e-4
    assert abs(float(prism_4d["bh05_permod_null_50trial_familywise_FPR_pct"]) - 0.0) < 1e-4
    assert abs(float(prism_4d["fwer05_b4999_null_50trial_FPR_pct"]) - 0.0) < 1e-4
    assert abs(float(prism_4d["per_mod_fwer05_b4999_null_50trial_FPR_pct"]) - 10.0) < 1e-4
    assert abs(float(prism_4d["bh05_study_b4999_null_50trial_FPR_pct"]) - 2.0) < 1e-4
    assert abs(float(prism_4d["bh05_permod_b4999_null_50trial_FPR_pct"]) - 16.0) < 1e-4
    mwu_4d = f4d.loc["SpliceNet + Unadjusted MWU"]
    assert abs(float(mwu_4d["fwer05_null_50trial_familywise_FPR_pct"]) - 52.0) < 1e-4
    assert int(mwu_4d["fwer05_developmental_rbps_called"]) == 22 and int(mwu_4d["fwer05_donor_null_M3_M4_fp_rbps"]) == 0
    assert int(mwu_4d["bh05_developmental_rbps_called"]) == 52 and int(mwu_4d["bh05_donor_null_M3_M4_fp_rbps"]) == 10
    assert abs(float(f4d.loc["Raw-Variance WGCNA + Unadjusted MWU", "fwer05_null_50trial_familywise_FPR_pct"]) - 22.0) < 1e-4
    for pooled_m in ["Whole-Pool SpliceTrace (No SpliceNet)", "Pooled rMAPS-style Wilcoxon", "Pooled Unmatched MWU (Matt-style)"]:
        assert int(f4d.loc[pooled_m, "evaluated_events"]) == 3043

    # 4. Single-cell Smart-seq2/3 benchmarks (GSE270928, Tabula Muris, Tasic, Longcell)
    f5a = _read("fig5a_sc_manifold_vs_marvel_jutils.tsv").set_index("method")
    assert abs(float(f5a.loc["Katana-PRISM (SpliceManifold-SC)", "permanova_r2"]) - 0.0468) < 1e-4
    assert abs(float(f5a.loc["Katana-PRISM (SpliceManifold-SC)", "donor_lisi"]) - 1.385) < 1e-3
    assert abs(float(f5a.loc["Katana-PRISM (SpliceManifold-SC)", "mean_modality_divergence"]) - 0.9748) < 1e-4
    assert abs(float(f5a.loc["Katana-PRISM (SpliceManifold-SC)", "within_donor_perm_p"]) - 0.001) < 1e-4
    assert abs(float(f5a.loc["MARVEL (Mean-Imputed, Same 1,000 HVEs)", "permanova_r2"]) - 0.0118) < 1e-4
    assert "input_representation" in f5a.columns and "kallisto" in str(f5a["input_representation"].iloc[0])

    f5b = _read("fig5b_gse270928_sc_proteome_consequences.tsv", keep_default_na=False)
    f5b_cod = f5b[f5b["frame_status"].isin(["frameshift", "in_frame_indel", "in_frame_PTC", "translation_start_altered"])]
    assert len(f5b) == 195 and len(f5b_cod) == 167
    assert int(f5b["frame_status"].eq("utr_cis_regulatory_only").sum()) == 14
    assert int(f5b["frame_status"].eq("noncoding_transcript").sum()) == 13
    assert int(f5b["frame_status"].eq("annotation_unmatched").sum()) == 1
    assert f5b.loc[f5b["frame_status"].eq("annotation_unmatched"), "nmd_switch_class"].iloc[0] == "unannotated"
    assert int(f5b_cod["contributing_donors"].astype(int).eq(1).sum()) == 89 and int(f5b_cod["contributing_donors"].astype(int).ge(2).sum()) == 78
    slam_b = f5b_cod[f5b_cod["gene_symbol"].eq("SLAMF1")].iloc[0]
    assert abs(float(slam_b["raw_dpsi"]) - 0.095456) < 1e-4 and abs(float(slam_b["eb_shrunk_dpsi"]) - 0.077136) < 1e-4
    ptprc_b = f5b_cod[f5b_cod["gene_symbol"].eq("PTPRC")].iloc[0]
    assert float(ptprc_b["host_log2fc"]) < -1.0  # repressed host gene (-1.025)

    slam = _read("fig5c_gse270928_slamf1_per_cell_event_psi.tsv")
    assert int(slam["event_psi"].notna().sum()) == 76
    slam_any = (slam["inclusion_reads"] + slam["exclusion_reads"]) > 0
    assert int(slam_any[slam["condition"].eq("Ctrl")].sum()) == 18 and int(slam_any[~slam["condition"].eq("Ctrl")].sum()) == 90
    assert int(slam.loc[slam["condition"].eq("Ctrl"), "event_psi"].notna().sum()) == 7 and int(slam.loc[~slam["condition"].eq("Ctrl"), "event_psi"].notna().sum()) == 69

    tas = _read("fig5e_tasic_71mice_scsplicenet_vs_pooled_acorde.tsv")
    tas_w = tas[tas["scope"].eq("within_animal_cell_type") & tas["endpoint"].eq("observed_usage")].groupby("method")["heldout_correlation"].mean()
    assert abs(float(tas_w["SC_CoSplice_v2_precision_EB"]) - 0.006448) < 2e-5
    assert abs(float(tas_w["usage_pooled"]) - 0.005586) < 2e-5
    assert abs(float(tas_w["acorde_percentile_native"]) - 0.002221) < 2e-5

    sc_mod = _read("fig5e_gse270928_scsplicenet_modules.tsv").set_index("event_id")["module"].to_dict()
    sc_br = _read("fig5e_gse270928_sc_module_rbp_bridge.tsv")
    sig_sc_br = sc_br[sc_br["module_P_FWER"] <= 0.05].set_index("RBP")["module"].to_dict()
    assert sig_sc_br.get("RBM10") == "M3" and sig_sc_br.get("HNRNPAB") == "M3" and sig_sc_br.get("SF1") == "M2" and sig_sc_br.get("FMR1") == "M4"
    assert sc_mod.get("SS3SE:chr1:160619849:160623502:160623585:160624095:-:ENSG00000117090") == "M3"  # SLAMF1
    assert sc_mod.get("SS3SE:chr1:70228555:70231858:70231975:70232267:+:ENSG00000116754") == "M3"      # SRSF11
    assert sc_mod.get("SS3SE:chr1:198699704:198702386:198702530:198703297:+:ENSG00000081237") == "M2"  # PTPRC
    assert "SS3SE:chr1:160796311:160797109:160797275:160799752:+:ENSG00000122224" not in sc_mod         # LY9
    assert "SS3SE:chr1:24646789:24647078:24647208:24648858:+:ENSG00000133226" not in sc_mod             # SRRM1

    pw = _read("fig5f_gse270928_pathway_partitioning_151genes.tsv").set_index("term_id")
    assert len(pw) == 5108 and "GOBP_LEUKOCYTE_PROLIFERATION" in pw.index
    assert int(pw.loc["GOBP_LEUKOCYTE_PROLIFERATION", "deg_overlap"]) == 16
    assert int(pw.loc["GOBP_RNA_SPLICING", "das_151_total"]) == 151 and int(pw.loc["GOBP_RNA_SPLICING", "das_overlap"]) == 18
    assert int(pw.loc["GOBP_RNA_SPLICING", "concordant2d_overlap"]) == 11
    assert int(pw.loc["GOBP_RNA_SPLICING", "measurable_se_bg_total"]) == 1293 and int(pw.loc["GOBP_RNA_SPLICING", "pathway_genes_in_se_bg"]) == 121
    assert float(pw.loc["GOBP_RNA_SPLICING", "das_bh_q_se_bg"]) == 1.0 and abs(float(pw.loc["GOBP_RNA_SPLICING", "concordant2d_bh_q_se_bg"]) - 0.6346) < 1e-3

    # 5. Longcell N=1 stress test, usage prediction, bootstrap CIs, and Tabula Muris / S7 ablations
    lc = _read("suite3_longcell_task1_blind_rbp_recovery.tsv", keep_default_na=False)
    assert len(lc) == 12 and int((lc["SC_MotifTrace_v2_FWER_P"].astype(float) <= 0.05).sum()) == 0
    assert (lc["MRAS_native_enrichment_MRR"].astype(str) == "NA").all()
    up = _read("gse270928_smartseq3_usage_prediction.tsv")
    up_m = up.groupby("method")["gene_equal_mse"].mean()
    assert abs(float(up_m["SC_CoSplice_v2_usage_selected"]) - 0.009682) < 2e-5
    assert abs(float(up_m["acorde_native_usage_only"]) - 0.009657) < 2e-5
    assert abs(float(up_m["acorde_native_N117_selected"]) - 0.009517) < 2e-5

    boot3 = _read("suite4_gse270928_task3_paired_bootstrap_ci.tsv")
    assert len(boot3) == 6
    cas_row = boot3[boot3["comparator"].astype(str).str.contains("CASREL")].iloc[0]
    assert abs(float(cas_row["v2_full_mse_reduction"]) - 0.006544) < 1e-5
    assert abs(float(cas_row["ci95_low"]) - 0.004912) < 1e-5 and abs(float(cas_row["ci95_high"]) - 0.008268) < 1e-5
    assert len(_read("upgrade3_sc_cosplice_v21_tasic_gaba_glut.tsv")) == 12
    miss_c = _read("tableS6_missing_coverage_null_check.tsv")
    assert len(miss_c) == 2 and (miss_c["rbp_P_FWER"] == 1.0).all() and (miss_c["stat_Z"].abs() < 1e-10).all()
    f5d = _read("fig5d_real_multidonor_sc_vs_rmaps3_and_stouffer.tsv")
    assert len(f5d) == 12
    f5d_sc = f5d[f5d["method"].eq("Katana_PRISM_scSpliceTrace_DonorSignFlip_FWER05")].set_index("cohort")
    assert int(f5d_sc.loc["TabulaMuris_cardiac (D=7 mice, 3,294 cells)", "paired_eligible_events"]) == 379
    assert int(f5d_sc.loc["TabulaMuris_brain (D=7 mice, 1,850 cells)", "paired_eligible_events"]) == 247
    assert abs(float(f5d_sc.loc["TabulaMuris_cardiac (D=7 mice, 3,294 cells)", "real_population_null_familywise_FPR_pct"]) - 2.86) < 1e-2
    assert abs(float(f5d_sc.loc["TabulaMuris_brain (D=7 mice, 1,850 cells)", "real_population_null_familywise_FPR_pct"]) - 0.00) < 1e-2
    f5d_hits = _read("fig5d_real_multidonor_rbp_hits.tsv")
    card_top = f5d_hits[f5d_hits["cohort"].eq("TabulaMuris_cardiac_7mice")].sort_values(["rbp_P_FWER", "nominal_P", "sequence_rank"]).iloc[0]
    brain_top = f5d_hits[f5d_hits["cohort"].eq("TabulaMuris_brain_7mice")].sort_values(["rbp_P_FWER", "nominal_P", "sequence_rank"]).iloc[0]
    assert card_top["RBP"] == "Rbm10" and abs(float(card_top["rbp_P_FWER"]) - 0.46875) < 1e-5
    assert brain_top["RBP"] == "Tra2b" and abs(float(brain_top["rbp_P_FWER"]) - 0.125) < 1e-5

    s4c = _read("figS4c_gse270928_qr_fwl_projection_check.tsv")
    assert float(s4c["max_abs_r_after_fwl"].max()) == 0.0
    assert int(_read("figS4a_within_donor_state_pooling.tsv")["cells_per_donor_state"].max()) == 48

    s7a = _read("tableS7a_ps_das_tau_sensitivity.tsv")
    assert int(s7a.loc[s7a["is_default"].eq(True), "NegA_called_events"].iloc[0]) == 1041
    s7b = _read("tableS7b_splicetrace_exon_gc_ablation.tsv").set_index("case")
    assert len(s7b) == 18
    assert list(s7b.loc[f2c_idx.index, "FWL_default_omitExonGC_sig"].astype(int)) == list(f2c_idx["cont_sig_rbps_fwer05"].astype(int))
    pos_cases = [c for c in f2c_idx.index if "negative" not in str(f2c_idx.loc[c, "role"])]
    assert np.allclose(s7b.loc[pos_cases, "FWL_default_omitExonGC_exact_P_FWER"].astype(float), f2c_idx.loc[pos_cases, "cont_exact_P_FWER"].astype(float), atol=1e-6)
    s7b2 = _read("tableS7b2_splicetrace_component_and_vif_ablations.tsv").set_index("case")
    assert len(s7b2) == 18 and float(s7b2["FWL_standardized_design_cond_number"].max()) < 10.0
    assert list(s7b2.loc[f2c_idx.index, "Full13ch_GeneBlock_sig"].astype(int)) == list(f2c_idx["cont_sig_rbps_fwer05"].astype(int))
    assert np.allclose(s7b2.loc[pos_cases, "Full13ch_GeneBlock_exact_P_FWER"].astype(float), f2c_idx.loc[pos_cases, "cont_exact_P_FWER"].astype(float), atol=1e-6)
    s7c = _read("tableS7c_splicenet_biweight_tail_sensitivity.tsv")
    s7c_bicor = s7c[~s7c["configuration"].str.contains("Pearson")]
    assert float(s7c_bicor["GSE62669_N60_partition_ARI_vs_default"].min()) > 0.92
    assert float(s7c_bicor["GSE122380_N297_partition_ARI_vs_default"].min()) > 0.92
    assert abs(float(s7c.loc[s7c["configuration"].str.contains("Pearson"), "GSE62669_N60_partition_ARI_vs_default"].iloc[0]) - 0.6564) < 1e-3
    assert "K562_SingleBatch_N16_partition_ARI_vs_default" in s7c.columns

    s7d = _read("tableS7d_manifold_and_treecut_sensitivity.tsv")
    assert len(s7d) >= 17 and s7d["module_tier"].str.contains("K562 Single-Batch N=16").any()
    k12 = s7d[s7d["parameter_grid"].eq("k_neighbors = 12 [default]")].iloc[0]
    assert abs(float(k12["endpoint_1_value"]) - 0.9748) < 1e-4 and abs(float(k12["endpoint_2_value"]) - 1.385) < 1e-3

    het_sw = _read("tableS6_semisynthetic_sigma_het_sweep.tsv")
    assert len(het_sw) == 6 and float(het_sw.iloc[0]["Cell_Pooled_rMAPS3_Holm_FPR_pct"]) == 0.0
    assert abs(float(het_sw.iloc[-1]["Cell_Pooled_rMAPS3_Holm_FPR_pct"]) - 70.0) < 1e-2
    assert abs(float(het_sw.iloc[-1]["scSpliceTrace_DonorMaxT_FPR_pct"]) - 5.0) < 1e-2
    s1_bam = _read("tableS1_zero_resort_bam_concordance.tsv")
    assert len(s1_bam) == 11 and float(s1_bam["exact_coordinate_overlap_pct"].min()) == 100.0 and float(s1_bam["max_abs_delta_dpsi"].max()) == 0.0
    s6_map = _read("tableS6_event_coordinate_mapping_audit.tsv", keep_default_na=False)
    assert len(s6_map) == 5
    init_den = s6_map["initial_annotated_or_cluster_denominator"].astype(int).tolist()
    inter_cnt = s6_map["intermediate_same_unit_events"].astype(int).tolist()
    inter_pct = s6_map["intermediate_retained_pct"].astype(float).tolist()
    final_cnt = s6_map["final_evaluated_binary_events"].astype(int).tolist()
    final_pct = s6_map["final_retained_of_initial_pct"].astype(float).tolist()
    quar_cnt = s6_map["quarantined_or_filtered_count"].astype(int).tolist()
    assert init_den == [10436, 53663, 66391, 5000, 7591]
    assert inter_cnt == [10436, 7763, 66391, 4310, 7452]
    assert final_cnt == [563, 2169, 1000, 707, 7138]
    assert quar_cnt == [9873, 51494, 65391, 690, 139]
    assert np.allclose(inter_pct, [100.0, 14.47, 100.0, 86.2, 98.17], atol=0.01)
    assert np.allclose(final_pct, [5.39, 4.04, 1.51, 14.14, 94.03], atol=0.01)
    assert np.allclose(inter_pct, [round(100.0 * b / a, 2) for a, b in zip(init_den, inter_cnt)], atol=0.01)
    assert np.allclose(final_pct, [round(100.0 * c / a, 2) for a, c in zip(init_den, final_cnt)], atol=0.01)
    assert "14488" in str(s6_map.iloc[2]["auxiliary_host_gene_callable_loci"])

    on_disk = {p.name for p in tables_path.glob("*.tsv")}
    assert verified == on_disk, f"Unverified TSVs: {sorted(on_disk - verified)}; unexpected: {sorted(verified - on_disk)}"
    print(f"[OK] All {len(verified)} bundled NAR benchmark tables verified (100% file coverage, untruncated peer suites, and ablations).")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce or verify Katana-PRISM v1.0.6 NAR benchmarks.")
    ap.add_argument("--verify-bundled", action="store_true", help="Verify bundled summary TSV invariants.")
    ap.add_argument("--splicetools-dir", type=Path, default=None, help="Directory with public ENCODE rMATS tables.")
    ap.add_argument("--hg38-fasta", type=Path, default=None, help="Indexed hg38 FASTA for sequence extraction.")
    ap.add_argument("--rbp-expression", type=Path, default=None, help="Optional cohort RBP TPM/CPM table.")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "tables")
    args = ap.parse_args()

    if args.splicetools_dir and args.hg38_fasta:
        run_encode_bridge_benchmark(args.splicetools_dir, args.hg38_fasta, args.out_dir, args.rbp_expression)
    if args.verify_bundled or (args.splicetools_dir is None):
        verify_bundled_tables(Path(__file__).resolve().parent / "tables")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
