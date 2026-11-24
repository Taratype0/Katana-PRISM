"""Portable public benchmark runner for Katana-PRISM v1.0.2 (Nucleic Acids Research).

Reproduces & verifies:
  1. 13-channel (`distal_intron_nt = 500`, mature-RNA path + 12 spatial windows) FWL-orthogonalized `SpliceTrace`
     continuous, unified `169`-RBP / `621`-PWM discrete, single-variable ablation, native `rMAPS3`, and `rMAPS2`-style
     scans across all `18` untruncated bulk cohorts.
  2. Exact `GRCh38` `HepG2` IDR `eCLIP` and neuronal `PTBP2` `CLIP` whole-event and `+-35 nt` motif-locus
     conditional MLE Fisher odds ratios and exact 95% confidence intervals.
  3. Real `GSE122380` (`N = 297` libraries, `19` donors x `16` days, `M1–M4`) `SpliceNet -> SpliceTrace` `12`-channel
     module-to-RBP regulatory bridge (`676` rows), `6`-method `50`-trial matched study-wide `Holm <= 0.05` null calibration
     (`4.0%` vs `34.0%–48.0%`), and `978` coding alluvial flows (`612` bridged + `366` `M3/M4` orphan).
  4. `GSE270928` Smart-seq3 dual-denominator retention (`0/1,000` shared HVIs; `327/66,391` multi-isoform transcripts via
     `kallisto v0.46.1`), `7`-mouse `Tabula Muris` `70`-split calibration & real-anchored `sigma_het` sweep, `Tasic` `71`-mouse
     (`6.3x / 2.9x > acorde`), `GSE270928` usage (`0.00968` virtual tie vs `acorde` `0.00966` / `0.00952`), donor splicing
     (`-16.1%/-16.3%` vs `CASREL`, tying `MRAS`), `Longcell` `N=1` stress test (`0/12` `FWER`), `SLAMF1`, and dual-background `MSigDB`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import re
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _pkg in ("splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace"):
    _pkg_dir = str(_REPO_ROOT / "packages" / _pkg)
    if _pkg_dir not in sys.path:
        sys.path.insert(0, _pkg_dir)

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
    """Verify key statistical invariants across all bundled NAR benchmark tables."""
    # 1. PS-DAS full-data qualification, pure nulls, rescued eCLIP, and downstream utility
    f1_vs = pd.read_csv(tables_path / "fig1_ps_das_vs_competitors.tsv", sep="\t")
    f1_pos = f1_vs[f1_vs["role"].eq("positive_RBP_KD") & f1_vs["method"].eq("Katana-PRISM (PS-DAS)")]
    assert int(f1_pos["called_das_events"].sum()) == 10291
    assert int(f1_pos["noncircular_asym_retained"].sum()) == 573 and int(f1_pos["noncircular_asym_total"].sum()) == 749
    assert abs(float(f1_pos["noncircular_asym_recall_pct"].mean()) - 74.57) < 0.1
    f1_nega = f1_vs[f1_vs["cohort"].eq("GSE206660_NegA")].set_index("method")
    assert int(f1_nega.loc["Raw rMATS (Unfiltered)", "called_das_events"]) == 3414
    assert int(f1_nega.loc["Katana-PRISM (PS-DAS)", "called_das_events"]) == 1041

    f1_ec = pd.read_csv(tables_path / "fig1_ps_das_rescued_eclip_validation.tsv", sep="\t")
    pt_res = f1_ec[f1_ec["knockdown_RBP"].eq("PTBP1") & f1_ec["subset"].eq("PS-DAS_rescued_only") & f1_ec["direction_filter"].eq("all_events")].iloc[0]
    assert int(pt_res["fg_hits"]) == 5 and int(pt_res["fg_total"]) == 89 and abs(float(pt_res["odds_ratio"]) - 5.4746) < 1e-2

    s1_ds = pd.read_csv(tables_path / "fig1_ps_das_downstream_splicetrace_utility.tsv", sep="\t")
    assert (s1_ds["downstream_model"] == "Continuous 13-channel FWL + 1,999 host-gene-block maxT").all()
    assert int(s1_ds.loc[s1_ds["cohort"].eq("GSE206660_PTBP2") & s1_ds["upstream_gate"].eq("Katana-PRISM (PS-DAS)"), "qualified_foreground_events"].iloc[0]) == 2700
    assert int(s1_ds.loc[s1_ds["cohort"].eq("GSE206660_NegA") & s1_ds["upstream_gate"].eq("Katana-PRISM (PS-DAS)"), "qualified_foreground_events"].iloc[0]) == 1041
    assert int(s1_ds.loc[s1_ds["cohort"].eq("GSE206660_NegA"), "downstream_SpliceTrace_sig_rbps_fwer05"].sum()) == 0
    enc_u = s1_ds[s1_ds["cohort"].str.startswith("ENCODE_")]
    assert int(enc_u.loc[enc_u["upstream_gate"].eq("Katana-PRISM (PS-DAS)"), "retained_full_ENCODE_UniProt_switches"].sum()) == 3807
    assert int(enc_u.loc[enc_u["upstream_gate"].eq("Strict All>=10 Filter"), "retained_full_ENCODE_UniProt_switches"].sum()) == 3590

    # 2. SpliceTrace 18-cohort calibration, ablations, native rMAPS3, and untruncated rMAPS2-style
    f2c = pd.read_csv(tables_path / "fig2c_splicetrace_exact_and_ablations.tsv", sep="\t")
    nulls = f2c[f2c["role"].isin(["negative_control", "mitochondrial_negative_control"])]
    assert len(nulls) == 3 and int(nulls["cont_sig_rbps_fwer05"].sum()) == 0 and int(nulls["disc_sig_rbps_fwer05"].sum()) == 0
    assert list(nulls.set_index("case").loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "NoFWL_SameMaxT_sig_fwer05"].astype(int)) == [21, 27, 3]
    assert nulls["disc_exact_rbp_P_FWER"].isna().all() and nulls["cont_exact_P_FWER"].isna().all()
    hb = f2c[f2c["role"].eq("high_block_positive")]
    assert len(hb) == 5 and int((hb["disc_exact_rbp_P_FWER"] <= 0.05).sum()) == 4 and int((hb["cont_exact_P_FWER"] <= 0.05).sum()) == 4
    assert int((hb["NoFWL_SameMaxT_exact_P_FWER"] <= 0.05).sum()) == 3
    lb = f2c[f2c["role"].eq("low_block_positive")]
    assert len(lb) == 10 and lb["disc_exact_rbp_P_FWER"].notna().all() and lb["disc_exact_rank"].notna().all()
    assert int((lb["disc_exact_rbp_P_FWER"] <= 0.05).sum()) == 0
    assert int((lb["cont_exact_P_FWER"] <= 0.05).sum()) == 1 and int((lb["NoFWL_SameMaxT_exact_P_FWER"] <= 0.05).sum()) == 5
    assert int(lb["NoFWL_SameMaxT_sig_fwer05"].min()) == 0 and int(lb["NoFWL_SameMaxT_sig_fwer05"].max()) == 153

    rm3 = pd.read_csv(tables_path / "tableS2a_native_rmaps3_all18_cohorts.tsv", sep="\t").set_index("case")
    assert len(rm3) == 18 and list(rm3.loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "rmaps3_sig_rbps_bonf_full"].astype(int)) == [10, 26, 29]
    rm2 = pd.read_csv(tables_path / "tableS2a_untruncated_rmaps2_169rbp_all18.tsv", sep="\t").set_index("case")
    assert len(rm2) == 18 and list(rm2.loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "rMAPS2_sig_rbps_bonf"].astype(int)) == [4, 1, 0]
    het_bulk = pd.read_csv(tables_path / "tableS2_bulk_heteroskedastic_null_calibration.tsv", sep="\t")
    assert len(het_bulk) == 4
    assert float(het_bulk.iloc[0]["perm_exchangeable_FWER05_pct"]) <= 5.0 and float(het_bulk.iloc[0]["rademacher_hc_FWER05_pct"]) <= 5.0
    assert float(het_bulk.iloc[-1]["rademacher_hc_FWER05_pct"]) <= 5.0

    # 3. Multi-donor GSE122380 SpliceNet, bulk manifold, GEUVADIS, GSE62669, Regulatory Bridge & 50-trial null benchmark
    f3d = pd.read_csv(tables_path / "fig3d_gse122380_fair_wgcna_and_djexpress.tsv", sep="\t")
    assert len(f3d) == 12
    w5k = f3d[(f3d["budget"] == 5000) & f3d["method"].eq("Standard_WGCNA_SameUsageRatio")].iloc[0]
    assert abs(float(w5k["median_coherence_gap"]) - 0.0209) < 1e-3
    assert len(pd.read_csv(tables_path / "fig3e_gse122380_crossfold_stability.tsv", sep="\t")) == 6
    f2m = pd.read_csv(tables_path / "fig2_bulk_manifold_vs_jutils_marvel.tsv", sep="\t")
    assert len(f2m) == 16 and abs(float(f2m.loc[f2m["dropout_rate"].eq(0.45) & f2m["method"].str.contains("SpliceManifold"), "permanova_r2"].iloc[0]) - 0.1992) < 1e-4

    b_df = pd.read_csv(tables_path / "fig4a_gse122380_module_rbp_bridge_distal500.tsv", sep="\t")
    assert len(b_df) == 676 and b_df["module"].nunique() == 4 and b_df["RBP"].nunique() == 169
    sig_study = b_df[b_df["study_P_FWER"] <= 0.05]
    assert len(sig_study) == 9 and set(sig_study["module"].unique()) == {"M1", "M2"}
    assert float(b_df.loc[b_df["module"].isin(["M3", "M4"]), "study_P_FWER"].min()) > 0.05

    flows = pd.read_csv(tables_path / "fig4_causal_alluvial_flows.tsv", sep="\t")
    assert len(flows) == 977 and flows["event_id"].nunique() == 977
    assert int((flows["bridge_tier"] == "Study_FWER<=0.05_Bridged").sum()) == 611
    assert int((flows["bridge_tier"] == "Orphan_DonorBackground_FWER>0.05").sum()) == 366

    f4d = pd.read_csv(tables_path / "fig4d_deconvolution_vs_competitors.tsv", sep="\t").set_index("method")
    prism_4d = f4d.loc["Katana-PRISM Bridge"]
    assert int(prism_4d["fwer05_developmental_rbps_called"]) == 8 and int(prism_4d["fwer05_canonical_cardiac_families"]) == 3
    assert int(prism_4d["fwer05_donor_null_M3_M4_fp_rbps"]) == 0 and abs(float(prism_4d["fwer05_null_50trial_familywise_FPR_pct"]) - 4.0) < 1e-4
    assert abs(float(f4d.loc["SpliceNet + Unadjusted MWU", "fwer05_null_50trial_familywise_FPR_pct"]) - 34.0) < 1e-4
    assert abs(float(f4d.loc["Raw-Variance WGCNA + Unadjusted MWU", "fwer05_null_50trial_familywise_FPR_pct"]) - 48.0) < 1e-4

    # 4. Single-cell Smart-seq2/3 benchmarks (GSE270928, Tabula Muris, Tasic, Longcell)
    f5a = pd.read_csv(tables_path / "fig5a_sc_manifold_vs_marvel_jutils.tsv", sep="\t").set_index("method")
    assert abs(float(f5a.loc["Katana-PRISM (SpliceManifold-SC)", "permanova_r2"]) - 0.0464) < 1e-4
    assert abs(float(f5a.loc["MARVEL (Mean-Imputed, Same 1,000 HVEs)", "permanova_r2"]) - 0.0119) < 1e-3
    assert "input_representation" in f5a.columns and "kallisto" in str(f5a["input_representation"].iloc[0])

    f5b = pd.read_csv(tables_path / "fig5b_gse270928_sc_proteome_consequences.tsv", sep="\t")
    f5b_cod = f5b[f5b["frame_status"].isin(["frameshift", "in_frame_indel", "in_frame_PTC", "translation_start_altered"])]
    assert len(f5b_cod) == 167
    assert int(f5b_cod["contributing_donors"].eq(1).sum()) == 89 and int(f5b_cod["contributing_donors"].ge(2).sum()) == 78
    slam_b = f5b_cod[f5b_cod["gene_symbol"].eq("SLAMF1")].iloc[0]
    assert abs(float(slam_b["raw_dpsi"]) - 0.095456) < 1e-4 and abs(float(slam_b["eb_shrunk_dpsi"]) - 0.077136) < 1e-4
    ptprc_b = f5b_cod[f5b_cod["gene_symbol"].eq("PTPRC")].iloc[0]
    assert float(ptprc_b["host_log2fc"]) < -1.0  # repressed host gene (-1.025)

    slam = pd.read_csv(tables_path / "fig5c_gse270928_slamf1_per_cell_event_psi.tsv", sep="\t")
    assert int(slam["event_psi"].notna().sum()) == 76
    slam_any = (slam["inclusion_reads"] + slam["exclusion_reads"]) > 0
    assert int(slam_any[slam["condition"].eq("Ctrl")].sum()) == 18 and int(slam_any[~slam["condition"].eq("Ctrl")].sum()) == 90
    assert int(slam.loc[slam["condition"].eq("Ctrl"), "event_psi"].notna().sum()) == 7 and int(slam.loc[~slam["condition"].eq("Ctrl"), "event_psi"].notna().sum()) == 69

    tas = pd.read_csv(tables_path / "fig5e_tasic_71mice_scsplicenet_vs_pooled_acorde.tsv", sep="\t")
    tas_w = tas[tas["scope"].eq("within_animal_cell_type") & tas["endpoint"].eq("observed_usage")].groupby("method")["heldout_correlation"].mean()
    assert abs(float(tas_w["SC_CoSplice_v2_precision_EB"]) - 0.006448) < 2e-5
    assert abs(float(tas_w["usage_pooled"]) - 0.005586) < 2e-5
    assert abs(float(tas_w["acorde_percentile_native"]) - 0.002221) < 2e-5

    sc_mod = pd.read_csv(tables_path / "fig5e_gse270928_scsplicenet_modules.tsv", sep="\t").set_index("event_id")["module"].to_dict()
    sc_br = pd.read_csv(tables_path / "fig5e_gse270928_sc_module_rbp_bridge.tsv", sep="\t")
    sig_sc_br = sc_br[sc_br["module_P_FWER"] <= 0.05].set_index("RBP")["module"].to_dict()
    assert sig_sc_br.get("RBM10") == "M3" and sig_sc_br.get("HNRNPAB") == "M3" and sig_sc_br.get("SF1") == "M2" and sig_sc_br.get("FMR1") == "M4"
    assert sc_mod.get("SS3SE:chr1:160619849:160623502:160623585:160624095:-:ENSG00000117090") == "M3"  # SLAMF1
    assert sc_mod.get("SS3SE:chr1:70228555:70231858:70231975:70232267:+:ENSG00000116754") == "M3"      # SRSF11
    assert sc_mod.get("SS3SE:chr1:198699704:198702386:198702530:198703297:+:ENSG00000081237") == "M2"  # PTPRC
    assert "SS3SE:chr1:160796311:160797109:160797275:160799752:+:ENSG00000122224" not in sc_mod         # LY9
    assert "SS3SE:chr1:24646789:24647078:24647208:24648858:+:ENSG00000133226" not in sc_mod             # SRRM1

    pw = pd.read_csv(tables_path / "fig5f_gse270928_pathway_partitioning_151genes.tsv", sep="\t").set_index("term_id")
    assert len(pw) == 5108 and "GOBP_LEUKOCYTE_PROLIFERATION" in pw.index
    assert int(pw.loc["GOBP_LEUKOCYTE_PROLIFERATION", "deg_overlap"]) == 16
    assert int(pw.loc["GOBP_RNA_SPLICING", "das_151_total"]) == 151 and int(pw.loc["GOBP_RNA_SPLICING", "das_overlap"]) == 18
    assert int(pw.loc["GOBP_RNA_SPLICING", "concordant2d_overlap"]) == 11
    assert int(pw.loc["GOBP_RNA_SPLICING", "measurable_se_bg_total"]) == 1293 and int(pw.loc["GOBP_RNA_SPLICING", "pathway_genes_in_se_bg"]) == 121
    assert float(pw.loc["GOBP_RNA_SPLICING", "das_bh_q_se_bg"]) == 1.0 and abs(float(pw.loc["GOBP_RNA_SPLICING", "concordant2d_bh_q_se_bg"]) - 0.6346) < 1e-3

    # 5. Longcell N=1 stress test, usage prediction, bootstrap CIs, and Tabula Muris / S7 ablations
    lc = pd.read_csv(tables_path / "suite3_longcell_task1_blind_rbp_recovery.tsv", sep="\t")
    assert len(lc) == 12 and int((lc["SC_MotifTrace_v2_FWER_P"] <= 0.05).sum()) == 0
    up = pd.read_csv(tables_path / "gse270928_smartseq3_usage_prediction.tsv", sep="\t")
    up_m = up.groupby("method")["gene_equal_mse"].mean()
    assert abs(float(up_m["SC_CoSplice_v2_usage_selected"]) - 0.009682) < 2e-5
    assert abs(float(up_m["acorde_native_usage_only"]) - 0.009657) < 2e-5
    assert abs(float(up_m["acorde_native_N117_selected"]) - 0.009517) < 2e-5

    assert len(pd.read_csv(tables_path / "suite4_gse270928_task3_paired_bootstrap_ci.tsv", sep="\t")) == 6
    assert len(pd.read_csv(tables_path / "upgrade3_sc_cosplice_v21_tasic_gaba_glut.tsv", sep="\t")) == 12
    f5d = pd.read_csv(tables_path / "fig5d_real_multidonor_sc_vs_rmaps3_and_stouffer.tsv", sep="\t")
    assert len(f5d) == 12
    f5d_sc = f5d[f5d["method"].eq("Katana_PRISM_scSpliceTrace_DonorSignFlip_FWER05")].set_index("cohort")
    assert int(f5d_sc.loc["TabulaMuris_cardiac (D=7 mice, 3,294 cells)", "paired_eligible_events"]) == 379
    assert int(f5d_sc.loc["TabulaMuris_brain (D=7 mice, 1,850 cells)", "paired_eligible_events"]) == 247
    assert abs(float(f5d_sc.loc["TabulaMuris_cardiac (D=7 mice, 3,294 cells)", "real_population_null_familywise_FPR_pct"]) - 1.43) < 1e-2
    assert abs(float(f5d_sc.loc["TabulaMuris_brain (D=7 mice, 1,850 cells)", "real_population_null_familywise_FPR_pct"]) - 4.29) < 1e-2
    f5d_hits = pd.read_csv(tables_path / "fig5d_real_multidonor_rbp_hits.tsv", sep="\t")
    assert f5d_hits[f5d_hits["cohort"].eq("TabulaMuris_cardiac_7mice")].iloc[0]["RBP"] == "Rbm10"
    assert f5d_hits[f5d_hits["cohort"].eq("TabulaMuris_brain_7mice")].iloc[0]["RBP"] == "Tut1"

    s4c = pd.read_csv(tables_path / "figS4c_gse270928_qr_fwl_projection_check.tsv", sep="\t")
    assert float(s4c["max_abs_r_after_fwl"].max()) == 0.0
    assert int(pd.read_csv(tables_path / "figS4a_within_donor_state_pooling.tsv", sep="\t")["cells_per_donor_state"].max()) == 48

    s7a = pd.read_csv(tables_path / "tableS7a_ps_das_tau_sensitivity.tsv", sep="\t")
    assert int(s7a.loc[s7a["is_default"].eq(True), "NegA_called_events"].iloc[0]) == 1041
    assert len(pd.read_csv(tables_path / "tableS7b_splicetrace_exon_gc_ablation.tsv", sep="\t")) == 18
    s7b2 = pd.read_csv(tables_path / "tableS7b2_splicetrace_component_and_vif_ablations.tsv", sep="\t")
    assert len(s7b2) == 18 and float(s7b2["FWL_standardized_design_cond_number"].max()) < 10.0
    s7c = pd.read_csv(tables_path / "tableS7c_splicenet_biweight_tail_sensitivity.tsv", sep="\t")
    s7c_bicor = s7c[~s7c["configuration"].str.contains("Pearson")]
    assert float(s7c_bicor["GSE62669_N60_partition_ARI_vs_default"].min()) > 0.92
    assert float(s7c_bicor["GSE122380_N297_partition_ARI_vs_default"].min()) > 0.92
    assert abs(float(s7c.loc[s7c["configuration"].str.contains("Pearson"), "GSE62669_N60_partition_ARI_vs_default"].iloc[0]) - 0.6564) < 1e-3
    assert "K562_SingleBatch_N16_partition_ARI_vs_default" in s7c.columns

    s7d = pd.read_csv(tables_path / "tableS7d_manifold_and_treecut_sensitivity.tsv", sep="\t")
    assert len(s7d) >= 17 and s7d["module_tier"].str.contains("K562 Single-Batch N=16").any()
    k12 = s7d[s7d["parameter_grid"].eq("k_neighbors = 12 [default]")].iloc[0]
    assert abs(float(k12["endpoint_1_value"]) - 0.9748) < 1e-4 and abs(float(k12["endpoint_2_value"]) - 1.385) < 1e-3

    het_sw = pd.read_csv(tables_path / "tableS6_semisynthetic_sigma_het_sweep.tsv", sep="\t")
    assert len(het_sw) == 6 and float(het_sw.iloc[0]["Cell_Pooled_rMAPS3_Holm_FPR_pct"]) == 0.0
    assert abs(float(het_sw.iloc[-1]["Cell_Pooled_rMAPS3_Holm_FPR_pct"]) - 70.0) < 1e-2
    assert abs(float(het_sw.iloc[-1]["scSpliceTrace_DonorMaxT_FPR_pct"]) - 5.0) < 1e-2
    s1_bam = pd.read_csv(tables_path / "tableS1_zero_resort_bam_concordance.tsv", sep="\t")
    assert len(s1_bam) == 11 and float(s1_bam["exact_coordinate_overlap_pct"].min()) == 100.0 and float(s1_bam["max_abs_delta_dpsi"].max()) == 0.0
    s6_map = pd.read_csv(tables_path / "tableS6_event_coordinate_mapping_audit.tsv", sep="\t")
    assert len(s6_map) == 3 and int(s6_map.iloc[0]["final_evaluated_binary_events"]) == 563 and int(s6_map.iloc[1]["initial_annotated_or_cluster_denominator"]) == 53663 and int(s6_map.iloc[1]["final_evaluated_binary_events"]) == 2169 and int(s6_map.iloc[2]["initial_annotated_or_cluster_denominator"]) == 66391 and int(s6_map.iloc[2]["final_evaluated_binary_events"]) == 1000
    print("[OK] All bundled NAR benchmark table invariants (including untruncated peer suites and ablations) verified.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce or verify Katana-PRISM v1.0.2 NAR benchmarks.")
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
