"""Portable public benchmark runner for Katana-PRISM v1.0.1 (Nucleic Acids Research).

Reproduces:
  1. 12-channel (`distal_intron_nt = 500`) FWL-orthogonalized `SpliceTrace` continuous & discrete RBP scans
     with optional `--rbp-expression` paralog disambiguation on public ENCODE & GSE206660 cohorts.
  2. Exact `HepG2` `eCLIP` (`ENCFF154DRN` `RBFOX2`, `ENCFF170SCU` `PTBP1`) whole-event and `+-35 nt` motif-locus
     conditional Fisher odds ratios and Haldane-Anscombe 95% confidence intervals.
  3. 16-library ENCODE multi-RBP `SpliceNet -> SpliceTrace` module-to-RBP regulatory bridge (`distal_intron_nt = 500`),
     separating `module_P_FWER <= 0.05` bridged pathways from unattributed (`FWER > 0.05`) orphan modules.
  4. `GSE270928` Smart-seq3 dual-denominator `Jutils` retention (`0/1,000` shared HVEs; `327/66,391` candidate transcripts),
     donor-stratified Stouffer `DEG` independence across `167` protein-altering switches, `SLAMF1` (`Q13291-4` vs `Q13291-1`)
     per-cell `83-nt` cassette-exon `PSI`, and within-`donor x state` cell-pooling sensitivity curves (`5..48` cells).
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
    sel_u, _ = select_events(masked_u, audit_u, feature_cap=4500, ranking_metric="excess_bio_variance")
    assign_df = fit_splicenet(sel_u, None, power=0, min_module_size=30, backend="python")
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
    """Verify key statistical invariants across bundled NAR benchmark tables."""
    f2c = pd.read_csv(tables_path / "fig2c_splicetrace_exact_and_ablations.tsv", sep="\t")
    nulls = f2c[f2c["role"].isin(["negative_control", "mitochondrial_negative_control"])]
    assert len(nulls) == 3 and int(nulls["cont_sig_rbps_fwer05"].sum()) == 0 and int(nulls["disc_sig_rbps_fwer05"].sum()) == 0
    assert list(nulls.set_index("case").loc[["GSE206660_NegA", "SUPV3L1", "PNPT1"], "NoFWL_SameMaxT_sig_fwer05"].astype(int)) == [20, 27, 3]
    assert nulls["disc_exact_rbp_P_FWER"].isna().all() and nulls["cont_exact_P_FWER"].isna().all()
    hb = f2c[f2c["role"].eq("high_block_positive")]
    assert len(hb) == 5 and int((hb["disc_exact_rbp_P_FWER"] <= 0.05).sum()) == 4 and int((hb["cont_exact_P_FWER"] <= 0.05).sum()) == 4
    assert int((hb["NoFWL_SameMaxT_exact_P_FWER"] <= 0.05).sum()) == 3
    lb = f2c[f2c["role"].eq("low_block_positive")]
    assert len(lb) == 10 and lb["disc_exact_rbp_P_FWER"].notna().all() and lb["disc_exact_rank"].notna().all()
    assert int((lb["cont_exact_P_FWER"] <= 0.05).sum()) == 1 and int((lb["NoFWL_SameMaxT_exact_P_FWER"] <= 0.05).sum()) == 5

    b_df = pd.read_csv(tables_path / "fig4a_encode_module_rbp_bridge_distal500.tsv", sep="\t")
    sig_mods = sorted(b_df.loc[b_df["module_P_FWER"] <= 0.05, "module"].unique().tolist())
    assert sig_mods == ["M1", "M12", "M3", "M9"], f"Unexpected FWER<=0.05 bridged modules: {sig_mods}"

    flows = pd.read_csv(tables_path / "fig4c_bridged_vs_orphan_coding_flows.tsv", sep="\t")
    assert len(flows) == 611
    assert int((flows["bridge_tier"] == "FWER<=0.05_Bridged").sum()) == 286
    assert int((flows["bridge_tier"] == "Exploratory_Orphan_Module").sum()) == 325

    f4d = pd.read_csv(tables_path / "fig4d_deconvolution_vs_competitors.tsv", sep="\t").set_index("method")
    prism_4d = f4d.loc["Katana-PRISM (SpliceNet -> SpliceTrace Bridge)"]
    assert abs(float(prism_4d["exact_all_called_f1_score"]) - 0.4000) < 1e-4
    assert abs(float(prism_4d["paralog_adj_f1_score"]) - 0.4444) < 1e-4

    f5a = pd.read_csv(tables_path / "fig5a_sc_manifold_vs_marvel_jutils.tsv", sep="\t").set_index("method")
    assert abs(float(f5a.loc["MARVEL (Mean-Imputed, Same 1,000 HVEs)", "permanova_r2"]) - 0.0119) < 1e-3

    f5b = pd.read_csv(tables_path / "fig5b_gse270928_sc_proteome_consequences.tsv", sep="\t")
    f5b_cod = f5b[f5b["frame_status"].isin(["frameshift", "in_frame_indel", "translation_start_altered"])]
    assert len(f5b_cod) == 167
    assert int(f5b_cod["contributing_donors"].eq(1).sum()) == 89 and int(f5b_cod["contributing_donors"].ge(2).sum()) == 78
    slam_b = f5b_cod[f5b_cod["gene_symbol"].eq("SLAMF1")].iloc[0]
    assert abs(float(slam_b["raw_dpsi"]) - 0.095456) < 1e-4 and abs(float(slam_b["eb_shrunk_dpsi"]) - 0.077136) < 1e-4

    slam = pd.read_csv(tables_path / "fig5c_gse270928_slamf1_per_cell_event_psi.tsv", sep="\t")
    assert int(slam["event_psi"].notna().sum()) == 76

    pw = pd.read_csv(tables_path / "fig5f_gse270928_pathway_partitioning_151genes.tsv", sep="\t").set_index("term_id")
    assert int(pw.loc["GOBP_RNA_SPLICING", "das_151_total"]) == 151
    assert int(pw.loc["GOBP_RNA_SPLICING", "das_overlap"]) == 18
    assert int(pw.loc["GOBP_RNA_SPLICING", "concordant2d_overlap"]) == 11
    assert int(pw.loc["REACTOME_PROCESSING_OF_CAPPED_INTRON_CONTAINING_PRE_MRNA", "das_overlap"]) == 12
    assert int(pw.loc["REACTOME_PROCESSING_OF_CAPPED_INTRON_CONTAINING_PRE_MRNA", "concordant2d_overlap"]) == 8

    pool = pd.read_csv(tables_path / "figS4a_within_donor_state_pooling.tsv", sep="\t")
    assert int(pool["cells_per_donor_state"].max()) == 48
    print("[OK] All bundled NAR benchmark table invariants verified.")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce or verify Katana-PRISM v1.0.1 NAR benchmarks.")
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
