"""Regression tests for paired-subject bulk support across SpliceManifold, SpliceNet, and Katana-Studio reports."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for pkg_dir in ("splicemanifold", "splicenet", "splicetrace", "scsplicetrace", "scsplicenet", "katana_studio"):
    p = ROOT / "packages" / pkg_dir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from katana_next.science.psi_landscape import (  # noqa: E402
    compute_beyond_jutils_landscape,
    permanova_subject_blocked as studio_permanova_subject_blocked,
    render_editorial_psi_pca,
)
from katana_studio.network_explorer import build_chapter6_splicenet_suite  # noqa: E402
from katana_studio.report_builder import run_bulk  # noqa: E402
from splicemanifold.cli import main as splicemanifold_main  # noqa: E402
from splicemanifold.manifold import (  # noqa: E402
    compute_splicing_manifold,
    permanova_subject_blocked,
    render_psi_pca,
)
from splicenet.cli import main as splicenet_main  # noqa: E402


def _make_lois_like_psi_and_meta() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    rng = np.random.default_rng(7)
    sids = ["S1_C", "S2_C", "S3_C", "S4_C", "S1_T", "S2_T", "S3_T", "S4_T"]
    meta = [
        {"sample_id": f"S{i}_C", "biological_sample_id": f"Ind{i}", "donor_id": f"Ind{i}", "condition": "Ctrl", "cell_type": "Ctrl"}
        for i in range(1, 5)
    ] + [
        {"sample_id": f"S{i}_T", "biological_sample_id": f"Ind{i}", "donor_id": f"Ind{i}", "condition": "Ab", "cell_type": "Ab"}
        for i in range(1, 5)
    ]
    subj_offsets = np.array([-0.18, -0.06, 0.08, 0.16])
    data = {}
    sup = {}
    eids = [f"SE:chr1:+:{100 + i * 50}:{200 + i * 50}" for i in range(40)]
    for j, sid in enumerate(sids):
        subj_i = j % 4
        is_ab = j >= 4
        base = np.linspace(0.30, 0.70, 40) + subj_offsets[subj_i] + (0.14 if is_ab else -0.14)
        vals = np.clip(base + rng.normal(0.0, 0.015, 40), 0.02, 0.98)
        data[sid] = vals
        sup[sid] = np.full(40, 80.0)
    return pd.DataFrame(data, index=eids), pd.DataFrame(sup, index=eids), meta


def _make_split_plot_psi_and_meta() -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    rng = np.random.default_rng(19)
    specs = [
        ("KA_W1", "KA_rep1", "WT_DEL_KA", -0.12, 0.0),
        ("KA_W2", "KA_rep2", "WT_DEL_KA", 0.14, 0.0),
        ("KA_M1", "KA_rep1", "MUT_DEL_KA", -0.12, 0.22),
        ("KA_M2", "KA_rep2", "MUT_DEL_KA", 0.14, 0.22),
        ("PBS_W1", "PBS_rep1", "WT_DEL_PBS", -0.20, -0.08),
        ("PBS_W2", "PBS_rep2", "WT_DEL_PBS", 0.00, -0.08),
        ("PBS_W3", "PBS_rep3", "WT_DEL_PBS", 0.18, -0.08),
        ("PBS_M1", "PBS_rep1", "MUT_DEL_PBS", -0.20, 0.06),
        ("PBS_M2", "PBS_rep2", "MUT_DEL_PBS", 0.00, 0.06),
        ("PBS_M3", "PBS_rep3", "MUT_DEL_PBS", 0.18, 0.06),
    ]
    meta = [
        {"sample_id": sid, "biological_sample_id": bio, "donor_id": bio, "condition": cond, "cell_type": cond}
        for sid, bio, cond, _, _ in specs
    ]
    eids = [f"SE:chr2:-:{500 + i * 50}:{600 + i * 50}" for i in range(45)]
    data = {}
    sup = {}
    for sid, _, _, subj_off, trt_off in specs:
        base = np.linspace(0.28, 0.72, 45) + subj_off + trt_off
        data[sid] = np.clip(base + rng.normal(0.0, 0.012, 45), 0.02, 0.98)
        sup[sid] = np.full(45, 70.0)
    return pd.DataFrame(data, index=eids), pd.DataFrame(sup, index=eids), meta


def _write_rmats_two_group(
    rdir: Path,
    *,
    p1_vals: np.ndarray,
    p2_vals: np.ndarray,
    dpsi_sign: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    rdir.mkdir(parents=True, exist_ok=True)
    n1 = len(p1_vals)
    n2 = len(p2_vals)
    n_events = 36
    rows = []
    for i in range(n_events):
        is_sig = i < 8
        shift = 0.22 * dpsi_sign if is_sig else 0.01
        p1 = np.clip(p1_vals + (0.04 * (i % 5) - 0.08) + shift + rng.normal(0.0, 0.01, n1), 0.05, 0.95)
        p2 = np.clip(p2_vals + (0.04 * (i % 5) - 0.08) + rng.normal(0.0, 0.01, n2), 0.05, 0.95)
        tot1 = np.full(n1, 80, dtype=int)
        tot2 = np.full(n2, 80, dtype=int)
        ijc1 = np.maximum(1, np.round(p1 * tot1).astype(int))
        sjc1 = np.maximum(1, tot1 - ijc1)
        ijc2 = np.maximum(1, np.round(p2 * tot2).astype(int))
        sjc2 = np.maximum(1, tot2 - ijc2)
        dpsi = round(float(p1.mean() - p2.mean()), 4)
        base = 1000 + i * 700
        g_idx = i + 1
        rows.append(
            {
                "ID": g_idx,
                "GeneID": f'"ENSMUSG{g_idx:011d}"',
                "geneSymbol": f'"Pgene{g_idx}"',
                "chr": "chr1",
                "strand": "+",
                "exonStart_0base": base + 200,
                "exonEnd": base + 290,
                "upstreamES": base,
                "upstreamEE": base + 90,
                "downstreamES": base + 420,
                "downstreamEE": base + 540,
                "ID.1": g_idx,
                "IJC_SAMPLE_1": ",".join(map(str, ijc1)),
                "SJC_SAMPLE_1": ",".join(map(str, sjc1)),
                "IJC_SAMPLE_2": ",".join(map(str, ijc2)),
                "SJC_SAMPLE_2": ",".join(map(str, sjc2)),
                "IncFormLen": 148,
                "SkipFormLen": 99,
                "PValue": 1e-5 if is_sig else 0.72,
                "FDR": 0.002 if is_sig else 0.88,
                "IncLevel1": ",".join(f"{x:.3f}" for x in p1),
                "IncLevel2": ",".join(f"{x:.3f}" for x in p2),
                "IncLevelDifference": dpsi,
            }
        )
    pd.DataFrame(rows).to_csv(rdir / "SE.MATS.JCEC.txt", sep="\t", index=False)


def test_permanova_subject_blocked_and_manifold_paired_cohorts(tmp_path: Path) -> None:
    psi_l, sup_l, meta_l = _make_lois_like_psi_and_meta()
    land_l = compute_splicing_manifold(psi_l, meta_l, depth_matrix=sup_l, run_umap=False)
    assert land_l["available"] is True
    assert land_l["permanova_strata"] == "subject_paired"
    assert land_l["paired_subjects"] == 4
    assert land_l["permanova_exact_perms"] == 16
    assert land_l["permanova_within_subject_r2"] > 0.75
    assert land_l["permanova_subject_r2"] > 0.25
    assert land_l["permanova_p_within_subject"] == pytest.approx(0.125)
    assert land_l["permanova_p"] == pytest.approx(0.125)

    blk_direct = studio_permanova_subject_blocked(
        land_l["_internal"]["D_prec"],
        [m["condition"] for m in meta_l],
        [m["donor_id"] for m in meta_l],
    )
    assert blk_direct["exact"] is True
    assert blk_direct["n_perms"] == 16
    assert blk_direct["within_subject_r2"] == pytest.approx(land_l["permanova_within_subject_r2"])

    pca_png = tmp_path / "lois_pca.png"
    render_psi_pca(land_l, pca_png, pca_png.with_suffix(".svg"))
    assert pca_png.is_file() and pca_png.stat().st_size > 0
    assert pca_png.with_suffix(".svg").is_file()

    # Editorial landscape parity
    land_ed = compute_beyond_jutils_landscape(psi_l, meta_l, depth_matrix=sup_l, run_umap=False)
    assert land_ed["permanova_strata"] == "subject_paired"
    assert land_ed["permanova_within_subject_r2"] > 0.75
    ed_pca_png = tmp_path / "lois_ed_pca.png"
    render_editorial_psi_pca(land_ed, ed_pca_png, ed_pca_png.with_suffix(".svg"))
    assert ed_pca_png.is_file() and ed_pca_png.stat().st_size > 0

    # Split-plot YTHDC1_rescue-like cohort (2 KA pairs + 3 PBS pairs = 5 paired subjects across 4 arms)
    psi_sp, sup_sp, meta_sp = _make_split_plot_psi_and_meta()
    land_sp = compute_splicing_manifold(psi_sp, meta_sp, depth_matrix=sup_sp, run_umap=False)
    assert land_sp["permanova_strata"] == "split_plot_paired"
    assert land_sp["paired_subjects"] == 5
    assert land_sp["permanova_exact_perms"] == 32
    assert land_sp["permanova_within_subject_r2"] > 0.70
    sp_png = tmp_path / "split_plot_pca.png"
    render_psi_pca(land_sp, sp_png, sp_png.with_suffix(".svg"))
    assert sp_png.is_file() and sp_png.stat().st_size > 0


def test_splicenet_paired_ttest_and_rm_anova_vs_mixed_and_unpaired(tmp_path: Path) -> None:
    # 1. Crossed paired cohort mimicking Lois M3: large baseline variation + consistent negative paired shift
    samples_l = [
        {"sample_id": "C1", "biological_sample_id": "Ind1", "condition": "Ctrl"},
        {"sample_id": "C2", "biological_sample_id": "Ind2", "condition": "Ctrl"},
        {"sample_id": "C3", "biological_sample_id": "Ind3", "condition": "Ctrl"},
        {"sample_id": "C4", "biological_sample_id": "Ind4", "condition": "Ctrl"},
        {"sample_id": "A1", "biological_sample_id": "Ind1", "condition": "Ab"},
        {"sample_id": "A2", "biological_sample_id": "Ind2", "condition": "Ab"},
        {"sample_id": "A3", "biological_sample_id": "Ind3", "condition": "Ab"},
        {"sample_id": "A4", "biological_sample_id": "Ind4", "condition": "Ab"},
    ]
    ctrl_vals = np.array([-1.50, -0.20, 0.80, 1.90])
    ab_vals = ctrl_vals + np.array([-1.00, -1.30, -1.20, -0.95])
    # Sanity check: unpaired Welch is NS while paired t-test is significant
    _, p_welch = stats.ttest_ind(ab_vals, ctrl_vals, equal_var=False)
    t_pair, p_pair = stats.ttest_rel(ab_vals, ctrl_vals)
    assert p_welch > 0.15
    assert p_pair < 0.01

    scores_l = pd.DataFrame(
        {
            "M1": [0.1, -0.2, 0.3, -0.1, 0.2, -0.1, 0.1, -0.2],
            "M3": np.concatenate([ctrl_vals, ab_vals]),
        },
        index=[s["sample_id"] for s in samples_l],
    )
    mod_df_l = pd.DataFrame(
        [
            {"event_key": f"SE:chr1:+:{i}:{i+50}", "module": 1 if i < 6 else 3, "gene_id": f"G{i}", "gene_symbol": f"Sym{i}", "kME": 0.85}
            for i in range(12)
        ]
    )
    psi_dummy_l = pd.DataFrame(
        np.tile(np.linspace(0.2, 0.8, 8), (12, 1)),
        index=mod_df_l["event_key"],
        columns=[s["sample_id"] for s in samples_l],
    )
    contrasts_l = [
        {
            "contrast_id": "Ab_vs_Ctrl",
            "label": "Ab vs Ctrl",
            "numerator": {"label": "Ab", "sample_ids": ["A1", "A2", "A3", "A4"]},
            "denominator": {"label": "Ctrl", "sample_ids": ["C1", "C2", "C3", "C4"]},
            "paired": True,
            "paired_subjects": 4,
            "design": "~ subject + condition",
        }
    ]
    out_l = tmp_path / "net_lois"
    build_chapter6_splicenet_suite(
        mod_df_raw=mod_df_l,
        scores_df_raw=scores_l,
        samples_list=samples_l,
        contrasts_spec=contrasts_l,
        psi_events_by_units=psi_dummy_l,
        gene_sym_map={},
        outdir=out_l,
        panels=[],
        tables=[],
        network_browsers=[],
        species="mouse",
        is_single_cell=False,
    )
    assoc_l = pd.read_csv(out_l / "outputs/network/report_tables/module_associations_complete.tsv", sep="\t")
    m3_con = assoc_l[(assoc_l["module"] == "M3") & (assoc_l["kind"] == "contrast")].iloc[0]
    m3_om = assoc_l[(assoc_l["module"] == "M3") & (assoc_l["kind"] == "omnibus")].iloc[0]
    assert m3_con["test_type"] == "paired_t"
    assert int(m3_con["paired_subjects"]) == 4
    assert int(m3_con["biological_units"]) == 4
    assert m3_con["p_value"] == pytest.approx(p_pair, rel=1e-4)
    assert m3_con["statistic"] == pytest.approx(t_pair, rel=1e-4)
    assert m3_con["q_value"] < 0.05
    assert m3_om["test_type"] == "paired_rm_anova"
    assert int(m3_om["paired_subjects"]) == 4
    assert m3_om["p_value"] == pytest.approx(p_pair, rel=1e-4)
    assert m3_om["effect"] == pytest.approx(t_pair ** 2, rel=1e-3)

    # 2. Mixed split-plot cohort: 2 paired contrasts + 2 unpaired contrasts
    samples_m = [
        {"sample_id": "R3", "biological_sample_id": "KA_rep1", "condition": "WT_DEL_KA"},
        {"sample_id": "R5", "biological_sample_id": "KA_rep2", "condition": "WT_DEL_KA"},
        {"sample_id": "R4", "biological_sample_id": "KA_rep1", "condition": "MUT_DEL_KA"},
        {"sample_id": "R6", "biological_sample_id": "KA_rep2", "condition": "MUT_DEL_KA"},
        {"sample_id": "R1", "biological_sample_id": "PBS_rep1", "condition": "WT_DEL_PBS"},
        {"sample_id": "R7", "biological_sample_id": "PBS_rep2", "condition": "WT_DEL_PBS"},
        {"sample_id": "R11", "biological_sample_id": "PBS_rep3", "condition": "WT_DEL_PBS"},
        {"sample_id": "R2", "biological_sample_id": "PBS_rep1", "condition": "MUT_DEL_PBS"},
        {"sample_id": "R8", "biological_sample_id": "PBS_rep2", "condition": "MUT_DEL_PBS"},
        {"sample_id": "R12", "biological_sample_id": "PBS_rep3", "condition": "MUT_DEL_PBS"},
    ]
    scores_m = pd.DataFrame(
        {"M1": [-0.5, 0.5, -1.2, -0.3, -1.4, 0.2, 0.8, -1.9, -0.1, 0.4]},
        index=[s["sample_id"] for s in samples_m],
    )
    psi_dummy_m = pd.DataFrame(
        np.tile(np.linspace(0.2, 0.8, 10), (12, 1)),
        index=mod_df_l["event_key"],
        columns=[s["sample_id"] for s in samples_m],
    )
    contrasts_m = [
        {
            "contrast_id": "MUT_DEL_KA_vs_WT_DEL_KA",
            "label": "MUT_DEL_KA vs WT_DEL_KA",
            "numerator": {"label": "MUT_DEL_KA", "sample_ids": ["R4", "R6"]},
            "denominator": {"label": "WT_DEL_KA", "sample_ids": ["R3", "R5"]},
            "paired": True,
        },
        {
            "contrast_id": "MUT_DEL_PBS_vs_WT_DEL_PBS",
            "label": "MUT_DEL_PBS vs WT_DEL_PBS",
            "numerator": {"label": "MUT_DEL_PBS", "sample_ids": ["R2", "R8", "R12"]},
            "denominator": {"label": "WT_DEL_PBS", "sample_ids": ["R1", "R7", "R11"]},
            "paired": True,
        },
        {
            "contrast_id": "WT_DEL_KA_vs_WT_DEL_PBS",
            "label": "WT_DEL_KA vs WT_DEL_PBS",
            "numerator": {"label": "WT_DEL_KA", "sample_ids": ["R3", "R5"]},
            "denominator": {"label": "WT_DEL_PBS", "sample_ids": ["R1", "R7", "R11"]},
            "paired": False,
        },
        {
            "contrast_id": "MUT_DEL_KA_vs_MUT_DEL_PBS",
            "label": "MUT_DEL_KA vs MUT_DEL_PBS",
            "numerator": {"label": "MUT_DEL_KA", "sample_ids": ["R4", "R6"]},
            "denominator": {"label": "MUT_DEL_PBS", "sample_ids": ["R2", "R8", "R12"]},
            "paired": False,
        },
    ]
    out_m = tmp_path / "net_mixed"
    build_chapter6_splicenet_suite(
        mod_df_raw=mod_df_l[mod_df_l["module"] == 1],
        scores_df_raw=scores_m,
        samples_list=samples_m,
        contrasts_spec=contrasts_m,
        psi_events_by_units=psi_dummy_m.iloc[:6],
        gene_sym_map={},
        outdir=out_m,
        panels=[],
        tables=[],
        network_browsers=[],
        species="mouse",
        is_single_cell=False,
    )
    assoc_m = pd.read_csv(out_m / "outputs/network/report_tables/module_associations_complete.tsv", sep="\t")
    assert assoc_m.loc[assoc_m["kind"] == "omnibus", "test_type"].iloc[0] == "oneway_anova"
    by_cid = assoc_m[assoc_m["kind"] == "contrast"].set_index("contrast_id")
    assert by_cid.loc["MUT_DEL_KA_vs_WT_DEL_KA", "test_type"] == "paired_t"
    assert int(by_cid.loc["MUT_DEL_KA_vs_WT_DEL_KA", "paired_subjects"]) == 2
    assert by_cid.loc["MUT_DEL_PBS_vs_WT_DEL_PBS", "test_type"] == "paired_t"
    assert int(by_cid.loc["MUT_DEL_PBS_vs_WT_DEL_PBS", "paired_subjects"]) == 3
    assert by_cid.loc["WT_DEL_KA_vs_WT_DEL_PBS", "test_type"] == "welch_t"
    assert int(by_cid.loc["WT_DEL_KA_vs_WT_DEL_PBS", "paired_subjects"]) == 0
    assert by_cid.loc["MUT_DEL_KA_vs_MUT_DEL_PBS", "test_type"] == "welch_t"


def test_cli_subject_alias_and_run_bulk_paired_and_unpaired_reports(tmp_path: Path) -> None:
    psi_l, sup_l, meta_l = _make_lois_like_psi_and_meta()
    psi_p = tmp_path / "psi.tsv"
    sup_p = tmp_path / "sup.tsv"
    samp_csv = tmp_path / "samples_subj.csv"
    psi_l.to_csv(psi_p, sep="\t", index_label="event_id")
    sup_l.to_csv(sup_p, sep="\t", index_label="event_id")
    pd.DataFrame(
        [{"sample_id": m["sample_id"], "condition": m["condition"], "subject": m["biological_sample_id"]} for m in meta_l]
    ).to_csv(samp_csv, index=False)

    sm_out = tmp_path / "sm_cli"
    assert splicemanifold_main(["bulk", "--psi-matrix", str(psi_p), "--support-matrix", str(sup_p), "--samples", str(samp_csv), "--outdir", str(sm_out)]) == 0
    rec = json.loads((sm_out / "run_receipt.json").read_text(encoding="utf-8"))
    assert rec["permanova_strata"] == "subject_paired"
    assert rec["paired_subjects"] == 4
    coords = pd.read_csv(sm_out / "sample_manifold_coordinates.tsv", sep="\t")
    assert set(coords["donor_id"]) == {"Ind1", "Ind2", "Ind3", "Ind4"}

    sn_out = tmp_path / "sn_cli"
    assert splicenet_main(["--usage", str(psi_p), "--support", str(sup_p), "--samples", str(samp_csv), "--outdir", str(sn_out), "--min-module-size", "6"]) == 0
    assert (sn_out / "module_assignments.tsv").is_file()

    # Full run_bulk on mixed paired + unpaired bundle
    bdir = tmp_path / "mixed_bundle"
    (bdir / "pipeline_info").mkdir(parents=True)
    samp_rows = [
        {"sample": "R3", "condition": "WT_DEL_KA", "subject": "KA_rep1"},
        {"sample": "R5", "condition": "WT_DEL_KA", "subject": "KA_rep2"},
        {"sample": "R4", "condition": "MUT_DEL_KA", "subject": "KA_rep1"},
        {"sample": "R6", "condition": "MUT_DEL_KA", "subject": "KA_rep2"},
        {"sample": "R1", "condition": "WT_DEL_PBS", "subject": "PBS_rep1"},
        {"sample": "R7", "condition": "WT_DEL_PBS", "subject": "PBS_rep2"},
        {"sample": "R11", "condition": "WT_DEL_PBS", "subject": "PBS_rep3"},
        {"sample": "R2", "condition": "MUT_DEL_PBS", "subject": "PBS_rep1"},
        {"sample": "R8", "condition": "MUT_DEL_PBS", "subject": "PBS_rep2"},
        {"sample": "R12", "condition": "MUT_DEL_PBS", "subject": "PBS_rep3"},
    ]
    pd.DataFrame(samp_rows).to_csv(bdir / "pipeline_info/samplesheet.valid.csv", index=False)
    pd.DataFrame(
        [
            {"contrast_id": "MUT_DEL_KA_vs_WT_DEL_KA", "treatment": "MUT_DEL_KA", "control": "WT_DEL_KA", "paired": "true", "design": "~ subject + condition"},
            {"contrast_id": "MUT_DEL_PBS_vs_WT_DEL_PBS", "treatment": "MUT_DEL_PBS", "control": "WT_DEL_PBS", "paired": "true", "design": "~ subject + condition"},
            {"contrast_id": "WT_DEL_KA_vs_WT_DEL_PBS", "treatment": "WT_DEL_KA", "control": "WT_DEL_PBS", "paired": "false", "design": "~ condition"},
        ]
    ).to_csv(bdir / "pipeline_info/contrastsheet.valid.csv", index=False)

    rm_root = bdir / "rnasplicing/rmats"
    _write_rmats_two_group(rm_root / "MUT_DEL_KA-WT_DEL_KA", p1_vals=np.array([0.70, 0.85]), p2_vals=np.array([0.30, 0.45]), dpsi_sign=1.0, seed=1)
    _write_rmats_two_group(rm_root / "MUT_DEL_PBS-WT_DEL_PBS", p1_vals=np.array([0.55, 0.70, 0.85]), p2_vals=np.array([0.25, 0.40, 0.55]), dpsi_sign=1.0, seed=2)
    _write_rmats_two_group(rm_root / "WT_DEL_KA-WT_DEL_PBS", p1_vals=np.array([0.30, 0.45]), p2_vals=np.array([0.25, 0.40, 0.55]), dpsi_sign=1.0, seed=3)

    rep_out = tmp_path / "rep_mixed"
    res = run_bulk(outdir=rep_out, bundle=bdir, species="mouse", study_name="Mixed Paired Study")
    assert res["missing_plot_data"] == 0

    manifest = json.loads((rep_out / "report-manifest.json").read_text(encoding="utf-8"))
    by_c = {c["contrast_id"]: c for c in manifest["display_project"]["contrasts"]}
    assert by_c["MUT_DEL_KA_vs_WT_DEL_KA"]["paired"] is True
    assert by_c["MUT_DEL_KA_vs_WT_DEL_KA"]["paired_subjects"] == 2
    assert by_c["MUT_DEL_KA_vs_WT_DEL_KA"]["design"] == "~ subject + condition"
    assert by_c["WT_DEL_KA_vs_WT_DEL_PBS"]["paired"] is False
    assert by_c["WT_DEL_KA_vs_WT_DEL_PBS"]["design"] == "~ condition"

    bd_df = pd.read_csv(rep_out / "outputs/landscape/biological-design-plot-data.tsv", sep="\t")
    assert int(bd_df["cohort_paired_subjects"].iloc[0]) == 5

    rc_df = pd.read_csv(rep_out / "outputs/landscape/replicate-consistency-plot-data.tsv", sep="\t")
    assert "Same subject (paired)" in set(rc_df["pair_type"])

    pca_df = pd.read_csv(rep_out / "outputs/landscape/psi-pca-coordinates.tsv", sep="\t")
    assert set(pca_df["donor_id"]) == {"KA_rep1", "KA_rep2", "PBS_rep1", "PBS_rep2", "PBS_rep3"}

    html_txt = (rep_out / "REPORT.html").read_text(encoding="utf-8")
    assert "Paired (n=2)" in html_txt
    assert "Unpaired · ~ condition" in html_txt

    assoc_rep = rep_out / "outputs/network/report_tables/module_associations_complete.tsv"
    if assoc_rep.is_file():
        adf = pd.read_csv(assoc_rep, sep="\t")
        c_rows = adf[adf["kind"] == "contrast"].drop_duplicates("contrast_id").set_index("contrast_id")
        assert c_rows.loc["MUT_DEL_KA_vs_WT_DEL_KA", "test_type"] == "paired_t"
        assert c_rows.loc["WT_DEL_KA_vs_WT_DEL_PBS", "test_type"] == "welch_t"

    # Backward compatibility on unpaired cohort
    ubdir = tmp_path / "unpaired_bundle"
    (ubdir / "pipeline_info").mkdir(parents=True)
    pd.DataFrame(
        [{"sample": f"U{i}", "condition": "Ctrl" if i <= 3 else "Trt", "subject": f"U{i}"} for i in range(1, 7)]
    ).to_csv(ubdir / "pipeline_info/samplesheet.valid.csv", index=False)
    pd.DataFrame(
        [{"contrast_id": "Trt_vs_Ctrl", "treatment": "Trt", "control": "Ctrl", "paired": "false", "design": "~ condition"}]
    ).to_csv(ubdir / "pipeline_info/contrastsheet.valid.csv", index=False)
    _write_rmats_two_group(ubdir / "rnasplicing/rmats/Trt-Ctrl", p1_vals=np.array([0.75, 0.78, 0.72]), p2_vals=np.array([0.25, 0.22, 0.28]), dpsi_sign=1.0, seed=9)
    u_out = tmp_path / "rep_unpaired"
    u_res = run_bulk(outdir=u_out, bundle=ubdir, species="mouse", study_name="Unpaired Study")
    assert u_res["missing_plot_data"] == 0
    u_rc = pd.read_csv(u_out / "outputs/landscape/replicate-consistency-plot-data.tsv", sep="\t")
    assert set(u_rc["pair_type"]) <= {"Within condition", "Between conditions"}
