"""Regression tests verifying Katana-PRISM v1.0.2 pre-submission audit fixes."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd

PRISM_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace"):
    _p = str(PRISM_ROOT / "packages" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def test_sc_paired_zero_pair_and_insufficient_donor_eligibility() -> None:
    """Priority 1: Events with 0 or <3 usable paired donors must never fall back to cell-pooled Z under paired label."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    donors = [f"D{i}" for i in range(1, 8)]
    rows = []
    psi_rows = []
    sup_rows = []
    rng = np.random.default_rng(7)
    for d_i, d in enumerate(donors):
        for arm in ("Case", "Ctrl"):
            for k in range(6):
                cid = f"{d}_{arm}_{k}"
                rows.append({"cell_id": cid, "donor_id": d, "condition": arm, "qc_pass": True})
                # E_valid: covered in all 7 paired donors
                e_valid = (0.80 if arm == "Case" else 0.20) + float(rng.normal(0.0, 0.03))
                # E_disjoint_0pair: D1-D4 covered ONLY in Case; D5-D7 covered ONLY in Ctrl => 0 paired donors!
                if d_i < 4 and arm == "Case":
                    e_0pair = 0.90
                    s_0pair = 25.0
                elif d_i >= 4 and arm == "Ctrl":
                    e_0pair = 0.10
                    s_0pair = 25.0
                else:
                    e_0pair = np.nan
                    s_0pair = 0.0
                # E_2pair: only D1 and D2 paired (< 3 minimum)
                if d_i < 2:
                    e_2pair = 0.85 if arm == "Case" else 0.15
                    s_2pair = 25.0
                else:
                    e_2pair = np.nan
                    s_2pair = 0.0
                psi_rows.append({"E_valid": np.clip(e_valid, 0.01, 0.99), "E_disjoint_0pair": e_0pair, "E_2pair": e_2pair})
                sup_rows.append({"E_valid": 25.0, "E_disjoint_0pair": s_0pair, "E_2pair": s_2pair})

    md = pd.DataFrame(rows)
    cids = md["cell_id"].tolist()
    usage = pd.DataFrame(psi_rows, index=cids)
    support = pd.DataFrame(sup_rows, index=cids)
    ev_genes = pd.Series({"E_valid": "G1", "E_disjoint_0pair": "G2", "E_2pair": "G3"})

    c_df, d_del, info = compute_sc_splicing_contrast(
        usage, md, ev_genes, support=support, numerator_group="Case", denominator_group="Ctrl", min_cells_per_group=5, min_support_per_group=8.0
    )
    assert info["inference_unit"] == "paired_donor_population"
    assert d_del is not None and d_del.shape == (7, 3)

    r_val = c_df.set_index("event_id").loc["E_valid"]
    assert bool(r_val["eligible"]) is True
    assert int(r_val["n_paired_donors"]) == 7
    assert r_val["event_inference_unit"] == "paired_donor_population"
    assert np.isfinite(r_val["p_value"]) and r_val["p_value"] < 0.01

    for bad_id, exp_pairs in (("E_disjoint_0pair", 0), ("E_2pair", 2)):
        r_bad = c_df.set_index("event_id").loc[bad_id]
        assert bool(r_bad["eligible"]) is False
        assert int(r_bad["n_paired_donors"]) == exp_pairs
        assert r_bad["event_inference_unit"] == "insufficient_paired_donors"
        assert pd.isna(r_bad["p_value"])
        assert pd.isna(r_bad["raw_dpsi"])
        assert pd.isna(r_bad["z_shift"])


def test_sc_unpaired_single_donor_per_arm_excluded() -> None:
    """Unpaired 3v3 cohorts must also exclude events covered in <2 donors in either arm."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    rows = []
    psi_rows = []
    sup_rows = []
    for d in ("M1", "M2", "M3"):
        for k in range(5):
            rows.append({"cell_id": f"{d}_A_{k}", "donor_id": d, "condition": "A", "qc_pass": True})
            psi_rows.append({"E_ok": 0.85, "E_single_A": 0.90 if d == "M1" else np.nan})
            sup_rows.append({"E_ok": 20.0, "E_single_A": 20.0 if d == "M1" else 0.0})
    for d in ("M4", "M5", "M6"):
        for k in range(5):
            rows.append({"cell_id": f"{d}_B_{k}", "donor_id": d, "condition": "B", "qc_pass": True})
            psi_rows.append({"E_ok": 0.15, "E_single_A": 0.10})
            sup_rows.append({"E_ok": 20.0, "E_single_A": 20.0})

    md = pd.DataFrame(rows)
    cids = md["cell_id"].tolist()
    c_df, _, info = compute_sc_splicing_contrast(
        pd.DataFrame(psi_rows, index=cids),
        md,
        pd.Series({"E_ok": "G1", "E_single_A": "G2"}),
        support=pd.DataFrame(sup_rows, index=cids),
        numerator_group="A",
        denominator_group="B",
        min_cells_per_group=5,
        min_support_per_group=10.0,
    )
    assert info["inference_unit"] == "unpaired_donor_pseudobulk"
    idx = c_df.set_index("event_id")
    assert bool(idx.loc["E_ok", "eligible"]) is True
    assert bool(idx.loc["E_single_A", "eligible"]) is False
    assert idx.loc["E_single_A", "event_inference_unit"] == "insufficient_group_donors"
    assert pd.isna(idx.loc["E_single_A", "p_value"])


def test_uniprot_exact_residue_alignment_f7_and_fn1() -> None:
    """Priority 2: F7 382-aa isoform lacking Gla exon must not report Gla disruption; FN1 +90 aa EDA maps to FN-III 13."""
    from splicetrace.protein_consequences import (
        _map_uniprot_domains_to_isoforms,
        load_bundled_uniprot_domains,
        load_bundled_uniprot_sequences,
    )

    seqs = load_bundled_uniprot_sequences()
    doms = load_bundled_uniprot_domains("human")
    assert "P08709" in seqs and "P02751" in seqs

    # F7 canonical P08709 (466 aa): construct realistic 382-aa exclusion lacking residues 22..105 (including Gla [61..105])
    # and a 29-aa PTC poison-exon inclusion isoform matching N-terminal signal peptide [1..21] + frameshift stop.
    u_f7 = seqs["P08709"]
    f7_exc_382 = u_f7[:21] + u_f7[105:]  # 382 aa, skips Gla [61..105], retains EGF-like 1, EGF-like 2, Peptidase S1
    f7_inc_29 = u_f7[:21] + "EAVLDFLQ"
    _, f7_str, f7_arch = _map_uniprot_domains_to_isoforms("F7", f7_inc_29, f7_exc_382, True, False, doms)
    assert "Gla" not in f7_str
    assert not any("Gla" in d["label"] for d in f7_arch)
    assert "Peptidase S1" in f7_str
    pep_s1 = next(d for d in f7_arch if "Peptidase S1" in d["label"])
    assert pep_s1["alignment_method"] == "residue_sequence_alignment"
    assert pep_s1["start"] == 213 - 84 and pep_s1["end"] == 452 - 84

    # FN1 canonical P02751 (2,477 aa): +90 aa EDA inclusion (residues 1722..1811) -> Fibronectin type-III 13
    u_fn1 = seqs["P02751"]
    fn1_exc_2387 = u_fn1[:1721] + u_fn1[1811:]
    _, fn1_str, fn1_arch = _map_uniprot_domains_to_isoforms("FN1", u_fn1, fn1_exc_2387, False, False, doms)
    assert "Fibronectin type-III 13" in fn1_str
    assert any(d["label"] == "DOMAIN:Fibronectin type-III 13" and d["state"] == "inserted_by_inclusion" for d in fn1_arch)


def test_bulk_rademacher_hc_handles_heteroskedastic_null() -> None:
    """Priority 3A: Self-normalized Rademacher gene-block maxT holds heteroskedastic null FWER <= 0.10."""
    from splicetrace.fast_accel import fast_gene_block_maxt_permutations, fast_gene_block_rademacher_maxt

    rng = np.random.default_rng(123)
    G, n_reg, n_rbp = 450, 13, 40
    C = n_reg * n_rbp
    F = rng.normal(0.0, 1.0, size=(G, C))
    F_gc = F - F.mean(axis=0, keepdims=True)
    # Sparse motif-coupled heteroskedastic null (6% motif+ genes carry 10x larger residual SD)
    sparse_hit = (rng.random(G) < 0.06).astype(float)
    F_gc[:, 0] = sparse_hit - sparse_hit.mean()
    valid_c = np.ones(C, dtype=bool)
    sigma = 0.25 + 2.50 * sparse_hit

    fp_perm = 0
    fp_rad = 0
    for t in range(120):
        yc = rng.normal(0.0, 1.0, size=G) * sigma
        yc -= yc.mean()
        sd_perm = np.sqrt(np.sum(yc * yc) * np.sum(F_gc * F_gc, axis=0) / (G - 1))
        _, _, fwer_p, _, _ = fast_gene_block_maxt_permutations(yc, F_gc, sd_perm, valid_c, n_reg, n_rbp, n_permutations=199, seed=t)
        _, _, fwer_r, _, _ = fast_gene_block_rademacher_maxt(yc, F_gc, valid_c, n_reg, n_rbp, n_permutations=199, seed=t)
        fp_perm += int(np.any(fwer_p <= 0.05))
        fp_rad += int(np.any(fwer_r <= 0.05))

    assert fp_perm > 25  # Confirms exchangeable permutation inflates under sparse heteroskedastic variance
    assert fp_rad <= 12  # <= 10% across 120 trials (near nominal 5%)


def test_sc_six_paired_donors_reaches_fwer_03125() -> None:
    """Priority 3B: D=6 paired donors achieves exact two-sided maxT P_FWER = 2/64 = 0.03125 <= 0.05 across >160 RBPs."""
    from scsplicetrace.contrast_discovery import discover_sc_rbp_programs

    rng = np.random.default_rng(2026)
    n_ev, n_reg, n_rbp = 400, 13, 168
    region_names = [f"r{i}" for i in range(n_reg)]
    rbps = np.array([f"RBP{j}" for j in range(n_rbp)], dtype=str)
    rbp_tensor = rng.normal(0.0, 1.0, size=(n_ev, n_reg, n_rbp))
    ev_ids = [f"E{i}" for i in range(n_ev)]
    # Plant RBP0 across D=6 donors plus genuine donor-specific null variation so background stays null
    sig_slope = rbp_tensor[:, 0, 0] * 1.2
    d_mat = np.vstack([sig_slope + rng.normal(0.0, 0.5, size=n_ev) for _ in range(6)])
    contrast_df = pd.DataFrame({
        "event_id": ev_ids,
        "gene_id": [f"G{i}" for i in range(n_ev)],
        "eligible": True,
        "baseline_PSI": np.full(n_ev, 0.5),
        "z_shift": d_mat.mean(axis=0),
    })
    cov_df = pd.DataFrame({
        "event_id": ev_ids,
        "exon_GC": np.full(n_ev, 0.5),
        "flank_GC": np.full(n_ev, 0.5),
        "log_length": np.full(n_ev, 4.6),
        "event_type": ["SE"] * n_ev,
    })
    donor_delta_df = pd.DataFrame(d_mat, index=[f"D{k}" for k in range(6)], columns=ev_ids)

    res = discover_sc_rbp_programs(contrast_df, rbp_tensor, rbps, region_names, cov_df, donor_delta_df=donor_delta_df)
    top = res.rbp_ranking.iloc[0]
    assert top["RBP"] == "RBP0"
    assert abs(float(top["rbp_P_FWER"]) - (2.0 / 64.0)) < 1e-12
    assert float(top["rbp_P_FWER"]) <= 0.05


def test_sc_unpaired_cell_count_imbalance_null() -> None:
    """v15 Re-audit Blocker 2A: Equal-donor-mean unpaired cohorts must NOT use cell-pooled PSI numerator."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    design = []
    values = []
    for arm in ("T", "C"):
        for d in range(20):
            usage = 0.9 if d < 10 else 0.1
            count = 80 if (usage == 0.9) == (arm == "T") else 8
            for j in range(count):
                design.append((f"{arm}.{d}.{j}", f"{arm}.{d}", arm))
                values.append(usage)
    md = pd.DataFrame(design, columns=["cell_id", "donor_id", "condition"])
    usage_df = pd.DataFrame({"E": values}, index=md["cell_id"])
    sup_df = pd.DataFrame({"E": [100.0] * len(md)}, index=md["cell_id"])
    c_df, d_del, info = compute_sc_splicing_contrast(
        usage_df, md, pd.Series({"E": "G"}), support=sup_df, group_col="condition", numerator_group="T", denominator_group="C"
    )
    assert info["inference_unit"] == "unpaired_donor_pseudobulk"
    assert d_del is None
    row = c_df.iloc[0]
    assert row["event_inference_unit"] == "unpaired_donor_pseudobulk"
    assert abs(float(row["raw_dpsi"])) < 1e-12
    assert abs(float(row["psi_numerator"]) - 0.5) < 1e-12
    assert abs(float(row["psi_denominator"]) - 0.5) < 1e-12
    assert float(row["p_value"]) > 0.99
    assert float(row["fdr"]) > 0.99


def test_sc_paired_sign_flip_midpoint_symmetry() -> None:
    """v15 Re-audit Blocker 2B: Paired baseline_PSI uses arm-symmetric midpoint so 2^D flips stay exact (<=2/64 at alpha=0.05)."""
    from itertools import product
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast, discover_sc_rbp_programs

    rng = np.random.default_rng(1602)
    n_donors, n_events, n_motifs = 6, 35, 12
    signs = np.asarray(list(product((-1, 1), repeat=n_donors)), dtype=float)
    ids = [f"e{i}" for i in range(n_events)]
    genes = pd.Series([f"g{i}" for i in range(n_events)], index=ids)
    nuisance = pd.DataFrame({"event_id": ids, "flank_GC": 0.5, "exon_GC": 0.5, "log_length": 4.0, "event_type": "SE"})
    for _ in range(87):
        motifs = rng.normal(size=(n_events, 1, n_motifs))
        midpoint = rng.uniform(0.43, 0.57, size=n_events)
        deltas = np.clip(rng.normal(0.0, 0.16, size=(n_donors, n_events)), -0.6, 0.6)
    md = pd.DataFrame(
        [(f"{d}.{arm}.{c}", f"{d}", arm) for d in range(n_donors) for arm in ("T", "C") for c in range(8)],
        columns=["cell_id", "donor_id", "condition"],
    )
    support = pd.DataFrame(np.full((len(md), len(ids)), 1e5), index=md["cell_id"], columns=ids)
    labels = np.array([f"r{k}" for k in range(n_motifs)])
    rejections = 0
    for s in signs:
        values = np.stack([
            midpoint + (1.0 if row.condition == "T" else -1.0) * s[int(row.donor_id)] * deltas[int(row.donor_id)] / 2.0
            for row in md.itertuples()
        ])
        u_df = pd.DataFrame(values, index=md["cell_id"], columns=ids)
        c_df, d_df, _ = compute_sc_splicing_contrast(
            u_df, md, genes, support=support, group_col="condition", numerator_group="T", denominator_group="C"
        )
        assert np.allclose(c_df["baseline_PSI"].to_numpy(float), midpoint, atol=1e-6)
        res = discover_sc_rbp_programs(c_df, motifs, labels, ["reg"], nuisance, donor_delta_df=d_df)
        if float(res.rbp_ranking["rbp_P_FWER"].min()) <= 0.05:
            rejections += 1
    assert rejections <= 2


def test_sc_partially_paired_and_arm_missingness_handling() -> None:
    """Explicit partial-pairing routing + arm-dependent missingness exclusion in unpaired cohorts."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    # Case A: 3 paired donors + 2 unpaired donors -> paired_donor_small_n strictly on paired donors
    rows_a, psi_a, sup_a = [], [], []
    for d in ("P1", "P2", "P3"):
        for arm in ("T", "C"):
            for k in range(5):
                rows_a.append({"cell_id": f"{d}_{arm}_{k}", "donor_id": d, "condition": arm})
                psi_a.append({"E1": 0.8 if arm == "T" else 0.2})
                sup_a.append({"E1": 25.0})
    for d, arm in (("U1", "T"), ("U2", "C")):
        for k in range(10):
            rows_a.append({"cell_id": f"{d}_{arm}_{k}", "donor_id": d, "condition": arm})
            psi_a.append({"E1": 0.1 if arm == "T" else 0.9})
            sup_a.append({"E1": 25.0})
    md_a = pd.DataFrame(rows_a)
    c_a, d_a, info_a = compute_sc_splicing_contrast(
        pd.DataFrame(psi_a, index=md_a["cell_id"]), md_a, pd.Series({"E1": "G1"}),
        support=pd.DataFrame(sup_a, index=md_a["cell_id"]), numerator_group="T", denominator_group="C", min_cells_per_group=5, min_support_per_group=10.0,
    )
    assert info_a["inference_unit"] == "paired_donor_small_n"
    assert d_a is None
    assert info_a["unpaired_donors_excluded"] == 2
    assert float(c_a.iloc[0]["raw_dpsi"]) > 0.55

    # Case B: 1 overlapping donor + 3 disjoint per arm + arm-dependent missingness in E_arm_miss
    rows_b, psi_b, sup_b = [], [], []
    for arm in ("T", "C"):
        for k in range(5):
            rows_b.append({"cell_id": f"OV_{arm}_{k}", "donor_id": "OV", "condition": arm})
            psi_b.append({"E_ok": 0.5, "E_arm_miss": 0.5})
            sup_b.append({"E_ok": 25.0, "E_arm_miss": 25.0})
    for i in range(3):
        for k in range(6):
            rows_b.append({"cell_id": f"DT{i}_{k}", "donor_id": f"DT{i}", "condition": "T"})
            psi_b.append({"E_ok": 0.85, "E_arm_miss": 0.85 if i == 0 else np.nan})
            sup_b.append({"E_ok": 25.0, "E_arm_miss": 25.0 if i == 0 else 0.0})
            rows_b.append({"cell_id": f"DC{i}_{k}", "donor_id": f"DC{i}", "condition": "C"})
            psi_b.append({"E_ok": 0.15, "E_arm_miss": 0.15})
            sup_b.append({"E_ok": 25.0, "E_arm_miss": 25.0})
    md_b = pd.DataFrame(rows_b)
    c_b, _, info_b = compute_sc_splicing_contrast(
        pd.DataFrame(psi_b, index=md_b["cell_id"]), md_b, pd.Series({"E_ok": "G1", "E_arm_miss": "G2"}),
        support=pd.DataFrame(sup_b, index=md_b["cell_id"]), numerator_group="T", denominator_group="C", min_cells_per_group=5, min_support_per_group=10.0,
    )
    assert info_b["inference_unit"] == "partially_paired_disjoint_unpaired"
    assert info_b["overlap_donors_excluded_from_unpaired"] == 1
    idx_b = c_b.set_index("event_id")
    assert bool(idx_b.loc["E_ok", "eligible"]) is True
    assert bool(idx_b.loc["E_arm_miss", "eligible"]) is False
    assert idx_b.loc["E_arm_miss", "event_inference_unit"] == "insufficient_group_donors"


def test_splicetrace_cli_calibration_option(tmp_path: Path) -> None:
    """Confirm splicetrace CLI accepts and routes --calibration rademacher_hc."""
    import json
    from splicetrace.cli import main as st_main

    fg_p = tmp_path / "fg.tsv"
    bg_p = tmp_path / "bg.tsv"
    # Tiny synthetic fixture just checking CLI invocation/fallback into run_bulk_continuous_rescue with calibration
    fa = PRISM_ROOT / "benchmarks"  # dummy fallback; use existing small fasta from test_katana_prism_1_0 or verify parser
    # Minimal 1-chrom FASTA + <250 matched events triggers auto small-sample rescue with calibration!
    fasta_p = tmp_path / "mini.fa"
    seq = "ACGTACGTGGCCAAATTTCCCGGG" * 100
    fasta_p.write_text(f">chr1\n{seq}\n", encoding="utf-8", newline="\n")
    (tmp_path / "mini.fa.fai").write_text(f"chr1\t{len(seq)}\t6\t{len(seq)}\t{len(seq) + 1}\n", encoding="utf-8")
    ev_rows = []
    for i in range(40):
        ev_rows.append({
            "event_id": f"EV{i}", "gene_id": f"G{i}", "gene_symbol": f"G{i}", "event_type": "SE",
            "chr": "chr1", "strand": "+", "exonStart_0base": 800 + i * 5, "exonEnd": 900 + i * 5,
            "upstreamES": 500, "upstreamEE": 600, "downstreamES": 1400, "downstreamEE": 1500,
            "dpsi": 0.25 if i < 12 else 0.0, "baseline_PSI": 0.5, "PValue": 1e-4 if i < 12 else 0.8,
            "fdr": 0.005 if i < 12 else 0.9, "depth": 50.0, "GC": 0.5, "length": 100, "ambiguous_fraction": 0.0,
        })
    df_ev = pd.DataFrame(ev_rows)
    df_ev.iloc[:15].to_csv(fg_p, sep="\t", index=False)
    df_ev.iloc[15:].to_csv(bg_p, sep="\t", index=False)
    out_d = tmp_path / "out_rad"
    rc = st_main([
        "--foreground", str(fg_p), "--background", str(bg_p), "--fasta", str(fasta_p),
        "--out", str(out_d), "--calibration", "rademacher_hc", "--permutations", "19", "--seed", "7",
    ])
    assert rc == 0
    receipt = json.loads((out_d / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["calibration"] == "rademacher_hc"
    assert receipt["inference_mode"] == "case_control_cpg_rademacher_hc_maxT"

