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
