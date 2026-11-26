"""Layered regression tests covering single-cell workflow, FWL, bicor, and receipt ledger fixes for Katana-PRISM v1.0.5."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
for pkg_dir in ("splicemanifold", "splicenet", "splicetrace", "scsplicetrace", "scsplicenet", "katana_studio"):
    p = ROOT / "packages" / pkg_dir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from katana_studio.full_suite_views import render_ch7_splicetrace_suite  # noqa: E402
from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast, discover_sc_rbp_programs  # noqa: E402
from splicenet.network import compute_tukey_bicor_matrix  # noqa: E402
import splicetrace.continuous_rescue as cr_mod  # noqa: E402
from splicetrace.library import CuratedMotifLibrary, load_bundled_human_rbp_library  # noqa: E402


def _load_bin_module(script_name: str):
    path = ROOT / "bin" / script_name
    spec = importlib.util.spec_from_file_location(f"katana_bin_v105_l_{script_name.replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sc_slice_contrast_strict_min_cells_and_missing_arm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slice script must fail loudly on missing arm and must never down-clamp --min-cells or emit top-10 fallbacks."""
    slice_mod = _load_bin_module("katana_sc_slice_contrast.py")
    cids = [f"C{i}" for i in range(12)]
    ev_df = pd.DataFrame([
        {"event_id": "SE:chr1:+:100:150:0:50:200:250", "gene_id": "G1", "gene_symbol": "G1"},
        {"event_id": "SE:chr1:+:300:350:200:250:400:450", "gene_id": "G2", "gene_symbol": "G2"},
    ])
    md_df = pd.DataFrame({
        "cell_id": cids,
        "donor_id": [f"D{i % 6}" for i in range(12)],
        "condition": ["T" if i < 6 else "C" for i in range(12)],
        "cell_type": ["B_cell"] * 12,
    })
    psi_df = pd.DataFrame(
        [[0.8, 0.2] if i < 6 else [0.2, 0.8] for i in range(12)],
        index=cids,
        columns=ev_df["event_id"],
    )
    sup_df = pd.DataFrame(np.full((12, 2), 25.0), index=cids, columns=ev_df["event_id"])

    ev_p = tmp_path / "events.tsv"
    md_p = tmp_path / "cells.tsv"
    psi_p = tmp_path / "psi.tsv"
    sup_p = tmp_path / "sup.tsv"
    ev_df.to_csv(ev_p, sep="\t", index=False)
    md_df.to_csv(md_p, sep="\t", index=False)
    psi_df.to_csv(psi_p, sep="\t", index_label="cell_id")
    sup_df.to_csv(sup_p, sep="\t", index_label="cell_id")

    # 1. Missing contrast arm after population filtering must raise ValueError and record missing_contrast_arm receipt
    out_miss = tmp_path / "out_missing_arm"
    md_one_arm = md_df.copy()
    md_one_arm.loc[md_one_arm["condition"] == "T", "cell_type"] = "Other"
    md_one_p = tmp_path / "cells_one_arm.tsv"
    md_one_arm.to_csv(md_one_p, sep="\t", index=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "katana_sc_slice_contrast.py",
            "--events", str(ev_p),
            "--psi", str(psi_p),
            "--support", str(sup_p),
            "--metadata", str(md_one_p),
            "--population-col", "cell_type",
            "--population-value", "B_cell",
            "--numerator", "T",
            "--denominator", "C",
            "--outdir", str(out_miss),
        ],
    )
    with pytest.raises(ValueError, match="Contrast arm\\(s\\) missing"):
        slice_mod.main()
    rec_err = json.loads((out_miss / "slice_receipt.json").read_text(encoding="utf-8"))
    assert rec_err["status"] == "missing_contrast_arm"
    assert rec_err["numerator_cells"] == 0 and rec_err["denominator_cells"] == 6

    # 2. Strict --min-cells 20 on 12 total cells must stay at 0 events (no down-clamping to len//5 or fallback top-10)
    out_strict = tmp_path / "out_strict_min_cells"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "katana_sc_slice_contrast.py",
            "--events", str(ev_p),
            "--psi", str(psi_p),
            "--support", str(sup_p),
            "--metadata", str(md_p),
            "--numerator", "T",
            "--denominator", "C",
            "--min-cells", "20",
            "--outdir", str(out_strict),
        ],
    )
    assert slice_mod.main() == 0
    ev_out = pd.read_csv(out_strict / "sc_events.tsv", sep="\t")
    assert len(ev_out) == 0
    rec_strict = json.loads((out_strict / "slice_receipt.json").read_text(encoding="utf-8"))
    assert rec_strict["status"] == "zero_passing_events"
    assert rec_strict["min_cells_requested"] == 20 and rec_strict["passing_events"] == 0


def test_sc_build_splice_matrices_group_col_and_pseudobulk_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Matrix builder must reject unknown explicit --group-col and donor-collapsing --pseudobulk-by while supporting donor-safe pseudobulk."""
    build_mod = _load_bin_module("katana_sc_build_splice_matrices.py")

    gtf_p = tmp_path / "se.gtf"
    gtf_p.write_text(
        "\n".join([
            'chr1\tsrc\texon\t1\t50\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T_INC"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t101\t150\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T_INC"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t201\t250\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T_INC"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t1\t50\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T_EXC"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t201\t250\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T_EXC"; transcript_type "protein_coding";',
        ]) + "\n",
        encoding="utf-8",
    )

    solo = tmp_path / "Solo.out/SJ/raw"
    solo.mkdir(parents=True)
    # Needed junctions from GTF: (51, 100), (51, 200), (151, 200)
    (solo / "features.tsv").write_text("chr1\t51\t100\t1\nchr1\t151\t200\t1\nchr1\t51\t200\t1\n", encoding="utf-8")
    bcs = [f"CELL_{i}" for i in range(12)]
    (solo / "barcodes.tsv").write_text("\n".join(bcs) + "\n", encoding="utf-8")
    # First 6 cells T (high inc), next 6 cells C (high exc) across 3 donors
    counts = np.zeros((3, 12), dtype=np.int32)
    counts[0, :6] = 20; counts[1, :6] = 20; counts[2, :6] = 4
    counts[0, 6:] = 4;  counts[1, 6:] = 4;  counts[2, 6:] = 20
    mmwrite(str(solo / "matrix.mtx"), coo_matrix(counts))

    md_p = tmp_path / "meta.tsv"
    pd.DataFrame({
        "cell_id": bcs,
        "donor_id": [f"D{(i % 3) + 1}" for i in range(12)],
        "condition": ["T" if i < 6 else "C" for i in range(12)],
        "cell_type": ["Tcell"] * 12,
    }).to_csv(md_p, sep="\t", index=False)

    # 1. Explicit unknown --group-col must raise ValueError
    with pytest.raises(ValueError, match="Group column 'nonexistent_col' not found"):
        build_mod.standardize_metadata(md_p, bcs, group_col="nonexistent_col", numerator="T", denominator="C")

    # 2. --pseudobulk-by omitting donor_id collapses D1..D3 -> must raise ValueError
    out_bad_pb = tmp_path / "out_bad_pb"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "katana_sc_build_splice_matrices.py",
            "--gtf", str(gtf_p),
            "--sj-dirs", str(solo),
            "--metadata", str(md_p),
            "--outdir", str(out_bad_pb),
            "--numerator", "T",
            "--denominator", "C",
            "--pseudobulk-by", "cell_type,condition",
        ],
    )
    with pytest.raises(ValueError, match="collapses multiple biological donors"):
        build_mod.main()

    # 3. Valid --pseudobulk-by donor_id,cell_type,condition succeeds & wires n_pooled_cells into scSpliceTrace
    out_ok_pb = tmp_path / "out_ok_pb"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "katana_sc_build_splice_matrices.py",
            "--gtf", str(gtf_p),
            "--sj-dirs", str(solo),
            "--metadata", str(md_p),
            "--outdir", str(out_ok_pb),
            "--numerator", "T",
            "--denominator", "C",
            "--min-cells", "4",
            "--pseudobulk-by", "donor_id,cell_type,condition",
        ],
    )
    assert build_mod.main() == 0
    pb_cells = pd.read_csv(out_ok_pb / "sc_cells.tsv", sep="\t")
    assert len(pb_cells) == 6 and (pb_cells["n_pooled_cells"] == 2).all()
    pb_psi = pd.read_csv(out_ok_pb / "sc_psi.tsv.gz", sep="\t", index_col=0)
    pb_sup = pd.read_csv(out_ok_pb / "sc_support.tsv.gz", sep="\t", index_col=0)
    pb_ev = pd.read_csv(out_ok_pb / "sc_events.tsv", sep="\t")
    c_df, _, info = compute_sc_splicing_contrast(
        pb_psi, pb_cells, pb_ev.set_index("event_id")["gene_id"],
        support=pb_sup, numerator_group="T", denominator_group="C",
        min_cells_per_group=4,
    )
    assert int(info["paired_donors"]) == 3
    assert bool(c_df["eligible"].iloc[0])


def test_barcode_shares_and_marvel_input_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """BAM extractor and MARVEL prep must reject ambiguous barcode_shares, filter QC=False first, and catch STAR barcode collisions."""
    bam_mod = _load_bin_module("katana_sc_extract_multiplexed_bam.py")
    prep_mod = _load_bin_module("katana_sc_prepare_marvel_inputs.py")

    md_p = tmp_path / "annot.tsv"
    md_p.write_text("cell_id\tdonor_id\tcondition\tqc_pass\nQC1\tD1\tT\tTrue\nQC2\tD2\tC\tTrue\n", encoding="utf-8")

    shares_cross = tmp_path / "shares_cross.tsv"
    shares_cross.write_text("QC1\tQC2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="links two distinct QC-passed cells"):
        bam_mod.load_qc_cells_and_aliases(md_p, shares_cross)

    shares_multi = tmp_path / "shares_multi.tsv"
    shares_multi.write_text("QC1\tSHARED_BC\nQC2\tSHARED_BC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="claimed by multiple QC cells"):
        bam_mod.load_qc_cells_and_aliases(md_p, shares_multi)

    # prepare_marvel_inputs: QC=False excluded before STAR barcode resolution; ACGT and ACGT-1 collision rejected
    gtf_p = tmp_path / "mini.gtf"
    gtf_p.write_text(
        'chr1\tsrc\texon\t1\t50\t.\t+\t.\tgene_id "G1"; gene_name "G1"; gene_type "protein_coding"; transcript_id "T1";\n',
        encoding="utf-8",
    )
    fgtf_dir = tmp_path / "fromgtf"
    fgtf_dir.mkdir()
    for fn in ("fromGTF.SE.txt", "fromGTF.MXE.txt", "fromGTF.RI.txt", "fromGTF.A5SS.txt", "fromGTF.A3SS.txt"):
        (fgtf_dir / fn).write_text("ID\tGeneID\tgeneSymbol\tchr\tstrand\texonStart_0base\texonEnd\tupstreamES\tupstreamEE\tdownstreamES\tdownstreamEE\n", encoding="utf-8")
    solo = tmp_path / "Solo.out/SJ/raw"
    solo.mkdir(parents=True)
    (solo / "features.tsv").write_text("chr1\t51\t100\t1\n", encoding="utf-8")
    (solo / "barcodes.tsv").write_text("ACGT-1\n", encoding="utf-8")
    mmwrite(str(solo / "matrix.mtx"), coo_matrix(np.array([[10]], dtype=np.int32)))

    md_coll = tmp_path / "md_coll.tsv"
    md_coll.write_text(
        "cell_id\tdonor_id\tcondition\tqc_pass\n"
        "FAILED_NOT_IN_STAR\tD0\tT\tFalse\n"
        "ACGT\tD1\tT\tTrue\n"
        "ACGT-1\tD2\tC\tTrue\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "katana_sc_prepare_marvel_inputs.py",
            "--gtf", str(gtf_p),
            "--fromgtf-dir", str(fgtf_dir),
            "--sj-dirs", str(solo),
            "--metadata", str(md_coll),
            "--outdir", str(tmp_path / "marvel_out"),
        ],
    )
    with pytest.raises(ValueError, match="same physical STARsolo barcode"):
        prep_mod.main()


def test_marvel_r_script_preserves_string_cell_and_donor_ids() -> None:
    """Static check ensuring katana_sc_marvel_quant.R preserves numeric/zero-padded IDs via colClasses='character'."""
    r_txt = (ROOT / "bin/katana_sc_marvel_quant.R").read_text(encoding="utf-8")
    assert 'colClasses = "character"' in r_txt
    assert "na.strings = character(0)" in r_txt
    assert "md_out$cell_id <- as.character" in r_txt
    assert "md_out$donor_id <- as.character" in r_txt


def test_sc_fwl_missing_coverage_null_and_sign_flip_symmetry() -> None:
    """Per-donor observed-mask FWL + numerical-zero moderation must yield FWER=1.0 under motif-coupled coverage missingness."""
    nD, nE = 8, 48
    cols = [f"e{k}" for k in range(nE)]
    shape = np.r_[-np.ones(nE // 2), np.ones(nE // 2)]
    motif = shape[:, None, None]
    coverage = np.repeat(((np.arange(nD) >= nD // 2)[:, None]) == (shape[None, :] > 0), 1, axis=0)
    coverage[:, [0, 1, 2, 3, nE // 2, nE // 2 + 1, nE // 2 + 2, nE // 2 + 3]] = True
    true_offsets = np.r_[np.repeat(-0.2, nD // 2), np.repeat(0.2, nD // 2)]

    rows, vals = [], []
    for d in range(nD):
        for arm in ("T", "C"):
            for k in range(8):
                rows.append((f"D{d}:{arm}:{k}", f"D{d}", arm))
                vals.append(np.repeat(0.5 + (0.5 if arm == "T" else -0.5) * true_offsets[d], nE))
    md = pd.DataFrame(rows, columns=["cell_id", "donor_id", "condition"])
    u_full = pd.DataFrame(vals, index=md.cell_id, columns=cols)
    u_sparse = u_full.where(pd.DataFrame(np.repeat(coverage, 16, axis=0), index=u_full.index, columns=cols))
    eg = pd.Series([f"g{i}" for i in range(nE)], index=cols)
    cov = pd.DataFrame({"event_id": cols, "exon_GC": 0.5, "flank_GC": 0.5, "log_length": 4.0, "event_type": "SE"})

    cs_sp, ds_sp, _ = compute_sc_splicing_contrast(
        u_sparse, md, eg, support=u_sparse.notna().astype(float) * 100.0, numerator_group="T", denominator_group="C"
    )
    res_sp = discover_sc_rbp_programs(
        cs_sp, motif, ["synth_cov_motif"], ["test_region"], cov, donor_delta_df=ds_sp, n_permutations=999
    )
    r_sp = res_sp.rbp_ranking.iloc[0]
    assert float(r_sp["rbp_P_FWER"]) == 1.0
    assert abs(float(r_sp["stat_Z"])) < 1e-10

    # Sign-flip half the donors -> still FWER == 1.0 & |Z| < 1e-10
    ds_flip = ds_sp.copy()
    ds_flip.iloc[:4] = -ds_flip.iloc[:4]
    res_flip = discover_sc_rbp_programs(
        cs_sp, motif, ["synth_cov_motif"], ["test_region"], cov, donor_delta_df=ds_flip, n_permutations=999
    )
    assert float(res_flip.rbp_ranking.iloc[0]["rbp_P_FWER"]) == 1.0
    assert abs(float(res_flip.rbp_ranking.iloc[0]["stat_Z"])) < 1e-10

    # Zero-mean iid noise around donor offsets + motif-coupled missingness must stay calibrated (FPR <= 0.15)
    rng = np.random.default_rng(20261126)
    fps = 0
    for trial in range(40):
        noise = rng.normal(0.0, 0.04, size=(nD, nE))
        ds_noisy = pd.DataFrame(
            np.where(coverage, true_offsets[:, None] + noise, np.nan),
            index=ds_sp.index,
            columns=cols,
        )
        cs_n = cs_sp.copy()
        d_bar = np.nanmean(ds_noisy.to_numpy(float), axis=0)
        cs_n["dpsi"] = d_bar
        cs_n["raw_dpsi"] = d_bar
        res_n = discover_sc_rbp_programs(
            cs_n, motif, ["synth_cov_motif"], ["test_region"], cov, donor_delta_df=ds_noisy, n_permutations=499, seed=100 + trial
        )
        if float(res_n.rbp_ranking.iloc[0]["rbp_P_FWER"]) <= 0.05:
            fps += 1
    assert fps / 40.0 <= 0.15


def test_splicenet_zero_mad_pearson_fallback_matches_wgcna() -> None:
    """Binary/sparse vectors with >50% zeros have MAD=0 and must fall back cleanly to Pearson correlation rather than 0."""
    mat = np.zeros((2, 30), dtype=float)
    mat[0, :2] = 1.0
    mat[1, :2] = 1.0
    bic = compute_tukey_bicor_matrix(mat)
    assert bic.shape == (2, 2)
    assert np.isfinite(bic).all()
    assert float(bic[0, 1]) == pytest.approx(1.0, abs=1e-6)


def test_studio_ch7_receipt_and_sha256_ledger_block_stale_artifacts(tmp_path: Path) -> None:
    """Studio Ch7 Trace ingestion must skip stale artifacts when receipt status is insufficient or SHA256 mismatches."""
    st_dir = tmp_path / "trace_dir"
    st_dir.mkdir()
    reg_p = st_dir / "region_tests.tsv"
    reg_txt = "RBP\tregion\tstat_Z\tnominal_P\tregion_P_FWER\tregion_FDR_BH\nPTBP1\t3ss_prox\t5.0\t0.001\t0.01\t0.01\n"
    reg_p.write_text(reg_txt, encoding="utf-8")
    good_sha = hashlib.sha256(reg_p.read_bytes()).hexdigest()

    # 1. Valid receipt + matching SHA ingests the artifact
    (st_dir / "run_receipt.json").write_text(
        json.dumps({"status": "complete", "contrast_id": "T_vs_C", "artifacts": {"region_tests.tsv": {"sha256": good_sha}}}),
        encoding="utf-8",
    )
    out1 = tmp_path / "studio1"
    panels1, tables1, _ = [], [], []
    render_ch7_splicetrace_suite(st_dir, "T_vs_C", "human", out1, panels1, tables1, [])
    assert any(t.get("analysis_id") == "prioritize.splicetrace_regions" for t in tables1)

    # 2. Tampered file (SHA mismatch against ledger) must be skipped
    reg_p.write_text(reg_txt.replace("5.0", "9.9"), encoding="utf-8")
    out2 = tmp_path / "studio2"
    tables2 = []
    render_ch7_splicetrace_suite(st_dir, "T_vs_C", "human", out2, [], tables2, [])
    assert not any(t.get("analysis_id") == "prioritize.splicetrace_regions" for t in tables2)

    # 3. Receipt marked insufficient_events_for_rescue must block all stale artifacts even if hash matched
    (st_dir / "run_receipt.json").write_text(
        json.dumps({"status": "insufficient_events_for_rescue", "contrast_id": "T_vs_C"}),
        encoding="utf-8",
    )
    out3 = tmp_path / "studio3"
    tables3 = []
    render_ch7_splicetrace_suite(st_dir, "T_vs_C", "human", out3, [], tables3, [])
    assert len(tables3) == 0


def test_splicetrace_vs_scsplicetrace_shared_modules_byte_identical() -> None:
    """Shared RNA-form and protein-consequence modules must remain byte-identical across bulk and sc packages."""
    for mod_name in ("rna_forms.py", "protein_consequences.py"):
        b1 = (ROOT / "packages/splicetrace/splicetrace" / mod_name).read_bytes()
        b2 = (ROOT / "packages/scsplicetrace/scsplicetrace" / mod_name).read_bytes()
        assert b1 == b2, f"Package mirror drift detected on {mod_name}"


def test_continuous_rescue_no_flank_gc_aliasing_and_6col_fwl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Continuous rescue FWL design must use explicit single intronic_gc column without in-place aliasing."""
    src_txt = (ROOT / "packages/splicetrace/splicetrace/continuous_rescue.py").read_text(encoding="utf-8")
    assert "out=flank_gc" not in src_txt
    assert "out=prox_gc.copy()" in src_txt

    # 16k bp so all 500-nt distal + 200-nt proximal windows are valid and vary across events
    rng = np.random.default_rng(123)
    seq_chars = list(rng.choice(["A", "C", "G", "T"], size=20000))
    events = []
    for k in range(14):
        base = 800 + k * 1200
        # Make upstream distal intron high-GC on even events, low-GC on odd events, while proximal intron stays ~0.5
        for p in range(base - 600, base - 200):
            seq_chars[p] = "G" if (k % 2 == 0 and p % 5 != 0) else ("A" if k % 2 == 1 and p % 5 != 0 else "C")
        events.append({
            "event_id": f"SE:chr1:+:{base+100}:{base+200}:{base-700}:{base-600}:{base+700}:{base+800}",
            "gene_id": f"G{k}",
            "gene_symbol": f"SYM{k}",
            "comparison_id": "test",
            "event_type": "SE",
            "chr": "chr1",
            "strand": "+",
            "exonStart_0base": base + 100,
            "exonEnd": base + 200,
            "upstreamES": base - 700,
            "upstreamEE": base - 600,
            "downstreamES": base + 700,
            "downstreamEE": base + 800,
            "dpsi": 0.35 if k < 7 else 0.01,
            "fdr": 0.01 if k < 7 else 0.80,
            "PValue": 0.001 if k < 7 else 0.70,
            "baseline_PSI": 0.30 + 0.03 * (k % 5),
            "depth": 40.0 + 5.0 * k,
        })
    fa_p = tmp_path / "mini.fa"
    seq_bytes = "".join(seq_chars).encode("ascii")
    fa_p.write_bytes(b">chr1\n" + seq_bytes + b"\n")
    (tmp_path / "mini.fa.fai").write_text(f"chr1\t{len(seq_bytes)}\t6\t{len(seq_bytes)}\t{len(seq_bytes) + 1}\n", encoding="utf-8")
    df_ev = pd.DataFrame(events)

    full_lib = load_bundled_human_rbp_library()
    m_id = full_lib.rbp_to_models["PTBP1"][0]
    lib = CuratedMotifLibrary(
        name="mini_test",
        models={m_id: full_lib.models[m_id]},
        rbp_to_models={"PTBP1": (m_id,)},
        excluded_models=(),
    )
    captured_X: list[np.ndarray] = []
    orig_fwl = cr_mod._fwl_residuals

    def _spy_fwl(Y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
        captured_X.append(X.copy())
        return orig_fwl(Y, X, w)

    monkeypatch.setattr(cr_mod, "_fwl_residuals", _spy_fwl)
    res = cr_mod.run_bulk_continuous_rescue(df_ev, fa_p, lib, distal_intron_nt=500, n_permutations=49, seed=7)
    assert res.diagnostics["status"] == "complete"
    assert len(captured_X) >= 3
    for X_mat in captured_X:
        assert X_mat.shape == (14, 7)  # intercept + 6 active continuous covariates (single intronic_gc col)
    assert float(np.std(captured_X[0][:, 1])) > 0.03  # distal intronic_gc varies cleanly without aliasing


def test_validate_inputs_biological_unit_aliases_and_conflicts(tmp_path: Path) -> None:
    """E03: Repeated biological_sample_id or conflicting alias columns must be rejected identically to subject."""
    kvi = _load_bin_module("katana_validate_inputs.py")
    fa = tmp_path / "ref.fa"
    gtf = tmp_path / "ref.gtf"
    fa.write_text(">chr1\nACGTACGT\n", encoding="utf-8")
    gtf.write_text('chr1\tHAVANA\texon\t1\t4\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n', encoding="utf-8")
    cs = tmp_path / "contrasts.csv"
    cs.write_text("contrast,treatment,control\nT_vs_C,T,C\n", encoding="utf-8")

    bams = []
    for i in range(6):
        b = tmp_path / f"lib_{i}.bam"
        b.write_bytes(b"BAM1")
        bams.append(str(b))

    # 1. Repeated biological_sample_id without subject -> must fail (technical partitions are not biological replicates)
    s_rep_bio = tmp_path / "s_rep_bio.csv"
    rows1 = ["sample,condition,genome_bam,single_end,strandedness,read_length,biological_sample_id"]
    for i in range(6):
        cond = "T" if i < 3 else "C"
        unit = f"D{'T' if i < 3 else 'C'}_1"
        rows1.append(f"L{i},{cond},{bams[i]},false,reverse,100,{unit}")
    s_rep_bio.write_text("\n".join(rows1) + "\n", encoding="utf-8")
    rc1 = kvi.main(["--samplesheet", str(s_rep_bio), "--contrasts", str(cs), "--outdir", str(tmp_path / "o1"), "--fasta", str(fa), "--gtf", str(gtf)])
    assert rc1 == 2

    # 2. Conflicting subject vs biological_sample_id -> must fail
    s_conf = tmp_path / "s_conf.csv"
    rows2 = ["sample,condition,genome_bam,single_end,strandedness,read_length,subject,biological_sample_id"]
    for i in range(6):
        cond = "T" if i < 3 else "C"
        rows2.append(f"L{i},{cond},{bams[i]},false,reverse,100,SUBJ_{i},BIO_{i}")
    s_conf.write_text("\n".join(rows2) + "\n", encoding="utf-8")
    rc2 = kvi.main(["--samplesheet", str(s_conf), "--contrasts", str(cs), "--outdir", str(tmp_path / "o2"), "--fasta", str(fa), "--gtf", str(gtf)])
    assert rc2 == 2

    # 3. Concordant independent units -> must pass
    s_ok = tmp_path / "s_ok.csv"
    rows3 = ["sample,condition,genome_bam,single_end,strandedness,read_length,subject,biological_sample_id"]
    for i in range(6):
        cond = "T" if i < 3 else "C"
        rows3.append(f"L{i},{cond},{bams[i]},false,reverse,100,U{i},U{i}")
    s_ok.write_text("\n".join(rows3) + "\n", encoding="utf-8")
    rc3 = kvi.main(["--samplesheet", str(s_ok), "--contrasts", str(cs), "--outdir", str(tmp_path / "o3"), "--fasta", str(fa), "--gtf", str(gtf)])
    assert rc3 == 0


def test_bundle_reference_sha_binding_and_reannotation_flag(tmp_path: Path) -> None:
    """E04: Bundled downstream reference mismatch fails unless explicit re-annotation is requested."""
    from splicetrace.rmats_io import load_rmats_measurable_events

    bdir = tmp_path / "bundle"
    rdir = bdir / "rmats/T_vs_C"
    rdir.mkdir(parents=True)
    (bdir / "provenance").mkdir(parents=True)

    seq_a = b"A" * 2000
    fa_orig = tmp_path / "orig.fa"
    fa_orig.write_bytes(b">chr1\n" + seq_a + b"\n")
    (tmp_path / "orig.fa.fai").write_text(f"chr1\t{len(seq_a)}\t6\t{len(seq_a)}\t{len(seq_a)+1}\n", encoding="utf-8")

    fa_other = tmp_path / "foreign.fa"
    seq_c = b"C" * 2000
    fa_other.write_bytes(b">chr1\n" + seq_c + b"\n")
    (tmp_path / "foreign.fa.fai").write_text(f"chr1\t{len(seq_c)}\t6\t{len(seq_c)}\t{len(seq_c)+1}\n", encoding="utf-8")

    orig_sha = hashlib.sha256(fa_orig.read_bytes()).hexdigest()
    (bdir / "provenance/reference_audit.json").write_text(json.dumps({"fasta": {"sha256": orig_sha}}), encoding="utf-8")
    (bdir / "contrastsheet.csv").write_text("contrast,treatment,control\nT_vs_C,T,C\n", encoding="utf-8")
    (rdir / "SE.MATS.JCEC.txt").write_text(
        "ID\tGeneID\tgeneSymbol\tchr\tstrand\texonStart_0base\texonEnd\tupES\tupEE\tupstreamES\tupstreamEE\tdownstreamES\tdownstreamEE\tPValue\tFDR\tIncLevelDifference\tIJC_SAMPLE_1\tSJC_SAMPLE_1\tIJC_SAMPLE_2\tSJC_SAMPLE_2\tIncLevel1\tIncLevel2\n"
        '1\t"ENSG000001"\t"G1"\tchr1\t+\t900\t1000\t700\t800\t700\t800\t1100\t1200\t0.001\t0.01\t0.4\t30,30\t5,5\t5,5\t30,30\t0.85,0.85\t0.15,0.15\n',
        encoding="utf-8",
    )

    # Matching reference passes
    df_ok = load_rmats_measurable_events(rdir, fa_orig, contrast_id="T_vs_C", bundle_dir=bdir)
    assert len(df_ok) == 1

    # Substituted reference fails by default
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_rmats_measurable_events(rdir, fa_other, contrast_id="T_vs_C", bundle_dir=bdir)

    # Explicit opt-in re-annotation succeeds
    df_re = load_rmats_measurable_events(rdir, fa_other, contrast_id="T_vs_C", bundle_dir=bdir, allow_reference_reannotation=True)
    assert len(df_re) == 1


def test_domain_and_reverse_direction_type_validation(tmp_path: Path) -> None:
    """E06: Out-of-range P/FDR/dPSI and string reverse_direction='False' are rejected cleanly."""
    from splicemanifold.ps_das import qualify_ps_das_events
    from splicetrace.rmats_io import _standardize_rmats_df

    base_row = {
        "ID": 1, "GeneID": "G1", "geneSymbol": "G1", "chr": "chr1", "strand": "+",
        "exonStart_0base": 100, "exonEnd": 200, "upstreamES": 10, "upstreamEE": 50, "downstreamES": 250, "downstreamEE": 300,
        "PValue": 0.001, "FDR": 0.01, "IncLevelDifference": 0.4,
        "IJC_SAMPLE_1": "30,30", "SJC_SAMPLE_1": "5,5", "IJC_SAMPLE_2": "5,5", "SJC_SAMPLE_2": "30,30",
        "IncLevel1": "0.85,0.85", "IncLevel2": "0.15,0.15",
    }
    with pytest.raises(TypeError, match="reverse_direction"):
        _standardize_rmats_df(pd.DataFrame([base_row]), "SE", {"chr1"}, "c1", reverse_direction="False")  # type: ignore[arg-type]

    bad_ps = pd.DataFrame([
        {**base_row, "PValue": -1.0},
        {**base_row, "FDR": -0.1},
        {**base_row, "IncLevelDifference": 1.25},
        {**base_row, "IncLevelDifference": np.inf},
    ])
    q_res = qualify_ps_das_events(bad_ps)
    assert (~q_res["passes_ordinary_das"]).all() and (~q_res["passes_ps_das"]).all()

    fa = tmp_path / "mini.fa"
    fa.write_bytes(b">chr1\n" + b"ACGT" * 500 + b"\n")
    (tmp_path / "mini.fa.fai").write_text("chr1\t2000\t6\t2000\t2001\n", encoding="utf-8")
    for col, bad_val in (("PValue", -0.5), ("fdr", -0.1), ("dpsi", float("inf")), ("dpsi", 1.25)):
        ev = {
            "event_id": "SE:chr1:+:900:1000:700:800:1100:1200", "gene_id": "G1", "gene_symbol": "G1",
            "event_type": "SE", "chr": "chr1", "strand": "+", "dpsi": 0.3, "fdr": 0.01, "PValue": 0.001,
            "baseline_PSI": 0.4, "depth": 40.0,
        }
        ev[col] = bad_val
        with pytest.raises(ValueError):
            cr_mod.run_bulk_continuous_rescue(pd.DataFrame([ev]), fa, load_bundled_human_rbp_library(), n_permutations=9, seed=1)


def test_splicemanifold_unit_permanova_coverage_and_supervised_guards() -> None:
    """A05, A06, A08: Unit-aware PERMANOVA, strict coverage/zero-support gate, and supervised P suppression."""
    from splicemanifold.manifold import compute_splicing_manifold, permanova_r2

    # A05: 8 independent donors x 4 technical libraries, treatment assigned at donor level (exact null 2-sample case)
    rng = np.random.default_rng(17)
    sids, meta = [], []
    # Deterministic identical distances within donors + mild donor shift -> under global donor null, exact donor perm P >= 0.05
    donor_vecs = rng.normal(0.0, 1.0, size=(8, 6))
    rows = []
    for d in range(8):
        cond = "T" if d < 4 else "C"
        for t in range(4):
            sid = f"D{d}_L{t}"
            sids.append(sid)
            meta.append({"sample_id": sid, "biological_sample_id": f"D{d}", "condition": cond})
            rows.append(donor_vecs[d] + rng.normal(0.0, 0.02, size=6))
    X = np.vstack(rows)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    conds = [m["condition"] for m in meta]
    units = [m["biological_sample_id"] for m in meta]
    _, p_naive = permanova_r2(D, conds, n_perm=199, seed=1)
    _, p_unit, scope_u, valid_u = permanova_r2(D, conds, n_perm=199, seed=1, units=units, return_details=True)
    assert valid_u is True and scope_u == "between_biological_unit"
    assert p_naive <= 0.05 and p_unit >= 0.25

    # A06: Strict 0.80 coverage gate (no silent fallback to 0.25) and all-zero support -> unavailable
    psi_cov = pd.DataFrame(
        np.where(np.arange(8)[:, None] < 99, rng.uniform(0.2, 0.8, size=(8, 4)), np.nan),
        index=[f"E{i}" for i in range(8)],
        columns=["S0", "S1", "S2", "S3"],
    )
    psi_cov.iloc[:, [2, 3]] = np.nan  # 50% observed < 0.80 requested
    m4 = [{"sample_id": f"S{i}", "biological_sample_id": f"S{i}", "condition": "A" if i < 2 else "B"} for i in range(4)]
    res_cov = compute_splicing_manifold(psi_cov, m4, min_present_fraction=0.80, run_umap=False)
    assert res_cov["available"] is False

    psi_full = pd.DataFrame(rng.uniform(0.2, 0.8, size=(8, 4)), index=[f"E{i}" for i in range(8)], columns=["S0", "S1", "S2", "S3"])
    sup_zero = pd.DataFrame(np.zeros((8, 4), dtype=float), index=psi_full.index, columns=psi_full.columns)
    res_zero = compute_splicing_manifold(psi_full, m4, depth_matrix=sup_zero, run_umap=False)
    assert res_zero["available"] is False

    # A08: Supervised mode suppresses PERMANOVA P and preserves R2 descriptively
    res_sup = compute_splicing_manifold(
        psi_full, m4, supervised_event_ids=set(psi_full.index[:6]), top_n=6, manifold_top_n=6, run_umap=False
    )
    assert res_sup["available"] is True
    assert res_sup["selection_mode"] == "supervised_ps_das"
    assert res_sup["permanova_p"] is None
    assert res_sup["permanova_inference_valid"] is False
    assert isinstance(res_sup["permanova_r2"], float)

