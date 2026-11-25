"""Round-5 regression tests covering all v20 deep re-audit fixes for Katana-PRISM v1.0.5 RC."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for pkg_dir in ("splicemanifold", "splicenet", "splicetrace", "scsplicetrace", "scsplicenet", "katana_studio"):
    p = ROOT / "packages" / pkg_dir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from katana_studio.report_builder import run_bulk, run_single_cell  # noqa: E402
from scsplicetrace import protein_consequences as sc_pc  # noqa: E402
from splicemanifold.ps_das import load_rmats_study, qualify_ps_das_events  # noqa: E402
from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats  # noqa: E402
from splicetrace import protein_consequences as st_pc  # noqa: E402
from splicetrace.rmats_io import load_rmats_measurable_events  # noqa: E402
from splicetrace.rna_forms import IndexedFasta  # noqa: E402


def _load_bin_module(script_name: str):
    path = ROOT / "bin" / script_name
    spec = importlib.util.spec_from_file_location(f"katana_bin_v105_{script_name.replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_fasta_with_fai(fa_path: Path, contigs: list[tuple[str, str]]) -> Path:
    fa_chunks: list[bytes] = []
    fai_lines: list[str] = []
    pos = 0
    for name, seq in contigs:
        hdr = f">{name}\n".encode("ascii")
        body = seq.encode("ascii") + b"\n"
        pos += len(hdr)
        fai_lines.append(f"{name}\t{len(seq)}\t{pos}\t{len(seq)}\t{len(seq) + 1}")
        pos += len(body)
        fa_chunks.extend([hdr, body])
    fa_path.write_bytes(b"".join(fa_chunks))
    Path(str(fa_path) + ".fai").write_text("\n".join(fai_lines) + "\n", encoding="utf-8")
    return fa_path


def _write_mini_gtf_fasta(tmp_path: Path) -> tuple[Path, Path]:
    fa_p = _write_fasta_with_fai(tmp_path / "mini.fa", [("chr1", "ATG" + ("GCT" * 350) + "TAA")])
    gtf_p = tmp_path / "mini.gtf"
    gtf_p.write_text(
        "\n".join([
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T1"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "G1"; gene_name "TP53"; transcript_id "T1"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t201\t260\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T1"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t201\t260\t.\t+\t0\tgene_id "G1"; gene_name "TP53"; transcript_id "T1"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t401\t460\t.\t+\t.\tgene_id "G1"; gene_name "TP53"; transcript_id "T1"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t401\t457\t.\t+\t0\tgene_id "G1"; gene_name "TP53"; transcript_id "T1"; transcript_type "protein_coding";',
        ]) + "\n",
        encoding="utf-8",
    )
    return fa_p, gtf_p


def _rmats_se_table_text(rows: list[list[str]]) -> str:
    header = "\t".join([
        "ID", "GeneID", "geneSymbol", "chr", "strand",
        "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
        "downstreamES", "downstreamEE", "ID.1",
        "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
        "IncFormLen", "SkipFormLen", "PValue", "FDR",
        "IncLevel1", "IncLevel2", "IncLevelDifference",
    ])
    return "\n".join([header] + ["\t".join(r) for r in rows]) + "\n"


def test_bare_rmats_post_reversed_orientation_and_short_arm_rejection(tmp_path: Path) -> None:
    """Direct Unt-PTBP2/rmats_post path must align arm PSI/sign and reject short samplesheets across all modules."""
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)
    post_dir = tmp_path / "Unt-PTBP2" / "rmats_post"
    post_dir.mkdir(parents=True, exist_ok=True)
    # Arm 1 = Unt (0.20, 0.22, 0.24), Arm 2 = PTBP2 (0.80, 0.82, 0.84) -> IncLevelDifference = Unt - PTBP2 = -0.60
    se_p = post_dir / "SE.MATS.JCEC.txt"
    se_p.write_text(
        _rmats_se_table_text([
            ["1", '"ENSG1"', '"TP53"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
             "20,22,24", "80,78,76", "80,82,84", "20,18,16", "149", "50", "1e-5", "0.001",
             "0.20,0.22,0.24", "0.80,0.82,0.84", "-0.60"],
        ]),
        encoding="utf-8",
    )

    samples_reordered = tmp_path / "samples_reordered.tsv"
    samples_reordered.write_text(
        "sample_id\tcondition\tbiological_sample_id\n"
        "PTBP21\tPTBP2\tD1\nPTBP22\tPTBP2\tD2\nPTBP23\tPTBP2\tD3\n"
        "Unt1\tUnt\tD4\nUnt2\tUnt\tD5\nUnt3\tUnt\tD6\n",
        encoding="utf-8",
    )
    contrasts_p = tmp_path / "contrasts.tsv"
    contrasts_p.write_text("contrast_id\tnumerator\tdenominator\nPTBP2_vs_Unt\tPTBP2\tUnt\n", encoding="utf-8")

    # 1. With explicit PTBP2 - Unt contrasts against direct Unt-PTBP2/rmats_post
    ev_sm, psi_sm, _, _, _ = load_rmats_study(
        rmats_dir=post_dir, samples_file=samples_reordered, contrasts_file=contrasts_p, contrast_id="PTBP2_vs_Unt",
    )
    q_sm = qualify_ps_das_events(ev_sm)
    assert bool(q_sm["reverse_direction"].iloc[0]) is True
    assert float(q_sm["dpsi"].iloc[0]) == pytest.approx(+0.60, abs=1e-4)
    assert float(psi_sm["PTBP21"].iloc[0]) == pytest.approx(0.80, abs=1e-4)
    assert float(psi_sm["Unt1"].iloc[0]) == pytest.approx(0.20, abs=1e-4)

    psi_sn, _, _, _ = extract_matrices_from_bundle_or_rmats(
        rmats_dir=post_dir, samples_path=samples_reordered, contrasts_path=contrasts_p,
    )
    assert float(psi_sn["PTBP21"].iloc[0]) == pytest.approx(0.80, abs=1e-4)
    assert float(psi_sn["Unt1"].iloc[0]) == pytest.approx(0.20, abs=1e-4)

    # 2. Without contrasts file, reordered samplesheet against Unt-PTBP2/rmats_post still maps Unt->arm1 and PTBP2->arm2
    _, psi_sm_noc, _, _, _ = load_rmats_study(rmats_dir=post_dir, samples_file=samples_reordered)
    assert float(psi_sm_noc["PTBP21"].iloc[0]) == pytest.approx(0.80, abs=1e-4)
    assert float(psi_sm_noc["Unt1"].iloc[0]) == pytest.approx(0.20, abs=1e-4)
    psi_sn_noc, _, _, _ = extract_matrices_from_bundle_or_rmats(rmats_dir=post_dir, samples_path=samples_reordered)
    assert float(psi_sn_noc["PTBP21"].iloc[0]) == pytest.approx(0.80, abs=1e-4)
    assert float(psi_sn_noc["Unt1"].iloc[0]) == pytest.approx(0.20, abs=1e-4)

    # 3. SpliceTrace bare directory also detects inverted PTBP2_vs_Unt over Unt-PTBP2/rmats_post
    st_df = load_rmats_measurable_events(post_dir, fa_p, contrast_id="PTBP2_vs_Unt", rmats_mode="JCEC")
    assert float(st_df["dpsi"].iloc[0]) == pytest.approx(+0.60, abs=1e-4)

    # 4. Studio bare Unt-PTBP2/rmats_post reverses dpsi (+0.60), binds PSI correctly, and emits provenance
    rep_bare = tmp_path / "studio_bare_rep"
    rec = run_bulk(
        outdir=rep_bare, rmats=post_dir, samples=samples_reordered, contrasts=contrasts_p,
        fasta=fa_p, gtf=gtf_p, study_name="Bare Dir Orientation & Provenance", species="human",
    )
    assert rec["status"] == "completed"
    das_pq = pd.read_parquet(rep_bare / "outputs/comparisons/PTBP2_vs_Unt/DAS-complete-original-and-report-flags.parquet")
    assert float(das_pq["dpsi"].iloc[0]) == pytest.approx(+0.60, abs=1e-4)
    rep_psi = pd.read_parquet(rep_bare / "outputs/results/sample_level_psi.parquet")
    assert float(rep_psi["PTBP21"].iloc[0]) == pytest.approx(0.80, abs=1e-4)
    assert float(rep_psi["Unt1"].iloc[0]) == pytest.approx(0.20, abs=1e-4)

    prov_path = rep_bare / "outputs/qc/rmats-input-provenance.json"
    assert prov_path.is_file()
    prov_j = json.loads(prov_path.read_text(encoding="utf-8"))
    c0 = prov_j["contrasts"][0]
    assert c0["arm1_samples"] == ["Unt1", "Unt2", "Unt3"]
    assert c0["arm2_samples"] == ["PTBP21", "PTBP22", "PTBP23"]
    exp_sha = hashlib.sha256(se_p.read_bytes()).hexdigest()
    assert c0["files"][0]["sha256"] == exp_sha

    # 5. Short arm samplesheet must raise ValueError rather than synthesizing Unt_rep*
    short_samples = tmp_path / "samples_short.tsv"
    short_samples.write_text(
        "sample_id\tcondition\tbiological_sample_id\n"
        "PTBP21\tPTBP2\tD1\nPTBP22\tPTBP2\tD2\nPTBP23\tPTBP2\tD3\n"
        "Unt1\tUnt\tD4\nUnt2\tUnt\tD5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="(?i)count mismatch"):
        load_rmats_study(rmats_dir=post_dir, samples_file=short_samples, contrasts_file=contrasts_p, contrast_id="PTBP2_vs_Unt")
    with pytest.raises(ValueError, match="(?i)count mismatch"):
        extract_matrices_from_bundle_or_rmats(rmats_dir=post_dir, samples_path=short_samples, contrasts_path=contrasts_p)
    with pytest.raises(ValueError, match="(?i)count mismatch"):
        run_bulk(outdir=tmp_path / "studio_short_fail", rmats=post_dir, samples=short_samples, contrasts=contrasts_p, fasta=fa_p, gtf=gtf_p)


def test_duplicate_comparison_id_in_manifest_rejected_across_all_bulk_loaders(tmp_path: Path) -> None:
    """Duplicate normalized comparison_id entries in manifest.json must be rejected across all 4 bulk loaders."""
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)
    bdir = tmp_path / "dup_manifest_bundle"
    (bdir / "inputs").mkdir(parents=True, exist_ok=True)
    (bdir / "inputs/samplesheet.csv").write_text(
        "sample,condition,biological_sample_id\nUnt1,Unt,D1\nUnt2,Unt,D2\nPTBP1,PTBP,D1\nPTBP2,PTBP,D2\n",
        encoding="utf-8",
    )
    (bdir / "inputs/contrastsheet.csv").write_text("contrast,treatment,control\nPTBP_vs_Unt,PTBP,Unt\n", encoding="utf-8")
    rdir = bdir / "rmats/PTBP_vs_Unt"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "SE.MATS.JCEC.txt").write_text(
        _rmats_se_table_text([
            ["1", '"ENSG1"', '"TP53"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
             "80,82", "20,18", "20,18", "80,82", "149", "50", "0.001", "0.01", "0.80,0.82", "0.20,0.18", "0.62"],
        ]),
        encoding="utf-8",
    )
    (bdir / "manifest.json").write_text(
        json.dumps({
            "comparisons": [
                {"comparison_id": "PTBP_vs_Unt", "rmats_group1": "PTBP", "rmats_group2": "Unt"},
                {"comparison_id": "ptbp_vs_unt", "rmats_group1": "Unt", "rmats_group2": "PTBP"},
            ]
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate normalized comparison_id"):
        load_rmats_study(bundle_dir=bdir, contrast_id="PTBP_vs_Unt")
    with pytest.raises(ValueError, match="Duplicate normalized comparison_id"):
        extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)
    with pytest.raises(ValueError, match="Duplicate normalized comparison_id"):
        load_rmats_measurable_events(rdir, fa_p, bundle_dir=bdir, contrast_id="PTBP_vs_Unt")
    with pytest.raises(ValueError, match="Duplicate normalized comparison_id"):
        run_bulk(outdir=tmp_path / "rep_dup_manifest", bundle=bdir, fasta=fa_p, gtf=gtf_p)


def test_splicetrace_archive_locks_contrast_before_mode_and_preserves_opaque_string_ids(tmp_path: Path) -> None:
    """Missing requested mode in target archive contrast must fail rather than fall back to another contrast; TSV '01' vs '1' stays string-exact."""
    fa_p, _ = _write_mini_gtf_fasta(tmp_path)
    a_jc_txt = _rmats_se_table_text([
        ["1", '"ENSG_A"', '"GA"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
         "80,82", "20,18", "18,16", "82,84", "149", "50", "0.0001", "0.005", "0.80,0.82", "0.18,0.16", "0.6412"],
    ])
    c_jcec_txt = _rmats_se_table_text([
        ["2", '"ENSG_C"', '"GC"', "chr1", "+", "210", "250", "0", "60", "400", "460", "2",
         "15,17", "85,83", "80,82", "20,18", "149", "50", "0.0001", "0.005", "0.15,0.17", "0.80,0.82", "-0.6493"],
    ])
    tp = tmp_path / "mixed_modes.tar.gz"
    with tarfile.open(tp, "w:gz") as tar:
        for arc_path, txt in [("A/SE.MATS.JC.txt", a_jc_txt), ("C/SE.MATS.JCEC.txt", c_jcec_txt)]:
            raw = txt.encode("utf-8")
            ti = tarfile.TarInfo(arc_path)
            ti.size = len(raw)
            tar.addfile(ti, io.BytesIO(raw))

    with pytest.raises(ValueError, match="lacks rMATS JCEC"):
        load_rmats_measurable_events(tp, fa_p, contrast_id="A", rmats_mode="JCEC")
    ok_jc = load_rmats_measurable_events(tp, fa_p, contrast_id="A", rmats_mode="JC")
    assert len(ok_jc) == 1 and ok_jc["gene_id"].iloc[0] == "ENSG_A"

    # Precomputed TSV vs Parquet with zero-padded string IDs "01" and "1"
    pre_df = pd.DataFrame([
        {
            "event_id": "SE:chr1:+:200:260:0:60:400:460", "gene_id": "G1", "gene_symbol": "G1",
            "event_type": "SE", "chr": "chr1", "strand": "+",
            "fdr": 0.001, "PValue": 0.0001, "dpsi": 0.6412, "depth": 50.0, "baseline_PSI": 0.2,
            "comparison_id": "01",
        },
        {
            "event_id": "SE:chr1:+:210:250:0:60:400:460", "gene_id": "G2", "gene_symbol": "G2",
            "event_type": "SE", "chr": "chr1", "strand": "+",
            "fdr": 0.002, "PValue": 0.0002, "dpsi": -0.6493, "depth": 50.0, "baseline_PSI": 0.8,
            "comparison_id": "1",
        },
    ])
    tsv_p = tmp_path / "opaque.tsv"
    pq_p = tmp_path / "opaque.parquet"
    pre_df.to_csv(tsv_p, sep="\t", index=False)
    pre_df.to_parquet(pq_p, index=False)

    got_tsv = load_rmats_measurable_events(tsv_p, fa_p, contrast_id="1")
    got_pq = load_rmats_measurable_events(pq_p, fa_p, contrast_id="1")
    assert len(got_tsv) == 1 and len(got_pq) == 1
    assert str(got_tsv["comparison_id"].iloc[0]) == "1"
    assert float(got_tsv["dpsi"].iloc[0]) == pytest.approx(-0.6493, abs=1e-4)
    assert float(got_pq["dpsi"].iloc[0]) == pytest.approx(-0.6493, abs=1e-4)


def test_studio_sc_multicontrast_scope_and_single_donor_conditional_separation(tmp_path: Path) -> None:
    """Studio SC must isolate multi-contrast Trace merges, reject pooled eligible=True as donor-aware, and mark 1-donor evidence conditional."""
    _, gtf_p = _write_mini_gtf_fasta(tmp_path)
    cells = [f"C{i}" for i in range(12)]
    md = pd.DataFrame({
        "cell_id": cells,
        "donor_id": [f"D{i % 6}" for i in range(12)],
        "condition": ["T" if i < 6 else "C" for i in range(12)],
        "cell_type": ["T"] * 12,
    })
    eids = [f"SE:chr1:+:{200 + k}:{260 + k}:0:60:400:460" for k in range(7)]
    base_rows = []
    for cmp_name, d_sign in (("T_vs_C", 1.0), ("Alt_vs_Ref", -1.0)):
        for idx, eid in enumerate(eids):
            base_rows.append({
                "event_id": eid, "event_type": "SE", "gene_id": f"G{idx + 1}", "gene_symbol": f"SYM{idx + 1}",
                "comparison_id": cmp_name,
                "dpsi": d_sign * 0.40,
                "pvalue": 1e-4 if idx < 7 else 0.70,
                "fdr": 1e-3,
                "eligible": True,
                "event_inference_unit": "exploratory_cell_pooled",
            })
    ev_multi = pd.DataFrame(base_rows)
    psi_mat = pd.DataFrame({c: [0.8 if i < 6 else 0.2] * 7 for i, c in enumerate(cells)}, index=eids)
    sup_mat = pd.DataFrame({c: [30.0] * 7 for c in cells}, index=eids)
    expr_mat = pd.DataFrame(
        {c: [5.0 + (i % 3), 3.0 + (i % 2), 8.0 - (i % 4), 2.0 + 0.1 * i] for i, c in enumerate(cells)},
        index=["G1", "G2", "G3", "G4"],
    )

    # A. Valid Trace only for T_vs_C must NOT overwrite Alt_vs_Ref or double-count 5 significant events as 10
    trace_tc = tmp_path / "trace_tc"
    trace_tc.mkdir()
    (trace_tc / "scsplicetrace_summary.json").write_text(
        json.dumps({"numerator_group": "T", "denominator_group": "C"}), encoding="utf-8",
    )
    pd.DataFrame([
        {
            "event_id": eid,
            "raw_dpsi": 0.50 if i < 5 else (0.14 if i == 5 else 0.60),
            "p_value": 1e-5 if i < 5 else (0.20 if i == 5 else 1e-20),
            "fdr": 8.37e-5 if i < 5 else (0.288 if i == 5 else 1e-18),
            "eligible": i < 6,
            "event_inference_unit": "paired_donor_population" if i < 6 else "insufficient_paired_donors",
            "n_paired_donors": 6 if i < 6 else 2,
            "contributing_donors": 6 if i < 6 else 2,
        }
        for i, eid in enumerate(eids)
    ]).to_csv(trace_tc / "scsplicetrace_event_contrasts.tsv", sep="\t", index=False)

    rep_multi = tmp_path / "rep_sc_multi"
    rec_m = run_single_cell(
        outdir=rep_multi, events=ev_multi, cells=md, psi_matrix=psi_mat, support_matrix=sup_mat,
        expression=expr_mat, gtf=gtf_p, scsplicetrace_dir=trace_tc, study_name="SC Multi-Contrast", species="human",
    )
    pq_m = pd.read_parquet(rep_multi / "outputs/results/sc_differential_splicing.parquet")
    assert set(pq_m["comparison_id"]) == {"T_vs_C", "Alt_vs_Ref"}
    assert float(pq_m.loc[pq_m["comparison_id"] == "Alt_vs_Ref", "dpsi"].iloc[0]) == pytest.approx(-0.40, abs=1e-4)
    man_m = json.loads(Path(rec_m["manifest_json"]).read_text(encoding="utf-8"))
    assert man_m["policy_data"]["stable_significant"]["T_vs_C"]["events"] == 5
    assert man_m["policy_data"]["stable_significant"]["Alt_vs_Ref"] == {"available": False, "events": 0, "genes": 0}
    assert man_m["policy_data"]["exploratory_cell_pooled"]["Alt_vs_Ref"]["events"] == 7

    # B. No-Trace pooled table with generic read-QC eligible=True must NOT be upgraded to donor-aware stable_significant
    rep_pooled = tmp_path / "rep_sc_pooled"
    rec_p = run_single_cell(
        outdir=rep_pooled, events=ev_multi[ev_multi["comparison_id"] == "T_vs_C"].copy(), cells=md,
        psi_matrix=psi_mat, support_matrix=sup_mat, expression=expr_mat, gtf=gtf_p,
        scsplicetrace_dir=None, study_name="SC Pooled Guard", species="human",
    )
    man_p = json.loads(Path(rec_p["manifest_json"]).read_text(encoding="utf-8"))
    assert man_p["policy_data"]["stable_significant"]["T_vs_C"] == {"available": False, "events": 0, "genes": 0}
    assert man_p["policy_data"]["exploratory_cell_pooled"]["T_vs_C"]["events"] == 7

    # C. D=2 study with only 1 contributing donor must keep provenance columns and flag inference as conditional single-donor
    trace_1d = tmp_path / "trace_1d"
    trace_1d.mkdir()
    (trace_1d / "scsplicetrace_summary.json").write_text(
        json.dumps({"numerator_group": "T", "denominator_group": "C", "inference_unit": "stratified_paired_2donor"}),
        encoding="utf-8",
    )
    pd.DataFrame([{
        "event_id": eids[0], "raw_dpsi": 0.45, "p_value": 1e-31, "fdr": 8.42e-30,
        "eligible": True, "event_inference_unit": "stratified_paired_2donor",
        "n_paired_donors": 1, "contributing_donors": 1,
    }]).to_csv(trace_1d / "scsplicetrace_event_contrasts.tsv", sep="\t", index=False)

    rep_1d = tmp_path / "rep_sc_1d"
    rec_1d = run_single_cell(
        outdir=rep_1d, events=ev_multi[(ev_multi["comparison_id"] == "T_vs_C") & (ev_multi["event_id"] == eids[0])].copy(),
        cells=md, psi_matrix=psi_mat.iloc[[0]], support_matrix=sup_mat.iloc[[0]], expression=expr_mat, gtf=gtf_p,
        scsplicetrace_dir=trace_1d, study_name="SC 1-Donor Conditional", species="human",
    )
    pq_1d = pd.read_parquet(rep_1d / "outputs/results/sc_differential_splicing.parquet")
    assert {"n_paired_donors", "contributing_donors", "donor_inference_scope"}.issubset(set(pq_1d.columns))
    assert int(pq_1d["contributing_donors"].iloc[0]) == 1
    assert pq_1d["donor_inference_scope"].iloc[0] == "single_donor_conditional"
    man_1d = json.loads(Path(rec_1d["manifest_json"]).read_text(encoding="utf-8"))
    assert man_1d["policy_data"]["stable_significant"]["T_vs_C"]["available"] is False
    assert man_1d["policy_data"]["stable_significant"]["T_vs_C"]["events"] == 0
    assert man_1d["policy_data"]["conditional_single_donor"]["T_vs_C"]["events"] == 1
    volc_1d = [p for p in man_1d["panels"] if p["id"] == "T_vs_C-sc-volcano"][0]
    assert "Conditional Single-Donor" in volc_1d["title"]


def test_star_barcode_normalization_requires_exact_or_terminal_digits(tmp_path: Path) -> None:
    """STAR barcode resolution must handle hyphenated prefixes cleanly without cross-binding unrelated samples."""
    build_mod = _load_bin_module("katana_sc_build_splice_matrices.py")

    md_hyph = tmp_path / "meta_hyph.tsv"
    md_hyph.write_text(
        "cell_id\tdonor_id\tcondition\nplate-A_ACGT\tD1\tT\nplate-B_CGTA-1\tD2\tC\n",
        encoding="utf-8",
    )
    out_h = build_mod.standardize_metadata(md_hyph, ["plate-A_ACGT-1", "plate-B_CGTA-1"], "condition", "T", "C")
    assert dict(zip(out_h["donor_id"], out_h["cell_id"])) == {"D1": "plate-A_ACGT-1", "D2": "plate-B_CGTA-1"}

    # Observed sample-X_AAAA-1 must NOT be bound to metadata sample-Y_TTTT -> Y-mouse
    md_nomatch = tmp_path / "meta_nomatch.tsv"
    md_nomatch.write_text(
        "cell_id\tdonor_id\tcondition\nsample-Y_TTTT\tY-mouse\tT\notherA-1\totherMouse\tC\n",
        encoding="utf-8",
    )
    out_nm = build_mod.standardize_metadata(md_nomatch, ["sample-X_AAAA-1", "otherA-1"], "condition", "T", "C")
    assert list(out_nm["cell_id"]) == ["otherA-1"]
    assert list(out_nm["donor_id"]) == ["otherMouse"]

    # ACGT-2 in metadata must NOT cross-match ACGT-1 in observed_cells
    md_lane2 = tmp_path / "meta_lane2.tsv"
    md_lane2.write_text("cell_id\tdonor_id\tcondition\nACGT-2\tD1\tT\nother-1\tD2\tC\n", encoding="utf-8")
    out_l2 = build_mod.standardize_metadata(md_lane2, ["ACGT-1", "other-1"], "condition", "T", "C")
    assert list(out_l2["cell_id"]) == ["other-1"]

    # Genuine terminal-digit ambiguity for bare barcode raises ValueError
    md_ambig = tmp_path / "meta_ambig.tsv"
    md_ambig.write_text("cell_id\tdonor_id\tcondition\nACGT\tD1\tT\n", encoding="utf-8")
    with pytest.raises(ValueError, match="terminal `-<digits>` suffix normalization"):
        build_mod.standardize_metadata(md_ambig, ["ACGT-1", "ACGT-2"], "condition", "T", "C")


def test_protein_consequences_restricts_coding_pool_to_validated_host_gene(tmp_path: Path) -> None:
    """A3SS/A5SS protein translation must stay strictly inside the validated pair gene rather than mixing a same-symbol paralog."""
    st_bytes = (ROOT / "packages/splicetrace/splicetrace/protein_consequences.py").read_bytes()
    sc_bytes = (ROOT / "packages/scsplicetrace/scsplicetrace/protein_consequences.py").read_bytes()
    assert hashlib.sha256(st_bytes).hexdigest() == hashlib.sha256(sc_bytes).hexdigest()

    contig_seq = "ATG" + ("GCT" * 200) + "TAA"
    fa_p = _write_fasta_with_fai(tmp_path / "dual_host.fa", [("chr1", contig_seq)])
    gtf_p = tmp_path / "dual_host.gtf"
    # GHOST shares gene_name="DUP_SYM", has longer single-arm exon [0,90) + flanking [200,350) -> 99/89 aa
    # GREAL has BOTH paired arms ([0,90) and [0,60) with flank [200,260)) -> 69/59 aa
    gtf_p.write_text(
        "\n".join([
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GHOST"; gene_name "DUP_SYM"; transcript_id "T_GHOST"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GHOST"; gene_name "DUP_SYM"; transcript_id "T_GHOST"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t201\t350\t.\t+\t.\tgene_id "GHOST"; gene_name "DUP_SYM"; transcript_id "T_GHOST"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t201\t347\t.\t+\t0\tgene_id "GHOST"; gene_name "DUP_SYM"; transcript_id "T_GHOST"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_L"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_L"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t201\t260\t.\t+\t.\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_L"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t201\t257\t.\t+\t0\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_L"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_S"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_S"; transcript_type "protein_coding";',
            'chr1\tsrc\texon\t201\t260\t.\t+\t.\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_S"; transcript_type "protein_coding";',
            'chr1\tsrc\tCDS\t201\t257\t.\t+\t0\tgene_id "GREAL"; gene_name "DUP_SYM"; transcript_id "T_GREAL_S"; transcript_type "protein_coding";',
        ]) + "\n",
        encoding="utf-8",
    )
    events = pd.DataFrame([
        {
            "event_id": "a5_wrong_id_symbol_fallback", "event_type": "A5SS", "gene_id": "OLD_ID", "gene_symbol": "DUP_SYM",
            "chr": "chr1", "strand": "+", "dpsi": 0.4,
            "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 260,
        },
        {
            "event_id": "a5_exact_gene_id", "event_type": "A5SS", "gene_id": "GREAL", "gene_symbol": "DUP_SYM",
            "chr": "chr1", "strand": "+", "dpsi": 0.4,
            "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 260,
        },
    ])
    for mod in (st_pc, sc_pc):
        res = mod.trace_protein_and_nmd_consequences(events, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human").set_index("event_id")
        for ev_k in ("a5_wrong_id_symbol_fallback", "a5_exact_gene_id"):
            assert res.loc[ev_k, "host_transcript_id"] == "T_GREAL_L"
            assert int(res.loc[ev_k, "aa_len_inclusion"]) == 50
            assert int(res.loc[ev_k, "aa_len_exclusion"]) == 40
