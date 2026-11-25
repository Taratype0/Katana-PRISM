"""Round-4 regression tests covering all v19 deep re-audit fixes for Katana-PRISM v1.0.4 RC."""
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
from splicemanifold.ps_das import load_rmats_study  # noqa: E402
from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats  # noqa: E402
from splicetrace import protein_consequences as st_pc  # noqa: E402
from splicetrace.rmats_io import load_rmats_measurable_events  # noqa: E402
from splicetrace.rna_forms import IndexedFasta  # noqa: E402


def _load_bin_module(script_name: str):
    path = ROOT / "bin" / script_name
    spec = importlib.util.spec_from_file_location(f"katana_bin_v104_{script_name.replace('.', '_')}", path)
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


def test_splicetrace_tar_archive_selects_contrast_invariant_to_member_order(tmp_path: Path) -> None:
    """Multi-contrast .tar.gz archives must resolve solely the requested contrast regardless of archive member order."""
    fa_p, _ = _write_mini_gtf_fasta(tmp_path)
    ab_txt = _rmats_se_table_text([
        ["1", '"ENSG_AB"', '"GAB"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
         "80,82", "20,18", "18,16", "82,84", "149", "50", "0.0001", "0.005", "0.80,0.82", "0.18,0.16", "0.6412"],
    ])
    cd_txt = _rmats_se_table_text([
        ["2", '"ENSG_CD"', '"GCD"', "chr1", "+", "210", "250", "0", "60", "400", "460", "2",
         "15,17", "85,83", "80,82", "20,18", "149", "50", "0.0001", "0.005", "0.15,0.17", "0.80,0.82", "-0.6493"],
        ["3", '"ENSG_CD2"', '"GCD2"', "chr1", "+", "220", "255", "0", "60", "400", "460", "3",
         "14,16", "86,84", "82,84", "18,16", "149", "50", "0.0002", "0.008", "0.14,0.16", "0.82,0.84", "-0.6800"],
    ])

    def _pack_tar(name: str, entries: list[tuple[str, str]]) -> Path:
        tp = tmp_path / name
        with tarfile.open(tp, "w:gz") as tar:
            for arc_path, txt in entries:
                raw = txt.encode("utf-8")
                ti = tarfile.TarInfo(arc_path)
                ti.size = len(raw)
                tar.addfile(ti, io.BytesIO(raw))
        return tp

    t_a_first = _pack_tar("a_first.tar.gz", [
        ("rmats/A_vs_B/SE.MATS.JCEC.txt", ab_txt),
        ("rmats/C_vs_D/SE.MATS.JCEC.txt", cd_txt),
    ])
    t_c_first = _pack_tar("c_first.tar.gz", [
        ("rmats/C_vs_D/SE.MATS.JCEC.txt", cd_txt),
        ("rmats/A_vs_B/SE.MATS.JCEC.txt", ab_txt),
    ])

    df_a = load_rmats_measurable_events(t_a_first, fa_p, contrast_id="A_vs_B", rmats_mode="JCEC")
    df_c = load_rmats_measurable_events(t_c_first, fa_p, contrast_id="A_vs_B", rmats_mode="JCEC")
    assert len(df_a) == 1 and len(df_c) == 1
    assert df_a["gene_id"].iloc[0] == "ENSG_AB" and df_c["gene_id"].iloc[0] == "ENSG_AB"
    assert float(df_a["dpsi"].iloc[0]) == pytest.approx(0.6412, abs=1e-4)
    assert float(df_c["dpsi"].iloc[0]) == pytest.approx(0.6412, abs=1e-4)

    df_cd = load_rmats_measurable_events(t_a_first, fa_p, contrast_id="C_vs_D", rmats_mode="JCEC")
    assert len(df_cd) == 2
    assert set(df_cd["gene_id"]) == {"ENSG_CD", "ENSG_CD2"}

    with pytest.raises(ValueError, match="multiple rMATS contrast"):
        load_rmats_measurable_events(t_c_first, fa_p, rmats_mode="JCEC")
    with pytest.raises(ValueError, match="Nonexistent"):
        load_rmats_measurable_events(t_c_first, fa_p, contrast_id="Nonexistent", rmats_mode="JCEC")

    # Single-contrast archive whose directory explicitly names a different contrast must be rejected
    t_only_cd = _pack_tar("only_cd.tar.gz", [("rmats/C_vs_D/SE.MATS.JCEC.txt", cd_txt)])
    with pytest.raises(ValueError, match="A_vs_B"):
        load_rmats_measurable_events(t_only_cd, fa_p, contrast_id="A_vs_B", rmats_mode="JCEC")


def test_splicetrace_precomputed_tsv_and_parquet_filter_or_reject_multi_contrast(tmp_path: Path) -> None:
    """Precomputed TSV and Parquet tables must filter by contrast_id and reject omitted multi-contrast targets."""
    fa_p, _ = _write_mini_gtf_fasta(tmp_path)
    df_pre = pd.DataFrame([
        {
            "event_id": "SE:chr1:+:200:260:0:60:400:460", "gene_id": "G1", "gene_symbol": "G1",
            "event_type": "SE", "chr": "chr1", "strand": "+",
            "fdr": 0.001, "PValue": 0.0001, "dpsi": 0.6412, "depth": 50.0, "baseline_PSI": 0.2,
            "comparison_id": "A_vs_B",
        },
        {
            "event_id": "SE:chr1:+:210:250:0:60:400:460", "gene_id": "G2", "gene_symbol": "G2",
            "event_type": "SE", "chr": "chr1", "strand": "+",
            "fdr": 0.002, "PValue": 0.0002, "dpsi": -0.6493, "depth": 50.0, "baseline_PSI": 0.8,
            "comparison_id": "C_vs_D",
        },
    ])
    tsv_p = tmp_path / "pre.tsv"
    pq_p = tmp_path / "pre.parquet"
    df_pre.to_csv(tsv_p, sep="\t", index=False)
    df_pre.to_parquet(pq_p, index=False)

    for src_p in (tsv_p, pq_p):
        sub_ab = load_rmats_measurable_events(src_p, fa_p, contrast_id="A_vs_B")
        assert list(sub_ab["comparison_id"]) == ["A_vs_B"]
        assert float(sub_ab["dpsi"].iloc[0]) == pytest.approx(0.6412, abs=1e-4)

        with pytest.raises(ValueError, match="multiple contrasts"):
            load_rmats_measurable_events(src_p, fa_p)
        with pytest.raises(ValueError, match="Missing_Contrast"):
            load_rmats_measurable_events(src_p, fa_p, contrast_id="Missing_Contrast")


def test_bulk_malformed_manifest_json_and_whitespace_sample_order(tmp_path: Path) -> None:
    """Malformed manifest.json must raise ValueError across all modules, while whitespace-padded sample IDs resolve cleanly."""
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)
    bdir = tmp_path / "manifest_hygiene"
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

    # 1. Corrupt manifest.json must fail loudly across all 4 packages
    (bdir / "manifest.json").write_text('{"comparisons": [BROKEN JSON', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed manifest.json"):
        load_rmats_study(bundle_dir=bdir, contrast_id="PTBP_vs_Unt")
    with pytest.raises(ValueError, match="Malformed manifest.json"):
        extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)
    with pytest.raises(ValueError, match="Malformed manifest.json"):
        load_rmats_measurable_events(rdir, fa_p, bundle_dir=bdir, contrast_id="PTBP_vs_Unt")
    with pytest.raises(ValueError, match="Malformed manifest.json"):
        run_bulk(outdir=tmp_path / "rep_malformed", bundle=bdir, fasta=fa_p, gtf=gtf_p)

    # 2. Valid manifest with padded sample IDs strips cleanly so SpliceManifold & SpliceNet match 100%
    (bdir / "manifest.json").write_text(
        json.dumps({
            "comparisons": [{
                "comparison_id": "PTBP_vs_Unt",
                "rmats_dir": "rmats/PTBP_vs_Unt",
                "rmats_group1": "PTBP",
                "rmats_group2": "Unt",
                "rmats_sample_order": {"PTBP": [" PTBP1 ", "PTBP2 "], "Unt": [" Unt1 ", "Unt2"]},
            }]
        }),
        encoding="utf-8",
    )
    _, psi_sm, _, _, _ = load_rmats_study(bundle_dir=bdir, contrast_id="PTBP_vs_Unt")
    psi_sn, _, _, _ = extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)
    assert list(psi_sm.columns) == ["PTBP1", "PTBP2", "Unt1", "Unt2"]
    assert set(psi_sn.columns) == {"PTBP1", "PTBP2", "Unt1", "Unt2"}


def test_studio_sc_incomplete_trace_schema_and_no_trace_isolation(tmp_path: Path) -> None:
    """Missing Trace fdr column must never backfill pooled q, and No-Trace runs must isolate exploratory stats."""
    _, gtf_p = _write_mini_gtf_fasta(tmp_path)
    cells = [f"C{i}" for i in range(12)]
    md = pd.DataFrame({
        "cell_id": cells,
        "donor_id": [f"D{i % 6}" for i in range(12)],
        "condition": ["stim" if i < 6 else "ctrl" for i in range(12)],
        "cell_type": ["T"] * 12,
    })
    ev_id = "SE:chr1:+:200:260:0:60:400:460"
    ev_df = pd.DataFrame([{
        "event_id": ev_id, "event_type": "SE", "gene_id": "G1", "gene_symbol": "TP53",
        "chr": "chr1", "strand": "+",
        "exonStart_0base": 200, "exonEnd": 260, "upstreamES": 0, "upstreamEE": 60, "downstreamES": 400, "downstreamEE": 460,
        "dpsi": 0.65, "pvalue": 1e-68, "fdr": 3.79e-66,
    }])
    psi_mat = pd.DataFrame({c: [0.85 if i < 6 else 0.20] for i, c in enumerate(cells)}, index=[ev_id])
    sup_mat = pd.DataFrame({c: [30.0] for c in cells}, index=[ev_id])
    expr_mat = pd.DataFrame({c: [5.0] for c in cells}, index=["TP53"])

    # A. Incomplete scsplicetrace_event_contrasts.tsv (missing `fdr` column)
    inc_trace = tmp_path / "incomplete_trace"
    inc_trace.mkdir()
    (inc_trace / "scsplicetrace_summary.json").write_text(
        json.dumps({"numerator_group": "stim", "denominator_group": "ctrl"}), encoding="utf-8",
    )
    pd.DataFrame([{
        "event_id": ev_id, "raw_dpsi": 0.1475, "p_value": 0.2879, "eligible": True, "event_inference_unit": "paired_donor_population",
    }]).to_csv(inc_trace / "scsplicetrace_event_contrasts.tsv", sep="\t", index=False)

    rep_inc = tmp_path / "rep_inc"
    rec_inc = run_single_cell(
        outdir=rep_inc, events=ev_df, cells=md, psi_matrix=psi_mat, support_matrix=sup_mat,
        expression=expr_mat, gtf=gtf_p, scsplicetrace_dir=inc_trace, study_name="Incomplete Trace Test", species="human",
    )
    pq_inc = pd.read_parquet(rep_inc / "outputs/results/sc_differential_splicing.parquet")
    assert not pq_inc["eligible"].astype(bool).any()
    assert pq_inc["event_inference_unit"].iloc[0] == "incomplete_donor_schema"
    assert pd.to_numeric(pq_inc["fdr"], errors="coerce").isna().all()
    assert pd.to_numeric(pq_inc["pooled_exploratory_fdr"], errors="coerce").notna().all()
    v_inc = pd.read_csv(rep_inc / "outputs/comparisons/stim_vs_ctrl/stim_vs_ctrl-sc-volcano-plot-data.tsv", sep="\t")
    assert not v_inc["_sig"].astype(bool).any()
    man_inc = json.loads(Path(rec_inc["manifest_json"]).read_text(encoding="utf-8"))
    assert man_inc["policy_data"]["stable_significant"]["stim_vs_ctrl"]["available"] is False

    # B. No Trace branch: exploratory label & caption, stable_significant unavailable, isolated exploratory_cell_pooled
    rep_none = tmp_path / "rep_no_trace"
    rec_none = run_single_cell(
        outdir=rep_none, events=ev_df, cells=md, psi_matrix=psi_mat, support_matrix=sup_mat,
        expression=expr_mat, gtf=gtf_p, scsplicetrace_dir=None, study_name="No Trace Exploratory Test", species="human",
    )
    man_none = json.loads(Path(rec_none["manifest_json"]).read_text(encoding="utf-8"))
    volc_none = [p for p in man_none["panels"] if p["id"].endswith("-sc-volcano")][0]
    assert "Exploratory Cell-Pooled" in volc_none["title"]
    assert "donor-aware" not in volc_none["caption"].lower()
    assert "not independent biological replicates" in volc_none["caption"].lower()
    assert man_none["policy_data"]["stable_significant"]["stim_vs_ctrl"] == {"available": False, "events": 0, "genes": 0}
    assert man_none["policy_data"]["exploratory_cell_pooled"]["stim_vs_ctrl"]["events"] == 1


def test_sc_build_splice_matrices_mixed_suffixed_and_unsuffixed_barcodes(tmp_path: Path) -> None:
    """Mixed unsuffixed (`A`) and `-1`-suffixed (`B-1`) STAR barcodes must preserve all unique cells."""
    build_mod = _load_bin_module("katana_sc_build_splice_matrices.py")
    md_p = tmp_path / "meta.tsv"
    md_p.write_text("cell_id\tdonor_id\tcondition\nA\tD1\tstim\nB\tD2\tctrl\n", encoding="utf-8")
    out_md = build_mod.standardize_metadata(md_p, ["A", "B-1"], "condition", "stim", "ctrl")
    assert set(out_md["cell_id"]) == {"A", "B-1"}

    with pytest.raises(ValueError, match="(?i)suffix normalization|duplicate observed"):
        build_mod.standardize_metadata(md_p, ["A", "B-1", "B-2"], "condition", "stim", "ctrl")


def test_nextflow_splicemanifold_sc_and_subtype_forward_donor_col() -> None:
    """Nextflow SPLICEMANIFOLD_SC and SC_SUBTYPE_DOWNSTREAM must forward `--donor-col` consistently."""
    mods_txt = (ROOT / "modules/local/katana_prism_modules.nf").read_text(encoding="utf-8")
    wf_txt = (ROOT / "workflows/katana_singlecell.nf").read_text(encoding="utf-8")

    sm_sc_block = mods_txt.split("process SPLICEMANIFOLD_SC", 1)[1].split("process SCSPLICENET", 1)[0]
    assert "val  donor_col" in sm_sc_block
    assert "${don_arg}" in sm_sc_block
    assert "params.sc_donor_col" in wf_txt.split("SPLICEMANIFOLD_SC(", 1)[1].split(")", 1)[0]

    sub_block = mods_txt.split("process SC_SUBTYPE_DOWNSTREAM", 1)[1].split("process SC_EXTRACT_MULTIPLEXED_BAM", 1)[0]
    assert sub_block.count("${don_arg}") >= 4


def test_protein_consequences_single_gene_a3_a5_and_positive_intron_strict(tmp_path: Path) -> None:
    """Split-gene same-symbol A3/A5 illusions and zero-length touching introns must be rejected in both packages."""
    st_bytes = (ROOT / "packages/splicetrace/splicetrace/protein_consequences.py").read_bytes()
    sc_bytes = (ROOT / "packages/scsplicetrace/scsplicetrace/protein_consequences.py").read_bytes()
    assert hashlib.sha256(st_bytes).hexdigest() == hashlib.sha256(sc_bytes).hexdigest()

    contig_seq = "ATG" + ("GCT" * 160) + "TAA"
    fa_p = _write_fasta_with_fai(tmp_path / "strict.fa", [("chr1", contig_seq)])
    gtf_p = tmp_path / "strict.gtf"
    gtf_p.write_text(
        "\n".join([
            # Valid A5SS + (`l0=0, l1=90`, `s0=0, s1=60`, `f0=200, f1=300`)
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5L";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5L";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5L";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5L";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5S";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5S";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5S";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GP5"; gene_name "GP5"; transcript_id "TP5S";',
            # Split across TWO genes sharing gene_name="SHARED" (GONLYL has long arm, GONLYS has short arm)
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GONLYL"; gene_name "SHARED"; transcript_id "TSHL";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GONLYL"; gene_name "SHARED"; transcript_id "TSHL";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GONLYL"; gene_name "SHARED"; transcript_id "TSHL";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GONLYL"; gene_name "SHARED"; transcript_id "TSHL";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GONLYS"; gene_name "SHARED"; transcript_id "TSHS";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GONLYS"; gene_name "SHARED"; transcript_id "TSHS";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GONLYS"; gene_name "SHARED"; transcript_id "TSHS";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GONLYS"; gene_name "SHARED"; transcript_id "TSHS";',
            # Zero-length intron touching flank (`l1=90 == f0=90`)
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZL";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZL";',
            'chr1\tsrc\texon\t91\t210\t.\t+\t.\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZL";',
            'chr1\tsrc\tCDS\t91\t207\t.\t+\t0\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZL";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZS";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZS";',
            'chr1\tsrc\texon\t91\t210\t.\t+\t.\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZS";',
            'chr1\tsrc\tCDS\t91\t207\t.\t+\t0\tgene_id "GZERO"; gene_name "GZERO"; transcript_id "TZS";',
        ]) + "\n",
        encoding="utf-8",
    )
    events = pd.DataFrame([
        {
            "event_id": "valid_A5_pos", "event_type": "A5SS", "gene_id": "GP5", "gene_symbol": "GP5",
            "chr": "chr1", "strand": "+", "dpsi": 0.4,
            "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300,
        },
        {
            "event_id": "split_symbol_fake_pair", "event_type": "A5SS", "gene_id": "wrong_id", "gene_symbol": "SHARED",
            "chr": "chr1", "strand": "+", "dpsi": 0.4,
            "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300,
        },
        {
            "event_id": "zero_intron_touching", "event_type": "A5SS", "gene_id": "GZERO", "gene_symbol": "GZERO",
            "chr": "chr1", "strand": "+", "dpsi": 0.4,
            "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 90, "flankingEE": 210,
        },
    ])

    for mod in (st_pc, sc_pc):
        res = mod.trace_protein_and_nmd_consequences(
            events, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human",
        ).set_index("event_id")
        assert res.loc["valid_A5_pos", "frame_status"] == "in_frame_indel"
        assert res.loc["split_symbol_fake_pair", "frame_status"] == "complex_junction_usage"
        assert res.loc["zero_intron_touching", "frame_status"] == "complex_junction_usage"

        res_opt = mod.trace_protein_and_nmd_consequences(
            events.iloc[[2]], gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human",
            require_annotated_a3_a5_pair=False,
        )
        assert res_opt["frame_status"].iloc[0] == "complex_junction_usage"
