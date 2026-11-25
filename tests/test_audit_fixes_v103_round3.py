"""Round-3 regression tests covering all v18 deep re-audit fixes for Katana-PRISM v1.0.3."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for pkg_dir in ("splicemanifold", "splicenet", "splicetrace", "scsplicetrace", "scsplicenet", "katana_studio"):
    p = ROOT / "packages" / pkg_dir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from katana_studio.report_builder import _read_tabular, run_bulk, run_single_cell  # noqa: E402
from scsplicenet.cli import main as scsplicenet_cli_main  # noqa: E402
from scsplicenet.input_contract import validate_sc_usage_input  # noqa: E402
from scsplicenet.network import fit_scsplicenet  # noqa: E402
from scsplicetrace import protein_consequences as sc_pc  # noqa: E402
from scsplicetrace.cli import main as scsplicetrace_cli_main  # noqa: E402
from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast  # noqa: E402
from splicemanifold.cli import main as splicemanifold_cli_main  # noqa: E402
from splicemanifold.ps_das import load_rmats_study, qualify_ps_das_events  # noqa: E402
from splicenet.network import fit_splicenet  # noqa: E402
from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats  # noqa: E402
from splicetrace import protein_consequences as st_pc  # noqa: E402
from splicetrace.rmats_io import load_rmats_measurable_events  # noqa: E402
from splicetrace.rna_forms import IndexedFasta  # noqa: E402


def _load_bin_module(script_name: str):
    path = ROOT / "bin" / script_name
    spec = importlib.util.spec_from_file_location(f"katana_bin_{script_name.replace('.', '_')}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_rmats_rows(
    rmats_dir: Path,
    *,
    rows: list[list[str]],
    mode: str = "JCEC",
) -> None:
    rmats_dir.mkdir(parents=True, exist_ok=True)
    header = "\t".join([
        "ID", "GeneID", "geneSymbol", "chr", "strand",
        "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE",
        "downstreamES", "downstreamEE", "ID.1",
        "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
        "IncFormLen", "SkipFormLen", "PValue", "FDR",
        "IncLevel1", "IncLevel2", "IncLevelDifference",
    ])
    lines = [header] + ["\t".join(r) for r in rows]
    (rmats_dir / f"SE.MATS.{mode}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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


def test_studio_sc_blocks_pooled_backfill_onto_ineligible_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Slice -> scSpliceTrace (eligible=False on partial pairing / zero donor effect) -> Studio must NEVER backfill pooled p/q/dpsi."""
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)

    # Partial pairing with zero within-donor effect but huge cell-count imbalance:
    # Donor D1 is shared (4 stim + 4 ctrl, identical PSI=0.50); D2 stim-only (12 cells PSI=0.92); D3 ctrl-only (12 cells PSI=0.08).
    cells = []
    ev1_psi, ev2_psi = [], []
    ev1_sup, ev2_sup = [], []
    for i in range(4):
        cells.append({"cell_id": f"D1_s_{i}", "donor_id": "D1", "condition": "stim", "cell_type": "T_cell"})
        cells.append({"cell_id": f"D1_c_{i}", "donor_id": "D1", "condition": "ctrl", "cell_type": "T_cell"})
        ev1_psi.extend([0.50, 0.50])
        ev2_psi.extend([0.55, 0.55])
        ev1_sup.extend([30.0, 30.0])
        ev2_sup.extend([30.0, 30.0])
    for i in range(12):
        cells.append({"cell_id": f"D2_s_{i}", "donor_id": "D2", "condition": "stim", "cell_type": "T_cell"})
        ev1_psi.append(0.92)
        ev2_psi.append(0.88)
        ev1_sup.append(35.0)
        ev2_sup.append(35.0)
        cells.append({"cell_id": f"D3_c_{i}", "donor_id": "D3", "condition": "ctrl", "cell_type": "T_cell"})
        ev1_psi.append(0.08)
        ev2_psi.append(0.12)
        ev1_sup.append(35.0)
        ev2_sup.append(35.0)

    cids = [c["cell_id"] for c in cells]
    md = pd.DataFrame(cells)
    ev1 = "SE:chr1:+:200:260:0:60:400:460"
    ev2 = "SE:chr1:+:210:255:0:60:400:460"
    ev_df = pd.DataFrame([
        {
            "event_id": ev1, "event_type": "SE", "gene_id": "G1", "gene_symbol": "TP53",
            "chr": "chr1", "strand": "+", "exonStart_0base": 200, "exonEnd": 260,
            "upstreamES": 0, "upstreamEE": 60, "downstreamES": 400, "downstreamEE": 460,
        },
        {
            "event_id": ev2, "event_type": "SE", "gene_id": "G2", "gene_symbol": "BRCA1",
            "chr": "chr1", "strand": "+", "exonStart_0base": 210, "exonEnd": 255,
            "upstreamES": 0, "upstreamEE": 60, "downstreamES": 400, "downstreamEE": 460,
        },
    ])
    psi_df = pd.DataFrame([ev1_psi, ev2_psi], index=[ev1, ev2], columns=cids).T
    sup_df = pd.DataFrame([ev1_sup, ev2_sup], index=[ev1, ev2], columns=cids).T
    expr_df = pd.DataFrame({"TP53": np.full(len(cids), 10.0), "BRCA1": np.full(len(cids), 12.0)}, index=cids)

    ev_p = tmp_path / "events.tsv"
    md_p = tmp_path / "cells.tsv"
    psi_p = tmp_path / "psi.tsv"
    sup_p = tmp_path / "support.tsv"
    expr_p = tmp_path / "expr.tsv"
    ev_df.to_csv(ev_p, sep="\t", index=False)
    md.to_csv(md_p, sep="\t", index=False)
    psi_df.to_csv(psi_p, sep="\t")
    sup_df.to_csv(sup_p, sep="\t")
    expr_df.to_csv(expr_p, sep="\t")

    slice_dir = tmp_path / "sliced"
    slice_mod = _load_bin_module("katana_sc_slice_contrast.py")
    monkeypatch.setattr(
        sys, "argv",
        [
            "katana_sc_slice_contrast.py",
            "--events", str(ev_p), "--psi", str(psi_p), "--support", str(sup_p),
            "--metadata", str(md_p), "--expression", str(expr_p),
            "--group-col", "condition", "--numerator", "stim", "--denominator", "ctrl",
            "--outdir", str(slice_dir),
        ],
    )
    assert slice_mod.main() == 0

    sliced_ev = pd.read_csv(slice_dir / "sc_events.tsv", sep="\t", dtype={"event_id": str})
    # Cell-pooled slicer sees huge artifactual significance before donor-aware scSpliceTrace runs
    assert float(sliced_ev["fdr"].min()) < 1e-3

    trace_dir = tmp_path / "scsplicetrace_out"
    assert scsplicetrace_cli_main([
        "--events", str(slice_dir / "sc_events.tsv"),
        "--psi", str(slice_dir / "sc_psi.tsv.gz"),
        "--support", str(slice_dir / "sc_support.tsv.gz"),
        "--metadata", str(slice_dir / "sc_cells.tsv"),
        "--fasta", str(fa_p),
        "--gtf", str(gtf_p),
        "--group-col", "condition",
        "--numerator", "stim",
        "--denominator", "ctrl",
        "--permutations", "49",
        "-o", str(trace_dir),
    ]) == 0
    st_res = pd.read_csv(trace_dir / "scsplicetrace_event_contrasts.tsv", sep="\t", dtype={"event_id": str})
    assert not st_res["eligible"].astype(bool).any()

    rep_dir = tmp_path / "sc_report"
    run_single_cell(
        outdir=rep_dir,
        events=slice_dir / "sc_events.tsv",
        cells=slice_dir / "sc_cells.tsv",
        psi_matrix=slice_dir / "sc_psi.tsv.gz",
        support_matrix=slice_dir / "sc_support.tsv.gz",
        expression=slice_dir / "sc_expression.tsv.gz",
        gtf=gtf_p,
        scsplicetrace_dir=trace_dir,
        study_name="Zero Effect Partial Pairing Guard Test",
        species="human",
    )

    pq_df = pd.read_parquet(rep_dir / "outputs/results/sc_differential_splicing.parquet")
    assert not pq_df["eligible"].astype(bool).any()
    assert not pq_df["passes_read_support"].astype(bool).any()
    assert pd.to_numeric(pq_df["dpsi"], errors="coerce").isna().all()
    assert pd.to_numeric(pq_df["pvalue"], errors="coerce").isna().all()
    assert pd.to_numeric(pq_df["fdr"], errors="coerce").isna().all()
    assert "pooled_exploratory_fdr" in pq_df.columns
    assert float(pd.to_numeric(pq_df["pooled_exploratory_fdr"], errors="coerce").min()) < 1e-3

    cid = str(pq_df["comparison_id"].iloc[0])
    v_tsv = rep_dir / "outputs/comparisons" / cid / f"{cid}-sc-volcano-plot-data.tsv"
    v_df = pd.read_csv(v_tsv, sep="\t")
    assert not v_df["_sig"].astype(bool).any()
    assert pd.to_numeric(v_df["dpsi"], errors="coerce").isna().all()
    assert not (rep_dir / "outputs/enrichment" / f"{cid}-sc-go-kegg-ora.tsv").exists()

    manifest = json.loads((rep_dir / "report-manifest.json").read_text(encoding="utf-8"))
    volc_panels = [p for p in manifest["panels"] if p["id"].endswith("-sc-volcano")]
    assert volc_panels and "0 donor-eligible events" in volc_panels[0]["title"]


def test_leading_zero_string_ids_preserved_across_modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Leading-zero IDs ('0001', '01', '001', '1') must never coerce to integer '1' across SC or bulk."""
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)
    sc_cells = ["0001", "01", "001", "1"]
    sc_donors = ["001", "001", "01", "01"]
    md = pd.DataFrame({
        "cell_id": sc_cells,
        "donor_id": sc_donors,
        "condition": ["stim", "ctrl", "stim", "ctrl"],
        "cell_type": ["T", "T", "T", "T"],
    })
    ev_id = "SE:chr1:+:200:260:0:60:400:460"
    ev2_id = "SE:chr1:+:210:255:0:60:400:460"
    psi_cxe = pd.DataFrame(
        [[0.80, 0.20], [0.25, 0.75], [0.85, 0.15], [0.30, 0.70]],
        index=sc_cells,
        columns=[ev_id, ev2_id],
    )
    sup_cxe = pd.DataFrame(np.full((4, 2), 30.0), index=sc_cells, columns=[ev_id, ev2_id])
    ev_df = pd.DataFrame([
        {
            "event_id": ev_id, "event_type": "SE", "gene_id": "G1", "gene_symbol": "G1", "chr": "chr1", "strand": "+",
            "exonStart_0base": 200, "exonEnd": 260, "upstreamES": 0, "upstreamEE": 60, "downstreamES": 400, "downstreamEE": 460,
        },
        {
            "event_id": ev2_id, "event_type": "SE", "gene_id": "G2", "gene_symbol": "G2", "chr": "chr1", "strand": "+",
            "exonStart_0base": 210, "exonEnd": 255, "upstreamES": 0, "upstreamEE": 60, "downstreamES": 400, "downstreamEE": 460,
        },
    ])
    md_p, ev_p, psi_p, sup_p = tmp_path / "md.tsv", tmp_path / "ev.tsv", tmp_path / "psi.tsv", tmp_path / "sup.tsv"
    md.to_csv(md_p, sep="\t", index=False)
    ev_df.to_csv(ev_p, sep="\t", index=False)
    psi_cxe.to_csv(psi_p, sep="\t")
    sup_cxe.to_csv(sup_p, sep="\t")

    # Report builder tabular helpers preserve leading zeroes on metadata and column-0 IDs
    md_read = _read_tabular(md_p, as_metadata=True)
    assert list(md_read["cell_id"]) == sc_cells
    assert list(md_read["donor_id"]) == sc_donors
    psi_read = _read_tabular(psi_p, string_index=True)
    assert list(psi_read.iloc[:, 0]) == sc_cells

    # sc_slice_contrast CLI preserves distinct '0001', '01', '001', '1'
    slice_dir = tmp_path / "sliced_ids"
    slice_mod = _load_bin_module("katana_sc_slice_contrast.py")
    monkeypatch.setattr(
        sys, "argv",
        [
            "katana_sc_slice_contrast.py",
            "--events", str(ev_p), "--psi", str(psi_p), "--support", str(sup_p),
            "--metadata", str(md_p), "--group-col", "condition",
            "--numerator", "stim", "--denominator", "ctrl", "--min-cells", "1",
            "--outdir", str(slice_dir),
        ],
    )
    assert slice_mod.main() == 0
    s_md = pd.read_csv(slice_dir / "sc_cells.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert set(s_md["cell_id"]) == set(sc_cells)
    assert set(s_md["donor_id"]) == {"001", "01"}

    # scsplicetrace CLI & summary preserve 4 distinct leading-zero cells and 2 distinct donors ('001' vs '01')
    tr_out = tmp_path / "tr_ids"
    assert scsplicetrace_cli_main([
        "--events", str(ev_p), "--psi", str(psi_p), "--support", str(sup_p), "--metadata", str(md_p),
        "--fasta", str(fa_p), "--gtf", str(gtf_p),
        "--group-col", "condition", "--numerator", "stim", "--denominator", "ctrl", "--permutations", "19",
        "-o", str(tr_out),
    ]) == 0
    tr_sum = json.loads((tr_out / "scsplicetrace_summary.json").read_text(encoding="utf-8"))
    assert int(tr_sum["paired_donors"]) == 2
    assert tr_sum["donors_evaluated"] == ["001", "01"]

    # Bulk leading-zero sample & biological_sample_id across bundle loaders
    bdir = tmp_path / "bulk_ids"
    (bdir / "inputs").mkdir(parents=True, exist_ok=True)
    (bdir / "inputs/samplesheet.csv").write_text(
        "sample,condition,biological_sample_id\n"
        "0001,A,001\n01,A,01\n001,B,001\n1,B,01\n",
        encoding="utf-8",
    )
    (bdir / "inputs/contrastsheet.csv").write_text("contrast,treatment,control\nA_vs_B,A,B\n", encoding="utf-8")
    _write_rmats_rows(
        bdir / "rmats/A_vs_B",
        rows=[[
            "1", '"ENSG1"', '"GENE1"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
            "80,75", "20,25", "20,25", "80,75", "149", "50", "0.001", "0.01", "0.80,0.75", "0.20,0.25", "0.55",
        ]],
    )
    (bdir / "manifest.json").write_text(
        json.dumps({
            "comparisons": [{
                "comparison_id": "A_vs_B",
                "rmats_dir": "rmats/A_vs_B",
                "rmats_group1": "A",
                "rmats_group2": "B",
                "rmats_sample_order": {"A": ["0001", "01"], "B": ["001", "1"]},
            }]
        }),
        encoding="utf-8",
    )
    _, psi_bm, _, sm_meta, _ = load_rmats_study(bundle_dir=bdir, contrast_id="A_vs_B")
    assert set(psi_bm.columns) == {"0001", "01", "001", "1"} and len(psi_bm.columns) == 4
    assert {r["biological_sample_id"] for r in sm_meta} == {"001", "01"}
    psi_bn, bn_meta, _, _ = extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)
    assert set(psi_bn.columns) == {"0001", "01", "001", "1"} and len(psi_bn.columns) == 4
    assert set(bn_meta["biological_sample_id"]) == {"001", "01"}


def test_conflicting_donor_aliases_rejected_unless_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Conflicting populated donor_id vs mouse_id must raise unless --donor-col selects one or aliases agree."""
    rows, cols = [], []
    for d in range(3):
        for grp in ("stim", "ctrl"):
            for k in range(2):
                cid = f"c_{d}_{grp}_{k}"
                cols.append(cid)
                rows.append({
                    "cell_id": cid,
                    "donor_id": "SingleCollapsed",
                    "mouse_id": f"M_{d}",
                    "condition": grp,
                    "cell_type": "Neuron",
                })
    ev_id = "SE:chr1:+:200:260:0:60:400:460"
    ev2_id = "SE:chr1:+:210:255:0:60:400:460"
    u_cxe = pd.DataFrame(
        [[0.8 if "_stim_" in c else 0.2, 0.7 if "_stim_" in c else 0.3] for c in cols],
        index=cols, columns=[ev_id, ev2_id],
    )
    s_cxe = pd.DataFrame(np.full((len(cols), 2), 30.0), index=cols, columns=[ev_id, ev2_id])
    ev_genes = pd.Series(["G1", "G2"], index=[ev_id, ev2_id])
    md_conflict = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="(?i)conflicting.*donor"):
        compute_sc_splicing_contrast(
            u_cxe, md_conflict, ev_genes, support=s_cxe,
            group_col="condition", numerator_group="stim", denominator_group="ctrl",
            donor_col=None,
        )
    with pytest.raises(ValueError, match="(?i)conflicting.*donor"):
        validate_sc_usage_input(u_cxe, md_conflict, ev_genes, support=s_cxe, donor_col=None)

    build_mod = _load_bin_module("katana_sc_build_splice_matrices.py")
    md_p = tmp_path / "md_conflict.tsv"
    md_conflict.to_csv(md_p, sep="\t", index=False)
    with pytest.raises(ValueError, match="(?i)conflicting.*donor"):
        build_mod.standardize_metadata(md_p, cols, "condition", "stim", "ctrl", donor_col=None)
    ok_md = build_mod.standardize_metadata(md_p, cols, "condition", "stim", "ctrl", donor_col="mouse_id")
    assert ok_md["donor_id"].nunique() == 3

    enum_mod = _load_bin_module("katana_sc_enumerate_testable_subtypes.py")
    monkeypatch.setattr(
        sys, "argv",
        [
            "katana_sc_enumerate_testable_subtypes.py",
            "--metadata", str(md_p), "--celltype-cols", "cell_type",
            "--group-col", "condition", "--numerator", "stim", "--denominator", "ctrl",
            "--outdir", str(tmp_path / "enum_bad"),
        ],
    )
    with pytest.raises(ValueError, match="(?i)conflicting.*donor"):
        enum_mod.main()

    psi_exc = tmp_path / "psi_exc.tsv"
    sup_exc = tmp_path / "sup_exc.tsv"
    u_cxe.T.to_csv(psi_exc, sep="\t")
    s_cxe.T.to_csv(sup_exc, sep="\t")
    with pytest.raises(ValueError, match="(?i)conflicting.*donor"):
        splicemanifold_cli_main(["single-cell", "--psi", str(psi_exc), "--support", str(sup_exc), "--metadata", str(md_p), "--outdir", str(tmp_path / "sm_bad")])
    with pytest.raises(ValueError, match="(?i)conflicting.*donor"):
        run_single_cell(
            outdir=tmp_path / "rep_bad",
            events=pd.DataFrame([{"event_id": ev_id, "event_type": "SE", "gene_symbol": "G1"}]),
            cells=md_p,
            donor_col=None,
        )

    _, _, diag_ok = compute_sc_splicing_contrast(
        u_cxe, md_conflict, ev_genes, support=s_cxe,
        group_col="condition", numerator_group="stim", denominator_group="ctrl",
        donor_col="mouse_id",
    )
    assert diag_ok["paired_donors"] == 3


def test_scsplicenet_cli_and_api_donor_col_wiring(tmp_path: Path) -> None:
    """scsplicenet CLI (--donor-col) and fit_scsplicenet(..., donor_col=) propagate explicit donor columns."""
    rng = np.random.default_rng(77)
    rows, cols = [], []
    for d in range(6):
        for g in ("A", "B"):
            for k in range(3):
                cid = f"cell_{d}_{g}_{k}"
                cols.append(cid)
                rows.append({"cell_id": cid, "donor_id": "Coll1", "mouse_id": f"M_{d}", "condition": g, "cell_type": "T", "qc_pass": True})
    md = pd.DataFrame(rows)
    ev_ids = [f"SE:chr1:+:{100 + i * 50}:{140 + i * 50}:0:40:900:950" for i in range(6)]
    latent = np.linspace(-1.0, 1.0, len(cols))
    u_mat = np.clip(0.5 + 0.3 * latent[:, None] + rng.normal(0, 0.05, (len(cols), len(ev_ids))), 0.05, 0.95)
    u_df = pd.DataFrame(u_mat, index=cols, columns=ev_ids)
    s_df = pd.DataFrame(np.full_like(u_mat, 30.0), index=cols, columns=ev_ids)
    ev_genes = pd.Series([f"GENE{i}" for i in range(6)], index=ev_ids)

    net_res = fit_scsplicenet(
        u_df, md, ev_genes, support=s_df, donor_col="mouse_id",
        level="cell_state_distribution", min_module_size=3, repeats=4, require_qc_whitelist=False,
    )
    assert net_res.diagnostics["training_donors"] == [f"M_{d}" for d in range(6)]

    psi_p, sup_p, md_p, eg_p = tmp_path / "u.tsv", tmp_path / "s.tsv", tmp_path / "m.tsv", tmp_path / "eg.tsv"
    u_df.to_csv(psi_p, sep="\t")
    s_df.to_csv(sup_p, sep="\t")
    md.to_csv(md_p, sep="\t", index=False)
    pd.DataFrame({"event_id": ev_ids, "gene_id": list(ev_genes.values)}).to_csv(eg_p, sep="\t", index=False)
    out_cli = tmp_path / "scsplicenet_cli_out"
    assert scsplicenet_cli_main([
        "--usage", str(psi_p), "--support", str(sup_p), "--metadata", str(md_p),
        "--event-genes", str(eg_p), "--donor-col", "mouse_id",
        "--min-module-size", "3", "--repeats", "4", "-o", str(out_cli),
    ]) == 0
    assert (out_cli / "scsplicenet_modules.tsv").is_file()
    cli_sum = json.loads((out_cli / "scsplicenet_summary.json").read_text(encoding="utf-8"))
    assert cli_sum["training_donors"] == [f"M_{d}" for d in range(6)]


def test_sc_build_splice_matrices_rejects_duplicate_sj_cells_and_suffix_collisions(tmp_path: Path) -> None:
    """Duplicate cell_id across raw STAR --sj-dirs and `-1` barcode normalization collisions must fail loudly."""
    build_mod = _load_bin_module("katana_sc_build_splice_matrices.py")
    jmap = {("chr1", 61, 400): 0}

    sj1 = tmp_path / "batch1_sj"
    sj2 = tmp_path / "batch2_sj"
    sj1.mkdir()
    sj2.mkdir()
    sj_line = "chr1\t61\t400\t1\t1\t1\t5\t0\t10\n"
    (sj1 / "CELL_DUP.SJ.out.tab").write_text(sj_line, encoding="utf-8")
    (sj2 / "CELL_DUP.SJ.out.tab").write_text(sj_line, encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)duplicate.*cell_id"):
        build_mod.load_junction_counts([str(sj1), str(sj2)], jmap)

    md_p = tmp_path / "meta.tsv"
    md_p.write_text("cell_id\tdonor_id\tcondition\nAAAC\tD1\tstim\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?i)suffix normalization|duplicate observed"):
        build_mod.standardize_metadata(md_p, ["AAAC-1", "AAAC-2"], "condition", "stim", "ctrl")


def test_bulk_rmats_reversed_contrast_id_conflict_and_neutral_dir(tmp_path: Path) -> None:
    """Unt-PTBP2 contrast-ID conflict is rejected across all 4 bulk modules while neutral F-E still resolves."""
    b_bad = tmp_path / "bad_rev"
    (b_bad / "inputs").mkdir(parents=True, exist_ok=True)
    (b_bad / "inputs/samplesheet.csv").write_text(
        "sample,condition,biological_sample_id\nU1,Unt,D1\nU2,Unt,D2\nP1,PTBP2,D1\nP2,PTBP2,D2\n",
        encoding="utf-8",
    )
    (b_bad / "inputs/contrastsheet.csv").write_text(
        "contrast,treatment,control\nUnt-PTBP2,PTBP2,Unt\n", encoding="utf-8",
    )
    _write_rmats_rows(
        b_bad / "rmats/Unt-PTBP2",
        rows=[[
            "1", '"ENSG1"', '"G1"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
            "80,82", "20,18", "20,18", "80,82", "149", "50", "0.001", "0.01", "0.80,0.82", "0.20,0.18", "0.62",
        ]],
    )
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)

    with pytest.raises(ValueError, match="(?i)ambiguous.*reversed|rmats_group1"):
        load_rmats_study(bundle_dir=b_bad, contrast_id="Unt-PTBP2")
    with pytest.raises(ValueError, match="(?i)ambiguous.*reversed|rmats_group1"):
        extract_matrices_from_bundle_or_rmats(bundle_dir=b_bad)
    with pytest.raises(ValueError, match="(?i)ambiguous.*reversed|rmats_group1"):
        load_rmats_measurable_events(b_bad / "rmats/Unt-PTBP2", bundle_dir=b_bad, contrast_id="Unt-PTBP2")
    with pytest.raises(ValueError, match="(?i)ambiguous.*reversed|rmats_group1"):
        run_bulk(outdir=tmp_path / "rep_rev_fail", bundle=b_bad, fasta=fa_p, gtf=gtf_p)

    # Neutral contrast_id="E_vs_F" with reversed physical folder name "F-E" STILL resolves with inverted=True
    b_neu = tmp_path / "neu_rev"
    (b_neu / "inputs").mkdir(parents=True, exist_ok=True)
    (b_neu / "inputs/samplesheet.csv").write_text(
        "sample,condition,biological_sample_id\nE1,E,D1\nE2,E,D2\nF1,F,D1\nF2,F,D2\n",
        encoding="utf-8",
    )
    (b_neu / "inputs/contrastsheet.csv").write_text("contrast,treatment,control\nE_vs_F,E,F\n", encoding="utf-8")
    _write_rmats_rows(
        b_neu / "rmats/F-E",
        rows=[[
            "1", '"ENSG1"', '"G1"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
            "20,22", "80,78", "80,82", "20,18", "149", "50", "0.001", "0.01", "0.20,0.22", "0.80,0.82", "-0.60",
        ]],
    )
    ev_n, psi_n, _, _, _ = load_rmats_study(bundle_dir=b_neu, contrast_id="E_vs_F")
    assert bool(ev_n.iloc[0]["reverse_direction"]) is True
    assert float(psi_n["E1"].iloc[0]) == pytest.approx(0.80, abs=1e-4)
    assert float(psi_n["F1"].iloc[0]) == pytest.approx(0.20, abs=1e-4)
    q_n = qualify_ps_das_events(ev_n)
    assert float(q_n.iloc[0]["dpsi"]) == pytest.approx(+0.60, abs=1e-4)
    st_n = load_rmats_measurable_events(b_neu / "rmats/F-E", bundle_dir=b_neu, contrast_id="E_vs_F")
    assert float(st_n.iloc[0]["dpsi"]) == pytest.approx(+0.60, abs=1e-4)


def test_bulk_manifest_strict_sample_order_and_row_length_checks(tmp_path: Path) -> None:
    """Unknown Ghost samples, duplicates, wrong-condition arms, and malformed row-2 replicate counts fail loudly."""
    fa_p, gtf_p = _write_mini_gtf_fasta(tmp_path)
    bdir = tmp_path / "strict_manifest"
    (bdir / "inputs").mkdir(parents=True, exist_ok=True)
    (bdir / "inputs/samplesheet.csv").write_text(
        "sample,condition,biological_sample_id\nA1,A,D1\nA2,A,D2\nB1,B,D1\nB2,B,D2\n",
        encoding="utf-8",
    )
    (bdir / "inputs/contrastsheet.csv").write_text("contrast,treatment,control\nA_vs_B,A,B\n", encoding="utf-8")
    good_rows = [[
        "1", '"ENSG1"', '"G1"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
        "80,82", "20,18", "20,18", "80,82", "149", "50", "0.001", "0.01", "0.80,0.82", "0.20,0.18", "0.62",
    ]]
    _write_rmats_rows(bdir / "rmats/A_vs_B", rows=good_rows)

    for bad_order in (
        {"A": ["A1", "Ghost"], "B": ["B1", "B2"]},
        {"A": ["A1", "A1"], "B": ["B1", "B2"]},
        {"A": ["A1", "B1"], "B": ["B1", "B2"]},
    ):
        (bdir / "manifest.json").write_text(
            json.dumps({
                "comparisons": [{
                    "comparison_id": "A_vs_B",
                    "rmats_dir": "rmats/A_vs_B",
                    "rmats_group1": "A",
                    "rmats_group2": "B",
                    "rmats_sample_order": bad_order,
                }]
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="rmats_sample_order"):
            load_rmats_study(bundle_dir=bdir, contrast_id="A_vs_B")
        with pytest.raises(ValueError, match="rmats_sample_order"):
            extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)
        with pytest.raises(ValueError, match="rmats_sample_order"):
            run_bulk(outdir=tmp_path / "rep_bad_ord", bundle=bdir, fasta=fa_p, gtf=gtf_p)

    # Restore valid manifest but truncate replicate list length on row 2
    (bdir / "manifest.json").write_text(
        json.dumps({
            "comparisons": [{
                "comparison_id": "A_vs_B",
                "rmats_dir": "rmats/A_vs_B",
                "rmats_group1": "A",
                "rmats_group2": "B",
                "rmats_sample_order": {"A": ["A1", "A2"], "B": ["B1", "B2"]},
            }]
        }),
        encoding="utf-8",
    )
    bad_rows = [
        good_rows[0],
        [
            "2", '"ENSG1"', '"G1"', "chr1", "+", "210", "255", "0", "60", "400", "460", "2",
            "80", "20,18", "20,18", "80,82", "149", "50", "0.001", "0.01", "0.80,0.82", "0.20,0.18", "0.62",
        ],
    ]
    _write_rmats_rows(bdir / "rmats/A_vs_B", rows=bad_rows)
    with pytest.raises(ValueError, match="(?i)Malformed rMATS replicate count list|list length"):
        load_rmats_study(bundle_dir=bdir, contrast_id="A_vs_B")
    with pytest.raises(ValueError, match="(?i)Malformed rMATS replicate count list|list length"):
        extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)
    with pytest.raises(ValueError, match="(?i)Malformed rMATS replicate count list|list length"):
        run_bulk(outdir=tmp_path / "rep_bad_len", bundle=bdir, fasta=fa_p, gtf=gtf_p)


def test_splicetrace_rmats_io_strict_mode_contrast_and_source_lock(tmp_path: Path) -> None:
    """SpliceTrace must never mix JC into JCEC, accept misspelled contrast_id, or load mismatched rmats_source."""
    bdir = tmp_path / "st_lock"
    (bdir / "inputs").mkdir(parents=True, exist_ok=True)
    (bdir / "inputs/samplesheet.csv").write_text("sample,condition\nA1,A\nB1,B\nC1,C\n", encoding="utf-8")
    (bdir / "inputs/contrastsheet.csv").write_text("contrast,treatment,control\nA_vs_B,A,B\n", encoding="utf-8")

    ab_dir = bdir / "rmats/A_vs_B"
    bc_dir = bdir / "rmats/B_vs_C"
    ab_row = [[
        "1", '"ENSG1"', '"G1"', "chr1", "+", "200", "260", "0", "60", "400", "460", "1",
        "80", "20", "20", "80", "149", "50", "0.001", "0.01", "0.80", "0.20", "0.60",
    ]]
    _write_rmats_rows(ab_dir, rows=ab_row, mode="JCEC")
    (ab_dir / "SE.MATS.JC.txt").unlink(missing_ok=True)
    header_a3 = "\t".join([
        "ID", "GeneID", "geneSymbol", "chr", "strand",
        "longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE",
        "ID.1", "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2",
        "IncFormLen", "SkipFormLen", "PValue", "FDR", "IncLevel1", "IncLevel2", "IncLevelDifference",
    ])
    (ab_dir / "A3SS.MATS.JC.txt").write_text(
        header_a3 + '\n9\t"ENSG2"\t"G2"\tchr1\t+\t200\t260\t220\t260\t0\t60\t9\t40\t10\t10\t40\t40\t20\t0.01\t0.05\t0.8\t0.2\t0.6\n',
        encoding="utf-8",
    )
    _write_rmats_rows(bc_dir, rows=ab_row, mode="JCEC")

    ev_jcec = load_rmats_measurable_events(ab_dir, bundle_dir=bdir, contrast_id="A_vs_B", rmats_mode="JCEC")
    assert set(ev_jcec["event_type"]) == {"SE"}

    jc_only_dir = tmp_path / "jcec_only"
    _write_rmats_rows(jc_only_dir, rows=ab_row, mode="JCEC")
    with pytest.raises(ValueError, match=r"No valid rMATS JC"):
        load_rmats_measurable_events(jc_only_dir, rmats_mode="JC")

    with pytest.raises(ValueError, match="WrongName"):
        load_rmats_measurable_events(ab_dir, bundle_dir=bdir, contrast_id="WrongName", rmats_mode="JCEC")

    with pytest.raises(ValueError, match="A_vs_B"):
        load_rmats_measurable_events(bc_dir, bundle_dir=bdir, contrast_id="A_vs_B", rmats_mode="JCEC")


def test_protein_consequences_strict_flank_adjacency_chrom_strand_and_parity(tmp_path: Path) -> None:
    """Both splicetrace & scsplicetrace must match byte-for-byte and reject non-adjacent/wrong-dir/wrong-strand/chr A3/A5."""
    st_bytes = (ROOT / "packages/splicetrace/splicetrace/protein_consequences.py").read_bytes()
    sc_bytes = (ROOT / "packages/scsplicetrace/scsplicetrace/protein_consequences.py").read_bytes()
    assert hashlib.sha256(st_bytes).hexdigest() == hashlib.sha256(sc_bytes).hexdigest()

    contig_seq = "ATG" + ("GCT" * 160) + "TAA"
    fa_p = _write_fasta_with_fai(tmp_path / "g.fa", [("chr1", contig_seq), ("chr2", contig_seq)])
    gtf_p = tmp_path / "adj.gtf"
    gtf_p.write_text(
        "\n".join([
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKL";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKL";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKL";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKL";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKS";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKS";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKS";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GOKL"; gene_name "GOKL"; transcript_id "TOKS";',
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADL";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADL";',
            'chr1\tsrc\texon\t151\t180\t.\t+\t.\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADL";',
            'chr1\tsrc\tCDS\t151\t180\t.\t+\t0\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADL";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADL";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADL";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADS";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADS";',
            'chr1\tsrc\texon\t151\t180\t.\t+\t.\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADS";',
            'chr1\tsrc\tCDS\t151\t180\t.\t+\t0\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADS";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADS";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GBAD"; gene_name "GBAD"; transcript_id "TBADS";',
        ]) + "\n",
        encoding="utf-8",
    )

    cases = pd.DataFrame([
        {"event_id": "A5SS:chr1:+:0:90:0:60:200:300", "event_type": "A5SS", "gene_id": "GOKL", "gene_symbol": "GOKL", "chr": "chr1", "strand": "+", "dpsi": 0.4, "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300},
        {"event_id": "A5SS:chr1:+:0:90:0:60:200:300_bad", "event_type": "A5SS", "gene_id": "GBAD", "gene_symbol": "GBAD", "chr": "chr1", "strand": "+", "dpsi": 0.4, "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300},
        {"event_id": "A3SS:chr1:+:0:90:0:60:200:300", "event_type": "A3SS", "gene_id": "GOKL", "gene_symbol": "GOKL", "chr": "chr1", "strand": "+", "dpsi": 0.4, "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300},
        {"event_id": "A5SS:chr2:+:0:90:0:60:200:300", "event_type": "A5SS", "gene_id": "GOKL", "gene_symbol": "GOKL", "chr": "chr2", "strand": "+", "dpsi": 0.4, "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300},
        {"event_id": "A5SS:chr1:-:0:90:0:60:200:300", "event_type": "A5SS", "gene_id": "GOKL", "gene_symbol": "GOKL", "chr": "chr1", "strand": "-", "dpsi": 0.4, "longExonStart_0base": 0, "longExonEnd": 90, "shortES": 0, "shortEE": 60, "flankingES": 200, "flankingEE": 300},
    ])

    for mod in (st_pc, sc_pc):
        out_strict = mod.trace_protein_and_nmd_consequences(
            cases, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human",
        ).set_index("event_id")
        assert out_strict.loc["A5SS:chr1:+:0:90:0:60:200:300", "frame_status"] == "in_frame_indel"
        assert int(out_strict.loc["A5SS:chr1:+:0:90:0:60:200:300", "delta_aa"]) == 10
        assert out_strict.loc["A5SS:chr1:+:0:90:0:60:200:300_bad", "frame_status"] == "complex_junction_usage"
        assert out_strict.loc["A3SS:chr1:+:0:90:0:60:200:300", "frame_status"] == "complex_junction_usage"
        assert out_strict.loc["A5SS:chr2:+:0:90:0:60:200:300", "frame_status"] == "complex_junction_usage"
        assert out_strict.loc["A5SS:chr1:-:0:90:0:60:200:300", "frame_status"] == "complex_junction_usage"

        out_opt = mod.trace_protein_and_nmd_consequences(
            cases.iloc[[1]], gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human",
            require_annotated_a3_a5_pair=False,
        )
        assert "[Conditional novel-site model]" in str(out_opt.iloc[0]["functional_consequence_summary"])


def test_splicenet_rejects_inf_psi(tmp_path: Path) -> None:
    """fit_splicenet must reject +inf and -inf PSI values directly."""
    for bad_val in (np.inf, -np.inf):
        psi = pd.DataFrame(
            [[bad_val, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]],
            index=[f"ev{i}" for i in range(4)],
            columns=[f"s{i}" for i in range(4)],
        )
        with pytest.raises(ValueError, match=r"finite|Inf|range"):
            fit_splicenet(psi, tmp_path / f"inf_{bad_val}")
