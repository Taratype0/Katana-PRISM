"""Round-2 regression tests covering v17 deep re-audit remediations for Katana-PRISM v1.0.3."""
from __future__ import annotations

import filecmp
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for pkg_dir in ("splicemanifold", "splicenet", "splicetrace", "scsplicetrace", "scsplicenet"):
    p = ROOT / "packages" / pkg_dir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scsplicenet.input_contract import validate_sc_usage_input  # noqa: E402
from scsplicetrace import protein_consequences as sc_pc  # noqa: E402
from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast  # noqa: E402
from splicemanifold.cli import main as splicemanifold_main  # noqa: E402
from splicemanifold.ps_das import load_rmats_study  # noqa: E402
from splicenet.cli import main as splicenet_cli_main  # noqa: E402
from splicenet.network import fit_splicenet  # noqa: E402
from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats  # noqa: E402
from splicetrace import protein_consequences as st_pc  # noqa: E402
from splicetrace.rmats_io import load_rmats_measurable_events  # noqa: E402
from splicetrace.rna_forms import IndexedFasta  # noqa: E402


def _write_minimal_rmats_dir(
    rmats_dir: Path,
    *,
    inc_ijc1: list[int],
    inc_sjc1: list[int],
    inc_ijc2: list[int],
    inc_sjc2: list[int],
    inc_lvl1: list[float],
    inc_lvl2: list[float],
    dpsi_val: float,
    write_jc: bool = False,
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
    row_jcec = "\t".join([
        "1", '"ENSG00000100001.1"', '"TEST1"', "chr1", "+",
        "200", "300", "100", "150", "400", "450", "1",
        ",".join(map(str, inc_ijc1)),
        ",".join(map(str, inc_sjc1)),
        ",".join(map(str, inc_ijc2)),
        ",".join(map(str, inc_sjc2)),
        "149", "50", "0.0001", "0.001",
        ",".join(f"{x:.4f}" for x in inc_lvl1),
        ",".join(f"{x:.4f}" for x in inc_lvl2),
        f"{dpsi_val:.4f}",
    ])
    (rmats_dir / "SE.MATS.JCEC.txt").write_text(header + "\n" + row_jcec + "\n", encoding="utf-8")
    if write_jc:
        # Distinct IncLevel1 in JC so JCEC-vs-JC preference is unambiguously observable.
        row_jc = "\t".join([
            "1", '"ENSG00000100001.1"', '"TEST1"', "chr1", "+",
            "200", "300", "100", "150", "400", "450", "1",
            "5,5,5", "50,50,50", "50,50,50", "5,5,5",
            "149", "50", "0.01", "0.05",
            "0.1100,0.1100,0.1100", "0.8900,0.8900,0.8900", "-0.7800",
        ])
        (rmats_dir / "SE.MATS.JC.txt").write_text(header + "\n" + row_jc + "\n", encoding="utf-8")


def test_bulk_rmats_multi_contrast_rejects_bare_dir_and_aligns_bundle(tmp_path: Path) -> None:
    # 1. Bare single rmats_dir with >1 contrast in contrasts.tsv must fail loudly.
    samples_p = tmp_path / "samples.csv"
    samples_p.write_text(
        "sample,condition,biological_sample_id\n"
        "A1,A,D1\nA2,A,D2\nA3,A,D3\n"
        "B1,B,D1\nB2,B,D2\nB3,B,D3\n"
        "C1,C,D1\nC2,C,D2\nC3,C,D3\n",
        encoding="utf-8",
    )
    contrasts_p = tmp_path / "contrasts.csv"
    contrasts_p.write_text(
        "id,treatment,control\n"
        "c1,A,B\n"
        "c2,C,B\n",
        encoding="utf-8",
    )
    bare_rmats = tmp_path / "bare_rmats"
    _write_minimal_rmats_dir(
        bare_rmats,
        inc_ijc1=[40, 42, 44], inc_sjc1=[10, 11, 12],
        inc_ijc2=[10, 12, 14], inc_sjc2=[40, 42, 44],
        inc_lvl1=[0.80, 0.81, 0.82], inc_lvl2=[0.20, 0.21, 0.22],
        dpsi_val=0.60,
    )
    with pytest.raises((ValueError, FileNotFoundError), match="(?i)multiple contrasts|shared bare|No rMATS contrast directory matched"):
        load_rmats_study(
            rmats_dir=bare_rmats,
            samples_file=samples_p,
            contrasts_file=contrasts_p,
            contrast_id="c2",
        )

    # 2. Bundle with manifest.json custom sample order & inverted rMATS arm order aligns across
    # SpliceManifold, SpliceNet, and SpliceTrace, defaulting to JCEC and preserving 3 biological donors.
    bdir = tmp_path / "bundle"
    (bdir / "inputs").mkdir(parents=True, exist_ok=True)
    (bdir / "inputs" / "samplesheet.csv").write_text(
        "sample,condition,biological_sample_id\n"
        "A1,A,Donor_Alpha\nA2,A,Donor_Beta\nA3,A,Donor_Gamma\n"
        "B1,B,Donor_Alpha\nB2,B,Donor_Beta\nB3,B,Donor_Gamma\n",
        encoding="utf-8",
    )
    (bdir / "inputs" / "contrastsheet.csv").write_text(
        "contrast,treatment,control\n"
        "A_vs_B,A,B\n",
        encoding="utf-8",
    )
    c_rmats = bdir / "diff" / "rmats" / "A_vs_B"
    # Stored as group1=B (den), group2=A (num) and permuted order [A3, A1, A2] inside rMATS!
    _write_minimal_rmats_dir(
        c_rmats,
        inc_ijc1=[10, 12, 14], inc_sjc1=[40, 42, 44],   # B3, B2, B1
        inc_ijc2=[85, 75, 80], inc_sjc2=[15, 25, 20],   # A3, A1, A2
        inc_lvl1=[0.20, 0.22, 0.24],
        inc_lvl2=[0.85, 0.75, 0.80],
        dpsi_val=-0.58,
        write_jc=True,
    )
    (bdir / "manifest.json").write_text(
        json.dumps({
            "comparisons": [
                {
                    "comparison_id": "A_vs_B",
                    "rmats_dir": "diff/rmats/A_vs_B",
                    "rmats_group1": "B",
                    "rmats_group2": "A",
                    "rmats_sample_order": {
                        "B": ["B3", "B2", "B1"],
                        "A": ["A3", "A1", "A2"],
                    },
                }
            ]
        }),
        encoding="utf-8",
    )

    ev_sm, psi_sm, _sup_sm, smeta_sm, _emeta = load_rmats_study(bundle_dir=bdir, contrast_id="A_vs_B")
    psi_sn, s_sn, _sup_sn, _gmap = extract_matrices_from_bundle_or_rmats(bundle_dir=bdir)

    assert len({r["biological_sample_id"] for r in smeta_sm}) == 3
    assert s_sn["biological_sample_id"].nunique() == 3
    # Sample PSI for A1/A2/A3/B1/B2/B3 must match identically between SpliceManifold and SpliceNet.
    row_sm = ev_sm.iloc[0]
    key = row_sm["event_id"] if "event_id" in ev_sm.columns else row_sm["coordinate_key"]
    assert "SE:chr1:+:200:300:100:150:400:450" in {
        str(row_sm.get("coordinate_key", "")),
        str(psi_sm.index[0]),
        str(psi_sn.index[0]),
    }
    for sid, expected in [("A1", 0.75), ("A2", 0.80), ("A3", 0.85), ("B1", 0.24), ("B2", 0.22), ("B3", 0.20)]:
        assert float(psi_sm.loc[:, sid].iloc[0]) == pytest.approx(expected, abs=1e-4)
        assert float(psi_sn.loc[:, sid].iloc[0]) == pytest.approx(expected, abs=1e-4)

    # SpliceTrace rMATS loader defaults to JCEC, auto-discovers manifest inversion, and anchors baseline_psi at IncLevel1
    st_ev = load_rmats_measurable_events(c_rmats, bundle_dir=bdir, contrast_id="A_vs_B")
    assert len(st_ev) == 1
    assert float(st_ev.iloc[0]["dpsi"]) == pytest.approx(+0.58, abs=1e-4)
    assert float(st_ev.iloc[0]["baseline_PSI"]) == pytest.approx(0.22, abs=1e-4)
    assert str(key)  # keep key referenced


def test_scsplicetrace_donor_alias_blank_exclusion_single_donor_gate_and_duplicate_cell(tmp_path: Path) -> None:
    rng = np.random.default_rng(20261125)
    # 6 donors in both conditions with unequal cell counts (6 vs 2) and zero true donor effect.
    rows = []
    cols = []
    psi_vals = []
    sup_vals = []
    for d_i in range(6):
        donor = f"M_{d_i + 1}"
        base = 0.25 + 0.1 * d_i
        for c_i in range(6):
            cid = f"{donor}_stim_{c_i}"
            rows.append({"cell_id": cid, "mouse_id": donor, "condition": "stim"})
            cols.append(cid)
            psi_vals.append(base)
            sup_vals.append(30.0)
        for c_i in range(2):
            cid = f"{donor}_ctrl_{c_i}"
            rows.append({"cell_id": cid, "mouse_id": donor, "condition": "ctrl"})
            cols.append(cid)
            psi_vals.append(base)
            sup_vals.append(30.0)
    # Add two blank-donor noise rows that must be excluded before donor counting
    for k, blank_val in enumerate(["", "nan"]):
        cid = f"blank_{k}"
        rows.append({"cell_id": cid, "mouse_id": blank_val, "condition": "stim" if k == 0 else "ctrl"})
        cols.append(cid)
        psi_vals.append(0.95 if k == 0 else 0.05)
        sup_vals.append(40.0)

    ev_id = "SE:chr1:+:200:300:100:150:400:450"
    usage = pd.DataFrame([psi_vals, np.clip(np.array(psi_vals) - 0.05, 0.01, 0.99)], index=[ev_id, ev_id + "_2"], columns=cols).T
    sup = pd.DataFrame([sup_vals, sup_vals], index=[ev_id, ev_id + "_2"], columns=cols).T
    md = pd.DataFrame(rows)
    ev_genes = pd.Series(["GENE1", "GENE2"], index=[ev_id, ev_id + "_2"])

    res_df, donor_df, diag = compute_sc_splicing_contrast(
        usage, md, ev_genes, support=sup,
        group_col="condition", numerator_group="stim", denominator_group="ctrl",
        donor_col=None,
    )
    assert diag["inference_unit"] == "paired_donor_population"
    assert diag["paired_donors"] == 6
    assert diag["cells_excluded_missing_donor_id"] == 2
    res_df_idx = res_df.set_index("event_id")
    assert abs(float(res_df_idx.loc[ev_id, "dpsi"])) < 1e-9
    assert abs(float(res_df_idx.loc[ev_id, "z_shift"])) < 1e-9
    assert donor_df is not None and donor_df.shape[0] == 6

    # Missing donor column fails unless allow_single_donor=True
    md_nodonor = md.drop(columns=["mouse_id"]).iloc[:-2].copy()
    u_nodonor = usage.loc[md_nodonor["cell_id"], :]
    s_nodonor = sup.loc[md_nodonor["cell_id"], :]
    with pytest.raises(ValueError, match="(?i)donor|allow_single_donor"):
        compute_sc_splicing_contrast(
            u_nodonor, md_nodonor, ev_genes, support=s_nodonor,
            group_col="condition", numerator_group="stim", denominator_group="ctrl",
            donor_col=None, allow_single_donor=False,
        )
    _, _, diag_single = compute_sc_splicing_contrast(
        u_nodonor, md_nodonor, ev_genes, support=s_nodonor,
        group_col="condition", numerator_group="stim", denominator_group="ctrl",
        donor_col=None, allow_single_donor=True,
    )
    assert diag_single["inference_unit"] == "single_donor_gene_block"

    # Duplicate cell_id must fail loudly across scsplicetrace, scsplicenet, and splicemanifold single-cell CLI
    md_dup = pd.concat([md.iloc[:-2], md.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        compute_sc_splicing_contrast(
            u_nodonor, md_dup, ev_genes, support=s_nodonor,
            group_col="condition", numerator_group="stim", denominator_group="ctrl",
            donor_col="mouse_id",
        )
    with pytest.raises(ValueError, match="(?i)duplicate"):
        validate_sc_usage_input(u_nodonor, md_dup, ev_genes, support=s_nodonor)

    psi_p = tmp_path / "sc_psi.tsv"
    md_p = tmp_path / "sc_meta.tsv"
    u_nodonor.T.to_csv(psi_p, sep="\t")
    md_dup.to_csv(md_p, sep="\t", index=False)
    with pytest.raises(ValueError, match="(?i)duplicate"):
        splicemanifold_main(["single-cell", "--psi", str(psi_p), "--metadata", str(md_p), "-o", str(tmp_path / "sm_out")])
    assert rng is not None


def test_workflow_sc_scripts_donor_ingress_and_duplicate_guards(tmp_path: Path) -> None:
    # Build minimal valid 2-transcript GTF (SE event: chr1 + 100-200 -> 300-400 -> 500-600; skip 100-200 -> 500-600)
    gtf_p = tmp_path / "mini.gtf"
    gtf_p.write_text(
        "\n".join([
            'chr1\tsrc\texon\t101\t200\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "T_inc";',
            'chr1\tsrc\texon\t301\t400\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "T_inc";',
            'chr1\tsrc\texon\t501\t600\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "T_inc";',
            'chr1\tsrc\texon\t101\t200\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "T_exc";',
            'chr1\tsrc\texon\t501\t600\t.\t+\t.\tgene_id "G1"; gene_name "G1"; transcript_id "T_exc";',
        ]) + "\n",
        encoding="utf-8",
    )
    sj_dir = tmp_path / "sjs"
    sj_dir.mkdir()
    meta_rows = []
    for d in ("Mouse_A", "Mouse_B"):
        for cond, i_cnt, s_cnt in [("Stim", 24, 6), ("Ctrl", 6, 24)]:
            for k in range(3):
                cid = f"{d}_{cond}_{k}"
                meta_rows.append({"cell_id": cid, "mouse_id": d, "condition": cond, "cell_type": "Tcell"})
                (sj_dir / f"{cid}.SJ.out.tab").write_text(
                    f"chr1\t201\t300\t1\t1\t1\t{i_cnt}\t0\t10\n"
                    f"chr1\t401\t500\t1\t1\t1\t{i_cnt}\t0\t10\n"
                    f"chr1\t201\t500\t1\t1\t1\t{s_cnt}\t0\t10\n",
                    encoding="utf-8",
                )

    md_p = tmp_path / "cells_mouse.tsv"
    pd.DataFrame(meta_rows).to_csv(md_p, sep="\t", index=False)

    # 1. katana_sc_build_splice_matrices.py preserves mouse_id (does not collapse to Donor1)
    out_ok = tmp_path / "out_ok"
    subprocess.check_call([
        sys.executable, str(ROOT / "bin" / "katana_sc_build_splice_matrices.py"),
        "--gtf", str(gtf_p), "--sj-dirs", str(sj_dir), "--metadata", str(md_p),
        "--outdir", str(out_ok), "--group-col", "condition", "--numerator", "Stim", "--denominator", "Ctrl",
        "--min-cells", "2", "--min-cell-support", "10",
    ])
    sc_cells = pd.read_csv(out_ok / "sc_cells.tsv", sep="\t")
    assert set(sc_cells["donor_id"]) == {"Mouse_A", "Mouse_B"}

    # 2. Missing donor col fails without --allow-single-donor, succeeds with it
    md_nod = tmp_path / "cells_nodonor.tsv"
    pd.DataFrame(meta_rows).drop(columns=["mouse_id"]).to_csv(md_nod, sep="\t", index=False)
    p_fail = subprocess.run([
        sys.executable, str(ROOT / "bin" / "katana_sc_build_splice_matrices.py"),
        "--gtf", str(gtf_p), "--sj-dirs", str(sj_dir), "--metadata", str(md_nod),
        "--outdir", str(tmp_path / "out_fail"),
    ], capture_output=True, text=True)
    assert p_fail.returncode != 0

    subprocess.check_call([
        sys.executable, str(ROOT / "bin" / "katana_sc_build_splice_matrices.py"),
        "--gtf", str(gtf_p), "--sj-dirs", str(sj_dir), "--metadata", str(md_nod),
        "--outdir", str(tmp_path / "out_single"), "--allow-single-donor",
        "--min-cells", "2", "--min-cell-support", "10",
    ])

    # 3. katana_sc_enumerate_testable_subtypes.py strips blank/NaN donors & rejects duplicate cell_id
    md_enum_rows = meta_rows + [{"cell_id": "blank_extra", "mouse_id": "", "condition": "Stim", "cell_type": "Tcell"}]
    md_enum = tmp_path / "cells_enum.tsv"
    pd.DataFrame(md_enum_rows).to_csv(md_enum, sep="\t", index=False)
    enum_out = tmp_path / "enum_ok"
    subprocess.check_call([
        sys.executable, str(ROOT / "bin" / "katana_sc_enumerate_testable_subtypes.py"),
        "--metadata", str(md_enum), "--celltype-cols", "cell_type", "--group-col", "condition",
        "--numerator", "Stim", "--denominator", "Ctrl", "--min-donors-per-arm", "2", "--outdir", str(enum_out),
    ])
    enum_df = pd.read_csv(enum_out / "sc_subtype_donor_audit.tsv", sep="\t")
    assert int(enum_df.iloc[0]["n_donors_numerator"]) == 2 and int(enum_df.iloc[0]["n_donors_denominator"]) == 2

    md_dup = tmp_path / "cells_dup.tsv"
    pd.DataFrame(meta_rows + [meta_rows[0]]).to_csv(md_dup, sep="\t", index=False)
    p_dup = subprocess.run([
        sys.executable, str(ROOT / "bin" / "katana_sc_enumerate_testable_subtypes.py"),
        "--metadata", str(md_dup), "--celltype-cols", "cell_type", "--group-col", "condition",
        "--numerator", "Stim", "--denominator", "Ctrl", "--outdir", str(tmp_path / "enum_dup"),
    ], capture_output=True, text=True)
    assert p_dup.returncode != 0


def test_protein_consequences_a3_a5_shared_boundary_guard_and_mirror(tmp_path: Path) -> None:
    assert filecmp.cmp(st_pc.__file__, sc_pc.__file__, shallow=False)

    # Construct 300-nt chr1 sequence starting with valid in-frame ORF (ATG + 99 Ala codons GCC/GCT)
    seq = "ATG" + ("GCT" * 99) + "TAA"
    fa_p = tmp_path / "mini.fa"
    fa_p.write_text(">chr1\n" + seq + "\n", encoding="utf-8")
    (tmp_path / "mini.fa.fai").write_text(f"chr1\t{len(seq)}\t6\t{len(seq)}\t{len(seq) + 1}\n", encoding="utf-8")

    # Two real annotated A5SS transcripts sharing downstream exon (200, 300):
    # - short upstream exon (0, 60)
    # - long  upstream exon (0, 90)
    gtf_p = tmp_path / "prot.gtf"
    gtf_p.write_text(
        "\n".join([
            'chr1\tsrc\texon\t1\t90\t.\t+\t.\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_long";',
            'chr1\tsrc\tCDS\t1\t90\t.\t+\t0\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_long";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_long";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_long";',
            'chr1\tsrc\texon\t1\t60\t.\t+\t.\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_short";',
            'chr1\tsrc\tCDS\t1\t60\t.\t+\t0\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_short";',
            'chr1\tsrc\texon\t201\t300\t.\t+\t.\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_short";',
            'chr1\tsrc\tCDS\t201\t297\t.\t+\t0\tgene_id "GPROT"; gene_name "GPROT"; transcript_id "T_short";',
        ]) + "\n",
        encoding="utf-8",
    )

    ev_df = pd.DataFrame([
        {
            "event_id": "A5SS:chr1:+:0:90:0:60:200:300",
            "event_type": "A5SS",
            "gene_id": "GPROT",
            "gene_symbol": "GPROT",
            "chr": "chr1",
            "strand": "+",
            "dpsi": 0.45,
            "IncLevel1": "0.80",
            "IncLevel2": "0.35",
            "exonStart_0base": 0, "exonEnd": 90,
            "longExonStart_0base": 0, "longExonEnd": 90,
            "shortES": 0, "shortEE": 60,
            "flankingES": 200, "flankingEE": 300,
            "upstreamES": 0, "upstreamEE": 60,
            "downstreamES": 200, "downstreamEE": 300,
        },
        {
            # Synthetic longExonEnd=120 absent from annotated GTF
            "event_id": "A5SS:chr1:+:0:120:0:60:200:300",
            "event_type": "A5SS",
            "gene_id": "GPROT",
            "gene_symbol": "GPROT",
            "chr": "chr1",
            "strand": "+",
            "dpsi": 0.45,
            "IncLevel1": "0.80",
            "IncLevel2": "0.35",
            "exonStart_0base": 0, "exonEnd": 120,
            "longExonStart_0base": 0, "longExonEnd": 120,
            "shortES": 0, "shortEE": 60,
            "flankingES": 200, "flankingEE": 300,
            "upstreamES": 0, "upstreamEE": 60,
            "downstreamES": 200, "downstreamEE": 300,
        },
    ])

    out_strict = st_pc.trace_protein_and_nmd_consequences(
        ev_df, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human", require_annotated_a3_a5_pair=True
    ).set_index("event_id")
    assert out_strict.loc["A5SS:chr1:+:0:90:0:60:200:300", "frame_status"] != "complex_junction_usage"
    assert int(out_strict.loc["A5SS:chr1:+:0:90:0:60:200:300", "delta_aa"]) != 0
    assert out_strict.loc["A5SS:chr1:+:0:120:0:60:200:300", "frame_status"] == "complex_junction_usage"
    assert int(out_strict.loc["A5SS:chr1:+:0:120:0:60:200:300", "delta_aa"]) == 0

    out_opt_out = st_pc.trace_protein_and_nmd_consequences(
        ev_df.iloc[[1]], gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="human", require_annotated_a3_a5_pair=False
    )
    assert "[Conditional novel-site model]" in str(out_opt_out.iloc[0]["functional_consequence_summary"])


def test_splicenet_psi_validation_and_small_panel_receipt(tmp_path: Path) -> None:
    bad_psi = pd.DataFrame(
        [[1.4, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.2, 0.3, 0.4, 0.5]],
        index=[f"ev{i}" for i in range(4)],
        columns=[f"s{i}" for i in range(4)],
    )
    with pytest.raises(ValueError, match=r"\[0,\s*1\]|PSI|range"):
        fit_splicenet(bad_psi, tmp_path / "bad_out")

    samples = [f"s{i}" for i in range(12)]
    small_psi = pd.DataFrame(
        [np.linspace(0.15, 0.85, 12), np.linspace(0.85, 0.15, 12), np.tile([0.2, 0.8], 6)],
        index=[f"ev{i}" for i in range(3)],
        columns=samples,
    )
    small_sup = pd.DataFrame(np.full((3, 12), 40.0), index=small_psi.index, columns=samples)
    small_meta = pd.DataFrame({
        "sample_id": samples,
        "biological_sample_id": [f"donor_{i // 2}" for i in range(12)],
        "condition": ["A" if i % 2 == 0 else "B" for i in range(12)],
    })
    u_p = tmp_path / "small_psi.tsv"
    s_p = tmp_path / "small_sup.tsv"
    m_p = tmp_path / "small_meta.tsv"
    small_psi.to_csv(u_p, sep="\t")
    small_sup.to_csv(s_p, sep="\t")
    small_meta.to_csv(m_p, sep="\t", index=False)

    small_out = tmp_path / "small_out"
    rc = splicenet_cli_main([
        "--usage", str(u_p), "--samples", str(m_p), "--support", str(s_p),
        "-o", str(small_out), "--backend", "python",
    ])
    assert rc == 0
    receipt = json.loads((small_out / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt["effective_backend"] == "none"
    assert receipt["skip_reason"] == "insufficient_events"
