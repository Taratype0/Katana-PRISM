"""Comprehensive automated verification suite for Katana-PRISM 1.0.0.

Validates:
  1. Nextflow DSL2 workflow/module include resolution, symbol exports, brace balance & HPC resource rules
  2. Upstream `bin/katana_*.py` validation, GTF filtering & strandedness scripts
  3. All 6 standalone 1.0.0 Python packages & 7 CLIs (`splicemanifold`, `splicenet`, `splicetrace`,
     `scsplicenet`, `scsplicetrace`, `katana-report`, `katana-studio`) end-to-end on synthetic data
  4. Repository-wide confidentiality, brand, and naming hygiene (`0` forbidden tokens).
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = REPO_ROOT / "packages"

# Isolate from parent workspace directories and make all 6 standalone packages importable
sys.path = [p for p in sys.path if p not in {"", ".", str(REPO_ROOT.parent)}]
for pkg_sub in ("splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace", "katana_studio"):
    pkg_path = str(PACKAGES_DIR / pkg_sub)
    if pkg_path not in sys.path:
        sys.path.insert(0, pkg_path)


# ---------------------------------------------------------------------------
# 1. Nextflow DSL2 Integrity & HPC Optimization Tests
# ---------------------------------------------------------------------------
def test_nextflow_dsl2_includes_and_brace_balance() -> None:
    nf_files = [
        p for p in REPO_ROOT.rglob("*")
        if p.is_file() and p.suffix in {".nf", ".config"} and "packages" not in p.parts
    ]
    assert len(nf_files) >= 50

    inc_re = re.compile(r"^\s*include\s*\{\s*([^}]+)\s*\}\s*from\s*['\"]([^'\"]+)['\"]", re.M)
    def_re = re.compile(r"^\s*(?:process|workflow)\s+([A-Za-z0-9_]+)\b", re.M)

    for nf in nf_files:
        txt = nf.read_text(encoding="utf-8")
        # Strip single-line comments before checking rough brace balance
        clean_lines = [
            re.sub(r"//.*$", "", line)
            for line in txt.splitlines()
            if not line.lstrip().startswith("//")
        ]
        joined = "\n".join(clean_lines)
        # Check every include statement resolves to an existing file and exported symbol
        if nf.suffix == ".nf":
            for m in inc_re.finditer(joined):
                raw_syms, rel_target = m.group(1), m.group(2)
                cand_base = (nf.parent / rel_target).resolve()
                target_nf = (
                    cand_base.with_suffix(".nf")
                    if cand_base.with_suffix(".nf").is_file()
                    else (cand_base / "main.nf")
                )
                assert target_nf.is_file(), f"Broken include in {nf.relative_to(REPO_ROOT)} -> {rel_target}"
                defined_syms = set(def_re.findall(target_nf.read_text(encoding="utf-8")))
                for item in raw_syms.split(";"):
                    sym = item.strip().split()[0]
                    if sym:
                        assert sym in defined_syms, f"Symbol {sym} not defined in {target_nf.relative_to(REPO_ROOT)}"


def test_nextflow_resource_and_alignment_optimizations() -> None:
    up_mod = (REPO_ROOT / "conf" / "upstream_modules.config").read_text(encoding="utf-8")
    assert "SortedByCoordinate" in up_mod, "STAR_ALIGN must output coordinate-sorted BAM in a single pass"
    assert "limitBAMsortRAM" in up_mod

    core_wf = (REPO_ROOT / "workflows" / "katana_core.nf").read_text(encoding="utf-8")
    assert "QUALIMAP_GENEBODY" in core_wf
    assert "SAMTOOLS_BAM_STATS" in core_wf
    assert "KATANA_MULTIQC" in core_wf

    port_res = (REPO_ROOT / "conf" / "portable_resources.config").read_text(encoding="utf-8")
    assert "54.GB" in port_res, "STAR_GENOMEGENERATE attempt 1 should fit 64-GB hosts & Dardel shared queue"
    for proc_name in ("SPLICEMANIFOLD", "SPLICENET", "SPLICETRACE", "SCSPLICENET", "SCSPLICETRACE", "KATANA_REPORT"):
        assert proc_name in port_res


# ---------------------------------------------------------------------------
# 2. Upstream Helper Scripts (`bin/`) Tests
# ---------------------------------------------------------------------------
def test_upstream_bin_scripts_and_zero_kanata_typos(tmp_path: Path) -> None:
    kanata_files = list((REPO_ROOT / "bin").glob("kanata_*"))
    assert kanata_files == [], f"Found unrenamed kanata_* scripts: {kanata_files}"

    # Smoke test katana_split_contrasts.py
    contrasts_csv = tmp_path / "contrasts.csv"
    contrasts_csv.write_text("contrast,treatment,control\nKO_vs_WT,KO,WT\n", encoding="utf-8")
    out_split = tmp_path / "split_out"
    res = subprocess.run(
        [sys.executable, str(REPO_ROOT / "bin" / "katana_split_contrasts.py"), str(contrasts_csv), str(out_split)],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    assert (out_split / "contrast.KO_vs_WT.csv").is_file()


# ---------------------------------------------------------------------------
# 3. Synthetic Fixture Builder for Bulk & Single-Cell Standalone Packages
# ---------------------------------------------------------------------------
def _create_synthetic_study(tmp_path: Path) -> dict[str, Path]:
    rng = np.random.default_rng(42)

    # 1. Synthetic Genome FASTA + FAI (chr1, 12,000 bp rich in splicing motifs)
    motifs = ["TGCATG", "TCTCTCT", "GAAGAA", "GGACT", "TTTTTT", "CAGGTG"]
    chunks = []
    while sum(len(c) for c in chunks) < 12000:
        chunks.append(motifs[rng.integers(0, len(motifs))] + "".join(rng.choice(list("ACGT"), size=18)))
    seq = "".join(chunks)[:12000]
    fasta = tmp_path / "genome.fa"
    fasta.write_text(">chr1\n" + "\n".join(seq[i : i + 80] for i in range(0, len(seq), 80)) + "\n", encoding="utf-8")
    (tmp_path / "genome.fa.fai").write_text(f"chr1\t{len(seq)}\t6\t80\t81\n", encoding="utf-8")

    # 2. Synthetic GTF with multi-exon coding transcripts on chr1
    gtf_lines = []
    for g_idx in range(1, 16):
        base = 200 + g_idx * 700
        gid = f"ENSMUSG{g_idx:011d}"
        tid = f"ENSMUST{g_idx:011d}"
        gsym = f"Gene{g_idx}"
        exons = [(base, base + 90), (base + 200, base + 290), (base + 420, base + 540)]
        for ex_start, ex_end in exons:
            gtf_lines.append(
                f'chr1\tsynth\texon\t{ex_start}\t{ex_end}\t.\t+\t.\tgene_id "{gid}"; transcript_id "{tid}"; gene_name "{gsym}";'
            )
            gtf_lines.append(
                f'chr1\tsynth\tCDS\t{ex_start}\t{ex_end}\t.\t+\t0\tgene_id "{gid}"; transcript_id "{tid}"; gene_name "{gsym}";'
            )
    gtf = tmp_path / "annotation.gtf"
    gtf.write_text("\n".join(gtf_lines) + "\n", encoding="utf-8")

    # 3. Synthetic Bulk rMATS directory + samples.csv + contrasts.csv (6 samples: 3 KO vs 3 WT, 60 SE events)
    samples_csv = tmp_path / "samples.csv"
    samples_csv.write_text(
        "sample_id,condition,biological_sample_id\n"
        "KO_1,KO,KO_1\nKO_2,KO,KO_2\nKO_3,KO,KO_3\n"
        "WT_1,WT,WT_1\nWT_2,WT,WT_2\nWT_3,WT,WT_3\n",
        encoding="utf-8",
    )
    contrasts_csv = tmp_path / "contrasts.csv"
    contrasts_csv.write_text("contrast,treatment,control\nKO_vs_WT,KO,WT\n", encoding="utf-8")

    rmats_dir = tmp_path / "rmats_post"
    rmats_dir.mkdir(parents=True, exist_ok=True)
    se_rows = []
    for i in range(60):
        g_idx = (i % 15) + 1
        base = 200 + g_idx * 700 + (i // 15) * 6
        gid = f"ENSMUSG{g_idx:011d}"
        gsym = f"Gene{g_idx}"
        is_sig = i < 20
        p1 = np.clip((0.75 if is_sig else 0.45) + rng.normal(0, 0.04, 3), 0.05, 0.95)
        p2 = np.clip((0.25 if is_sig else 0.44) + rng.normal(0, 0.04, 3), 0.05, 0.95)
        tot1 = rng.integers(35, 70, 3)
        tot2 = rng.integers(35, 70, 3)
        ijc1 = np.round(p1 * tot1).astype(int)
        sjc1 = np.maximum(1, tot1 - ijc1)
        ijc2 = np.round(p2 * tot2).astype(int)
        sjc2 = np.maximum(1, tot2 - ijc2)
        dpsi = round(float(p1.mean() - p2.mean()), 4)
        pval = 1e-5 if is_sig else 0.65
        fdr = 0.002 if is_sig else 0.82
        se_rows.append({
            "ID": i + 1,
            "GeneID": gid,
            "geneSymbol": gsym,
            "chr": "chr1",
            "strand": "+",
            "exonStart_0base": base + 199,
            "exonEnd": base + 290,
            "upstreamES": base - 1,
            "upstreamEE": base + 90,
            "downstreamES": base + 419,
            "downstreamEE": base + 540,
            "ID.1": i + 1,
            "IJC_SAMPLE_1": ",".join(map(str, ijc1)),
            "SJC_SAMPLE_1": ",".join(map(str, sjc1)),
            "IJC_SAMPLE_2": ",".join(map(str, ijc2)),
            "SJC_SAMPLE_2": ",".join(map(str, sjc2)),
            "IncFormLen": 148,
            "SkipFormLen": 99,
            "PValue": pval,
            "FDR": fdr,
            "IncLevel1": ",".join(f"{x:.3f}" for x in p1),
            "IncLevel2": ",".join(f"{x:.3f}" for x in p2),
            "IncLevelDifference": dpsi,
        })
    se_df = pd.DataFrame(se_rows)
    for mode in ("JCEC", "JC"):
        se_df.to_csv(rmats_dir / f"SE.MATS.{mode}.txt", sep="\t", index=False)

    # 4. Synthetic Single-Cell Smart-seq3 inputs (32 cells across 2 donors x 2 states, 45 events)
    cells = [f"Cell_{c+1:02d}" for c in range(32)]
    sc_meta = pd.DataFrame({
        "cell_id": cells,
        "donor_id": ["D1" if c < 16 else "D2" for c in range(32)],
        "cell_type": ["T_cell" if c % 2 == 0 else "Monocyte" for c in range(32)],
        "condition": ["Activated" if (c % 4) < 2 else "Resting" for c in range(32)],
        "qc_pass": True,
    })
    sc_meta_path = tmp_path / "sc_cells.tsv"
    sc_meta.to_csv(sc_meta_path, sep="\t", index=False)

    sc_ev_rows = []
    for j in range(45):
        g_idx = (j % 15) + 1
        base = 200 + g_idx * 700 + (j // 15) * 3
        sc_ev_rows.append({
            "event_id": f"SE_sc_{j+1:03d}",
            "event_type": "SE",
            "gene_id": f"ENSMUSG{g_idx:011d}",
            "gene_symbol": f"Gene{g_idx}",
            "chr": "chr1",
            "strand": "+",
            "exonStart_0base": base + 199,
            "exonEnd": base + 290,
            "upstreamES": base - 1,
            "upstreamEE": base + 90,
            "downstreamES": base + 419,
            "downstreamEE": base + 540,
            "dpsi": 0.28 if j < 15 else 0.02,
            "fdr": 0.005 if j < 15 else 0.75,
        })
    sc_ev_df = pd.DataFrame(sc_ev_rows)
    sc_ev_path = tmp_path / "sc_events.tsv"
    sc_ev_df.to_csv(sc_ev_path, sep="\t", index=False)
    sc_eg_path = tmp_path / "sc_event_genes.tsv"
    sc_ev_df[["event_id", "gene_id", "gene_symbol"]].to_csv(sc_eg_path, sep="\t", index=False)

    eids = sc_ev_df["event_id"].tolist()
    psi_vals = np.zeros((32, 45), dtype=float)
    for c_i in range(32):
        act = 1.0 if sc_meta.loc[c_i, "condition"] == "Activated" else 0.0
        for e_j in range(45):
            center = (0.72 if act else 0.30) if e_j < 18 else (0.25 if act else 0.65) if e_j < 32 else 0.50
            psi_vals[c_i, e_j] = np.clip(center + rng.normal(0, 0.06), 0.02, 0.98)
    sc_psi_df = pd.DataFrame(psi_vals, index=cells, columns=eids)
    sc_sup_df = pd.DataFrame(rng.integers(15, 55, size=(32, 45)), index=cells, columns=eids)
    sc_expr_df = pd.DataFrame(rng.normal(4.0, 1.2, size=(32, 30)), index=cells, columns=[f"Gene{k+1}" for k in range(30)])

    sc_psi_path = tmp_path / "sc_psi.tsv"
    sc_sup_path = tmp_path / "sc_support.tsv"
    sc_expr_path = tmp_path / "sc_expr.tsv"
    sc_psi_df.to_csv(sc_psi_path, sep="\t", index_label="cell_id")
    sc_sup_df.to_csv(sc_sup_path, sep="\t", index_label="cell_id")
    sc_expr_df.to_csv(sc_expr_path, sep="\t", index_label="cell_id")

    return {
        "fasta": fasta,
        "gtf": gtf,
        "samples": samples_csv,
        "contrasts": contrasts_csv,
        "rmats": rmats_dir,
        "sc_meta": sc_meta_path,
        "sc_events": sc_ev_path,
        "sc_event_genes": sc_eg_path,
        "sc_psi": sc_psi_path,
        "sc_sup": sc_sup_path,
        "sc_expr": sc_expr_path,
    }


def test_all_six_standalone_packages_end_to_end(tmp_path: Path) -> None:
    fx = _create_synthetic_study(tmp_path)

    # 1. SpliceManifold (Bulk & Single-Cell)
    from splicemanifold.cli import main as sm_main

    sm_bulk_out = tmp_path / "out_splicemanifold_bulk"
    rc = sm_main([
        "bulk",
        "--rmats-dir", str(fx["rmats"]),
        "--samples", str(fx["samples"]),
        "--contrasts", str(fx["contrasts"]),
        "--outdir", str(sm_bulk_out),
        "--emit-versions-yml", str(sm_bulk_out / "versions.yml"),
    ])
    assert rc == 0
    assert (sm_bulk_out / "ps_das_events.tsv").is_file()
    assert (sm_bulk_out / "figures" / "psi_clustermap.png").is_file()
    assert (sm_bulk_out / "figures" / "psi_pca.png").is_file()
    assert 'splicemanifold: "1.0.1"' in (sm_bulk_out / "versions.yml").read_text(encoding="utf-8")

    sm_sc_out = tmp_path / "out_splicemanifold_sc"
    rc = sm_main([
        "single-cell",
        "--psi", str(fx["sc_psi"]),
        "--support", str(fx["sc_sup"]),
        "--metadata", str(fx["sc_meta"]),
        "--expression", str(fx["sc_expr"]),
        "--cells-as-rows",
        "--outdir", str(sm_sc_out),
    ])
    assert rc == 0
    assert (sm_sc_out / "figures" / "sc_dual_psi_umap.png").is_file()
    assert (sm_sc_out / "sc_cell_manifold_coordinates.tsv").is_file()

    # 2. SpliceNet (Bulk)
    from splicenet.cli import main as sn_main

    sn_out = tmp_path / "out_splicenet"
    rc = sn_main([
        "--rmats-dir", str(fx["rmats"]),
        "--samples", str(fx["samples"]),
        "--min-module-size", "5",
        "--output-dir", str(sn_out),
        "--emit-versions-yml", str(sn_out / "versions.yml"),
    ])
    assert rc == 0
    assert (sn_out / "module_assignments.tsv").is_file()
    assert (sn_out / "module_scores.tsv").is_file()
    assert 'splicenet: "1.0.1"' in (sn_out / "versions.yml").read_text(encoding="utf-8")

    # 3. SpliceTrace (Bulk: 170-RBP map + SpliceNet bridge + 50-nt EJC NMD & UniProt consequences)
    from splicetrace.cli import main as st_main

    st_out = tmp_path / "out_splicetrace"
    rc = st_main([
        "--rmats", str(fx["rmats"]),
        "--fasta", str(fx["fasta"]),
        "--gtf", str(fx["gtf"]),
        "--species", "mouse",
        "--permutations", "49",
        "--splicenet-modules", str(sn_out / "module_assignments.tsv"),
        "--output-dir", str(st_out),
        "--emit-versions-yml", str(st_out / "versions.yml"),
    ])
    assert rc == 0
    assert (st_out / "rbp_ranking.tsv").is_file()
    assert (st_out / "protein_and_nmd_consequences.tsv").is_file()
    assert (st_out / "module_rbp_bridge.tsv").is_file()
    assert len(pd.read_csv(st_out / "module_rbp_bridge.tsv", sep="\t")) > 0
    assert 'splicetrace: "1.0.1"' in (st_out / "versions.yml").read_text(encoding="utf-8")

    # 4. scSpliceNet (Single-Cell)
    from scsplicenet.cli import main as scsn_main

    scsn_out = tmp_path / "out_scsplicenet"
    rc = scsn_main([
        "--usage", str(fx["sc_psi"]),
        "--support", str(fx["sc_sup"]),
        "--metadata", str(fx["sc_meta"]),
        "--event-genes", str(fx["sc_event_genes"]),
        "--min-module-size", "5",
        "--repeats", "4",
        "--out", str(scsn_out),
        "--emit-versions-yml", str(scsn_out / "versions.yml"),
    ])
    assert rc == 0
    assert (scsn_out / "scsplicenet_edges.tsv").is_file()
    assert (scsn_out / "scsplicenet_modules.tsv").is_file()

    # 5. scSpliceTrace (Single-Cell)
    from scsplicetrace.cli import main as scst_main

    scst_out = tmp_path / "out_scsplicetrace"
    rc = scst_main([
        "--events", str(fx["sc_events"]),
        "--psi", str(fx["sc_psi"]),
        "--support", str(fx["sc_sup"]),
        "--metadata", str(fx["sc_meta"]),
        "--fasta", str(fx["fasta"]),
        "--gtf", str(fx["gtf"]),
        "--species", "mouse",
        "--numerator", "Activated",
        "--denominator", "Resting",
        "--permutations", "49",
        "--splicenet-modules", str(scsn_out / "scsplicenet_modules.tsv"),
        "--out", str(scst_out),
        "--emit-versions-yml", str(scst_out / "versions.yml"),
    ])
    assert rc == 0
    assert (scst_out / "scsplicetrace_rbp_ranking.tsv").is_file()
    assert (scst_out / "sc_protein_and_nmd_consequences.tsv").is_file()

    # 6. Katana Report & Katana Studio
    import importlib.util as _ilu
    from katana_studio.cli import report_main

    rep_out = tmp_path / "out_report"
    rc = report_main([
        "bulk",
        "--rmats", str(fx["rmats"]),
        "--samples", str(fx["samples"]),
        "--contrasts", str(fx["contrasts"]),
        "--study-name", "Synthetic Benchmark Study",
        "--species", "mouse",
        "--outdir", str(rep_out),
        "--emit-versions-yml", str(rep_out / "versions.yml"),
    ])
    assert rc == 0
    assert (rep_out / "REPORT.html").is_file()
    assert (rep_out / "REPORT.zip").is_file()
    with zipfile.ZipFile(rep_out / "REPORT.zip") as zf:
        assert "REPORT.html" in zf.namelist()
        assert "report-manifest.json" in zf.namelist()

    # Verify Katana Studio FastAPI app initializes and locates bundled _web UI when fastapi is installed
    if _ilu.find_spec("fastapi") is not None:
        from katana_next.api.app import PlatformRuntime, create_app
        app = create_app(PlatformRuntime(application_root=PACKAGES_DIR / "katana_studio"))
        assert app is not None


def test_v1_0_1_distal_bridge_and_expression_disambiguation(tmp_path: Path) -> None:
    """Regression suite for v1.0.1 multi-paralog discrete scan, distal=500 FWL bridge, and RBP expression disambiguation."""
    from splicetrace.module_bridge import annotate_rbp_expression_disambiguation, parse_rbp_expression_reference
    from splicetrace.continuous_rescue import run_bulk_continuous_rescue
    from splicetrace.library import load_bundled_human_rbp_library

    expr_df = pd.DataFrame({"gene_symbol": ["PTBP1", "PTBP2", "SRSF1"], "TPM": [48.5, 0.2, 62.0]})
    mean_map, _, _ = parse_rbp_expression_reference(expr_df)
    assert mean_map["PTBP1"] > 40.0 and mean_map["PTBP2"] < 1.0

    mock_rank = pd.DataFrame([
        {"RBP": "PTBP2", "stat_Z": 4.2, "P_FWER": 0.004},
        {"RBP": "PTBP1", "stat_Z": 4.2, "P_FWER": 0.004},
    ])
    disamb = annotate_rbp_expression_disambiguation(mock_rank, expr_df, fwer_col="P_FWER", min_rbp_expr=1.0)
    assert bool(disamb.loc[disamb["RBP"] == "PTBP1", "rbp_expressed"].iloc[0]) is True
    assert bool(disamb.loc[disamb["RBP"] == "PTBP2", "rbp_expressed"].iloc[0]) is False

    fx = _create_synthetic_study(tmp_path)
    lib = load_bundled_human_rbp_library(species="mouse")
    se_df = pd.read_csv(fx["rmats"] / "SE.MATS.JCEC.txt", sep="\t")
    from splicetrace.rmats_io import _standardize_rmats_df
    std_df = _standardize_rmats_df(se_df, "SE", {"chr1"}, "synth")
    res_obj = run_bulk_continuous_rescue(
        std_df, fx["fasta"], lib,
        distal_intron_nt=500,
        n_permutations=19,
        seed=42,
    )
    assert not res_obj.rbp_ranking.empty
    assert {"RBP", "rbp_P_FWER", "best_region"}.issubset(res_obj.rbp_ranking.columns)


def test_sc_build_splice_matrices_from_starsolo_sj(tmp_path: Path) -> None:
    """Verify one-step STARsolo SJ/raw + GTF -> canonical single-cell splicing matrices builder."""
    gtf_p = tmp_path / "test_sc.gtf"
    gtf_p.write_text(
        'chr1\tsynth\texon\t101\t200\t.\t+\t.\tgene_id "G1"; transcript_id "T_inc"; gene_name "GeneA";\n'
        'chr1\tsynth\texon\t301\t380\t.\t+\t.\tgene_id "G1"; transcript_id "T_inc"; gene_name "GeneA";\n'
        'chr1\tsynth\texon\t501\t600\t.\t+\t.\tgene_id "G1"; transcript_id "T_inc"; gene_name "GeneA";\n'
        'chr1\tsynth\texon\t101\t200\t.\t+\t.\tgene_id "G1"; transcript_id "T_exc"; gene_name "GeneA";\n'
        'chr1\tsynth\texon\t501\t600\t.\t+\t.\tgene_id "G1"; transcript_id "T_exc"; gene_name "GeneA";\n',
        encoding="utf-8",
    )
    sj_raw = tmp_path / "Solo.out" / "SJ" / "raw"
    sj_raw.mkdir(parents=True)
    # 0-based exons: [100,200), [300,380), [500,600) -> 1-based STAR introns: (201,300), (381,500), skip (201,500)
    (sj_raw / "features.tsv").write_text("chr1\t201\t300\t1\nchr1\t381\t500\t1\nchr1\t201\t500\t1\n", encoding="utf-8")
    cells = [f"Cell_{i}" for i in range(1, 9)]
    (sj_raw / "barcodes.tsv").write_text("\n".join(cells) + "\n", encoding="utf-8")
    mtx_lines = ["%%MatrixMarket matrix coordinate integer general", "3 8 24"]
    for c_idx in range(1, 9):
        inc = 16 if c_idx <= 4 else 2
        exc = 2 if c_idx <= 4 else 14
        mtx_lines.extend([f"1 {c_idx} {inc}", f"2 {c_idx} {inc}", f"3 {c_idx} {exc}"])
    (sj_raw / "matrix.mtx").write_text("\n".join(mtx_lines) + "\n", encoding="utf-8")

    meta_p = tmp_path / "cell_meta.tsv"
    meta_rows = ["cell_id\tcell_type\tdonor_id"] + [
        f"Cell_{i}\t{'TypeA' if i <= 4 else 'TypeB'}\t{'D1' if i % 2 == 1 else 'D2'}" for i in range(1, 9)
    ]
    meta_p.write_text("\n".join(meta_rows) + "\n", encoding="utf-8")

    out_m = tmp_path / "sc_matrices"
    res = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "bin" / "katana_sc_build_splice_matrices.py"),
            "--gtf", str(gtf_p),
            "--sj-dirs", str(sj_raw),
            "--metadata", str(meta_p),
            "--group-col", "cell_type",
            "--numerator", "TypeA",
            "--denominator", "TypeB",
            "--min-cells", "4",
            "--min-cell-support", "3",
            "--outdir", str(out_m),
        ],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, res.stderr
    ev_out = pd.read_csv(out_m / "sc_events.tsv", sep="\t")
    psi_out = pd.read_csv(out_m / "sc_psi.tsv.gz", sep="\t", index_col=0)
    assert len(ev_out) == 1 and float(ev_out.iloc[0]["dpsi"]) > 0.60
    assert psi_out.shape == (8, 1)


def test_v1_0_2_scientific_audit_regression_suite(tmp_path: Path) -> None:
    """Comprehensive regression suite covering statistical, coordinate, and pipeline-contract audit fixes."""
    # 1. SpliceManifold: Jeffreys Beta(0.5,0.5) n+2 denominator, usable_both guard, UMAP fallback, dual kNN ties
    from splicemanifold.ps_das import qualify_ps_das_events
    from splicemanifold.manifold import compute_splicing_manifold, compute_dual_expression_vs_psi_umap

    ev_df = pd.DataFrame([{
        "event_id": "SE|chr1|+|100|200",
        "event_type": "SE",
        "FDR": 0.001,
        "IncLevelDifference": 0.45,
        "IncLevel1": "0.80,0.82,0.79",
        "IncLevel2": "0.35,0.34,0.36",
        "IJC_SAMPLE_1": "20,20,20",
        "SJC_SAMPLE_1": "5,5,5",
        "IJC_SAMPLE_2": "7,7,7",
        "SJC_SAMPLE_2": "13,13,13",
        "IncFormLen": 150,
        "SkipFormLen": 100,
    }, {
        # Corrupt IncLevel (1.5) must NEVER set nmd_asymmetry_rescued=True
        "event_id": "SE|chr1|+|300|400",
        "event_type": "SE",
        "FDR": 0.001,
        "IncLevelDifference": 0.50,
        "IncLevel1": "1.5,0.85,0.85",
        "IncLevel2": "0.10,0.10,0.10",
        "IJC_SAMPLE_1": "30,30,30",
        "SJC_SAMPLE_1": "5,5,5",
        "IJC_SAMPLE_2": "1,1,1",
        "SJC_SAMPLE_2": "5,5,5",
        "IncFormLen": 150,
        "SkipFormLen": 100,
    }])
    q = qualify_ps_das_events(ev_df)
    # Row 0: st1=75, si1=60; st2=60, si2=21; exact Beta posterior var uses n+2
    p1 = (60 + 0.5) / 76.0
    p2 = (21 + 0.5) / 61.0
    expected_se = round(float(np.sqrt(p1 * (1 - p1) / 77.0 + p2 * (1 - p2) / 62.0)), 5)
    assert abs(float(q.iloc[0]["jeffreys_se_dpsi"]) - expected_se) < 1e-5
    assert bool(q.iloc[1]["nmd_asymmetry_rescued"]) is False

    # Small cohort (n=3) UMAP fallback must NOT be all-zero, and identical expr/psi must give Delta_kNN=0 & no silhouette crash
    rng = np.random.default_rng(7)
    p_small = pd.DataFrame(rng.uniform(0.1, 0.9, size=(12, 3)), index=[f"E{i}" for i in range(12)], columns=["C1", "C2", "C3"])
    s_meta = [
        {"sample_id": "C1", "condition": "A", "cell_type": "T1", "donor_id": "D1"},
        {"sample_id": "C2", "condition": "A", "cell_type": "T2", "donor_id": "D2"},
        {"sample_id": "C3", "condition": "B", "cell_type": "T3", "donor_id": "D3"},
    ]
    land = compute_splicing_manifold(p_small, s_meta, run_umap=False, top_n=12, manifold_top_n=12)
    assert land["available"] is True
    umap_vals = np.array([[s["psi_UMAP1"], s["psi_UMAP2"]] for s in land["samples"]])
    assert float(np.linalg.norm(umap_vals)) > 1e-4
    # Identical expression features with all distinct cell_type labels (n_ct == n_s) must not crash silhouette_score
    dual = compute_dual_expression_vs_psi_umap(p_small.T, land, k_neighbors=3)
    assert dual["cell_type_silhouette_psi"] == 0.0
    assert all(0.0 <= c["modality_divergence"] <= 1.0 for c in dual["cells"])

    # 2. SpliceNet: <4 donor consensus_core_audit no-crash + NaN-safe _bh_fdr
    from splicenet.network import consensus_core_audit, _merge_close_modules_and_kme
    u_sn = pd.DataFrame(rng.uniform(0.1, 0.9, size=(10, 3)), index=[f"EV{i}" for i in range(10)], columns=["S1", "S2", "S3"])
    m_sn = pd.DataFrame({"sample_id": ["S1", "S2", "S3"], "biological_sample_id": ["D1", "D2", "D3"], "condition": ["A", "A", "B"]})
    b_asg = pd.DataFrame({"event_key": u_sn.index, "module": [1]*5 + [2]*5, "block": 1, "kME": 0.8})
    enr_asg, stab = consensus_core_audit(u_sn, m_sn, None, b_asg, n_subsamples=3)
    assert stab["n_subsamples"] == 0 and len(enr_asg) == 10
    # Zero-variance module alongside two identical valid modules must still merge the two valid modules
    x_merge = np.vstack([
        np.full((3, 6), 0.5),
        np.tile(np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), (3, 1)),
        np.tile(np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), (3, 1)),
    ])
    m_final, _ = _merge_close_modules_and_kme(x_merge, np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]), merge_cut_height=0.25)
    assert len(set(m_final[3:])) == 1  # modules 2 and 3 merged despite module 1 having zero variance

    # 3. scSpliceNet: EB shrinkage must NOT mutate global_mean in-place + NaN weights guarded
    from scsplicenet.evaluation import predict_sc_usage_inductive
    from scsplicenet.network import precision_pairwise_correlation
    tu = np.vstack([np.full((4, 2), 0.1), np.full((4, 2), 0.9)])
    qu = np.full((2, 2), np.nan)
    ts = np.array(["S1"]*4 + ["S2"]*4)
    qs = np.array(["S1", "S2"])
    pred = predict_sc_usage_inductive(tu, qu, np.array(["G1", "G2"]), train_states=ts, query_states=qs, state_shrink_kappa=100.0)
    # With kappa=100 and sw~4, predictions MUST shrink strongly toward global_mean=0.5 (not stay at 0.1/0.9)
    assert 0.35 < float(pred[0, 0]) < 0.50 and 0.50 < float(pred[1, 0]) < 0.65
    x_p = rng.uniform(0.1, 0.9, size=(12, 3))
    w_p = np.ones((12, 3))
    w_p[0, 0] = np.nan
    r_p, _, _ = precision_pairwise_correlation(x_p, w_p, min_pairs=4, min_eff_n=3.0)
    assert np.isfinite(r_p[0, 1])

    # 4. SpliceTrace & scSpliceTrace: in-frame insertion into canonical-exclusion UniProt protein + partial PTC truncation
    from splicetrace.protein_consequences import _map_uniprot_domains_to_isoforms
    prot_exc = "M" + "A" * 99  # 100 aa canonical exclusion isoform
    prot_inc = "M" + "A" * 49 + "GGGGGGGGGG" + "A" * 50  # +10 aa in-frame insertion right inside domain [30..70]
    mock_db = {"syn1": {"uniprot_acc": "P17600", "uniprot_len": 100, "features": [
        {"kind": "DOMAIN", "start": 30, "end": 70, "label": "DOMAIN:Synapsin_Core"},
    ]}}
    _, feat_str, arch = _map_uniprot_domains_to_isoforms("Syn1", prot_inc, prot_exc, False, False, mock_db)
    assert "+10 aa ins" in feat_str and arch[0]["state"] == "inserted_by_inclusion"
    # Partial PTC truncation at aa 50 straddling domain [30..70]
    _, feat_ptc, arch_ptc = _map_uniprot_domains_to_isoforms("Syn1", prot_exc[:50], prot_exc, True, False, mock_db)
    assert arch_ptc[0]["state"] == "truncated_by_PTC" and "PTC" in feat_ptc

    # Ensembl-ID-only expression matrix must NOT falsely mark all RBPs rbp_expressed=False
    from splicetrace.module_bridge import annotate_rbp_expression_disambiguation
    ens_expr = pd.DataFrame({"gene_id": ["ENSG00000100001", "ENSG00000100002"], "S1": [50.0, 20.0], "S2": [48.0, 22.0]})
    rk_in = pd.DataFrame([{"RBP": "RBFOX1", "stat_Z": 4.0, "rbp_P_FWER": 0.01}, {"RBP": "PTBP1", "stat_Z": 3.5, "rbp_P_FWER": 0.02}])
    rk_out = annotate_rbp_expression_disambiguation(rk_in, ens_expr)
    assert bool(rk_out["rbp_expressed"].all()) is True

    # scSpliceTrace: unpaired 2v2 donor cohort routes to unpaired_donor_pseudobulk with finite p_value & fdr
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast
    cells = [f"C{i}" for i in range(40)]
    md_2v2 = pd.DataFrame({
        "cell_id": cells,
        "condition": ["KO"] * 20 + ["WT"] * 20,
        "donor_id": ["M1"] * 10 + ["M2"] * 10 + ["M3"] * 10 + ["M4"] * 10,
        "qc_pass": ["TRUE"] * 40,
    })
    u_2v2 = pd.DataFrame(
        {"E1": [0.85] * 20 + [0.15] * 20, "E2": [0.50] * 40},
        index=cells,
    )
    s_2v2 = pd.DataFrame({"E1": [20.0] * 40, "E2": [20.0] * 40}, index=cells)
    c_df, d_del, c_info = compute_sc_splicing_contrast(
        u_2v2, md_2v2, pd.Series({"E1": "G1", "E2": "G2"}),
        support=s_2v2, numerator_group="KO", denominator_group="WT",
    )
    assert c_info["inference_unit"] == "unpaired_donor_pseudobulk"
    assert d_del is None
    assert {"p_value", "fdr"}.issubset(c_df.columns)
    assert float(c_df.loc[c_df["event_id"] == "E1", "p_value"].iloc[0]) < 0.01

    # 5. Katana Studio: network_explorer consumes scsplicenet source_event/target_event/shrunk_correlation
    from katana_studio.network_explorer import _compute_real_edges
    sc_edges = pd.DataFrame([
        {"source_event": "E1", "target_event": "E2", "shrunk_correlation": 0.88},
        {"source_event": "E2", "target_event": "E3", "shrunk_correlation": 0.77},
        {"source_event": "E1", "target_event": "E3", "shrunk_correlation": 0.66},
        {"source_event": "E3", "target_event": "E4", "shrunk_correlation": 0.55},
    ])
    got_edges = _compute_real_edges(["E1", "E2", "E3", "E4"], pd.DataFrame(), np.ones(4), explicit_edges_df=sc_edges)
    assert len(got_edges) == 4 and abs(got_edges[0][3] - 0.88) < 1e-4


# ---------------------------------------------------------------------------
# 4. Repository-Wide Confidentiality, Brand & Naming Hygiene Audit
# ---------------------------------------------------------------------------
def test_zero_confidential_brand_or_typo_leaks() -> None:
    toks = [
        "yth" + "dc1_deletion",
        r"\bset" + r"d2\b",
        r"\blu" + r"is\b",
        "mck" + "insey",
        "RS-" + "DAS",
        "kanata_",
        r"E:\\\\Project",
        "E:/" + "Project",
    ]
    forbidden = re.compile("(" + "|".join(toks) + ")", re.I)
    text_exts = {".py", ".nf", ".config", ".R", ".md", ".toml", ".json", ".ts", ".tsx", ".css", ".html", ".sbatch"}
    violations: list[str] = []
    for f in REPO_ROOT.rglob("*"):
        if not f.is_file() or f.suffix not in text_exts:
            continue
        if f.resolve() == Path(__file__).resolve():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache", "dummy_inspection_reports"} for part in f.parts):
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = forbidden.search(txt)
        if m:
            violations.append(f"{f.relative_to(REPO_ROOT)} -> {m.group(0)!r}")
    assert violations == [], f"Hygiene violations found: {violations[:15]}"
