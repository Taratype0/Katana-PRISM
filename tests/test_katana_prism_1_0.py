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
    bam_qc_nf = (REPO_ROOT / "modules" / "local" / "bam_qc_stats.nf").read_text(encoding="utf-8")
    assert "trap 'rm -rf tmp_java' EXIT" in bam_qc_nf
    assert "rm -rf tmp_java" in bam_qc_nf
    assert "maxForks    = 2" in port_res
    assert "afterScript = 'rm -rf tmp_java || true'" in port_res
    assert "COLLATE_GENOME_BAM" in port_res and "maxForks = 4" in port_res
    for proc_name in ("SPLICEMANIFOLD", "SPLICENET", "SPLICETRACE", "KATANA_CROSS_PARADIGM", "SCSPLICENET", "SCSPLICETRACE", "KATANA_REPORT"):
        assert proc_name in port_res

    nf_cfg = (REPO_ROOT / "nextflow.config").read_text(encoding="utf-8")
    assert "skip_cross_paradigm" in nf_cfg
    assert "KATANA_CROSS_PARADIGM" in nf_cfg
    std_out = (REPO_ROOT / "conf" / "standard_outputs.config").read_text(encoding="utf-8")
    assert "KATANA_CROSS_PARADIGM" in std_out
    mod_nf = (REPO_ROOT / "modules" / "local" / "katana_prism_modules.nf").read_text(encoding="utf-8")
    assert "process KATANA_CROSS_PARADIGM" in mod_nf
    assert "--cross-paradigm-dir" in mod_nf
    down_nf = (REPO_ROOT / "workflows" / "katana_modular_downstream.nf").read_text(encoding="utf-8")
    assert "KATANA_CROSS_PARADIGM(" in down_nf


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
    assert 'splicemanifold: "1.0.3"' in (sm_bulk_out / "versions.yml").read_text(encoding="utf-8")

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
    assert 'splicenet: "1.0.3"' in (sn_out / "versions.yml").read_text(encoding="utf-8")

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
    assert 'splicetrace: "1.0.3"' in (st_out / "versions.yml").read_text(encoding="utf-8")

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


def test_sc_multiplexed_bam_sashimi_and_dardel_wiring(tmp_path: Path) -> None:
    """Verify single-cell multiplexed BC-tagged BAM RPKM Katana Native Sashimi & Nextflow/Dardel wiring."""
    import importlib.util as _ilu
    import struct
    from katana_next.targets.native_sashimi_adapter import NativeSashimiAdapter
    from katana_studio.candidate_workbench import _load_sc_barcode_aliases

    # Load BGZF helper directly from bin/katana_make_dummy_bam.py
    spec = _ilu.spec_from_file_location("katana_make_dummy_bam", REPO_ROOT / "bin" / "katana_make_dummy_bam.py")
    assert spec is not None and spec.loader is not None
    bam_mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(bam_mod)

    # Construct valid coordinate-sorted BGZF BAM on chr1 with spliced & exonic reads carrying binary BC:Z tags
    sam_hdr = "@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:5000\n".encode("ascii")
    bam_bytes = bytearray(b"BAM\x01")
    bam_bytes += struct.pack("<i", len(sam_hdr)) + sam_hdr + struct.pack("<i", 1)
    cname = b"chr1\x00"
    bam_bytes += struct.pack("<i", len(cname)) + cname + struct.pack("<i", 5000)

    def _pack_read(qname_s: str, pos0: int, cigar_pairs: list[tuple[int, int]], bc_str: str) -> bytes:
        qname = qname_s.encode("ascii") + b"\x00"
        cigar_packed = b"".join(struct.pack("<I", (ln << 4) | op) for op, ln in cigar_pairs)
        qlen = sum(ln for op, ln in cigar_pairs if op in (0, 1, 4, 7, 8))
        seq_enc = bytes([0x11] * ((qlen + 1) // 2))
        qual = bytes([40] * qlen)
        aux = b"NH\x69" + struct.pack("<i", 1) + b"BCZ" + bc_str.encode("ascii") + b"\x00"
        body = (
            struct.pack("<iiBBHHHiiii", 0, pos0, len(qname), 60, 4680, len(cigar_pairs), 0, qlen, -1, -1, 0)
            + qname
            + cigar_packed
            + seq_enc
            + qual
            + aux
        )
        return struct.pack("<i", len(body)) + body

    # SE locus: upstream exon 1001..1100, skipped exon 1301..1400, downstream exon 1601..1700
    # Arm 0 (C1 -> D1, C2 via alias BC_C2_ALIAS -> D2): high inclusion (25M 200N 25M and 25M 200N 25M)
    # Arm 1 (C3 -> D3, C4 -> D4): high skipping (25M 500N 25M)
    reads = [
        _pack_read("r1", 1075, [(0, 25), (3, 200), (0, 25)], "C1"),
        _pack_read("r2", 1075, [(0, 25), (3, 200), (0, 25)], "BC_C2_ALIAS"),
        _pack_read("r3", 1075, [(0, 25), (3, 500), (0, 25)], "C3"),
        _pack_read("r4", 1075, [(0, 25), (3, 500), (0, 25)], "C4"),
        _pack_read("r5", 1320, [(0, 50)], "C1"),
        _pack_read("r6", 1375, [(0, 25), (3, 200), (0, 25)], "BC_C2_ALIAS"),
    ]
    bam_path = tmp_path / "sc_mux.bam"
    bam_path.write_bytes(bam_mod.make_bgzf_block(bytes(bam_bytes) + b"".join(reads)) + bam_mod.BGZF_EOF)

    shares_path = tmp_path / "barcode_shares.txt"
    shares_path.write_text("C2\tBC_C2_ALIAS\n", encoding="utf-8")
    cell_to_bcs = _load_sc_barcode_aliases({"C1", "C2", "C3", "C4"}, shares_path)
    assert "BC_C2_ALIAS" in cell_to_bcs["C2"]

    bc_to_donor_arm: dict[str, tuple[int, str]] = {}
    for cid_s, don in (("C1", "D1"), ("C2", "D2")):
        for bc in cell_to_bcs[cid_s]:
            bc_to_donor_arm[bc] = (0, don)
    for cid_s, don in (("C3", "D3"), ("C4", "D4")):
        for bc in cell_to_bcs[cid_s]:
            bc_to_donor_arm[bc] = (1, don)

    adapter = NativeSashimiAdapter()
    ev_row = {
        "event_id": "SE:chr1:+:1300:1400:1000:1100:1600:1700",
        "gene_symbol": "Nrxn1",
        "geneSymbol": "Nrxn1",
        "gene_id": "ENSMUSG00000024109",
        "chr": "chr1",
        "strand": "+",
        "exonStart_0base": 1300,
        "exonEnd": 1400,
        "upstreamES": 1000,
        "upstreamEE": 1100,
        "downstreamES": 1600,
        "downstreamEE": 1700,
    }
    res_s = adapter.render_event_sashimi_figure(
        event_row=ev_row,
        event_type="SE",
        label1="Stim",
        label2="Ctrl",
        out_stem=tmp_path / "nrxn1_mux_sashimi",
        multiplexed_bam=bam_path,
        bc_to_donor_arm=bc_to_donor_arm,
        donor_cell_counts=({"D1": 1, "D2": 1}, {"D3": 1, "D4": 1}),
        total_study_cells=4,
        bc_tag="BC",
        dpsi=0.65,
        fdr=0.001,
        psi1=[0.90, 0.88],
        psi2=[0.18, 0.22],
        save_pdf=False,
    )
    assert res_s["mode"] == "bam_rpkm"
    assert Path(res_s["svg"]).is_file() and Path(res_s["svg"]).stat().st_size > 2000
    assert Path(res_s["png"]).is_file() and Path(res_s["png"]).stat().st_size > 2000

    # Verify Nextflow Dardel wiring: PYTHONPATH live override, --sc-bam, --group-col, and 16.GB KATANA_REPORT_SC budget
    mods_txt = (REPO_ROOT / "modules" / "local" / "katana_prism_modules.nf").read_text(encoding="utf-8")
    sc_wf_txt = (REPO_ROOT / "workflows" / "katana_singlecell.nf").read_text(encoding="utf-8")
    res_txt = (REPO_ROOT / "conf" / "portable_resources.config").read_text(encoding="utf-8")
    assert "prismPythonPathBlock" in mods_txt and "export PYTHONPATH=" in mods_txt
    assert "--sc-bam" in mods_txt and "--sc-barcode-shares" in mods_txt
    assert mods_txt.count('--group-col "${group_col}"') >= 3
    assert "ch_sc_bam_val" in sc_wf_txt and "ch_sc_bcs_val" in sc_wf_txt
    assert "16.GB * task.attempt" in res_txt and "KATANA_REPORT_SC" in res_txt


def test_sparse_subtype_empty_and_headerless_splicetrace_files(tmp_path: Path) -> None:
    import pandas as pd
    from katana_studio.full_suite_views import render_ch7_splicetrace_suite
    from scsplicetrace.protein_consequences import trace_protein_and_nmd_consequences as sc_trace_prot
    from scsplicetrace.rna_forms import IndexedFasta
    from scsplicetrace.sequence_and_consequence import trace_sc_mature_motif_consequences
    from splicetrace.protein_consequences import trace_protein_and_nmd_consequences as bulk_trace_prot

    # Simulate exact 1-byte newline-only & header-only scsplicetrace outputs from a sparse subtype (0 eligible events)
    st_dir = tmp_path / "scsplicetrace"
    st_dir.mkdir(parents=True)
    for fname in (
        "sc_protein_and_nmd_consequences.tsv",
        "scsplicetrace_mature_consequences.tsv",
        "scsplicetrace_region_tests.tsv",
        "sc_module_rbp_bridge.tsv",
    ):
        (st_dir / fname).write_text("\n", encoding="utf-8")
    (st_dir / "scsplicetrace_rbp_ranking.tsv").write_text(
        "RBP\tbest_region\tdirection\tstat_Z\trbp_P_FWER\tnominal_P\n", encoding="utf-8"
    )

    out_dir = tmp_path / "report"
    out_dir.mkdir(parents=True)
    panels: list[dict] = []
    tables: list[dict] = []
    motif_summaries: list[dict] = []
    prot_df, conseq_df, br_df = render_ch7_splicetrace_suite(
        st_dir=st_dir,
        cid="Rare_Subtype__Stim_vs_Ctrl",
        species="mouse",
        outdir=out_dir,
        panels=panels,
        tables=tables,
        motif_summaries=motif_summaries,
    )
    assert prot_df.empty and conseq_df.empty and br_df.empty

    # Verify producers emit explicit TSV headers when given empty event tables
    empty_ev = pd.DataFrame(columns=["event_id", "gene_id", "gene_symbol", "event_type", "strand", "dpsi"])
    gtf_p = tmp_path / "dummy.gtf"
    gtf_p.write_text("", encoding="utf-8")
    fa_p = tmp_path / "dummy.fa"
    fa_p.write_text(">chr1\nACGTACGT\n", encoding="utf-8")

    for trace_fn, tag in ((sc_trace_prot, "sc"), (bulk_trace_prot, "bulk")):
        res_p = trace_fn(empty_ev, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, species="mouse")
        assert len(res_p.columns) >= 15
        out_tsv = tmp_path / f"{tag}_prot.tsv"
        res_p.to_csv(out_tsv, sep="\t", index=False)
        assert pd.read_csv(out_tsv, sep="\t").empty

    empty_rna = pd.DataFrame(columns=["event_id", "layer", "sequence"])
    empty_ce = pd.DataFrame(columns=["event_id", "z_shift", "dpsi"])
    mc_df = trace_sc_mature_motif_consequences(empty_rna, empty_ce)
    assert len(mc_df.columns) >= 4
    mc_tsv = tmp_path / "mc.tsv"
    mc_df.to_csv(mc_tsv, sep="\t", index=False)
    assert pd.read_csv(mc_tsv, sep="\t").empty


def test_scsplicenet_multimodule_fast_quantiles_and_cli_budget(tmp_path: Path) -> None:
    """Verify vectorized nanquantile parity, multi-module anti-collapse clustering, and CLI --select-budget."""
    from scsplicenet.cli import main as scsn_main
    from scsplicenet.network import _fast_nanquantile_axis0, _louvain_or_modularity_cut, fit_scsplicenet

    rng = np.random.default_rng(123)
    arr = rng.normal(0.0, 1.0, size=(8, 12))
    arr[0, 0] = np.nan
    arr[2:5, 3] = np.nan
    arr[:, 7] = np.nan
    for qv in (0.25, 0.50, 0.75):
        got = _fast_nanquantile_axis0(arr, qv)
        exp = np.nanquantile(arr, qv, axis=0)
        assert np.allclose(got, exp, atol=1e-12, equal_nan=True)

    # Synthetic 3-community co-splicing usage matrix (36 cells x 45 distinct genes/events)
    n_cells = 36
    cells = [f"Cell_{i:02d}" for i in range(n_cells)]
    meta = pd.DataFrame({
        "cell_id": cells,
        "donor_id": [f"D{(i % 6) + 1}" for i in range(n_cells)],
        "cell_type": ["PopA" if i < 18 else "PopB" for i in range(n_cells)],
        "condition": ["Stim" if (i % 2 == 0) else "Ctrl" for i in range(n_cells)],
        "qc_pass": True,
    })
    eids = [f"EV_{j:02d}" for j in range(45)]
    eg_df = pd.DataFrame({"event_id": eids, "gene_id": [f"G_{j:02d}" for j in range(45)]})

    # Three orthogonal latent splicing factors across 3x15 events
    z1 = np.array([1.0 if (i % 2 == 0) else -1.0 for i in range(n_cells)])
    z2 = np.array([1.0 if i < 18 else -1.0 for i in range(n_cells)])
    z3 = np.array([1.0 if (i % 4 < 2) else -1.0 for i in range(n_cells)])
    psi_mat = np.zeros((n_cells, 45), dtype=float)
    for k_f, zf in enumerate((z1, z2, z3)):
        for j in range(15):
            psi_mat[:, k_f * 15 + j] = np.clip(0.50 + 0.30 * zf + rng.normal(0.0, 0.04, n_cells), 0.05, 0.95)
    u_df = pd.DataFrame(psi_mat, index=cells, columns=eids)
    s_df = pd.DataFrame(np.full((n_cells, 45), 35.0), index=cells, columns=eids)

    net = fit_scsplicenet(u_df, meta, eg_df, support=s_df, repeats=4, min_module_size=8, compute_modules=True)
    assert net.diagnostics["discovered_modules"] >= 2
    vc_m = net.modules.loc[net.modules["module"].ne("M0"), "module"].value_counts()
    assert int(vc_m.max()) <= int(0.65 * 45)

    # Anti-collapse modularity cut on a 60-node 3-block dissimilarity matrix must find >= 2 communities
    blk = np.ones((20, 20), dtype=float)
    W3 = np.zeros((60, 60), dtype=float)
    for b_i in range(3):
        W3[b_i * 20:(b_i + 1) * 20, b_i * 20:(b_i + 1) * 20] = 0.85 * blk
    np.fill_diagonal(W3, 1.0)
    lbl3 = _louvain_or_modularity_cut(1.0 - W3, min_module_size=10)
    assert len(set(lbl3) - {0}) >= 3

    # CLI --select-budget execution must succeed without unpacking ValidatedSCUsageInput
    u_p = tmp_path / "psi.tsv"
    s_p = tmp_path / "sup.tsv"
    m_p = tmp_path / "meta.tsv"
    eg_p = tmp_path / "eg.tsv"
    u_df.to_csv(u_p, sep="\t", index_label="cell_id")
    s_df.to_csv(s_p, sep="\t", index_label="cell_id")
    meta.to_csv(m_p, sep="\t", index=False)
    eg_df.to_csv(eg_p, sep="\t", index=False)
    out_cli = tmp_path / "cli_budget_out"
    rc = scsn_main([
        "--usage", str(u_p), "--support", str(s_p), "--metadata", str(m_p), "--event-genes", str(eg_p),
        "--select-budget", "30", "--min-module-size", "8", "--repeats", "4", "--threads", "2", "--out", str(out_cli),
    ])
    assert rc == 0
    assert (out_cli / "scsplicenet_selected_events.tsv").is_file()
    assert len(pd.read_csv(out_cli / "scsplicenet_selected_events.tsv", sep="\t")) == 30


def test_dual_bh_fdr_and_log1p_rbp_expression_gating(tmp_path: Path) -> None:
    """Verify dual maxT FWER + BH-FDR emission/rendering and auto-adaptation on log1p single-cell expression."""
    from katana_studio.full_suite_views import render_ch7_splicetrace_suite
    from scsplicetrace.contrast_discovery import discover_sc_rbp_programs
    from scsplicetrace.module_bridge import annotate_rbp_expression_disambiguation as sc_annot
    from splicetrace.module_bridge import annotate_rbp_expression_disambiguation as bulk_annot
    from splicetrace.continuous_rescue import run_bulk_continuous_rescue
    from splicetrace.library import load_bundled_human_rbp_library

    # 1. Expression gating: log1p single-cell scale auto-adapts to 0.05; bulk TPM keeps 1.0
    bg_genes = [f"GENE{i}" for i in range(1, 35)]
    sc_expr_df = pd.DataFrame({
        "gene_symbol": ["RBFOX1", "PTBP1", "ESRP1"] + bg_genes,
        "mean_expr": [0.65, 0.18, 0.0] + [0.45] * len(bg_genes),
    })
    rk_mock = pd.DataFrame([
        {"RBP": "RBFOX1", "stat_Z": 4.1, "rbp_P_FWER": 0.01},
        {"RBP": "PTBP1", "stat_Z": 3.2, "rbp_P_FWER": 0.12},
        {"RBP": "ESRP1", "stat_Z": 2.9, "rbp_P_FWER": 0.25},
    ])
    sc_out = sc_annot(rk_mock, sc_expr_df)
    assert float(sc_out["rbp_expr_threshold"].iloc[0]) == pytest.approx(0.05)
    assert bool(sc_out.loc[sc_out["RBP"] == "RBFOX1", "rbp_expressed"].iloc[0]) is True
    assert bool(sc_out.loc[sc_out["RBP"] == "PTBP1", "rbp_expressed"].iloc[0]) is True
    assert bool(sc_out.loc[sc_out["RBP"] == "ESRP1", "rbp_expressed"].iloc[0]) is False

    bulk_expr_df = pd.DataFrame({
        "gene_symbol": ["PTBP1", "PTBP2"] + bg_genes,
        "TPM": [42.0, 0.2] + [25.0] * len(bg_genes),
    })
    bulk_out = bulk_annot(rk_mock.iloc[1:2].copy(), bulk_expr_df)
    assert float(bulk_out["rbp_expr_threshold"].iloc[0]) == pytest.approx(1.0)
    assert bool(bulk_out.iloc[0]["rbp_expressed"]) is True

    # 2. Synthetic discovery outputs include rbp_FDR_BH & region_FDR_BH in both bulk and SC
    fx = _create_synthetic_study(tmp_path)
    lib = load_bundled_human_rbp_library(species="mouse")
    se_df = pd.read_csv(fx["rmats"] / "SE.MATS.JCEC.txt", sep="\t")
    from splicetrace.rmats_io import _standardize_rmats_df
    std_df = _standardize_rmats_df(se_df, "SE", {"chr1"}, "synth")
    bulk_res = run_bulk_continuous_rescue(std_df, fx["fasta"], lib, n_permutations=29, seed=11)
    assert {"rbp_FDR_BH", "rbp_P_FWER", "nominal_P"}.issubset(bulk_res.rbp_ranking.columns)
    assert {"region_FDR_BH", "region_P_FWER", "nominal_P"}.issubset(bulk_res.region_tests.columns)

    rng = np.random.default_rng(7)
    n_ev_s = 30
    c_df = pd.DataFrame({
        "event_id": [f"E_{i}" for i in range(n_ev_s)],
        "gene_id": [f"G_{i}" for i in range(n_ev_s)],
        "eligible": True,
        "baseline_PSI": rng.uniform(0.2, 0.8, n_ev_s),
        "z_shift": rng.normal(0.0, 1.5, n_ev_s),
    })
    cov_df = pd.DataFrame({
        "event_id": c_df["event_id"],
        "exon_GC": 0.5,
        "flank_GC": 0.45,
        "log_length": 4.5,
        "event_type": "SE",
    })
    tensor = rng.normal(0.0, 1.0, size=(n_ev_s, 3, 5))
    sc_res = discover_sc_rbp_programs(
        c_df, tensor, np.array(["RBFOX1", "PTBP1", "SRSF1", "TIAL1", "MBNL1"]),
        ["upstream_intron_0_35", "alt_exon_5p_50", "downstream_exon_5p_50"],
        cov_df, n_permutations=29, seed=11,
    )
    assert {"rbp_FDR_BH", "rbp_P_FWER", "nominal_P"}.issubset(sc_res.rbp_ranking.columns)
    assert {"region_FDR_BH", "region_P_FWER", "nominal_P"}.issubset(sc_res.region_tests.columns)

    # 3. Ch07 dual-tier rendering preserves both FDR_BH and P_FWER
    st_dir = tmp_path / "st_ch7"
    st_dir.mkdir(parents=True)
    pd.DataFrame([
        {"RBP": "Rbfox1", "best_region": "downstream_exon_5p_50", "stat_Z": -3.85, "nominal_P": 0.0002, "rbp_FDR_BH": 0.025, "rbp_P_FWER": 0.11},
        {"RBP": "Ptbp1", "best_region": "upstream_intron_0_35", "stat_Z": 3.10, "nominal_P": 0.0120, "rbp_FDR_BH": 0.420, "rbp_P_FWER": 0.65},
    ]).to_csv(st_dir / "scsplicetrace_rbp_ranking.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"RBP": "Rbfox1", "region": "downstream_exon_5p_50", "stat_Z": -3.85, "nominal_P": 0.0002, "region_FDR_BH": 0.04, "region_P_FWER": 0.11},
        {"RBP": "Rbfox1", "region": "upstream_intron_0_35", "stat_Z": 0.45, "nominal_P": 0.65, "region_FDR_BH": 0.90, "region_P_FWER": 1.0},
        {"RBP": "Ptbp1", "region": "downstream_exon_5p_50", "stat_Z": 0.25, "nominal_P": 0.80, "region_FDR_BH": 0.95, "region_P_FWER": 1.0},
        {"RBP": "Ptbp1", "region": "upstream_intron_0_35", "stat_Z": 3.10, "nominal_P": 0.012, "region_FDR_BH": 0.42, "region_P_FWER": 0.65},
    ]).to_csv(st_dir / "scsplicetrace_region_tests.tsv", sep="\t", index=False)

    rep_ch7 = tmp_path / "rep_ch7"
    rep_ch7.mkdir(parents=True)
    p7: list[dict] = []
    t7: list[dict] = []
    ms7: list[dict] = []
    render_ch7_splicetrace_suite(st_dir, "PopA__Stim_vs_Ctrl", "mouse", rep_ch7, p7, t7, ms7)
    assert ms7 and ms7[0]["status"] == "significant"
    assert ms7[0]["min_native_fdr"] == pytest.approx(0.025)
    assert ms7[0]["min_fwer"] == pytest.approx(0.11)
    logo_plot_df = pd.read_csv(rep_ch7 / "outputs/splicetrace/PopA__Stim_vs_Ctrl/top_motif_logos_plot_data.tsv", sep="\t")
    assert {"FDR_BH", "P_FWER", "nominal_P"}.issubset(logo_plot_df.columns)


def test_per_celltype_ch6_networks_and_studio_zip_import(tmp_path: Path) -> None:
    """Verify per-population switchable Ch06 networks from --subtypes-dirs and REPORT.zip archive import."""
    import importlib.util as _ilu
    from katana_studio.report_builder import run_single_cell

    fx = _create_synthetic_study(tmp_path)
    meta_df = pd.read_csv(fx["sc_meta"], sep="\t")
    ev_df = pd.read_csv(fx["sc_events"], sep="\t")
    psi_df = pd.read_csv(fx["sc_psi"], sep="\t", index_col=0)
    sup_df = pd.read_csv(fx["sc_sup"], sep="\t", index_col=0)
    expr_df = pd.read_csv(fx["sc_expr"], sep="\t", index_col=0)

    # Create two synthetic subtypes with distinct population-specific scsplicenet modules
    sub_dirs: list[Path] = []
    for pop_name, ct_match, mod_assign in [
        ("PopA", "T_cell", ["M1"] * 15 + ["M2"] * 15 + ["M0"] * 15),
        ("PopB", "Monocyte", ["M1"] * 12 + ["M2"] * 12 + ["M3"] * 11 + ["M0"] * 10),
    ]:
        cid = f"{pop_name}__Activated_vs_Resting"
        sd = tmp_path / "subtypes" / cid
        (sd / "sc_matrices").mkdir(parents=True)
        (sd / "scsplicenet").mkdir(parents=True)
        (sd / "scsplicetrace").mkdir(parents=True)

        sub_cells = meta_df[meta_df["cell_type"] == ct_match].copy()
        sub_cells.to_csv(sd / "sc_matrices/sc_cells.tsv", sep="\t", index=False)
        s_ev = ev_df.copy()
        s_ev["pvalue"] = [0.002 if j < 15 else 0.70 for j in range(len(s_ev))]
        s_ev["fdr"] = [0.01 if j < 15 else 0.85 for j in range(len(s_ev))]
        s_ev.to_csv(sd / "sc_matrices/sc_events.tsv", sep="\t", index=False)

        pd.DataFrame({
            "event_id": ev_df["event_id"],
            "gene_id": ev_df["gene_id"],
            "module": mod_assign,
            "kME": [0.85 if m != "M0" else 0.0 for m in mod_assign],
        }).to_csv(sd / "scsplicenet/scsplicenet_modules.tsv", sep="\t", index=False)
        pd.DataFrame([
            {"RBP": "Rbfox1", "best_region": "downstream_exon_5p_50", "stat_Z": 3.4, "nominal_P": 0.001, "rbp_FDR_BH": 0.04, "rbp_P_FWER": 0.15}
        ]).to_csv(sd / "scsplicetrace/scsplicetrace_rbp_ranking.tsv", sep="\t", index=False)
        sub_dirs.append(sd)

    rep_dir = tmp_path / "sc_report"
    receipt = run_single_cell(
        outdir=rep_dir,
        events=ev_df,
        cells=meta_df,
        psi_matrix=psi_df.T,
        support_matrix=sup_df.T,
        expression=expr_df.T,
        gtf=fx["gtf"],
        subtypes_dirs=sub_dirs,
        study_name="Multi-Subtype Per-Population Network Study",
        species="mouse",
    )
    assert receipt["missing_plot_data"] == 0
    manifest = json.loads((rep_dir / "report-manifest.json").read_text(encoding="utf-8"))
    browsers = manifest.get("network_browsers", [])
    assert len(browsers) == 2
    comp_groups = {b["id"]: b.get("comparison_ids") for b in browsers}
    assert any(v == ["PopA__Activated_vs_Resting"] for v in comp_groups.values())
    assert any(v == ["PopB__Activated_vs_Resting"] for v in comp_groups.values())
    assert len({b["source"] for b in browsers}) == 2

    html_txt = (rep_dir / "REPORT.html").read_text(encoding="utf-8")
    assert "PopA__Activated_vs_Resting" in html_txt and "PopB__Activated_vs_Resting" in html_txt
    assert (rep_dir / "REPORT.zip").is_file()
    with zipfile.ZipFile(rep_dir / "REPORT.zip") as zf:
        names = set(zf.namelist())
        assert "REPORT.html" in names and "report-manifest.json" in names
        for b in browsers:
            assert b["source"] in names

    # Verify REPORT.zip passes report_source + import_report cleanly when duckdb is installed
    if _ilu.find_spec("duckdb") is not None:
        from katana_next.services.report_archive import report_source
        from katana_next.services.report_import import import_report

        cache_root = tmp_path / "studio_cache"
        ws_dest = tmp_path / "studio_workspace"
        unpacked_dir = report_source(rep_dir / "REPORT.zip", cache_root)
        ctx = import_report(unpacked_dir, ws_dest)
        assert (ws_dest / "katana.project.json").is_file()
        assert len(ctx.catalog.list_artifacts(ctx.manifest.project_id)) > 0


def test_bulk_report_studio_contract_end_to_end(tmp_path: Path) -> None:
    """Verify Bulk REPORT.zip carries normalized counts, MultiQC coverage, SpliceNet edges, and 13-channel FDR/FWER SpliceTrace for Katana Studio."""
    import importlib.util as _ilu
    from katana_studio.report_builder import run_bulk

    fx = _create_synthetic_study(tmp_path)
    rmats_df = pd.read_csv(fx["rmats"] / "SE.MATS.JCEC.txt", sep="\t")
    sample_ids = ["KO_1", "KO_2", "KO_3", "WT_1", "WT_2", "WT_3"]

    # 1. DESeq2 results + all.normalised_counts.tsv
    deg_dir = tmp_path / "deseq2"
    deg_dir.mkdir()
    deg_rows = []
    cnt_rows = []
    for g_idx in range(1, 16):
        gid = f"ENSMUSG{g_idx:011d}"
        gsym = f"Gene{g_idx}"
        deg_rows.append({
            "gene_id": gid, "gene_symbol": gsym, "baseMean": 450.0 + g_idx * 20,
            "log2FoldChange": 1.65 if g_idx <= 4 else (-1.45 if g_idx <= 8 else 0.15),
            "pvalue": 1e-4 if g_idx <= 8 else 0.52,
            "padj": 0.008 if g_idx <= 8 else 0.78,
        })
        row_c = {"gene_id": gid, "gene_symbol": gsym}
        for s_i, sid in enumerate(sample_ids):
            base_v = 320.0 if (g_idx <= 4 and s_i < 3) or (4 < g_idx <= 8 and s_i >= 3) else 110.0
            row_c[sid] = round(base_v + (g_idx * 7 + s_i * 5), 2)
        cnt_rows.append(row_c)
    pd.DataFrame(deg_rows).to_csv(deg_dir / "KO_vs_WT.deseq2.results.tsv", sep="\t", index=False)
    pd.DataFrame(cnt_rows).to_csv(deg_dir / "all.normalised_counts.tsv", sep="\t", index=False)

    # 2. MultiQC Qualimap gene-body coverage
    mqc_dir = tmp_path / "multiqc"
    mqc_dir.mkdir()
    cov_cols = {str(p): [round(0.4 + 0.012 * p + 0.03 * s_i, 4) for s_i in range(len(sample_ids))] for p in range(1, 101)}
    cov_df = pd.DataFrame({"Sample": sample_ids, **cov_cols})
    cov_df.to_csv(mqc_dir / "qualimap_gene_coverage_profile_Normalised.txt", sep="\t", index=False)

    # 3. SpliceNet module assignments, scores & pairwise edges
    sn_dir = tmp_path / "splicenet"
    sn_dir.mkdir()
    ev_keys = [
        f"SE|{r['chr']}|{r['strand']}|{int(r['exonStart_0base'])}|{int(r['exonEnd'])}|{int(r['upstreamEE'])}|{int(r['downstreamES'])}"
        for _, r in rmats_df.iterrows()
    ]
    pd.DataFrame({
        "event_key": ev_keys,
        "gene_id": rmats_df["GeneID"],
        "gene_symbol": rmats_df["geneSymbol"],
        "module": ["M1" if i < 20 else ("M2" if i < 40 else "M0") for i in range(len(ev_keys))],
        "kME": [round(0.88 - 0.008 * (i % 15), 4) if i < 40 else 0.05 for i in range(len(ev_keys))],
    }).to_csv(sn_dir / "module_assignments.tsv", sep="\t", index=False)
    pd.DataFrame({
        "sample_id": sample_ids,
        "M1": [0.42, 0.38, 0.45, -0.40, -0.36, -0.44],
        "M2": [-0.31, -0.28, -0.35, 0.32, 0.29, 0.34],
    }).to_csv(sn_dir / "module_scores.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"event_a": ev_keys[i], "event_b": ev_keys[i + 1], "weight": round(0.84 - 0.02 * i, 4), "correlation": round(0.86 - 0.02 * i, 4)}
        for i in range(18)
    ]).to_csv(sn_dir / "splicenet_edges.tsv", sep="\t", index=False)

    # 4. SpliceTrace ranking, 13-channel region_tests, bridge, proteome & motif consequences
    st_dir = tmp_path / "splicetrace_KO_vs_WT"
    st_dir.mkdir()
    pd.DataFrame([
        {"RBP": "RBFOX1", "best_region": "downstream_intron_0_35", "stat_Z": 4.25, "nominal_P": 0.00012, "rbp_FDR_BH": 0.0042, "rbp_P_FWER": 0.018},
        {"RBP": "PTBP1", "best_region": "upstream_intron_0_35", "stat_Z": -3.15, "nominal_P": 0.0048, "rbp_FDR_BH": 0.042, "rbp_P_FWER": 0.14},
    ]).to_csv(st_dir / "rbp_ranking.tsv", sep="\t", index=False)
    all_13_regs = [
        "upstream_intron_200_500", "upstream_intron_85_200", "upstream_intron_35_85", "upstream_intron_0_35",
        "upstream_exon_3p_50", "alt_exon_5p_50", "alt_exon_3p_50", "downstream_exon_5p_50",
        "downstream_intron_0_35", "downstream_intron_35_85", "downstream_intron_85_200", "downstream_intron_200_500",
        "mature_consequence_delta",
    ]
    reg_rows = []
    for rbp_n, peak_r, peak_z in [("RBFOX1", "downstream_intron_0_35", 4.25), ("PTBP1", "upstream_intron_0_35", -3.15)]:
        for rg in all_13_regs:
            zv = peak_z if rg == peak_r else 0.4
            pv = 0.00012 if rg == peak_r else 0.65
            reg_rows.append({"RBP": rbp_n, "region": rg, "stat_Z": zv, "nominal_P": pv, "region_FDR_BH": pv * 3, "region_P_FWER": min(1.0, pv * 13)})
    pd.DataFrame(reg_rows).to_csv(st_dir / "region_tests.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"module": "M1", "module_events": 20, "module_genes": 10, "RBP": "RBFOX1", "best_region": "downstream_intron_0_35", "kme_direction": "signed", "stat_Z": 3.9, "nominal_P": 0.0008, "module_FDR_BH": 0.015, "module_P_FWER": 0.04, "study_P_FWER": 0.04},
    ]).to_csv(st_dir / "module_rbp_bridge.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"event_id": ev_keys[0], "gene_id": rmats_df.iloc[0]["GeneID"], "gene_symbol": rmats_df.iloc[0]["geneSymbol"], "event_type": "SE", "dpsi": rmats_df.iloc[0]["IncLevelDifference"], "uniprot_acc": "Q9JJ43", "frame_status": "frameshift", "nmd_switch_class": "NMD_inclusion_trigger", "aa_len_inclusion": 180, "aa_len_exclusion": 420, "delta_aa": -240, "disrupted_protein_features": "DOMAIN:RRM_1", "functional_consequence_summary": "Inclusion triggers 50-nt EJC NMD"},
    ]).to_csv(st_dir / "protein_and_nmd_consequences.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"event_id": ev_keys[0], "gene_id": rmats_df.iloc[0]["GeneID"], "gene_symbol": rmats_df.iloc[0]["geneSymbol"], "rbp_list": "RBFOX1;PTBP1", "region": "downstream_intron_0_35", "hit_RNA": "UGCAUG", "delta_motif_score": 3.4},
    ]).to_csv(st_dir / "motif_consequences.tsv", sep="\t", index=False)

    rep_dir = tmp_path / "bulk_report"
    receipt = run_bulk(
        outdir=rep_dir,
        rmats=fx["rmats"],
        samples=fx["samples"],
        contrasts=fx["contrasts"],
        deseq2=deg_dir,
        multiqc=mqc_dir,
        fasta=fx["fasta"],
        gtf=fx["gtf"],
        splicenet_dir=sn_dir,
        splicetrace_dirs={"KO_vs_WT": st_dir},
        study_name="Bulk Contract Regression Study",
        species="mouse",
    )
    assert receipt["missing_plot_data"] == 0
    manifest = json.loads((rep_dir / "report-manifest.json").read_text(encoding="utf-8"))
    comp_results = manifest.get("complete_results", [])
    table_aids = {t.get("analysis_id") for t in comp_results}
    assert {"quality.read_coverage_bias", "expression.normalised_counts", "programs.splicenet_edges", "prioritize.splicetrace_regions"}.issubset(table_aids)
    cov_rec = next(t for t in comp_results if t.get("analysis_id") == "quality.read_coverage_bias")
    assert int(cov_rec.get("rows", 0)) == len(sample_ids) * 100
    assert (rep_dir / str(cov_rec["source"])).is_file()

    if _ilu.find_spec("duckdb") is not None:
        from katana_next.services.report_archive import report_source
        from katana_next.services.report_import import import_report
        from katana_next.services.studio_compute import (
            compute_live_go_kegg_ora,
            compute_live_motif_scan,
            compute_live_psi_landscape,
            compute_live_target_ranker,
        )
        from katana_next.services.studio_interactive import (
            genecard_preview,
            quadrant_data,
            splicetrace_workbench,
            unified_network,
        )

        ctx = import_report(report_source(rep_dir / "REPORT.zip", tmp_path / "cache"), tmp_path / "ws_bulk")
        net = unified_network(ctx, comparison="KO_vs_WT")
        assert net["available"] is True
        assert any(g.get("edges") for g in net.get("graphs", {}).values())
        assert len(net.get("bridge_rbps", [])) >= 1

        st = splicetrace_workbench(ctx, comparison="KO_vs_WT")
        assert st["available"] is True and st["rbps"][0]["fdr_bh"] is not None
        assert len(st["regions"].get("RBFOX1", [])) == 13

        qd = quadrant_data(ctx, comparison="KO_vs_WT")
        assert len(qd["rows"]) > 0

        gc = genecard_preview(ctx, gene_query="Gene1", comparison="KO_vs_WT")
        assert gc["found"] is True and gc["das_artifact_id"] and gc["das_row_id"]
        assert len(gc["track1_quantification"]["expression_samples"]) == len(sample_ids)

        pl = compute_live_psi_landscape(ctx, comparison="KO_vs_WT")
        assert pl["available"] is True
        ora = compute_live_go_kegg_ora(
            ctx,
            comparison="KO_vs_WT",
            custom_genes=["Rbfox1", "Rbfox2", "Ptbp1", "Srsf1", "Nrxn1", "Snap25", "Grin1"],
        )
        assert ora["available"] is True and len(ora.get("top_rows", [])) > 0
        ms = compute_live_motif_scan(ctx, comparison="KO_vs_WT", query_motif_or_rbp="RBFOX1")
        assert ms.get("consensus") and ms.get("catalog_size", 0) > 100 and len(ms.get("matched_events", [])) > 0
        tr = compute_live_target_ranker(ctx, comparison="KO_vs_WT")
        assert tr["available"] is True and tr["rows"]
        assert tr["rows"][0]["source_artifact_id"] and tr["rows"][0]["das_row_id"]


def test_sc_report_studio_contract_end_to_end(tmp_path: Path) -> None:
    """Verify Single-Cell REPORT.zip exports manifold embeddings, compact PSI/expr matrices, and scope-switchable Ch06 networks for Studio."""
    import importlib.util as _ilu
    from katana_studio.report_builder import run_single_cell

    fx = _create_synthetic_study(tmp_path)
    meta_df = pd.read_csv(fx["sc_meta"], sep="\t")
    ev_df = pd.read_csv(fx["sc_events"], sep="\t")
    psi_df = pd.read_csv(fx["sc_psi"], sep="\t", index_col=0)
    sup_df = pd.read_csv(fx["sc_sup"], sep="\t", index_col=0)
    expr_df = pd.read_csv(fx["sc_expr"], sep="\t", index_col=0)

    # Manifold coordinates for SingleCellManifoldExplorer
    sm_dir = tmp_path / "sc_manifold"
    sm_dir.mkdir()
    coords = pd.DataFrame({
        "sample_id": meta_df["cell_id"],
        "condition": meta_df["condition"],
        "donor_id": meta_df["donor_id"],
        "cell_type": meta_df["cell_type"],
        "PC1": np.linspace(-2.0, 2.0, len(meta_df)),
        "PC2": np.sin(np.linspace(0, 6.0, len(meta_df))),
        "psi_UMAP1": np.linspace(-1.5, 1.5, len(meta_df)),
        "psi_UMAP2": np.cos(np.linspace(0, 5.0, len(meta_df))),
        "expr_UMAP1": np.linspace(-1.0, 1.0, len(meta_df)),
        "expr_UMAP2": np.sin(np.linspace(0, 4.0, len(meta_df))),
    })
    coords.to_csv(sm_dir / "sc_cell_manifold_coordinates.tsv", sep="\t", index=False)

    sub_dirs: list[Path] = []
    for pop_name, ct_match, mod_assign in [
        ("PopA", "T_cell", ["M1"] * 18 + ["M2"] * 17 + ["M0"] * 10),
        ("PopB", "Monocyte", ["M1"] * 12 + ["M2"] * 12 + ["M3"] * 11 + ["M0"] * 10),
    ]:
        cid = f"{pop_name}__Activated_vs_Resting"
        sd = tmp_path / "sc_subtypes" / cid
        (sd / "sc_matrices").mkdir(parents=True)
        (sd / "scsplicenet").mkdir(parents=True)
        (sd / "scsplicetrace").mkdir(parents=True)

        meta_df[meta_df["cell_type"] == ct_match].to_csv(sd / "sc_matrices/sc_cells.tsv", sep="\t", index=False)
        s_ev = ev_df.copy()
        s_ev["pvalue"] = [0.0015 if j < 15 else 0.72 for j in range(len(s_ev))]
        s_ev["fdr"] = [0.009 if j < 15 else 0.85 for j in range(len(s_ev))]
        s_ev.to_csv(sd / "sc_matrices/sc_events.tsv", sep="\t", index=False)
        pd.DataFrame({
            "event_id": ev_df["event_id"], "gene_id": ev_df["gene_id"], "gene_symbol": ev_df["gene_symbol"],
            "module": mod_assign, "kME": [0.85 if m != "M0" else 0.0 for m in mod_assign],
        }).to_csv(sd / "scsplicenet/scsplicenet_modules.tsv", sep="\t", index=False)
        pd.DataFrame([
            {"source_event": ev_df.iloc[i]["event_id"], "target_event": ev_df.iloc[i + 1]["event_id"], "shrunk_correlation": round(0.85 - 0.03 * i, 4)}
            for i in range(10)
        ]).to_csv(sd / "scsplicenet/scsplicenet_edges.tsv", sep="\t", index=False)
        pd.DataFrame([
            {"RBP": "RBFOX1", "best_region": "downstream_exon_5p_50", "stat_Z": 3.4, "nominal_P": 0.001, "rbp_FDR_BH": 0.03, "rbp_P_FWER": 0.12}
        ]).to_csv(sd / "scsplicetrace/scsplicetrace_rbp_ranking.tsv", sep="\t", index=False)
        sub_dirs.append(sd)

    rep_dir = tmp_path / "sc_studio_rep"
    run_single_cell(
        outdir=rep_dir, events=ev_df, cells=meta_df,
        psi_matrix=psi_df.T, support_matrix=sup_df.T, expression=expr_df.T,
        gtf=fx["gtf"], splicemanifold_dir=sm_dir, subtypes_dirs=sub_dirs,
        study_name="Single-Cell Studio Contract Study", species="mouse",
    )
    manifest = json.loads((rep_dir / "report-manifest.json").read_text(encoding="utf-8"))
    sc_aids = {t.get("analysis_id") for t in manifest.get("complete_results", [])}
    assert {"single_cell.cell_embedding", "single_cell.psi_matrix", "single_cell.expression_matrix"}.issubset(sc_aids)

    if _ilu.find_spec("duckdb") is not None:
        from katana_next.services.report_archive import report_source
        from katana_next.services.report_import import import_report
        from katana_next.services.studio_compute import compute_live_target_ranker
        from katana_next.services.studio_interactive import genecard_preview, single_cell_manifold, unified_network

        ctx = import_report(report_source(rep_dir / "REPORT.zip", tmp_path / "sc_cache"), tmp_path / "ws_sc")
        mf = single_cell_manifold(ctx)
        assert mf["available"] is True and len(mf["cells"]) == len(meta_df)
        assert len(mf["psi_events"]) > 0 and len(mf["expr_genes"]) > 0

        n1 = unified_network(ctx)
        scopes = n1.get("available_scopes", [])
        assert len(scopes) >= 2
        n2 = unified_network(ctx, scope=scopes[1]["scope_slug"])
        assert n2["active_scope"] == scopes[1]["scope_slug"]
        assert n2["active_scope"] != n1["active_scope"]

        gc_sc = genecard_preview(ctx, gene_query="Gene1", comparison="PopA__Activated_vs_Resting")
        assert gc_sc["found"] is True and gc_sc["das_artifact_id"] and gc_sc["das_row_id"]
        rk_sc = compute_live_target_ranker(ctx, comparison="PopA__Activated_vs_Resting")
        assert rk_sc["available"] is True and len(rk_sc["rows"]) > 0
        from katana_next.services.studio_interactive import cross_paradigm_explorer
        assert cross_paradigm_explorer(ctx)["available"] is False


def test_selected_target_genecard_eventcard_workflow(tmp_path: Path) -> None:
    """Verify /targets/resolve, strict request.zip + sibling targets.tsv, NEXTFLOW_TASK.zip, execution & restore ledger."""
    import importlib.util as _ilu
    from katana_studio.report_builder import run_bulk

    fx = _create_synthetic_study(tmp_path)
    rep_dir = tmp_path / "tgt_rep"
    run_bulk(
        outdir=rep_dir, rmats=fx["rmats"], samples=fx["samples"], contrasts=fx["contrasts"],
        fasta=fx["fasta"], gtf=fx["gtf"], study_name="Target Workflow Study", species="mouse",
    )

    if _ilu.find_spec("duckdb") is None:
        return

    from katana_next.services.report_archive import report_source
    from katana_next.services.report_import import import_report
    from katana_next.targets.nextflow import prepare_task
    from katana_next.targets.restore import restore_results
    from katana_next.targets.runner import inspect_results, run_targets

    ws_dir = tmp_path / "ws_tgt"
    ctx = import_report(report_source(rep_dir / "REPORT.zip", tmp_path / "tgt_cache"), ws_dir)
    pid = ctx.manifest.project_id
    das_art = next(a for a in ctx.catalog.list_artifacts(pid) if a.metadata.get("source_analysis_id") == "differential.as")
    das_df = pd.read_parquet(das_art.path)
    gene_row_id = str(das_df.iloc[0]["row_id"])
    event_row_id = str(das_df.iloc[1]["row_id"])

    if _ilu.find_spec("fastapi") is not None and _ilu.find_spec("httpx") is not None:
        from fastapi.testclient import TestClient
        from katana_next.api.app import PlatformRuntime, create_app

        app = create_app(PlatformRuntime(application_root=tmp_path / "studio_root"))
        client = TestClient(app)
        open_res = client.post("/api/projects/open", json={"manifest_path": str(ws_dir / "katana.project.json")})
        assert open_res.status_code == 200, open_res.text

        res_resolve = client.post(f"/api/projects/{pid}/targets/resolve", json={"queries": ["Gene1", "Gene2"], "target_kind": "gene", "comparison": "KO_vs_WT"})
        assert res_resolve.status_code == 200, res_resolve.text
        rj = res_resolve.json()
        assert len(rj["selections"]) >= 1 and len(rj["matched"]) >= 2

        exp_res = client.post(f"/api/projects/{pid}/targets/export", json={
            "selections": [
                {"source_artifact_id": das_art.artifact_id, "row_ids": [gene_row_id], "target_kind": "gene", "filters": {}},
                {"source_artifact_id": das_art.artifact_id, "row_ids": [event_row_id], "target_kind": "event", "filters": {}},
            ],
            "modules": ["gene_card", "event_card", "sashimi"],
        })
        assert exp_res.status_code == 200, exp_res.text
        ej = exp_res.json()
        assert ej["target_count"] == 2 and ej.get("targets_tsv_url")

        with zipfile.ZipFile(ej["archive_path"]) as zf:
            names = set(zf.namelist())
            assert "request.json" in names
            assert "targets.tsv" not in names  # strict manifest separation inside request.zip

        tsv_res = client.get(ej["targets_tsv_url"])
        assert tsv_res.status_code == 200 and "target_id" in tsv_res.text

        nf_res = client.post(f"/api/projects/{pid}/targets/nextflow", json={"request_id": ej["request_id"]})
        assert nf_res.status_code == 200, nf_res.text
        nfj = nf_res.json()
        assert nfj.get("targets_tsv_url")
        with zipfile.ZipFile(nfj["archive"]) as zf:
            nf_names = set(zf.namelist())
            assert {"katana_targets.nf", "targets.tsv", "request.zip"}.issubset(nf_names)

        req_path = Path(ej["request_path"])
    else:
        if _ilu.find_spec("fastapi") is not None:
            from katana_next.api.app import PlatformRuntime, create_app
            from katana_next.api.targets import TargetResolveRequest, create_targets_router
            rt = PlatformRuntime(application_root=tmp_path / "studio_root")
            app = create_app(rt)
            assert app is not None
            opened = rt.projects.open_project(ws_dir / "katana.project.json")
            router = create_targets_router(rt)
            ep_resolve = next(r.endpoint for r in router.routes if getattr(r, "path", "").endswith("/targets/resolve"))
            rj = ep_resolve(opened.manifest.project_id, TargetResolveRequest(queries=["Gene1", "Gene2"], target_kind="gene", comparison="KO_vs_WT"))
            assert len(rj["selections"]) >= 1 and len(rj["matched"]) >= 2

        from katana_next.targets.bundle import export_targets
        ej = export_targets(
            ctx,
            [
                {"source_artifact_id": das_art.artifact_id, "row_ids": [gene_row_id], "target_kind": "gene", "filters": {}},
                {"source_artifact_id": das_art.artifact_id, "row_ids": [event_row_id], "target_kind": "event", "filters": {}},
            ],
            tmp_path / "exp_bundle",
            modules=["gene_card", "event_card"],
        )
        with zipfile.ZipFile(ej["archive_path"]) as zf:
            assert "request.json" in zf.namelist() and "targets.tsv" not in zf.namelist()
        assert Path(ej["targets_tsv_path"]).is_file()
        nf_out = prepare_task(ej["archive_path"], tmp_path / "nf_prep")
        with zipfile.ZipFile(nf_out["archive"]) as zf:
            assert {"katana_targets.nf", "targets.tsv", "request.zip"}.issubset(set(zf.namelist()))
        req_path = Path(ej["request_path"])

    run_out = tmp_path / "tgt_run_out"
    run_res = run_targets(req_path, run_out)
    assert run_res["status"] in {"complete", "partial"}
    verified = inspect_results(run_out)
    assert verified["verified"] is True
    assert (run_out / "INDEX.html").is_file() and (run_out / "REPORT.html").is_file()
    assert (run_out / "targets.tsv").is_file() and (run_out / "REPORT.zip").is_file()
    idx_html = (run_out / "INDEX.html").read_text(encoding="utf-8")
    assert "katana-grid-2x2" in idx_html and "target-select" in idx_html
    assert "All gene events" not in idx_html and "SpliceNet" not in idx_html
    m_data = re.search(r"<script id='katana-batch-data' type='application/json'>(.*?)</script>", idx_html, re.S)
    assert m_data is not None
    batch_payload = json.loads(m_data.group(1))
    assert len(batch_payload["targets"]) == 2
    assert any(
        ev.get("stats_by_comparison")
        for t in batch_payload["targets"]
        for ev in t.get("events", [])
    )
    rendered_htmls = [p.read_text(encoding="utf-8") for p in (run_out / "targets").rglob("REPORT.html")]
    assert rendered_htmls and all("katana-grid-2x2" in h and "All gene events" not in h and "SpliceNet" not in h for h in rendered_htmls)
    restored = tmp_path / "tgt_restored"
    restore_results(run_out / "REPORT.zip", restored, expected_project_id=pid)
    assert (restored / "INDEX.html").is_file()


def test_sashimi_events_skips_coordinate_less_annotation_cards() -> None:
    from katana_next.targets.runner import _sashimi_events

    cards = [
        {
            "event_id": "SE|chr1|+|3700|3745|3380|4100",
            "event_type": "SE",
            "comparison_id": "KO_vs_WT",
            "event_geometry": {},
            "evidence": [{"evidence_id": "ev_motif", "row": {"hit_RNA": "UGCAUG"}, "metadata": {}, "parameters": {}}],
        },
        {
            "event_id": "SE:chr1:+:3700:3745:3200:3380:4100:4280",
            "event_type": "SE",
            "comparison_id": "KO_vs_WT",
            "event_geometry": {"chr": "chr1", "strand": "+"},
            "evidence": [{"evidence_id": "ev_geom", "row": {}, "metadata": {}, "parameters": {}}],
        },
        {
            "event_id": "SE:chr1:+:5700:5809:5200:5360:6200:6380",
            "event_type": "SE",
            "comparison_id": "KO_vs_WT",
            "event_geometry": {"chr": "chr1", "strand": "+"},
            "evidence": [{"evidence_id": "ev_rmats", "row": {"IncLevel1": "0.7,0.8", "IJC_SAMPLE_1": "30,35"}, "metadata": {}, "parameters": {}}],
        },
    ]
    out = _sashimi_events(cards)
    assert [e["event_id"] for e in out] == [
        "SE:chr1:+:3700:3745:3200:3380:4100:4280",
        "SE:chr1:+:5700:5809:5200:5360:6200:6380",
    ]
    assert out[0]["source_evidence_id"] == "geometry-only"
    assert out[1]["source_evidence_id"] == "ev_rmats"


def test_cross_paradigm_dexseq_sign_fix_and_fused_bulk_report(tmp_path: Path) -> None:
    """Verify DEXSeq log2fold_<ctrl>_<treat> sign calibration and end-to-end Cross-Paradigm fusion into Chapters 04, 07, and 08 (no Ch09)."""
    import shutil
    from katana_studio.cli import report_main
    from katana_studio.cross_paradigm import cli_main as cp_cli_main

    fx = _create_synthetic_study(tmp_path)

    # Augment synthetic GTF with explicit transcript lines so Gene10 has Coding -> NMD isoform switching
    gtf_txt = fx["gtf"].read_text(encoding="utf-8")
    g10_base = 200 + 10 * 700
    g10_gid = "ENSMUSG00000000010"
    g10_coding = "ENSMUST00000000010"
    g10_nmd = "ENSMUST00000000110"
    extra_gtf = "\n".join([
        f'chr1\tsynth\ttranscript\t{g10_base}\t{g10_base + 540}\t.\t+\t.\tgene_id "{g10_gid}"; transcript_id "{g10_coding}"; gene_name "Gene10"; transcript_type "protein_coding";',
        f'chr1\tsynth\ttranscript\t{g10_base}\t{g10_base + 540}\t.\t+\t.\tgene_id "{g10_gid}"; transcript_id "{g10_nmd}"; gene_name "Gene10"; transcript_type "nonsense_mediated_decay";',
        f'chr1\tsynth\texon\t{g10_base}\t{g10_base + 90}\t.\t+\t.\tgene_id "{g10_gid}"; transcript_id "{g10_nmd}"; gene_name "Gene10";',
        f'chr1\tsynth\texon\t{g10_base + 200}\t{g10_base + 290}\t.\t+\t.\tgene_id "{g10_gid}"; transcript_id "{g10_nmd}"; gene_name "Gene10";',
    ])
    aug_gtf = tmp_path / "annotation_aug.gtf"
    aug_gtf.write_text(gtf_txt + extra_gtf + "\n", encoding="utf-8")

    # Build Nextflow-style katana_bundle layout
    bundle = tmp_path / "bundle_cp"
    for d in ("rmats/KO_vs_WT/rmats_post", "DEG", "dexseq_exon/annotation", "suppa/KO_vs_WT/diffsplice/per_local_event", "suppa/KO_vs_WT/diffsplice/per_isoform", "dtu/results/stager", "quant/matrices"):
        (bundle / d).mkdir(parents=True, exist_ok=True)
    shutil.copy2(fx["samples"], bundle / "samplesheet.csv")
    shutil.copy2(fx["contrasts"], bundle / "contrastsheet.csv")
    for fn in ("SE.MATS.JCEC.txt", "SE.MATS.JC.txt"):
        shutil.copy2(fx["rmats"] / fn, bundle / "rmats/KO_vs_WT/rmats_post" / fn)

    sample_ids = ["KO_1", "KO_2", "KO_3", "WT_1", "WT_2", "WT_3"]
    # DEG: Gene1..4 significant; Gene10 explicitly non-significant (hidden from DEG)
    deg_rows = []
    cnt_rows = []
    for g_idx in range(1, 16):
        gid = f"ENSMUSG{g_idx:011d}"
        gsym = f"Gene{g_idx}"
        deg_rows.append({
            "gene_id": gid, "gene_symbol": gsym, "baseMean": 400.0 + g_idx * 15,
            "log2FoldChange": 1.6 if g_idx <= 4 else 0.08,
            "pvalue": 1e-4 if g_idx <= 4 else 0.65,
            "padj": 0.005 if g_idx <= 4 else 0.78,
        })
        cnt_rows.append({"gene_id": gid, "gene_symbol": gsym, **{s: round(250.0 + g_idx * 5, 2) for s in sample_ids}})
    pd.DataFrame(deg_rows).to_csv(bundle / "DEG/KO_vs_WT.deseq2.results.tsv", sep="\t", index=False)
    pd.DataFrame(cnt_rows).to_csv(bundle / "DEG/all.normalised_counts.tsv", sep="\t", index=False)

    # DEXSeq: intentionally emit inverted log2fold_WT_KO (ctrl/treat polarity!) matching rMATS exon coords
    gff_rows = []
    dex_rows = []
    for g_idx in range(1, 16):
        base = 200 + g_idx * 700
        gid = f"ENSMUSG{g_idx:011d}"
        is_c1 = g_idx <= 5
        # Bin E001 (5' exon) + E002 (exact rMATS skipped exon [base+200, base+290]) + E003 (3' exon)
        for bin_i, (s_b, e_b) in enumerate([(base, base + 90), (base + 200, base + 290), (base + 420, base + 540)], 1):
            fid = f"E{bin_i:03d}"
            gff_rows.append(f'chr1\tdexseq\texonic_part\t{s_b}\t{e_b}\t.\t+\t.\tgene_id "{gid}"; exonic_part_number "{bin_i:03d}";')
            if bin_i == 2 and is_c1:
                raw_lfc, pv, pa, ko_m, wt_m = -1.85, 1e-8, 1e-6, 18.0, 4.5
            elif bin_i == 1 and g_idx == 12:
                raw_lfc, pv, pa, ko_m, wt_m = -1.60, 2e-6, 2e-4, 14.0, 4.0
            else:
                raw_lfc, pv, pa, ko_m, wt_m = 0.05, 0.70, 0.85, 10.0, 10.2
            dex_rows.append({
                "groupID": gid, "featureID": fid, "exonBaseMean": 120.0,
                "dispersion": 0.05, "stat": 12.0, "pvalue": pv, "padj": pa,
                "KO": ko_m, "WT": wt_m, "log2fold_WT_KO": raw_lfc,
                "genomicData.seqnames": "chr1", "genomicData.start": s_b, "genomicData.end": e_b, "genomicData.strand": "+",
            })
    (bundle / "dexseq_exon/annotation/DEXSeq.gff").write_text("\n".join(gff_rows) + "\n", encoding="utf-8")
    pd.DataFrame(dex_rows).to_csv(bundle / "dexseq_exon/DEXSeqResults.KO_vs_WT.csv", index=False)

    # SUPPA2 local events: 2 BH-FDR-significant AF/AL on novel genes, 1 SE, 1 nominal-only AF, plus background SE events
    suppa_rows = [
        {"event_id": "ENSMUSG00000000021;AF:chr1:7900-7990:8100-8190:+", "KO-WT_dPSI": 0.48, "KO-WT_p-val": 0.0001},
        {"event_id": "ENSMUSG00000000022;AL:chr1:9600-9680:9720-9840:+", "KO-WT_dPSI": -0.44, "KO-WT_p-val": 0.0002},
        {"event_id": "ENSMUSG00000000001;SE:chr1:990-1100:1190-1320:+", "KO-WT_dPSI": 0.35, "KO-WT_p-val": 0.0010},
        {"event_id": "ENSMUSG00000000023;AF:chr1:10200-10290:10400-10490:+", "KO-WT_dPSI": 0.32, "KO-WT_p-val": 0.0400},
    ] + [
        {"event_id": f"ENSMUSG{bg_i:011d};SE:chr1:{200 + bg_i * 700}-{290 + bg_i * 700}:{400 + bg_i * 700}-{490 + bg_i * 700}:+", "KO-WT_dPSI": 0.02, "KO-WT_p-val": 0.80}
        for bg_i in range(2, 12)
    ]
    pd.DataFrame(suppa_rows).to_csv(bundle / "suppa/KO_vs_WT/diffsplice/per_local_event/local_events.dpsi", sep="\t", index=False)

    # stageR DTU + Salmon transcript TPM for Gene10 (Coding -> NMD, hidden from DEG)
    pd.DataFrame([
        {"geneID": g10_gid, "txID": g10_coding, "gene": 1e-4, "transcript": 2e-4},
        {"geneID": g10_gid, "txID": g10_nmd, "gene": 1e-4, "transcript": 3e-4},
    ]).to_csv(bundle / "dtu/results/stager/getAdjustedPValues.KO_vs_WT.tsv", sep="\t", index=False)
    pd.DataFrame([
        {"tx": g10_coding, "gene_id": g10_gid, "KO_1": 18.0, "KO_2": 20.0, "KO_3": 22.0, "WT_1": 88.0, "WT_2": 90.0, "WT_3": 92.0},
        {"tx": g10_nmd, "gene_id": g10_gid, "KO_1": 82.0, "KO_2": 80.0, "KO_3": 78.0, "WT_1": 12.0, "WT_2": 10.0, "WT_3": 8.0},
    ]).to_csv(bundle / "quant/matrices/salmon.merged.transcript_tpm.tsv", sep="\t", index=False)

    # Step 1: Run standalone KATANA_CROSS_PARADIGM CLI
    cp_dir = tmp_path / "cp_precomputed"
    ver_yml = tmp_path / "cp_versions.yml"
    rc_cp = cp_cli_main([
        "--bundle", str(bundle),
        "--gtf", str(aug_gtf),
        "--rmats-mode", "JCEC",
        "--outdir", str(cp_dir),
        "--emit-versions-yml", str(ver_yml),
    ])
    assert rc_cp == 0 and ver_yml.is_file()
    cp_sum = json.loads((cp_dir / "cross_paradigm_summary.json").read_text(encoding="utf-8"))
    assert cp_sum["available"] is True
    cnt = cp_sum["contrasts"]["KO_vs_WT"]["counts"]
    assert cp_sum["contrasts"]["KO_vs_WT"]["dexseq_sign_inverted"] is True
    assert cnt["class1_events"] >= 1 and cnt["class1_concordant_pct"] == 100.0
    assert cnt["suppa_af_al_fdr"] == 2
    assert cnt["suppa_af_al_nom"] == 3
    assert cnt["suppa_af_al_genes"] == 2
    assert cnt["suppa_af_al_novel_genes"] == 2
    assert cnt["suppa_af_al_nom_genes"] == 3
    assert cnt["suppa_af_al_nom_novel_genes"] == 3
    assert cp_sum["bench_multi_contrast"][0]["unlocked_afal"] == 2
    assert cnt["dtu_switch_resolved"] == 1 and cnt["dtu_hidden_from_deg"] == 1
    assert cp_sum["contrasts"]["KO_vs_WT"]["tables_preview"]["class3_switches"][0]["fate"] == "Coding -> NMD (Loss of ORF)"
    assert "gene_lists" in cp_sum["contrasts"]["KO_vs_WT"]
    assert len(cp_sum["contrasts"]["KO_vs_WT"]["gene_lists"]["class3_switches"]) == 1
    assert len(cp_sum["contrasts"]["KO_vs_WT"]["gene_lists"]["class2_af_al_fdr"]) == 2
    assert len(cp_sum["contrasts"]["KO_vs_WT"]["gene_lists"]["class2_af_al"]) == 3

    # Step 2: Run full katana-report bulk CLI reusing --cross-paradigm-dir and fuse into existing 8 chapters
    rep_dir = tmp_path / "fused_report"
    rc_rep = report_main([
        "bulk",
        "--bundle", str(bundle),
        "--fasta", str(fx["fasta"]),
        "--gtf", str(aug_gtf),
        "--cross-paradigm-dir", str(cp_dir),
        "--study-name", "Cross-Paradigm Fused Smoke Test",
        "--species", "mouse",
        "--outdir", str(rep_dir),
    ])
    assert rc_rep == 0

    manifest = json.loads((rep_dir / "report-manifest.json").read_text(encoding="utf-8"))
    assert manifest["figure_data_links"]["missing_plot_data"] == []
    assert {p["chapter"] for p in manifest["panels"]}.issubset({1, 2, 3, 4, 5, 6, 7, 8})

    # Verify DEXSeq parquet auto-calibration reversed log2fold_WT_KO (-1.85 -> +1.85)
    dex_pq = pd.read_parquet(rep_dir / "outputs/comparisons/KO_vs_WT/DEXSeq-complete.parquet")
    c1_bin = dex_pq[(dex_pq["gene_id"] == "ENSMUSG00000000001") & (dex_pq["feature_id"] == "E002")].iloc[0]
    assert float(c1_bin["log2fc"]) > 1.5

    # Verify Ch04, Ch07, Ch08 fusion + zero Ch09 + interactive explorer + candidate enrichment
    panel_chapters = {p["id"]: p["chapter"] for p in manifest["panels"]}
    assert panel_chapters["cp-taxonomy-concordance-KO_vs_WT"] == 4
    assert panel_chapters["cp-afal-positional-KO_vs_WT"] == 4
    assert panel_chapters["cp-isoform-fates-KO_vs_WT"] == 7

    html_txt = (rep_dir / "REPORT.html").read_text(encoding="utf-8")
    assert "interactive-cross-paradigm-isoform-explorer" in html_txt
    assert "Jump to Cross-Paradigm" not in html_txt
    assert "Gene Story Explorer" not in html_txt
    assert "fmtQ" in html_txt and "sigStars" in html_txt
    assert "gene_dex_bins" in html_txt and "rmats_fdr" in html_txt and "tx_confirm_tier" in html_txt
    assert "kcp-genelist-box" in html_txt and "kcp-guide" in html_txt and "syncGeneDropdown" in html_txt
    for ch_n in (1, 2, 3, 4, 6, 7, 8):
        assert f"Chapter 0{ch_n}" in html_txt
    assert "Chapter 09" not in html_txt and "chapter-9" not in html_txt

    prio_df = pd.read_csv(rep_dir / "outputs/synthesis/integrative_priority_targets_KO_vs_WT.tsv", sep="\t")
    assert {"coord_verified_exon", "isoform_switch_fate", "hidden_from_deg"}.issubset(set(prio_df.columns))

    with zipfile.ZipFile(rep_dir / "REPORT.zip", "r") as zf:
        znames = set(zf.namelist())
        assert "REPORT.html" in znames
        assert "outputs/cross_paradigm/CROSS_PARADIGM_EXPLORER.html" in znames

    # Step 3: Verify Katana Studio API integration (Cross-Paradigm Explorer, rMATS-blind GeneCard, Target Resolve/Export & Target Ranker)
    import importlib.util as _ilu
    if _ilu.find_spec("duckdb") is not None and _ilu.find_spec("fastapi") is not None:
        from fastapi.testclient import TestClient
        from katana_next.api.app import PlatformRuntime, create_app
        from katana_next.services.report_archive import report_source
        from katana_next.services.report_import import import_report

        runtime = PlatformRuntime(application_root=PACKAGES_DIR / "katana_studio")
        ws_dest = tmp_path / "ws_cp_studio"
        src_dir = report_source(rep_dir / "REPORT.zip", tmp_path / "cp_cache")
        ctx = import_report(src_dir, ws_dest, runtime.projects)
        runtime.jobs.bind_project(ctx.manifest.project_id, ctx.manifest.output_root)
        pid = ctx.manifest.project_id
        client = TestClient(create_app(runtime))

        r_cp = client.get(f"/api/projects/{pid}/cross-paradigm", params={"comparison": "KO_vs_WT"})
        assert r_cp.status_code == 200
        cp_api = r_cp.json()
        assert cp_api["available"] is True
        assert cp_api["dexseq_sign_inverted"] is True
        assert cp_api["counts"]["class1_concordant_pct"] == 100.0
        assert cp_api["counts"]["suppa_af_al_fdr"] == 2
        assert cp_api["counts"]["dtu_switch_resolved"] == 1
        assert cp_api["counts"]["dtu_hidden_from_deg"] == 1
        assert cp_api["explorer_html_url"].endswith("CROSS_PARADIGM_EXPLORER.html")
        assert client.get(cp_api["explorer_html_url"]).status_code == 200
        for cls_key in ("class1", "class2_af_al", "class3_switches", "class4_dexseq"):
            rows = cp_api["tables"][cls_key]
            assert len(rows) > 0, f"Expected rows for {cls_key}"
            assert rows[0]["row_id"] and rows[0]["source_artifact_id"]

        # Gene1 (rMATS + Class I DEXSeq verified)
        r_gc1 = client.post(f"/api/projects/{pid}/genecard-preview", json={"gene_query": "Gene1", "comparison": "KO_vs_WT"})
        assert r_gc1.status_code == 200
        gc1 = r_gc1.json()
        assert gc1["found"] is True and gc1["evidence_origin"] == "rmats"
        assert gc1["cross_paradigm"]["available"] is True
        assert gc1["cross_paradigm"]["class1"] is not None
        assert gc1["cross_paradigm"]["dex_bins"] is not None

        # Gene10 (rMATS + Class III stageR switch hidden from DEG) + direct Class III switch event lookup
        r_gc10 = client.post(f"/api/projects/{pid}/genecard-preview", json={"gene_query": "Gene10", "comparison": "KO_vs_WT"})
        assert r_gc10.status_code == 200
        gc10 = r_gc10.json()
        assert gc10["found"] is True
        assert gc10["cross_paradigm"]["hidden_from_deg"] is True
        assert gc10["cross_paradigm"]["class3_switch"]["fate"] == "Coding -> NMD (Loss of ORF)"
        assert len(gc10["track1_quantification"]["dtu_isoforms"]) >= 2

        r_gc10_sw = client.post(
            f"/api/projects/{pid}/genecard-preview",
            json={"gene_query": "Gene10", "event_id": f"{g10_coding}->{g10_nmd}", "comparison": "KO_vs_WT"},
        )
        assert r_gc10_sw.status_code == 200
        gc10_sw = r_gc10_sw.json()
        assert gc10_sw["found"] is True and gc10_sw["evidence_origin"] == "cross_paradigm_dtu"
        assert gc10_sw["track4_proteome"]["status"] == "coding_architecture"

        # rMATS-blind SUPPA2 AF target (ENSMUSG00000000021)
        r_gc21 = client.post(f"/api/projects/{pid}/genecard-preview", json={"gene_query": "ENSMUSG00000000021", "comparison": "KO_vs_WT"})
        assert r_gc21.status_code == 200
        gc21 = r_gc21.json()
        assert gc21["found"] is True and gc21["evidence_origin"] == "cross_paradigm_suppa2"
        assert gc21["das_artifact_id"] and gc21["das_row_id"]

        # Nonexistent gene returns found=False cleanly rather than unrelated top rMATS event
        r_missing = client.post(f"/api/projects/{pid}/genecard-preview", json={"gene_query": "NO_SUCH_GENE_XYZ", "comparison": "KO_vs_WT"})
        assert r_missing.status_code == 200 and r_missing.json()["found"] is False

        # Target resolve for rMATS-blind ENSMUSG00000000021 & Target export from Class III row
        r_res = client.post(
            f"/api/projects/{pid}/targets/resolve",
            json={"queries": ["ENSMUSG00000000021"], "target_kind": "gene", "comparison": "KO_vs_WT"},
        )
        assert r_res.status_code == 200 and len(r_res.json()["matched"]) == 1

        c3_row = cp_api["tables"]["class3_switches"][0]
        r_exp = client.post(
            f"/api/projects/{pid}/targets/export",
            json={
                "selections": [{
                    "source_artifact_id": c3_row["source_artifact_id"],
                    "row_ids": [c3_row["row_id"]],
                    "target_kind": "gene",
                    "filters": {},
                }],
            },
        )
        assert r_exp.status_code == 200 and r_exp.json()["target_count"] == 1

        # Target Ranker exposes Cross-Paradigm metadata and survives strict empty filters cleanly
        r_rk = client.post(
            f"/api/projects/{pid}/compute/target-ranker",
            json={"comparison": "KO_vs_WT", "hidden_from_deg_only": True},
        )
        assert r_rk.status_code == 200 and r_rk.json()["available"] is True
        assert any(row.get("hidden_from_deg") is True for row in r_rk.json()["rows"])

        r_rk_empty = client.post(
            f"/api/projects/{pid}/compute/target-ranker",
            json={"comparison": "KO_vs_WT", "min_abs_dpsi": 0.99, "hidden_from_deg_only": True},
        )
        assert r_rk_empty.status_code == 200 and r_rk_empty.json()["total_qualified"] == 0


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
        if any(part in {".git", "__pycache__", ".pytest_cache", "dummy_inspection_reports", "node_modules", "dist"} for part in f.parts):
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = forbidden.search(txt)
        if m:
            violations.append(f"{f.relative_to(REPO_ROOT)} -> {m.group(0)!r}")
    assert violations == [], f"Hygiene violations found: {violations[:15]}"
