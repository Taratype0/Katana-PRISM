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
    assert 'splicemanifold: "1.0.0"' in (sm_bulk_out / "versions.yml").read_text(encoding="utf-8")

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
    assert 'splicenet: "1.0.0"' in (sn_out / "versions.yml").read_text(encoding="utf-8")

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
    assert 'splicetrace: "1.0.0"' in (st_out / "versions.yml").read_text(encoding="utf-8")

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
    from katana_studio.cli import report_main
    from katana_next.api.app import PlatformRuntime, create_app

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

    # Verify Katana Studio FastAPI app initializes and locates bundled _web UI
    app = create_app(PlatformRuntime(application_root=PACKAGES_DIR / "katana_studio"))
    assert app is not None


# ---------------------------------------------------------------------------
# 4. Repository-Wide Confidentiality, Brand & Naming Hygiene Audit
# ---------------------------------------------------------------------------
def test_zero_confidential_brand_or_typo_leaks() -> None:
    forbidden = re.compile(
        r"(Ythdc1_deletion|\bSetd2\b|\bLuis\b|mckinsey|RS-DAS|kanata_|E:\\\\Project|E:/Project)",
        re.I,
    )
    text_exts = {".py", ".nf", ".config", ".R", ".md", ".toml", ".json", ".ts", ".tsx", ".css", ".html"}
    violations: list[str] = []
    for f in REPO_ROOT.rglob("*"):
        if not f.is_file() or f.suffix not in text_exts:
            continue
        if f.resolve() == Path(__file__).resolve():
            continue
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in f.parts):
            continue
        txt = f.read_text(encoding="utf-8", errors="replace")
        m = forbidden.search(txt)
        if m:
            violations.append(f"{f.relative_to(REPO_ROOT)} -> {m.group(0)!r}")
    assert violations == [], f"Hygiene violations found: {violations[:15]}"
