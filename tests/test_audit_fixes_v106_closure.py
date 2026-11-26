"""Comprehensive v1.0.6 closure regression suite covering all v22 From-Source Audit software items:
- E03: Case-colliding headers & row-level biological unit alias conflicts/merging across preflight + all 3 bulk loaders
- E04: Reference lineage across raw, gunzipped FASTA, pipeline-filtered GTF, and native manifest.reference
- E05: Three-arm single-cell slice isolation (`Other` cells excluded BEFORE eligibility & network export) + scSpliceNet contrast flags
- E06: Strict numeric probability/effect domain checks & blind-depth fallback suppression when nominal P is absent
- E07: Enclosing-bundle within-arm sample slot verification across SpliceNet, SpliceManifold, and Katana-Studio
- A04: SpliceNet single-block exact-tie module rescue, subgroup step stability under jitter, and inverse-replicate donor weighting
- A05: Incomplete biological unit PERMANOVA suppression (API + CLI) & exact paired subsample-block permutation
- A06: Empty depth matrix rejection & zero-shared-pair inference suppression under explicit coverage fallback
- A08: Supervised & mixed-design PSI clustermap/PCA rendering without TypeError across SpliceManifold & Studio
- R05: Atomic staging in SpliceTrace & scSpliceTrace preserving prior valid outputs on failed retry while purging stale outputs on completion
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for pkg_dir in ("splicemanifold", "splicenet", "splicetrace", "scsplicetrace", "scsplicenet", "katana_studio"):
    p = ROOT / "packages" / pkg_dir
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from katana_next.science.psi_landscape import (  # noqa: E402
    compute_beyond_jutils_landscape,
    render_editorial_psi_clustermap,
    render_editorial_psi_pca,
)
from katana_studio.report_builder import run_bulk  # noqa: E402
from scsplicenet.cli import main as scsplicenet_main  # noqa: E402
from splicemanifold.cli import main as splicemanifold_main  # noqa: E402
from splicemanifold.manifold import (  # noqa: E402
    compute_splicing_manifold,
    render_psi_clustermap,
    render_psi_pca,
)
from splicemanifold.ps_das import load_rmats_study  # noqa: E402
from splicenet.network import compute_tukey_bicor_matrix, fit_splicenet  # noqa: E402
from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats  # noqa: E402
from splicetrace.cli import main as splicetrace_main  # noqa: E402
from splicetrace.continuous_rescue import run_bulk_continuous_rescue  # noqa: E402
from splicetrace.rmats_io import verify_bundle_reference_sha  # noqa: E402


def _write_simple_rmats_se(rdir: Path, p1_list: list[float], p2_list: list[float]) -> None:
    rdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in range(12):
        base = 1000 + idx * 500
        a1 = [round(min(0.95, max(0.05, v + 0.01 * (idx % 3))), 3) for v in p1_list]
        a2 = [round(min(0.95, max(0.05, v + 0.01 * (idx % 3))), 3) for v in p2_list]
        rows.append(
            {
                "ID": idx + 1,
                "GeneID": f"ENSG000001{idx:05d}",
                "geneSymbol": f"GENE{idx + 1}",
                "chr": "chr1",
                "strand": "+",
                "exonStart_0base": base + 100,
                "exonEnd": base + 180,
                "upstreamES": base,
                "upstreamEE": base + 50,
                "downstreamES": base + 250,
                "downstreamEE": base + 320,
                "IJC_SAMPLE_1": ",".join(str(int(round(x * 100))) for x in a1),
                "SJC_SAMPLE_1": ",".join(str(100 - int(round(x * 100))) for x in a1),
                "IJC_SAMPLE_2": ",".join(str(int(round(x * 100))) for x in a2),
                "SJC_SAMPLE_2": ",".join(str(100 - int(round(x * 100))) for x in a2),
                "IncFormLen": 129,
                "SkipFormLen": 50,
                "PValue": 0.0002 if idx < 4 else 0.65,
                "FDR": 0.004 if idx < 4 else 0.85,
                "IncLevel1": ",".join(f"{x:.3f}" for x in a1),
                "IncLevel2": ",".join(f"{x:.3f}" for x in a2),
                "IncLevelDifference": round(float(np.mean(a1) - np.mean(a2)), 4),
            }
        )
    pd.DataFrame(rows).to_csv(rdir / "SE.MATS.JCEC.txt", sep="\t", index=False)


def test_e03_case_collision_and_row_alias_contracts(tmp_path: Path) -> None:
    fa = tmp_path / "ref.fa"
    gtf = tmp_path / "ref.gtf"
    fa.write_text(">chr1\n" + "A" * 200 + "\n", encoding="utf-8")
    gtf.write_text('chr1\tsrc\texon\t1\t80\t.\t+\t.\tgene_id "G1"; transcript_id "T1";\n', encoding="utf-8")
    ct = tmp_path / "contrasts.csv"
    ct.write_text("contrast,treatment,control\nT_vs_C,T,C\n", encoding="utf-8")

    fqs = []
    for arm in ("T", "C"):
        for i in range(4):
            sid = f"{arm}{i + 1}"
            fq = tmp_path / f"{sid}.fastq.gz"
            with gzip.open(fq, "wt", encoding="utf-8") as fh:
                fh.write(f"@{sid}\n" + "A" * 50 + "\n+\n" + "I" * 50 + "\n")
            fqs.append((sid, arm, fq))

    # 1. Case-colliding headers (`subject` + `SUBJECT`) rejected in preflight AND all 3 bulk loaders
    coll_csv = tmp_path / "case_collision.csv"
    lines = ["sample,condition,strandedness,read_length,fastq_1,subject,SUBJECT"]
    for idx, (sid, arm, fq) in enumerate(fqs):
        lines.append(f"{sid},{arm},unstranded,50,{fq},{arm}u{idx // 2},{sid}")
    coll_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, "-B", str(ROOT / "bin/katana_validate_inputs.py"), "--samplesheet", str(coll_csv), "--contrasts", str(ct), "--outdir", str(tmp_path / "val_coll"), "--fasta", str(fa), "--gtf", str(gtf)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "case-insensitive" in proc.stderr.lower() or "collid" in proc.stderr.lower() or "duplicate" in proc.stderr.lower()

    rdir = tmp_path / "rmats/T-C/rmats_post"
    _write_simple_rmats_se(rdir, [0.8, 0.82, 0.79, 0.81], [0.2, 0.22, 0.19, 0.21])
    with pytest.raises(ValueError):
        extract_matrices_from_bundle_or_rmats(rmats_dir=rdir, samples_path=coll_csv, contrasts_path=ct)
    with pytest.raises(ValueError):
        load_rmats_study(rmats_dir=rdir, samples_file=coll_csv, contrasts_file=ct)
    with pytest.raises(ValueError):
        run_bulk(outdir=tmp_path / "rep_coll", rmats=rdir, samples=coll_csv, contrasts=ct, dummy=True)

    # 2. Mixed non-conflicting aliases (`subject` on odd rows, `biological_sample_id` on even rows) merge cleanly
    mix_csv = tmp_path / "mixed_alias.csv"
    lines_m = ["sample,condition,strandedness,read_length,fastq_1,subject,biological_sample_id"]
    for idx, (sid, arm, fq) in enumerate(fqs):
        subj, bio = (sid, "") if idx % 2 else ("", sid)
        lines_m.append(f"{sid},{arm},unstranded,50,{fq},{subj},{bio}")
    mix_csv.write_text("\n".join(lines_m) + "\n", encoding="utf-8")
    proc_m = subprocess.run(
        [sys.executable, "-B", str(ROOT / "bin/katana_validate_inputs.py"), "--samplesheet", str(mix_csv), "--contrasts", str(ct), "--outdir", str(tmp_path / "val_mix"), "--fasta", str(fa), "--gtf", str(gtf)],
        capture_output=True, text=True,
    )
    assert proc_m.returncode == 0
    _, meta_sn, _, _ = extract_matrices_from_bundle_or_rmats(rmats_dir=rdir, samples_path=mix_csv, contrasts_path=ct)
    assert meta_sn["biological_sample_id"].nunique() == 8


def test_e04_and_r05_reference_lineage_and_atomic_staging(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    (bundle / "provenance").mkdir(parents=True)
    rdir = bundle / "rmats/A_vs_B"
    _write_simple_rmats_se(rdir, [0.82, 0.80], [0.18, 0.20])

    fa = tmp_path / "genome.fa"
    fa_bytes = b">chr1\n" + (b"ACGT" * 25 + b"\n") * 100
    fa.write_bytes(fa_bytes)
    (tmp_path / "genome.fa.fai").write_text("chr1\t10000\t6\t100\t101\n", encoding="utf-8")
    fa_gz = tmp_path / "genome.fa.gz"
    fa_gz.write_bytes(gzip.compress(fa_bytes))
    wrong_fa = tmp_path / "foreign.fa"
    wrong_fa.write_bytes(b">chr1\n" + (b"CCCC" * 25 + b"\n") * 100)
    (tmp_path / "foreign.fa.fai").write_text("chr1\t10000\t6\t100\t101\n", encoding="utf-8")

    raw_gtf = tmp_path / "raw.gtf"
    raw_gtf.write_text(
        'chr1\tsrc\texon\t1\t400\t.\t+\t.\tgene_id "ENSG00000100000"; transcript_id "T1"; gene_type "protein_coding";\n'
        'chr1\tsrc\texon\t500\t700\t.\t+\t.\tgene_id "ENSG00000999999"; transcript_id "T2"; gene_type "processed_pseudogene";\n',
        encoding="utf-8",
    )
    sidecar = bundle / "provenance/reference_audit.json"
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "bin/katana_validate_reference.py"), "--fasta", str(fa_gz), "--gtf", str(raw_gtf), "--output", str(sidecar)],
        check=True,
    )
    filt_dir = tmp_path / "filtered_gtf"
    subprocess.run(
        [sys.executable, "-B", str(ROOT / "bin/katana_filter_analysis_gtf.py"), "--gtf", str(raw_gtf), "--outdir", str(filt_dir)],
        check=True,
    )
    analysis_gtf = filt_dir / "annotation.analysis.gtf"

    manifest_obj = {
        "comparisons": [{"comparison_id": "A_vs_B", "treatment": "A", "control": "B", "rmats_group1": "A", "rmats_group2": "B", "rmats_dir": "rmats/A_vs_B"}],
        "reference": json.loads(sidecar.read_text(encoding="utf-8")),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest_obj), encoding="utf-8")

    # Pipeline-decompressed FASTA + pipeline-filtered GTF accepted without reannotated flag
    res_filt = verify_bundle_reference_sha(bundle, fasta_path=fa, gtf_path=analysis_gtf)
    assert res_filt["verified"] is True
    assert res_filt["reannotated"] is False

    # Native manifest.reference alone still blocks foreign FASTA
    sidecar.unlink()
    with pytest.raises(ValueError, match="FASTA SHA256 mismatch"):
        verify_bundle_reference_sha(bundle, fasta_path=wrong_fa, gtf_path=analysis_gtf)

    # R05: Successful run followed by failed retry into SAME outdir preserves prior receipt & tables
    trace_out = tmp_path / "trace_out"
    rc = splicetrace_main([
        "--rmats", str(rdir), "--bundle", str(bundle), "--contrast-id", "A_vs_B",
        "--fasta", str(fa), "--gtf", str(analysis_gtf), "--out", str(trace_out),
        "--threads", "1", "--no-user-packs", "--no-extended-motifs",
    ])
    assert rc == 0
    receipt_before = json.loads((trace_out / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt_before["artifacts"]
    assert all((trace_out / rel).is_file() for rel in receipt_before["artifacts"])

    with pytest.raises(ValueError, match="FASTA SHA256 mismatch"):
        splicetrace_main([
            "--rmats", str(rdir), "--bundle", str(bundle), "--contrast-id", "A_vs_B",
            "--fasta", str(wrong_fa), "--gtf", str(analysis_gtf), "--out", str(trace_out),
            "--threads", "1", "--no-user-packs", "--no-extended-motifs",
        ])
    receipt_after = json.loads((trace_out / "run_receipt.json").read_text(encoding="utf-8"))
    assert receipt_after == receipt_before
    assert all((trace_out / rel).is_file() for rel in receipt_before["artifacts"])


def test_e05_three_arm_single_cell_slice_isolation_and_scsplicenet_flags(tmp_path: Path) -> None:
    cells = []
    for cond in ("T", "C", "Other"):
        for d in ("D1", "D2"):
            for k in range(4):
                cells.append({"cell_id": f"{cond}_{d}_{k}", "donor_id": d, "cell_type": "Neuron", "condition": cond, "qc_pass": "true"})
    md = pd.DataFrame(cells)
    cids = md["cell_id"].tolist()
    events = pd.DataFrame([
        {"event_id": f"E{i}", "gene_id": f"G{i}", "gene_symbol": f"SYM{i}", "event_type": "SE"}
        for i in range(16)
    ])
    # Events E0..E11 are ONLY observed in `Other` (NaN/0 in T and C); E12..E15 are observed in T and C
    psi_mat = pd.DataFrame(np.nan, index=events["event_id"], columns=cids)
    sup_mat = pd.DataFrame(0.0, index=events["event_id"], columns=cids)
    rng = np.random.default_rng(42)
    for i in range(12):
        other_cols = [c for c in cids if c.startswith("Other_")]
        psi_mat.loc[f"E{i}", other_cols] = np.where(np.arange(len(other_cols)) < 4, 0.85, 0.15)
        sup_mat.loc[f"E{i}", other_cols] = 25.0
    for i in range(12, 16):
        tc_cols = [c for c in cids if not c.startswith("Other_")]
        psi_mat.loc[f"E{i}", tc_cols] = rng.uniform(0.2, 0.8, len(tc_cols))
        sup_mat.loc[f"E{i}", tc_cols] = 20.0

    md.to_csv(tmp_path / "cells.tsv", sep="\t", index=False)
    events.to_csv(tmp_path / "events.tsv", sep="\t", index=False)
    psi_mat.T.to_csv(tmp_path / "psi.tsv", sep="\t", index_label="cell_id")
    sup_mat.T.to_csv(tmp_path / "sup.tsv", sep="\t", index_label="cell_id")

    slice_out = tmp_path / "sliced"
    subprocess.run(
        [
            sys.executable, "-B", str(ROOT / "bin/katana_sc_slice_contrast.py"),
            "--psi", str(tmp_path / "psi.tsv"),
            "--support", str(tmp_path / "sup.tsv"),
            "--events", str(tmp_path / "events.tsv"),
            "--metadata", str(tmp_path / "cells.tsv"),
            "--population-col", "cell_type",
            "--population-value", "Neuron",
            "--group-col", "condition",
            "--numerator", "T",
            "--denominator", "C",
            "--min-cells", "4",
            "--outdir", str(slice_out),
        ],
        check=True,
    )
    sliced_cells = pd.read_csv(slice_out / "sc_cells.tsv", sep="\t")
    sliced_ev = pd.read_csv(slice_out / "sc_events.tsv", sep="\t")
    assert set(sliced_cells["condition"]) == {"T", "C"}
    assert set(sliced_ev["event_id"]) == {"E12", "E13", "E14", "E15"}

    # Direct scSpliceNet CLI contrast flags also exclude `Other`
    psi_cells_rows = psi_mat.T
    sup_cells_rows = sup_mat.T
    psi_cells_rows.to_csv(tmp_path / "usage_cr.tsv", sep="\t", index_label="cell_id")
    sup_cells_rows.to_csv(tmp_path / "sup_cr.tsv", sep="\t", index_label="cell_id")
    events[["event_id", "gene_id", "gene_symbol"]].to_csv(tmp_path / "eg.tsv", sep="\t", index=False)
    net_out = tmp_path / "scnet_tc"
    assert scsplicenet_main([
        "--usage", str(tmp_path / "usage_cr.tsv"),
        "--support", str(tmp_path / "sup_cr.tsv"),
        "--metadata", str(tmp_path / "cells.tsv"),
        "--event-genes", str(tmp_path / "eg.tsv"),
        "--group-col", "condition",
        "--numerator", "T",
        "--denominator", "C",
        "-o", str(net_out),
    ]) == 0
    aud = pd.read_csv(net_out / "scsplicenet_event_audit.tsv", sep="\t").set_index("event_id")
    assert not aud.loc[[f"E{i}" for i in range(12)], "qc_eligible"].any()


def test_e06_strict_probability_and_blind_depth_fallback_status(tmp_path: Path) -> None:
    from splicetrace.library import CuratedMotifLibrary, load_bundled_human_rbp_library

    rng = np.random.default_rng(99)
    n = 20
    seq_bytes = "".join(rng.choice(["A", "C", "G", "T"], size=32000)).encode("ascii")
    fa_p = tmp_path / "mini_e06.fa"
    fa_p.write_bytes(b">chr1\n" + seq_bytes + b"\n")
    (tmp_path / "mini_e06.fa.fai").write_text(f"chr1\t{len(seq_bytes)}\t6\t{len(seq_bytes)}\t{len(seq_bytes) + 1}\n", encoding="utf-8")

    rows = []
    for i in range(n):
        base = 900 + i * 1400
        rows.append({
            "event_id": f"SE:chr1:+:{base+100}:{base+200}:{base-700}:{base-600}:{base+700}:{base+800}",
            "gene_id": f"G{i}",
            "gene_symbol": f"SYM{i}",
            "event_type": "SE",
            "chr": "chr1",
            "strand": "+",
            "exonStart_0base": base + 100,
            "exonEnd": base + 200,
            "upstreamES": base - 700,
            "upstreamEE": base - 600,
            "downstreamES": base + 700,
            "downstreamEE": base + 800,
            "dpsi": float(rng.uniform(-0.4, 0.4)),
            "baseline_PSI": float(rng.uniform(0.2, 0.8)),
            "depth": 80.0,
        })
    base_df = pd.DataFrame(rows)
    full_lib = load_bundled_human_rbp_library()
    m_id = full_lib.rbp_to_models["PTBP1"][0]
    mini_lib = CuratedMotifLibrary(name="mini_e06", models={m_id: full_lib.models[m_id]}, rbp_to_models={"PTBP1": (m_id,)}, excluded_models=())

    # 1. Text token in PValue must raise ValueError
    bad_p = base_df.assign(fdr=0.5, PValue=["xyz"] + [0.5] * (n - 1))
    with pytest.raises(ValueError, match="Non-numeric"):
        run_bulk_continuous_rescue(bad_p, fa_p, mini_lib, n_permutations=29)

    # 2. Completely missing PValue and FDR -> insufficient_probability_evidence (NOT complete with 20 foregrounds!)
    res_no_p = run_bulk_continuous_rescue(base_df, fa_p, mini_lib, n_permutations=29)
    assert res_no_p.diagnostics["status"] == "insufficient_probability_evidence"
    assert res_no_p.diagnostics["inference_valid"] is False
    assert res_no_p.diagnostics["foreground_events"] == 0


def test_e07_enclosing_bundle_within_arm_order_across_all_three_loaders(tmp_path: Path) -> None:
    bdir = tmp_path / "bundle_order"
    (bdir / "pipeline_info").mkdir(parents=True)
    rdir = bdir / "rnasplicing/rmats/Unt-KD/rmats_post"
    # True slot order: Unt1=0.10, Unt2=0.30, Unt3=0.50; KD1=0.60, KD2=0.75, KD3=0.90
    _write_simple_rmats_se(rdir, [0.10, 0.30, 0.50], [0.60, 0.75, 0.90])
    (bdir / "manifest.json").write_text(
        json.dumps({
            "rmats_sample_order": {"Unt": ["Unt1", "Unt2", "Unt3"], "KD": ["KD1", "KD2", "KD3"]},
            "comparisons": [{"comparison_id": "Unt-KD", "rmats_group1": "Unt", "rmats_group2": "KD"}],
        }),
        encoding="utf-8",
    )
    # Resort samplesheet in REVERSE within-arm order!
    s_rev = bdir / "pipeline_info/samplesheet.valid.csv"
    pd.DataFrame([
        {"sample_id": "Unt3", "condition": "Unt"},
        {"sample_id": "Unt2", "condition": "Unt"},
        {"sample_id": "Unt1", "condition": "Unt"},
        {"sample_id": "KD3", "condition": "KD"},
        {"sample_id": "KD2", "condition": "KD"},
        {"sample_id": "KD1", "condition": "KD"},
    ]).to_csv(s_rev, index=False)
    c_rev = bdir / "pipeline_info/contrastsheet.valid.csv"
    pd.DataFrame([{"contrast": "KD_vs_Unt", "treatment": "KD", "control": "Unt"}]).to_csv(c_rev, index=False)

    u_net, _, _, _ = extract_matrices_from_bundle_or_rmats(rmats_dir=rdir, samples_path=s_rev, contrasts_path=c_rev)
    _, u_man, _, _, _ = load_rmats_study(rmats_dir=rdir, samples_file=s_rev, contrasts_file=c_rev)
    rep_dir = tmp_path / "studio_ord"
    run_bulk(outdir=rep_dir, rmats=rdir, samples=s_rev, contrasts=c_rev, dummy=True)
    u_stu = pd.read_parquet(rep_dir / "outputs/results/sample_level_psi.parquet").set_index("event_id")

    assert float(u_net["Unt1"].iloc[0]) < float(u_net["Unt3"].iloc[0])
    assert float(u_man["Unt1"].iloc[0]) < float(u_man["Unt3"].iloc[0])
    assert float(u_stu["Unt1"].iloc[0]) < float(u_stu["Unt3"].iloc[0])
    prov = json.loads((rep_dir / "outputs/qc/rmats-input-provenance.json").read_text(encoding="utf-8"))["contrasts"][0]
    assert prov["sample_order_verified"] is True
    assert prov["arm1_samples"] == ["Unt1", "Unt2", "Unt3"]


def test_a04_splicenet_exact_tie_jitter_and_donor_weights() -> None:
    rng = np.random.default_rng(20261126)
    base = np.round(np.linspace(0.18, 0.24, 40) * 2000) / 2000
    x = np.tile(base[:, None], (1, 30))
    x[:, -2:] += 0.55
    px = pd.DataFrame(x, index=[f"E{e}" for e in range(40)], columns=[f"S{s}" for s in range(30)])
    fit_df, bc, tom = fit_splicenet(px, backend="python", return_matrices=True, min_module_size=8)
    assert bc[0, 1] > 0.999 and tom[0, 1] > 0.999
    assert int(fit_df["module"].gt(0).sum()) == 40

    for sd in (1e-4, 1e-3, 0.01):
        arr = x + rng.normal(0.0, sd, x.shape)
        cc = compute_tukey_bicor_matrix(arr)
        tri = np.triu_indices(len(arr), 1)
        assert float(np.median(cc[tri])) > 0.95

    don = np.linspace(-1, 1, 11)
    b_vals = []
    for rep in (2, 6):
        xx = np.append(0.50 + 0.15 * don, np.repeat(0.62, rep))
        yy = np.append(0.50 + 0.15 * don, np.repeat(0.38, rep))
        d_ids = [f"D{j}" for j in range(11)] + ["D11"] * rep
        b_vals.append(float(compute_tukey_bicor_matrix(np.stack((xx, yy)), donor_ids=d_ids)[0, 1]))
    assert b_vals[0] == pytest.approx(b_vals[1], abs=1e-6)


def test_a05_a06_a08_manifold_and_studio_guards(tmp_path: Path) -> None:
    rng = np.random.default_rng(740113)
    # A05: One blank donor suppresses P in both API and CLI
    D = np.repeat(np.arange(8), 4)
    sids = [f"S{i}" for i in range(32)]
    X = rng.binomial(500, rng.beta(8, 8, size=(30, 8))[:, D]) / 500.0
    psi = pd.DataFrame(X, index=[f"E{i}" for i in range(30)], columns=sids)
    dep = pd.DataFrame(500.0, index=psi.index, columns=psi.columns)
    meta_partial = [
        {"sample_id": s, "condition": "A" if d < 4 else "B", "biological_sample_id": "" if idx == 0 else f"D{d}"}
        for idx, (s, d) in enumerate(zip(sids, D))
    ]
    for fn in (compute_splicing_manifold, compute_beyond_jutils_landscape):
        res = fn(psi, meta_partial, depth_matrix=dep, run_umap=False)
        assert res["permanova_p"] is None
        assert res["permanova_scope"] == "incomplete_biological_units"
        assert res["permanova_inference_valid"] is False

    psi_p = tmp_path / "psi.tsv"
    dep_p = tmp_path / "dep.tsv"
    smp_p = tmp_path / "smp.tsv"
    psi.to_csv(psi_p, sep="\t")
    dep.to_csv(dep_p, sep="\t")
    pd.DataFrame(meta_partial).to_csv(smp_p, sep="\t", index=False)
    cli_out = tmp_path / "cli_partial"
    assert splicemanifold_main(["bulk", "--psi-matrix", str(psi_p), "--support-matrix", str(dep_p), "--samples", str(smp_p), "-o", str(cli_out)]) == 0
    rec = json.loads((cli_out / "run_receipt.json").read_text(encoding="utf-8"))
    assert rec["permanova_p"] is None
    assert rec["permanova_scope"] == "incomplete_biological_units"

    # A05: Nested paired technical clones (8 donors x 8 libraries = 4+4 clones) match exact 2^8 subsample-block permutation
    D64 = np.repeat(np.arange(8), 8)
    tech = (np.arange(64) % 8) < 4
    obs_id = D64 * 2 + tech
    X64 = rng.binomial(500, rng.beta(6, 6, size=(30, 16))[:, obs_id]) / 500.0
    psi64 = pd.DataFrame(X64, index=[f"E{i}" for i in range(30)], columns=[f"C{i}" for i in range(64)])
    dep64 = pd.DataFrame(500.0, index=psi64.index, columns=psi64.columns)
    meta64 = [{"sample_id": f"C{i}", "condition": str(bool(t)), "biological_sample_id": f"D{d}"} for i, (d, t) in enumerate(zip(D64, tech))]
    res64 = compute_splicing_manifold(psi64, meta64, depth_matrix=dep64, run_umap=False)
    assert res64["permanova_scope"] == "within_biological_unit"
    assert res64["permanova_p"] > 0.05

    # A06: Explicit empty depth matrix rejected; zero-shared-pairs fallback suppresses inferential P
    empty_dep = pd.DataFrame(index=pd.Index([], name="event"), columns=sids)
    for fn in (compute_splicing_manifold, compute_beyond_jutils_landscape):
        res_empty = fn(psi, meta_partial, depth_matrix=empty_dep, run_umap=False)
        assert res_empty["available"] is False

    x_sp = np.tile([0.95, 0.05, np.nan, np.nan], (8, 1))
    x_sp[4:] = np.tile([np.nan, np.nan, 0.95, 0.05], (4, 1))
    psi_sp = pd.DataFrame(x_sp, index=[f"E{i}" for i in range(8)], columns=["S0", "S1", "S2", "S3"])
    dep_sp = pd.DataFrame(np.isfinite(x_sp) * 300.0, index=psi_sp.index, columns=psi_sp.columns)
    meta_sp = [{"sample_id": f"S{i}", "condition": "A" if i < 2 else "B", "biological_sample_id": f"U{i}"} for i in range(4)]
    for fn in (compute_splicing_manifold, compute_beyond_jutils_landscape):
        sp_res = fn(psi_sp, meta_sp, depth_matrix=dep_sp, min_present_fraction=0.8, allow_coverage_fallback=True, run_umap=False)
        assert sp_res["available"] is True
        assert sp_res["zero_shared_pairs"] == 4
        assert sp_res["permanova_p"] is None
        assert sp_res["permanova_inference_valid"] is False

    # A08: Supervised & mixed-design clustermap + PCA render cleanly in both SpliceManifold and Studio
    sup_set = {f"E{i}" for i in range(10)}
    for tag, fn, hm_fn, pca_fn in (
        ("sm", compute_splicing_manifold, render_psi_clustermap, render_psi_pca),
        ("st", compute_beyond_jutils_landscape, render_editorial_psi_clustermap, render_editorial_psi_pca),
    ):
        sup_res = fn(psi, meta_partial, depth_matrix=dep, supervised_event_ids=sup_set, run_umap=False)
        assert sup_res["permanova_p"] is None
        hm_p = tmp_path / f"{tag}_sup_hm.png"
        pca_p = tmp_path / f"{tag}_sup_pca.png"
        hm_fn(sup_res, hm_p)
        pca_fn(sup_res, pca_p)
        assert hm_p.is_file() and pca_p.is_file()


def test_terminal_exon_support_and_selenocysteine_modes(tmp_path: Path) -> None:
    """Verify 3' terminal extension requires continuous annotated exon support and GTF Selenocysteine is honored."""
    from splicetrace.protein_consequences import trace_protein_and_nmd_consequences
    from splicetrace.rna_forms import IndexedFasta

    # Build a synthetic '+' strand chromosome:
    # Exon 1: [10, 40) (30 nt: ATG + 9 sense codons)
    # Alt Exon A: [60, 80) (20 nt, shifts reading frame!)
    # Exon 2 (clipped in TX1): [100, 120) (20 nt, no stop in shifted frame)
    # Intronic gap [120, 160) contains an intronic TAA at [130, 133) NOT covered by any exon!
    # Sibling TX2 has a separate downstream exon [160, 190) (so gene_3p_max=190, but [100, 133) is NOT covered by one exon!)
    # Sibling TX3 has complete Exon 1 + Exon 2 with native in-frame stop before Alt Exon A (5'-UTR / short intact ORF)
    seq = list("A" * 300)
    # Put ATG at 10..13, then GCA repeats
    def put(pos: int, s: str) -> None:
        seq[pos : pos + len(s)] = list(s)

    put(10, "ATG" + "GCA" * 9)       # [10, 40): 10 codons (no stop)
    put(60, "G" + "CCA" * 6 + "C")   # [60, 80): 20 nt frameshift exon (no stop)
    put(100, "GCA" * 6 + "TA")       # [100, 120): ends with 'TA', in Exc (30+20=50 nt) put stop at 100..103? Let's place stop cleanly:
    put(112, "TAA")                  # Exc frame: 30 nt + 12 nt = 42 nt -> TAA at [112, 115) (14 aa)! Inc frame: 30+20+12=62 -> 62%3=2 (out of frame, misses TAA!)
    put(131, "TAA")                  # Intronic TAA at [131, 134): would be hit ONLY if [100, 120) were stretched across intron [120, 160)!
    # Also build a Selenocysteine gene GSEC on chr1 [210, 270): ATG + GGA + TGA(Sec) + GCA + TAA
    put(210, "ATGGGATGAGCATAA")      # [210, 225): M - G - U(216..219) - A - *(222..225) -> 4 aa with Sec, 2 aa without!

    fa_p = tmp_path / "ref.fa"
    fa_p.write_bytes((">chr1\n" + "".join(seq) + "\n").encode("ascii"))
    (tmp_path / "ref.fa.fai").write_bytes(b"chr1\t300\t6\t300\t301\n")

    gtf_lines = [
        'chr1\ttest\ttranscript\t11\t120\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding"; tag "Ensembl_canonical";',
        'chr1\ttest\texon\t11\t40\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\tCDS\t11\t40\t.\t+\t0\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\tstart_codon\t11\t13\t.\t+\t0\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\texon\t61\t80\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\texon\t101\t120\t.\t+\t.\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\tCDS\t101\t112\t.\t+\t0\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\tstop_codon\t113\t115\t.\t+\t0\tgene_id "G1"; transcript_id "TX1"; gene_name "G1"; transcript_type "protein_coding";',
        # TX2 has a disjoint downstream exon [161, 190] so gene_3p_max = 190, leaving [120, 160) unannotated intron
        'chr1\ttest\ttranscript\t11\t190\t.\t+\t.\tgene_id "G1"; transcript_id "TX2"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\texon\t11\t40\t.\t+\t.\tgene_id "G1"; transcript_id "TX2"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\tCDS\t11\t40\t.\t+\t0\tgene_id "G1"; transcript_id "TX2"; gene_name "G1"; transcript_type "protein_coding";',
        'chr1\ttest\texon\t161\t190\t.\t+\t.\tgene_id "G1"; transcript_id "TX2"; gene_name "G1"; transcript_type "protein_coding";',
        # GSEC transcript with annotated Selenocysteine at 1-based [217, 219] (0-based [216, 219))
        'chr1\ttest\ttranscript\t211\t240\t.\t+\t.\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding"; tag "Ensembl_canonical";',
        'chr1\ttest\texon\t211\t240\t.\t+\t.\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
        'chr1\ttest\tCDS\t211\t222\t.\t+\t0\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
        'chr1\ttest\tstart_codon\t211\t213\t.\t+\t0\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
        'chr1\ttest\tSelenocysteine\t217\t219\t.\t+\t0\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
        'chr1\ttest\tstop_codon\t223\t225\t.\t+\t0\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
        'chr1\ttest\texon\t251\t270\t.\t+\t.\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
        'chr1\ttest\texon\t281\t295\t.\t+\t.\tgene_id "GSEC"; transcript_id "TXSEC"; gene_name "GSEC"; transcript_type "protein_coding";',
    ]
    gtf_p = tmp_path / "ref.gtf"
    gtf_p.write_text("\n".join(gtf_lines) + "\n", encoding="utf-8")

    ev = pd.DataFrame([
        {
            "event_id": "SE:chr1:+:60:80:10:40:100:120",
            "gene_id": "G1",
            "gene_symbol": "G1",
            "event_type": "SE",
            "chr": "chr1",
            "strand": "+",
            "exonStart_0base": 60,
            "exonEnd": 80,
            "upstreamES": 10,
            "upstreamEE": 40,
            "downstreamES": 100,
            "downstreamEE": 120,
            "dpsi": 0.25,
        },
        {
            "event_id": "SE:chr1:+:250:270:210:240:280:295",
            "gene_id": "GSEC",
            "gene_symbol": "GSEC",
            "event_type": "SE",
            "chr": "chr1",
            "strand": "+",
            "exonStart_0base": 250,
            "exonEnd": 270,
            "upstreamES": 210,
            "upstreamEE": 240,
            "downstreamES": 280,
            "downstreamEE": 295,
            "dpsi": 0.20,
        },
    ])
    out_cond = trace_protein_and_nmd_consequences(ev, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, selenocysteine_mode="conditional")
    r_g1 = out_cond[out_cond["gene_id"] == "G1"].iloc[0]
    # Intronic TAA at [131, 134) MUST NOT be used because no single exon spans [100, 134)!
    assert r_g1["nmd_switch_class"] == "incomplete_3prime_CDS_or_nonstop"

    r_sec = out_cond[out_cond["gene_id"] == "GSEC"].iloc[0]
    assert int(r_sec["aa_len_inclusion"]) == 2 and "Conditional selenocysteine" in str(r_sec["functional_consequence_summary"])
    assert "4 vs 4 aa" in str(r_sec["functional_consequence_summary"])

    out_rec = trace_protein_and_nmd_consequences(ev, gtf_p, fa_p, indexed_fasta_cls=IndexedFasta, selenocysteine_mode="recode")
    r_sec_rec = out_rec[out_rec["gene_id"] == "GSEC"].iloc[0]
    assert int(r_sec_rec["aa_len_inclusion"]) == 4 and int(r_sec_rec["aa_len_exclusion"]) == 4

