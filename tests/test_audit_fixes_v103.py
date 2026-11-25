"""Regression tests verifying Katana-PRISM v1.0.3 deep re-audit remediations."""
from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PRISM_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace", "katana_studio"):
    _p = str(PRISM_ROOT / "packages" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _build_fixture(spec: list[tuple[object, str, int, object]], events: list[str] | None = None):
    if events is None:
        events = ["E"]
    rows = []
    for donor, arm, n, val in spec:
        for _ in range(n):
            rows.append((f"cell_{len(rows)}", donor, arm, np.broadcast_to(val, (len(events),))))
    md = pd.DataFrame([(cid, d, arm) for cid, d, arm, _ in rows], columns=["cell_id", "donor_id", "condition"])
    u = pd.DataFrame(np.vstack([v for *_, v in rows]), index=md["cell_id"], columns=events)
    sup = u.notna().astype(float) * 100.0
    genes = pd.Series([f"gene_{e}" for e in events], index=events)
    return u, md, genes, sup


def _write_mini_fasta(tmp_dir: Path) -> Path:
    rng = np.random.default_rng(17)
    seq = "".join(rng.choice(list("ACGT"), size=1000))
    fa = tmp_dir / "genome.fa"
    fa.write_text(f">chr1\n{seq}\n", encoding="utf-8")
    (tmp_dir / "genome.fa.fai").write_text("chr1\t1000\t6\t1000\t1001\n", encoding="utf-8")
    return fa


def _run_cli_contrast(tmp_dir: Path, u: pd.DataFrame, md: pd.DataFrame, genes: pd.Series, sup: pd.DataFrame, *, design_mode: str = "auto") -> tuple[pd.DataFrame, dict]:
    from scsplicetrace.cli import main as sc_main

    run_dir = tmp_dir / "cli_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    fa = _write_mini_fasta(run_dir)
    ev_rows = []
    for idx, eid in enumerate(u.columns):
        ev_rows.append({
            "event_id": str(eid), "gene_id": str(genes.loc[eid]), "gene_symbol": str(genes.loc[eid]),
            "chr": "chr1", "strand": "+", "event_type": "SE",
            "upstreamES": 50, "upstreamEE": 100,
            "exonStart_0base": 200 + idx * 3, "exonEnd": 300 + idx * 3,
            "downstreamES": 450, "downstreamEE": 550,
        })
    pd.DataFrame(ev_rows).to_csv(run_dir / "events.tsv", sep="\t", index=False)
    md.to_csv(run_dir / "md.tsv", sep="\t", index=False)
    u.to_csv(run_dir / "usage.tsv", sep="\t")
    sup.to_csv(run_dir / "support.tsv", sep="\t")
    out_dir = run_dir / "out"
    rc = sc_main([
        "--events", str(run_dir / "events.tsv"),
        "--fasta", str(fa),
        "--psi", str(run_dir / "usage.tsv"),
        "--metadata", str(run_dir / "md.tsv"),
        "--support", str(run_dir / "support.tsv"),
        "--out", str(out_dir),
        "--numerator", "T",
        "--denominator", "C",
        "--design-mode", design_mode,
        "--no-user-packs",
        "--threads", "1",
        "--permutations", "49",
    ])
    assert rc == 0
    return pd.read_csv(out_dir / "scsplicetrace_event_contrasts.tsv", sep="\t"), json.loads((out_dir / "scsplicetrace_summary.json").read_text(encoding="utf-8"))


def test_sc_one_shared_one_disjoint_never_pools_cells(tmp_path: Path) -> None:
    """v16 Blocker 2.1: 1 shared + 1 disjoint/arm must return insufficient_multi_donor_design in BOTH API and CLI."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast, discover_sc_rbp_programs

    spec = [("lowShared", "T", 8, 0.1), ("lowShared", "C", 80, 0.1), ("highT", "T", 80, 0.9), ("highC", "C", 8, 0.9)]
    u, md, genes, sup = _build_fixture(spec)
    c_api, d_api, info_api = compute_sc_splicing_contrast(u, md, genes, support=sup, numerator_group="T", denominator_group="C")
    assert info_api["inference_unit"] == "insufficient_multi_donor_design"
    assert info_api["eligible_events"] == 0
    assert d_api is None
    assert bool(c_api.iloc[0]["eligible"]) is False
    assert pd.isna(c_api.iloc[0]["raw_dpsi"])

    c_cli, info_cli = _run_cli_contrast(tmp_path / "one_shared", u, md, genes, sup)
    assert info_cli["inference_unit"] == "insufficient_multi_donor_design"
    assert info_cli["eligible_events"] == 0
    assert info_cli["status"] == "insufficient_eligible_events"
    assert bool(c_cli.iloc[0]["eligible"]) is False

    # Confirm 35-gene motif level also returns insufficient_eligible_events rather than false P=0.001
    ev35 = [f"e{k}" for k in range(35)]
    motif_m = np.random.default_rng(246).uniform(-1.0, 1.0, len(ev35))
    low = 0.5 - 0.4 * motif_m
    high = 0.5 + 0.4 * motif_m
    spec35 = [("lowShared", "T", 8, low), ("lowShared", "C", 80, low), ("highT", "T", 80, high), ("highC", "C", 8, high)]
    u35, md35, g35, s35 = _build_fixture(spec35, ev35)
    c35, d35, _ = compute_sc_splicing_contrast(u35, md35, g35, support=s35, numerator_group="T", denominator_group="C")
    cov35 = pd.DataFrame({"event_id": ev35, "event_type": "SE", "exon_GC": 0.5, "flank_GC": 0.5, "log_length": 4.0})
    r35 = discover_sc_rbp_programs(c35, motif_m[:, None, None], np.array(["R"]), ["region"], cov35, donor_delta_df=d35, n_permutations=99)
    assert r35.diagnostics["status"] == "insufficient_eligible_events"


def test_sc_two_shared_cell_imbalance_null_api_and_cli(tmp_path: Path) -> None:
    """v16 Blocker 2.1: 2 shared donors with true 0 donor delta & cell-yield imbalance -> dPSI=0, P=1 in BOTH API & CLI."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    spec = [("oneLow", "T", 8, 0.1), ("oneLow", "C", 80, 0.1), ("oneHigh", "T", 80, 0.9), ("oneHigh", "C", 8, 0.9)]
    u, md, genes, sup = _build_fixture(spec)
    c_api, _, info_api = compute_sc_splicing_contrast(u, md, genes, support=sup, numerator_group="T", denominator_group="C")
    c_cli, info_cli = _run_cli_contrast(tmp_path / "two_shared", u, md, genes, sup)
    for tag, df, inf in (("API", c_api, info_api), ("CLI", c_cli, info_cli)):
        assert inf["inference_unit"] == "stratified_paired_2donor", tag
        assert bool(df.iloc[0]["eligible"]) is True, tag
        assert abs(float(df.iloc[0]["raw_dpsi"])) < 1e-10, tag
        assert float(df.iloc[0]["p_value"]) > 0.99, tag


def test_sc_cli_api_identical_on_two_paired_plus_two_disjoint(tmp_path: Path) -> None:
    """v16 Blocker 2.2: CLI and API must choose identical donor subsets and sign under default design_mode='auto'."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    spec = [(f"p{d}", a, 10, 0.8 if a == "T" else 0.2) for d in range(2) for a in ("T", "C")]
    spec += [(f"x{d}{a}", a, 10, 0.2 if a == "T" else 0.8) for d in range(2) for a in ("T", "C")]
    u, md, genes, sup = _build_fixture(spec)

    c_api, _, info_api = compute_sc_splicing_contrast(u, md, genes, support=sup, numerator_group="T", denominator_group="C")
    c_cli, info_cli = _run_cli_contrast(tmp_path / "mixed", u, md, genes, sup)
    assert info_api["inference_unit"] == info_cli["inference_unit"] == "stratified_paired_2donor"
    assert abs(float(c_api.iloc[0]["raw_dpsi"]) - 0.6) < 1e-6
    assert abs(float(c_cli.iloc[0]["raw_dpsi"]) - 0.6) < 1e-6
    assert info_api["unpaired_donors_excluded"] == info_cli["unpaired_donors_excluded"] == 4

    # Explicit design_mode="unpaired" consistently evaluates disjoint donors in both
    c_unp, _, info_unp = compute_sc_splicing_contrast(
        u, md, genes, support=sup, numerator_group="T", denominator_group="C", design_mode="unpaired"
    )
    assert info_unp["inference_unit"] == "partially_paired_disjoint_unpaired"
    assert abs(float(c_unp.iloc[0]["raw_dpsi"]) - (-0.6)) < 1e-6


def test_sc_blank_and_nan_donor_ids_dropped() -> None:
    """v16 Blocker 2.3: Missing/blank/NaN donor IDs must never fabricate a 'nan' pseudo-donor."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast, discover_sc_rbp_programs

    ev35 = [f"e{k}" for k in range(35)]
    motif = np.linspace(-1.0, 1.0, len(ev35))
    cov = pd.DataFrame({"event_id": ev35, "event_type": "SE", "exon_GC": 0.5, "flank_GC": 0.5, "log_length": 4.0})
    spec_5 = [(f"d{i}", a, 8, 0.5 + (0.06 if a == "T" else -0.06) * motif) for i in range(5) for a in ("T", "C")]
    spec_bad = spec_5 + [(np.nan, a, 8, 0.5 + (0.06 if a == "T" else -0.06) * motif) for a in ("T", "C")] + [("", "T", 8, 0.5 + 0.06 * motif)]
    u, md, genes, sup = _build_fixture(spec_bad, ev35)
    c, d_df, info = compute_sc_splicing_contrast(u, md, genes, support=sup, numerator_group="T", denominator_group="C")
    assert info["paired_donors"] == 5
    assert info["cells_excluded_missing_donor_id"] == 24
    assert "nan" not in d_df.index
    res = discover_sc_rbp_programs(c, motif[:, None, None], np.array(["R"]), ["region"], cov, donor_delta_df=d_df)
    assert abs(float(res.rbp_ranking["rbp_P_FWER"].iloc[0]) - 0.0625) < 1e-12


def test_sc_paired_eligibility_invariant_under_within_donor_arm_swaps() -> None:
    """v16 Blocker 2.4: Paired eligibility must be strictly invariant under arbitrary within-donor arm swaps."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast

    spec = [(str(d), a, 2, 0.4 if a == "T" else 0.6) for d in range(3) for a in ("T", "C")]
    spec += [(str(d), a, 2, 0.6 if d in (3, 4) and a == "T" else np.nan) for d in range(3, 6) for a in ("T", "C")]
    u, md, genes, sup = _build_fixture(spec)
    orig, _, _ = compute_sc_splicing_contrast(u, md, genes, support=sup, numerator_group="T", denominator_group="C")

    md_flip = md.copy()
    mask4 = md_flip["donor_id"] == "4"
    md_flip.loc[mask4, "condition"] = md_flip.loc[mask4, "condition"].map({"T": "C", "C": "T"})
    flipped, _, _ = compute_sc_splicing_contrast(u, md_flip, genes, support=sup, numerator_group="T", denominator_group="C")
    assert bool(orig["eligible"].iloc[0]) == bool(flipped["eligible"].iloc[0])
    assert int(orig["n_paired_donors"].iloc[0]) == int(flipped["n_paired_donors"].iloc[0]) == 3


def test_splicenet_psi_bounds_and_missing_r_backend_fail_loudly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """v16 Blocker 3: Out-of-range PSI must raise ValueError and missing Rscript with backend='r-wgcna' must fail loudly."""
    from splicenet.network import fit_splicenet
    from splicenet.qc_and_selection import unit_robust_audit

    cols = [f"S{i}" for i in range(12)]
    meta = pd.DataFrame({"sample_id": cols, "biological_sample_id": [f"D{i // 2}" for i in range(12)]})
    good = pd.DataFrame(np.tile(np.linspace(0.1, 0.9, 12), (25, 1)), index=[f"E{k}" for k in range(25)], columns=cols)
    bad_hi = good.copy()
    bad_hi.iloc[0, 0] = 1.4
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        unit_robust_audit(bad_hi, meta)
    bad_lo = good.copy()
    bad_lo.iloc[0, 0] = -0.1
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        unit_robust_audit(bad_lo, meta)

    monkeypatch.setattr("splicenet.network.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError, match="Rscript was not found"):
        fit_splicenet(good, tmp_path / "r_fail", power=6, min_module_size=5, backend="r-wgcna")


def test_studio_cross_paradigm_fallback_counts_untruncated_rows(tmp_path: Path) -> None:
    """v16 Section 5: Missing summary JSON fallback must count all full TSV rows (450) while previewing 300."""
    import importlib.util as _ilu
    if _ilu.find_spec("duckdb") is None:
        pytest.skip("duckdb not installed")

    from katana_next.api.app import PlatformRuntime
    from katana_next.services.report_import import import_report
    from katana_next.services.studio_interactive import cross_paradigm_explorer

    rep_src = tmp_path / "src_report"
    cp_dir = rep_src / "outputs/cross_paradigm"
    cp_dir.mkdir(parents=True)
    rows_450 = pd.DataFrame([
        {
            "comparison_id": "A_vs_B",
            "gene_id": f"ENSG{i:011d}",
            "gene_symbol": f"G{i}",
            "event_id": f"SE_{i}",
            "concordant_fixed": (i % 2 == 0),
        }
        for i in range(450)
    ])
    tsv_p = cp_dir / "A_vs_B_class1.tsv"
    rows_450.to_csv(tsv_p, sep="\t", index=False)

    manifest = {
        "schema": "katana.approved_report/v1",
        "project_id": "proj_cp_450",
        "mode": "bulk",
        "display_project": {
            "project_id": "proj_cp_450",
            "name": "CP 450 Fallback Test",
            "contrasts": [{
                "contrast_id": "A_vs_B",
                "numerator": {"label": "A", "sample_ids": ["s1"]},
                "denominator": {"label": "B", "sample_ids": ["s2"]},
            }],
            "samples": [
                {"sample_id": "s1", "biological_sample_id": "s1", "condition": "A"},
                {"sample_id": "s2", "biological_sample_id": "s2", "condition": "B"},
            ],
            "annotation": {"genome_build": "GRCh38"},
            "metadata": {"species": "human"},
        },
        "complete_results": [
            {
                "source": "outputs/cross_paradigm/A_vs_B_class1.tsv",
                "contrast_id": "A_vs_B",
                "analysis_id": "cross_paradigm.class1",
                "name": "Class 1",
                "chapter": 4,
                "metadata": {"comparison_ids": ["A_vs_B"]},
            }
        ],
        "panels": [],
        "source_files": [],
    }
    (rep_src / "report-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (rep_src / "REPORT.html").write_text("<html><body>test</body></html>", encoding="utf-8")

    rt = PlatformRuntime(application_root=PRISM_ROOT / "packages/katana_studio")
    ctx = import_report(rep_src, tmp_path / "ws_dest", rt.projects)
    res = cross_paradigm_explorer(ctx, comparison="A_vs_B")
    assert res["available"] is True
    assert len(res["tables"]["class1"]) == 300
    assert res["counts"]["class1_events"] == 450
    assert res["counts"]["class1_genes"] == 450
    assert res["counts"]["class1_concordant"] == 225


def test_sc_sharp_null_6donor_25common_invariance() -> None:
    """Under sharp-null within-donor label flips on 6 donors (25 common + 15 optional events), eligibility is invariant and FWER <= 2/64."""
    from scsplicetrace.contrast_discovery import compute_sc_splicing_contrast, discover_sc_rbp_programs

    rng = np.random.default_rng(20261125)
    ev40 = [f"e{k}" for k in range(40)]
    cov = pd.DataFrame({
        "event_id": ev40,
        "event_type": "SE",
        "exon_GC": np.linspace(0.35, 0.65, 40),
        "flank_GC": np.linspace(0.40, 0.60, 40),
        "log_length": np.linspace(3.5, 5.0, 40),
    })
    motif = np.zeros((40, 2, 1), dtype=float)
    motif[:18, 0, 0] = rng.normal(0.8, 0.2, 18)
    motif[18:, 1, 0] = rng.normal(0.8, 0.2, 22)

    specs = []
    psi_base = np.full(40, 0.5)
    for d in range(6):
        specs.append((f"d{d}", "T", 4, psi_base.copy()))
        specs.append((f"d{d}", "C", 4, psi_base.copy()))
    u, md, genes, sup = _build_fixture(specs, ev40)
    # Donors 0..1 paired on all 40 events (< min_paired_donors=3 on optional 15); d2 one-sided T on optional 15; d3..d5 paired only on 25 common events.
    for d in range(6):
        t_cols = md[(md["donor_id"] == f"d{d}") & (md["condition"] == "T")]["cell_id"].tolist()
        c_cols = md[(md["donor_id"] == f"d{d}") & (md["condition"] == "C")]["cell_id"].tolist()
        shift = (0.12 if d % 2 == 0 else -0.10) * (np.arange(40) < 25)
        u.loc[t_cols, :] = np.clip(0.5 + shift[None, :], 0.05, 0.95)
        u.loc[c_cols, :] = np.clip(0.5 - shift[None, :], 0.05, 0.95)
        if d < 2:
            opt_shift = 0.15 if d == 0 else -0.05
            u.loc[t_cols, ev40[25:]] = 0.5 + opt_shift
            u.loc[c_cols, ev40[25:]] = 0.5 - opt_shift
        elif d == 2:
            # One-sided extra coverage in T only on optional 15 events -> must NOT trigger asymmetric pooling or flip eligibility!
            u.loc[t_cols, ev40[25:]] = 0.85
            u.loc[c_cols, ev40[25:]] = np.nan
            sup.loc[c_cols, ev40[25:]] = 0.0
        else:
            u.loc[t_cols + c_cols, ev40[25:]] = np.nan
            sup.loc[t_cols + c_cols, ev40[25:]] = 0.0

    el_sets = set()
    calls = []
    for mask in range(64):
        md_m = md.copy()
        for d in range(6):
            if (mask >> d) & 1:
                sel = md_m["donor_id"] == f"d{d}"
                md_m.loc[sel, "condition"] = md_m.loc[sel, "condition"].map({"T": "C", "C": "T"})
        c_df, d_df, _ = compute_sc_splicing_contrast(u, md_m, genes, support=sup, numerator_group="T", denominator_group="C")
        el_sets.add(tuple(c_df["eligible"].astype(bool).tolist()))
        res = discover_sc_rbp_programs(c_df, motif, np.array(["R0", "R1"]), ["region"], cov, donor_delta_df=d_df, n_permutations=499)
        if float(res.rbp_ranking["rbp_P_FWER"].min()) <= 0.05:
            calls.append(mask)
    assert len(el_sets) == 1
    assert next(iter(el_sets)).count(True) == 25
    assert len(calls) <= 3

