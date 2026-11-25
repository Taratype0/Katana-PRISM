"""Regression tests for multi-contrast hyphenated `rnasplicing/rmats/<treat>-<ctrl>` resolution."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PRISM_ROOT = Path(__file__).resolve().parent.parent
for _pkg in ("katana_studio", "splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace"):
    _p = str(PRISM_ROOT / "packages" / _pkg)
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _write_rmats_se(
    rdir: Path,
    *,
    n_events: int,
    n_sig: int,
    dpsi_sign: float,
    psi1_base: float,
    psi2_base: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    rdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n_events):
        is_sig = i < n_sig
        p1 = np.clip((psi1_base if is_sig else 0.50) + rng.normal(0.0, 0.02, 3), 0.05, 0.95)
        p2 = np.clip((psi2_base if is_sig else 0.50) + rng.normal(0.0, 0.02, 3), 0.05, 0.95)
        tot = np.full(3, 50, dtype=int)
        ijc1 = np.round(p1 * tot).astype(int)
        sjc1 = np.maximum(1, tot - ijc1)
        ijc2 = np.round(p2 * tot).astype(int)
        sjc2 = np.maximum(1, tot - ijc2)
        dpsi = round(float(dpsi_sign * abs(p1.mean() - p2.mean())), 4) if is_sig else 0.01
        base = 300 + i * 700
        g_idx = i + 1
        rows.append(
            {
                "ID": g_idx,
                "GeneID": f'"ENSMUSG{g_idx:011d}"',
                "geneSymbol": f'"Gene{g_idx}"',
                "chr": "chr1",
                "strand": "+",
                "exonStart_0base": base + 199,
                "exonEnd": base + 290,
                "upstreamES": base - 1,
                "upstreamEE": base + 90,
                "downstreamES": base + 419,
                "downstreamEE": base + 540,
                "ID.1": g_idx,
                "IJC_SAMPLE_1": ",".join(map(str, ijc1)),
                "SJC_SAMPLE_1": ",".join(map(str, sjc1)),
                "IJC_SAMPLE_2": ",".join(map(str, ijc2)),
                "SJC_SAMPLE_2": ",".join(map(str, sjc2)),
                "IncFormLen": 148,
                "SkipFormLen": 99,
                "PValue": 1e-5 if is_sig else 0.72,
                "FDR": 0.001 if is_sig else 0.85,
                "IncLevel1": ",".join(f"{x:.3f}" for x in p1),
                "IncLevel2": ",".join(f"{x:.3f}" for x in p2),
                "IncLevelDifference": dpsi,
            }
        )
    pd.DataFrame(rows).to_csv(rdir / "SE.MATS.JCEC.txt", sep="\t", index=False)


def _contrast_spec(cid: str, num: str, den: str) -> dict[str, object]:
    return {
        "contrast_id": cid,
        "numerator": {"label": num, "sample_ids": [f"{num}_{i}" for i in range(1, 4)]},
        "denominator": {"label": den, "sample_ids": [f"{den}_{i}" for i in range(1, 4)]},
    }


def test_resolve_rmats_dir_multi_contrast_hyphen_and_reversed(tmp_path: Path) -> None:
    from katana_studio.report_builder import _discover_rmats_contrasts, _resolve_rmats_dir_for_contrast

    rmats_root = tmp_path / "rnasplicing/rmats"
    _write_rmats_se(rmats_root / "A-B/rmats_post", n_events=6, n_sig=2, dpsi_sign=1.0, psi1_base=0.8, psi2_base=0.2, seed=1)
    _write_rmats_se(rmats_root / "C-D/rmats_post", n_events=10, n_sig=4, dpsi_sign=1.0, psi1_base=0.8, psi2_base=0.2, seed=2)
    _write_rmats_se(rmats_root / "F-E/rmats_post", n_events=8, n_sig=3, dpsi_sign=1.0, psi1_base=0.8, psi2_base=0.2, seed=3)

    specs = [
        _contrast_spec("A_vs_B", "A", "B"),
        _contrast_spec("C_vs_D", "C", "D"),
        _contrast_spec("E_vs_F", "E", "F"),
    ]
    rmats_map = _discover_rmats_contrasts(rmats_root, specs)

    dir_ab, inv_ab = _resolve_rmats_dir_for_contrast(rmats_map, specs[0], len(specs))
    dir_cd, inv_cd = _resolve_rmats_dir_for_contrast(rmats_map, specs[1], len(specs))
    dir_ef, inv_ef = _resolve_rmats_dir_for_contrast(rmats_map, specs[2], len(specs))

    assert dir_ab == rmats_root / "A-B/rmats_post" and inv_ab is False
    assert dir_cd == rmats_root / "C-D/rmats_post" and inv_cd is False
    assert dir_ef == rmats_root / "F-E/rmats_post" and inv_ef is True


def test_resolve_rmats_dir_forbids_silent_multi_contrast_fallback_and_allows_single(tmp_path: Path) -> None:
    from katana_studio.report_builder import _discover_rmats_contrasts, _resolve_rmats_dir_for_contrast

    rmats_root = tmp_path / "rnasplicing/rmats"
    _write_rmats_se(rmats_root / "A-B/rmats_post", n_events=6, n_sig=2, dpsi_sign=1.0, psi1_base=0.8, psi2_base=0.2, seed=4)

    multi_specs = [_contrast_spec("A_vs_B", "A", "B"), _contrast_spec("C_vs_D", "C", "D")]
    rmats_map_multi = _discover_rmats_contrasts(rmats_root, multi_specs)
    with pytest.raises(FileNotFoundError, match="No rMATS contrast directory matched 'C_vs_D'"):
        _resolve_rmats_dir_for_contrast(rmats_map_multi, multi_specs[1], len(multi_specs))

    single_specs = [_contrast_spec("CustomContrastName", "Trt", "Ctrl")]
    single_map = _discover_rmats_contrasts(rmats_root, single_specs)
    resolved_dir, inv = _resolve_rmats_dir_for_contrast(single_map, single_specs[0], len(single_specs))
    assert resolved_dir == rmats_root / "A-B/rmats_post" and inv is False


def test_splicemanifold_and_splicenet_resolve_hyphenated_bundle(tmp_path: Path) -> None:
    from splicemanifold.ps_das import load_rmats_study
    from splicenet.rmats_loader import extract_matrices_from_bundle_or_rmats

    bundle = tmp_path / "bundle"
    (bundle / "pipeline_info").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"sample": f"A_{i}", "condition": "A"} for i in range(1, 4)
        ]
        + [{"sample": f"B_{i}", "condition": "B"} for i in range(1, 4)]
        + [{"sample": f"C_{i}", "condition": "C"} for i in range(1, 4)]
        + [{"sample": f"D_{i}", "condition": "D"} for i in range(1, 4)]
    ).to_csv(bundle / "pipeline_info/samplesheet.valid.csv", index=False)
    pd.DataFrame(
        [
            {"contrast": "A_vs_B", "treatment": "A", "control": "B"},
            {"contrast": "C_vs_D", "treatment": "C", "control": "D"},
        ]
    ).to_csv(bundle / "pipeline_info/contrastsheet.valid.csv", index=False)

    _write_rmats_se(bundle / "rnasplicing/rmats/A-B/rmats_post", n_events=8, n_sig=2, dpsi_sign=1.0, psi1_base=0.80, psi2_base=0.20, seed=10)
    _write_rmats_se(bundle / "rnasplicing/rmats/C-D/rmats_post", n_events=14, n_sig=5, dpsi_sign=1.0, psi1_base=0.25, psi2_base=0.75, seed=20)

    ev_df, psi_sm, _, _, _ = load_rmats_study(bundle_dir=bundle)
    assert int((ev_df["comparison_id"] == "A_vs_B").sum()) == 8
    assert int((ev_df["comparison_id"] == "C_vs_D").sum()) == 14
    assert float(psi_sm["C_1"].dropna().iloc[0]) == pytest.approx(0.25, abs=0.08)

    psi_sn, _, _, _ = extract_matrices_from_bundle_or_rmats(bundle_dir=bundle)
    assert psi_sn.shape == (14, 12)
    assert int(psi_sn["A_1"].notna().sum()) == 8
    assert int(psi_sn["C_1"].notna().sum()) == 14
    assert float(psi_sn["A_1"].dropna().iloc[0]) == pytest.approx(0.80, abs=0.08)
    assert float(psi_sn["C_1"].dropna().iloc[0]) == pytest.approx(0.25, abs=0.08)


def test_run_bulk_two_contrasts_hyphenated_rmats_end_to_end(tmp_path: Path) -> None:
    from katana_studio.report_builder import run_bulk

    bundle = tmp_path / "bundle"
    (bundle / "pipeline_info").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"sample": f"A_{i}", "condition": "A"} for i in range(1, 4)]
        + [{"sample": f"B_{i}", "condition": "B"} for i in range(1, 4)]
        + [{"sample": f"C_{i}", "condition": "C"} for i in range(1, 4)]
        + [{"sample": f"D_{i}", "condition": "D"} for i in range(1, 4)]
    ).to_csv(bundle / "pipeline_info/samplesheet.valid.csv", index=False)
    pd.DataFrame(
        [
            {"contrast": "A_vs_B", "treatment": "A", "control": "B"},
            {"contrast": "C_vs_D", "treatment": "C", "control": "D"},
        ]
    ).to_csv(bundle / "pipeline_info/contrastsheet.valid.csv", index=False)

    _write_rmats_se(bundle / "rnasplicing/rmats/A-B/rmats_post", n_events=12, n_sig=3, dpsi_sign=1.0, psi1_base=0.85, psi2_base=0.15, seed=31)
    _write_rmats_se(bundle / "rnasplicing/rmats/C-D/rmats_post", n_events=19, n_sig=7, dpsi_sign=1.0, psi1_base=0.20, psi2_base=0.80, seed=32)

    outdir = tmp_path / "report_out"
    res = run_bulk(
        outdir=outdir,
        bundle=bundle,
        study_name="Hyphenated rMATS Two-Contrast Check",
        species="mouse",
        dummy=True,
    )
    assert res["status"] == "completed"

    das_ab = pd.read_parquet(outdir / "outputs/comparisons/A_vs_B/DAS-complete-original-and-report-flags.parquet")
    das_cd = pd.read_parquet(outdir / "outputs/comparisons/C_vs_D/DAS-complete-original-and-report-flags.parquet")
    assert len(das_ab) == 12
    assert len(das_cd) == 19
    assert int(das_ab["report_das"].sum()) == 3
    assert int(das_cd["report_das"].sum()) == 7

    psi_mat = pd.read_parquet(outdir / "outputs/results/sample_level_psi.parquet")
    assert float(psi_mat["A_1"].dropna().iloc[0]) == pytest.approx(0.85, abs=0.08)
    assert float(psi_mat["B_1"].dropna().iloc[0]) == pytest.approx(0.15, abs=0.08)
    assert float(psi_mat["C_1"].dropna().iloc[0]) == pytest.approx(0.20, abs=0.08)
    assert float(psi_mat["D_1"].dropna().iloc[0]) == pytest.approx(0.80, abs=0.08)
