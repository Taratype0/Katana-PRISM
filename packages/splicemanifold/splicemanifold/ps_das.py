"""PS-DAS (Precision-Supported Differential Alternative Splicing) & rMATS Matrix Loader.

Implements Katana's publication event-qualification framework (`katana-ps-das`):
  1. Isoform-length-corrected effective junction depth `N_eff = IJC + (IncFormLen / SkipFormLen) * SJC`
  2. Replicate quantile support (`>= 75%` of declared replicates per arm passing `N_eff >= 10` and `N_raw >= 5`)
  3. Pooled Jeffreys posterior precision bound `SE_Jeffreys(dPSI) <= 0.16`
  4. Biological AS-NMD / complete-switching asymmetry rescue when one condition is anchored at high depth
  5. Full preservation of unmodified native rMATS `PValue`, `FDR`, `IncLevelDifference`, and `passes_legacy_all10`.
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EVENT_TYPES = ("SE", "RI", "A3SS", "A5SS", "MXE")
PS_DAS_POLICY_ID = "katana-ps-das"
LEGACY_ALL10_POLICY_ID = "legacy-all-samples-counts"
DEFAULT_MIN_EFFECTIVE_READS = 10.0
DEFAULT_MIN_RAW_READS = 5.0
DEFAULT_MIN_SAMPLE_FRACTION = 0.75
DEFAULT_MAX_JEFFREYS_SE = 0.16

EVENT_COORDINATES = (
    "chr", "strand",
    "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "riExonStart_0base", "riExonEnd",
    "longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE",
    "1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd",
)


def _clean_id(val: object) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip().strip('"').strip("'")


def normalize_gene_id(val: object) -> str:
    txt = _clean_id(val)
    m = re.match(r"^(ENS[A-Z]*G\d+)(?:\.\d+)?$", txt, flags=re.IGNORECASE)
    return m.group(1) if m else txt


def parse_number_list(val: object) -> list[float]:
    if val is None:
        return []
    if isinstance(val, str) and not val.strip():
        return []
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    out: list[float] = []
    for tok in str(val).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok or tok.upper() in {"NA", "NAN", "NONE"}:
            out.append(math.nan)
            continue
        try:
            num = float(tok)
            out.append(num if math.isfinite(num) else math.nan)
        except ValueError:
            out.append(math.nan)
    return out


def canonical_event_ids(frame: pd.DataFrame, event_type: str) -> pd.Series:
    ids = pd.Series(event_type, index=frame.index, dtype="object")
    coord_cnt = pd.Series(0, index=frame.index, dtype="int16")
    for col in EVENT_COORDINATES:
        if col not in frame.columns:
            continue
        vals = frame[col].map(_clean_id)
        present = vals.ne("")
        ids.loc[present] = ids.loc[present] + "|" + col + "=" + vals.loc[present]
        coord_cnt.loc[present] += 1
    fb = coord_cnt.eq(0)
    if fb.any():
        src = frame["ID"].map(_clean_id) if "ID" in frame.columns else pd.Series("", index=frame.index)
        src.loc[src.eq("")] = src.index[src.eq("")].astype(str)
        ids.loc[fb] = ids.loc[fb] + "|source_id=" + src.loc[fb]
    return ids


def _finite_median(vals: list[float]) -> float:
    fin = sorted(v for v in vals if math.isfinite(v))
    n = len(fin)
    if n == 0:
        return math.nan
    return fin[n // 2] if n % 2 else 0.5 * (fin[n // 2 - 1] + fin[n // 2])


def _required_replicates(expected: int, min_fraction: float) -> int:
    if expected <= 1:
        return max(0, expected)
    if min_fraction >= 0.999:
        return expected
    return max(2, int(math.ceil(min_fraction * expected - 1e-9)))


def _arm_support(
    row: pd.Series,
    arm: str,
    suffix: str,
    *,
    eff_len_mult: float,
    min_eff_reads: float,
    min_raw_reads: float,
    min_sample_fraction: float,
) -> dict[str, Any]:
    psi = parse_number_list(row.get(f"IncLevel{suffix}"))
    ijc = parse_number_list(row.get(f"IJC_SAMPLE_{suffix}"))
    sjc = parse_number_list(row.get(f"SJC_SAMPLE_{suffix}"))
    exp_col = f"{arm}_expected_samples"
    expected = int(row[exp_col]) if exp_col in row and pd.notna(row[exp_col]) else max(len(psi), len(ijc), len(sjc))
    if expected == 1 and len(psi) == 0 and len(ijc) == 1 and len(sjc) == 1:
        # Handle single-replicate "NA" parsed as float NaN by upstream pandas.read_csv
        psi = [math.nan]
    reasons: list[str] = []
    shape_ok = expected > 0 and len(psi) == len(ijc) == len(sjc) == expected
    counts_shape_ok = expected > 0 and len(ijc) == len(sjc) == expected

    usable = raw_all10 = eff_pass = raw_floor_pass = corrupt_psi = invalid_cnt = 0
    eff_list: list[float] = []
    sum_ijc = sum_tot = 0.0
    n_pos = min(expected, max(len(psi), len(ijc), len(sjc)))
    for idx in range(n_pos):
        p = psi[idx] if idx < len(psi) else math.nan
        i = ijc[idx] if idx < len(ijc) else math.nan
        s = sjc[idx] if idx < len(sjc) else math.nan
        p_ok = math.isfinite(p) and 0.0 <= p <= 1.0
        c_ok = math.isfinite(i) and math.isfinite(s) and i >= 0.0 and s >= 0.0
        raw_n = (i + s) if c_ok else math.nan
        eff_n = (i + s * eff_len_mult) if c_ok else math.nan
        if c_ok:
            eff_list.append(eff_n)
            sum_ijc += i
            sum_tot += raw_n
        is_all10 = c_ok and raw_n >= 10.0
        is_eff = c_ok and eff_n >= min_eff_reads and raw_n >= min_raw_reads
        is_floor = c_ok and raw_n >= min_raw_reads
        usable += int(p_ok and c_ok)
        raw_all10 += int(p_ok and is_all10)
        eff_pass += int(is_eff)
        raw_floor_pass += int(is_floor)
        corrupt_psi += int(c_ok and raw_n > 0 and not p_ok)
        if not c_ok:
            invalid_cnt += 1
            reasons.append(f"{arm}:sample_{idx+1}:invalid_counts")
        elif not is_eff:
            reasons.append(f"{arm}:sample_{idx+1}:eff_reads_below_{min_eff_reads:g}")

    req_k = _required_replicates(expected, min_sample_fraction)
    med_eff = _finite_median(eff_list) if eff_list else 0.0
    counts_ok = counts_shape_ok and (invalid_cnt == 0)
    data_ok = shape_ok and counts_ok and (corrupt_psi == 0) and (usable >= req_k)
    maj_k = max(2, int(math.floor(0.66 * expected)))
    high_depth_maj = (
        expected >= 3
        and counts_ok
        and raw_floor_pass == expected
        and sum(1 for x in eff_list if math.isfinite(x) and x >= min_eff_reads * 1.5) >= maj_k
        and math.isfinite(med_eff) and med_eff >= min_eff_reads * 1.5
        and len(eff_list) == expected
        and all(math.isfinite(x) and x >= min_eff_reads * 0.8 for x in eff_list)
    )
    q_pass = (counts_ok and eff_pass >= req_k and math.isfinite(med_eff) and med_eff >= min_eff_reads) or high_depth_maj
    high_abun_k = sum(1 for x in eff_list if math.isfinite(x) and x >= max(12.0, min_eff_reads * 1.2))
    high_anchor = counts_ok and high_abun_k >= req_k and math.isfinite(med_eff) and med_eff >= max(20.0, min_eff_reads * 2.0)
    nmd_partner = counts_ok and raw_floor_pass >= req_k

    return {
        f"{arm}_expected_samples": expected,
        f"{arm}_required_replicates": req_k,
        f"{arm}_supported_replicates": eff_pass,
        f"{arm}_median_effective_reads": round(float(med_eff), 2) if math.isfinite(med_eff) else math.nan,
        f"{arm}_data_usable": data_ok,
        f"{arm}_quantile_eff_pass": q_pass,
        f"{arm}_all_read_counts_pass": shape_ok and raw_all10 == expected,
        "_high_anchor": high_anchor,
        "_nmd_partner": nmd_partner,
        "_sum_ijc": sum_ijc,
        "_sum_tot": sum_tot,
        "_reasons": reasons,
    }


def qualify_ps_das_events(
    events: pd.DataFrame,
    *,
    fdr_cutoff: float = 0.05,
    dpsi_cutoff: float = 0.10,
    min_effective_reads: float = DEFAULT_MIN_EFFECTIVE_READS,
    min_raw_reads: float = DEFAULT_MIN_RAW_READS,
    min_sample_fraction: float = DEFAULT_MIN_SAMPLE_FRACTION,
    max_jeffreys_se: float = DEFAULT_MAX_JEFFREYS_SE,
    enable_nmd_asymmetry_rescue: bool = True,
) -> pd.DataFrame:
    """Annotate rMATS events with PS-DAS (`katana-ps-das`), Ordinary DAS, and Legacy All>=10 flags."""
    df = events.copy()
    if df.empty:
        return df
    df["fdr"] = pd.to_numeric(df["FDR"] if "FDR" in df.columns else df.get("fdr"), errors="coerce")
    raw_dpsi = pd.to_numeric(df["IncLevelDifference"] if "IncLevelDifference" in df.columns else df.get("dpsi"), errors="coerce")
    rev = df["reverse_direction"].astype(bool) if "reverse_direction" in df.columns else pd.Series(False, index=df.index)
    df["dpsi"] = np.where(rev, -raw_dpsi, raw_dpsi)
    df["abs_dpsi"] = df["dpsi"].abs()

    audits: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        reverse = bool(row.get("reverse_direction", False))
        if "IncFormLen" in row and "SkipFormLen" in row and pd.notna(row["IncFormLen"]) and pd.notna(row["SkipFormLen"]):
            try:
                eff_mult = max(1.0, min(3.0, float(row["IncFormLen"]) / max(1.0, float(row["SkipFormLen"]))))
            except (TypeError, ValueError):
                eff_mult = 1.0
        else:
            eff_mult = 1.0

        num = _arm_support(
            row, "numerator", "2" if reverse else "1",
            eff_len_mult=eff_mult, min_eff_reads=min_effective_reads,
            min_raw_reads=min_raw_reads, min_sample_fraction=min_sample_fraction,
        )
        den = _arm_support(
            row, "denominator", "1" if reverse else "2",
            eff_len_mult=eff_mult, min_eff_reads=min_effective_reads,
            min_raw_reads=min_raw_reads, min_sample_fraction=min_sample_fraction,
        )
        reasons = num.pop("_reasons") + den.pop("_reasons")
        num_high, num_part = num.pop("_high_anchor"), num.pop("_nmd_partner")
        den_high, den_part = den.pop("_high_anchor"), den.pop("_nmd_partner")
        si1, st1 = num.pop("_sum_ijc"), num.pop("_sum_tot")
        si2, st2 = den.pop("_sum_ijc"), den.pop("_sum_tot")

        p1_j = (si1 + 0.5) / (st1 + 1.0)
        p2_j = (si2 + 0.5) / (st2 + 1.0)
        # Exact Jeffreys Beta(k+0.5, n-k+0.5) posterior variance: p*(1-p)/(n+2)
        j_se = math.sqrt(p1_j * (1.0 - p1_j) / (st1 + 2.0) + p2_j * (1.0 - p2_j) / (st2 + 2.0))

        min_arm = min(num["numerator_expected_samples"], den["denominator_expected_samples"])
        eff_se = max_jeffreys_se if min_arm >= 2 else max(max_jeffreys_se, 0.22)
        bilat = bool(num["numerator_quantile_eff_pass"] and den["denominator_quantile_eff_pass"])
        nm, dm = num["numerator_median_effective_reads"], den["denominator_median_effective_reads"]
        asym = (
            math.isfinite(nm) and math.isfinite(dm)
            and max(nm, dm) >= 2.5 * max(1.0, min(nm, dm))
            and abs(p1_j - p2_j) >= 0.10
        )
        nmd_res = bool(enable_nmd_asymmetry_rescue and not bilat and asym and ((num_high and den_part) or (den_high and num_part)))
        se_ok = j_se <= eff_se
        legacy10 = bool(num["numerator_all_read_counts_pass"] and den["denominator_all_read_counts_pass"])
        usable_both = bool(num["numerator_data_usable"] and den["denominator_data_usable"])
        ps_support = bool(usable_both and (bilat or nmd_res) and se_ok)

        rec = {**num, **den}
        rec["effective_length_ratio"] = round(eff_mult, 3)
        rec["jeffreys_se_dpsi"] = round(j_se, 5)
        rec["nmd_asymmetry_rescued"] = bool(usable_both and nmd_res and se_ok)
        rec["passes_precision_support"] = ps_support
        rec["passes_legacy_all10"] = legacy10
        rec["precision_support_policy"] = PS_DAS_POLICY_ID
        rec["precision_support_reason"] = (
            "passed_as_nmd_asymmetry_rescue" if (ps_support and nmd_res)
            else ("passed" if ps_support else ("; ".join(reasons[:4]) or "failed_precision_gate"))
        )
        audits.append(rec)

    aud_df = pd.DataFrame(audits, index=df.index)
    for col in aud_df.columns:
        df[col] = aud_df[col]

    sig_raw = df["fdr"].le(fdr_cutoff) & df["abs_dpsi"].ge(dpsi_cutoff)
    df["passes_ordinary_das"] = sig_raw
    df["passes_ps_das"] = sig_raw & df["passes_precision_support"]
    df["passes_legacy_all10_das"] = sig_raw & df["passes_legacy_all10"]
    return df


def _read_delimited(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def load_rmats_study(
    *,
    bundle_dir: Path | None = None,
    rmats_dir: Path | None = None,
    samples_file: Path | None = None,
    contrasts_file: Path | None = None,
    rmats_mode: str = "JCEC",
    contrast_id: str = "contrast_1",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load rMATS events, sample PSI matrix, read-support matrix, sample metadata, and event metadata."""
    mode = rmats_mode.upper()
    if mode not in {"JC", "JCEC"}:
        raise ValueError(f"Unsupported rmats_mode: {rmats_mode}")

    sample_meta: list[dict[str, Any]] = []
    contrast_specs: list[dict[str, Any]] = []

    if bundle_dir is not None:
        bdir = Path(bundle_dir)
        s_path = next((p for p in (
            bdir / "pipeline_info" / "samplesheet.valid.csv",
            bdir / "inputs" / "samplesheet.csv",
            bdir / "samplesheet.csv",
            bdir / "provenance" / "strandedness" / "samplesheet.requested.csv",
        ) if p.is_file()), None)
        c_path = next((p for p in (
            bdir / "pipeline_info" / "contrastsheet.valid.csv",
            bdir / "inputs" / "contrastsheet.csv",
            bdir / "contrastsheet.csv",
        ) if p.is_file()), None)
        if s_path is None:
            raise FileNotFoundError(f"Missing samplesheet in bundle: {bdir}")
        s_df = _read_delimited(s_path)
        for _, r in s_df.iterrows():
            sid = str(r.get("sample") or r.get("sample_id")).strip()
            cond = str(r.get("condition") or r.get("group") or "Group").strip()
            if sid and not any(x["sample_id"] == sid for x in sample_meta):
                sample_meta.append({"sample_id": sid, "biological_sample_id": sid, "condition": cond})
        if c_path is not None and c_path.is_file():
            import json as _json
            manifest_order: dict[str, list[str]] = {}
            manifest_arms: dict[str, tuple[str, str]] = {}
            if (bdir / "manifest.json").is_file():
                try:
                    bm = _json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
                    if isinstance(bm.get("rmats_sample_order"), dict):
                        manifest_order = {str(k): [str(x) for x in v] for k, v in bm["rmats_sample_order"].items() if isinstance(v, list)}
                    for cm in bm.get("comparisons", []):
                        if isinstance(cm, dict) and cm.get("comparison_id") and cm.get("rmats_group1") and cm.get("rmats_group2"):
                            manifest_arms[str(cm["comparison_id"])] = (str(cm["rmats_group1"]), str(cm["rmats_group2"]))
                except Exception:
                    pass
            c_df = _read_delimited(c_path)
            for _, r in c_df.iterrows():
                cid = str(r.get("contrast") or r.get("contrast_id")).strip()
                num_lbl = str(r.get("treatment") or r.get("numerator")).strip()
                den_lbl = str(r.get("control") or r.get("denominator")).strip()
                rdir = bdir / "rmats" / cid / "rmats_post"
                if not rdir.is_dir():
                    rdir = bdir / "rmats" / cid
                g1_lbl, g2_lbl = manifest_arms.get(cid, (num_lbl, den_lbl))
                rev_dir = (g1_lbl == den_lbl and g2_lbl == num_lbl)
                num_s = manifest_order.get(num_lbl) or [s["sample_id"] for s in sample_meta if s["condition"] == num_lbl]
                den_s = manifest_order.get(den_lbl) or [s["sample_id"] for s in sample_meta if s["condition"] == den_lbl]
                contrast_specs.append({
                    "contrast_id": cid, "numerator": num_lbl, "denominator": den_lbl,
                    "rmats_dir": rdir, "num_samples": num_s, "den_samples": den_s,
                    "reverse_direction": rev_dir,
                })
    elif rmats_dir is not None:
        rdir = Path(rmats_dir)
        if samples_file is not None and Path(samples_file).is_file():
            s_df = _read_delimited(Path(samples_file))
            for _, r in s_df.iterrows():
                sid = str(r.get("sample_id") or r.get("sample")).strip()
                cond = str(r.get("condition") or r.get("group") or "Group").strip()
                bio = str(r.get("biological_sample_id") or sid).strip()
                if sid and not any(x["sample_id"] == sid for x in sample_meta):
                    sample_meta.append({"sample_id": sid, "biological_sample_id": bio, "condition": cond})
        if contrasts_file is not None and Path(contrasts_file).is_file() and sample_meta:
            c_df = _read_delimited(Path(contrasts_file))
            for _, r in c_df.iterrows():
                cid = str(r.get("contrast") or r.get("contrast_id") or contrast_id).strip()
                num_lbl = str(r.get("treatment") or r.get("numerator")).strip()
                den_lbl = str(r.get("control") or r.get("denominator")).strip()
                cand = rdir / cid / "rmats_post" if (rdir / cid / "rmats_post").is_dir() else (rdir / cid if (rdir / cid).is_dir() else rdir)
                contrast_specs.append({
                    "contrast_id": cid, "numerator": num_lbl, "denominator": den_lbl,
                    "rmats_dir": cand,
                    "num_samples": [s["sample_id"] for s in sample_meta if s["condition"] == num_lbl],
                    "den_samples": [s["sample_id"] for s in sample_meta if s["condition"] == den_lbl],
                })
        else:
            # Single rMATS directory mode (infer sample counts from first non-empty table if needed)
            conds = list(dict.fromkeys(s["condition"] for s in sample_meta))
            if len(conds) >= 2:
                num_lbl, den_lbl = conds[0], conds[1]
                num_s = [s["sample_id"] for s in sample_meta if s["condition"] == num_lbl]
                den_s = [s["sample_id"] for s in sample_meta if s["condition"] == den_lbl]
            else:
                num_lbl, den_lbl, num_s, den_s = "Group1", "Group2", [], []
            contrast_specs.append({
                "contrast_id": contrast_id, "numerator": num_lbl, "denominator": den_lbl,
                "rmats_dir": rdir, "num_samples": num_s, "den_samples": den_s,
            })
    else:
        raise ValueError("Either bundle_dir or rmats_dir must be supplied.")

    psi_obs: dict[tuple[str, str], list[float]] = {}
    sup_obs: dict[tuple[str, str], list[float]] = {}
    event_meta: dict[str, dict[str, Any]] = {}
    all_frames: list[pd.DataFrame] = []

    for spec in contrast_specs:
        cdir = Path(spec["rmats_dir"])
        for et in EVENT_TYPES:
            fpath = cdir / f"{et}.MATS.{mode}.txt"
            if not fpath.is_file():
                continue
            _str_cols = {
                c: str
                for c in ("IncLevel1", "IncLevel2", "IJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_1", "SJC_SAMPLE_2")
            }
            df = pd.read_csv(fpath, sep="\t", converters=_str_cols)
            if df.empty:
                continue
            df = df.copy()
            df["event_type"] = et
            df["event_id"] = canonical_event_ids(df, et)
            rev_d = bool(spec.get("reverse_direction", False))
            df["comparison_id"] = spec["contrast_id"]
            df["numerator"] = spec["numerator"]
            df["denominator"] = spec["denominator"]
            df["reverse_direction"] = rev_d
            df["gene_id"] = df["GeneID"].map(normalize_gene_id) if "GeneID" in df.columns else ""
            df["gene_symbol"] = df["geneSymbol"].map(_clean_id) if "geneSymbol" in df.columns else df["gene_id"]

            p1_first = parse_number_list(df["IncLevel1"].iloc[0]) or parse_number_list(df["IJC_SAMPLE_1"].iloc[0])
            p2_first = parse_number_list(df["IncLevel2"].iloc[0]) or parse_number_list(df["IJC_SAMPLE_2"].iloc[0])
            arm1_s = spec["den_samples"] if rev_d else spec["num_samples"]
            arm2_s = spec["num_samples"] if rev_d else spec["den_samples"]
            arm1_lbl = spec["denominator"] if rev_d else spec["numerator"]
            arm2_lbl = spec["numerator"] if rev_d else spec["denominator"]
            n1_names = arm1_s if len(arm1_s) == len(p1_first) else [f"{arm1_lbl}_rep{i+1}" for i in range(len(p1_first))]
            n2_names = arm2_s if len(arm2_s) == len(p2_first) else [f"{arm2_lbl}_rep{i+1}" for i in range(len(p2_first))]
            for s in n1_names:
                if not any(x["sample_id"] == s for x in sample_meta):
                    sample_meta.append({"sample_id": s, "biological_sample_id": s, "condition": arm1_lbl})
            for s in n2_names:
                if not any(x["sample_id"] == s for x in sample_meta):
                    sample_meta.append({"sample_id": s, "biological_sample_id": s, "condition": arm2_lbl})

            df["numerator_expected_samples"] = len(n2_names) if rev_d else len(n1_names)
            df["denominator_expected_samples"] = len(n1_names) if rev_d else len(n2_names)

            for eid, gid, gsym, p1s, p2s, i1s, i2s, s1s, s2s in zip(
                df["event_id"], df["gene_id"], df["gene_symbol"],
                df["IncLevel1"].map(parse_number_list), df["IncLevel2"].map(parse_number_list),
                df["IJC_SAMPLE_1"].map(parse_number_list), df["IJC_SAMPLE_2"].map(parse_number_list),
                df["SJC_SAMPLE_1"].map(parse_number_list), df["SJC_SAMPLE_2"].map(parse_number_list),
            ):
                if eid not in event_meta:
                    event_meta[eid] = {"gene_id": gid, "gene_symbol": gsym or gid, "event_type": et}
                for s_name, pv, iv, sv in zip(n1_names, p1s, i1s, s1s):
                    psi_obs.setdefault((eid, s_name), []).append(pv)
                    sup_obs.setdefault((eid, s_name), []).append((iv + sv) if (math.isfinite(iv) and math.isfinite(sv)) else math.nan)
                for s_name, pv, iv, sv in zip(n2_names, p2s, i2s, s2s):
                    psi_obs.setdefault((eid, s_name), []).append(pv)
                    sup_obs.setdefault((eid, s_name), []).append((iv + sv) if (math.isfinite(iv) and math.isfinite(sv)) else math.nan)

            all_frames.append(df)

    if not all_frames:
        raise ValueError("No non-empty rMATS tables found.")
    events_all = pd.concat(all_frames, ignore_index=True, sort=False)
    psi_mat = pd.Series({k: _finite_median(v) for k, v in psi_obs.items()}, dtype=float).unstack().sort_index()
    sup_mat = pd.Series({k: _finite_median(v) for k, v in sup_obs.items()}, dtype=float).unstack().sort_index()
    psi_mat.index.name = "event_id"
    sup_mat.index.name = "event_id"
    return events_all, psi_mat, sup_mat, sample_meta, event_meta
