"""Direct rMATS directory, .tar.gz archive, TSV, and Parquet loader for splicetrace."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
import numpy as np
import pandas as pd

EVENT_COORD_COLS = {
    "SE": ["exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"],
    "A3SS": ["longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"],
    "A5SS": ["longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"],
    "MXE": ["1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"],
    "RI": ["riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"],
}


def _mode_file_order(rmats_mode: str) -> tuple[str, ...]:
    mode = str(rmats_mode or "JCEC").upper()
    if mode not in {"JCEC", "JC"}:
        raise ValueError(f"Unsupported rmats_mode: {rmats_mode}")
    return (f".MATS.{mode}.txt",)


def _has_rmats_files(d: Path, mode: str = "JCEC") -> bool:
    if not d.is_dir():
        return False
    sufs = _mode_file_order(mode)
    return any((d / f"{k}{s}").is_file() for k in EVENT_COORD_COLS for s in sufs)


def _dir_contrast_label(p: Path) -> str:
    return p.parent.name if p.name.lower() == "rmats_post" else p.name


def _infer_arms_from_cid(cid: str) -> tuple[str, str]:
    q_str = str(cid or "").strip()
    if not q_str:
        return "", ""
    parts = re.split(r"_vs_", q_str, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2 and q_str.count("-") == 1:
        parts = q_str.split("-", 1)
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return "", ""


def _find_enclosing_bundle(src: Path) -> Path | None:
    """Locate an enclosing Katana bundle root (direct path or up to 4 ancestors) if present."""
    candidates = [src, *list(src.parents)[:4]]
    for cand in candidates:
        if (cand / "manifest.json").is_file() or any(
            (cand / rel).is_file()
            for rel in (
                "pipeline_info/contrastsheet.valid.csv",
                "inputs/contrastsheet.csv",
                "contrastsheet.csv",
            )
        ):
            return cand
    return None


def _find_expected_dir_for_contrast(
    bdir: Path,
    eff_cid: str,
    num_lbl: str,
    den_lbl: str,
    mc: dict[str, str] | None,
    rmats_mode: str,
) -> tuple[Path | None, bool]:
    if mc and mc.get("rmats_dir"):
        raw_d = Path(mc["rmats_dir"])
        cand_m = raw_d if raw_d.is_absolute() else (bdir / raw_d)
        if _has_rmats_files(cand_m, rmats_mode):
            return cand_m, False
        raise FileNotFoundError(f"Manifest rmats_dir for contrast '{eff_cid}' lacks *.MATS.{rmats_mode.upper()}.txt files: {cand_m}")
    roots = [bdir / "rmats", bdir / "rnasplicing/rmats"]
    fwd = [k for k in (eff_cid, f"{num_lbl}-{den_lbl}", f"{num_lbl}_vs_{den_lbl}") if k and k != "-"]
    rev = [k for k in (f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}") if k and k != "-"]
    for root in roots:
        if not root.is_dir():
            continue
        for k in fwd:
            for p in (root / k / "rmats_post", root / k):
                if _has_rmats_files(p, rmats_mode):
                    return p, False
        for k in rev:
            for p in (root / k / "rmats_post", root / k):
                if _has_rmats_files(p, rmats_mode):
                    return p, True
    return None, False


def _resolve_bundle_context(
    bdir: Path,
    *,
    src_dir: Path | None,
    contrast_id: str | None,
    rmats_mode: str,
    explicit_reverse_direction: bool | None = None,
) -> tuple[Path | None, bool, str | None]:
    """Resolve contrast rMATS dir and biological orientation from bundle manifest/contrastsheet."""
    c_path = next(
        (
            bdir / rel
            for rel in (
                "pipeline_info/contrastsheet.valid.csv",
                "inputs/contrastsheet.csv",
                "contrastsheet.csv",
            )
            if (bdir / rel).is_file()
        ),
        None,
    )
    contrasts: list[tuple[str, str, str]] = []
    if c_path is not None:
        sep = "\t" if c_path.suffix.lower() in {".tsv", ".txt"} else ","
        c_df = pd.read_csv(c_path, sep=sep, dtype=str, keep_default_na=False)
        seen_cids: set[str] = set()
        for _, r in c_df.iterrows():
            cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id") or "").strip()
            num_l = str(r.get("treatment") or r.get("numerator") or r.get("group_a") or "").strip()
            den_l = str(r.get("control") or r.get("denominator") or r.get("group_b") or "").strip()
            if cid:
                if cid.lower() in seen_cids:
                    raise ValueError(f"Duplicate normalized contrast_id {cid!r} in contrastsheet ({c_path}).")
                seen_cids.add(cid.lower())
                contrasts.append((cid, num_l, den_l))

    manifest_comps: dict[str, dict[str, str]] = {}
    mpath = bdir / "manifest.json"
    if mpath.is_file():
        try:
            bm = json.loads(mpath.read_text(encoding="utf-8"))
            if not isinstance(bm, dict):
                raise ValueError("top-level JSON value must be an object")
            comps_raw = bm.get("comparisons", [])
            if comps_raw is not None and not isinstance(comps_raw, list):
                raise ValueError("'comparisons' must be a list")
            seen_manifest_cids: set[str] = set()
            manifest_comp_keys: list[str] = []
            for cm in comps_raw or []:
                if not isinstance(cm, dict) or not cm.get("comparison_id"):
                    continue
                c_key = str(cm["comparison_id"]).strip()
                c_low = c_key.lower()
                if c_low in seen_manifest_cids:
                    raise ValueError(f"Duplicate normalized comparison_id {c_key!r} in manifest.json comparisons.")
                seen_manifest_cids.add(c_low)
                manifest_comp_keys.append(c_key)
                rec = {
                    "treatment": str(cm.get("treatment") or cm.get("numerator") or cm.get("group_a") or "").strip(),
                    "control": str(cm.get("control") or cm.get("denominator") or cm.get("group_b") or "").strip(),
                    "rmats_group1": str(cm.get("rmats_group1") or "").strip(),
                    "rmats_group2": str(cm.get("rmats_group2") or "").strip(),
                    "rmats_dir": str(cm.get("rmats_dir") or "").strip(),
                }
                manifest_comps[c_key] = rec
                manifest_comps.setdefault(c_low, rec)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"Malformed manifest.json in {mpath}: {exc}") from exc
    else:
        manifest_comp_keys = []

    target_cid = contrast_id.strip() if (contrast_id is not None and contrast_id.strip()) else None
    matched_c: tuple[str, str, str] | None = None
    if target_cid is not None:
        matched_c = next((c for c in contrasts if c[0] == target_cid or c[0].lower() == target_cid.lower()), None)
        if matched_c is None and contrasts:
            raise ValueError(
                f"Requested contrast_id {target_cid!r} not found in bundle contrasts: {[c[0] for c in contrasts]}."
            )
        if matched_c is None and manifest_comps and target_cid not in manifest_comps and target_cid.lower() not in manifest_comps:
            raise ValueError(
                f"Requested contrast_id {target_cid!r} not found in bundle manifest comparisons."
            )
    elif len(contrasts) == 1:
        matched_c = contrasts[0]
    elif len(manifest_comp_keys) == 1 and not contrasts:
        target_cid = manifest_comp_keys[0]
    elif len(contrasts) >= 2 and src_dir is not None and src_dir.resolve() != bdir.resolve():
        # Match explicit src_dir back to the owning contrast when caller passed --rmats <subdir> without --contrast-id
        src_res = src_dir.resolve()
        for c_cand in contrasts:
            mc_cand = manifest_comps.get(c_cand[0]) or manifest_comps.get(c_cand[0].lower())
            try:
                exp_c_dir, _ = _find_expected_dir_for_contrast(bdir, c_cand[0], c_cand[1], c_cand[2], mc_cand, rmats_mode)
            except FileNotFoundError:
                exp_c_dir = None
            if exp_c_dir is not None and exp_c_dir.resolve() == src_res:
                matched_c = c_cand
                break

    eff_cid = matched_c[0] if matched_c is not None else target_cid
    num_lbl, den_lbl = (matched_c[1], matched_c[2]) if matched_c is not None else ("", "")

    mc = (manifest_comps.get(eff_cid) or manifest_comps.get(eff_cid.lower())) if eff_cid else None
    if mc is None and len(manifest_comp_keys) == 1 and target_cid is None:
        eff_cid = manifest_comp_keys[0]
        mc = manifest_comps.get(eff_cid)
    if mc is not None:
        if not num_lbl and mc.get("treatment"):
            num_lbl = mc["treatment"]
        if not den_lbl and mc.get("control"):
            den_lbl = mc["control"]
    if (not num_lbl or not den_lbl) and eff_cid:
        inf_n, inf_d = _infer_arms_from_cid(eff_cid)
        if inf_n and inf_d:
            if mc and mc.get("rmats_group1") and mc.get("rmats_group2"):
                if {inf_n, inf_d} == {mc["rmats_group1"], mc["rmats_group2"]}:
                    num_lbl = num_lbl or inf_n
                    den_lbl = den_lbl or inf_d
            elif not mc:
                num_lbl = num_lbl or inf_n
                den_lbl = den_lbl or inf_d

    has_manifest_arms = bool(mc and mc.get("rmats_group1") and mc.get("rmats_group2"))
    if has_manifest_arms and num_lbl and den_lbl and {mc["rmats_group1"], mc["rmats_group2"]} != {num_lbl, den_lbl}:
        raise ValueError(
            f"manifest.json rmats_group1/2 ({mc['rmats_group1']}, {mc['rmats_group2']}) do not match contrast '{eff_cid}' ({num_lbl}, {den_lbl})."
        )
    if (
        explicit_reverse_direction is None
        and (not has_manifest_arms)
        and eff_cid
        and num_lbl
        and den_lbl
        and eff_cid.lower() in {f"{den_lbl}-{num_lbl}".lower(), f"{den_lbl}_vs_{num_lbl}".lower(), f"{den_lbl}_{num_lbl}".lower()}
    ):
        raise ValueError(
            f"Ambiguous contrast_id {eff_cid!r} matches reversed condition order '{den_lbl}-{num_lbl}' "
            f"for numerator={num_lbl!r}, denominator={den_lbl!r} without explicit manifest rmats_group1/rmats_group2 or reverse_direction."
        )

    src_is_explicit = src_dir is not None and src_dir.resolve() != bdir.resolve()
    resolved_dir: Path | None = None
    path_inverted = False

    if eff_cid is not None:
        expected_dir, expected_inv = _find_expected_dir_for_contrast(bdir, eff_cid, num_lbl, den_lbl, mc, rmats_mode)
        if src_is_explicit and _has_rmats_files(src_dir, rmats_mode):
            src_res = src_dir.resolve()
            if expected_dir is not None and src_res != expected_dir.resolve():
                raise ValueError(
                    f"Explicit rmats_source {src_dir} does not match expected contrast directory {expected_dir} for contrast '{eff_cid}'."
                )
            alt_mode = "JC" if rmats_mode.upper() == "JCEC" else "JCEC"
            if expected_dir is None:
                try:
                    alt_exp_dir, _ = _find_expected_dir_for_contrast(bdir, eff_cid, num_lbl, den_lbl, mc, alt_mode)
                except FileNotFoundError:
                    alt_exp_dir = None
                if alt_exp_dir is not None and src_res != alt_exp_dir.resolve():
                    raise ValueError(
                        f"Contrast '{eff_cid}' exists under mode={alt_mode} at {alt_exp_dir} and lacks mode={rmats_mode.upper()}; "
                        f"cannot bind explicit rmats_source {src_dir}."
                    )
            # Also reject if src_dir belongs to any different declared contrast under EITHER mode
            other_entries = [
                (oc, onum, oden, manifest_comps.get(oc) or manifest_comps.get(oc.lower()))
                for oc, onum, oden in contrasts
                if oc.lower() != eff_cid.lower()
            ] + [
                (mk, manifest_comps[mk].get("treatment", ""), manifest_comps[mk].get("control", ""), manifest_comps[mk])
                for mk in manifest_comp_keys
                if mk.lower() != eff_cid.lower()
            ]
            for other_cid, other_num, other_den, other_mc in other_entries:
                for chk_mode in ("JCEC", "JC"):
                    try:
                        other_dir, _ = _find_expected_dir_for_contrast(bdir, other_cid, other_num, other_den, other_mc, chk_mode)
                    except FileNotFoundError:
                        other_dir = None
                    if other_dir is not None and other_dir.resolve() == src_res:
                        raise ValueError(
                            f"Explicit rmats_source {src_dir} belongs to contrast '{other_cid}', conflicting with requested contrast '{eff_cid}'."
                        )
            # Reject unregistered neighboring directory when bundle declares contrasts/manifest comparisons
            if expected_dir is None and (contrasts or manifest_comp_keys):
                cand_labels = {eff_cid.lower()}
                if num_lbl and den_lbl:
                    cand_labels.update({
                        f"{num_lbl}-{den_lbl}".lower(),
                        f"{num_lbl}_vs_{den_lbl}".lower(),
                        f"{den_lbl}-{num_lbl}".lower(),
                        f"{den_lbl}_vs_{num_lbl}".lower(),
                    })
                src_lbl = _dir_contrast_label(src_dir).lower()
                if src_lbl not in cand_labels:
                    raise ValueError(
                        f"Explicit rmats_source {src_dir} ('{_dir_contrast_label(src_dir)}') is not registered for bundle contrast '{eff_cid}'."
                    )
            resolved_dir = src_dir
            if num_lbl and den_lbl:
                dname = _dir_contrast_label(src_dir)
                if dname.lower() in {f"{den_lbl}-{num_lbl}".lower(), f"{den_lbl}_vs_{num_lbl}".lower()}:
                    path_inverted = True
        else:
            resolved_dir = expected_dir
            path_inverted = expected_inv
            if resolved_dir is None and (len(contrasts) >= 1 or len(manifest_comp_keys) >= 1):
                raise FileNotFoundError(f"No rMATS contrast directory matched '{eff_cid}' (mode={rmats_mode.upper()}) under {bdir}")
    elif src_is_explicit and _has_rmats_files(src_dir, rmats_mode):
        resolved_dir = src_dir

    rev_dir = path_inverted
    if has_manifest_arms and num_lbl and den_lbl:
        rev_dir = (mc["rmats_group1"] == den_lbl and mc["rmats_group2"] == num_lbl)
    return resolved_dir, rev_dir, eff_cid


def _clean_str_tok(val: object) -> str:
    if val is None:
        return ""
    try:
        if pd.isna(val):
            return ""
    except (TypeError, ValueError):
        pass
    return str(val).strip().strip('"').strip("'")


def _strict_int_coord_col(ser: pd.Series, col_name: str, kind: str) -> pd.Series:
    out: list[str] = []
    for idx_r, raw in enumerate(ser.tolist()):
        s = _clean_str_tok(raw)
        if not s:
            raise ValueError(f"Missing required coordinate '{col_name}' in {kind} row {idx_r}.")
        try:
            fv = float(s)
        except ValueError as exc:
            raise ValueError(f"Non-numeric coordinate {s!r} in '{col_name}' ({kind} row {idx_r}).") from exc
        if (not math.isfinite(fv)) or fv != math.floor(fv) or fv < 0:
            raise ValueError(f"Non-integer or negative coordinate {s!r} in '{col_name}' ({kind} row {idx_r}).")
        out.append(str(int(round(fv))))
    return pd.Series(out, index=ser.index, dtype="object")


def _parse_scalar_col_range(
    ser: pd.Series, *, col_name: str, lo: float, hi: float, kind: str,
) -> np.ndarray:
    vals = np.full(len(ser), np.nan, dtype=float)
    for i, raw in enumerate(ser.tolist()):
        if raw is None:
            continue
        try:
            if pd.isna(raw):
                continue
        except (TypeError, ValueError):
            pass
        s = str(raw).strip()
        if not s or s.upper() in {"NA", "NAN", "NONE", "NULL"}:
            continue
        try:
            fv = float(s)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric token {s!r} in column '{col_name}' ({kind} row {i}).") from exc
        if (not math.isfinite(fv)) or fv < lo or fv > hi:
            raise ValueError(
                f"Out-of-range value {fv} in '{col_name}' ({kind} row {i}); expected [{lo}, {hi}]."
            )
        vals[i] = fv
    return vals


def _parse_replicate_vec(raw: object, *, col_name: str, kind: str, row_idx: int) -> np.ndarray:
    if raw is None:
        raise ValueError(f"Missing replicate vector in '{col_name}' ({kind} row {row_idx}).")
    try:
        if pd.isna(raw):
            raise ValueError(f"Missing replicate vector in '{col_name}' ({kind} row {row_idx}).")
    except (TypeError, ValueError):
        pass
    s = str(raw).strip()
    if not s:
        raise ValueError(f"Empty replicate vector in '{col_name}' ({kind} row {row_idx}).")
    toks = [t.strip() for t in s.split(",")]
    arr = np.empty(len(toks), dtype=float)
    for j, tok in enumerate(toks):
        if not tok or tok.upper() in {"NA", "NAN", "NONE", "NULL"}:
            arr[j] = np.nan
            continue
        try:
            fv = float(tok)
        except ValueError as exc:
            raise ValueError(f"Non-numeric replicate value {tok!r} in '{col_name}' ({kind} row {row_idx}).") from exc
        if not math.isfinite(fv):
            raise ValueError(f"Non-finite replicate value {tok!r} in '{col_name}' ({kind} row {row_idx}).")
        arr[j] = fv
    return arr


def _check_duplicate_event_conflicts(
    df: pd.DataFrame, *, stat_cols: tuple[str, ...], context: str,
) -> pd.DataFrame:
    if df.empty or "event_id" not in df.columns:
        return df
    dup_mask = df["event_id"].astype(str).duplicated(keep=False)
    if not dup_mask.any():
        return df
    sub = df.loc[dup_mask].copy()
    present_stats = [c for c in stat_cols if c in sub.columns]
    num_map = {c: pd.to_numeric(sub[c], errors="coerce") for c in present_stats}
    for eid, grp_idx in sub.groupby(sub["event_id"].astype(str), sort=False).groups.items():
        idxs = list(grp_idx)
        if len(idxs) <= 1:
            continue
        if "gene_id" in sub.columns:
            gvals = {
                "" if pd.isna(v) else str(v).strip().strip('"').split(".")[0]
                for v in sub.loc[idxs, "gene_id"].tolist()
            }
            if len(gvals) > 1:
                raise ValueError(f"Conflicting duplicate event_id '{eid}' gene_id values {sorted(gvals)} in {context}.")
        for c in present_stats:
            s_vals = num_map[c].loc[idxs].to_numpy(dtype=float)
            fin = np.isfinite(s_vals)
            if fin.any() and not fin.all():
                raise ValueError(f"Conflicting duplicate event_id '{eid}' in column '{c}' ({context}): finite vs missing.")
            if fin.all() and float(np.max(s_vals) - np.min(s_vals)) > 1e-6:
                raise ValueError(
                    f"Conflicting duplicate event_id '{eid}' statistics in column '{c}' ({context}): {s_vals.tolist()}."
                )
    return df.drop_duplicates(subset=["event_id"]).copy()


def _validate_bool_arg(val: object, name: str) -> bool | None:
    if val is None:
        return None
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    raise TypeError(f"{name} must be a bool or None, got {type(val).__name__} ({val!r}).")


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(blk)
    return h.hexdigest()


def _extract_expected_ref_sha(bdir: Path) -> tuple[dict[str, str], Path | None]:
    expected: dict[str, str] = {}
    source_path: Path | None = None
    for rel in (
        "provenance/reference_audit.json",
        "pipeline_info/reference_audit.json",
        "reference_audit.json",
        "pipeline_info/input_audit.json",
        "manifest.json",
    ):
        ap = bdir / rel
        if not ap.is_file():
            continue
        try:
            data = json.loads(ap.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            lbl = "manifest.json" if rel == "manifest.json" else "reference audit JSON"
            raise ValueError(f"Malformed {lbl} in {ap}: {exc}") from exc
        if not isinstance(data, dict):
            lbl = "manifest.json" if rel == "manifest.json" else "reference audit JSON"
            raise ValueError(f"Malformed {lbl} in {ap}: expected JSON object.")
        refs_obj = data.get("references") if isinstance(data.get("references"), dict) else data
        found_here = False
        for kind in ("fasta", "gtf"):
            entry = refs_obj.get(kind) if isinstance(refs_obj, dict) else None
            if isinstance(entry, dict) and entry.get("sha256"):
                sha = str(entry["sha256"]).strip().lower()
                if sha and kind not in expected:
                    expected[kind] = sha
                    found_here = True
        if found_here and source_path is None:
            source_path = ap
        if "fasta" in expected and "gtf" in expected:
            break
    return expected, source_path


def verify_bundle_reference_sha(
    bdir: Path,
    *,
    fasta_path: Path | str | None = None,
    gtf_path: Path | str | None = None,
    allow_reference_reannotation: bool = False,
) -> dict[str, object]:
    """Verify FASTA/GTF SHA256 against the bundle reference audit unless explicit re-annotation is requested."""
    expected, audit_path = _extract_expected_ref_sha(bdir)
    info: dict[str, object] = {
        "reference_audit_path": str(audit_path) if audit_path is not None else None,
        "expected": expected,
        "actual": {},
        "reannotated": False,
    }
    if not expected:
        return info
    mismatches: list[str] = []
    actual_map: dict[str, str] = {}
    for kind, raw_p in (("fasta", fasta_path), ("gtf", gtf_path)):
        exp_sha = expected.get(kind)
        if not exp_sha or raw_p is None:
            continue
        p = Path(raw_p)
        if not p.is_file():
            continue
        act_sha = _file_sha256(p).lower()
        actual_map[kind] = act_sha
        if act_sha != exp_sha:
            mismatches.append(f"{kind.upper()} SHA256 mismatch vs {audit_path}: expected {exp_sha}, got {act_sha}")
    info["actual"] = actual_map
    if mismatches:
        if allow_reference_reannotation:
            info["reannotated"] = True
        else:
            raise ValueError(
                "; ".join(mismatches)
                + ". Supply the bundle-matched reference or pass allow_reference_reannotation=True / --allow-reference-reannotation."
            )
    return info


def _standardize_rmats_df(
    d: pd.DataFrame,
    kind: str,
    valid_contigs: set[str] | None,
    comparison_id: str,
    *,
    reverse_direction: bool = False,
) -> pd.DataFrame:
    rev_bool = _validate_bool_arg(reverse_direction, "reverse_direction")
    reverse_direction = bool(rev_bool) if rev_bool is not None else False
    if d.empty:
        return pd.DataFrame()
    coord_cols = EVENT_COORD_COLS[kind]
    req_cols = [
        "chr", "strand", *coord_cols, "GeneID", "FDR", "IncLevelDifference",
        "IJC_SAMPLE_1", "SJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_2", "IncLevel1", "IncLevel2",
    ]
    missing_cols = [c for c in req_cols if c not in d.columns]
    if missing_cols:
        raise ValueError(f"rMATS {kind} table missing required column(s): {missing_cols}.")
    chr_s = d["chr"].map(_clean_str_tok)
    strand_s = d["strand"].map(_clean_str_tok)
    if valid_contigs is not None:
        keep_mask = chr_s.isin(valid_contigs) & strand_s.isin(["+", "-"])
    else:
        keep_mask = strand_s.isin(["+", "-"])
    d = d.loc[keep_mask].copy()
    if d.empty:
        return pd.DataFrame()
    d["chr"] = chr_s.loc[d.index]
    d["strand"] = strand_s.loc[d.index]
    for c_col in coord_cols:
        d[c_col] = _strict_int_coord_col(d[c_col], c_col, kind)

    d["event_type"] = kind
    d["gene_id"] = d["GeneID"].astype(str).str.strip('"').str.split(".").str[0]
    d["gene_symbol"] = d.get("geneSymbol", d["gene_id"]).astype(str).str.strip('"')
    d["event_id"] = (
        kind + ":" + d["chr"].astype(str) + ":" + d["strand"].astype(str) + ":"
        + d[coord_cols].astype(str).agg(":".join, axis=1)
    )

    d["fdr"] = _parse_scalar_col_range(d["FDR"], col_name="FDR", lo=0.0, hi=1.0, kind=kind)
    pv_ser = d["PValue"] if "PValue" in d.columns else d["FDR"]
    d["PValue"] = _parse_scalar_col_range(pv_ser, col_name="PValue", lo=0.0, hi=1.0, kind=kind)
    raw_dpsi = _parse_scalar_col_range(d["IncLevelDifference"], col_name="IncLevelDifference", lo=-1.0, hi=1.0, kind=kind)
    d["dpsi"] = -raw_dpsi if reverse_direction else raw_dpsi
    d["reverse_direction"] = bool(reverse_direction)

    base_col = "IncLevel1" if reverse_direction else "IncLevel2"
    r1_m, r2_m, base_p = [], [], []
    expected_arity: tuple[int, int] | None = None
    for row_idx, (a1, b1, inc1, a2, b2, inc2, inc_base) in enumerate(zip(
        d["IJC_SAMPLE_1"], d["SJC_SAMPLE_1"], d["IncLevel1"],
        d["IJC_SAMPLE_2"], d["SJC_SAMPLE_2"], d["IncLevel2"],
        d[base_col],
    )):
        x1 = _parse_replicate_vec(a1, col_name="IJC_SAMPLE_1", kind=kind, row_idx=row_idx)
        y1 = _parse_replicate_vec(b1, col_name="SJC_SAMPLE_1", kind=kind, row_idx=row_idx)
        p1 = _parse_replicate_vec(inc1, col_name="IncLevel1", kind=kind, row_idx=row_idx)
        x2 = _parse_replicate_vec(a2, col_name="IJC_SAMPLE_2", kind=kind, row_idx=row_idx)
        y2 = _parse_replicate_vec(b2, col_name="SJC_SAMPLE_2", kind=kind, row_idx=row_idx)
        p2 = _parse_replicate_vec(inc2, col_name="IncLevel2", kind=kind, row_idx=row_idx)

        if len(x1) != len(y1) or len(x1) != len(p1) or len(x2) != len(y2) or len(x2) != len(p2):
            raise ValueError(
                f"Inconsistent rMATS replicate list lengths in {kind} row {row_idx}: "
                f"arm1=({len(x1)},{len(y1)},{len(p1)}), arm2=({len(x2)},{len(y2)},{len(p2)})."
            )
        arity = (len(x1), len(x2))
        if expected_arity is None:
            expected_arity = arity
        elif arity != expected_arity:
            raise ValueError(
                f"Inconsistent replicate arity across rows in {kind}: expected {expected_arity}, got {arity} at row {row_idx}."
            )

        for label, vec in (("IJC_SAMPLE_1", x1), ("SJC_SAMPLE_1", y1), ("IJC_SAMPLE_2", x2), ("SJC_SAMPLE_2", y2)):
            if np.any(np.isfinite(vec) & (vec < 0.0)):
                raise ValueError(f"Negative rMATS count in '{label}' at {kind} row {row_idx}.")
        for label, vec in (("IncLevel1", p1), ("IncLevel2", p2)):
            if np.any(np.isfinite(vec) & ((vec < 0.0) | (vec > 1.0))):
                raise ValueError(f"Out-of-range PSI in '{label}' at {kind} row {row_idx}; expected [0, 1].")

        s1 = x1 + y1
        s2 = x2 + y2
        fin_s1 = s1[np.isfinite(s1)]
        fin_s2 = s2[np.isfinite(s2)]
        r1_m.append(float(np.mean(fin_s1)) if len(fin_s1) >= 1 else np.nan)
        r2_m.append(float(np.mean(fin_s2)) if len(fin_s2) >= 1 else np.nan)

        p_b = p1 if reverse_direction else p2
        fin_p = p_b[np.isfinite(p_b)]
        base_p.append(float(np.mean(fin_p)) if len(fin_p) >= 1 else np.nan)

    d["depth"] = np.minimum(r1_m, r2_m)
    d["baseline_PSI"] = base_p
    d["comparison_id"] = comparison_id
    d = _check_duplicate_event_conflicts(
        d,
        stat_cols=("fdr", "PValue", "dpsi", "depth", "baseline_PSI"),
        context=f"{kind} ({comparison_id})",
    )
    return d


def _select_tar_contrast_members(
    tar: tarfile.TarFile,
    prio_sufs: tuple[str, ...],
    eff_cid_query: str | None,
    src: Path,
) -> dict[str, tarfile.TarInfo]:
    """Select rMATS files from a single contrast directory locked BEFORE mode validation inside `.tar`/`.tar.gz`."""
    all_groups: dict[str, dict[str, dict[str, tarfile.TarInfo]]] = {}
    mode_sufs = (("JCEC", ".MATS.JCEC.txt"), ("JC", ".MATS.JC.txt"))
    for member in tar:
        norm_name = member.name.replace("\\", "/").lstrip("./")
        base = norm_name.rsplit("/", 1)[-1]
        if not member.isfile() or base.startswith("._"):
            continue
        parent_parts = [p for p in norm_name.split("/")[:-1] if p and p != "."]
        if parent_parts and parent_parts[-1].lower() == "rmats_post":
            parent_parts = parent_parts[:-1]
        group_key = "/".join(parent_parts)
        for kind in EVENT_COORD_COLS:
            for m_name, suf in mode_sufs:
                if base == f"{kind}{suf}":
                    g_mode = all_groups.setdefault(group_key, {}).setdefault(m_name, {})
                    if kind in g_mode:
                        raise ValueError(
                            f"Duplicate rMATS '{kind}{suf}' entries in archive contrast directory {group_key or '.'!r} of {src}."
                        )
                    g_mode[kind] = member

    if not all_groups:
        return {}
    if len(all_groups) > 1:
        if not eff_cid_query:
            raise ValueError(
                f"Archive {src} contains multiple rMATS contrast directories ({sorted(all_groups)}); "
                "pass explicit --contrast-id / contrast_id= to select one unambiguously."
            )
        q_exact = eff_cid_query.strip()
        q_low = q_exact.lower()
        matches = [k for k in all_groups if k == q_exact or Path(k).name == q_exact]
        if not matches:
            matches = [k for k in all_groups if k.lower() == q_low or Path(k).name.lower() == q_low]
        if not matches:
            raise ValueError(
                f"Requested contrast_id {eff_cid_query!r} not found among archive contrast directories {sorted(all_groups)} in {src}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous contrast_id {eff_cid_query!r} matched multiple archive directories {sorted(matches)} in {src}."
            )
        chosen = matches[0]
    else:
        chosen = next(iter(all_groups))
        if eff_cid_query and chosen:
            leaf = Path(chosen).name
            generic_names = {"rmats", "rmats_post", "rmats_out", "output", "outputs", "results"}
            q_low = eff_cid_query.strip().lower()
            if leaf.lower() not in generic_names and chosen.lower() != q_low and leaf.lower() != q_low:
                raise ValueError(
                    f"Requested contrast_id {eff_cid_query!r} does not match archive contrast directory {chosen!r} in {src}."
                )

    req_mode = "JCEC" if ".MATS.JCEC.txt" in prio_sufs else "JC"
    mode_bucket = all_groups[chosen].get(req_mode, {})
    if not mode_bucket:
        raise ValueError(
            f"Archive contrast directory {chosen or '.'!r} in {src} lacks rMATS {req_mode} files "
            f"(available modes: {sorted(all_groups[chosen])})."
        )
    return mode_bucket


def _filter_precomputed_contrast(
    df: pd.DataFrame,
    *,
    eff_cid_query: str | None,
    src: Path,
) -> pd.DataFrame:
    """Validate and filter precomputed `.tsv`/`.parquet` splicing tables by requested `contrast_id`."""
    present_c_cols = [c for c in ("comparison_id", "contrast_id", "contrast") if c in df.columns]
    if not present_c_cols or df.empty:
        return df
    out_df = df.copy()
    norm_cols: dict[str, pd.Series] = {}
    for col in present_c_cols:
        norm_cols[col] = out_df[col].map(lambda v: "" if pd.isna(v) else str(v).strip())
        out_df[col] = norm_cols[col]

    if len(present_c_cols) >= 2:
        for r_i in range(len(out_df)):
            row_vals = {norm_cols[col].iloc[r_i] for col in present_c_cols if norm_cols[col].iloc[r_i] != ""}
            if len(row_vals) > 1:
                raise ValueError(
                    f"Conflicting contrast identifier columns {present_c_cols} at row {r_i} in {src}: {sorted(row_vals)}."
                )

    c_col = present_c_cols[0]
    vals = norm_cols[c_col].copy()
    for alt_col in present_c_cols[1:]:
        vals = vals.where(vals != "", norm_cols[alt_col])
    out_df[c_col] = vals

    uniq_c = [v for v in dict.fromkeys(vals) if v != ""]
    if uniq_c and (vals == "").any():
        raise ValueError(
            f"Precomputed event table {src} mixes annotated contrast IDs {uniq_c} with blank/missing values in '{c_col}'."
        )
    if eff_cid_query is not None:
        q_s = eff_cid_query.strip()
        mask = vals == q_s
        if not mask.any():
            mask = vals.str.lower() == q_s.lower()
            if mask.any():
                matched_raw = list(dict.fromkeys(vals.loc[mask]))
                if len(matched_raw) > 1:
                    raise ValueError(
                        f"Ambiguous case-insensitive contrast_id {eff_cid_query!r} matched multiple raw IDs {matched_raw} in {src}."
                    )
        if not mask.any():
            raise ValueError(
                f"Requested contrast_id {eff_cid_query!r} not found in {src} column '{c_col}' (available: {uniq_c})."
            )
        out_df = out_df.loc[mask].reset_index(drop=True)
    elif len(uniq_c) > 1:
        raise ValueError(
            f"Precomputed event table {src} contains multiple contrasts in '{c_col}': {uniq_c}; "
            "specify explicit --contrast-id / contrast_id=."
        )

    return _check_duplicate_event_conflicts(
        out_df,
        stat_cols=("dpsi", "delta_psi", "raw_dpsi", "fdr", "FDR", "PValue", "pvalue", "p_value", "depth", "baseline_PSI"),
        context=str(src),
    ).reset_index(drop=True)


def _validate_precomputed_domains(df: pd.DataFrame, context: str) -> None:
    for col in ("fdr", "FDR", "PValue", "pvalue", "p_value", "baseline_PSI"):
        if col in df.columns:
            _parse_scalar_col_range(df[col], col_name=col, lo=0.0, hi=1.0, kind=context)
    for col in ("dpsi", "delta_psi", "raw_dpsi"):
        if col in df.columns:
            _parse_scalar_col_range(df[col], col_name=col, lo=-1.0, hi=1.0, kind=context)


def load_rmats_measurable_events(
    rmats_source: Path | str,
    fasta_path: Path | str | None = None,
    *,
    min_depth: float = 10.0,
    comparison_id: str = "rmats_contrast",
    rmats_mode: str = "JCEC",
    reverse_direction: bool | None = None,
    bundle_dir: Path | str | None = None,
    contrast_id: str | None = None,
    allow_reference_reannotation: bool = False,
) -> pd.DataFrame:
    """Load measurable splicing events from an rMATS output directory, `.tar.gz` archive, or `.tsv`/`.parquet` file."""
    rev_arg = _validate_bool_arg(reverse_direction, "reverse_direction")
    allow_reannot = bool(_validate_bool_arg(allow_reference_reannotation, "allow_reference_reannotation"))
    src = Path(rmats_source)
    prio_sufs = _mode_file_order(rmats_mode)
    valid_contigs: set[str] | None = None
    if fasta_path is not None:
        fai = Path(str(fasta_path) + ".fai")
        if fai.is_file():
            raw_contigs = {ln.split("\t")[0] for ln in fai.read_text(encoding="utf-8").splitlines() if ln.strip()}
            valid_contigs = set(raw_contigs)
            for c in raw_contigs:
                valid_contigs.add(c[3:] if c.startswith("chr") else f"chr{c}")

    eff_cid_query = contrast_id if contrast_id is not None else (comparison_id if comparison_id != "rmats_contrast" else None)
    eff_comp_id = contrast_id or comparison_id
    eff_rev = bool(rev_arg) if rev_arg is not None else False

    bdir_resolved = Path(bundle_dir) if bundle_dir is not None else (_find_enclosing_bundle(src) if src.exists() else None)
    if bdir_resolved is not None:
        verify_bundle_reference_sha(
            bdir_resolved,
            fasta_path=fasta_path,
            allow_reference_reannotation=allow_reannot,
        )
    if bdir_resolved is not None and (src.is_dir() or bundle_dir is not None):
        rdir_res, bundle_rev, bundle_cid = _resolve_bundle_context(
            bdir_resolved,
            src_dir=src if src.is_dir() else None,
            contrast_id=eff_cid_query,
            rmats_mode=rmats_mode,
            explicit_reverse_direction=rev_arg,
        )
        if src.is_dir() and rdir_res is not None:
            src = rdir_res
        if rev_arg is None:
            eff_rev = bundle_rev
        if comparison_id == "rmats_contrast" and bundle_cid:
            eff_comp_id = bundle_cid
    elif src.is_dir() and rev_arg is None and eff_cid_query:
        dir_label = src.parent.name if src.name.lower() == "rmats_post" else src.name
        q_str = eff_cid_query.strip()
        q_parts = re.split(r"_vs_", q_str, maxsplit=1, flags=re.IGNORECASE)
        if len(q_parts) != 2 and q_str.count("-") == 1:
            q_parts = q_str.split("-", 1)
        if len(q_parts) == 2:
            num_q, den_q = q_parts[0].strip(), q_parts[1].strip()
            if num_q and den_q and dir_label.lower() in {
                f"{den_q}-{num_q}".lower(),
                f"{den_q}_vs_{num_q}".lower(),
                f"{den_q}_{num_q}".lower(),
            }:
                eff_rev = True

    frames: list[pd.DataFrame] = []
    found_any_rmats_file = False
    if src.is_dir():
        eff_suf = prio_sufs[0]
        for kind in EVENT_COORD_COLS:
            cand = src / f"{kind}{eff_suf}"
            if cand.is_file():
                found_any_rmats_file = True
                raw = pd.read_csv(cand, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
                frames.append(_standardize_rmats_df(raw, kind, valid_contigs, eff_comp_id, reverse_direction=eff_rev))
    elif str(src).endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(src, "r:*") as tar:
            selected_members = _select_tar_contrast_members(tar, prio_sufs, eff_cid_query, src)
            for kind, member in selected_members.items():
                fobj = tar.extractfile(member)
                if fobj is not None:
                    found_any_rmats_file = True
                    raw = pd.read_csv(fobj, sep="\t", dtype=str, keep_default_na=False, low_memory=False)
                    frames.append(_standardize_rmats_df(raw, kind, valid_contigs, eff_comp_id, reverse_direction=eff_rev))
    elif src.suffix == ".parquet":
        raw_p = _filter_precomputed_contrast(pd.read_parquet(src), eff_cid_query=eff_cid_query, src=src)
        if "IncLevelDifference" in raw_p.columns and "dpsi" not in raw_p.columns and "event_type" in raw_p.columns:
            for kind, grp in raw_p.groupby("event_type"):
                if str(kind) in EVENT_COORD_COLS:
                    found_any_rmats_file = True
                    frames.append(_standardize_rmats_df(grp.copy(), str(kind), valid_contigs, eff_comp_id, reverse_direction=eff_rev))
        else:
            _validate_precomputed_domains(raw_p, str(src))
            return raw_p
    else:
        raw_t = _filter_precomputed_contrast(
            pd.read_csv(src, sep="\t", dtype=str, keep_default_na=False, low_memory=False),
            eff_cid_query=eff_cid_query,
            src=src,
        )
        if "IncLevelDifference" in raw_t.columns and "dpsi" not in raw_t.columns and "event_type" in raw_t.columns:
            for kind, grp in raw_t.groupby("event_type"):
                if str(kind) in EVENT_COORD_COLS:
                    found_any_rmats_file = True
                    frames.append(_standardize_rmats_df(grp.copy(), str(kind), valid_contigs, eff_comp_id, reverse_direction=eff_rev))
        else:
            _validate_precomputed_domains(raw_t, str(src))
            _num_pre_cols = {
                "fdr", "FDR", "PValue", "pvalue", "p_value", "dpsi", "delta_psi", "raw_dpsi",
                "depth", "baseline_PSI", "IncFormLen", "SkipFormLen", "GC", "length", "ambiguous_fraction",
                *{c for cols in EVENT_COORD_COLS.values() for c in cols},
            }
            for nc in _num_pre_cols & set(raw_t.columns):
                raw_t[nc] = pd.to_numeric(raw_t[nc], errors="coerce")
            return raw_t

    frames = [f for f in frames if not f.empty]
    if not frames:
        if found_any_rmats_file:
            return pd.DataFrame(columns=["event_id", "gene_id", "gene_symbol", "event_type", "chr", "strand", "fdr", "PValue", "dpsi", "depth", "baseline_PSI", "comparison_id", "reverse_direction"])
        raise ValueError(f"No valid rMATS {rmats_mode.upper()} event tables found in {src}")
    all_ev = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["event_id"])
    ok = (
        all_ev["fdr"].notna()
        & all_ev["dpsi"].notna()
        & all_ev["baseline_PSI"].notna()
        & (all_ev["depth"] >= float(min_depth))
        & (all_ev["gene_id"] != "")
    )
    return all_ev.loc[ok].reset_index(drop=True)
