"""Lightweight rMATS & Katana-bundle PSI/support matrix extractor for standalone SpliceNet runs."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any
import pandas as pd

EVENT_TYPES = ("SE", "RI", "A3SS", "A5SS", "MXE")
_DONOR_ID_ALIASES = (
    "biological_sample_id",
    "donor_id",
    "donor",
    "mouse_id",
    "subject",
    "patient",
    "individual",
    "animal_id",
)
_BLANK_DONOR_TOKENS = {"", "nan", "none", "na", "null", "n/a"}
EVENT_COORDINATES = (
    "chr", "strand",
    "exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE",
    "riExonStart_0base", "riExonEnd",
    "longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE",
    "1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd",
)


def _clean(val: object) -> str:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    return str(val).strip().strip('"').strip("'")


def _extract_donor_id(row: pd.Series | dict[str, Any], fallback_sid: str) -> str:
    """Resolve biological donor identity across canonical metadata aliases, falling back to sample_id."""
    col_map = {str(k).strip().lower(): k for k in row.keys()}
    for alias in _DONOR_ID_ALIASES:
        orig_col = col_map.get(alias)
        if orig_col is None:
            continue
        raw_val = row[orig_col]
        try:
            if pd.isna(raw_val):
                continue
        except (TypeError, ValueError):
            pass
        cleaned = str(raw_val).strip().strip('"').strip("'")
        if cleaned.lower() not in _BLANK_DONOR_TOKENS:
            return cleaned
    return fallback_sid


def _has_rmats_files(d: Path, mode: str = "JCEC") -> bool:
    if not d.is_dir():
        return False
    m = mode.upper()
    return any((d / f"{et}.MATS.{m}.txt").is_file() for et in EVENT_TYPES)


def _dir_contrast_label(cand: Path) -> str:
    return cand.parent.name if cand.name.lower() == "rmats_post" else cand.name


def _validate_bare_dir_contrast(
    cand: Path,
    *,
    cid: str,
    num_lbl: str,
    den_lbl: str,
    fwd_low: set[str],
    rev_low: set[str],
    explicit_arm_order: bool,
) -> tuple[Path, bool]:
    dir_label = _dir_contrast_label(cand)
    dl_low = dir_label.lower()
    if (not explicit_arm_order) and dl_low in rev_low:
        return cand, True
    if dl_low in fwd_low:
        return cand, False
    if (not explicit_arm_order) and num_lbl and den_lbl and ("_vs_" in dl_low or "-" in dl_low):
        parts = dl_low.split("_vs_") if "_vs_" in dl_low else dl_low.split("-")
        toks = {p.strip() for p in parts if p.strip()}
        if toks & {num_lbl.lower(), den_lbl.lower()}:
            raise ValueError(
                f"Bare rMATS directory {cand} ('{dir_label}') conflicts with declared contrast "
                f"'{cid}' ({num_lbl} vs {den_lbl})."
            )
    return cand, False


def _resolve_contrast_rmats_dir(
    search_roots: list[Path],
    *,
    cid: str,
    num_lbl: str,
    den_lbl: str,
    n_contrasts: int,
    mode: str,
    explicit_dir: Path | None = None,
    allow_bare_root: Path | None = None,
    explicit_arm_order: bool = False,
) -> tuple[Path, bool]:
    """Strictly match a declared contrast to its rMATS directory without silent orientation conflicts."""
    rev_names = tuple(dict.fromkeys(k for k in (f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}") if k and num_lbl and den_lbl))
    rev_conflict_set = {k.lower() for k in rev_names}
    if num_lbl and den_lbl:
        rev_conflict_set.add(f"{den_lbl}_{num_lbl}".lower())
    if (not explicit_arm_order) and cid and cid.strip().lower() in rev_conflict_set:
        raise ValueError(
            f"Ambiguous contrast_id {cid!r} matches reversed condition order '{den_lbl}-{num_lbl}' "
            f"for numerator={num_lbl!r}, denominator={den_lbl!r} without explicit manifest rmats_group1/rmats_group2."
        )

    if explicit_dir is not None:
        if not _has_rmats_files(explicit_dir, mode):
            raise FileNotFoundError(f"Declared manifest rmats_dir for '{cid}' lacks *.MATS.{mode.upper()}.txt files: {explicit_dir}")
        if explicit_arm_order:
            return explicit_dir, False
        rev_name = _dir_contrast_label(explicit_dir).lower() in rev_conflict_set
        return explicit_dir, rev_name

    fwd_names = tuple(dict.fromkeys(k for k in (cid, f"{num_lbl}-{den_lbl}", f"{num_lbl}_vs_{den_lbl}") if k))

    for root in search_roots:
        if not root.is_dir():
            continue
        for k in fwd_names:
            for cand in (root / k / "rmats_post", root / k):
                if _has_rmats_files(cand, mode):
                    return cand, False
        for k in rev_names:
            for cand in (root / k / "rmats_post", root / k):
                if _has_rmats_files(cand, mode):
                    return cand, True

    fwd_low = {k.lower() for k in fwd_names}
    rev_low = {k.lower() for k in rev_names} | rev_conflict_set
    available_subs: list[Path] = []
    for root in search_roots:
        if not root.is_dir():
            continue
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            target = (
                sub / "rmats_post"
                if _has_rmats_files(sub / "rmats_post", mode)
                else (sub if _has_rmats_files(sub, mode) else None)
            )
            if target is None:
                continue
            available_subs.append(target)
            low_name = sub.name.lower()
            if low_name in fwd_low:
                return target, False
            if low_name in rev_low:
                return target, True

    if n_contrasts == 1:
        if allow_bare_root is not None and _has_rmats_files(allow_bare_root, mode):
            return _validate_bare_dir_contrast(
                allow_bare_root,
                cid=cid,
                num_lbl=num_lbl,
                den_lbl=den_lbl,
                fwd_low=fwd_low,
                rev_low=rev_low,
                explicit_arm_order=explicit_arm_order,
            )
        if len(available_subs) == 1:
            return _validate_bare_dir_contrast(
                available_subs[0],
                cid=cid,
                num_lbl=num_lbl,
                den_lbl=den_lbl,
                fwd_low=fwd_low,
                rev_low=rev_low,
                explicit_arm_order=explicit_arm_order,
            )

    raise FileNotFoundError(
        f"No rMATS contrast directory matched '{cid}' "
        f"(tried {list(fwd_names) + list(rev_names)}) under {[str(r) for r in search_roots]}"
    )


def _norm_gene(val: object) -> str:
    txt = _clean(val)
    m = re.match(r"^(ENS[A-Z]*G\d+)(?:\.\d+)?$", txt, flags=re.IGNORECASE)
    return m.group(1) if m else txt


def _parse_nums(val: object) -> list[float]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    out: list[float] = []
    for tok in str(val).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok or tok.upper() in {"NA", "NAN", "NONE"}:
            out.append(math.nan)
            continue
        try:
            n = float(tok)
            out.append(n if math.isfinite(n) else math.nan)
        except ValueError:
            out.append(math.nan)
    return out


_COORD_COLS = {
    "SE": ("exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "A3SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "A5SS": ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"),
    "MXE": ("1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
    "RI": ("riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"),
}


def _canon_ids(df: pd.DataFrame, et: str) -> pd.Series:
    cols = _COORD_COLS.get(et, ())
    if cols and {"chr", "strand", *cols}.issubset(df.columns):
        return (
            et + ":" + df["chr"].map(_clean) + ":" + df["strand"].map(_clean) + ":"
            + df[list(cols)].apply(lambda c: c.map(_clean)).agg(":".join, axis=1)
        )
    ids = pd.Series(et, index=df.index, dtype="object")
    for col in EVENT_COORDINATES:
        if col in df.columns:
            vals = df[col].map(_clean)
            ok = vals.ne("")
            ids.loc[ok] = ids.loc[ok] + "|" + col + "=" + vals.loc[ok]
    return ids


def _med(vals: list[float]) -> float:
    fin = sorted(v for v in vals if math.isfinite(v))
    n = len(fin)
    return (fin[n // 2] if n % 2 else 0.5 * (fin[n // 2 - 1] + fin[n // 2])) if n else math.nan


def _parse_manifest_rmats(bdir: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    manifest_order: dict[str, list[str]] = {}
    manifest_comps: dict[str, dict[str, Any]] = {}
    mpath = bdir / "manifest.json"
    if not mpath.is_file():
        return manifest_order, manifest_comps
    try:
        bm = json.loads(mpath.read_text(encoding="utf-8"))
        if not isinstance(bm, dict):
            raise ValueError("top-level JSON value must be an object")
        top_ord = bm.get("rmats_sample_order")
        if top_ord is not None:
            if not isinstance(top_ord, dict):
                raise ValueError("'rmats_sample_order' must be an object")
            for k, v in top_ord.items():
                if not isinstance(v, list):
                    raise ValueError(f"'rmats_sample_order[{k!r}]' must be a list of sample identifiers")
                k_s = str(k).strip()
                if k_s in manifest_order:
                    raise ValueError(f"Duplicate group key {k_s!r} in rmats_sample_order.")
                manifest_order[k_s] = [str(x).strip() for x in v]
        comps_raw = bm.get("comparisons", [])
        if comps_raw is not None and not isinstance(comps_raw, list):
            raise ValueError("'comparisons' must be a list")
        seen_cids: set[str] = set()
        for cm in comps_raw or []:
            if not isinstance(cm, dict) or not cm.get("comparison_id"):
                continue
            cid_m = str(cm["comparison_id"]).strip()
            cid_low = cid_m.lower()
            if cid_low in seen_cids:
                raise ValueError(f"Duplicate normalized comparison_id {cid_m!r} in manifest.json comparisons.")
            seen_cids.add(cid_low)
            cm_ord_raw = cm.get("rmats_sample_order")
            cm_ord: dict[str, list[str]] = {}
            if cm_ord_raw is not None:
                if not isinstance(cm_ord_raw, dict):
                    raise ValueError(f"comparisons[{cid_m!r}].rmats_sample_order must be an object")
                for k, v in cm_ord_raw.items():
                    if not isinstance(v, list):
                        raise ValueError(f"comparisons[{cid_m!r}].rmats_sample_order[{k!r}] must be a list")
                    k_s = str(k).strip()
                    if k_s in cm_ord:
                        raise ValueError(f"Duplicate group key {k_s!r} in comparisons[{cid_m!r}].rmats_sample_order.")
                    cm_ord[k_s] = [str(x).strip() for x in v]
            comp_rec = {
                "rmats_group1": str(cm["rmats_group1"]).strip() if cm.get("rmats_group1") else None,
                "rmats_group2": str(cm["rmats_group2"]).strip() if cm.get("rmats_group2") else None,
                "rmats_dir": str(cm["rmats_dir"]).strip() if cm.get("rmats_dir") else None,
                "rmats_sample_order": cm_ord,
            }
            manifest_comps[cid_m] = comp_rec
            manifest_comps.setdefault(cid_low, comp_rec)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        raise ValueError(f"Malformed manifest.json in {mpath}: {exc}") from exc
    return manifest_order, manifest_comps


def _read_delimited(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)


def _validate_manifest_sample_order(
    arm1_ids: list[str],
    arm2_ids: list[str],
    arm1_lbl: str,
    arm2_lbl: str,
    sample_rows: list[dict[str, str]],
    cid: str,
) -> None:
    if not arm1_ids or not arm2_ids:
        raise ValueError(f"manifest.json rmats_sample_order for '{cid}' must specify non-empty sample lists for both '{arm1_lbl}' and '{arm2_lbl}'.")
    a1 = [str(x).strip() for x in arm1_ids]
    a2 = [str(x).strip() for x in arm2_ids]
    if any(not x for x in a1) or any(not x for x in a2):
        raise ValueError(f"manifest.json rmats_sample_order for '{cid}' contains empty sample identifiers.")
    if len(set(a1)) != len(a1) or len(set(a2)) != len(a2):
        raise ValueError(f"manifest.json rmats_sample_order for '{cid}' contains duplicate sample identifiers within an arm.")
    overlap = set(a1) & set(a2)
    if overlap:
        raise ValueError(f"manifest.json rmats_sample_order for '{cid}' assigns sample(s) {sorted(overlap)} to both arms.")
    cond_by_sid = {str(s["sample_id"]): str(s["condition"]) for s in sample_rows}
    unknown = [s for s in a1 + a2 if s not in cond_by_sid]
    if unknown:
        raise ValueError(f"manifest.json rmats_sample_order for '{cid}' references unknown samples absent from samplesheet: {unknown}")
    bad_a1 = [(s, cond_by_sid[s]) for s in a1 if cond_by_sid[s] != arm1_lbl]
    bad_a2 = [(s, cond_by_sid[s]) for s in a2 if cond_by_sid[s] != arm2_lbl]
    if bad_a1 or bad_a2:
        raise ValueError(
            f"manifest.json rmats_sample_order for '{cid}' has condition mismatches against samplesheet: "
            f"expected arm1='{arm1_lbl}' bad={bad_a1}, expected arm2='{arm2_lbl}' bad={bad_a2}"
        )


def extract_matrices_from_bundle_or_rmats(
    *,
    bundle_dir: Path | None = None,
    rmats_dir: Path | None = None,
    samples_path: Path | None = None,
    contrasts_path: Path | None = None,
    rmats_mode: str = "JCEC",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return `(usage_df, samples_df, support_df, gene_map)` from a Katana bundle or rMATS directory."""
    mode = rmats_mode.upper()
    if mode not in {"JC", "JCEC"}:
        raise ValueError(f"Unsupported rmats_mode: {rmats_mode}")
    sample_rows: list[dict[str, str]] = []
    specs: list[tuple[Path, list[str], list[str], str, str, bool]] = []

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
        if s_path is None or c_path is None:
            raise FileNotFoundError(f"Missing samplesheet/contrastsheet in bundle: {bdir}")
        s_df = _read_delimited(s_path)
        for _, r in s_df.iterrows():
            sid = str(r.get("sample") or r.get("sample_id")).strip()
            cond = str(r.get("condition") or r.get("group") or "Group").strip()
            bio = _extract_donor_id(r, sid)
            if sid and not any(x["sample_id"] == sid for x in sample_rows):
                sample_rows.append({"sample_id": sid, "biological_sample_id": bio, "condition": cond})
        manifest_order, manifest_comps = _parse_manifest_rmats(bdir)
        c_df = _read_delimited(c_path)
        n_contrasts = max(1, len(c_df))
        seen_cids: set[str] = set()
        for _, r in c_df.iterrows():
            cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id")).strip()
            if cid.lower() in seen_cids:
                raise ValueError(f"Duplicate normalized contrast_id {cid!r} in contrastsheet ({c_path}).")
            seen_cids.add(cid.lower())
            num_c = str(r.get("treatment") or r.get("numerator") or r.get("group_a")).strip()
            den_c = str(r.get("control") or r.get("denominator") or r.get("group_b")).strip()
            mc = manifest_comps.get(cid) or manifest_comps.get(cid.lower()) or {}
            exp_dir = None
            if mc.get("rmats_dir"):
                raw_d = Path(mc["rmats_dir"])
                exp_dir = raw_d if raw_d.is_absolute() else (bdir / raw_d)
            has_manifest_arms = bool(mc.get("rmats_group1") and mc.get("rmats_group2"))
            g1_lbl = mc.get("rmats_group1") or num_c
            g2_lbl = mc.get("rmats_group2") or den_c
            if has_manifest_arms and {g1_lbl, g2_lbl} != {num_c, den_c}:
                raise ValueError(
                    f"manifest.json rmats_group1/2 ({g1_lbl}, {g2_lbl}) do not match contrast '{cid}' ({num_c}, {den_c})."
                )
            rpath, path_inverted = _resolve_contrast_rmats_dir(
                [bdir / "rmats", bdir / "rnasplicing/rmats"],
                cid=cid,
                num_lbl=num_c,
                den_lbl=den_c,
                n_contrasts=n_contrasts,
                mode=mode,
                explicit_dir=exp_dir,
                allow_bare_root=None,
                explicit_arm_order=has_manifest_arms,
            )
            rev_dir = (g1_lbl == den_c and g2_lbl == num_c) if has_manifest_arms else path_inverted
            cm_order = mc.get("rmats_sample_order") or {}
            arm1_lbl = den_c if rev_dir else num_c
            arm2_lbl = num_c if rev_dir else den_c
            ord1 = cm_order.get(arm1_lbl) if arm1_lbl in cm_order else manifest_order.get(arm1_lbl)
            ord2 = cm_order.get(arm2_lbl) if arm2_lbl in cm_order else manifest_order.get(arm2_lbl)
            if ord1 is not None:
                ord1 = [str(x).strip() for x in ord1]
            if ord2 is not None:
                ord2 = [str(x).strip() for x in ord2]
            explicit_order = bool(cm_order or manifest_order or ord1 is not None or ord2 is not None)
            if explicit_order:
                _validate_manifest_sample_order(ord1 or [], ord2 or [], arm1_lbl, arm2_lbl, sample_rows, cid)
                specs.append((rpath, list(ord1 or []), list(ord2 or []), arm1_lbl, arm2_lbl, True))
            else:
                n1 = [s["sample_id"] for s in sample_rows if s["condition"] == arm1_lbl]
                n2 = [s["sample_id"] for s in sample_rows if s["condition"] == arm2_lbl]
                specs.append((rpath, n1, n2, arm1_lbl, arm2_lbl, False))
    elif rmats_dir is not None:
        rpath = Path(rmats_dir)
        if samples_path is not None and Path(samples_path).is_file():
            s_df = _read_delimited(Path(samples_path))
            for _, r in s_df.iterrows():
                sid = str(r.get("sample_id") or r.get("sample")).strip()
                bio = _extract_donor_id(r, sid)
                cond = str(r.get("condition") or r.get("group") or "Group").strip()
                if sid and not any(x["sample_id"] == sid for x in sample_rows):
                    sample_rows.append({"sample_id": sid, "biological_sample_id": bio, "condition": cond})
        if contrasts_path is not None and Path(contrasts_path).is_file() and sample_rows:
            c_df = _read_delimited(Path(contrasts_path))
            n_contrasts = max(1, len(c_df))
            seen_cids_r: set[str] = set()
            for _, r in c_df.iterrows():
                cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id")).strip()
                if cid.lower() in seen_cids_r:
                    raise ValueError(f"Duplicate normalized contrast_id {cid!r} in contrasts file ({contrasts_path}).")
                seen_cids_r.add(cid.lower())
                num_c = str(r.get("treatment") or r.get("numerator") or r.get("group_a")).strip()
                den_c = str(r.get("control") or r.get("denominator") or r.get("group_b")).strip()
                cand, rev_dir = _resolve_contrast_rmats_dir(
                    [rpath],
                    cid=cid,
                    num_lbl=num_c,
                    den_lbl=den_c,
                    n_contrasts=n_contrasts,
                    mode=mode,
                    allow_bare_root=rpath,
                )
                n1 = [s["sample_id"] for s in sample_rows if s["condition"] == num_c]
                n2 = [s["sample_id"] for s in sample_rows if s["condition"] == den_c]
                if rev_dir:
                    specs.append((cand, n2, n1, den_c, num_c, False))
                else:
                    specs.append((cand, n1, n2, num_c, den_c, False))
        else:
            conds = list(dict.fromkeys(s["condition"] for s in sample_rows))
            if len(conds) >= 2:
                dir_label = _dir_contrast_label(rpath)
                dl_low = dir_label.lower()
                matched_pairs = [
                    (ca, cb)
                    for i, ca in enumerate(conds)
                    for j, cb in enumerate(conds)
                    if i != j and dl_low in {f"{ca}-{cb}".lower(), f"{ca}_vs_{cb}".lower(), f"{ca}_{cb}".lower()}
                ]
                if len(matched_pairs) == 1:
                    arm1_lbl, arm2_lbl = matched_pairs[0]
                elif len(matched_pairs) > 1:
                    raise ValueError(f"Ambiguous rMATS directory label {dir_label!r} matched multiple condition pairs: {matched_pairs}.")
                elif len(conds) == 2:
                    arm1_lbl, arm2_lbl = conds[0], conds[1]
                else:
                    raise ValueError(f"Samplesheet has >2 conditions ({conds}) without contrasts file or matching rMATS directory {rpath}.")
                n1 = [s["sample_id"] for s in sample_rows if s["condition"] == arm1_lbl]
                n2 = [s["sample_id"] for s in sample_rows if s["condition"] == arm2_lbl]
                specs.append((rpath, n1, n2, arm1_lbl, arm2_lbl, False))
            else:
                specs.append((rpath, [], [], "Group1", "Group2", False))
    else:
        raise ValueError("Provide either bundle_dir or rmats_dir.")

    has_declared_samples = bool(sample_rows)

    psi_obs: dict[tuple[str, str], list[float]] = {}
    sup_obs: dict[tuple[str, str], list[float]] = {}
    genes: dict[str, str] = {}

    for rpath, n1_decl, n2_decl, lbl1, lbl2, explicit_order in specs:
        eff_mode = mode
        for et in EVENT_TYPES:
            fp = rpath / f"{et}.MATS.{eff_mode}.txt"
            if not fp.is_file():
                continue
            _str_cols = {
                c: str
                for c in ("IncLevel1", "IncLevel2", "IJC_SAMPLE_1", "IJC_SAMPLE_2", "SJC_SAMPLE_1", "SJC_SAMPLE_2")
            }
            df = pd.read_csv(fp, sep="\t", converters=_str_cols)
            if df.empty:
                continue
            eids = _canon_ids(df, et)
            gids = df["GeneID"].map(_norm_gene) if "GeneID" in df.columns else pd.Series("", index=df.index)
            p1_0 = _parse_nums(df["IncLevel1"].iloc[0]) or _parse_nums(df["IJC_SAMPLE_1"].iloc[0])
            p2_0 = _parse_nums(df["IncLevel2"].iloc[0]) or _parse_nums(df["IJC_SAMPLE_2"].iloc[0])
            if (explicit_order or has_declared_samples) and (len(n1_decl) != len(p1_0) or len(n2_decl) != len(p2_0)):
                raise ValueError(
                    f"Declared sample count mismatch in {fp.name} ({rpath}): "
                    f"{lbl1}={len(n1_decl)} vs rMATS={len(p1_0)}, {lbl2}={len(n2_decl)} vs rMATS={len(p2_0)}."
                )
            n1 = n1_decl if len(n1_decl) == len(p1_0) else [f"{lbl1}_rep{i+1}" for i in range(len(p1_0))]
            n2 = n2_decl if len(n2_decl) == len(p2_0) else [f"{lbl2}_rep{i+1}" for i in range(len(p2_0))]
            for s in n1:
                if not any(x["sample_id"] == s for x in sample_rows):
                    sample_rows.append({"sample_id": s, "biological_sample_id": s, "condition": lbl1})
            for s in n2:
                if not any(x["sample_id"] == s for x in sample_rows):
                    sample_rows.append({"sample_id": s, "biological_sample_id": s, "condition": lbl2})

            exp_n1, exp_n2 = len(n1), len(n2)
            p1_lists = df["IncLevel1"].map(_parse_nums).tolist()
            p2_lists = df["IncLevel2"].map(_parse_nums).tolist()
            i1_lists = df["IJC_SAMPLE_1"].map(_parse_nums).tolist()
            i2_lists = df["IJC_SAMPLE_2"].map(_parse_nums).tolist()
            s1_lists = df["SJC_SAMPLE_1"].map(_parse_nums).tolist()
            s2_lists = df["SJC_SAMPLE_2"].map(_parse_nums).tolist()

            for r_idx, (eid, gid, p1s, p2s, i1s, i2s, s1s, s2s) in enumerate(zip(
                eids, gids, p1_lists, p2_lists, i1_lists, i2_lists, s1_lists, s2_lists,
            )):
                if (
                    len(p1s) != exp_n1 or len(i1s) != exp_n1 or len(s1s) != exp_n1
                    or len(p2s) != exp_n2 or len(i2s) != exp_n2 or len(s2s) != exp_n2
                ):
                    raise ValueError(
                        f"Malformed rMATS replicate count list at {fp.name} row {r_idx} ({eid}): "
                        f"expected (g1={exp_n1}, g2={exp_n2}), got IncLevel=({len(p1s)},{len(p2s)}), "
                        f"IJC=({len(i1s)},{len(i2s)}), SJC=({len(s1s)},{len(s2s)})."
                    )
                genes.setdefault(eid, gid)
                for s, pv, iv, sv in zip(n1, p1s, i1s, s1s):
                    psi_obs.setdefault((eid, s), []).append(pv)
                    sup_obs.setdefault((eid, s), []).append((iv + sv) if (math.isfinite(iv) and math.isfinite(sv)) else math.nan)
                for s, pv, iv, sv in zip(n2, p2s, i2s, s2s):
                    psi_obs.setdefault((eid, s), []).append(pv)
                    sup_obs.setdefault((eid, s), []).append((iv + sv) if (math.isfinite(iv) and math.isfinite(sv)) else math.nan)

    if not psi_obs:
        raise ValueError("No non-empty rMATS tables found to construct co-splicing matrices.")
    usage = pd.Series({k: _med(v) for k, v in psi_obs.items()}, dtype=float).unstack().sort_index()
    support = pd.Series({k: _med(v) for k, v in sup_obs.items()}, dtype=float).unstack().sort_index()
    usage.index.name = "event_key"
    support.index.name = "event_key"
    meta_df = pd.DataFrame(sample_rows)
    gene_ser = pd.Series(genes, name="gene_id")
    return usage, meta_df, support, gene_ser
