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
    modes = tuple(dict.fromkeys((mode.upper(), "JCEC", "JC")))
    return any((d / f"{et}.MATS.{m}.txt").is_file() for et in EVENT_TYPES for m in modes)


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
) -> tuple[Path, bool]:
    """Strictly match a declared contrast to its rMATS directory without multi-contrast bare reuse."""
    if explicit_dir is not None and _has_rmats_files(explicit_dir, mode):
        rev_name = explicit_dir.name in {f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}"} or (
            explicit_dir.name == "rmats_post"
            and explicit_dir.parent.name in {f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}"}
        )
        return explicit_dir, rev_name

    fwd_names = tuple(dict.fromkeys(k for k in (cid, f"{num_lbl}-{den_lbl}", f"{num_lbl}_vs_{den_lbl}") if k))
    rev_names = tuple(dict.fromkeys(k for k in (f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}") if k))

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
    rev_low = {k.lower() for k in rev_names}
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
            return allow_bare_root, False
        if len(available_subs) == 1:
            return available_subs[0], False

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
        if isinstance(bm.get("rmats_sample_order"), dict):
            manifest_order = {
                str(k): [str(x) for x in v]
                for k, v in bm["rmats_sample_order"].items()
                if isinstance(v, list)
            }
        for cm in bm.get("comparisons", []):
            if not isinstance(cm, dict) or not cm.get("comparison_id"):
                continue
            cid_m = str(cm["comparison_id"]).strip()
            comp_rec = {
                "rmats_group1": str(cm["rmats_group1"]).strip() if cm.get("rmats_group1") else None,
                "rmats_group2": str(cm["rmats_group2"]).strip() if cm.get("rmats_group2") else None,
                "rmats_dir": str(cm["rmats_dir"]).strip() if cm.get("rmats_dir") else None,
                "rmats_sample_order": (
                    {str(k): [str(x) for x in v] for k, v in cm["rmats_sample_order"].items() if isinstance(v, list)}
                    if isinstance(cm.get("rmats_sample_order"), dict)
                    else {}
                ),
            }
            manifest_comps[cid_m] = comp_rec
            manifest_comps.setdefault(cid_m.lower(), comp_rec)
    except Exception:
        pass
    return manifest_order, manifest_comps


def _read_delimited(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


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
        for _, r in c_df.iterrows():
            cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id")).strip()
            num_c = str(r.get("treatment") or r.get("numerator") or r.get("group_a")).strip()
            den_c = str(r.get("control") or r.get("denominator") or r.get("group_b")).strip()
            mc = manifest_comps.get(cid) or manifest_comps.get(cid.lower()) or {}
            exp_dir = None
            if mc.get("rmats_dir"):
                raw_d = Path(mc["rmats_dir"])
                exp_dir = raw_d if raw_d.is_absolute() else (bdir / raw_d)
            rpath, path_inverted = _resolve_contrast_rmats_dir(
                [bdir / "rmats", bdir / "rnasplicing/rmats"],
                cid=cid,
                num_lbl=num_c,
                den_lbl=den_c,
                n_contrasts=n_contrasts,
                mode=mode,
                explicit_dir=exp_dir,
                allow_bare_root=None,
            )
            g1_lbl = mc.get("rmats_group1") or num_c
            g2_lbl = mc.get("rmats_group2") or den_c
            has_manifest_arms = bool(mc.get("rmats_group1") and mc.get("rmats_group2"))
            rev_dir = (g1_lbl == den_c and g2_lbl == num_c) if has_manifest_arms else path_inverted
            cm_order = mc.get("rmats_sample_order") or {}
            num_ord = cm_order.get(num_c) or manifest_order.get(num_c)
            den_ord = cm_order.get(den_c) or manifest_order.get(den_c)
            n1 = num_ord or [s["sample_id"] for s in sample_rows if s["condition"] == num_c]
            n2 = den_ord or [s["sample_id"] for s in sample_rows if s["condition"] == den_c]
            explicit_order = bool(num_ord or den_ord)
            if rev_dir:
                specs.append((rpath, n2, n1, den_c, num_c, explicit_order))
            else:
                specs.append((rpath, n1, n2, num_c, den_c, explicit_order))
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
            for _, r in c_df.iterrows():
                cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id")).strip()
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
                n1 = [s["sample_id"] for s in sample_rows if s["condition"] == conds[0]]
                n2 = [s["sample_id"] for s in sample_rows if s["condition"] == conds[1]]
                specs.append((rpath, n1, n2, conds[0], conds[1], False))
            else:
                specs.append((rpath, [], [], "Group1", "Group2", False))
    else:
        raise ValueError("Provide either bundle_dir or rmats_dir.")

    psi_obs: dict[tuple[str, str], list[float]] = {}
    sup_obs: dict[tuple[str, str], list[float]] = {}
    genes: dict[str, str] = {}

    for rpath, n1_decl, n2_decl, lbl1, lbl2, explicit_order in specs:
        eff_mode = (
            mode
            if any((rpath / f"{et}.MATS.{mode}.txt").is_file() for et in EVENT_TYPES)
            else next((m for m in ("JCEC", "JC") if any((rpath / f"{et}.MATS.{m}.txt").is_file() for et in EVENT_TYPES)), mode)
        )
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
            if explicit_order and ((n1_decl and len(n1_decl) != len(p1_0)) or (n2_decl and len(n2_decl) != len(p2_0))):
                raise ValueError(
                    f"manifest.json rmats_sample_order length mismatch in {rpath}: "
                    f"group1={len(n1_decl)} vs rMATS={len(p1_0)}, group2={len(n2_decl)} vs rMATS={len(p2_0)}"
                )
            n1 = n1_decl if len(n1_decl) == len(p1_0) else [f"{lbl1}_rep{i+1}" for i in range(len(p1_0))]
            n2 = n2_decl if len(n2_decl) == len(p2_0) else [f"{lbl2}_rep{i+1}" for i in range(len(p2_0))]
            for s in n1:
                if not any(x["sample_id"] == s for x in sample_rows):
                    sample_rows.append({"sample_id": s, "biological_sample_id": s, "condition": lbl1})
            for s in n2:
                if not any(x["sample_id"] == s for x in sample_rows):
                    sample_rows.append({"sample_id": s, "biological_sample_id": s, "condition": lbl2})

            for eid, gid, p1s, p2s, i1s, i2s, s1s, s2s in zip(
                eids, gids,
                df["IncLevel1"].map(_parse_nums), df["IncLevel2"].map(_parse_nums),
                df["IJC_SAMPLE_1"].map(_parse_nums), df["IJC_SAMPLE_2"].map(_parse_nums),
                df["SJC_SAMPLE_1"].map(_parse_nums), df["SJC_SAMPLE_2"].map(_parse_nums),
            ):
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
