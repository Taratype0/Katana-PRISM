"""Lightweight rMATS & Katana-bundle PSI/support matrix extractor for standalone SpliceNet runs."""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any
import pandas as pd

EVENT_TYPES = ("SE", "RI", "A3SS", "A5SS", "MXE")
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


def extract_matrices_from_bundle_or_rmats(
    *,
    bundle_dir: Path | None = None,
    rmats_dir: Path | None = None,
    samples_path: Path | None = None,
    rmats_mode: str = "JCEC",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return `(usage_df, samples_df, support_df, gene_map)` from a Katana bundle or rMATS directory."""
    mode = rmats_mode.upper()
    sample_rows: list[dict[str, str]] = []
    specs: list[tuple[Path, list[str], list[str], str, str]] = []

    if bundle_dir is not None:
        bdir = Path(bundle_dir)
        s_path = next((p for p in (
            bdir / "pipeline_info" / "samplesheet.valid.csv",
            bdir / "inputs" / "samplesheet.csv",
            bdir / "samplesheet.csv",
        ) if p.is_file()), None)
        c_path = next((p for p in (
            bdir / "pipeline_info" / "contrastsheet.valid.csv",
            bdir / "inputs" / "contrastsheet.csv",
            bdir / "contrastsheet.csv",
        ) if p.is_file()), None)
        if s_path is None or c_path is None:
            raise FileNotFoundError(f"Missing samplesheet/contrastsheet in bundle: {bdir}")
        s_df = pd.read_csv(s_path)
        for _, r in s_df.iterrows():
            sid = str(r.get("sample") or r.get("sample_id")).strip()
            cond = str(r.get("condition") or "Group").strip()
            if sid and not any(x["sample_id"] == sid for x in sample_rows):
                sample_rows.append({"sample_id": sid, "biological_sample_id": sid, "condition": cond})
        c_df = pd.read_csv(c_path)
        for _, r in c_df.iterrows():
            cid = str(r.get("contrast") or r.get("contrast_id")).strip()
            num_c = str(r.get("treatment") or r.get("numerator")).strip()
            den_c = str(r.get("control") or r.get("denominator")).strip()
            rpath_candidates = (
                (bdir / "rmats" / cid / "rmats_post", False),
                (bdir / "rmats" / cid, False),
                (bdir / "rnasplicing/rmats" / f"{num_c}-{den_c}" / "rmats_post", False),
                (bdir / "rnasplicing/rmats" / cid / "rmats_post", False),
                (bdir / "rnasplicing/rmats" / f"{den_c}-{num_c}" / "rmats_post", True),
            )
            rpath, rev_dir = next(
                ((p, inv) for p, inv in rpath_candidates if p.is_dir()),
                (bdir / "rmats" / cid / "rmats_post", False),
            )
            n1 = [s["sample_id"] for s in sample_rows if s["condition"] == num_c]
            n2 = [s["sample_id"] for s in sample_rows if s["condition"] == den_c]
            if rev_dir:
                specs.append((rpath, n2, n1, den_c, num_c))
            else:
                specs.append((rpath, n1, n2, num_c, den_c))
    elif rmats_dir is not None:
        rpath = Path(rmats_dir)
        if samples_path is not None and Path(samples_path).is_file():
            sep = "\t" if str(samples_path).endswith(".tsv") else ","
            s_df = pd.read_csv(samples_path, sep=sep)
            for _, r in s_df.iterrows():
                sid = str(r.get("sample_id") or r.get("sample")).strip()
                bio = str(r.get("biological_sample_id") or sid).strip()
                cond = str(r.get("condition") or r.get("group") or "Group").strip()
                if sid and not any(x["sample_id"] == sid for x in sample_rows):
                    sample_rows.append({"sample_id": sid, "biological_sample_id": bio, "condition": cond})
        conds = list(dict.fromkeys(s["condition"] for s in sample_rows))
        if len(conds) >= 2:
            n1 = [s["sample_id"] for s in sample_rows if s["condition"] == conds[0]]
            n2 = [s["sample_id"] for s in sample_rows if s["condition"] == conds[1]]
            specs.append((rpath, n1, n2, conds[0], conds[1]))
        else:
            specs.append((rpath, [], [], "Group1", "Group2"))
    else:
        raise ValueError("Provide either bundle_dir or rmats_dir.")

    psi_obs: dict[tuple[str, str], list[float]] = {}
    sup_obs: dict[tuple[str, str], list[float]] = {}
    genes: dict[str, str] = {}

    for rpath, n1_decl, n2_decl, lbl1, lbl2 in specs:
        for et in EVENT_TYPES:
            fp = rpath / f"{et}.MATS.{mode}.txt"
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
