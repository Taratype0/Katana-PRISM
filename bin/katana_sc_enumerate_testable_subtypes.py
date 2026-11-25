#!/usr/bin/env python3
"""Enumerate testable single-cell population & subtype contrasts directly from the user's
QC-filtered, cell-typed metadata table (`--sc_celltype_cols`), verifying that both contrast arms
contain at least `--min-donors-per-arm` independent biological donors/animals.

Preserves 100% of the user's cell-type/subclass annotations while logging rare subtypes that lack
biological replication into `sc_subtype_donor_audit.tsv` without halting workflow execution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import pandas as pd


_DONOR_COL_ALIASES = (
    "donor_id",
    "mouse_id",
    "biological_sample_id",
    "donor",
    "subject_id",
    "subject",
    "mouse",
    "patient_id",
    "patient",
    "individual",
    "individual_id",
    "sample_donor",
    "animal_id",
)
_MISSING_DONOR_TOKENS = {"", "nan", "none", "na", "null", "n/a"}


def _valid_donors(series: pd.Series) -> list[str]:
    out: list[str] = []
    for v in series.tolist():
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s.lower() not in _MISSING_DONOR_TOKENS:
            out.append(s)
    return out


def _norm_donors(series: pd.Series) -> list[str]:
    return [
        "" if (pd.isna(v) or str(v).strip().lower() in _MISSING_DONOR_TOKENS) else str(v).strip()
        for v in series.tolist()
    ]


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_")


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit donor replication per user-annotated cell subtype and emit testable contrasts.")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--celltype-cols", required=True, help="Comma-separated metadata columns (e.g., cell_type,atlas_subclass)")
    ap.add_argument("--group-col", default="condition")
    ap.add_argument("--donor-col", default=None, help="Explicit donor/animal column name (auto-detects canonical aliases if omitted).")
    ap.add_argument("--numerator", default="")
    ap.add_argument("--denominator", default="")
    ap.add_argument("--condition-pairs", default="", help="Optional comma-separated NUM:DEN pairs (e.g., KO_Stim:WT_Stim,KO_Basal:WT_Basal)")
    ap.add_argument("--min-donors-per-arm", type=int, default=2)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    sep = "," if args.metadata.suffix.lower() == ".csv" else "\t"
    md = pd.read_csv(args.metadata, sep=sep, dtype=str, keep_default_na=False)
    for qc_col in ("qc_pass", "qc_pass_nuclei", "keep"):
        if qc_col in md.columns:
            md = md[md[qc_col].astype(str).str.lower().isin(("true", "1", "yes", "pass"))].copy()
            break

    cols_l = {str(c).strip().lower(): str(c) for c in md.columns}
    cid_col = next((cols_l[k] for k in ("cell_id", "cell_barcode", "barcode", "sample_id", "sample.id") if k in cols_l), None)
    if cid_col is not None:
        cids_clean = md[cid_col].astype(str).str.strip()
        dup_cid = cids_clean.duplicated(keep=False)
        if dup_cid.any():
            bad_ids = cids_clean[dup_cid].drop_duplicates().head(5).tolist()
            raise ValueError(f"Duplicate physical cell_id entries in metadata (e.g. {bad_ids}).")

    if args.donor_col and str(args.donor_col).strip():
        req_d = str(args.donor_col).strip()
        donor_col = req_d if req_d in md.columns else cols_l.get(req_d.lower())
        if donor_col is None:
            raise ValueError(f"Donor column {args.donor_col!r} not found in metadata columns: {list(md.columns)}")
    else:
        populated_cols: list[str] = []
        present_cols: list[str] = []
        norm_by_col: dict[str, list[str]] = {}
        for d_alias in _DONOR_COL_ALIASES:
            orig_d = cols_l.get(d_alias)
            if orig_d is not None and orig_d not in present_cols:
                present_cols.append(orig_d)
                norm_v = _norm_donors(md[orig_d])
                if any(v != "" for v in norm_v):
                    populated_cols.append(orig_d)
                    norm_by_col[orig_d] = norm_v
        if len(populated_cols) >= 2:
            base_col = populated_cols[0]
            conflicts = [c for c in populated_cols[1:] if norm_by_col[c] != norm_by_col[base_col]]
            if conflicts:
                raise ValueError(
                    f"Conflicting populated donor identifier columns in metadata: {[base_col, *conflicts]}. "
                    "Pass explicit --donor-col to disambiguate."
                )
        donor_col = populated_cols[0] if populated_cols else (present_cols[0] if present_cols else None)
        if donor_col is None:
            raise ValueError(f"Metadata {args.metadata} requires a donor/animal identifier column (donor_id or mouse_id).")
    if args.group_col not in md.columns:
        raise ValueError(f"Group column {args.group_col!r} not found in metadata columns: {list(md.columns)}")

    pairs: list[tuple[str, str]] = []
    if args.condition_pairs.strip():
        for tok in args.condition_pairs.split(","):
            if ":" in tok:
                n_s, d_s = tok.split(":", 1)
                if n_s.strip() and d_s.strip():
                    pairs.append((n_s.strip(), d_s.strip()))
    if not pairs and args.numerator.strip() and args.denominator.strip():
        pairs.append((args.numerator.strip(), args.denominator.strip()))
    if not pairs:
        raise ValueError("Provide either --condition-pairs 'NUM1:DEN1,NUM2:DEN2' or --numerator / --denominator.")

    populations: list[tuple[str, str, str]] = [("All_QC_Cells", "ALL", "ALL")]
    skip_labels = {"", "na", "nan", "none", "unknown", "uncertain", "uncertain_subclass", "unassigned"}
    for col in [c.strip() for c in args.celltype_cols.split(",") if c.strip()]:
        if col not in md.columns:
            continue
        vals = sorted(md[col].dropna().astype(str).unique().tolist(), key=lambda v: -int(md[col].astype(str).eq(v).sum()))
        for v in vals:
            if v.strip().lower() in skip_labels:
                continue
            populations.append((slugify(v), col, v))

    contrast_rows = []
    audit_rows = []
    seen_cids: set[str] = set()
    for pop_slug, pop_col, pop_val in populations:
        sub = md if pop_col == "ALL" else md[md[pop_col].astype(str).eq(pop_val)]
        for num, den in pairs:
            s_num = sub[sub[args.group_col].astype(str).eq(num)]
            s_den = sub[sub[args.group_col].astype(str).eq(den)]
            valid_num = _valid_donors(s_num[donor_col])
            valid_den = _valid_donors(s_den[donor_col])
            d_num = len(set(valid_num))
            d_den = len(set(valid_den))
            testable = (d_num >= args.min_donors_per_arm) and (d_den >= args.min_donors_per_arm)
            cid = f"{pop_slug}__{slugify(num)}_vs_{slugify(den)}"
            if cid in seen_cids:
                continue
            seen_cids.add(cid)

            if testable:
                contrast_rows.append({
                    "contrast_id": cid,
                    "population_col": pop_col,
                    "population_value": pop_val,
                    "numerator": num,
                    "denominator": den,
                })
            audit_rows.append({
                "contrast_id": cid,
                "population_col": pop_col,
                "population_value": pop_val,
                "numerator": num,
                "denominator": den,
                "n_cells_total_in_population": int(len(sub)),
                "n_cells_numerator": int(len(s_num)),
                "n_cells_denominator": int(len(s_den)),
                "n_donors_numerator": d_num,
                "n_donors_denominator": d_den,
                "min_donors_required": int(args.min_donors_per_arm),
                "testable": testable,
                "status": "testable" if testable else "skipped_insufficient_donors_per_arm",
            })

    pd.DataFrame(
        contrast_rows,
        columns=["contrast_id", "population_col", "population_value", "numerator", "denominator"],
    ).to_csv(args.outdir / "testable_subtype_contrasts.tsv", sep="\t", index=False)
    pd.DataFrame(audit_rows).to_csv(args.outdir / "sc_subtype_donor_audit.tsv", sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
