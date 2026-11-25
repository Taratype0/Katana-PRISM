#!/usr/bin/env python3
"""Build canonical 0-based single-cell alternative splicing matrices (`sc_events.tsv`, `sc_event_genes.tsv`,
`sc_psi.tsv.gz`, `sc_support.tsv.gz`, and standardized `sc_cells.tsv`) directly from STARsolo `SJ/raw` output
or per-cell `*.SJ.out.tab` files plus a GENCODE/Ensembl GTF annotation and cell metadata table.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
from pathlib import Path
import re
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, hstack
from scipy.stats import mannwhitneyu


_ATTR_RE = re.compile(r'([A-Za-z0-9_]+)\s+"([^"]+)"')


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def build_events_from_gtf(gtf_path: Path) -> tuple[pd.DataFrame, dict[tuple[str, int, int], int]]:
    """Extract canonical 0-based binary SE, A5SS, and A3SS events and unique splice junctions from GTF."""
    tx_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    tx_info: dict[str, tuple[str, str, str, str]] = {}

    with _open_text(gtf_path) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "exon":
                continue
            chrom, _, _, start_s, end_s, _, strand, _, attrs_s = parts
            if strand not in ("+", "-"):
                continue
            attrs = dict(_ATTR_RE.findall(attrs_s))
            tid = attrs.get("transcript_id", "")
            gid = attrs.get("gene_id", "")
            if not tid or not gid:
                continue
            gsym = attrs.get("gene_name") or attrs.get("gene_symbol") or gid.split(".")[0]
            s0 = int(start_s) - 1
            e1 = int(end_s)
            if e1 <= s0:
                continue
            tx_exons[tid].append((s0, e1))
            if tid not in tx_info:
                tx_info[tid] = (gid, gsym, chrom, strand)

    gene_txs: dict[tuple[str, str, str, str], list[list[tuple[int, int]]]] = defaultdict(list)
    for tid, exs in tx_exons.items():
        if len(exs) >= 2:
            exs_sorted = sorted(set(exs))
            gene_txs[tx_info[tid]].append(exs_sorted)

    events: dict[str, dict] = {}
    needed_juncs: set[tuple[str, int, int]] = set()

    for (gid, gsym, chrom, strand), tx_list in gene_txs.items():
        skip_introns: dict[tuple[int, int], tuple[int, int]] = {}
        triplets: list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]] = []
        intron_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []

        for exs in tx_list:
            for k in range(len(exs) - 1):
                u_ex, d_ex = exs[k], exs[k + 1]
                if d_ex[0] > u_ex[1]:
                    skip_introns[(u_ex[1], d_ex[0])] = (u_ex[0], d_ex[1])
                    intron_pairs.append((u_ex, d_ex))
            for k in range(len(exs) - 2):
                u_ex, c_ex, d_ex = exs[k], exs[k + 1], exs[k + 2]
                if u_ex[1] < c_ex[0] and c_ex[1] < d_ex[0]:
                    triplets.append((u_ex, c_ex, d_ex))

        # 1. Cassette / Skipped Exon (SE)
        for u_ex, c_ex, d_ex in triplets:
            key_skip = (u_ex[1], d_ex[0])
            if key_skip in skip_introns:
                eid = f"SE:{chrom}:{strand}:{c_ex[0]}:{c_ex[1]}:{u_ex[0]}:{u_ex[1]}:{d_ex[0]}:{d_ex[1]}"
                if eid not in events:
                    j_inc1 = (chrom, u_ex[1] + 1, c_ex[0])
                    j_inc2 = (chrom, c_ex[1] + 1, d_ex[0])
                    j_exc = (chrom, u_ex[1] + 1, d_ex[0])
                    needed_juncs.update((j_inc1, j_inc2, j_exc))
                    events[eid] = {
                        "event_id": eid,
                        "event_type": "SE",
                        "gene_id": gid,
                        "gene_symbol": gsym,
                        "chr": chrom,
                        "strand": strand,
                        "exonStart_0base": c_ex[0],
                        "exonEnd": c_ex[1],
                        "upstreamES": u_ex[0],
                        "upstreamEE": u_ex[1],
                        "downstreamES": d_ex[0],
                        "downstreamEE": d_ex[1],
                        "_inc_juncs": [j_inc1, j_inc2],
                        "_exc_juncs": [j_exc],
                    }

        # 2. Alternative 5' / 3' Splice Sites (A5SS / A3SS)
        by_right: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
        by_left: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = defaultdict(list)
        for u_ex, d_ex in set(intron_pairs):
            by_right[d_ex[0]].append((u_ex, d_ex))
            by_left[u_ex[1]].append((u_ex, d_ex))

        # Shared right intron boundary (`d_ex[0]`), variable left exon end (`u_ex[1]`): A5SS on '+', A3SS on '-'
        for right_s0, plist in by_right.items():
            uniq_ends = sorted({p[0][1] for p in plist})
            if len(uniq_ends) == 2:
                short_e, long_e = uniq_ends[0], uniq_ends[1]
                u_short = min((p[0] for p in plist if p[0][1] == short_e), key=lambda x: x[0])
                u_long = min((p[0] for p in plist if p[0][1] == long_e), key=lambda x: x[0])
                if max(u_short[0], u_long[0]) >= min(u_short[1], u_long[1]):
                    continue
                d_ref = max((p[1] for p in plist), key=lambda x: x[1])
                ls0 = min(u_short[0], u_long[0])
                etype = "A5SS" if strand == "+" else "A3SS"
                eid = f"{etype}:{chrom}:{strand}:{ls0}:{long_e}:{ls0}:{short_e}:{right_s0}:{d_ref[1]}"
                if eid not in events:
                    j_long = (chrom, long_e + 1, right_s0)
                    j_short = (chrom, short_e + 1, right_s0)
                    needed_juncs.update((j_long, j_short))
                    events[eid] = {
                        "event_id": eid,
                        "event_type": etype,
                        "gene_id": gid,
                        "gene_symbol": gsym,
                        "chr": chrom,
                        "strand": strand,
                        "exonStart_0base": ls0,
                        "exonEnd": long_e,
                        "upstreamES": ls0,
                        "upstreamEE": short_e,
                        "downstreamES": right_s0,
                        "downstreamEE": d_ref[1],
                        "longExonStart_0base": ls0,
                        "longExonEnd": long_e,
                        "shortES": ls0,
                        "shortEE": short_e,
                        "flankingES": right_s0,
                        "flankingEE": d_ref[1],
                        "_inc_juncs": [j_long],
                        "_exc_juncs": [j_short],
                    }

        # Shared left intron boundary (`u_ex[1]`), variable right exon start (`d_ex[0]`): A3SS on '+', A5SS on '-'
        for left_e, plist in by_left.items():
            uniq_starts = sorted({p[1][0] for p in plist})
            if len(uniq_starts) == 2:
                long_s0, short_s0 = uniq_starts[0], uniq_starts[1]
                d_long = max((p[1] for p in plist if p[1][0] == long_s0), key=lambda x: x[1])
                d_short = max((p[1] for p in plist if p[1][0] == short_s0), key=lambda x: x[1])
                if max(d_long[0], d_short[0]) >= min(d_long[1], d_short[1]):
                    continue
                u_ref = min((p[0] for p in plist), key=lambda x: x[0])
                le_end = max(d_long[1], d_short[1])
                etype = "A3SS" if strand == "+" else "A5SS"
                eid = f"{etype}:{chrom}:{strand}:{long_s0}:{le_end}:{short_s0}:{le_end}:{u_ref[0]}:{left_e}"
                if eid not in events:
                    j_long = (chrom, left_e + 1, long_s0)
                    j_short = (chrom, left_e + 1, short_s0)
                    needed_juncs.update((j_long, j_short))
                    events[eid] = {
                        "event_id": eid,
                        "event_type": etype,
                        "gene_id": gid,
                        "gene_symbol": gsym,
                        "chr": chrom,
                        "strand": strand,
                        "exonStart_0base": long_s0,
                        "exonEnd": le_end,
                        "upstreamES": u_ref[0],
                        "upstreamEE": left_e,
                        "downstreamES": short_s0,
                        "downstreamEE": le_end,
                        "longExonStart_0base": long_s0,
                        "longExonEnd": le_end,
                        "shortES": short_s0,
                        "shortEE": le_end,
                        "flankingES": u_ref[0],
                        "flankingEE": left_e,
                        "_inc_juncs": [j_long],
                        "_exc_juncs": [j_short],
                    }

    junc_list = sorted(needed_juncs)
    junc_to_idx = {j: i for i, j in enumerate(junc_list)}
    ev_df = pd.DataFrame(list(events.values()))
    return ev_df, junc_to_idx


def _norm_chrom(c: str, target_chroms: set[str]) -> str:
    if c in target_chroms:
        return c
    if c.startswith("chr") and c[3:] in target_chroms:
        return c[3:]
    if ("chr" + c) in target_chroms:
        return "chr" + c
    return c


def load_junction_counts(sj_specs: list[str], junc_to_idx: dict[tuple[str, int, int], int]) -> tuple[csr_matrix, list[str]]:
    """Load sparse `junctions x cells` counts from STARsolo `SJ/raw` folders or per-cell `*.SJ.out.tab` folders."""
    target_chroms = {j[0] for j in junc_to_idx}
    n_j = len(junc_to_idx)
    blocks: list[csr_matrix] = []
    all_cells: list[str] = []
    seen_cells: dict[str, str] = {}

    def _register_batch_cells(bcs: list[str], source_label: str) -> None:
        dups: list[str] = []
        for bc in bcs:
            if bc in seen_cells:
                dups.append(f"{bc} ({seen_cells[bc]} vs {source_label})")
            else:
                seen_cells[bc] = source_label
        if dups:
            raise ValueError(
                "Duplicate physical cell_id barcodes detected across raw junction inputs: "
                f"{dups[:5]}; assign unique sj-dir prefixes (`prefix::path`) or deduplicate inputs."
            )

    for spec in sj_specs:
        if "::" in spec:
            prefix, raw_p = spec.split("::", 1)
            prefix = prefix.strip() + "_"
        else:
            prefix, raw_p = "", spec
        p = Path(raw_p)

        # Case 1: STARsolo `Solo.out/SJ/raw` directory (or parent containing `Solo.out/SJ/raw`)
        cand_dirs = [p, p / "Solo.out/SJ/raw", p / "SJ/raw"]
        solo_dir = next((d for d in cand_dirs if (d / "matrix.mtx").is_file() or (d / "matrix.mtx.gz").is_file()), None)
        if solo_dir is not None:
            mtx_f = solo_dir / ("matrix.mtx.gz" if (solo_dir / "matrix.mtx.gz").is_file() else "matrix.mtx")
            feat_f = solo_dir / ("features.tsv.gz" if (solo_dir / "features.tsv.gz").is_file() else "features.tsv")
            bc_f = solo_dir / ("barcodes.tsv.gz" if (solo_dir / "barcodes.tsv.gz").is_file() else "barcodes.tsv")

            mat = mmread(str(mtx_f)).tocsr()
            with _open_text(bc_f) as fh:
                bcs = [prefix + ln.strip().split("\t")[0] for ln in fh if ln.strip()]
            _register_batch_cells(bcs, spec)
            row_map_src: list[int] = []
            row_map_dst: list[int] = []
            with _open_text(feat_f) as fh:
                for r_i, ln in enumerate(fh):
                    pts = ln.rstrip("\n").split("\t")
                    if len(pts) >= 3:
                        ch = _norm_chrom(pts[0], target_chroms)
                        key = (ch, int(pts[1]), int(pts[2]))
                        dst = junc_to_idx.get(key)
                        if dst is not None:
                            row_map_src.append(r_i)
                            row_map_dst.append(dst)
            if row_map_src:
                sub = mat[row_map_src, :].tocoo()
                mapped_rows = np.asarray(row_map_dst, dtype=int)[sub.row]
                proj = csr_matrix((sub.data.astype(np.float32), (mapped_rows, sub.col)), shape=(n_j, len(bcs)))
            else:
                proj = csr_matrix((n_j, len(bcs)), dtype=np.float32)
            blocks.append(proj)
            all_cells.extend(bcs)
            continue

        # Case 2: Directory of per-cell `*.SJ.out.tab` files
        sj_files = sorted(list(p.glob("*.SJ.out.tab")) + list(p.glob("*/*.SJ.out.tab")))
        if not sj_files:
            raise FileNotFoundError(f"No STARsolo matrix.mtx or *.SJ.out.tab found in {p}")
        rows_i, cols_j, vals_v = [], [], []
        bcs = []
        for c_j, sj_f in enumerate(sj_files):
            cid = sj_f.name.replace(".SJ.out.tab", "").rstrip(".")
            if cid == "SJ.out.tab" or not cid:
                cid = sj_f.parent.name
            bcs.append(prefix + cid)
        _register_batch_cells(bcs, spec)
        for c_j, sj_f in enumerate(sj_files):
            with _open_text(sj_f) as fh:
                for ln in fh:
                    pts = ln.rstrip("\n").split("\t")
                    if len(pts) >= 7:
                        ch = _norm_chrom(pts[0], target_chroms)
                        dst = junc_to_idx.get((ch, int(pts[1]), int(pts[2])))
                        if dst is not None:
                            u_cnt = float(pts[6])
                            if u_cnt > 0:
                                rows_i.append(dst)
                                cols_j.append(c_j)
                                vals_v.append(u_cnt)
        proj = csr_matrix((np.asarray(vals_v, dtype=np.float32), (rows_i, cols_j)), shape=(n_j, len(bcs)))
        blocks.append(proj)
        all_cells.extend(bcs)

    full_mat = hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    return full_mat, all_cells


_DONOR_ID_ALIASES = (
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


def _sanitize_donor_values(series: pd.Series) -> pd.Series:
    return pd.Series(
        [
            "" if (pd.isna(v) or str(v).strip().lower() in _MISSING_DONOR_TOKENS) else str(v).strip()
            for v in series.tolist()
        ],
        index=series.index,
        dtype="object",
    )


def standardize_metadata(
    meta_path: Path,
    observed_cells: list[str],
    group_col: str,
    numerator: str | None,
    denominator: str | None,
    *,
    donor_col: str | None = None,
    allow_single_donor: bool = False,
) -> pd.DataFrame:
    sep = "," if meta_path.suffix.lower() == ".csv" else "\t"
    md = pd.read_csv(meta_path, sep=sep, dtype=str, keep_default_na=False)
    cols_l = {str(c).strip().lower(): str(c) for c in md.columns}
    cid_c = next((cols_l[k] for k in ("cell_id", "cell", "barcode", "sample_id", "sample", "id") if k in cols_l), md.columns[0])
    md = md.rename(columns={cid_c: "cell_id"})
    md["cell_id"] = md["cell_id"].astype(str).str.strip()
    dup_cid = md["cell_id"].duplicated(keep=False)
    if dup_cid.any():
        bad_cids = md.loc[dup_cid, "cell_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Duplicate physical cell_id entries in metadata (e.g. {bad_cids}).")

    # Match barcodes cleanly per cell even when some STARsolo barcodes carry `-1` suffixes and others do not
    obs_set = set(observed_cells)
    base_to_obs: dict[str, list[str]] = defaultdict(list)
    for c_obs in observed_cells:
        base_to_obs[c_obs.split("-")[0]].append(c_obs)

    resolved_cids: list[str] = []
    for cid_raw in md["cell_id"]:
        if cid_raw in obs_set:
            resolved_cids.append(cid_raw)
        else:
            cands = base_to_obs.get(cid_raw.split("-")[0], [])
            if len(cands) > 1:
                raise ValueError(
                    f"Duplicate observed cell barcodes after `-1` suffix normalization for {cid_raw!r}: {cands[:5]}."
                )
            resolved_cids.append(cands[0] if len(cands) == 1 else cid_raw)
    md["cell_id"] = resolved_cids
    dup_stripped = md["cell_id"].duplicated(keep=False)
    if dup_stripped.any():
        bad_s = md.loc[dup_stripped, "cell_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Duplicate cell_id entries after barcode suffix normalization (e.g. {bad_s}).")

    cols_l = {str(c).strip().lower(): str(c) for c in md.columns}
    if donor_col is not None and str(donor_col).strip():
        req_d = str(donor_col).strip()
        resolved_don = req_d if req_d in md.columns else cols_l.get(req_d.lower())
        if resolved_don is None:
            raise ValueError(f"Donor column {donor_col!r} not found in metadata columns: {list(md.columns)}")
        md["donor_id"] = _sanitize_donor_values(md[resolved_don])
    else:
        populated_cols: list[str] = []
        present_cols: list[str] = []
        norm_by_col: dict[str, list[str]] = {}
        for d_alias in _DONOR_ID_ALIASES:
            orig_d = cols_l.get(d_alias)
            if orig_d is not None and orig_d not in present_cols:
                present_cols.append(orig_d)
                san = _sanitize_donor_values(md[orig_d]).tolist()
                if any(v != "" for v in san):
                    populated_cols.append(orig_d)
                    norm_by_col[orig_d] = san
        if len(populated_cols) >= 2:
            base_col = populated_cols[0]
            conflicts = [c for c in populated_cols[1:] if norm_by_col[c] != norm_by_col[base_col]]
            if conflicts:
                raise ValueError(
                    f"Conflicting populated donor identifier columns in metadata: {[base_col, *conflicts]}. "
                    "Pass explicit --donor-col to disambiguate."
                )
        if populated_cols:
            md["donor_id"] = _sanitize_donor_values(md[populated_cols[0]])
        elif present_cols:
            md["donor_id"] = _sanitize_donor_values(md[present_cols[0]])
        elif allow_single_donor:
            md["donor_id"] = "Donor1"
        else:
            raise ValueError(
                f"Metadata {meta_path} requires a donor/animal identifier column (e.g., donor_id, mouse_id, biological_sample_id). "
                "Pass --allow-single-donor only for single-donor exploratory analysis."
            )

    if "cell_type" not in md.columns:
        ct_c = next((cols_l[k] for k in ("cell_type", "celltype", "cluster", "annotation", "lineage") if k in cols_l), None)
        md["cell_type"] = md[ct_c].astype(str) if ct_c else md.get("condition", "Cell")
    if "condition" not in md.columns:
        cond_c = next((cols_l[k] for k in ("condition", "group", "state", "treatment", "status") if k in cols_l), None)
        md["condition"] = md[cond_c].astype(str) if cond_c else md["cell_type"].astype(str)

    # If user requested contrast on `group_col` (e.g. comparing two cell_types!), expose that column as `condition` for downstream modules
    if group_col in md.columns and group_col != "condition":
        md["original_condition"] = md["condition"]
        md["condition"] = md[group_col].astype(str)
    elif numerator and denominator and "condition" in md.columns:
        cond_vals = set(md["condition"].astype(str))
        if not ({numerator, denominator} <= cond_vals) and "cell_type" in md.columns and ({numerator, denominator} <= set(md["cell_type"].astype(str))):
            md["original_condition"] = md["condition"]
            md["condition"] = md["cell_type"].astype(str)

    if "qc_pass" not in md.columns:
        md["qc_pass"] = True

    md = md[md["cell_id"].isin(obs_set)].reset_index(drop=True)
    if md.empty:
        raise ValueError("Zero overlapping cell_ids between --metadata and STAR splice junction barcodes.")
    return md


def main() -> int:
    ap = argparse.ArgumentParser(description="Build Katana-PRISM single-cell splicing matrices from STAR junctions + GTF.")
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--sj-dirs", nargs="+", required=True, help="STARsolo SJ/raw dirs (optional prefix::path) or *.SJ.out.tab dirs.")
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--donor-col", default=None, help="Explicit donor/animal column name (auto-detects donor_id/mouse_id/biological_sample_id if omitted).")
    ap.add_argument("--allow-single-donor", action="store_true", help="Allow single-donor fallback when metadata lacks a donor column.")
    ap.add_argument("--group-col", default="condition", help="Metadata column for contrast (e.g., condition or cell_type).")
    ap.add_argument("--numerator", default=None)
    ap.add_argument("--denominator", default=None)
    ap.add_argument("--min-cells", type=int, default=10, help="Minimum cells with >= min-cell-support junction reads.")
    ap.add_argument("--min-cell-support", type=float, default=3.0, help="Minimum competing junction reads in a cell to call PSI.")
    ap.add_argument("--pseudobulk-by", default=None, help="Optional comma-separated columns (e.g., donor_id,cell_type) to aggregate sparse droplet cells.")
    args = ap.parse_args()

    ev_df, junc_to_idx = build_events_from_gtf(args.gtf)
    if ev_df.empty:
        raise RuntimeError(f"No alternative splicing events constructed from GTF: {args.gtf}")

    j_mat, obs_cells = load_junction_counts(args.sj_dirs, junc_to_idx)
    md = standardize_metadata(
        args.metadata,
        obs_cells,
        args.group_col,
        args.numerator,
        args.denominator,
        donor_col=args.donor_col,
        allow_single_donor=args.allow_single_donor,
    )

    cell_to_col = {c: i for i, c in enumerate(obs_cells)}
    keep_cols = [cell_to_col[c] for c in md["cell_id"]]
    j_mat = j_mat[:, keep_cols].tocsr()

    # Optional donor x cell_type pseudobulk aggregation for sparse 10x droplet libraries
    if args.pseudobulk_by:
        pb_cols = [c.strip() for c in args.pseudobulk_by.split(",") if c.strip() in md.columns]
        if pb_cols:
            grp_keys = md[pb_cols].astype(str).agg("__".join, axis=1)
            uniq_k = grp_keys.unique().tolist()
            pb_blocks = []
            pb_meta = []
            for uk in uniq_k:
                mask = (grp_keys == uk).to_numpy()
                pb_blocks.append(np.asarray(j_mat[:, mask].sum(axis=1)))
                r0 = md.loc[mask].iloc[0].to_dict()
                r0["cell_id"] = uk
                r0["n_pooled_cells"] = int(mask.sum())
                pb_meta.append(r0)
            j_mat = csr_matrix(np.hstack(pb_blocks))
            md = pd.DataFrame(pb_meta)

    n_ev = len(ev_df)
    n_cells = j_mat.shape[1]
    j_dense = j_mat.toarray()

    inc_eff = np.zeros((n_ev, n_cells), dtype=np.float32)
    exc_eff = np.zeros((n_ev, n_cells), dtype=np.float32)
    tot_sup = np.zeros((n_ev, n_cells), dtype=np.float32)

    for e_i, (inc_js, exc_js) in enumerate(zip(ev_df["_inc_juncs"], ev_df["_exc_juncs"])):
        inc_idx = [junc_to_idx[j] for j in inc_js]
        exc_idx = [junc_to_idx[j] for j in exc_js]
        inc_sum = j_dense[inc_idx, :].sum(axis=0)
        exc_sum = j_dense[exc_idx, :].sum(axis=0)
        inc_eff[e_i, :] = inc_sum / float(len(inc_idx))
        exc_eff[e_i, :] = exc_sum / float(len(exc_idx))
        tot_sup[e_i, :] = inc_sum + exc_sum

    denom_eff = inc_eff + exc_eff
    valid_mask = (tot_sup >= float(args.min_cell_support)) & (denom_eff > 0)
    psi_mat = np.where(valid_mask, inc_eff / np.maximum(denom_eff, 1e-9), np.nan)

    # Filter to events observed in >= min_cells AND non-constant across cells
    n_obs_per_ev = valid_mask.sum(axis=1)
    min_c = min(int(args.min_cells), max(2, n_cells // 4))
    mean_psi = np.nanmean(np.where(valid_mask, psi_mat, np.nan), axis=1)
    keep_ev = (n_obs_per_ev >= min_c) & np.isfinite(mean_psi) & (mean_psi >= 0.01) & (mean_psi <= 0.99)
    if int(keep_ev.sum()) == 0:
        keep_ev = n_obs_per_ev >= max(1, min(3, n_cells // 2))

    ev_kept = ev_df.loc[keep_ev].drop(columns=["_inc_juncs", "_exc_juncs"]).reset_index(drop=True)
    psi_kept = psi_mat[keep_ev, :]
    sup_kept = np.where(valid_mask[keep_ev, :], tot_sup[keep_ev, :], 0.0)

    # Compute contrast dpsi & BH-FDR when numerator & denominator are present
    dpsi_arr = np.zeros(len(ev_kept), dtype=float)
    pval_arr = np.ones(len(ev_kept), dtype=float)
    if args.numerator and args.denominator and "condition" in md.columns:
        m_num = md["condition"].astype(str).eq(str(args.numerator)).to_numpy()
        m_den = md["condition"].astype(str).eq(str(args.denominator)).to_numpy()
        if m_num.any() and m_den.any():
            for i in range(len(ev_kept)):
                v1 = psi_kept[i, m_num]
                v2 = psi_kept[i, m_den]
                v1 = v1[np.isfinite(v1)]
                v2 = v2[np.isfinite(v2)]
                if len(v1) >= 2 and len(v2) >= 2:
                    dpsi_arr[i] = float(np.mean(v1) - np.mean(v2))
                    if np.std(np.concatenate([v1, v2])) > 0:
                        _, pval_arr[i] = mannwhitneyu(v1, v2, alternative="two-sided")

    # Simple Benjamini-Hochberg FDR without requiring statsmodels
    n_p = len(pval_arr)
    if n_p > 0:
        order = np.argsort(pval_arr)
        ranked = pval_arr[order] * n_p / np.arange(1, n_p + 1)
        q_ord = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
        fdr_arr = np.empty_like(q_ord)
        fdr_arr[order] = q_ord
    else:
        fdr_arr = np.ones(0, dtype=float)

    ev_kept["dpsi"] = np.round(dpsi_arr, 5)
    ev_kept["pvalue"] = pval_arr
    ev_kept["fdr"] = fdr_arr

    args.outdir.mkdir(parents=True, exist_ok=True)
    ev_kept.to_csv(args.outdir / "sc_events.tsv", sep="\t", index=False)
    ev_kept[["event_id", "gene_id", "gene_symbol"]].to_csv(args.outdir / "sc_event_genes.tsv", sep="\t", index=False)
    md.to_csv(args.outdir / "sc_cells.tsv", sep="\t", index=False)

    cell_ids = md["cell_id"].tolist()
    eids = ev_kept["event_id"].tolist()
    pd.DataFrame(psi_kept.T, index=cell_ids, columns=eids).to_csv(
        args.outdir / "sc_psi.tsv.gz", sep="\t", index_label="cell_id", compression="gzip"
    )
    pd.DataFrame(np.round(sup_kept.T, 1), index=cell_ids, columns=eids).to_csv(
        args.outdir / "sc_support.tsv.gz", sep="\t", index_label="cell_id", compression="gzip"
    )
    print(f"[katana_sc_build_splice_matrices] Emitted {len(ev_kept)} events across {len(cell_ids)} cells/units -> {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
