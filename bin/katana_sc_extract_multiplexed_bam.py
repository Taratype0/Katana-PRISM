#!/usr/bin/env python3
"""Extract BC-tagged spliced junctions (and optional zUMIs UMI expression) directly from a
multiplexed single-cell/single-nucleus BAM (e.g., Smart-seq3 / Smart-seq3xpress / zUMIs)
using the user's QC-filtered and cell-typed metadata table.

Supports optional `barcode_shares.txt` pairing so both 5'-UMI reads (`XC_DNBPE_TS`) and
internal splice-junction reads (`XC_DNBPE`) map onto the canonical QC-passed cell barcode.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import coo_matrix

_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def load_qc_cells_and_aliases(
    annotations_path: Path,
    barcode_shares_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    sep = "," if annotations_path.suffix.lower() == ".csv" else "\t"
    md = pd.read_csv(annotations_path, sep=sep, dtype=str, keep_default_na=False)
    for qc_col in ("qc_pass", "qc_pass_nuclei", "keep"):
        if qc_col in md.columns:
            md = md[md[qc_col].astype(str).str.lower().isin(("true", "1", "yes", "pass"))].copy()
            break
    cid_col = next((c for c in ("cell_id", "cell_barcode", "barcode", "sample.id") if c in md.columns), md.columns[0])
    md["cell_id"] = md[cid_col].astype(str).str.strip()
    dup_cid = md["cell_id"].duplicated(keep=False)
    if dup_cid.any():
        bad_cids = md.loc[dup_cid, "cell_id"].drop_duplicates().head(5).tolist()
        raise ValueError(f"Duplicate physical cell_id entries in metadata (e.g. {bad_cids}).")
    md = md.reset_index(drop=True)
    qc_set = set(md["cell_id"])
    if not qc_set:
        raise RuntimeError(f"Zero QC-passed cells found in {annotations_path}")

    bc_to_cell = {bc: bc for bc in qc_set}
    if barcode_shares_path and barcode_shares_path.is_file():
        with barcode_shares_path.open("r", encoding="utf-8-sig") as fh:
            for line in fh:
                pts = line.strip().split("\t")
                if len(pts) >= 2:
                    b1, b2 = pts[0].strip(), pts[1].strip()
                    if b1 in qc_set:
                        bc_to_cell[b1] = b1
                        bc_to_cell[b2] = b1
                    elif b2 in qc_set:
                        bc_to_cell[b1] = b2
                        bc_to_cell[b2] = b2
    return md, bc_to_cell


def list_nuclear_host_contigs(bam_path: Path) -> list[str]:
    excluded = {"MT", "M", "chrM", "chrMT"}
    bad_toks = ("ercc", "sirv", "spike", "tdtomato", "cre", "ai75", "egfp", "phi")
    out = subprocess.check_output(["samtools", "idxstats", str(bam_path)], text=True)
    keep = []
    for ln in out.splitlines():
        if not ln or ln.startswith("*"):
            continue
        c = ln.split("\t", 1)[0]
        if c not in excluded and not any(t in c.lower() for t in bad_toks):
            keep.append(c)
    return keep


def _scan_contig(args_tuple: tuple[str, str, dict[str, int], str]) -> tuple[str, dict[tuple[int, int, int], int]]:
    bam_str, contig, bc_to_cidx, bc_tag = args_tuple
    junc_counts: dict[tuple[int, int, int], int] = defaultdict(int)
    try:
        import pysam
        with pysam.AlignmentFile(bam_str, "rb", threads=1) as bam:
            for read in bam.fetch(contig):
                if read.is_unmapped or read.is_secondary or read.is_supplementary:
                    continue
                cigar = read.cigartuples
                if not cigar or not any(op == 3 for op, _ in cigar):
                    continue
                if not read.has_tag(bc_tag):
                    continue
                cidx = bc_to_cidx.get(str(read.get_tag(bc_tag)))
                if cidx is None:
                    continue
                pos0 = read.reference_start
                for op, ln in cigar:
                    if op == 3:
                        if ln >= 20:
                            junc_counts[(pos0 + 1, pos0 + ln, cidx)] += 1
                        pos0 += ln
                    elif op in (0, 2, 7, 8):
                        pos0 += ln
        return contig, dict(junc_counts)
    except ImportError:
        pass

    proc = subprocess.Popen(
        ["samtools", "view", "-F", "2308", bam_str, contig],
        stdout=subprocess.PIPE, text=True, bufsize=1 << 20,
    )
    assert proc.stdout is not None
    tag_pref = f"{bc_tag}:Z:"
    for line in proc.stdout:
        pts = line.rstrip("\n").split("\t")
        if len(pts) < 12 or "N" not in pts[5]:
            continue
        bc_val = None
        for f in pts[11:]:
            if f.startswith(tag_pref):
                bc_val = f[len(tag_pref):]
                break
        if bc_val is None:
            continue
        cidx = bc_to_cidx.get(bc_val)
        if cidx is None:
            continue
        pos0 = int(pts[3]) - 1
        for ln_s, op in _CIGAR_RE.findall(pts[5]):
            ln = int(ln_s)
            if op == "N":
                if ln >= 20:
                    junc_counts[(pos0 + 1, pos0 + ln, cidx)] += 1
                pos0 += ln
            elif op in ("M", "D", "=", "X"):
                pos0 += ln
    proc.wait()
    return contig, dict(junc_counts)


def build_zumis_log1p_expression(
    umi_matrix: Path,
    gene_names_file: Path | None,
    cell_ids: list[str],
    out_gz: Path,
    max_genes: int = 8000,
) -> None:
    opener = gzip.open if str(umi_matrix).endswith(".gz") else open
    with opener(umi_matrix, "rt", encoding="utf-8", errors="replace") as fh:
        h_line = fh.readline().rstrip("\r\n")
        r1_line = fh.readline().rstrip("\r\n")
    sep = "\t" if "\t" in r1_line else r"\s+"
    h_toks = [t.strip('"') for t in (h_line.split("\t") if sep == "\t" else h_line.split())]
    r1_toks = r1_line.split("\t") if sep == "\t" else r1_line.split()
    if len(r1_toks) == len(h_toks) + 1:
        df = pd.read_csv(umi_matrix, sep=sep, header=0, index_col=0, engine="c" if sep == "\t" else "python")
        df.columns = h_toks
    else:
        df = pd.read_csv(umi_matrix, sep=sep, header=0, index_col=0, engine="c" if sep == "\t" else "python")
    df.columns = [str(c).strip('"') for c in df.columns]
    df.index = [str(i).strip('"') for i in df.index]

    gmap: dict[str, str] = {}
    if gene_names_file and gene_names_file.is_file():
        gdf = pd.read_csv(gene_names_file, sep=None, engine="python")
        if gdf.shape[1] >= 2:
            for g_id, g_nm in zip(gdf.iloc[:, 0].astype(str), gdf.iloc[:, 1].astype(str)):
                g1 = g_id.strip('"')
                g2 = g_nm.strip('"')
                if g2 and g2 != "NA":
                    gmap[g1] = g2
                    gmap[g1.split(".")[0]] = g2

    endog = [i for i, g in enumerate(df.index) if not any(k in g.lower() for k in ("ercc", "sirv", "spike", "tdtomato", "cre"))]
    df = df.iloc[endog].reindex(columns=cell_ids, fill_value=0.0)
    syms = [gmap.get(g) or gmap.get(g.split(".")[0]) or g.split(".")[0] for g in df.index]
    df.index = syms
    df = df.groupby(level=0, sort=False).sum()

    raw = df.to_numpy(dtype=np.float32)
    lib = np.maximum(raw.sum(axis=0, keepdims=True), 1.0)
    log_mat = np.log1p(raw / lib * 1e4)
    if log_mat.shape[0] > max_genes:
        keep_idx = np.argsort(-log_mat.var(axis=1))[:max_genes]
        log_mat = log_mat[keep_idx]
        kept_syms = [df.index[i] for i in keep_idx]
    else:
        kept_syms = list(df.index)

    out_df = pd.DataFrame(np.round(log_mat.T, 4), index=cell_ids, columns=kept_syms)
    out_gz.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_gz, sep="\t", index_label="cell_id", compression="gzip")


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract STARsolo SJ/raw and optional log1p expression from multiplexed BAM.")
    ap.add_argument("--bam", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--barcode-shares", type=Path, default=None)
    ap.add_argument("--umi-matrix", type=Path, default=None)
    ap.add_argument("--gene-names", type=Path, default=None)
    ap.add_argument("--bc-tag", default="BC")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    sj_dir = args.outdir / "Solo.out" / "SJ" / "raw"
    sj_dir.mkdir(parents=True, exist_ok=True)
    md, bc_to_cell = load_qc_cells_and_aliases(args.metadata, args.barcode_shares)
    cell_ids = md["cell_id"].tolist()
    c2i = {c: i for i, c in enumerate(cell_ids)}
    bc_to_cidx = {bc: c2i[c] for bc, c in bc_to_cell.items()}

    contigs = list_nuclear_host_contigs(args.bam)
    tasks = [(str(args.bam), c, bc_to_cidx, args.bc_tag) for c in contigs]
    res_map: dict[str, dict[tuple[int, int, int], int]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.threads)) as pool:
        futs = {pool.submit(_scan_contig, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            ctg, jd = fut.result()
            res_map[ctg] = jd

    feat_list, rows_i, cols_j, vals_v = [], [], [], []
    for ctg in contigs:
        by_j: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for (s1, e1, cidx), cnt in res_map[ctg].items():
            by_j[(s1, e1)].append((cidx, cnt))
        for s1, e1 in sorted(by_j):
            f_idx = len(feat_list)
            feat_list.append((ctg, s1, e1))
            for cidx, cnt in by_j[(s1, e1)]:
                rows_i.append(f_idx)
                cols_j.append(cidx)
                vals_v.append(cnt)

    (sj_dir / "barcodes.tsv").write_text("\n".join(cell_ids) + "\n", encoding="utf-8")
    (sj_dir / "features.tsv").write_text("\n".join(f"{c}\t{s}\t{e}" for c, s, e in feat_list) + "\n", encoding="utf-8")
    mat = coo_matrix(
        (np.asarray(vals_v, dtype=np.int32), (np.asarray(rows_i, dtype=np.int32), np.asarray(cols_j, dtype=np.int32))),
        shape=(len(feat_list), len(cell_ids)),
    )
    mmwrite(str(sj_dir / "matrix.mtx"), mat)

    if args.umi_matrix and args.umi_matrix.is_file():
        build_zumis_log1p_expression(args.umi_matrix, args.gene_names, cell_ids, args.outdir / "sc_expression.tsv.gz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
