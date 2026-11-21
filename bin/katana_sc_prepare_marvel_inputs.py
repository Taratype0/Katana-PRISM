#!/usr/bin/env python3
"""Prepare standardized inputs (`SpliceJunction`, `IntronCounts`, host-normalized `GTF`,
`SplicePheno`, `GeneFeature`, and linear `Exp`) for official `MARVEL` (`v2.0.5`, Wen et al., NAR 2023)
from STAR/STARsolo `SJ/raw` matrices (or per-cell `*.SJ.out.tab`), optional BAM (`BC`-tagged or per-cell),
and `rMATS` `fromGTF.{SE,MXE,RI,A5SS,A3SS}.txt` event tables.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
from pathlib import Path
import re
import subprocess
import numpy as np
import pandas as pd
from scipy.io import mmread
from scipy.sparse import csr_matrix, hstack

_ATTR_RE = re.compile(r'([A-Za-z0-9_]+)\s+"([^"]+)"')
_CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")


def _open_text(path: Path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def normalize_gtf_for_marvel(gtf_in: Path, gtf_out: Path) -> pd.DataFrame:
    """Emit a clean 9-column nuclear host GTF guaranteeing `gene` rows with `gene_id` and `gene_type`
    (required by `MARVEL::Preprocess_rMATS`), and return `GeneFeature` (`gene_id`, `gene_short_name`, `gene_type`).
    """
    excluded_chroms = {"MT", "M", "chrM", "chrMT"}
    bad_tokens = ("ercc", "sirv", "spike", "tdtomato", "cre", "ai75", "egfp", "phi")

    raw_lines: list[tuple[str, str, str, str, str, str, str, str, str, dict[str, str]]] = []
    gene_span: dict[str, list] = {}
    has_gene_row: set[str] = set()

    with _open_text(gtf_in) as fh:
        for line in fh:
            if not line or line.startswith("#"):
                continue
            pts = line.rstrip("\n").split("\t")
            if len(pts) < 9:
                continue
            chrom = pts[0]
            if chrom in excluded_chroms or any(t in chrom.lower() for t in bad_tokens):
                continue
            attrs = dict(_ATTR_RE.findall(pts[8]))
            gid = attrs.get("gene_id", "")
            if not gid:
                continue
            gtype = attrs.get("gene_type") or attrs.get("gene_biotype") or "protein_coding"
            gsym = attrs.get("gene_name") or attrs.get("gene_symbol") or gid.split(".")[0]
            attrs["gene_type"] = gtype
            attrs["gene_name"] = gsym
            s1, e1 = int(pts[3]), int(pts[4])
            if pts[2] == "gene":
                has_gene_row.add(gid)
            if gid not in gene_span:
                gene_span[gid] = [chrom, pts[1], s1, e1, pts[6], gsym, gtype]
            else:
                gene_span[gid][2] = min(gene_span[gid][2], s1)
                gene_span[gid][3] = max(gene_span[gid][3], e1)
            raw_lines.append((pts[0], pts[1], pts[2], pts[3], pts[4], pts[5], pts[6], pts[7], pts[8], attrs))

    gtf_out.parent.mkdir(parents=True, exist_ok=True)
    with gtf_out.open("w", encoding="utf-8") as out:
        for gid, (chrom, src, s1, e1, strand, gsym, gtype) in gene_span.items():
            if gid not in has_gene_row:
                attr_s = f'gene_id "{gid}"; gene_name "{gsym}"; gene_type "{gtype}";'
                out.write(f"{chrom}\t{src}\tgene\t{s1}\t{e1}\t.\t{strand}\t.\t{attr_s}\n")
        for p0, p1, p2, p3, p4, p5, p6, p7, p8, attrs in raw_lines:
            if p2 == "gene" and "gene_type" not in p8:
                p8 = p8.rstrip().rstrip(";") + f'; gene_type "{attrs["gene_type"]}";'
            out.write(f"{p0}\t{p1}\t{p2}\t{p3}\t{p4}\t{p5}\t{p6}\t{p7}\t{p8}\n")

    gf_rows = [
        {"gene_id": gid, "gene_short_name": info[5], "gene_type": info[6]}
        for gid, info in sorted(gene_span.items())
    ]
    return pd.DataFrame(gf_rows)


def collect_candidate_junctions_and_ri_intervals(
    fromgtf_dir: Path,
) -> tuple[set[tuple[str, int, int]], dict[str, list[tuple[int, int]]]]:
    """Collect all 1-based splice junctions `(chr, start1, end1)` and retained-intron spans `(start1, end1)`
    referenced by `rMATS` `fromGTF.{SE,MXE,RI,A5SS,A3SS}.txt` so `MARVEL` receives a compact, exact SJ table.
    """
    exact_juncs: set[tuple[str, int, int]] = set()
    ri_by_chr: dict[str, list[tuple[int, int]]] = defaultdict(list)

    def _read_rmats(etype: str) -> pd.DataFrame:
        p = fromgtf_dir / f"fromGTF.{etype}.txt"
        return pd.read_csv(p, sep="\t") if p.is_file() else pd.DataFrame()

    se = _read_rmats("SE")
    for r in se.itertuples(index=False):
        ch = str(r.chr)
        exact_juncs.add((ch, int(r.upstreamEE) + 1, int(r.exonStart_0base)))
        exact_juncs.add((ch, int(r.exonEnd) + 1, int(r.downstreamES)))
        exact_juncs.add((ch, int(r.upstreamEE) + 1, int(r.downstreamES)))

    mxe = _read_rmats("MXE")
    if not mxe.empty:
        c1s = mxe["1stExonStart_0base"].to_numpy(dtype=int)
        c1e = mxe["1stExonEnd"].to_numpy(dtype=int)
        c2s = mxe["2ndExonStart_0base"].to_numpy(dtype=int)
        c2e = mxe["2ndExonEnd"].to_numpy(dtype=int)
        ues = mxe["upstreamEE"].to_numpy(dtype=int)
        des = mxe["downstreamES"].to_numpy(dtype=int)
        chs = mxe["chr"].astype(str).to_numpy()
        for ch, ue, s1, e1, s2, e2, ds in zip(chs, ues, c1s, c1e, c2s, c2e, des):
            exact_juncs.add((ch, int(ue) + 1, int(s1)))
            exact_juncs.add((ch, int(e1) + 1, int(ds)))
            exact_juncs.add((ch, int(ue) + 1, int(s2)))
            exact_juncs.add((ch, int(e2) + 1, int(ds)))

    for et in ("A5SS", "A3SS"):
        ss = _read_rmats(et)
        for r in ss.itertuples(index=False):
            ch, st = str(r.chr), str(r.strand)
            if (et == "A5SS" and st == "+") or (et == "A3SS" and st == "-"):
                exact_juncs.add((ch, int(r.longExonEnd) + 1, int(r.flankingES)))
                exact_juncs.add((ch, int(r.shortEE) + 1, int(r.flankingES)))
            else:
                exact_juncs.add((ch, int(r.flankingEE) + 1, int(r.longExonStart_0base)))
                exact_juncs.add((ch, int(r.flankingEE) + 1, int(r.shortES)))

    ri = _read_rmats("RI")
    seen_ri: set[tuple[str, int, int]] = set()
    for r in ri.itertuples(index=False):
        ch = str(r.chr)
        s1 = int(r.upstreamEE) + 1
        e1 = int(r.downstreamES)
        if e1 >= s1 and (ch, s1, e1) not in seen_ri:
            seen_ri.add((ch, s1, e1))
            ri_by_chr[ch].append((s1, e1))
            exact_juncs.add((ch, s1, e1))

    for ch in ri_by_chr:
        ri_by_chr[ch].sort()
    return exact_juncs, ri_by_chr


def build_marvel_sj_table(
    sj_dirs: list[str],
    keep_cells: list[str],
    exact_juncs: set[tuple[str, int, int]],
    ri_by_chr: dict[str, list[tuple[int, int]]],
    out_tsv_gz: Path,
) -> None:
    """Filter STARsolo `SJ/raw` (or per-cell `*.SJ.out.tab`) to candidate junctions for `MARVEL`."""
    target_chroms = {j[0] for j in exact_juncs} | set(ri_by_chr.keys())

    def _norm_chr(c: str) -> str:
        if c in target_chroms:
            return c
        if c.startswith("chr") and c[3:] in target_chroms:
            return c[3:]
        if ("chr" + c) in target_chroms:
            return "chr" + c
        return c

    def _is_relevant(ch: str, s1: int, e1: int) -> bool:
        if (ch, s1, e1) in exact_juncs:
            return True
        for rs, re_ in ri_by_chr.get(ch, ()):
            if s1 <= rs and e1 >= re_:
                return True
        return False

    blocks_raw: list[tuple[csr_matrix, list[int], list[int], int]] = []
    obs_cells: list[str] = []
    coord_to_idx: dict[str, int] = {}
    coord_list: list[str] = []

    for spec in sj_dirs:
        prefix, raw_p = (spec.split("::", 1)[0].strip() + "_", spec.split("::", 1)[1]) if "::" in spec else ("", spec)
        p = Path(raw_p)
        cand = [p, p / "Solo.out/SJ/raw", p / "SJ/raw"]
        solo_dir = next((d for d in cand if (d / "matrix.mtx").is_file() or (d / "matrix.mtx.gz").is_file()), None)
        if solo_dir is None:
            raise FileNotFoundError(f"Expected STARsolo SJ/raw directory with matrix.mtx in {p}")
        mtx_f = solo_dir / ("matrix.mtx.gz" if (solo_dir / "matrix.mtx.gz").is_file() else "matrix.mtx")
        feat_f = solo_dir / ("features.tsv.gz" if (solo_dir / "features.tsv.gz").is_file() else "features.tsv")
        bc_f = solo_dir / ("barcodes.tsv.gz" if (solo_dir / "barcodes.tsv.gz").is_file() else "barcodes.tsv")

        mat = mmread(str(mtx_f)).tocsr()
        with _open_text(bc_f) as fh:
            bcs = [prefix + ln.strip().split("\t")[0] for ln in fh if ln.strip()]

        src_rows: list[int] = []
        dst_rows: list[int] = []
        with _open_text(feat_f) as fh:
            for r_i, ln in enumerate(fh):
                pts = ln.rstrip("\n").split("\t")
                if len(pts) >= 3:
                    ch = _norm_chr(pts[0])
                    s1, e1 = int(pts[1]), int(pts[2])
                    if _is_relevant(ch, s1, e1):
                        key_c = f"{ch}:{s1}:{e1}"
                        dst = coord_to_idx.get(key_c)
                        if dst is None:
                            dst = len(coord_list)
                            coord_to_idx[key_c] = dst
                            coord_list.append(key_c)
                        src_rows.append(r_i)
                        dst_rows.append(dst)
        blocks_raw.append((mat, src_rows, dst_rows, len(bcs)))
        obs_cells.extend(bcs)

    n_total_coords = len(coord_list)
    blocks: list[csr_matrix] = []
    for mat, src_rows, dst_rows, n_bcs in blocks_raw:
        if src_rows and n_total_coords > 0:
            sub_coo = mat[src_rows, :].tocoo()
            mapped_r = np.asarray(dst_rows, dtype=np.int32)[sub_coo.row]
            blocks.append(csr_matrix((sub_coo.data.astype(np.int32), (mapped_r, sub_coo.col)), shape=(n_total_coords, n_bcs)))
        else:
            blocks.append(csr_matrix((n_total_coords, n_bcs), dtype=np.int32))

    full = hstack(blocks, format="csr") if len(blocks) > 1 else blocks[0]
    c2i = {c: i for i, c in enumerate(obs_cells)}
    col_idx = [c2i[c] for c in keep_cells if c in c2i]
    kept_cids = [c for c in keep_cells if c in c2i]
    if not kept_cids:
        raise RuntimeError("Zero overlapping barcodes between metadata and STARsolo SJ/raw.")

    full = full[:, col_idx].tocsr()
    nonzero_mask = np.asarray(full.sum(axis=1)).ravel() > 0
    full = full[nonzero_mask, :]
    coords_final = [coord_list[i] for i in np.flatnonzero(nonzero_mask)]

    dense = full.toarray().astype(np.int32)
    df_sj = pd.DataFrame(dense, columns=kept_cids)
    df_sj.insert(0, "coord.intron", coords_final)
    out_tsv_gz.parent.mkdir(parents=True, exist_ok=True)
    df_sj.to_csv(out_tsv_gz, sep="\t", index=False, compression="gzip")


def _ri_contig_worker(
    args_tuple: tuple[str, str, list[tuple[int, int]], dict[str, int], str],
) -> tuple[str, np.ndarray]:
    """Count per-base unspliced intronic coverage over `fromGTF.RI.txt` intervals `(start1, end1)` on one contig."""
    bam_str, contig, intervals_1b, bc_to_cidx, bc_tag = args_tuple
    n_int = len(intervals_1b)
    n_cells = max(bc_to_cidx.values()) + 1 if bc_to_cidx else 0
    cov_mat = np.zeros((n_int, n_cells), dtype=np.int64)
    if n_int == 0 or n_cells == 0:
        return contig, cov_mat

    # Group nearby RI intervals into merged fetch windows (gap <= 5000 bp) for fast indexed BAM access
    windows: list[tuple[int, int, list[int]]] = []
    cur_s0, cur_e0, cur_idxs = intervals_1b[0][0] - 1, intervals_1b[0][1], [0]
    for idx in range(1, n_int):
        s0, e0 = intervals_1b[idx][0] - 1, intervals_1b[idx][1]
        if s0 <= cur_e0 + 5000:
            cur_e0 = max(cur_e0, e0)
            cur_idxs.append(idx)
        else:
            windows.append((cur_s0, cur_e0, cur_idxs))
            cur_s0, cur_e0, cur_idxs = s0, e0, [idx]
    windows.append((cur_s0, cur_e0, cur_idxs))

    try:
        import pysam
        with pysam.AlignmentFile(bam_str, "rb", threads=1) as bam:
            if contig not in bam.references:
                alt = contig[3:] if contig.startswith("chr") else "chr" + contig
                if alt not in bam.references:
                    return contig, cov_mat
                fetch_ctg = alt
            else:
                fetch_ctg = contig

            for w_s0, w_e0, idxs in windows:
                sub_iv = [(i, intervals_1b[i][0] - 1, intervals_1b[i][1]) for i in idxs]
                for read in bam.fetch(fetch_ctg, w_s0, w_e0):
                    if read.is_unmapped or read.is_secondary or read.is_supplementary:
                        continue
                    if not read.has_tag(bc_tag):
                        continue
                    cidx = bc_to_cidx.get(str(read.get_tag(bc_tag)))
                    if cidx is None:
                        continue
                    blocks = read.get_blocks()
                    if not blocks:
                        continue
                    r_min, r_max = blocks[0][0], blocks[-1][1]
                    for r_i, iv_s0, iv_e0 in sub_iv:
                        if iv_e0 <= r_min:
                            continue
                        if iv_s0 >= r_max:
                            break
                        ov = 0
                        for b_s, b_e in blocks:
                            if b_e > iv_s0 and b_s < iv_e0:
                                ov += min(b_e, iv_e0) - max(b_s, iv_s0)
                        if ov > 0:
                            cov_mat[r_i, cidx] += ov
        return contig, cov_mat
    except ImportError:
        pass

    # Single-process indexed multi-region stream via `samtools view -F 2308 -M`
    import bisect
    all_ivs = [(i, intervals_1b[i][0] - 1, intervals_1b[i][1]) for i in range(n_int)]
    max_iv_len = max((e0 - s0 for _, s0, e0 in all_ivs), default=0)
    iv_starts = [s0 for _, s0, _ in all_ivs]
    regions = [f"{contig}:{w_s0 + 1}-{w_e0}" for w_s0, w_e0, _ in windows]

    proc = subprocess.Popen(
        ["samtools", "view", "-F", "2308", "-M", bam_str, *regions],
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1 << 20,
    )
    assert proc.stdout is not None
    tag_pref = f"{bc_tag}:Z:"
    for line in proc.stdout:
        pts = line.rstrip("\n").split("\t")
        if len(pts) < 12:
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
        blocks = []
        for ln_s, op in _CIGAR_RE.findall(pts[5]):
            ln = int(ln_s)
            if op in ("M", "=", "X"):
                blocks.append((pos0, pos0 + ln))
                pos0 += ln
            elif op in ("D", "N"):
                pos0 += ln
        if not blocks:
            continue
        r_min, r_max = blocks[0][0], blocks[-1][1]
        lo = bisect.bisect_left(iv_starts, max(0, r_min - max_iv_len))
        for k in range(lo, n_int):
            r_i, iv_s0, iv_e0 = all_ivs[k]
            if iv_s0 >= r_max:
                break
            if iv_e0 <= r_min:
                continue
            ov = 0
            for b_s, b_e in blocks:
                if b_e > iv_s0 and b_s < iv_e0:
                    ov += min(b_e, iv_e0) - max(b_s, iv_s0)
            if ov > 0:
                cov_mat[r_i, cidx] += ov
    proc.wait()
    return contig, cov_mat


def build_marvel_ri_intron_counts(
    bam_path: Path,
    ri_by_chr: dict[str, list[tuple[int, int]]],
    keep_cells: list[str],
    bc_to_cell: dict[str, str],
    bc_tag: str,
    threads: int,
    out_tsv_gz: Path,
) -> None:
    """Compute per-base intronic coverage matrix `IntronCounts` (`coord.intron` x `cell_id`) for `MARVEL::ComputePSI.RI`."""
    c2i = {c: i for i, c in enumerate(keep_cells)}
    bc_to_cidx = {bc: c2i[cell] for bc, cell in bc_to_cell.items() if cell in c2i}

    coords_all: list[str] = []
    mats_all: list[np.ndarray] = []
    tasks = [(str(bam_path), ch, ivs, bc_to_cidx, bc_tag) for ch, ivs in sorted(ri_by_chr.items()) if ivs]

    res_by_chr: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=max(1, threads)) as pool:
        futs = {pool.submit(_ri_contig_worker, t): t[1] for t in tasks}
        for fut in as_completed(futs):
            ctg, arr = fut.result()
            res_by_chr[ctg] = arr

    for ch, ivs in sorted(ri_by_chr.items()):
        if not ivs:
            continue
        coords_all.extend(f"{ch}:{s1}:{e1}" for s1, e1 in ivs)
        mats_all.append(res_by_chr[ch])

    full = np.vstack(mats_all) if mats_all else np.zeros((0, len(keep_cells)), dtype=np.int64)
    df_ri = pd.DataFrame(full, columns=keep_cells)
    df_ri.insert(0, "coord.intron", coords_all)
    out_tsv_gz.parent.mkdir(parents=True, exist_ok=True)
    df_ri.to_csv(out_tsv_gz, sep="\t", index=False, compression="gzip")


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare standardized MARVEL v2.0.5 inputs from STARsolo SJ/raw + optional BAM.")
    ap.add_argument("--gtf", type=Path, required=True)
    ap.add_argument("--fromgtf-dir", type=Path, required=True, help="Directory containing rMATS fromGTF.{SE,MXE,RI,A5SS,A3SS}.txt")
    ap.add_argument("--sj-dirs", nargs="+", required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--bam", type=Path, default=None, help="Optional BC-tagged BAM to compute MARVEL RI IntronCounts")
    ap.add_argument("--barcode-shares", type=Path, default=None)
    ap.add_argument("--bc-tag", default="BC")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    md = pd.read_csv(args.metadata, sep="," if args.metadata.suffix.lower() == ".csv" else "\t")
    cid_col = next((c for c in ("cell_id", "cell_barcode", "barcode", "sample.id", "sample_id") if c in md.columns), md.columns[0])
    keep_cells = md[cid_col].astype(str).str.strip().drop_duplicates().tolist()

    bc_to_cell = {c: c for c in keep_cells}
    if args.barcode_shares and args.barcode_shares.is_file():
        qc_set = set(keep_cells)
        with args.barcode_shares.open("r", encoding="utf-8-sig") as fh:
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

    print("[MARVEL Prep 1/3] Normalizing host GTF & GeneFeature for MARVEL ...", flush=True)
    gf_df = normalize_gtf_for_marvel(args.gtf, args.outdir / "marvel_host.gtf")
    gf_df.to_csv(args.outdir / "marvel_gene_feature.tsv", sep="\t", index=False)

    pheno = md.copy()
    pheno.insert(0, "sample.id", pheno[cid_col].astype(str).str.strip())
    pheno = pheno.drop_duplicates("sample.id").reset_index(drop=True)
    pheno.to_csv(args.outdir / "marvel_pheno.tsv", sep="\t", index=False)

    print("[MARVEL Prep 2/3] Extracting candidate splice junctions for MARVEL ...", flush=True)
    exact_juncs, ri_by_chr = collect_candidate_junctions_and_ri_intervals(args.fromgtf_dir)
    build_marvel_sj_table(args.sj_dirs, keep_cells, exact_juncs, ri_by_chr, args.outdir / "marvel_sj_counts.tsv.gz")

    if args.bam and args.bam.is_file() and ri_by_chr:
        print(f"[MARVEL Prep 3/3] Computing MARVEL RI per-base IntronCounts across {sum(len(v) for v in ri_by_chr.values())} RI intervals ...", flush=True)
        build_marvel_ri_intron_counts(
            args.bam,
            ri_by_chr,
            keep_cells,
            bc_to_cell,
            args.bc_tag,
            args.threads,
            args.outdir / "marvel_ri_intron_counts.tsv.gz",
        )
    else:
        print("[MARVEL Prep 3/3] No BAM supplied for RI IntronCounts; MARVEL will quantify SE, MXE, A5SS, A3SS.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
