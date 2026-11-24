"""Generic Cross-Paradigm Splicing & Isoform Resolution Engine (`cross_paradigm`).

Resolves four domain-agnostic evidence classes directly from standard Katana-PRISM /
nf-core upstream outputs without any gene-family or study-specific hardcoding:
  - Class I  : 1-bp Coordinate-Verified Exons (`rMATS × DEXSeq` + automated sign check)
  - Class II : Boundary Reprogramming (`SUPPA2 AF/AL` + strand-aware `5' -> 3'` DEXSeq gradient)
  - Class III: Full-Length Isoform Switching (`stageR DTU × SUPPA2 IOI × Salmon TPM × GTF × SpliceTrace`)
  - Class IV : Blind-Spot Exons & Net Discovery Uplift (`DEXSeq-only` non-rMATS regions across contrasts)

Gracefully auto-skips on single-cell cohorts or bundles lacking bulk multi-layer outputs.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import html as html_lib
import json
import math
from pathlib import Path
import platform
import re
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

ED_NAVY = "#051C2C"
ED_BLUE = "#2251FF"
ED_CERULEAN = "#0077C8"
ED_TEAL = "#0D9488"
ED_CORAL = "#B83A4B"
ED_AMBER = "#D97706"
ED_VIOLET = "#6D28D9"
ED_SLATE = "#64748B"


def _clean_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (float, np.floating)):
        fv = float(obj)
        return None if (math.isnan(fv) or math.isinf(fv)) else fv
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _safe_slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "study"


def _read_tabular_safe(path: Path, sep: str | None = None) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    if sep is None:
        sep = "," if path.suffix.lower() == ".csv" else "\t"
    try:
        return pd.read_csv(path, sep=sep, low_memory=False)
    except Exception:
        return pd.DataFrame()


def _save_dual(fig: plt.Figure, png_path: Path) -> Path:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path = png_path.with_suffix(".svg")
    fig.savefig(png_path, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return svg_path


def _attr_gtf(attrs: str, key: str) -> str:
    pat = key + ' "'
    i = attrs.find(pat)
    if i >= 0:
        i += len(pat)
        j = attrs.find('"', i)
        if j >= 0:
            return attrs[i:j]
    # Fallback unquoted GFF3/GTF key=value
    pat_eq = key + "="
    k = attrs.find(pat_eq)
    if k >= 0:
        k += len(pat_eq)
        j = attrs.find(";", k)
        return (attrs[k:] if j < 0 else attrs[k:j]).strip().strip('"')
    return ""


def _parse_gtf_structures(gtf_path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, list[tuple[int, int]]], dict[str, list[tuple[int, int]]], dict[str, tuple[str, str, int, int, str]]]:
    tx_meta: dict[str, dict[str, Any]] = {}
    tx_exons: dict[str, list[tuple[int, int]]] = defaultdict(list)
    tx_cds: dict[str, list[tuple[int, int]]] = defaultdict(list)
    gene_span: dict[str, tuple[str, str, int, int, str]] = {}
    if gtf_path is None or not Path(gtf_path).is_file():
        return tx_meta, tx_exons, tx_cds, gene_span

    gp = Path(gtf_path)
    opener = gzip.open if gp.suffix.lower() == ".gz" else open
    gene_bounds: dict[str, list[Any]] = {}
    with opener(gp, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[call-overload]
        for line in fh:
            if not line or line[0] == "#":
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9:
                continue
            ftype = p[2]
            if ftype not in ("gene", "transcript", "exon", "CDS"):
                continue
            chrom, start, end, strand, attrs = p[0], int(p[3]), int(p[4]), p[6], p[8]
            if ftype == "gene":
                gb = _attr_gtf(attrs, "gene_id").split(".")[0]
                if not gb:
                    continue
                gs = _attr_gtf(attrs, "gene_name") or _attr_gtf(attrs, "gene_symbol") or gb
                gene_span[gb] = (chrom, strand, start, end, gs)
            elif ftype == "transcript":
                tb = _attr_gtf(attrs, "transcript_id").split(".")[0]
                gb = _attr_gtf(attrs, "gene_id").split(".")[0]
                gs = _attr_gtf(attrs, "gene_name") or _attr_gtf(attrs, "gene_symbol") or gb
                bt = _attr_gtf(attrs, "transcript_type") or _attr_gtf(attrs, "transcript_biotype") or "protein_coding"
                if tb:
                    tx_meta[tb] = {"tx_base": tb, "gene_base": gb, "gene_symbol": gs, "biotype": bt, "chr": chrom, "strand": strand, "start": start, "end": end}
                if gb:
                    cur = gene_bounds.get(gb)
                    if cur is None:
                        gene_bounds[gb] = [chrom, strand, start, end, gs]
                    else:
                        cur[2] = min(cur[2], start)
                        cur[3] = max(cur[3], end)
            elif ftype == "exon":
                tb = _attr_gtf(attrs, "transcript_id").split(".")[0]
                if tb:
                    tx_exons[tb].append((start, end))
                    gb = _attr_gtf(attrs, "gene_id").split(".")[0]
                    gs = _attr_gtf(attrs, "gene_name") or gb
                    if tb not in tx_meta and gb:
                        tx_meta[tb] = {"tx_base": tb, "gene_base": gb, "gene_symbol": gs, "biotype": "protein_coding", "chr": chrom, "strand": strand, "start": start, "end": end}
                    if gb:
                        cur = gene_bounds.get(gb)
                        if cur is None:
                            gene_bounds[gb] = [chrom, strand, start, end, gs]
                        else:
                            cur[2] = min(cur[2], start)
                            cur[3] = max(cur[3], end)
            elif ftype == "CDS":
                tb = _attr_gtf(attrs, "transcript_id").split(".")[0]
                if tb:
                    tx_cds[tb].append((start, end))

    for gb, vals in gene_bounds.items():
        gene_span.setdefault(gb, (vals[0], vals[1], int(vals[2]), int(vals[3]), vals[4]))
    for tb in tx_exons:
        tx_exons[tb].sort()
    for tb in tx_cds:
        tx_cds[tb].sort()
    return tx_meta, tx_exons, tx_cds, gene_span


def _parse_dexseq_gff(gff_path: Path | None) -> pd.DataFrame:
    if gff_path is None or not Path(gff_path).is_file():
        return pd.DataFrame(columns=["chr", "start", "end", "strand", "gene_base", "feature_id", "transcripts"])
    rows: list[tuple[str, int, int, str, str, str, str]] = []
    with Path(gff_path).open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] != "exonic_part":
                continue
            gb = _attr_gtf(p[8], "gene_id").split("+", 1)[0].split(".")[0]
            ep_num = _attr_gtf(p[8], "exonic_part_number") or _attr_gtf(p[8], "feature_id")
            fid = ep_num if ep_num.startswith("E") else f"E{ep_num}"
            txs = _attr_gtf(p[8], "transcripts")
            rows.append((p[0], int(p[3]), int(p[4]), p[6], gb, fid, txs))
    return pd.DataFrame(rows, columns=["chr", "start", "end", "strand", "gene_base", "feature_id", "transcripts"])


def _orient_dexseq_df(raw_x: pd.DataFrame, lfc_col: str, treat_lbl: str, ctrl_lbl: str) -> tuple[pd.Series, bool]:
    raw_lfc = pd.to_numeric(raw_x[lfc_col], errors="coerce")
    if treat_lbl and ctrl_lbl and treat_lbl != ctrl_lbl:
        t_col = treat_lbl if treat_lbl in raw_x.columns else None
        c_col = ctrl_lbl if ctrl_lbl in raw_x.columns else None
        if t_col and c_col:
            dt = pd.to_numeric(raw_x[t_col], errors="coerce") - pd.to_numeric(raw_x[c_col], errors="coerce")
            q_col = next((c for c in ("padj", "FDR", "fdr", "qvalue") if c in raw_x.columns), None)
            mask = raw_lfc.notna() & dt.notna() & raw_lfc.abs().ge(0.25) & (dt != 0)
            if q_col:
                sig_m = mask & pd.to_numeric(raw_x[q_col], errors="coerce").lt(0.05)
                if int(sig_m.sum()) >= 5:
                    mask = sig_m
            if int(mask.sum()) >= 3:
                corr_sign = float(np.sign(dt[mask] * raw_lfc[mask]).mean())
                if corr_sign < 0.0:
                    return -raw_lfc, True
                if corr_sign > 0.0:
                    return raw_lfc, False
        if lfc_col.lower().startswith(f"log2fold_{ctrl_lbl.lower()}_{treat_lbl.lower()}"):
            return -raw_lfc, True
    return raw_lfc, False


def _cds_aa_len(tx_cds: dict[str, list[tuple[int, int]]], tb: str) -> int:
    segs = tx_cds.get(tb, [])
    if not segs:
        return 0
    return int(round(sum(e - s + 1 for s, e in segs) / 3.0))


def _utr_lens(tx_meta: dict[str, dict[str, Any]], tx_exons: dict[str, list[tuple[int, int]]], tx_cds: dict[str, list[tuple[int, int]]], tb: str) -> tuple[int, int]:
    segs = tx_exons.get(tb, [])
    cds = tx_cds.get(tb, [])
    if not segs or not cds:
        return (0, 0)
    cmin, cmax = cds[0][0], cds[-1][1]
    left = sum(max(0, min(e, cmin - 1) - s + 1) for s, e in segs if s < cmin)
    right = sum(max(0, e - max(s, cmax + 1) + 1) for s, e in segs if e > cmax)
    strand = tx_meta.get(tb, {}).get("strand", "+")
    return (left, right) if strand == "+" else (right, left)


def _classify_isoform_fate(b_dn: str, b_up: str, aa_dn: int, aa_up: int, u5_dn: int, u5_up: int, u3_dn: int, u3_up: int) -> str:
    is_cod_dn = b_dn == "protein_coding"
    is_cod_up = b_up == "protein_coding"
    if is_cod_dn and b_up == "nonsense_mediated_decay":
        return "Coding -> NMD (Loss of ORF)"
    if b_dn == "nonsense_mediated_decay" and is_cod_up:
        return "NMD -> Coding (Rescue)"
    if is_cod_dn and b_up == "retained_intron":
        return "Coding -> Retained Intron"
    if b_dn == "retained_intron" and is_cod_up:
        return "Retained Intron -> Coding"
    if is_cod_dn and not is_cod_up:
        return f"Coding -> Non-coding ({b_up})"
    if not is_cod_dn and is_cod_up:
        return f"Non-coding ({b_dn}) -> Coding"
    if is_cod_dn and is_cod_up:
        if abs(aa_up - aa_dn) >= 15:
            return "Coding -> Coding (ORF Remodeling >=15 aa)"
        if abs(u3_up - u3_dn) >= 100 or abs(u5_up - u5_dn) >= 50:
            return "Coding -> Coding (UTR Switch)"
        return "Coding -> Coding (Minor/Synonymous)"
    return "Non-coding -> Non-coding / Other"


def _discover_contrasts_and_samples(bundle_dir: Path | None, samples_arg: Path | None = None, contrasts_arg: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    s_f = samples_arg
    if s_f is None and bundle_dir is not None:
        s_f = next((p for p in (
            bundle_dir / "pipeline_info/samplesheet.valid.csv",
            bundle_dir / "inputs/samplesheet.csv",
            bundle_dir / "samplesheet.csv",
            bundle_dir / "samples.csv",
            bundle_dir / "splicemanifold/splicemanifold_out/samples_metadata.tsv",
        ) if p.is_file()), None)
    samples_list: list[dict[str, Any]] = []
    cond_to_sids: dict[str, list[str]] = {}
    if s_f is not None and s_f.is_file():
        s_df = _read_tabular_safe(s_f)
        if not s_df.empty:
            sc = next((c for c in ("sample_id", "sample", "id") if c in s_df.columns), s_df.columns[0])
            cc = next((c for c in ("condition", "group") if c in s_df.columns), s_df.columns[1])
            for _, r in s_df.iterrows():
                sid = str(r[sc]).strip()
                cond = str(r[cc]).strip()
                samples_list.append({"sample_id": sid, "condition": cond})
                cond_to_sids.setdefault(cond, []).append(sid)

    c_f = contrasts_arg
    if c_f is None and bundle_dir is not None:
        c_f = next((p for p in (
            bundle_dir / "pipeline_info/contrastsheet.valid.csv",
            bundle_dir / "inputs/contrastsheet.csv",
            bundle_dir / "contrasts.csv",
            bundle_dir / "contrastsheet.csv",
        ) if p.is_file()), None)
    contrasts_list: list[dict[str, Any]] = []
    if c_f is not None and c_f.is_file():
        c_df = _read_tabular_safe(c_f)
        if not c_df.empty:
            cid_c = next((c for c in ("contrast_id", "contrast", "comparison_id") if c in c_df.columns), c_df.columns[0])
            num_c = next((c for c in ("treatment", "numerator", "group_a") if c in c_df.columns), c_df.columns[1])
            den_c = next((c for c in ("control", "denominator", "group_b") if c in c_df.columns), c_df.columns[2])
            for _, r in c_df.iterrows():
                cid = str(r[cid_c]).strip()
                tr = str(r[num_c]).strip()
                ct = str(r[den_c]).strip()
                contrasts_list.append({
                    "contrast_id": cid,
                    "label": f"{tr} vs {ct}",
                    "numerator": {"label": tr, "sample_ids": cond_to_sids.get(tr, [])},
                    "denominator": {"label": ct, "sample_ids": cond_to_sids.get(ct, [])},
                })
    elif len(cond_to_sids) >= 2:
        ks = list(cond_to_sids.keys())
        contrasts_list.append({
            "contrast_id": f"{ks[0]}_vs_{ks[1]}",
            "label": f"{ks[0]} vs {ks[1]}",
            "numerator": {"label": ks[0], "sample_ids": cond_to_sids[ks[0]]},
            "denominator": {"label": ks[1], "sample_ids": cond_to_sids[ks[1]]},
        })
    return samples_list, contrasts_list


def _load_rmats_contrast_df(bundle_dir: Path | None, cid: str, treat_lbl: str, ctrl_lbl: str, rmats_mode: str = "JCEC") -> pd.DataFrame:
    if bundle_dir is None:
        return pd.DataFrame()
    pair = f"{treat_lbl}-{ctrl_lbl}" if (treat_lbl and ctrl_lbl) else cid
    r_dirs = [
        bundle_dir / "rmats" / cid / "rmats_post",
        bundle_dir / "rmats" / cid,
        bundle_dir / "rnasplicing/rmats" / pair / "rmats_post",
        bundle_dir / "rnasplicing/rmats" / cid / "rmats_post",
    ]
    rdir = next((d for d in r_dirs if d.is_dir()), None)
    if rdir is None:
        return pd.DataFrame()
    frames = []
    for et in ("SE", "MXE", "RI", "A3SS", "A5SS"):
        fp = rdir / f"{et}.MATS.{rmats_mode.upper()}.txt"
        if not fp.is_file():
            alt = "JC" if rmats_mode.upper() == "JCEC" else "JCEC"
            fp = rdir / f"{et}.MATS.{alt}.txt"
        if not fp.is_file():
            continue
        df = _read_tabular_safe(fp, sep="\t")
        if df.empty:
            continue
        df["event_type"] = et
        df["gene_id"] = df["GeneID"].astype(str).str.strip('"')
        df["gene_symbol"] = df["geneSymbol"].astype(str).str.strip('"')
        df["fdr"] = pd.to_numeric(df["FDR"], errors="coerce")
        df["pvalue"] = pd.to_numeric(df["PValue"], errors="coerce")
        df["dpsi"] = pd.to_numeric(df["IncLevelDifference"], errors="coerce")
        df["abs_dpsi"] = df["dpsi"].abs()
        df["stable_significant"] = (df["fdr"] < 0.05) & (df["abs_dpsi"] >= 0.10)
        if et == "SE":
            df["event_id"] = df.apply(lambda r: f"SE:{r['chr']}:{r['strand']}:{int(r['exonStart_0base'])}:{int(r['exonEnd'])}:{int(r['upstreamES'])}:{int(r['upstreamEE'])}:{int(r['downstreamES'])}:{int(r['downstreamEE'])}", axis=1)
        elif et in ("A3SS", "A5SS"):
            df["event_id"] = df.apply(lambda r: f"{et}:{r['chr']}:{r['strand']}:{int(r['longExonStart_0base'])}:{int(r['longExonEnd'])}:{int(r['shortES'])}:{int(r['shortEE'])}:{int(r['flankingES'])}:{int(r['flankingEE'])}", axis=1)
        elif et == "RI":
            df["event_id"] = df.apply(lambda r: f"RI:{r['chr']}:{r['strand']}:{int(r['riExonStart_0base'])}:{int(r['riExonEnd'])}:{int(r['upstreamES'])}:{int(r['upstreamEE'])}:{int(r['downstreamES'])}:{int(r['downstreamEE'])}", axis=1)
        elif et == "MXE":
            df["event_id"] = df.apply(lambda r: f"MXE:{r['chr']}:{r['strand']}:{int(r['1stExonStart_0base'])}:{int(r['1stExonEnd'])}:{int(r['2ndExonStart_0base'])}:{int(r['2ndExonEnd'])}:{int(r['upstreamES'])}:{int(r['upstreamEE'])}:{int(r['downstreamES'])}:{int(r['downstreamEE'])}", axis=1)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _extract_rmats_target_intervals(df: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, int, int]]]]:
    targets: list[dict[str, Any]] = []
    all_by_gene: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
    if df is None or df.empty:
        return targets, all_by_gene
    work = df.copy()
    if "gene_base" not in work.columns:
        gc = "gene_id" if "gene_id" in work.columns else "GeneID"
        work["gene_base"] = work[gc].astype(str).str.strip('"').str.split(".").str[0]
    if "fdr" not in work.columns and "FDR" in work.columns:
        work["fdr"] = pd.to_numeric(work["FDR"], errors="coerce")
    if "dpsi" not in work.columns and "IncLevelDifference" in work.columns:
        work["dpsi"] = pd.to_numeric(work["IncLevelDifference"], errors="coerce")
    if "abs_dpsi" not in work.columns:
        work["abs_dpsi"] = pd.to_numeric(work["dpsi"], errors="coerce").abs()
    if "stable_significant" not in work.columns:
        work["stable_significant"] = (work["fdr"] < 0.05) & (work["abs_dpsi"] >= 0.10)

    coord_cols = [
        "gene_base", "gene_symbol", "event_id", "event_type", "chr", "strand", "dpsi", "abs_dpsi", "fdr", "stable_significant",
        "exonStart_0base", "exonEnd", "longExonStart_0base", "longExonEnd", "shortES", "shortEE",
        "upstreamEE", "downstreamES", "riExonStart_0base", "riExonEnd",
        "1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd",
    ]
    present = [c for c in coord_cols if c in work.columns]
    for rec in work[present].to_dict("records"):
        gb = str(rec.get("gene_base", ""))
        et = str(rec.get("event_type", "SE"))
        ch = str(rec.get("chr", ""))
        ivs_all: list[tuple[int, int]] = []
        ivs_diff: list[tuple[int, int]] = []
        if et == "SE" and pd.notna(rec.get("exonStart_0base")) and pd.notna(rec.get("exonEnd")):
            iv = (int(rec["exonStart_0base"]) + 1, int(rec["exonEnd"]))
            ivs_all.append(iv); ivs_diff.append(iv)
        elif et in ("A3SS", "A5SS") and pd.notna(rec.get("longExonStart_0base")) and pd.notna(rec.get("longExonEnd")):
            ls, le = int(rec["longExonStart_0base"]) + 1, int(rec["longExonEnd"])
            ivs_all.append((ls, le))
            if pd.notna(rec.get("shortES")) and pd.notna(rec.get("shortEE")):
                ss, se = int(rec["shortES"]) + 1, int(rec["shortEE"])
                if ls < ss:
                    ivs_diff.append((ls, ss - 1))
                if se < le:
                    ivs_diff.append((se + 1, le))
            if not ivs_diff:
                ivs_diff.append((ls, le))
        elif et == "RI" and pd.notna(rec.get("upstreamEE")) and pd.notna(rec.get("downstreamES")):
            us, ue = int(rec["upstreamEE"]) + 1, int(rec["downstreamES"])
            if us <= ue:
                ivs_diff.append((us, ue))
            elif pd.notna(rec.get("riExonStart_0base")) and pd.notna(rec.get("riExonEnd")):
                ivs_diff.append((int(rec["riExonStart_0base"]) + 1, int(rec["riExonEnd"])))
            if pd.notna(rec.get("riExonStart_0base")) and pd.notna(rec.get("riExonEnd")):
                ivs_all.append((int(rec["riExonStart_0base"]) + 1, int(rec["riExonEnd"])))
        elif et == "MXE":
            for s_k, e_k in (("1stExonStart_0base", "1stExonEnd"), ("2ndExonStart_0base", "2ndExonEnd")):
                if pd.notna(rec.get(s_k)) and pd.notna(rec.get(e_k)):
                    iv = (int(rec[s_k]) + 1, int(rec[e_k]))
                    ivs_all.append(iv); ivs_diff.append(iv)
        for s, e in ivs_all:
            all_by_gene[gb].append((ch, s, e))
        if (rec.get("fdr", 1.0) < 0.05) and (rec.get("abs_dpsi", 0.0) >= 0.05) and ivs_diff:
            rec["ivs"] = ivs_diff
            targets.append(rec)
    return targets, all_by_gene


def run_cross_paradigm_analysis(
    *,
    outdir: str | Path,
    bundle_dir: str | Path | None = None,
    gtf_path: str | Path | None = None,
    contrasts_list: list[dict[str, Any]] | None = None,
    samples_list: list[dict[str, Any]] | None = None,
    das_by_contrast: dict[str, pd.DataFrame] | None = None,
    deg_by_contrast: dict[str, pd.DataFrame] | None = None,
    splicetrace_dirs: list[str | Path] | dict[str, Path] | None = None,
    rmats_mode: str = "JCEC",
    fdr_cutoff: float = 0.05,
) -> dict[str, Any]:
    """Compute generic 4-class cross-paradigm concordance, AF/AL boundaries, 5'->3' exon gradient,
    DTU isoform switching fates, and multi-contrast uplift from completed bulk upstream outputs."""
    out = Path(outdir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    bdir = Path(bundle_dir).resolve() if bundle_dir is not None else None

    if contrasts_list is None or samples_list is None:
        disc_s, disc_c = _discover_contrasts_and_samples(bdir)
        samples_list = samples_list or disc_s
        contrasts_list = contrasts_list or disc_c
    if not contrasts_list:
        summary = {"available": False, "reason": "no_contrasts_declared"}
        (out / "cross_paradigm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # Locate optional GTF, DEXSeq.gff, and Salmon transcript TPM
    gtf_cand = Path(gtf_path) if (gtf_path and Path(gtf_path).is_file()) else None
    if gtf_cand is None and bdir is not None:
        gtf_cand = next((p for p in (bdir / "refs/annotation.gtf", bdir / "refs/genes.gtf", bdir / "refs/annotation.gtf.gz") if p.is_file()), None)

    dex_gff_f = next((p for p in (
        (bdir / "dexseq_exon/annotation/DEXSeq.gff") if bdir else None,
        (bdir / "differential/dexseq_exon/annotation/DEXSeq.gff") if bdir else None,
    ) if p is not None and p.is_file()), None)

    tpm_f = next((p for p in (
        (bdir / "quant/matrices/salmon.merged.transcript_tpm.tsv") if bdir else None,
        (bdir / "rnaseq/matrices/salmon.merged.transcript_tpm.tsv") if bdir else None,
        (bdir / "quant/matrices/salmon.merged.transcript_tpm_dtu_scaled.tsv") if bdir else None,
        (bdir / "rnaseq/matrices/salmon.merged.transcript_tpm_dtu_scaled.tsv") if bdir else None,
    ) if p is not None and p.is_file()), None)

    # Check if ANY bulk cross-paradigm upstream layers exist
    has_any_layer = False
    for c_sp in contrasts_list:
        cid = str(c_sp["contrast_id"])
        tr = str(c_sp.get("numerator", {}).get("label", "")).strip()
        ct = str(c_sp.get("denominator", {}).get("label", "")).strip()
        if bdir is not None:
            for p in (
                bdir / "dexseq_exon" / f"DEXSeqResults.{cid}.csv",
                bdir / "differential/dexseq_exon" / f"DEXSeqResults.{cid}.csv",
                bdir / "suppa" / cid / "diffsplice/per_local_event/local_events.dpsi",
                bdir / "rnasplicing/suppa/diffsplice_ioe" / f"{ct}-{tr}_local_diffsplice.dpsi",
                bdir / "dtu/results/stager" / f"getAdjustedPValues.{cid}.tsv",
                bdir / "dtu/results/stager" / f"getAdjustedPValues.{tr}-{ct}.tsv",
                bdir / "differential/dtu/stager" / f"getAdjustedPValues.{cid}.tsv",
                bdir / "differential/dtu/stager" / f"getAdjustedPValues.{tr}-{ct}.tsv",
            ):
                if p.is_file():
                    has_any_layer = True
                    break
        if has_any_layer:
            break
    if not has_any_layer:
        summary = {"available": False, "reason": "no_bulk_multiparadigm_upstream_files"}
        (out / "cross_paradigm_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # Parse shared GTF & DEXSeq.gff + Salmon TPM once
    tx_meta, tx_exons, tx_cds, gene_span = _parse_gtf_structures(gtf_cand)
    gff_df = _parse_dexseq_gff(dex_gff_f)

    gff_pos_frac: dict[tuple[str, str], float] = {}
    if not gff_df.empty:
        ep_rank = gff_df.sort_values(["gene_base", "start"]).assign(idx=lambda d: d.groupby("gene_base").cumcount() + 1)
        n_ep = gff_df.groupby("gene_base")["feature_id"].count().to_dict()
        for rec in ep_rank[["gene_base", "feature_id", "idx", "start", "end"]].to_dict("records"):
            gb = rec["gene_base"]
            gs_info = gene_span.get(gb)
            if gs_info and gs_info[3] > gs_info[2]:
                mid = 0.5 * (rec["start"] + rec["end"])
                frac = (mid - gs_info[2]) / float(gs_info[3] - gs_info[2])
                if gs_info[1] == "-":
                    frac = 1.0 - frac
            else:
                n = n_ep.get(gb, 1)
                frac = (rec["idx"] - 1) / max(1, n - 1) if n > 1 else 0.5
            gff_pos_frac[(gb, rec["feature_id"])] = max(0.0, min(1.0, float(frac)))

    sym_map: dict[str, str] = {gb: info[4] for gb, info in gene_span.items()}
    if bdir and (bdir / "refs/annotation.tsv").is_file():
        ann = _read_tabular_safe(bdir / "refs/annotation.tsv", sep="\t")
        if not ann.empty:
            gc = next((c for c in ("gene_id", "ensembl_gene_id") if c in ann.columns), None)
            sc = next((c for c in ("gene_name", "gene_symbol", "symbol") if c in ann.columns), None)
            if gc and sc:
                for gid, gs in zip(ann[gc].astype(str), ann[sc].astype(str)):
                    if gs and gs != "nan":
                        sym_map[gid.split(".")[0]] = gs

    tpm_df = _read_tabular_safe(tpm_f, sep="\t") if tpm_f else pd.DataFrame()
    if not tpm_df.empty:
        tx_col_t = next((c for c in ("tx", "transcript_id", "TXNAME", tpm_df.columns[0]) if c in tpm_df.columns), tpm_df.columns[0])
        gid_col_t = next((c for c in ("gene_id", "GENEID", tpm_df.columns[1]) if c in tpm_df.columns), tpm_df.columns[1])
        tpm_df["tx_base"] = tpm_df[tx_col_t].astype(str).str.split(".").str[0]
        tpm_df["gene_base"] = tpm_df[gid_col_t].astype(str).str.split(".").str[0]

    # Map SpliceTrace dirs
    st_dir_map: dict[str, Path] = {}
    if isinstance(splicetrace_dirs, dict):
        st_dir_map = {str(k): Path(v) for k, v in splicetrace_dirs.items() if Path(v).is_dir()}
    elif splicetrace_dirs:
        for item in splicetrace_dirs:
            raw_s = str(item).strip()
            if "=" in raw_s and not Path(raw_s).exists():
                k, v = raw_s.split("=", 1)
                if Path(v.strip()).is_dir():
                    st_dir_map[k.strip()] = Path(v.strip())
            elif Path(raw_s).is_dir():
                p_st = Path(raw_s)
                for c_sp in contrasts_list:
                    cid = str(c_sp["contrast_id"])
                    if p_st.name in (cid, f"splicetrace_{cid}") or p_st.name.endswith(f"_{cid}"):
                        st_dir_map[cid] = p_st

    contrast_results: dict[str, Any] = {}
    bench_rows: list[dict[str, Any]] = []

    for c_sp in contrasts_list:
        cid = str(c_sp["contrast_id"])
        slug = _safe_slug(cid)
        treat_lbl = str(c_sp.get("numerator", {}).get("label", "")).strip()
        ctrl_lbl = str(c_sp.get("denominator", {}).get("label", "")).strip()
        treat_sids = list(c_sp.get("numerator", {}).get("sample_ids", []))
        ctrl_sids = list(c_sp.get("denominator", {}).get("sample_ids", []))

        # Load DEG
        deg_df = deg_by_contrast[cid].copy() if (deg_by_contrast and cid in deg_by_contrast) else pd.DataFrame()
        if deg_df.empty and bdir is not None:
            deg_f = next((p for p in (
                bdir / "DEG" / f"{cid}.deseq2.results.tsv",
                bdir / "differential/deg" / f"{cid}.deseq2.results.tsv",
                bdir / "DEG" / f"{cid}.tsv",
            ) if p.is_file()), None)
            if deg_f is not None:
                raw_g = _read_tabular_safe(deg_f, sep="\t")
                if not raw_g.empty:
                    gc = next((c for c in ("gene_id", "GeneID", "id") if c in raw_g.columns), raw_g.columns[0])
                    fc = next((c for c in ("log2FoldChange", "log2fc", "logFC") if c in raw_g.columns), None)
                    qc = next((c for c in ("padj", "fdr", "FDR", "qvalue") if c in raw_g.columns), None)
                    sc = next((c for c in ("gene_symbol", "gene_name") if c in raw_g.columns), None)
                    if fc and qc:
                        deg_df = pd.DataFrame({
                            "gene_id": raw_g[gc].astype(str),
                            "gene_symbol": raw_g[sc].astype(str) if sc else raw_g[gc].astype(str).map(lambda g: sym_map.get(g.split(".")[0], g)),
                            "log2fc": pd.to_numeric(raw_g[fc], errors="coerce"),
                            "fdr": pd.to_numeric(raw_g[qc], errors="coerce"),
                        })
        deg_sig_genes: set[str] = set()
        deg_lfc_map: dict[str, float] = {}
        deg_fdr_map: dict[str, float] = {}
        if not deg_df.empty:
            deg_df["gene_base"] = deg_df["gene_id"].astype(str).str.split(".").str[0]
            if "gene_symbol" in deg_df.columns:
                for gb, gs in zip(deg_df["gene_base"], deg_df["gene_symbol"].astype(str)):
                    if gs and gs != "nan" and not gs.startswith("ENS"):
                        sym_map.setdefault(gb, gs)
            fc_col = "log2fc" if "log2fc" in deg_df.columns else "log2FoldChange"
            fd_col = "fdr" if "fdr" in deg_df.columns else "padj"
            deg_sig_genes = set(deg_df.loc[pd.to_numeric(deg_df[fd_col], errors="coerce") < fdr_cutoff, "gene_base"])
            deg_lfc_map = dict(zip(deg_df["gene_base"], pd.to_numeric(deg_df[fc_col], errors="coerce")))
            deg_fdr_map = dict(zip(deg_df["gene_base"], pd.to_numeric(deg_df[fd_col], errors="coerce")))

        # Load rMATS
        das_df = das_by_contrast[cid].copy() if (das_by_contrast and cid in das_by_contrast) else _load_rmats_contrast_df(bdir, cid, treat_lbl, ctrl_lbl, rmats_mode=rmats_mode)
        if not das_df.empty and {"gene_id", "gene_symbol"} <= set(das_df.columns):
            for gb, gs in zip(das_df["gene_id"].astype(str).str.split(".").str[0], das_df["gene_symbol"].astype(str)):
                if gs and gs != "nan" and not gs.startswith("ENS"):
                    sym_map.setdefault(gb, gs)
        rmats_targets, all_rmats_by_gene = _extract_rmats_target_intervals(das_df)
        if not das_df.empty:
            st_col = "stable_significant" if "stable_significant" in das_df.columns else ("report_rs_das" if "report_rs_das" in das_df.columns else None)
            if st_col:
                das_sig_genes = set(das_df.loc[das_df[st_col].astype(bool), "gene_id"].astype(str).str.split(".").str[0])
            else:
                das_sig_genes = {r["gene_base"] for r in rmats_targets}
        else:
            das_sig_genes = set()
        baseline_g = deg_sig_genes | das_sig_genes

        # Load DEXSeq + coordinate merge + auto sign orientation
        dex_f = next((p for p in (
            (bdir / "dexseq_exon" / f"DEXSeqResults.{cid}.csv") if bdir else None,
            (bdir / "differential/dexseq_exon" / f"DEXSeqResults.{cid}.csv") if bdir else None,
            (bdir / "dexseq_exon" / f"DEXSeqResults.{cid}.tsv") if bdir else None,
        ) if p is not None and p.is_file()), None)
        dex_raw = _read_tabular_safe(dex_f) if dex_f else pd.DataFrame()
        dex_inverted = False
        lfc_col_name = ""
        df_c1_best = pd.DataFrame()
        df_c4 = pd.DataFrame()
        dex_sig_genes: set[str] = set()
        dex_m = pd.DataFrame()
        pos_stats = {"deciles": [f"{i*10}-{(i+1)*10}%" for i in range(10)], "up_oe": [1.0]*10, "dn_oe": [1.0]*10, "up_n": 0, "dn_n": 0, "p_up_vs_bg": None, "p_dn_vs_bg": None, "p_up_vs_dn": None}

        if not dex_raw.empty:
            lfc_c = next((c for c in dex_raw.columns if c.lower().startswith("log2fold") or c.lower() in ("log2fc", "logfc")), None)
            grp_c = next((c for c in ("groupID", "gene_id", "geneID") if c in dex_raw.columns), dex_raw.columns[0])
            feat_c = next((c for c in ("featureID", "exon_id") if c in dex_raw.columns), None)
            if lfc_c and feat_c:
                lfc_col_name = lfc_c
                dex_raw["gene_base"] = dex_raw[grp_c].astype(str).str.split("+").str[0].str.split(".").str[0]
                dex_raw["lfc_tc"], dex_inverted = _orient_dexseq_df(dex_raw, lfc_c, treat_lbl, ctrl_lbl)
                dex_raw["raw_lfc"] = pd.to_numeric(dex_raw[lfc_c], errors="coerce")
                dex_raw["padj"] = pd.to_numeric(dex_raw["padj"], errors="coerce")
                dex_raw["exonBaseMean"] = pd.to_numeric(dex_raw.get("exonBaseMean", 0.0), errors="coerce")
                if not gff_df.empty:
                    dex_m = dex_raw.merge(gff_df, left_on=["gene_base", feat_c], right_on=["gene_base", "feature_id"], how="left")
                else:
                    dex_m = dex_raw.copy()
                    dex_m["feature_id"] = dex_m[feat_c].astype(str)
                    dex_m["chr"] = ""
                    dex_m["start"] = np.nan
                    dex_m["end"] = np.nan
                    dex_m["strand"] = "+"
                dex_m["gene_symbol"] = dex_m["gene_base"].map(lambda g: sym_map.get(g, g))
                dex_m["pos_5to3"] = [gff_pos_frac.get((g, f), np.nan) for g, f in zip(dex_m["gene_base"], dex_m[feat_c])]

                dex_sig_padj = dex_m[dex_m["padj"] < fdr_cutoff].copy()
                dex_sig_lfc = dex_m[(dex_m["padj"] < fdr_cutoff) & (dex_m["lfc_tc"].abs() >= 0.5)].copy()
                dex_sig_genes = set(dex_sig_lfc["gene_base"])

                # Class I: Coordinate-Verified Exons
                dex_by_gene: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for rec in dex_sig_padj[["gene_base", feat_c, "chr", "start", "end", "strand", "lfc_tc", "raw_lfc", "padj", "exonBaseMean"]].to_dict("records"):
                    if pd.notna(rec["start"]):
                        dex_by_gene[rec["gene_base"]].append(rec)
                c1_rows: list[dict[str, Any]] = []
                for r in rmats_targets:
                    gb = r["gene_base"]
                    if gb not in dex_by_gene or not r["ivs"]:
                        continue
                    for d in dex_by_gene[gb]:
                        if d["chr"] != r["chr"]:
                            continue
                        if any(max(s, d["start"]) <= min(e, d["end"]) for s, e in r["ivs"]):
                            ov_bp = max(max(0, min(e, d["end"]) - max(s, d["start"]) + 1) for s, e in r["ivs"])
                            c1_rows.append({
                                "gene_base": gb,
                                "gene_symbol": r.get("gene_symbol", sym_map.get(gb, gb)),
                                "event_id": r["event_id"],
                                "event_type": r["event_type"],
                                "stable_significant": bool(r.get("stable_significant", False)),
                                "rmats_dpsi": float(r["dpsi"]),
                                "rmats_fdr": float(r["fdr"]),
                                "dexseq_exon": str(d[feat_c]),
                                "dexseq_coord": f"{d['chr']}:{int(d['start'])}-{int(d['end'])}({d['strand']})",
                                "dexseq_lfc": float(d["lfc_tc"]),
                                "dexseq_raw_lfc": float(d["raw_lfc"]),
                                "dexseq_padj": float(d["padj"]),
                                "overlap_bp": int(ov_bp),
                                "concordant_fixed": bool(float(r["dpsi"]) * float(d["lfc_tc"]) > 0),
                                "concordant_raw": bool(float(r["dpsi"]) * float(d["raw_lfc"]) > 0),
                            })
                if c1_rows:
                    df_c1_best = pd.DataFrame(c1_rows).sort_values("dexseq_padj").drop_duplicates("event_id")

                # Class IV: DEXSeq-only exons
                sig_rmats_ivs: dict[str, list[tuple[str, int, int]]] = defaultdict(list)
                for r in rmats_targets:
                    for s, e in r["ivs"]:
                        sig_rmats_ivs[r["gene_base"]].append((r["chr"], s, e))
                c4_rows = []
                for rec in dex_sig_lfc.to_dict("records"):
                    gb = rec["gene_base"]
                    if pd.isna(rec.get("start")):
                        continue
                    if any(ch == rec["chr"] and max(s, rec["start"]) <= min(e, rec["end"]) for ch, s, e in sig_rmats_ivs.get(gb, [])):
                        continue
                    in_tested = any(ch == rec["chr"] and max(s, rec["start"]) <= min(e, rec["end"]) for ch, s, e in all_rmats_by_gene.get(gb, []))
                    p = rec.get("pos_5to3", np.nan)
                    reg = "Other" if pd.isna(p) else ("5' First/TSS Region (0-15%)" if p <= 0.15 else ("3' Terminal/PAS Region (85-100%)" if p >= 0.85 else "Internal Exon Body (15-85%)"))
                    c4_rows.append({
                        "gene_base": gb, "gene_symbol": rec["gene_symbol"], "feature_id": str(rec[feat_c]),
                        "coord": f"{rec['chr']}:{int(rec['start'])}-{int(rec['end'])}({rec['strand']})",
                        "lfc": float(rec["lfc_tc"]), "padj": float(rec["padj"]),
                        "exonBaseMean": float(rec["exonBaseMean"]) if pd.notna(rec["exonBaseMean"]) else 0.0,
                        "pos_5to3": None if pd.isna(p) else float(p),
                        "region_bin": reg, "in_tested_rmats": bool(in_tested),
                        "gene_in_deg": gb in deg_sig_genes, "gene_in_rmats": gb in das_sig_genes,
                    })
                df_c4 = pd.DataFrame(c4_rows)

                # 5'->3' decile profile
                pos_df = dex_m.dropna(subset=["pos_5to3", "padj"]).copy()
                if len(pos_df) >= 10:
                    pos_df["decile"] = np.clip((pos_df["pos_5to3"] * 10).astype(int), 0, 9)
                    bg_cnt = pos_df["decile"].value_counts().reindex(range(10), fill_value=0).to_numpy(dtype=float)
                    up_cnt = pos_df[(pos_df["padj"] < fdr_cutoff) & (pos_df["lfc_tc"] >= 0.5)]["decile"].value_counts().reindex(range(10), fill_value=0).to_numpy(dtype=float)
                    dn_cnt = pos_df[(pos_df["padj"] < fdr_cutoff) & (pos_df["lfc_tc"] <= -0.5)]["decile"].value_counts().reindex(range(10), fill_value=0).to_numpy(dtype=float)
                    bg_pct = bg_cnt / max(1.0, bg_cnt.sum()) * 100.0
                    up_oe = (up_cnt / max(1.0, up_cnt.sum()) * 100.0) / np.maximum(bg_pct, 1e-6) if up_cnt.sum() > 0 else np.ones(10)
                    dn_oe = (dn_cnt / max(1.0, dn_cnt.sum()) * 100.0) / np.maximum(bg_pct, 1e-6) if dn_cnt.sum() > 0 else np.ones(10)
                    p_up_bg = None
                    p_dn_bg = None
                    p_up_dn = None
                    if up_cnt.sum() >= 10 and np.all(bg_cnt > 0):
                        exp_u = bg_cnt * (up_cnt.sum() / bg_cnt.sum())
                        _, p_up_bg = stats.chisquare(up_cnt, f_exp=exp_u)
                    if dn_cnt.sum() >= 10 and np.all(bg_cnt > 0):
                        exp_d = bg_cnt * (dn_cnt.sum() / bg_cnt.sum())
                        _, p_dn_bg = stats.chisquare(dn_cnt, f_exp=exp_d)
                    if up_cnt.sum() >= 10 and dn_cnt.sum() >= 10:
                        obs_ud = up_cnt + 1.0
                        exp_ud = (dn_cnt + 1.0) * (obs_ud.sum() / (dn_cnt + 1.0).sum())
                        _, p_up_dn = stats.chisquare(obs_ud, f_exp=exp_ud)
                    pos_stats = {
                        "deciles": [f"{i*10}-{(i+1)*10}%" for i in range(10)],
                        "up_oe": [round(float(v), 3) for v in up_oe],
                        "dn_oe": [round(float(v), 3) for v in dn_oe],
                        "up_n": int(up_cnt.sum()),
                        "dn_n": int(dn_cnt.sum()),
                        "p_up_vs_bg": p_up_bg,
                        "p_dn_vs_bg": p_dn_bg,
                        "p_up_vs_dn": p_up_dn,
                    }

        # Class II: SUPPA2 full 7-class + AF/AL
        sup_pair = f"{ctrl_lbl}-{treat_lbl}" if (ctrl_lbl and treat_lbl) else cid
        sup_local_f = next((p for p in (
            (bdir / "suppa" / cid / "diffsplice/per_local_event/local_events.dpsi") if bdir else None,
            *((sorted((bdir / "suppa" / cid / "diffsplice/per_local_event").glob("*.dpsi"))) if (bdir and (bdir / "suppa" / cid / "diffsplice/per_local_event").is_dir()) else []),
            (bdir / "rnasplicing/suppa/diffsplice_ioe" / f"{sup_pair}_local_diffsplice.dpsi") if bdir else None,
        ) if p is not None and p.is_file()), None)
        df_sup_sum = pd.DataFrame([{"etype": et, "tested": 0, "nom_n": 0, "nom_genes": 0, "fdr_n": 0, "fdr_genes": 0} for et in ("SE", "A3", "A5", "MX", "RI", "AF", "AL")])
        af_al_nom = pd.DataFrame()
        af_al_fdr = pd.DataFrame()
        af_al_genes: set[str] = set()
        af_al_nom_genes: set[str] = set()
        sup_fdr_total = 0
        if sup_local_f is not None:
            raw_s = _read_tabular_safe(sup_local_f, sep="\t")
            if not raw_s.empty:
                if (not isinstance(raw_s.index, pd.RangeIndex)) or (len(raw_s.columns) >= 2 and "dpsi" in str(raw_s.columns[0]).lower()):
                    raw_s = raw_s.reset_index().rename(columns={raw_s.index.name or "index": "event_id"})
                ev_c = next((c for c in ("event_id", "Event_id", "id") if c in raw_s.columns), raw_s.columns[0])
                dp_c = next((c for c in raw_s.columns if c != ev_c and "dpsi" in str(c).lower()), raw_s.columns[1])
                pv_c = next((c for c in raw_s.columns if c not in (ev_c, dp_c) and ("p-val" in str(c).lower() or "pval" in str(c).lower() or "p_val" in str(c).lower() or "pvalue" in str(c).lower())), raw_s.columns[2])
                sup_df = pd.DataFrame({
                    "event_id": raw_s[ev_c].astype(str),
                    "dpsi": pd.to_numeric(raw_s[dp_c], errors="coerce"),
                    "pvalue": pd.to_numeric(raw_s[pv_c], errors="coerce"),
                }).dropna(subset=["dpsi", "pvalue"])
                if not sup_df.empty:
                    sup_df["etype"] = sup_df["event_id"].str.extract(r";([A-Z0-9]+):")[0].fillna("OTHER")
                    sup_df["gene_base"] = sup_df["event_id"].str.split(";").str[0].str.split("_and_").str[0].str.split(".").str[0]
                    sup_df["gene_symbol"] = sup_df["gene_base"].map(lambda g: sym_map.get(g, g))
                    sup_df["bh_fdr"] = multipletests(sup_df["pvalue"], method="fdr_bh")[1]
                    sup_df["sig_nom"] = (sup_df["pvalue"] < fdr_cutoff) & (sup_df["dpsi"].abs() >= 0.10)
                    sup_df["sig_fdr"] = (sup_df["bh_fdr"] < fdr_cutoff) & (sup_df["dpsi"].abs() >= 0.10)
                    sup_fdr_total = int(sup_df["sig_fdr"].sum())
                    sum_rows = []
                    for et in ("SE", "A3", "A5", "MX", "RI", "AF", "AL"):
                        sub = sup_df[sup_df["etype"] == et]
                        sum_rows.append({
                            "etype": et, "tested": int(len(sub)),
                            "nom_n": int(sub["sig_nom"].sum()), "nom_genes": int(sub.loc[sub["sig_nom"], "gene_base"].nunique()),
                            "fdr_n": int(sub["sig_fdr"].sum()), "fdr_genes": int(sub.loc[sub["sig_fdr"], "gene_base"].nunique()),
                        })
                    df_sup_sum = pd.DataFrame(sum_rows)
                    af_al_nom = sup_df[sup_df["sig_nom"] & sup_df["etype"].isin(["AF", "AL"])].copy()
                    af_al_fdr = sup_df[sup_df["sig_fdr"] & sup_df["etype"].isin(["AF", "AL"])].copy()
                    af_al_nom["in_ps_das"] = af_al_nom["gene_base"].isin(das_sig_genes)
                    af_al_nom["in_deg"] = af_al_nom["gene_base"].isin(deg_sig_genes)
                    af_al_fdr["in_ps_das"] = af_al_fdr["gene_base"].isin(das_sig_genes)
                    af_al_fdr["in_deg"] = af_al_fdr["gene_base"].isin(deg_sig_genes)
                    af_al_genes = set(af_al_fdr["gene_base"])
                    af_al_nom_genes = set(af_al_nom["gene_base"])

        # Class III: Full-Length Isoform Switching (stageR DTU + SUPPA IOI + Salmon TPM + GTF + SpliceTrace)
        dtu_pair = f"{treat_lbl}-{ctrl_lbl}" if (treat_lbl and ctrl_lbl) else cid
        dtu_f = next((p for p in (
            (bdir / "dtu/results/stager" / f"getAdjustedPValues.{cid}.tsv") if bdir else None,
            (bdir / "dtu/results/stager" / f"getAdjustedPValues.{dtu_pair}.tsv") if bdir else None,
            (bdir / "differential/dtu/stager" / f"getAdjustedPValues.{cid}.tsv") if bdir else None,
            (bdir / "differential/dtu/stager" / f"getAdjustedPValues.{dtu_pair}.tsv") if bdir else None,
            (bdir / "dtu/stager" / f"getAdjustedPValues.{cid}.tsv") if bdir else None,
            (bdir / "dtu/stager" / f"getAdjustedPValues.{dtu_pair}.tsv") if bdir else None,
        ) if p is not None and p.is_file()), None)
        sup_ioi_f = next((p for p in (
            (bdir / "suppa" / cid / "diffsplice/per_isoform/isoform_events.dpsi") if bdir else None,
            *((sorted((bdir / "suppa" / cid / "diffsplice/per_isoform").glob("*.dpsi"))) if (bdir and (bdir / "suppa" / cid / "diffsplice/per_isoform").is_dir()) else []),
            (bdir / "rnasplicing/suppa/diffsplice_ioi" / f"{sup_pair}_transcript_diffsplice.dpsi") if bdir else None,
        ) if p is not None and p.is_file()), None)

        df_sw = pd.DataFrame()
        sel_sw = pd.DataFrame()
        dtu_gene_q05_set: set[str] = set()
        dtu_tx_q05_cnt = 0
        ioi_concordant_cnt = 0
        ioi_total_cnt = 0
        interactive_genes: list[dict[str, Any]] = []
        st_sig = pd.DataFrame()

        if cid in st_dir_map and (st_dir_map[cid] / "protein_and_nmd_consequences.tsv").is_file():
            st_raw = _read_tabular_safe(st_dir_map[cid] / "protein_and_nmd_consequences.tsv", sep="\t")
            if not st_raw.empty and "gene_id" in st_raw.columns:
                st_raw["gene_base"] = st_raw["gene_id"].astype(str).str.split(".").str[0]
                sig_eids = {r["event_id"] for r in rmats_targets}
                st_sig = st_raw[st_raw["event_id"].isin(sig_eids)].copy() if sig_eids else st_raw.head(0)

        if dtu_f is not None and not tpm_df.empty:
            dtu_st = _read_tabular_safe(dtu_f, sep="\t")
            if not dtu_st.empty:
                gid_dtu = next((c for c in ("geneID", "gene_id", "groupID") if c in dtu_st.columns), dtu_st.columns[0])
                tx_dtu = next((c for c in ("txID", "transcript_id", "featureID") if c in dtu_st.columns), dtu_st.columns[1])
                gq_dtu = next((c for c in ("gene", "gene_q", "padj") if c in dtu_st.columns), dtu_st.columns[2])
                tq_dtu = next((c for c in ("transcript", "transcript_q", "tx_q") if c in dtu_st.columns), gq_dtu)
                dtu_st["gene_base"] = dtu_st[gid_dtu].astype(str).str.split(".").str[0]
                dtu_st["tx_base"] = dtu_st[tx_dtu].astype(str).str.split(".").str[0]
                dtu_st["gene_q_v"] = pd.to_numeric(dtu_st[gq_dtu], errors="coerce")
                dtu_st["tx_q_v"] = pd.to_numeric(dtu_st[tq_dtu], errors="coerce")

                # Resolve replicate TPM columns dynamically
                tr_cols = [c for c in treat_sids if c in tpm_df.columns] or [c for c in tpm_df.columns if treat_lbl and c.startswith(treat_lbl + "_")]
                ct_cols = [c for c in ctrl_sids if c in tpm_df.columns] or [c for c in tpm_df.columns if ctrl_lbl and c.startswith(ctrl_lbl + "_")]
                if tr_cols and ct_cols:
                    samp_cols = ct_cols + tr_cols
                    tpm_work = tpm_df[["tx_base", "gene_base", *samp_cols]].copy()
                    for sc in samp_cols:
                        tpm_work[sc] = pd.to_numeric(tpm_work[sc], errors="coerce").fillna(0.0)
                    g_sums = tpm_work.groupby("gene_base")[samp_cols].transform("sum")
                    if_mat = tpm_work[samp_cols].div(g_sums.replace(0.0, np.nan))
                    tpm_work["mean_tpm_ctrl"] = tpm_work[ct_cols].mean(axis=1)
                    tpm_work["mean_tpm_treat"] = tpm_work[tr_cols].mean(axis=1)
                    tpm_work["gene_tpm_ctrl"] = g_sums[ct_cols].mean(axis=1)
                    tpm_work["gene_tpm_treat"] = g_sums[tr_cols].mean(axis=1)
                    tpm_work["mean_if_ctrl"] = if_mat[ct_cols].mean(axis=1)
                    tpm_work["mean_if_treat"] = if_mat[tr_cols].mean(axis=1)
                    tpm_work["dIF"] = tpm_work["mean_if_treat"] - tpm_work["mean_if_ctrl"]
                    for sc in samp_cols:
                        tpm_work[f"IF_{sc}"] = if_mat[sc]

                    dtu_m = dtu_st.merge(tpm_work.drop(columns=["gene_base"]), on="tx_base", how="left")
                    if sup_ioi_f is not None:
                        raw_i = _read_tabular_safe(sup_ioi_f, sep="\t")
                        if not raw_i.empty:
                            if (not isinstance(raw_i.index, pd.RangeIndex)) or (len(raw_i.columns) >= 2 and "dpsi" in str(raw_i.columns[0]).lower()):
                                raw_i = raw_i.reset_index().rename(columns={raw_i.index.name or "index": "event_id"})
                            ev_i = raw_i.columns[0]
                            dp_i = next((c for c in raw_i.columns if c != ev_i and "dpsi" in str(c).lower()), raw_i.columns[1])
                            ioi_df = pd.DataFrame({
                                "tx_base": raw_i[ev_i].astype(str).str.split(";").str[-1].str.split(".").str[0],
                                "ioi_dpsi": pd.to_numeric(raw_i[dp_i], errors="coerce"),
                            }).dropna(subset=["ioi_dpsi"]).drop_duplicates("tx_base")
                            dtu_m = dtu_m.merge(ioi_df, on="tx_base", how="left")
                    else:
                        dtu_m["ioi_dpsi"] = np.nan

                    dtu_m["biotype"] = dtu_m["tx_base"].map(lambda t: tx_meta.get(t, {}).get("biotype", "protein_coding"))
                    dtu_m["aa_len"] = dtu_m["tx_base"].map(lambda t: _cds_aa_len(tx_cds, t))
                    dtu_m["utr5_len"] = dtu_m["tx_base"].map(lambda t: _utr_lens(tx_meta, tx_exons, tx_cds, t)[0])
                    dtu_m["utr3_len"] = dtu_m["tx_base"].map(lambda t: _utr_lens(tx_meta, tx_exons, tx_cds, t)[1])
                    dtu_m["gene_symbol"] = dtu_m["gene_base"].map(lambda g: sym_map.get(g, g))

                    sig_dtu = dtu_m[dtu_m["gene_q_v"] < fdr_cutoff].copy()
                    dtu_gene_q05_set = set(sig_dtu["gene_base"])
                    dtu_tx_q05_cnt = int((dtu_st["tx_q_v"] < fdr_cutoff).sum())
                    st_genes_set = set(st_sig["gene_base"]) if not st_sig.empty else set()

                    _tier_rank = {"both_tx_sig": 0, "one_tx_sig": 1, "gene_only": 2}
                    sw_rows = []
                    for gb, grp in sig_dtu.groupby("gene_base"):
                        gv = grp.dropna(subset=["dIF"])
                        gv_pos = gv[gv["dIF"] > 0]
                        gv_neg = gv[gv["dIF"] < 0]
                        if gv_pos.empty or gv_neg.empty:
                            continue
                        gv_pos_sig = gv_pos[gv_pos["tx_q_v"] < fdr_cutoff]
                        gv_neg_sig = gv_neg[gv_neg["tx_q_v"] < fdr_cutoff]
                        tup = (gv_pos_sig if not gv_pos_sig.empty else gv_pos).sort_values("dIF", ascending=False).iloc[0]
                        tdn = (gv_neg_sig if not gv_neg_sig.empty else gv_neg).sort_values("dIF", ascending=True).iloc[0]
                        tx_up_q = float(tup["tx_q_v"]) if pd.notna(tup["tx_q_v"]) else 1.0
                        tx_dn_q = float(tdn["tx_q_v"]) if pd.notna(tdn["tx_q_v"]) else 1.0
                        if tx_up_q < fdr_cutoff and tx_dn_q < fdr_cutoff:
                            tx_tier = "both_tx_sig"
                        elif tx_up_q < fdr_cutoff or tx_dn_q < fdr_cutoff:
                            tx_tier = "one_tx_sig"
                        else:
                            tx_tier = "gene_only"
                        fate = _classify_isoform_fate(
                            str(tdn["biotype"]), str(tup["biotype"]),
                            int(tdn["aa_len"]), int(tup["aa_len"]),
                            int(tdn["utr5_len"]), int(tup["utr5_len"]),
                            int(tdn["utr3_len"]), int(tup["utr3_len"]),
                        )
                        g_lfc = deg_lfc_map.get(gb, np.nan)
                        g_fdr = deg_fdr_map.get(gb, np.nan)
                        is_deg = bool(pd.notna(g_fdr) and g_fdr < fdr_cutoff)
                        sw_rows.append({
                            "gene_base": gb, "gene_symbol": str(tup["gene_symbol"]), "gene_q": float(tup["gene_q_v"]),
                            "tx_up": str(tup["tx_base"]), "tx_up_biotype": str(tup["biotype"]), "tx_up_aa": int(tup["aa_len"]),
                            "tx_up_dif": float(tup["dIF"]), "tx_up_q": tx_up_q,
                            "tx_up_ioi_dpsi": None if pd.isna(tup.get("ioi_dpsi")) else float(tup["ioi_dpsi"]),
                            "tx_dn": str(tdn["tx_base"]), "tx_dn_biotype": str(tdn["biotype"]), "tx_dn_aa": int(tdn["aa_len"]),
                            "tx_dn_dif": float(tdn["dIF"]), "tx_dn_q": tx_dn_q,
                            "tx_confirm_tier": tx_tier,
                            "delta_aa": int(tup["aa_len"] - tdn["aa_len"]),
                            "switch_mag": float(0.5 * (abs(tup["dIF"]) + abs(tdn["dIF"]))),
                            "max_abs_dif": float(max(abs(tup["dIF"]), abs(tdn["dIF"]))),
                            "fate": fate,
                            "deg_log2fc": None if pd.isna(g_lfc) else float(g_lfc),
                            "deg_fdr": None if pd.isna(g_fdr) else float(g_fdr),
                            "hidden_from_deg": not is_deg,
                            "gene_tpm_ctrl": float(tup["gene_tpm_ctrl"]) if pd.notna(tup["gene_tpm_ctrl"]) else 0.0,
                            "gene_tpm_treat": float(tup["gene_tpm_treat"]) if pd.notna(tup["gene_tpm_treat"]) else 0.0,
                            "has_splicetrace": gb in st_genes_set,
                        })
                    if sw_rows:
                        df_sw = pd.DataFrame(sw_rows)
                        df_sw["_tier_ord"] = df_sw["tx_confirm_tier"].map(_tier_rank).fillna(3)
                        df_sw["_min_tx_q"] = np.minimum(df_sw["tx_up_q"], df_sw["tx_dn_q"])
                        df_sw = (
                            df_sw.sort_values(["_tier_ord", "_min_tx_q", "gene_q", "switch_mag"], ascending=[True, True, True, False])
                            .drop(columns=["_tier_ord", "_min_tx_q"])
                            .reset_index(drop=True)
                        )
                        sw_ioi = df_sw.dropna(subset=["tx_up_ioi_dpsi"])
                        ioi_total_cnt = int(len(sw_ioi))
                        ioi_concordant_cnt = int((sw_ioi["tx_up_dif"] * sw_ioi["tx_up_ioi_dpsi"] > 0).sum()) if ioi_total_cnt else 0

                        # Pick up to 45 balanced representative interactive switch genes matching table preview
                        p_top = df_sw.head(30)
                        p_st = df_sw[df_sw["has_splicetrace"]].head(8)
                        p_nmd = df_sw[df_sw["hidden_from_deg"] & df_sw["fate"].str.contains("NMD")].head(10)
                        p_ri = df_sw[df_sw["fate"].str.contains("Retained|ORF")].head(10)
                        sel_sw = pd.concat([p_top, p_st, p_nmd, p_ri]).drop_duplicates("gene_base").head(45)
                        for rec in sel_sw.to_dict("records"):
                            gb = rec["gene_base"]
                            g_txs = dtu_m[dtu_m["gene_base"] == gb].sort_values("mean_tpm_treat", ascending=False).head(5)
                            must = dtu_m[(dtu_m["gene_base"] == gb) & (dtu_m["tx_base"].isin([rec["tx_up"], rec["tx_dn"]]))]
                            g_txs = pd.concat([must, g_txs]).drop_duplicates("tx_base").head(5)
                            all_ex = [ex for tb in g_txs["tx_base"] for ex in tx_exons.get(tb, [])]
                            gmin = min((s for s, _ in all_ex), default=1)
                            gmax = max((e for _, e in all_ex), default=1000)
                            strand = tx_meta.get(rec["tx_up"], {}).get("strand", "+")
                            chrom = tx_meta.get(rec["tx_up"], {}).get("chr", "")
                            tx_list = []
                            for tr in g_txs.to_dict("records"):
                                tb = tr["tx_base"]
                                cds = tx_cds.get(tb, [])
                                cspan = [cds[0][0], cds[-1][1]] if cds else None
                                role = "T_up" if tb == rec["tx_up"] else ("T_dn" if tb == rec["tx_dn"] else "Other")
                                tx_list.append({
                                    "tx_id": tb, "role": role, "biotype": str(tr["biotype"]), "aa_len": int(tr["aa_len"]),
                                    "tx_q": float(tr["tx_q_v"]) if pd.notna(tr["tx_q_v"]) else 1.0,
                                    "if_ctrl": round(float(tr["mean_if_ctrl"] or 0), 4),
                                    "if_treat": round(float(tr["mean_if_treat"] or 0), 4),
                                    "dIF": round(float(tr["dIF"] or 0), 4),
                                    "tpm_ctrl": round(float(tr["mean_tpm_ctrl"] or 0), 2),
                                    "tpm_treat": round(float(tr["mean_tpm_treat"] or 0), 2),
                                    "exons": [[s, e] for s, e in tx_exons.get(tb, [])],
                                    "cds_span": cspan,
                                })
                            dex_bins = []
                            if not dex_m.empty:
                                gd = dex_m[dex_m["gene_base"] == gb].dropna(subset=["start"]).sort_values("start")
                                dex_bins = [
                                    {"fid": str(b.get("feature_id", "")), "start": int(b["start"]), "end": int(b["end"]),
                                     "lfc": round(float(b["lfc_tc"]), 3) if pd.notna(b["lfc_tc"]) else 0.0,
                                     "padj": float(b["padj"]) if pd.notna(b["padj"]) else 1.0}
                                    for b in gd.to_dict("records")
                                ]
                            st_notes = []
                            if not st_sig.empty:
                                for x in st_sig[st_sig["gene_base"] == gb].head(3).to_dict("records"):
                                    st_notes.append({
                                        "event_id": str(x["event_id"]),
                                        "nmd_switch": str(x.get("nmd_switch_class", "")),
                                        "frame": str(x.get("frame_status", "")),
                                        "dpsi": float(x.get("dpsi", 0.0)),
                                        "fdr": None if pd.isna(x.get("fdr")) else float(x.get("fdr")),
                                        "summary": str(x.get("functional_consequence_summary", "")),
                                    })
                            interactive_genes.append({
                                "gene_base": gb, "gene_symbol": rec["gene_symbol"], "chr": chrom, "strand": strand,
                                "span": [int(gmin), int(gmax)], "gene_q": float(rec["gene_q"]),
                                "tx_up_q": float(rec["tx_up_q"]), "tx_dn_q": float(rec["tx_dn_q"]),
                                "tx_confirm_tier": rec["tx_confirm_tier"],
                                "deg_log2fc": rec["deg_log2fc"], "deg_fdr": rec["deg_fdr"],
                                "hidden_from_deg": bool(rec["hidden_from_deg"]),
                                "gene_tpm_ctrl": round(float(rec["gene_tpm_ctrl"]), 2),
                                "gene_tpm_treat": round(float(rec["gene_tpm_treat"]), 2),
                                "fate": rec["fate"], "delta_aa": int(rec["delta_aa"]), "switch_mag": round(float(rec["switch_mag"]), 4),
                                "txs": tx_list, "dex_bins": dex_bins, "splicetrace": st_notes,
                            })

        # Uplift metrics
        unlocked_afal = af_al_genes - baseline_g
        unlocked_dtu = dtu_gene_q05_set - baseline_g - unlocked_afal
        unlocked_dex = dex_sig_genes - baseline_g - unlocked_afal - unlocked_dtu
        union_all = baseline_g | af_al_genes | dtu_gene_q05_set | dex_sig_genes
        bench_rows.append({
            "contrast": cid, "deg_genes": len(deg_sig_genes), "rmats_ps_genes": len(das_sig_genes),
            "baseline_genes": len(baseline_g),
            "unlocked_afal": len(unlocked_afal), "unlocked_dtu": len(unlocked_dtu), "unlocked_dexseq": len(unlocked_dex),
            "total_union": len(union_all),
            "uplift_pct": round((len(unlocked_afal) + len(unlocked_dtu) + len(unlocked_dex)) / max(1, len(baseline_g)) * 100.0, 1),
        })

        # Save per-contrast tabular outputs
        if not df_c1_best.empty:
            df_c1_best.to_csv(out / f"{slug}_class1_coord_verified.tsv", sep="\t", index=False)
        df_sup_sum.to_csv(out / f"{slug}_class2_suppa_7class.tsv", sep="\t", index=False)
        if not af_al_nom.empty:
            af_al_nom.sort_values(["bh_fdr", "pvalue"]).to_csv(out / f"{slug}_class2_af_al_events.tsv", sep="\t", index=False)
        pd.DataFrame({
            "decile": pos_stats["deciles"],
            "up_obs_over_exp": pos_stats["up_oe"],
            "dn_obs_over_exp": pos_stats["dn_oe"],
            "up_n": pos_stats["up_n"],
            "dn_n": pos_stats["dn_n"],
        }).to_csv(out / f"{slug}_class2_positional_gradient.tsv", sep="\t", index=False)
        if not df_sw.empty:
            df_sw.to_csv(out / f"{slug}_class3_isoform_switches.tsv", sep="\t", index=False)
        if not df_c4.empty:
            df_c4.to_csv(out / f"{slug}_class4_dexseq_only.tsv", sep="\t", index=False)

        c4_genes = set(df_c4["gene_base"]) if not df_c4.empty else set()
        sw_genes = set(df_sw["gene_base"]) if not df_sw.empty else set()
        c1_genes = int(df_c1_best["gene_base"].nunique()) if not df_c1_best.empty else 0
        c1_conc = float(df_c1_best["concordant_fixed"].mean() * 100.0) if not df_c1_best.empty else 0.0
        c1_raw = float(df_c1_best["concordant_raw"].mean() * 100.0) if not df_c1_best.empty else 0.0
        rho_c1, p_c1 = (None, None)
        if len(df_c1_best) >= 3 and df_c1_best["rmats_dpsi"].nunique() > 1 and df_c1_best["dexseq_lfc"].nunique() > 1:
            r_val, p_val = stats.spearmanr(df_c1_best["rmats_dpsi"], df_c1_best["dexseq_lfc"])
            rho_c1, p_c1 = float(r_val), float(p_val)

        c1_prev_df = df_c1_best.head(60) if not df_c1_best.empty else pd.DataFrame()
        c2_prev_df = af_al_nom.sort_values(["bh_fdr", "pvalue"]).head(60)[["gene_symbol", "gene_base", "etype", "event_id", "dpsi", "pvalue", "bh_fdr", "sig_fdr", "in_ps_das", "in_deg"]] if not af_al_nom.empty else pd.DataFrame()
        c3_cols = ["gene_symbol", "gene_base", "gene_q", "tx_up", "tx_up_biotype", "tx_up_aa", "tx_up_dif", "tx_up_q", "tx_dn", "tx_dn_biotype", "tx_dn_aa", "tx_dn_dif", "tx_dn_q", "tx_confirm_tier", "delta_aa", "switch_mag", "fate", "deg_log2fc", "deg_fdr", "hidden_from_deg", "gene_tpm_ctrl", "gene_tpm_treat", "has_splicetrace"]
        c3_prev_src = sel_sw if not sel_sw.empty else (df_sw.head(45) if not df_sw.empty else pd.DataFrame())
        c3_prev_df = c3_prev_src[c3_cols] if not c3_prev_src.empty else pd.DataFrame()
        c4_prev_df = df_c4.sort_values("padj").head(60) if not df_c4.empty else pd.DataFrame()

        gene_lists = {
            "class3_switches": df_sw["gene_symbol"].dropna().astype(str).drop_duplicates().tolist() if not df_sw.empty else [],
            "class1": df_c1_best["gene_symbol"].dropna().astype(str).drop_duplicates().tolist() if not df_c1_best.empty else [],
            "class2_af_al": af_al_nom.sort_values(["bh_fdr", "pvalue"])["gene_symbol"].dropna().astype(str).drop_duplicates().tolist() if not af_al_nom.empty else [],
            "class2_af_al_fdr": af_al_fdr.sort_values(["bh_fdr", "pvalue"])["gene_symbol"].dropna().astype(str).drop_duplicates().tolist() if not af_al_fdr.empty else [],
            "class4_dexseq": df_c4.sort_values("padj")["gene_symbol"].dropna().astype(str).drop_duplicates().tolist() if not df_c4.empty else [],
        }
        full_tsv_files = {
            "class1": f"{slug}_class1_coord_verified.tsv",
            "class2_af_al": f"{slug}_class2_af_al_events.tsv",
            "class3_switches": f"{slug}_class3_isoform_switches.tsv",
            "class4_dexseq": f"{slug}_class4_dexseq_only.tsv",
        }

        # Whole-gene DEXSeq bin tracks for Class I and Class IV preview genes
        gene_dex_bins: dict[str, dict[str, Any]] = {}
        if not dex_m.empty:
            target_gbs: set[str] = set()
            if not c1_prev_df.empty:
                target_gbs.update(c1_prev_df["gene_base"].astype(str))
            if not c4_prev_df.empty:
                target_gbs.update(c4_prev_df["gene_base"].astype(str))
            if not c2_prev_df.empty:
                target_gbs.update(c2_prev_df["gene_base"].astype(str))
            for gb in sorted(target_gbs):
                gd = dex_m[dex_m["gene_base"] == gb].dropna(subset=["start"]).sort_values("start")
                if gd.empty:
                    continue
                if len(gd) > 28:
                    sig_idx = set(gd[gd["padj"] < fdr_cutoff].sort_values("padj").head(16).index)
                    step = max(1, len(gd) // 12)
                    even_idx = set(gd.iloc[::step].index)
                    gd = gd.loc[gd.index.isin(sig_idx | even_idx)].sort_values("start").head(28)
                gene_dex_bins[gb] = {
                    "chr": str(gd["chr"].iloc[0]),
                    "strand": str(gd["strand"].iloc[0]),
                    "bins": [
                        {
                            "fid": str(b.get("feature_id", "")),
                            "start": int(b["start"]),
                            "end": int(b["end"]),
                            "lfc": round(float(b["lfc_tc"]), 3) if pd.notna(b["lfc_tc"]) else 0.0,
                            "padj": float(b["padj"]) if pd.notna(b["padj"]) else 1.0,
                            "pos_5to3": None if pd.isna(b.get("pos_5to3")) else round(float(b["pos_5to3"]), 3),
                            "exonBaseMean": round(float(b["exonBaseMean"]), 1) if pd.notna(b.get("exonBaseMean")) else 0.0,
                        }
                        for b in gd.to_dict("records")
                    ],
                }

        contrast_results[cid] = {
            "contrast_id": cid, "slug": slug, "treat_label": treat_lbl, "ctrl_label": ctrl_lbl,
            "dexseq_lfc_col": lfc_col_name, "dexseq_sign_inverted": dex_inverted,
            "counts": {
                "deg_genes": len(deg_sig_genes),
                "rmats_ps_genes": len(das_sig_genes),
                "baseline_genes": len(baseline_g),
                "class1_events": len(df_c1_best),
                "class1_genes": c1_genes,
                "class1_concordant": int(df_c1_best["concordant_fixed"].sum()) if not df_c1_best.empty else 0,
                "class1_concordant_pct": round(c1_conc, 1),
                "class1_raw_bug_pct": round(c1_raw, 1),
                "class1_spearman_rho": None if rho_c1 is None else round(rho_c1, 3),
                "class1_spearman_p": p_c1,
                "suppa_af_al_nom": len(af_al_nom),
                "suppa_af_al_fdr": len(af_al_fdr),
                "suppa_fdr_total": sup_fdr_total,
                "suppa_af_al_genes": len(af_al_genes),
                "suppa_af_al_novel_genes": len(af_al_genes - baseline_g),
                "suppa_af_al_nom_genes": len(af_al_nom_genes),
                "suppa_af_al_nom_novel_genes": len(af_al_nom_genes - baseline_g),
                "dtu_gene_q05": len(dtu_gene_q05_set),
                "dtu_tx_q05": dtu_tx_q05_cnt,
                "dtu_switch_resolved": len(df_sw),
                "dtu_switch_novel_genes": len(sw_genes - baseline_g),
                "dtu_hidden_from_deg": int(df_sw["hidden_from_deg"].sum()) if not df_sw.empty else 0,
                "dtu_hidden_strong_dif10": int((df_sw["hidden_from_deg"] & (df_sw["max_abs_dif"] >= 0.10)).sum()) if not df_sw.empty else 0,
                "dtu_ioi_concordant": f"{ioi_concordant_cnt}/{ioi_total_cnt}",
                "class4_exons": len(df_c4),
                "class4_genes": len(c4_genes),
                "class4_novel_genes": len(c4_genes - baseline_g),
                "class4_untested_exons": int((~df_c4["in_tested_rmats"]).sum()) if not df_c4.empty else 0,
            },
            "positional": pos_stats,
            "suppa_7class": df_sup_sum.to_dict("records"),
            "interactive_genes": interactive_genes,
            "gene_dex_bins": gene_dex_bins,
            "gene_lists": gene_lists,
            "full_tsv_files": full_tsv_files,
            "tables_preview": {
                "class1": c1_prev_df.to_dict("records") if not c1_prev_df.empty else [],
                "class2_af_al": c2_prev_df.to_dict("records") if not c2_prev_df.empty else [],
                "class3_switches": c3_prev_df.to_dict("records") if not c3_prev_df.empty else [],
                "class4_dexseq": c4_prev_df.to_dict("records") if not c4_prev_df.empty else [],
            },
            "_df_c1": df_c1_best,
            "_df_sup": df_sup_sum,
            "_df_sw": df_sw,
            "_df_c4": df_c4,
            "_af_al_nom": af_al_nom,
        }

    df_bench = pd.DataFrame(bench_rows)
    df_bench.to_csv(out / "multi_contrast_uplift.tsv", sep="\t", index=False)

    # Persist serializable JSON summary without internal DataFrames
    json_payload = _clean_json({
        "available": True,
        "contrasts": {
            cid: {k: v for k, v in val.items() if not k.startswith("_")}
            for cid, val in contrast_results.items()
        },
        "bench_multi_contrast": df_bench.to_dict("records"),
    })
    (out / "cross_paradigm_summary.json").write_text(json.dumps(json_payload, indent=2, allow_nan=False), encoding="utf-8")
    return {"available": True, "outdir": str(out), "results": contrast_results, "bench_multi_contrast": df_bench}


def _render_cp_figs_for_contrast(
    c_res: dict[str, Any],
    cp_out_dir: Path,
    fig_dir: Path,
    outdir: Path,
    panels: list[dict[str, Any]],
) -> None:
    cid = c_res["contrast_id"]
    slug = c_res["slug"]
    cnt = c_res["counts"]
    plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 9.5, "axes.titlesize": 11.2, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white", "axes.facecolor": "white"})

    # 1. Chapter 04: Taxonomy Inventory + Class I Coordinate Concordance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.9), dpi=240)
    fig.subplots_adjust(wspace=0.34, top=0.84, bottom=0.14, left=0.22, right=0.97)
    layers_info = [
        ("DESeq2 DEG\n(FDR < 0.05)", cnt["deg_genes"], 0, ED_SLATE),
        ("rMATS PS-DAS\n(FDR < 0.05, stable)", cnt["rmats_ps_genes"], 0, ED_NAVY),
        ("Class I: Coord-Verified\n(rMATS∩DEXSeq padj<.05)", cnt["class1_genes"], 0, ED_CERULEAN),
        ("Class II: SUPPA2 AF/AL\n(BH-FDR < 0.05)", cnt["suppa_af_al_genes"], cnt["suppa_af_al_novel_genes"], ED_AMBER),
        ("Class III: stageR DTU\n(stageR gene_q < 0.05)", cnt["dtu_switch_resolved"], cnt["dtu_switch_novel_genes"], ED_TEAL),
        ("Class IV: DEXSeq-Only\n(padj < 0.05, |lfc|≥0.5)", cnt["class4_genes"], cnt["class4_novel_genes"], ED_VIOLET),
    ]
    y = np.arange(len(layers_info))[::-1]
    xmax1 = max(10, max(x[1] for x in layers_info))
    for idx, (lbl, tot, nov, col) in enumerate(layers_info):
        yi = y[idx]
        ax1.barh(yi, tot, height=0.62, color=col, alpha=0.88)
        if nov > 0:
            ax1.barh(yi, nov, height=0.62, left=max(0, tot - nov), color="none", edgecolor="white", hatch="///", linewidth=0.8)
            ax1.text(tot + xmax1 * 0.02, yi, f"{tot:,} genes\n(+{nov:,} outside baseline)", va="center", ha="left", fontsize=8.4, fontweight="bold", color=ED_NAVY)
        else:
            sub_txt = f" ({cnt['class1_events']} events, {cnt['class1_concordant_pct']:.1f}% same-sign)" if "Class I" in lbl else ""
            ax1.text(tot + xmax1 * 0.02, yi, f"{tot:,} genes{sub_txt}", va="center", ha="left", fontsize=8.4, color=ED_NAVY)
    ax1.set_yticks(y)
    ax1.set_yticklabels([x[0] for x in layers_info], fontsize=8.8)
    ax1.set_xlim(0, xmax1 * 1.58)
    ax1.set_ylim(-0.6, 6.35)
    ax1.set_xlabel(f"Unique Significant Genes ({cid})", fontsize=9.5)
    ax1.set_title("A. 4-Class Cross-Paradigm Layer Inventory & Net New Genes", loc="left", pad=10)
    hatched_patch = mpatches.Patch(facecolor="#CBD5E1", edgecolor="white", hatch="///", label="Novel genes outside DEG + PS-DAS baseline")
    ax1.legend(handles=[hatched_patch], loc="upper right", frameon=True, fontsize=8.0)

    df_c1 = c_res["_df_c1"]
    if not df_c1.empty:
        ax2.axhline(0, color="#CBD5E1", lw=1, ls="--")
        ax2.axvline(0, color="#CBD5E1", lw=1, ls="--")
        y_hi = max(0.5, float(df_c1["dexseq_lfc"].max()) * 1.18)
        y_lo = min(-0.5, float(df_c1["dexseq_lfc"].min()) * 1.48)
        x_lo = min(-0.2, float(df_c1["rmats_dpsi"].min()) * 1.15)
        x_hi = max(0.2, float(df_c1["rmats_dpsi"].max()) * 1.25)
        ax2.axhspan(0, y_hi, xmin=0.5, xmax=1.0, color="#ECFDF5", alpha=0.5, zorder=0)
        ax2.axhspan(y_lo, 0, xmin=0.0, xmax=0.5, color="#ECFDF5", alpha=0.5, zorder=0)
        et_colors = {"SE": ED_NAVY, "A3SS": ED_TEAL, "A5SS": ED_AMBER, "RI": ED_CORAL, "MXE": ED_VIOLET}
        for et, sub in df_c1.groupby("event_type"):
            ax2.scatter(sub["rmats_dpsi"], sub["dexseq_lfc"], s=40, alpha=0.82, c=et_colors.get(et, ED_SLATE), edgecolors="white", lw=0.6, label=f"{et} (n={len(sub)})", zorder=3)
        conc_df = df_c1[df_c1["concordant_fixed"]].copy()
        seen_syms: set[str] = set()
        picked = []
        for sub_c, off_xy in (
            (conc_df.sort_values("dexseq_lfc", ascending=False), (-64, -6)),
            (conc_df.sort_values("rmats_dpsi", ascending=True), (10, 14)),
            (conc_df.sort_values("dexseq_lfc", ascending=True), (-10, -18)),
            (conc_df[conc_df["dexseq_lfc"] < 8].sort_values("rmats_dpsi", ascending=False), (-32, 16)),
        ):
            for _, r in sub_c.iterrows():
                gs = str(r["gene_symbol"])
                if gs not in seen_syms:
                    seen_syms.add(gs); picked.append((r, off_xy)); break
        for r, (ox, oy) in picked:
            ax2.annotate(f"{r['gene_symbol']}\n({r['dexseq_exon']})", xy=(r["rmats_dpsi"], r["dexseq_lfc"]), xytext=(ox, oy), textcoords="offset points", fontsize=7.8, fontweight="bold", color=ED_NAVY, arrowprops=dict(arrowstyle="-", color="#94A3B8", lw=0.7))
        ax2.set_xlim(x_lo, x_hi); ax2.set_ylim(y_lo, y_hi)
        rho_txt = f"Spearman ρ = {cnt['class1_spearman_rho']:.3f} (p = {cnt['class1_spearman_p']:.1e})\n" if cnt["class1_spearman_rho"] is not None else ""
        inv_txt = f"[Auto-corrected DEXSeq col: raw was {cnt['class1_raw_bug_pct']:.1f}%]" if c_res["dexseq_sign_inverted"] else "[DEXSeq orientation verified]"
        ax2.text(0.03, 0.96, f"All pts: rMATS FDR<.05 ∩ DEXSeq padj<.05\nn = {cnt['class1_events']} events ({cnt['class1_genes']} genes) · {cnt['class1_concordant_pct']:.1f}% same-sign\n{rho_txt}{inv_txt}", transform=ax2.transAxes, va="top", ha="left", fontsize=8.0, bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CBD5E1", alpha=0.95))
        ax2.legend(loc="lower right", frameon=True, fontsize=8.0, ncol=2)
    else:
        ax2.text(0.5, 0.5, "No 1-bp overlapping DEXSeq × rMATS pairs", ha="center", va="center", color=ED_SLATE)
    ax2.set_xlabel(f"rMATS ΔΨ ({c_res['treat_label']} − {c_res['ctrl_label']})", fontsize=9.5)
    ax2.set_ylabel("Oriented DEXSeq log2FC (Treat / Ctrl)", fontsize=9.5)
    ax2.set_title("B. Class I Exact 1-bp Coordinate Overlap Concordance", loc="left", pad=10)
    fig.suptitle(f"Cross-Paradigm Integration: Coordinate Concordance & Blind-Spot Expansion — {cid}", fontsize=13, fontweight="bold", color=ED_NAVY, y=0.96)
    png1 = fig_dir / f"cp-taxonomy-concordance-{slug}.png"
    svg1 = _save_dual(fig, png1)
    tsv1 = cp_out_dir / f"{slug}_class1_coord_verified.tsv"
    if not tsv1.is_file():
        pd.DataFrame(layers_info, columns=["layer", "genes", "novel_outside_baseline", "color"]).drop(columns=["color"]).to_csv(tsv1, sep="\t", index=False)
    panels.append({
        "id": f"cp-taxonomy-concordance-{slug}", "chapter": 4, "contrast_id": cid,
        "title": f"Cross-paradigm 4-class layer inventory & 1-bp coordinate concordance ({cnt['class1_concordant_pct']:.1f}%) | {cid}",
        "caption": "Quantifies net new genes unlocked beyond DEG + PS-DAS alongside exact 1-bp exon-coordinate concordance between rMATS and direction-calibrated DEXSeq.",
        "image": png1.relative_to(outdir).as_posix(), "svg": svg1.relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "Class I coordinate pairs (TSV)", "path": tsv1.relative_to(outdir).as_posix(), "unit": "exon_event", "contrast_id": cid, "policy": None}],
    })

    # 2. Chapter 04: SUPPA2 AF/AL + Strand-Aware 5'->3' DEXSeq Exon Gradient
    df_sup = c_res["_df_sup"]
    pos = c_res["positional"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 5.8), dpi=240)
    fig.subplots_adjust(wspace=0.28, top=0.83, bottom=0.15, left=0.07, right=0.97)
    x = np.arange(len(df_sup))
    w = 0.38
    b1 = ax1.bar(x - w / 2, df_sup["nom_n"], width=w, color=[ED_AMBER if et in ("AF", "AL") else ED_NAVY for et in df_sup["etype"]], alpha=0.88, label="Nominal (p < 0.05 & |ΔΨ| ≥ 0.10)")
    b2 = ax1.bar(x + w / 2, df_sup["fdr_n"], width=w, color=[ED_CORAL if et in ("AF", "AL") else ED_TEAL for et in df_sup["etype"]], alpha=0.92, label="BH-FDR < 0.05 & |ΔΨ| ≥ 0.10")
    max_s_h = max(10.0, float(df_sup["nom_n"].max()))
    for bar in b1:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + max_s_h * 0.015, f"{int(h)}", ha="center", va="bottom", fontsize=8.3, color=ED_NAVY)
    for bar in b2:
        h = bar.get_height()
        if h > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2, h + max_s_h * 0.015, f"{int(h)}", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=ED_CORAL)
    ax1.axvspan(4.5, 6.5, color="#FEF3C7", alpha=0.45, zorder=0)
    fdr_pct = (cnt["suppa_af_al_fdr"] / max(1, cnt["suppa_fdr_total"]) * 100.0) if cnt["suppa_fdr_total"] > 0 else 0.0
    ax1.text(2.65, max_s_h * 0.97, f"rMATS-Blind Boundary Zone\nAF+AL BH-FDR<0.05: {cnt['suppa_af_al_fdr']} ({cnt['suppa_af_al_genes']} genes)\n{cnt['suppa_af_al_fdr']}/{cnt['suppa_fdr_total']} ({fdr_pct:.0f}%) of ALL FDR<0.05 · +{cnt['suppa_af_al_novel_genes']} new\n(Nominal p<0.05: {cnt['suppa_af_al_nom']} events)", ha="center", va="top", fontsize=8.0, fontweight="bold", color="#92400E", bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#F59E0B", alpha=0.95))
    ax1.set_xticks(x)
    ax1.set_xticklabels(["SE", "A3", "A5", "MX", "RI", "AF ★\n(Alt First)", "AL ★\n(Alt Last)"], fontsize=8.9)
    ax1.set_ylim(0, max_s_h * 1.38)
    ax1.set_ylabel("Significant SUPPA2 Local Events", fontsize=9.5)
    ax1.set_title("A. SUPPA2 7-Class Splicing & AF/AL Dominance", loc="left", pad=10)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.1)

    dec_x = np.arange(1, 11)
    up_oe = np.asarray(pos["up_oe"], dtype=float)
    dn_oe = np.asarray(pos["dn_oe"], dtype=float)
    ax2.axhline(1.0, color="#94A3B8", ls="--", lw=1.2, label="Uniform Background (1.0x)")
    ax2.plot(dec_x, up_oe, marker="o", lw=2.3, ms=6.5, color=ED_TEAL, label=f"Higher Usage (log2FC ≥ +0.5, n={pos['up_n']:,})")
    ax2.plot(dec_x, dn_oe, marker="s", lw=2.3, ms=6.5, color=ED_CORAL, label=f"Lower Usage (log2FC ≤ −0.5, n={pos['dn_n']:,})")
    ax2.fill_between(dec_x, 1.0, up_oe, where=up_oe >= 1.0, color=ED_TEAL, alpha=0.14)
    ax2.fill_between(dec_x, 1.0, dn_oe, where=dn_oe >= 1.0, color=ED_CORAL, alpha=0.14)
    ax2.set_xticks(dec_x)
    ax2.set_xticklabels(["5' TSS\n0-10%", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "3' PAS\n90-100%"], fontsize=8.2)
    ax2.set_ylabel("Observed / Expected Density (Fold vs Bg)", fontsize=9.5)
    ax2.set_ylim(0, max(1.6, float(max(up_oe.max(), dn_oe.max())) * 1.28))
    ax2.set_title("B. Strand-Aware 5' → 3' DEXSeq Exon Positional Gradient", loc="left", pad=10)
    p_txt = (f"UP vs Bg χ² p = {pos['p_up_vs_bg']:.2e}\n" if pos["p_up_vs_bg"] is not None else "") + \
            (f"DOWN vs Bg χ² p = {pos['p_dn_vs_bg']:.2e}\n" if pos["p_dn_vs_bg"] is not None else "") + \
            (f"UP vs DOWN asym p = {pos['p_up_vs_dn']:.2e}" if pos["p_up_vs_dn"] is not None else "Insufficient counts for χ²")
    ax2.text(0.98, 0.96, p_txt.strip(), transform=ax2.transAxes, va="top", ha="right", fontsize=8.1, bbox=dict(boxstyle="round,pad=0.36", facecolor="white", edgecolor="#CBD5E1", alpha=0.95))
    ax2.legend(loc="upper left", frameon=True, fontsize=8.0)
    fig.suptitle(f"Boundary Reprogramming: Alternative First/Last Exons & 5'→3' Positional Exon Gradient — {cid}", fontsize=13, fontweight="bold", color=ED_NAVY, y=0.96)
    png2 = fig_dir / f"cp-afal-positional-{slug}.png"
    svg2 = _save_dual(fig, png2)
    tsv2 = cp_out_dir / f"{slug}_class2_positional_gradient.tsv"
    if not tsv2.is_file():
        pd.DataFrame({
            "decile": pos["deciles"],
            "up_obs_over_exp": pos["up_oe"],
            "dn_obs_over_exp": pos["dn_oe"],
            "up_n": pos["up_n"],
            "dn_n": pos["dn_n"],
        }).to_csv(tsv2, sep="\t", index=False)
    panels.append({
        "id": f"cp-afal-positional-{slug}", "chapter": 4, "contrast_id": cid,
        "title": f"SUPPA2 AF/AL promoter/polyA switches & 5'→3' DEXSeq exon positional gradient | {cid}",
        "caption": "Captures rMATS-blind Alternative First/Last exon switches alongside strand-normalized 5'->3' exon usage polarization.",
        "image": png2.relative_to(outdir).as_posix(), "svg": svg2.relative_to(outdir).as_posix(),
        "data_links": [{"role": "plot_data", "label": "5'->3' positional gradient (TSV)", "path": tsv2.relative_to(outdir).as_posix(), "unit": "decile", "contrast_id": cid, "policy": None}],
    })

    # 3. Chapter 07: Class III Isoform Switch Fates & Hidden Isoform-Only Genes
    df_sw = c_res["_df_sw"]
    if not df_sw.empty:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 5.9), dpi=240)
        fig.subplots_adjust(wspace=0.30, top=0.83, bottom=0.15, left=0.23, right=0.96)
        fate_counts = df_sw["fate"].value_counts()
        fate_order = fate_counts.index.tolist()[::-1]
        fate_vals = [int(fate_counts[f]) for f in fate_order]
        fate_hidden = [int(df_sw[(df_sw["fate"] == f) & (df_sw["hidden_from_deg"])].shape[0]) for f in fate_order]
        fy = np.arange(len(fate_order))
        f_colors = [ED_CORAL if "-> NMD" in f else (ED_TEAL if "NMD -> Coding" in f else (ED_AMBER if "Retained Intron" in f else (ED_NAVY if "ORF Remodeling" in f else (ED_CERULEAN if "UTR Switch" in f else ED_VIOLET)))) for f in fate_order]
        ax1.barh(fy, fate_vals, height=0.62, color=f_colors, alpha=0.88)
        ax1.barh(fy, fate_hidden, height=0.62, color="none", edgecolor="white", hatch="///", lw=0.8)
        max_fv = max(1, max(fate_vals))
        for i, (tot, hid) in enumerate(zip(fate_vals, fate_hidden)):
            ax1.text(tot + max_fv * 0.02, fy[i], f"{tot} ({hid} hidden DEG)", va="center", ha="left", fontsize=8.2, color=ED_NAVY)
        ax1.set_yticks(fy)
        ax1.set_yticklabels([f.replace(" (", "\n(") for f in fate_order], fontsize=8.3)
        ax1.set_xlim(0, max_fv * 1.44)
        ax1.set_xlabel(f"stageR Confirmed DTU Genes (gene_q < 0.05, n = {len(df_sw):,})", fontsize=9.5)
        ax1.set_title("A. Full-Length Isoform Switch Consequences (T_down → T_up)", loc="left", pad=10)
        hp2 = mpatches.Patch(facecolor="#CBD5E1", edgecolor="white", hatch="///", label="Hidden from DESeq2 DEG (FDR ≥ 0.05)")
        ax1.legend(handles=[hp2], loc="lower right", frameon=True, fontsize=8.0)

        sc3 = df_sw.dropna(subset=["deg_log2fc"]).copy()
        if not sc3.empty:
            sc3["abs_lfc"] = sc3["deg_log2fc"].abs()
            hid_m = sc3["hidden_from_deg"] & (sc3["max_abs_dif"] >= 0.10)
            both_m = (~sc3["hidden_from_deg"]) & (sc3["max_abs_dif"] >= 0.10)
            mild_m = sc3["max_abs_dif"] < 0.10
            ymax3 = max(0.25, float(sc3["max_abs_dif"].max()) * 1.32)
            ax2.axvline(1.0, color="#94A3B8", ls="--", lw=1)
            ax2.axhline(0.10, color="#94A3B8", ls="--", lw=1)
            ax2.axvspan(0, 1.0, ymin=min(1.0, 0.10 / ymax3), ymax=1.0, color="#FEF2F2", alpha=0.5, zorder=0)
            ax2.scatter(sc3.loc[mild_m, "abs_lfc"], sc3.loc[mild_m, "max_abs_dif"], s=18, c="#CBD5E1", alpha=0.65, label=f"Moderate |ΔIF|<0.10 (n={int(mild_m.sum())})", zorder=2)
            ax2.scatter(sc3.loc[both_m, "abs_lfc"], sc3.loc[both_m, "max_abs_dif"], s=28, c=ED_NAVY, alpha=0.78, label=f"DEG + Strong Switch (n={int(both_m.sum())})", zorder=3)
            ax2.scatter(sc3.loc[hid_m, "abs_lfc"], sc3.loc[hid_m, "max_abs_dif"], s=34, c=ED_CORAL, alpha=0.85, edgecolors="white", lw=0.5, label=f"Hidden Isoform-Only (n={int(hid_m.sum())})", zorder=4)
            picked_hid = []
            for _, r in sc3[hid_m].sort_values("max_abs_dif", ascending=False).iterrows():
                rx, ry = float(r["abs_lfc"]), float(r["max_abs_dif"])
                if any(abs(rx - px) <= 0.30 and abs(ry - py) <= 0.085 for px, py in picked_hid):
                    continue
                picked_hid.append((rx, ry))
                ox, oy = [(14, -12), (44, 14), (86, 2)][len(picked_hid) - 1]
                sf = "→NMD" if "-> NMD" in r["fate"] else ("→RI" if "Retained" in r["fate"] else ("ORFΔ" if "ORF" in r["fate"] else "Switch"))
                g_star = "***" if r["gene_q"] < 0.001 else ("**" if r["gene_q"] < 0.01 else "*")
                ax2.annotate(f"{r['gene_symbol']} ({sf} {g_star})", xy=(rx, ry), xytext=(ox, oy), textcoords="offset points", fontsize=7.9, fontweight="bold", color="#7F1D1D", arrowprops=dict(arrowstyle="-", color="#DC2626", lw=0.8))
                if len(picked_hid) >= 3:
                    break
            ax2.set_xlim(-0.05, min(6.5, max(1.5, float(sc3["abs_lfc"].max()) * 1.1)))
            ax2.set_ylim(0, ymax3)
            ax2.text(0.03, 0.96, "Gate: stageR gene_q < 0.05\nCoral: DESeq2 FDR ≥ 0.05", transform=ax2.transAxes, va="top", ha="left", fontsize=7.8, bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#CBD5E1", alpha=0.95))
            ax2.legend(loc="upper right", frameon=True, fontsize=7.9)
        ax2.set_xlabel("Gene-Level DESeq2 |log2FC|", fontsize=9.5)
        ax2.set_ylabel("Max Isoform Fraction Shift |ΔIF|", fontsize=9.5)
        ax2.set_title("B. 'Hidden' Switches Masked at Gene-Level", loc="left", pad=10)
        fig.suptitle(f"Full-Length Isoform Fate Remodeling & Gene-Level Masking — {cid}", fontsize=13, fontweight="bold", color=ED_NAVY, y=0.96)
        png3 = fig_dir / f"cp-isoform-fates-{slug}.png"
        svg3 = _save_dual(fig, png3)
        tsv3 = cp_out_dir / f"{slug}_class3_isoform_switches.tsv"
        panels.append({
            "id": f"cp-isoform-fates-{slug}", "chapter": 7, "contrast_id": cid,
            "title": f"Full-length isoform switch fates & hidden DEG-masked switches ({len(df_sw):,} pairs) | {cid}",
            "caption": "Annotates stageR T_down -> T_up switches via GENCODE CDS/UTR/biotype boundaries and highlights functional isoform switches masked at gene-sum expression.",
            "image": png3.relative_to(outdir).as_posix(), "svg": svg3.relative_to(outdir).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Isoform switch pairs (TSV)", "path": tsv3.relative_to(outdir).as_posix(), "unit": "gene", "contrast_id": cid, "policy": None}],
        })


def _render_ch8_cross_paradigm_explorer(
    cp_summary_data: dict[str, Any],
    standalone_path: Path,
    standalone_rel: str,
) -> tuple[str, str]:
    """Render Katana-PRISM visual-first Cross-Paradigm Explorer HTML (offline, zero CDN)."""
    safe_json = json.dumps(_clean_json(cp_summary_data), allow_nan=False).replace("</", "<\\/")
    safe_rel = html_lib.escape(standalone_rel)
    inline_tsv_prefix = (standalone_rel.rsplit("/", 1)[0] + "/") if "/" in standalone_rel else ""
    template_body = r"""
<style>
  .interactive-workbench, #interactive-cross-paradigm-isoform-explorer { grid-column:1 / -1 !important; width:100%; min-width:0; box-sizing:border-box; }
  .kcp-shell { font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; color:#051c2c; background:#fff; border:1px solid #cbd5e1; border-radius:6px; padding:16px 18px; margin:8px 0 16px; width:100%; box-sizing:border-box; }
  .kcp-head { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px; border-bottom:2px solid #051c2c; padding-bottom:10px; margin-bottom:12px; }
  .kcp-eyebrow { font-size:10px; font-weight:700; letter-spacing:0.6px; text-transform:uppercase; color:#0d9488; display:block; }
  .kcp-tabs { display:grid; grid-template-columns:repeat(auto-fit,minmax(175px,1fr)); gap:8px; margin-bottom:8px; }
  @media (min-width:900px) { .kcp-tabs { grid-template-columns:repeat(4, minmax(0, 1fr)); } }
  .kcp-tab { text-align:left; background:#f8fafc; border:1px solid #cbd5e1; border-top:3px solid #94a3b8; border-radius:5px; padding:9px 12px; cursor:pointer; transition:all 0.12s ease; min-width:0; }
  .kcp-tab:hover { background:#f1f5f9; border-color:#051c2c; }
  .kcp-tab.active { background:#f0fdfa; border-color:#0d9488; border-top-color:#0d9488; box-shadow:0 2px 6px rgba(13,148,136,0.12); }
  .kcp-tab .kcp-tname { font-size:12.5px; font-weight:700; color:#051c2c; display:block; }
  .kcp-tab .kcp-tval { font-size:15px; font-weight:800; color:#051c2c; display:block; margin:2px 0; }
  .kcp-tab .kcp-tgate { font-size:10.5px; color:#475569; font-weight:600; display:block; }
  .kcp-guide { font-size:11.8px; color:#334155; background:#f8fafc; border:1px solid #e2e8f0; border-left:3px solid #051c2c; border-radius:4px; padding:6px 11px; margin-bottom:10px; line-height:1.4; }
  .kcp-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; background:#f1f5f9; padding:8px 12px; border-radius:5px; margin-bottom:10px; }
  .kcp-bar select, .kcp-bar input { padding:5px 8px; border:1px solid #cbd5e1; border-radius:4px; background:#fff; font-size:12px; color:#051c2c; max-width:255px; }
  .kcp-story { border:1px solid #cbd5e1; border-radius:6px; background:#fcfdff; padding:12px 14px; margin-bottom:12px; }
  .kcp-sig-strip { background:#f8fafc; border:1px solid #e2e8f0; border-left:4px solid #0d9488; border-radius:4px; padding:10px 12px; margin-bottom:12px; }
  .kcp-sig-strip.warn { border-left-color:#dc2626; background:#fef2f2; }
  .kcp-sig-top { display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px; margin-bottom:8px; }
  .kcp-sig-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:8px; }
  .kcp-mcard { background:#fff; border:1px solid #e2e8f0; border-radius:4px; padding:6px 9px; min-width:0; }
  .kcp-mlbl { font-size:10.2px; font-weight:700; text-transform:uppercase; color:#64748b; letter-spacing:0.4px; display:block; }
  .kcp-mval { font-size:12.5px; font-weight:700; color:#051c2c; display:block; margin:2px 0; }
  .kcp-split { display:grid; grid-template-columns:minmax(0, 1fr) minmax(0, 1.32fr); gap:12px; margin-bottom:10px; }
  @media (max-width:960px) { .kcp-split { grid-template-columns:1fr; } }
  .kcp-box { border:1px solid #e2e8f0; border-radius:5px; padding:10px 12px; background:#fff; overflow-x:auto; min-width:0; }
  .kcp-box svg { width:100% !important; max-width:none !important; height:auto; display:block; }
  .kcp-box h4 { margin:0 0 8px; font-size:12.2px; font-weight:700; color:#051c2c; border-bottom:1px solid #f1f5f9; padding-bottom:4px; }
  .kcp-tbl-wrap { max-height:360px; overflow:auto; border:1px solid #e2e8f0; border-radius:5px; width:100%; }
  .kcp-tbl { width:100%; border-collapse:collapse; font-size:11.2px; }
  .kcp-tbl th { position:sticky; top:0; background:#f1f5f9; color:#051c2c; padding:5px 6px; text-align:left; border-bottom:1px solid #cbd5e1; white-space:nowrap; z-index:2; }
  .kcp-tbl td { padding:4px 6px; border-bottom:1px solid #f1f5f9; white-space:nowrap; }
  .kcp-tbl tbody tr.kcp-row { cursor:pointer; }
  .kcp-tbl tbody tr.kcp-row:hover { background:#f0fdfa; }
  .kcp-tbl tbody tr.kcp-row.selected { background:#ccfbf1; font-weight:600; }
  .kcp-gl-wrap { margin-top:10px; border:1px solid #e2e8f0; border-radius:5px; background:#f8fafc; padding:8px 12px; }
  .kcp-gl-wrap summary { cursor:pointer; font-size:12px; font-weight:700; color:#051c2c; }
  .kcp-gl-chips { display:flex; flex-wrap:wrap; gap:5px; margin-top:8px; max-height:190px; overflow-y:auto; padding-right:4px; }
  .kcp-gchip { font-size:11px; padding:2px 8px; border-radius:999px; border:1px solid #cbd5e1; background:#fff; color:#334155; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .kcp-gchip.clickable { border-color:#0d9488; color:#051c2c; font-weight:600; cursor:pointer; }
  .kcp-gchip.clickable:hover { background:#f0fdfa; }
  .kcp-gchip.active { background:#051c2c; color:#fff; border-color:#051c2c; }
  .kcp-gchip.extra { background:#f1f5f9; color:#64748b; border-color:#e2e8f0; }
  .kcp-pill { display:inline-block; padding:2px 7px; border-radius:999px; font-size:10.5px; font-weight:700; }
  .kcp-p-teal { background:#ccfbf1; color:#0f766e; }
  .kcp-p-cor { background:#fee2e2; color:#991b1b; }
  .kcp-p-amb { background:#fef3c7; color:#92400e; }
  .kcp-p-vio { background:#ede9fe; color:#5b21b6; }
  .kcp-p-sl { background:#e2e8f0; color:#334155; }
  .kcp-sbadge { display:inline-block; padding:1px 6px; border-radius:3px; font-size:10.5px; font-weight:700; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  .kcp-sig-3 { background:#0f766e; color:#fff; }
  .kcp-sig-2 { background:#0d9488; color:#fff; }
  .kcp-sig-1 { background:#e0f2fe; color:#0369a1; border:1px solid #bae6fd; }
  .kcp-sig-ns { background:#f1f5f9; color:#64748b; border:1px solid #cbd5e1; }
  .kcp-btn { border:1px solid #cbd5e1; background:#fff; color:#051c2c; padding:4px 9px; border-radius:4px; font-size:11.2px; font-weight:600; cursor:pointer; text-decoration:none; display:inline-block; }
  .kcp-btn:hover { background:#f1f5f9; border-color:#051c2c; }
  .kcp-pt { cursor:pointer; transition:r 0.1s ease; }
</style>
<section class="exhibit wide interactive-workbench" id="interactive-cross-paradigm-isoform-explorer" style="grid-column:1 / -1;width:100%;min-width:0;border-top:none;padding:0;">
<div class="kcp-shell" id="kcp-root">
  <div class="kcp-head">
    <div>
      <span class="kcp-eyebrow">Cross-Paradigm Visual Explorer · Multi-Tool Statistical Synthesis</span>
      <h3 style="margin:2px 0 0;font-size:17px;color:#051c2c;">Isoform Switches, Dual-Verified Exons &amp; rMATS Blind Spots</h3>
    </div>
    <div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
      <label style="font-size:12px;font-weight:700;">Contrast: <select id="kcp-cid-sel"></select></label>
      <span id="kcp-bug-note" style="font-size:11.5px;"></span>
      <a href="__STANDALONE_REL__" target="_blank" style="font-size:11.8px;font-weight:700;color:#2251ff;text-decoration:none;border:1px solid #cbd5e1;padding:4px 10px;border-radius:4px;">Open standalone ↗</a>
    </div>
  </div>
  <div class="kcp-tabs" id="kcp-cards"></div>
  <div class="kcp-guide" id="kcp-guide"></div>
  <div class="kcp-story" id="kcp-story-card">
    <div id="kcp-verdict" class="kcp-sig-strip"></div>
    <div class="kcp-split">
      <div class="kcp-box">
        <h4 id="kcp-left-title">A. Quantitative Profile</h4>
        <div id="kcp-left-svg"></div>
      </div>
      <div class="kcp-box">
        <h4 id="kcp-right-title">B. Genomic Exon/CDS &amp; DEXSeq Track</h4>
        <div id="kcp-svg-track"></div>
        <div id="kcp-exon-pills" style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px;align-items:center;"></div>
      </div>
    </div>
    <div id="kcp-st-box"></div>
    <details id="kcp-tx-wrap" style="margin-top:6px;font-size:11.8px;">
      <summary style="cursor:pointer;font-weight:700;color:#475569;">Per-Transcript Raw Counts &amp; Exact Q-Values</summary>
      <div id="kcp-tx-tbl" style="margin-top:6px;overflow-x:auto;"></div>
    </details>
  </div>
  <div class="kcp-bar" style="justify-content:space-between;">
    <div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center;">
      <span id="kcp-tab-count" style="font-size:11.8px;font-weight:700;color:#051c2c;"></span>
      <input type="text" id="kcp-search" placeholder="Filter gene, tx, exon, fate..." style="min-width:170px;" />
      <label style="font-size:11.5px;cursor:pointer;"><input type="checkbox" id="kcp-novel-only" /> Missed by DEG + PS-DAS</label>
      <label id="kcp-strict-wrap" style="font-size:11.5px;cursor:pointer;"><input type="checkbox" id="kcp-strict-sig" /> <span id="kcp-strict-lbl">Confirmed both tx_q &lt; 0.05</span></label>
      <label style="font-size:11.5px;font-weight:600;">Gene: <select id="kcp-gene-sel"></select></label>
    </div>
    <div style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;">
      <button type="button" id="kcp-copy-genes" class="kcp-btn">Copy Gene List</button>
      <button type="button" id="kcp-dl-genes" class="kcp-btn">Gene List (.txt)</button>
      <button type="button" id="kcp-export-tsv" class="kcp-btn">Table (.tsv)</button>
      <a id="kcp-full-tsv" class="kcp-btn" href="#" target="_blank">Complete TSV ↗</a>
    </div>
  </div>
  <div class="kcp-tbl-wrap" id="kcp-table-box"></div>
  <details class="kcp-gl-wrap" id="kcp-genelist-wrap" open>
    <summary id="kcp-genelist-sum">Complete Gene List for Active Tab</summary>
    <div id="kcp-genelist-box" class="kcp-gl-chips"></div>
  </details>
</div>
<script>
(function() {
  const CP = __SAFE_JSON__;
  const cids = Object.keys(CP.contrasts || {});
  if (!cids.length) return;
  let activeCid = cids[0];
  let activeTab = 'class3_switches';
  let activeRowKey = '';
  let focusObj = null;

  const esc = (s) => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const fmtQ = (q) => {
    if (q === null || q === undefined || Number.isNaN(Number(q))) return 'NA';
    const v = Number(q);
    if (v <= 0 || v < 1e-16) return '< 1e-16';
    if (v < 0.001) return v.toExponential(1);
    return v.toFixed(3);
  };
  const sigStars = (q) => {
    if (q === null || q === undefined || Number.isNaN(Number(q))) return 'NA';
    const v = Number(q);
    if (v < 0.001) return '***';
    if (v < 0.01) return '**';
    if (v < 0.05) return '*';
    return 'n.s.';
  };
  const sigCls = (q) => {
    if (q === null || q === undefined || Number.isNaN(Number(q))) return 'kcp-sig-ns';
    const v = Number(q);
    if (v < 0.001) return 'kcp-sig-3';
    if (v < 0.01) return 'kcp-sig-2';
    if (v < 0.05) return 'kcp-sig-1';
    return 'kcp-sig-ns';
  };
  const sigBadge = (q) => `<span class="kcp-sbadge ${sigCls(q)}">${fmtQ(q)} (${sigStars(q)})</span>`;
  const tierInfo = (t) => {
    if (t === 'both_tx_sig') return ['Both tx sig', 'kcp-p-teal'];
    if (t === 'one_tx_sig') return ['1 tx sig', 'kcp-p-amb'];
    return ['Gene-level only', 'kcp-p-sl'];
  };
  const fmtS = (v, d=2) => (v == null || Number.isNaN(Number(v))) ? 'NA' : `${Number(v) >= 0 ? '+' : ''}${Number(v).toFixed(d)}`;
  const shortBt = (b) => String(b || 'other').replace('nonsense_mediated_decay', 'NMD').replace('protein_coding_CDS_not_defined', 'No-CDS').replace('retained_intron', 'RI').replace('protein_coding', 'Coding');

  const root = document.getElementById('kcp-root');
  const cidSel = document.getElementById('kcp-cid-sel');
  const geneSel = document.getElementById('kcp-gene-sel');

  cids.forEach(c => {
    const o = document.createElement('option');
    o.value = c;
    o.textContent = `${c} (${CP.contrasts[c].counts.dtu_hidden_from_deg} hidden DEG · ${CP.contrasts[c].counts.class1_concordant_pct}% verified)`;
    cidSel.appendChild(o);
  });

  window._kcpCopyCoord = (btn, txt) => {
    if (!txt) return;
    const orig = btn.textContent;
    const done = () => { btn.textContent = 'Copied'; setTimeout(() => { btn.textContent = orig; }, 1400); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(done).catch(done);
    } else {
      const ta = document.createElement('textarea');
      ta.value = txt; document.body.appendChild(ta); ta.select();
      try { document.execCommand('copy'); } catch (e) {}
      document.body.removeChild(ta); done();
    }
  };

  function parseCoord(cstr) {
    const m = String(cstr || '').match(/^([^:]+):(\d+)-(\d+)\(([+-])\)/);
    if (!m) return null;
    return { chr: m[1], start: parseInt(m[2], 10), end: parseInt(m[3], 10), strand: m[4] };
  }

  function findIg(sym, base) {
    const C = CP.contrasts[activeCid];
    return (C.interactive_genes || []).find(g => (sym && g.gene_symbol === sym) || (base && g.gene_base === base)) || null;
  }
  function findC1(sym, base) {
    const C = CP.contrasts[activeCid];
    return ((C.tables_preview && C.tables_preview.class1) || []).find(r => (sym && r.gene_symbol === sym) || (base && r.gene_base === base)) || null;
  }
  function findC3(sym, base) {
    const C = CP.contrasts[activeCid];
    return ((C.tables_preview && C.tables_preview.class3_switches) || []).find(r => (sym && r.gene_symbol === sym) || (base && r.gene_base === base)) || null;
  }

  function getGeneBins(geneBase, fallbackCoord, fallbackFid, fallbackLfc, fallbackPadj) {
    const C = CP.contrasts[activeCid];
    const gbObj = C.gene_dex_bins && C.gene_dex_bins[geneBase];
    if (gbObj && gbObj.bins && gbObj.bins.length) return gbObj;
    const ig = findIg(null, geneBase);
    if (ig && ig.dex_bins && ig.dex_bins.length) {
      return { chr: ig.chr, strand: ig.strand, bins: ig.dex_bins };
    }
    const pc = parseCoord(fallbackCoord);
    if (pc) {
      return { chr: pc.chr, strand: pc.strand, bins: [{ fid: fallbackFid || 'Exon', start: pc.start, end: pc.end, lfc: Number(fallbackLfc || 0), padj: Number(fallbackPadj ?? 1), pos_5to3: 0.5, exonBaseMean: 0 }] };
    }
    return { chr: '', strand: '+', bins: [] };
  }

  function filteredRowsAndCols() {
    const C = CP.contrasts[activeCid];
    const q = (document.getElementById('kcp-search').value || '').toLowerCase().trim();
    const novel = document.getElementById('kcp-novel-only').checked;
    const strictSig = document.getElementById('kcp-strict-sig').checked;
    const prev = C.tables_preview || {};
    let rows = prev[activeTab] || [];
    let cols = [];
    if (activeTab === 'class1') {
      if (novel) rows = rows.filter(r => !r.stable_significant);
      cols = [['gene_symbol','Gene'],['event_type','Class'],['dexseq_exon','Exon Bin'],['overlap_bp','Overlap(bp)'],['rmats_dpsi','rMATS ΔΨ'],['rmats_fdr','rMATS FDR'],['dexseq_lfc','DEXSeq lfc'],['dexseq_padj','DEXSeq padj'],['concordant_fixed','Agree']];
    } else if (activeTab === 'class2_af_al') {
      if (novel) rows = rows.filter(r => !r.in_ps_das && !r.in_deg);
      if (strictSig) rows = rows.filter(r => r.sig_fdr);
      cols = [['gene_symbol','Gene'],['etype','Type'],['dpsi','SUPPA2 ΔΨ'],['pvalue','Nominal P'],['bh_fdr','BH-FDR'],['in_ps_das','In PS-DAS'],['in_deg','In DEG'],['event_id','Event Coordinates']];
    } else if (activeTab === 'class3_switches') {
      if (novel) rows = rows.filter(r => r.hidden_from_deg);
      if (strictSig) rows = rows.filter(r => r.tx_confirm_tier === 'both_tx_sig');
      cols = [['gene_symbol','Gene'],['fate','Switch Fate'],['tx_confirm_tier','Tx Confirm'],['gene_q','stageR gene_q'],['tx_up_dif','T_up ΔIF'],['tx_up_q','T_up tx_q'],['tx_dn_dif','T_dn ΔIF'],['tx_dn_q','T_dn tx_q'],['delta_aa','Δaa'],['deg_log2fc','DESeq2 lfc'],['deg_fdr','DESeq2 FDR']];
    } else {
      if (novel) rows = rows.filter(r => !r.gene_in_deg && !r.gene_in_rmats);
      cols = [['gene_symbol','Gene'],['feature_id','Exon Bin'],['region_bin','5′→3′ Region'],['lfc','DEXSeq lfc'],['padj','DEXSeq padj'],['exonBaseMean','Mean Counts'],['in_tested_rmats','Tested rMATS'],['gene_in_deg','In DEG']];
    }
    if (q) rows = rows.filter(r => Object.values(r).some(v => String(v).toLowerCase().includes(q)));
    return { rows, cols };
  }

  function updateStrictFilterUI() {
    const w = document.getElementById('kcp-strict-wrap');
    const lbl = document.getElementById('kcp-strict-lbl');
    if (activeTab === 'class3_switches') {
      w.style.display = 'inline-block';
      lbl.textContent = 'Confirmed both tx_q < 0.05';
    } else if (activeTab === 'class2_af_al') {
      w.style.display = 'inline-block';
      lbl.textContent = 'Strict BH-FDR < 0.05 only';
    } else {
      w.style.display = 'none';
    }
  }

  function defaultFocusForTab() {
    const C = CP.contrasts[activeCid];
    const { rows } = filteredRowsAndCols();
    const prev = C.tables_preview || {};
    const useRows = rows.length ? rows : (prev[activeTab] || []);
    if (activeTab === 'class3_switches') {
      const r0 = useRows.find(r => r.hidden_from_deg && r.tx_confirm_tier === 'both_tx_sig') || useRows[0];
      if (r0) return { kind: 'c3', row: r0, ig: findIg(r0.gene_symbol, r0.gene_base), ridx: useRows.indexOf(r0) };
      const igs = C.interactive_genes || [];
      return igs[0] ? { kind: 'ig', ig: igs[0], row: igs[0], ridx: 0 } : null;
    }
    const kind = activeTab === 'class1' ? 'c1' : (activeTab === 'class2_af_al' ? 'c2' : 'c4');
    const r0 = useRows[0];
    return r0 ? { kind, row: r0, ig: findIg(r0.gene_symbol, r0.gene_base), ridx: 0 } : null;
  }

  function renderTabs() {
    const C = CP.contrasts[activeCid];
    const cnt = C.counts;
    const defs = [
      ['class3_switches', '1. Isoform Switches', `${cnt.dtu_switch_resolved} switches · ${cnt.dtu_hidden_from_deg} missed DEG`, 'Gate: stageR gene_q < 0.05'],
      ['class1', '2. Dual-Verified Exons', `${cnt.class1_concordant}/${cnt.class1_events} concordant (${cnt.class1_concordant_pct}%)`, 'Gate: rMATS FDR<.05 ∩ DEXSeq padj<.05'],
      ['class2_af_al', '3. 5′ / 3′ Boundaries', `${cnt.suppa_af_al_fdr} BH-FDR (+${cnt.suppa_af_al_novel_genes} new)`, `Gate: BH-FDR < 0.05 · ${cnt.suppa_af_al_nom} nom shown`],
      ['class4_dexseq', '4. DEXSeq-Only Exons', `${cnt.class4_exons} exons (+${cnt.class4_novel_genes} genes)`, 'Gate: padj < 0.05 & |log2FC| ≥ 0.5'],
    ];
    document.getElementById('kcp-cards').innerHTML = defs.map(([id, name, val, gate]) =>
      `<button type="button" class="kcp-tab ${activeTab === id ? 'active' : ''}" data-tab="${id}">`
      + `<span class="kcp-tname">${name}</span><span class="kcp-tval">${val}</span><span class="kcp-tgate">${gate}</span></button>`
    ).join('');
    root.querySelectorAll('.kcp-tab').forEach(b => b.addEventListener('click', () => {
      activeTab = b.dataset.tab;
      activeRowKey = '';
      updateStrictFilterUI();
      focusObj = defaultFocusForTab();
      renderTabs();
      renderStory();
      renderTable();
    }));
  }

  function biotypeCat(bt) {
    const b = String(bt || '').toLowerCase();
    if (b.includes('nonsense_mediated_decay')) return 'nmd';
    if (b.includes('retained_intron')) return 'ri';
    if (b === 'protein_coding') return 'coding';
    if (b.includes('cds_not_defined') || b.includes('lncrna') || b.includes('processed')) return 'noncoding';
    return 'other';
  }

  function renderTab1LeftSvg(ig, r, C) {
    const txs = (ig && ig.txs && ig.txs.length) ? ig.txs : [
      { tx_id: r.tx_up, role: 'T_up', biotype: r.tx_up_biotype, aa_len: r.tx_up_aa || 0, dIF: r.tx_up_dif || 0, if_ctrl: Math.max(0.05, 0.5 - (r.tx_up_dif || 0)/2), if_treat: Math.min(0.95, 0.5 + (r.tx_up_dif || 0)/2), tx_q: r.tx_up_q },
      { tx_id: r.tx_dn, role: 'T_dn', biotype: r.tx_dn_biotype, aa_len: r.tx_dn_aa || 0, dIF: r.tx_dn_dif || 0, if_ctrl: Math.min(0.95, 0.5 - (r.tx_dn_dif || 0)/2), if_treat: Math.max(0.05, 0.5 + (r.tx_dn_dif || 0)/2), tx_q: r.tx_dn_q },
    ];
    const degFdr = ig ? ig.deg_fdr : r.deg_fdr;
    const degLfc = ig ? ig.deg_log2fc : r.deg_log2fc;
    const geneQ = ig ? ig.gene_q : r.gene_q;
    const hid = ig ? ig.hidden_from_deg : r.hidden_from_deg;
    const totCtrl = (ig && ig.gene_tpm_ctrl != null) ? Number(ig.gene_tpm_ctrl) : (r.gene_tpm_ctrl != null ? Number(r.gene_tpm_ctrl) : txs.reduce((s, t) => s + Number(t.tpm_ctrl || 0), 0));
    const totTreat = (ig && ig.gene_tpm_treat != null) ? Number(ig.gene_tpm_treat) : (r.gene_tpm_treat != null ? Number(r.gene_tpm_treat) : txs.reduce((s, t) => s + Number(t.tpm_treat || 0), 0));

    const cats = [
      ['coding', 'Coding', '#0d9488'],
      ['nmd', 'NMD', '#dc2626'],
      ['ri', 'Retained Intron', '#d97706'],
      ['noncoding', 'Non-Coding', '#7c3aed'],
      ['other', 'Other Isoforms', '#cbd5e1'],
    ];
    const sumCtrl = { coding:0, nmd:0, ri:0, noncoding:0, other:0 };
    const sumTreat = { coding:0, nmd:0, ri:0, noncoding:0, other:0 };
    let covCtrl = 0, covTreat = 0;
    txs.forEach(t => {
      const k = biotypeCat(t.biotype);
      sumCtrl[k] += Number(t.if_ctrl || 0); covCtrl += Number(t.if_ctrl || 0);
      sumTreat[k] += Number(t.if_treat || 0); covTreat += Number(t.if_treat || 0);
    });
    if (covCtrl < 1.0) sumCtrl.other += Math.max(0, 1.0 - covCtrl);
    if (covTreat < 1.0) sumTreat.other += Math.max(0, 1.0 - covTreat);

    const W = 540, padL = 104, barW = 350;
    const maxTpm = Math.max(1, totCtrl, totTreat);
    const wCtrl = (totCtrl / maxTpm) * 235, wTreat = (totTreat / maxTpm) * 235;
    const rowH = 26, dbTop = 226;
    const H = dbTop + txs.length * rowH + 14;

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:540px;height:auto;font-family:sans-serif;">`;
    // 1. Gene-level TPM bars
    svg += `<text x="4" y="15" font-size="10.6" font-weight="700" fill="#051c2c">1. Total Gene mRNA (similar bars = missed by DESeq2)</text>`;
    svg += `<text x="${W-6}" y="15" font-size="10.2" font-weight="700" text-anchor="end" fill="${hid ? '#b91c1c' : '#0f766e'}">DESeq2 FDR = ${esc(fmtQ(degFdr))} (${sigStars(degFdr)})</text>`;
    svg += `<text x="4" y="38" font-size="10.5" font-weight="700" fill="#334155">${esc(C.ctrl_label)}</text>`;
    svg += `<rect x="${padL}" y="26" width="${Math.max(4, wCtrl).toFixed(1)}" height="15" rx="2" fill="#64748b" opacity="0.85"/>`;
    svg += `<text x="${padL + wCtrl + 6}" y="38" font-size="10.5" font-weight="700" fill="#051c2c">${totCtrl.toFixed(1)} TPM</text>`;
    svg += `<text x="4" y="60" font-size="10.5" font-weight="700" fill="#334155">${esc(C.treat_label)}</text>`;
    svg += `<rect x="${padL}" y="48" width="${Math.max(4, wTreat).toFixed(1)}" height="15" rx="2" fill="${hid ? '#64748b' : '#051c2c'}" opacity="0.9"/>`;
    svg += `<text x="${padL + wTreat + 6}" y="60" font-size="10.5" font-weight="700" fill="#051c2c">${totTreat.toFixed(1)} TPM (lfc ${fmtS(degLfc, 2)})</text>`;

    // 2. 100% Stacked Isoform Composition
    svg += `<line x1="4" y1="74" x2="${W-4}" y2="74" stroke="#e2e8f0" stroke-width="1"/>`;
    svg += `<text x="4" y="92" font-size="10.6" font-weight="700" fill="#051c2c">2. Normal Protein vs Non-Functional mRNA Share (100%)</text>`;
    svg += `<text x="${W-6}" y="92" font-size="10.2" font-weight="700" text-anchor="end" fill="#0f766e">stageR gene_q = ${esc(fmtQ(geneQ))} (${sigStars(geneQ)})</text>`;
    [[C.ctrl_label, sumCtrl, 102], [C.treat_label, sumTreat, 134]].forEach(([lbl, sums, y]) => {
      svg += `<text x="4" y="${y+17}" font-size="10.5" font-weight="700" fill="#051c2c">${esc(lbl)}</text>`;
      let cx = padL;
      const totS = cats.reduce((a, [k]) => a + sums[k], 0) || 1.0;
      cats.forEach(([k, name, col]) => {
        const frac = sums[k] / totS, sw = frac * barW;
        if (sw > 0.8) {
          svg += `<rect x="${cx.toFixed(1)}" y="${y}" width="${sw.toFixed(1)}" height="23" fill="${col}" stroke="#fff" stroke-width="0.8"><title>${esc(name)}: ${(frac*100).toFixed(1)}%</title></rect>`;
          if (frac >= 0.095) {
            svg += `<text x="${(cx + sw/2).toFixed(1)}" y="${y+15}" font-size="10" font-weight="700" fill="${k==='other'?'#051c2c':'#fff'}" text-anchor="middle">${(frac*100).toFixed(0)}%</text>`;
          }
        }
        cx += sw;
      });
    });
    let lx = 4;
    cats.forEach(([k, name, col]) => {
      if (sumCtrl[k] + sumTreat[k] < 0.005) return;
      svg += `<rect x="${lx}" y="166" width="9" height="9" rx="2" fill="${col}"/><text x="${lx+12}" y="174" font-size="9.5" fill="#334155">${esc(name)}</text>`;
      lx += name.length * 5.6 + 22;
    });

    // 3. Per-Transcript Shift Dumbbell + tx_q
    svg += `<line x1="4" y1="186" x2="${W-4}" y2="186" stroke="#e2e8f0" stroke-width="1"/>`;
    svg += `<text x="4" y="203" font-size="10.6" font-weight="700" fill="#051c2c">3. Per-Isoform Share Shift (Control ○ → Treated ●) &amp; tx_q</text>`;
    const dx0 = 195, dw = 140;
    [0, 0.5, 1].forEach(p => {
      const gx = dx0 + p * dw;
      svg += `<line x1="${gx}" y1="210" x2="${gx}" y2="${H-6}" stroke="#e2e8f0" stroke-dasharray="2,2"/>`;
      svg += `<text x="${gx}" y="218" font-size="8.8" fill="#64748b" text-anchor="middle">${(p*100).toFixed(0)}%</text>`;
    });
    svg += `<text x="348" y="218" font-size="9" font-weight="700" fill="#64748b">ΔIF &amp; tx_q</text>`;
    txs.forEach((t, i) => {
      const cy = dbTop + i * rowH + 10;
      const x1 = dx0 + Math.max(0, Math.min(1, Number(t.if_ctrl || 0))) * dw;
      const x2 = dx0 + Math.max(0, Math.min(1, Number(t.if_treat || 0))) * dw;
      const up = Number(t.dIF || 0) >= 0;
      const col = Number(t.tx_q ?? 1) < 0.05 ? (up ? '#0d9488' : '#b83a4b') : '#64748b';
      const roleP = t.role === 'T_up' ? '▲ ' : (t.role === 'T_dn' ? '▼ ' : '');
      svg += `<text x="4" y="${cy+3.5}" font-size="9.5" font-weight="${t.role==='Other'?'400':'700'}" fill="${col}">${roleP}${esc(t.tx_id)} (${esc(shortBt(t.biotype))})</text>`;
      svg += `<line x1="${dx0}" y1="${cy}" x2="${dx0+dw}" y2="${cy}" stroke="#f1f5f9" stroke-width="2"/>`;
      svg += `<line x1="${x1.toFixed(1)}" y1="${cy}" x2="${x2.toFixed(1)}" y2="${cy}" stroke="${col}" stroke-width="2.5"/>`;
      svg += `<circle cx="${x1.toFixed(1)}" cy="${cy}" r="3.5" fill="#fff" stroke="${col}" stroke-width="1.6"/>`;
      svg += `<circle cx="${x2.toFixed(1)}" cy="${cy}" r="4.2" fill="${col}"/>`;
      const sigTxt = `q=${fmtQ(t.tx_q)} (${sigStars(t.tx_q)})`;
      svg += `<text x="348" y="${cy+3.5}" font-size="9.5" font-weight="700" fill="${col}">${fmtS(Number(t.dIF||0)*100,1)}% · ${esc(sigTxt)}</text>`;
    });
    svg += '</svg>';
    return svg;
  }

  function renderTab1RightSvg(ig, r) {
    if (!ig || !ig.txs || !ig.txs.length) {
      const W = 720, H = 220;
      const u = Number(r.tx_up_dif || 0) * 100, d = Number(r.tx_dn_dif || 0) * 100, mid = 340, maxL = 220;
      const wu = Math.min(maxL, Math.abs(u)/100*maxL), wd = Math.min(maxL, Math.abs(d)/100*maxL);
      return `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:740px;height:auto;font-family:sans-serif;">
        <text x="6" y="20" font-size="11.5" font-weight="700" fill="#051c2c">Competing Isoform Divergence (stageR gene_q = ${esc(fmtQ(r.gene_q))} ${sigStars(r.gene_q)})</text>
        <line x1="${mid}" y1="36" x2="${mid}" y2="145" stroke="#94a3b8" stroke-dasharray="3,2"/>
        <text x="6" y="68" font-size="11" font-weight="700" fill="#0d9488">▲ ${esc(r.tx_up)} (${r.tx_up_aa}aa)</text>
        <rect x="${mid}" y="53" width="${Math.max(5,wu)}" height="20" rx="2" fill="#0d9488"/>
        <text x="${mid+wu+8}" y="68" font-size="11" font-weight="700" fill="#0d9488">${fmtS(u,1)}% · tx_q=${esc(fmtQ(r.tx_up_q))} (${sigStars(r.tx_up_q)})</text>
        <text x="6" y="114" font-size="11" font-weight="700" fill="#b83a4b">▼ ${esc(r.tx_dn)} (${r.tx_dn_aa}aa)</text>
        <rect x="${mid-wd}" y="99" width="${Math.max(5,wd)}" height="20" rx="2" fill="#b83a4b"/>
        <text x="${mid-wd-8}" y="114" font-size="11" font-weight="700" text-anchor="end" fill="#b83a4b">${fmtS(d,1)}% · tx_q=${esc(fmtQ(r.tx_dn_q))} (${sigStars(r.tx_dn_q)})</text>
      </svg>`;
    }
    const W = 740, padL = 185, padR = 24, trackW = W - padL - padR;
    const minC = ig.span[0], maxC = Math.max(ig.span[1], minC + 1);
    const sx = (pos) => padL + (ig.strand === '-' ? (1 - (pos - minC) / (maxC - minC)) : ((pos - minC) / (maxC - minC))) * trackW;
    const rowH = 30, headerH = 22, dexH = 82;
    const H = headerH + ig.txs.length * rowH + 22 + dexH;
    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:740px;height:auto;font-family:sans-serif;">`;
    svg += `<text x="${padL}" y="14" font-size="10.5" font-weight="700" fill="#475569">5′ (${esc(ig.strand)}) ──────────────────────────▶ 3′ (${((maxC-minC)/1000).toFixed(1)} kb · solid box = CDS)</text>`;
    ig.txs.forEach((tx, i) => {
      const cy = headerH + i * rowH + 15;
      const col = tx.role === 'T_up' ? '#0d9488' : (tx.role === 'T_dn' ? '#b83a4b' : '#64748b');
      const tag = tx.role === 'T_up' ? '▲ ' : (tx.role === 'T_dn' ? '▼ ' : '• ');
      svg += `<text x="4" y="${cy+4}" font-size="10.2" font-weight="700" fill="${col}">${tag}${esc(tx.tx_id)} (${sigStars(tx.tx_q)})</text>`;
      svg += `<line x1="${padL}" y1="${cy}" x2="${W-padR}" y2="${cy}" stroke="#cbd5e1" stroke-width="1.2"/>`;
      tx.exons.forEach(([es, ee]) => {
        const x1 = sx(es), x2 = sx(ee);
        svg += `<rect x="${Math.min(x1,x2).toFixed(1)}" y="${cy-4}" width="${Math.max(2,Math.abs(x2-x1)).toFixed(1)}" height="8" fill="${col}" opacity="0.42"/>`;
        if (tx.cds_span) {
          const cs = Math.max(es, tx.cds_span[0]), ce = Math.min(ee, tx.cds_span[1]);
          if (cs <= ce) {
            const cx1 = sx(cs), cx2 = sx(ce);
            svg += `<rect x="${Math.min(cx1,cx2).toFixed(1)}" y="${cy-7.5}" width="${Math.max(2,Math.abs(cx2-cx1)).toFixed(1)}" height="15" rx="1" fill="${col}"/>`;
          }
        }
      });
    });
    const dexY = headerH + ig.txs.length * rowH + 46;
    const bins = ig.dex_bins || [];
    const maxDex = Math.max(1.0, ...bins.map(b => Math.abs(b.lfc)));
    svg += `<text x="4" y="${dexY+4}" font-size="10.5" font-weight="700" fill="#051c2c">DEXSeq Bins</text>`;
    svg += `<text x="4" y="${dexY+16}" font-size="9.2" fill="#64748b">padj &lt; 0.05 solid</text>`;
    svg += `<line x1="${padL}" y1="${dexY}" x2="${W-padR}" y2="${dexY}" stroke="#94a3b8" stroke-dasharray="3,2" stroke-width="1"/>`;
    const topBins = new Set(bins.filter(b => b.padj < 0.05).sort((a,b) => a.padj - b.padj).slice(0, 3).map(b => b.fid));
    bins.forEach(b => {
      const x1 = sx(b.start), x2 = sx(b.end), bx = Math.min(x1, x2), bw = Math.max(2.8, Math.abs(x2 - x1));
      const h = Math.min(24, Math.abs(b.lfc) / maxDex * 22), sig = b.padj < 0.05;
      const fill = b.lfc >= 0 ? (sig ? '#0d9488' : '#99f6e4') : (sig ? '#b83a4b' : '#fecaca');
      const by = b.lfc >= 0 ? (dexY - h) : dexY;
      svg += `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(2,h).toFixed(1)}" fill="${fill}" stroke="${sig ? '#051c2c' : 'none'}" stroke-width="0.6"><title>${esc(b.fid)} (${b.start}-${b.end}): log2FC=${b.lfc}, padj=${fmtQ(b.padj)}</title></rect>`;
      if (topBins.has(b.fid)) {
        const lx = Math.max(padL + 16, Math.min(W - padR - 16, bx + bw / 2));
        const ly = b.lfc >= 0 ? (by - 4) : (by + Math.max(2,h) + 11);
        svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="8.8" font-weight="700" text-anchor="middle" fill="#051c2c">${esc(b.fid)} ${sigStars(b.padj)}</text>`;
      }
    });
    svg += '</svg>';
    return svg;
  }

  function renderTab2ScatterSvg(rows, selIdx, C) {
    const W = 540, H = 330, x0 = 66, x1 = 516, y0 = 42, y1 = 278;
    const xMin = Math.min(-0.2, ...rows.map(r => Number(r.rmats_dpsi || 0))) * 1.12;
    const xMax = Math.max(0.2, ...rows.map(r => Number(r.rmats_dpsi || 0))) * 1.12;
    const yMin = Math.min(-1.0, ...rows.map(r => Number(r.dexseq_lfc || 0))) * 1.15;
    const yMax = Math.max(1.0, ...rows.map(r => Number(r.dexseq_lfc || 0))) * 1.15;
    const px = (v) => x0 + (v - xMin) / Math.max(1e-6, xMax - xMin) * (x1 - x0);
    const py = (v) => y1 - (v - yMin) / Math.max(1e-6, yMax - yMin) * (y1 - y0);
    const zx = Math.max(x0, Math.min(x1, px(0))), zy = Math.max(y0, Math.min(y1, py(0)));
    const etCol = { SE: '#051c2c', A3SS: '#0d9488', A5SS: '#d97706', RI: '#dc2626', MXE: '#7c3aed' };
    const cnt = C.counts;

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:540px;height:auto;font-family:sans-serif;">`;
    const rhoStr = cnt.class1_spearman_rho != null ? `Spearman ρ=${cnt.class1_spearman_rho.toFixed(2)} (p=${fmtQ(cnt.class1_spearman_p)})` : `${cnt.class1_concordant_pct}% concordant`;
    svg += `<text x="${x0}" y="26" font-size="10.2" font-weight="700" fill="#051c2c">Green = Both Tools Agree · ${esc(rhoStr)} · Click any dot</text>`;
    svg += `<rect x="${zx}" y="${y0}" width="${Math.max(0, x1-zx)}" height="${Math.max(0, zy-y0)}" fill="#ecfdf5" opacity="0.65"/>`;
    svg += `<rect x="${x0}" y="${zy}" width="${Math.max(0, zx-x0)}" height="${Math.max(0, y1-zy)}" fill="#ecfdf5" opacity="0.65"/>`;
    svg += `<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="none" stroke="#cbd5e1"/>`;
    svg += `<line x1="${x0}" y1="${zy}" x2="${x1}" y2="${zy}" stroke="#94a3b8" stroke-dasharray="3,2"/>`;
    svg += `<line x1="${zx}" y1="${y0}" x2="${zx}" y2="${y1}" stroke="#94a3b8" stroke-dasharray="3,2"/>`;
    svg += `<text x="${(x0+x1)/2}" y="314" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c">rMATS Splicing ΔΨ (← Lower in Treated | Higher in Treated →)</text>`;
    svg += `<text x="16" y="${(y0+y1)/2}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c" transform="rotate(-90 16 ${(y0+y1)/2})">DEXSeq Exon log2FC (← Lower | Higher →)</text>`;
    svg += `<text x="${x0+4}" y="${y1+14}" font-size="9.5" fill="#64748b">${xMin.toFixed(2)}</text>`;
    svg += `<text x="${x1-4}" y="${y1+14}" font-size="9.5" fill="#64748b" text-anchor="end">${xMax.toFixed(2)}</text>`;
    svg += `<text x="${x0-6}" y="${y0+8}" font-size="9.5" fill="#64748b" text-anchor="end">${yMax.toFixed(1)}</text>`;
    svg += `<text x="${x0-6}" y="${y1}" font-size="9.5" fill="#64748b" text-anchor="end">${yMin.toFixed(1)}</text>`;

    rows.forEach((r, idx) => {
      const cx = px(Number(r.rmats_dpsi || 0)), cy = py(Number(r.dexseq_lfc || 0));
      const col = etCol[r.event_type] || '#475569';
      const sel = idx === selIdx;
      svg += `<circle class="kcp-pt" data-ridx="${idx}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${sel ? 7.2 : 4.8}" fill="${col}" fill-opacity="${sel ? 1 : 0.82}" stroke="${sel ? '#051c2c' : '#fff'}" stroke-width="${sel ? 2.4 : 0.8}"><title>${esc(r.gene_symbol)} (${r.event_type}/${r.dexseq_exon}): rMATS ΔΨ=${fmtS(r.rmats_dpsi,3)} (FDR=${fmtQ(r.rmats_fdr)}), DEXSeq lfc=${fmtS(r.dexseq_lfc,2)} (padj=${fmtQ(r.dexseq_padj)})</title></circle>`;
    });
    const sr = rows[selIdx];
    if (sr) {
      const sx = px(Number(sr.rmats_dpsi || 0)), sy = py(Number(sr.dexseq_lfc || 0));
      svg += `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="10.5" fill="none" stroke="#d97706" stroke-width="2"/>`;
      const lx = Math.max(x0 + 95, Math.min(x1 - 95, sx));
      const ly = sy < y0 + 28 ? sy + 19 : sy - 14;
      svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c" paint-order="stroke" stroke="#ffffff" stroke-width="3.5" stroke-linejoin="round">${esc(sr.gene_symbol)} (${esc(sr.dexseq_exon)}) · ${sigStars(sr.dexseq_padj)}</text>`;
    }
    svg += '</svg>';
    return svg;
  }

  function renderTab2TrackSvg(r, C) {
    const W = 740, H = 292, x0 = 54, x1 = 712;
    const gInfo = getGeneBins(r.gene_base, r.dexseq_coord, r.dexseq_exon, r.dexseq_lfc, r.dexseq_padj);
    const bins = gInfo.bins || [];
    const eParts = String(r.event_id || '').split(':');
    let rmS = null, rmE = null;
    if (eParts.length >= 5 && !Number.isNaN(Number(eParts[3])) && !Number.isNaN(Number(eParts[4]))) {
      rmS = Math.min(Number(eParts[3]), Number(eParts[4]));
      rmE = Math.max(Number(eParts[3]), Number(eParts[4]));
    }
    const allPos = bins.flatMap(b => [b.start, b.end]);
    if (rmS != null) allPos.push(rmS, rmE);
    const minC = Math.min(...allPos, 0), maxC = Math.max(...allPos, minC + 1);
    const sx = (p) => x0 + (gInfo.strand === '-' ? (1 - (p - minC)/(maxC - minC)) : ((p - minC)/(maxC - minC))) * (x1 - x0);
    const dexY = 122, maxDex = Math.max(1.0, ...bins.map(b => Math.abs(b.lfc)));

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:740px;height:auto;font-family:sans-serif;">`;
    svg += `<text x="6" y="16" font-size="11" font-weight="700" fill="#051c2c">1. Whole-Gene DEXSeq Bins &amp; Physical ${r.overlap_bp}-bp rMATS Overlap (${esc(gInfo.chr)}, ${esc(gInfo.strand)})</text>`;
    if (rmS != null) {
      const rx1 = sx(rmS), rx2 = sx(rmE), rx = Math.min(rx1, rx2), rw = Math.max(6, Math.abs(rx2 - rx1));
      svg += `<rect x="${rx.toFixed(1)}" y="34" width="${rw.toFixed(1)}" height="126" fill="#fef3c7" opacity="0.55" stroke="#d97706" stroke-dasharray="3,2"/>`;
      const cxLabel = Math.max(x0 + 85, Math.min(x1 - 85, rx + rw / 2));
      svg += `<text x="${cxLabel.toFixed(1)}" y="48" font-size="9.8" font-weight="700" text-anchor="middle" fill="#92400e">rMATS ${esc(r.event_type)} (${r.overlap_bp} bp overlap)</text>`;
    }
    svg += `<line x1="${x0}" y1="${dexY}" x2="${x1}" y2="${dexY}" stroke="#64748b" stroke-width="1.2"/>`;
    bins.forEach(b => {
      const bx1 = sx(b.start), bx2 = sx(b.end), bx = Math.min(bx1, bx2), bw = Math.max(4, Math.abs(bx2 - bx1));
      const h = Math.min(46, Math.abs(b.lfc) / maxDex * 42), sig = b.padj < 0.05;
      const isHit = b.fid === r.dexseq_exon;
      const fill = b.lfc >= 0 ? (sig ? '#0d9488' : '#99f6e4') : (sig ? '#b83a4b' : '#fecaca');
      const by = b.lfc >= 0 ? (dexY - h) : dexY;
      svg += `<rect x="${bx.toFixed(1)}" y="${by.toFixed(1)}" width="${bw.toFixed(1)}" height="${Math.max(3,h).toFixed(1)}" fill="${fill}" stroke="${isHit ? '#051c2c' : 'none'}" stroke-width="${isHit ? 2.2 : 0}"><title>${esc(b.fid)}: log2FC=${b.lfc}, padj=${fmtQ(b.padj)}</title></rect>`;
      if (isHit) {
        const lx = Math.max(x0 + 115, Math.min(x1 - 115, bx + bw / 2));
        const ly = b.lfc >= 0 ? (by - 5) : (by + Math.max(3,h) + 12);
        svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10" font-weight="700" text-anchor="middle" fill="#051c2c">${esc(b.fid)} (${fmtQ(b.padj)} ${sigStars(b.padj)})</text>`;
      }
    });

    // Bottom paired signed bars with FDR & padj
    svg += `<line x1="6" y1="184" x2="${W-6}" y2="184" stroke="#e2e8f0"/>`;
    svg += `<text x="6" y="202" font-size="11" font-weight="700" fill="#051c2c">2. Dual-Tool Signed Effect &amp; Significance Orientation Check</text>`;
    const mid = 335, maxB = 155;
    const wd = Math.min(maxB, Math.abs(Number(r.rmats_dpsi || 0)) * maxB);
    const wl = Math.min(maxB, Math.abs(Number(r.dexseq_lfc || 0)) / 4.0 * maxB);
    const dp = Number(r.rmats_dpsi || 0), dl = Number(r.dexseq_lfc || 0);
    svg += `<line x1="${mid}" y1="214" x2="${mid}" y2="278" stroke="#94a3b8" stroke-dasharray="3,2"/>`;
    svg += `<text x="6" y="232" font-size="10.5" font-weight="700" fill="#051c2c">rMATS ${esc(r.event_type)} ΔΨ</text>`;
    svg += `<rect x="${dp>=0 ? mid : mid-wd}" y="220" width="${Math.max(5,wd)}" height="16" rx="2" fill="${dp>=0?'#0d9488':'#b83a4b'}"/>`;
    svg += `<text x="${dp>=0 ? mid+wd+8 : mid-wd-8}" y="232" font-size="10.2" font-weight="700" text-anchor="${dp>=0?'start':'end'}" fill="#051c2c">${fmtS(dp,3)} · FDR=${esc(fmtQ(r.rmats_fdr))} (${sigStars(r.rmats_fdr)})</text>`;
    svg += `<text x="6" y="264" font-size="10.5" font-weight="700" fill="#051c2c">DEXSeq ${esc(r.dexseq_exon)} log2FC</text>`;
    svg += `<rect x="${dl>=0 ? mid : mid-wl}" y="252" width="${Math.max(5,wl)}" height="16" rx="2" fill="${dl>=0?'#0d9488':'#b83a4b'}"/>`;
    svg += `<text x="${dl>=0 ? mid+wl+8 : mid-wl-8}" y="264" font-size="10.2" font-weight="700" text-anchor="${dl>=0?'start':'end'}" fill="#051c2c">${fmtS(dl,2)} (raw ${fmtS(r.dexseq_raw_lfc,2)}) · padj=${esc(fmtQ(r.dexseq_padj))} (${sigStars(r.dexseq_padj)})</text>`;
    svg += '</svg>';
    return svg;
  }

  function renderTab3VolcanoSvg(rows, selIdx, C) {
    const W = 540, H = 330, x0 = 66, x1 = 516, y0 = 42, y1 = 278;
    const xMin = Math.min(-0.45, ...rows.map(r => Number(r.dpsi || 0))) * 1.1;
    const xMax = Math.max(0.45, ...rows.map(r => Number(r.dpsi || 0))) * 1.1;
    const yVals = rows.map(r => -Math.log10(Math.max(Number(r.bh_fdr ?? r.pvalue ?? 1), 1e-16)));
    const yMax = Math.max(2.8, ...yVals) * 1.12;
    const px = (v) => x0 + (v - xMin) / Math.max(1e-6, xMax - xMin) * (x1 - x0);
    const py = (y) => y1 - (y / yMax) * (y1 - y0);
    const yCut = py(1.30103);

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:540px;height:auto;font-family:sans-serif;">`;
    svg += `<text x="${x0}" y="26" font-size="10.2" font-weight="700" fill="#051c2c">Solid = BH-FDR &lt; 0.05 · Hollow = Nominal p &lt; 0.05 · Click any dot</text>`;
    svg += `<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${Math.max(0, yCut-y0)}" fill="#fffbeb" opacity="0.55"/>`;
    svg += `<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="none" stroke="#cbd5e1"/>`;
    svg += `<line x1="${x0}" y1="${yCut.toFixed(1)}" x2="${x1}" y2="${yCut.toFixed(1)}" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.2"/>`;
    svg += `<text x="${x1-4}" y="${(yCut-4).toFixed(1)}" font-size="9.2" font-weight="700" text-anchor="end" fill="#dc2626">BH-FDR = 0.05 (y=1.30)</text>`;
    svg += `<line x1="${px(0)}" y1="${y0}" x2="${px(0)}" y2="${y1}" stroke="#94a3b8" stroke-dasharray="3,2"/>`;
    svg += `<text x="${(x0+x1)/2}" y="314" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c">SUPPA2 Promoter/PolyA ΔΨ (← Lower in Treated | Higher in Treated →)</text>`;
    svg += `<text x="16" y="${(y0+y1)/2}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c" transform="rotate(-90 16 ${(y0+y1)/2})">Significance: −log10(BH-FDR)</text>`;
    svg += `<text x="${x0+4}" y="${y1+14}" font-size="9.2" fill="#64748b">${xMin.toFixed(2)}</text>`;
    svg += `<text x="${px(0).toFixed(1)}" y="${y1+14}" font-size="9.2" fill="#64748b" text-anchor="middle">0</text>`;
    svg += `<text x="${x1-4}" y="${y1+14}" font-size="9.2" fill="#64748b" text-anchor="end">${xMax.toFixed(2)}</text>`;
    svg += `<text x="${x0-6}" y="${y0+8}" font-size="9.2" fill="#64748b" text-anchor="end">${yMax.toFixed(1)}</text>`;
    svg += `<text x="${x0-6}" y="${y1}" font-size="9.2" fill="#64748b" text-anchor="end">0</text>`;

    rows.forEach((r, idx) => {
      const cx = px(Number(r.dpsi || 0)), cy = py(yVals[idx]);
      const sig = Boolean(r.sig_fdr);
      const col = r.etype === 'AL' ? '#dc2626' : '#d97706';
      const sel = idx === selIdx;
      svg += `<circle class="kcp-pt" data-ridx="${idx}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${sel ? 7.2 : 5}" fill="${sig ? col : '#fff'}" fill-opacity="${sig ? 0.88 : 0.9}" stroke="${sel ? '#051c2c' : col}" stroke-width="${sel ? 2.4 : 1.5}"><title>${esc(r.gene_symbol)} (${r.etype}): ΔΨ=${fmtS(r.dpsi,3)}, p=${fmtQ(r.pvalue)}, BH-FDR=${fmtQ(r.bh_fdr)}</title></circle>`;
    });
    const sr = rows[selIdx];
    if (sr) {
      const sx = px(Number(sr.dpsi || 0)), sy = py(yVals[selIdx]);
      svg += `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="10.5" fill="none" stroke="#051c2c" stroke-width="2"/>`;
      const lx = Math.max(x0 + 95, Math.min(x1 - 95, sx));
      const ly = sy < y0 + 28 ? sy + 19 : sy - 14;
      svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c" paint-order="stroke" stroke="#ffffff" stroke-width="3.5" stroke-linejoin="round">${esc(sr.gene_symbol)} (${sr.etype}) · ${sigStars(sr.bh_fdr)}</text>`;
    }
    svg += '</svg>';
    return svg;
  }

  function renderTab3ArchSvg(r, C) {
    const W = 740, H = 292;
    const isAF = r.etype === 'AF';
    const parts = String(r.event_id || '').split(':');
    const strand = parts.length >= 6 ? parts[parts.length - 1] : '+';
    const chr = parts.length >= 2 ? parts[1] : '';
    const dpsi = Number(r.dpsi || 0);

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:740px;height:auto;font-family:sans-serif;">`;
    svg += `<text x="6" y="18" font-size="11" font-weight="700" fill="#051c2c">1. ${isAF ? 'Alternative First Exon (5′ Promoter Choice)' : 'Alternative Last Exon (3′ PolyA / Terminal Choice)'} (${esc(chr)}, ${esc(strand)})</text>`;
    svg += `<text x="6" y="34" font-size="9.5" fill="#475569">rMATS cannot model alternative 5′ promoters or 3′ polyA termini · SUPPA2 ΔΨ = ${fmtS(dpsi, 3)}</text>`;

    // Draw schematic two competing boundary isoforms
    if (isAF) {
      svg += `<rect x="70" y="54" width="85" height="18" rx="2" fill="#d97706"/><text x="112" y="66" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Alt 5′ E1a</text>`;
      svg += `<path d="M 155 63 Q 315 38 470 63" fill="none" stroke="#d97706" stroke-width="2"/>`;
      svg += `<rect x="470" y="54" width="130" height="18" rx="2" fill="#64748b"/><text x="535" y="66" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Shared Exon 2</text>`;
      svg += `<rect x="260" y="96" width="85" height="18" rx="2" fill="#0d9488"/><text x="302" y="108" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Alt 5′ E1b</text>`;
      svg += `<path d="M 345 105 Q 408 82 470 105" fill="none" stroke="#0d9488" stroke-width="2"/>`;
      svg += `<rect x="470" y="96" width="130" height="18" rx="2" fill="#64748b"/><text x="535" y="108" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Shared Exon 2</text>`;
    } else {
      svg += `<rect x="70" y="54" width="130" height="18" rx="2" fill="#64748b"/><text x="135" y="66" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Shared Exon</text>`;
      svg += `<path d="M 200 63 Q 275 40 350 63" fill="none" stroke="#d97706" stroke-width="2"/>`;
      svg += `<rect x="350" y="54" width="90" height="18" rx="2" fill="#d97706"/><text x="395" y="66" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Proximal PAS</text>`;
      svg += `<rect x="70" y="96" width="130" height="18" rx="2" fill="#64748b"/><text x="135" y="108" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Shared Exon</text>`;
      svg += `<path d="M 200 105 Q 370 78 535 105" fill="none" stroke="#dc2626" stroke-width="2"/>`;
      svg += `<rect x="535" y="96" width="90" height="18" rx="2" fill="#dc2626"/><text x="580" y="108" font-size="9.5" font-weight="700" fill="#fff" text-anchor="middle">Distal PAS</text>`;
    }
    svg += `<text x="6" y="138" font-size="9" fill="#475569">Locus Key: ${esc(r.event_id)}</text>`;

    // 2. Significance Gauge (-log10 P and BH-FDR)
    svg += `<line x1="6" y1="154" x2="${W-6}" y2="154" stroke="#e2e8f0"/>`;
    svg += `<text x="6" y="172" font-size="11" font-weight="700" fill="#051c2c">2. Statistical Significance Gauge (−log10 Scale vs 0.05 Cutoff)</text>`;
    const lp = -Math.log10(Math.max(Number(r.pvalue ?? 1), 1e-16));
    const lf = -Math.log10(Math.max(Number(r.bh_fdr ?? 1), 1e-16));
    const maxLog = Math.max(4.0, lp, lf) * 1.1;
    const gx0 = 155, gw = 380;
    const xg = (v) => gx0 + Math.min(1, v / maxLog) * gw;
    const xCut = xg(1.30103);
    svg += `<line x1="${xCut.toFixed(1)}" y1="184" x2="${xCut.toFixed(1)}" y2="250" stroke="#dc2626" stroke-dasharray="3,2" stroke-width="1.4"/>`;
    svg += `<text x="${xCut.toFixed(1)}" y="182" font-size="8.8" font-weight="700" fill="#dc2626" text-anchor="middle">0.05 Cutoff</text>`;

    svg += `<text x="6" y="205" font-size="10.5" font-weight="700" fill="#051c2c">SUPPA2 Nominal P</text>`;
    svg += `<rect x="${gx0}" y="192" width="${Math.max(4, xg(lp)-gx0).toFixed(1)}" height="16" rx="2" fill="#0d9488"/>`;
    svg += `<text x="${xg(lp)+8}" y="205" font-size="10.2" font-weight="700" fill="#051c2c">p = ${esc(fmtQ(r.pvalue))} (${sigStars(r.pvalue)})</text>`;

    svg += `<text x="6" y="236" font-size="10.5" font-weight="700" fill="#051c2c">SUPPA2 BH-FDR</text>`;
    svg += `<rect x="${gx0}" y="223" width="${Math.max(4, xg(lf)-gx0).toFixed(1)}" height="16" rx="2" fill="${r.sig_fdr ? '#d97706' : '#94a3b8'}"/>`;
    svg += `<text x="${xg(lf)+8}" y="236" font-size="10.2" font-weight="700" fill="${r.sig_fdr ? '#92400e' : '#64748b'}">BH-FDR = ${esc(fmtQ(r.bh_fdr))} (${sigStars(r.bh_fdr)})</text>`;

    svg += `<text x="6" y="274" font-size="10.2" font-weight="700" fill="${r.in_ps_das ? '#0f766e' : '#b91c1c'}">rMATS PS-DAS: ${r.in_ps_das ? 'Also had internal event' : 'COMPLETELY MISSED (Blind Spot)'}</text>`;
    svg += `<text x="420" y="274" font-size="10.2" font-weight="700" fill="${r.in_deg ? '#0f766e' : '#b91c1c'}">DESeq2 DEG: ${r.in_deg ? 'Significant gene' : 'COMPLETELY MISSED (Flat total RNA)'}</text>`;
    svg += '</svg>';
    return svg;
  }

  function renderTab4VolcanoSvg(rows, selIdx, C) {
    const W = 540, H = 330, x0 = 66, x1 = 516, y0 = 42, y1 = 278;
    const xMin = Math.min(-1.0, ...rows.map(r => Number(r.lfc || 0))) * 1.12;
    const xMax = Math.max(1.0, ...rows.map(r => Number(r.lfc || 0))) * 1.12;
    const yVals = rows.map(r => Math.min(35, -Math.log10(Math.max(Number(r.padj ?? 1), 1e-35))));
    const yMax = Math.max(4.0, ...yVals) * 1.12;
    const px = (v) => x0 + (v - xMin) / Math.max(1e-6, xMax - xMin) * (x1 - x0);
    const py = (y) => y1 - (y / yMax) * (y1 - y0);
    const yCut = py(1.30103);
    const regCol = (reg) => String(reg).includes('5') ? '#d97706' : (String(reg).includes('3') ? '#0d9488' : '#7c3aed');

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:540px;height:auto;font-family:sans-serif;">`;
    svg += `<text x="${x0}" y="26" font-size="10.2" font-weight="700" fill="#051c2c">Position: 5′ TSS (amber) · Internal (violet) · 3′ PAS (teal) · Click dot</text>`;
    svg += `<rect x="${x0}" y="${y0}" width="${x1-x0}" height="${y1-y0}" fill="none" stroke="#cbd5e1"/>`;
    svg += `<line x1="${x0}" y1="${yCut.toFixed(1)}" x2="${x1}" y2="${yCut.toFixed(1)}" stroke="#dc2626" stroke-dasharray="4,3" stroke-width="1.1"/>`;
    svg += `<line x1="${px(-0.5)}" y1="${y0}" x2="${px(-0.5)}" y2="${y1}" stroke="#94a3b8" stroke-dasharray="3,2"/>`;
    svg += `<line x1="${px(0.5)}" y1="${y0}" x2="${px(0.5)}" y2="${y1}" stroke="#94a3b8" stroke-dasharray="3,2"/>`;
    svg += `<text x="${(x0+x1)/2}" y="314" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c">DEXSeq Exon log2FC (← Lower in Treated | Higher in Treated →)</text>`;
    svg += `<text x="16" y="${(y0+y1)/2}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c" transform="rotate(-90 16 ${(y0+y1)/2})">Significance: −log10(padj, cap 35)</text>`;
    svg += `<text x="${x0+4}" y="${y1+14}" font-size="9.2" fill="#64748b">${xMin.toFixed(1)}</text>`;
    svg += `<text x="${px(0).toFixed(1)}" y="${y1+14}" font-size="9.2" fill="#64748b" text-anchor="middle">0</text>`;
    svg += `<text x="${x1-4}" y="${y1+14}" font-size="9.2" fill="#64748b" text-anchor="end">${xMax.toFixed(1)}</text>`;
    svg += `<text x="${x0-6}" y="${y0+8}" font-size="9.2" fill="#64748b" text-anchor="end">${yMax.toFixed(0)}</text>`;
    svg += `<text x="${x0-6}" y="${y1}" font-size="9.2" fill="#64748b" text-anchor="end">0</text>`;

    rows.forEach((r, idx) => {
      const cx = px(Number(r.lfc || 0)), cy = py(yVals[idx]);
      const col = regCol(r.region_bin);
      const sel = idx === selIdx;
      svg += `<circle class="kcp-pt" data-ridx="${idx}" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${sel ? 7.2 : 5}" fill="${col}" fill-opacity="${sel ? 1 : 0.85}" stroke="${sel ? '#051c2c' : '#fff'}" stroke-width="${sel ? 2.4 : 0.8}"><title>${esc(r.gene_symbol)} ${esc(r.feature_id)} (${esc(r.region_bin)}): lfc=${fmtS(r.lfc,2)}, padj=${fmtQ(r.padj)}, counts=${Math.round(r.exonBaseMean||0)}</title></circle>`;
    });
    const sr = rows[selIdx];
    if (sr) {
      const sx = px(Number(sr.lfc || 0)), sy = py(yVals[selIdx]);
      svg += `<circle cx="${sx.toFixed(1)}" cy="${sy.toFixed(1)}" r="10.5" fill="none" stroke="#051c2c" stroke-width="2"/>`;
      const lx = Math.max(x0 + 100, Math.min(x1 - 100, sx));
      const ly = sy < y0 + 28 ? sy + 19 : sy - 14;
      svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c" paint-order="stroke" stroke="#ffffff" stroke-width="3.5" stroke-linejoin="round">${esc(sr.gene_symbol)}:${esc(sr.feature_id)} (${sigStars(sr.padj)})</text>`;
    }
    svg += '</svg>';
    return svg;
  }

  function renderTab4ProfileSvg(r, C) {
    const W = 740, H = 292, x0 = 54, x1 = 712;
    const gInfo = getGeneBins(r.gene_base, r.coord, r.feature_id, r.lfc, r.padj);
    const bins = gInfo.bins || [];
    const nB = Math.max(1, bins.length);
    const dexY = 128, maxDex = Math.max(1.0, ...bins.map(b => Math.abs(b.lfc)));

    let svg = `<svg viewBox="0 0 ${W} ${H}" style="width:100%;max-width:740px;height:auto;font-family:sans-serif;">`;
    svg += `<text x="6" y="18" font-size="11" font-weight="700" fill="#051c2c">1. Whole-Gene 5′ → 3′ DEXSeq Exon Body Profile (${esc(r.gene_symbol)} · ${bins.length} bins shown)</text>`;
    svg += `<line x1="${x0}" y1="${dexY}" x2="${x1}" y2="${dexY}" stroke="#64748b" stroke-width="1.4"/>`;
    svg += `<text x="${x0}" y="196" font-size="9.5" font-weight="700" fill="#64748b">5′ TSS (0%)</text>`;
    svg += `<text x="${x1}" y="196" font-size="9.5" font-weight="700" fill="#64748b" text-anchor="end">3′ PAS (100%)</text>`;

    bins.forEach((b, i) => {
      const posFrac = b.pos_5to3 != null ? Math.max(0, Math.min(1, Number(b.pos_5to3))) : ((i + 0.5) / nB);
      const bx = x0 + posFrac * (x1 - x0);
      const h = Math.min(54, Math.abs(b.lfc) / maxDex * 48);
      const sig = b.padj < 0.05;
      const isHit = b.fid === r.feature_id;
      const col = b.lfc >= 0 ? (sig ? '#0d9488' : '#cbd5e1') : (sig ? '#b83a4b' : '#cbd5e1');
      const cy = b.lfc >= 0 ? (dexY - h) : (dexY + h);
      svg += `<line x1="${bx.toFixed(1)}" y1="${dexY}" x2="${bx.toFixed(1)}" y2="${cy.toFixed(1)}" stroke="${col}" stroke-width="${isHit ? 4.5 : 2.5}"/>`;
      svg += `<circle cx="${bx.toFixed(1)}" cy="${cy.toFixed(1)}" r="${isHit ? 6.5 : (sig ? 4.2 : 3)}" fill="${col}" stroke="${isHit ? '#051c2c' : 'none'}" stroke-width="${isHit ? 2.2 : 0}"><title>${esc(b.fid)}: pos=${(posFrac*100).toFixed(0)}%, lfc=${b.lfc}, padj=${fmtQ(b.padj)}</title></circle>`;
      if (isHit) {
        const lx = Math.max(x0 + 135, Math.min(x1 - 135, bx));
        const ly = b.lfc >= 0 ? Math.max(36, cy - 11) : Math.min(182, cy + 16);
        svg += `<text x="${lx.toFixed(1)}" y="${ly.toFixed(1)}" font-size="10.2" font-weight="700" text-anchor="middle" fill="#051c2c">${esc(b.fid)} · lfc ${fmtS(r.lfc,2)} · padj ${fmtQ(r.padj)} (${sigStars(r.padj)})</text>`;
      }
    });

    svg += `<line x1="6" y1="210" x2="${W-6}" y2="210" stroke="#e2e8f0"/>`;
    svg += `<text x="6" y="232" font-size="10.8" font-weight="700" fill="#051c2c">Exon Locus: ${esc(r.coord || '')}   ·   Positional Region: ${esc(r.region_bin || '')}</text>`;
    svg += `<text x="6" y="254" font-size="10.5" font-weight="700" fill="#051c2c">Mean Base Counts (exonBaseMean): ${Math.round(Number(r.exonBaseMean || 0)).toLocaleString()} reads</text>`;
    svg += `<text x="6" y="276" font-size="10.5" font-weight="700" fill="${r.in_tested_rmats ? '#0f766e' : '#7c3aed'}">rMATS Coverage: ${r.in_tested_rmats ? 'Tested exon in rMATS' : 'NEVER TESTED by rMATS (DEXSeq Blind Spot)'}</text>`;
    svg += `<text x="435" y="276" font-size="10.5" font-weight="700" fill="${r.gene_in_deg ? '#0f766e' : '#b91c1c'}">DESeq2 DEG: ${r.gene_in_deg ? 'Significant' : 'Hidden from DEG'}</text>`;
    svg += '</svg>';
    return svg;
  }

  function wirePlotPoints(rows) {
    document.querySelectorAll('#kcp-left-svg .kcp-pt').forEach(el => {
      el.addEventListener('click', () => {
        const idx = parseInt(el.dataset.ridx, 10);
        const r = rows[idx];
        if (!r) return;
        activeRowKey = `${activeTab}_${r.gene_base || ''}_${r.event_id || r.feature_id || idx}`;
        const tKind = activeTab === 'class1' ? 'c1' : (activeTab === 'class2_af_al' ? 'c2' : (activeTab === 'class3_switches' ? 'c3' : 'c4'));
        focusObj = { kind: tKind, row: r, ig: findIg(r.gene_symbol, r.gene_base), ridx: idx };
        renderStory();
        renderTable();
      });
    });
  }

  function mCard(lbl, val, badgeHtml) {
    return `<div class="kcp-mcard"><span class="kcp-mlbl">${lbl}</span><span class="kcp-mval">${val}</span>${badgeHtml || ''}</div>`;
  }

  const tabGuides = {
    class3_switches: '<strong>How to read:</strong> Left compares flat Total Gene mRNA (missed by DESeq2) vs shifted Protein-Coding / NMD / Retained Intron share (○ Control → ● Treated); Right aligns rising (▲) vs falling (▼) exons/CDS with DEXSeq bins.',
    class1: '<strong>How to read:</strong> Green quadrants = rMATS and DEXSeq agree in direction on the exact same DNA exon (best RT-PCR candidates). Click any dot or table row to inspect its genomic overlap on the right.',
    class2_af_al: '<strong>How to read:</strong> Alternative 5′ promoters (AF) and 3′ polyA ends (AL) are invisible to rMATS. Solid colored dots above the red dashed line pass strict BH-FDR &lt; 0.05 (used for uplift counts); hollow circles are exploratory nominal p &lt; 0.05.',
    class4_dexseq: '<strong>How to read:</strong> Significant DEXSeq exons missed or never tested by rMATS, colored by position (5′ TSS / Internal / 3′ PAS). Click any dot or row to inspect the whole-gene 5′→3′ profile on the right.',
  };
  const tabLabels = {
    class3_switches: 'Tab 1 Isoform Switches',
    class1: 'Tab 2 Dual-Verified',
    class2_af_al: 'Tab 3 5′/3′ Boundaries',
    class4_dexseq: 'Tab 4 DEXSeq-Only',
  };

  function renderStory() {
    const C = CP.contrasts[activeCid];
    document.getElementById('kcp-guide').innerHTML = tabGuides[activeTab] || '';
    if (!focusObj) focusObj = defaultFocusForTab();
    if (!focusObj) {
      document.getElementById('kcp-verdict').innerHTML = 'No events match current filter.';
      document.getElementById('kcp-left-svg').innerHTML = '';
      document.getElementById('kcp-svg-track').innerHTML = '';
      document.getElementById('kcp-exon-pills').innerHTML = '';
      document.getElementById('kcp-st-box').innerHTML = '';
      document.getElementById('kcp-tx-wrap').style.display = 'none';
      return;
    }
    const { rows } = filteredRowsAndCols();
    const r = focusObj.row;
    const ig = focusObj.ig || findIg(r.gene_symbol, r.gene_base);
    const sym = (ig && ig.gene_symbol) || (r && r.gene_symbol) || 'Gene';
    const selIdx = Math.max(0, rows.findIndex(x => (x.gene_base === r.gene_base && (x.event_id === r.event_id) && (x.feature_id === r.feature_id))));

    const vBox = document.getElementById('kcp-verdict');
    const stBox = document.getElementById('kcp-st-box');
    const txWrap = document.getElementById('kcp-tx-wrap');
    const exonPills = document.getElementById('kcp-exon-pills');
    let coordCopy = '';

    if (activeTab === 'class3_switches' || focusObj.kind === 'ig') {
      const topSigBin = ig ? (ig.dex_bins || []).filter(b => b.padj < 0.05).sort((a, b) => a.padj - b.padj)[0] : null;
      coordCopy = topSigBin ? `${ig.chr}:${topSigBin.start}-${topSigBin.end}(${ig.strand})` : (ig ? `${ig.chr}:${ig.span[0]}-${ig.span[1]}(${ig.strand})` : sym);
      const hid = ig ? ig.hidden_from_deg : r.hidden_from_deg;
      const fate = (ig && ig.fate) || r.fate || '';
      const [tLbl, tCls] = tierInfo((ig && ig.tx_confirm_tier) || r.tx_confirm_tier);
      const tup = (ig && ig.txs && ig.txs.find(t => t.role === 'T_up')) || { tx_id: r.tx_up, dIF: r.tx_up_dif, tx_q: r.tx_up_q };
      const tdn = (ig && ig.txs && ig.txs.find(t => t.role === 'T_dn')) || { tx_id: r.tx_dn, dIF: r.tx_dn_dif, tx_q: r.tx_dn_q };
      const c1Hit = findC1(sym, (ig && ig.gene_base) || r.gene_base);

      vBox.className = 'kcp-sig-strip ' + (hid ? 'warn' : '');
      const cards = [
        mCard('DESeq2 DEG (Gene Sum)', `lfc ${fmtS(ig ? ig.deg_log2fc : r.deg_log2fc, 2)}`, `${sigBadge(ig ? ig.deg_fdr : r.deg_fdr)} ${hid ? '<span class="kcp-pill kcp-p-cor">Missed DEG</span>' : '<span class="kcp-pill kcp-p-teal">DEG Sig</span>'}`),
        mCard('stageR DTU (Gene)', `Switch ${((ig ? ig.switch_mag : r.switch_mag)*100).toFixed(1)}%`, `${sigBadge(ig ? ig.gene_q : r.gene_q)} <span class="kcp-pill ${tCls}">${tLbl}</span>`),
        mCard(`Rising Isoform (${esc(tup.tx_id)})`, `ΔIF ${fmtS(Number(tup.dIF||0)*100,1)}%`, `tx_q ${sigBadge(tup.tx_q)}`),
        mCard(`Falling Isoform (${esc(tdn.tx_id)})`, `ΔIF ${fmtS(Number(tdn.dIF||0)*100,1)}%`, `tx_q ${sigBadge(tdn.tx_q)}`),
      ];
      if (topSigBin || c1Hit) {
        const bLfc = c1Hit ? c1Hit.dexseq_lfc : topSigBin.lfc;
        const bPadj = c1Hit ? c1Hit.dexseq_padj : topSigBin.padj;
        const bId = c1Hit ? c1Hit.dexseq_exon : topSigBin.fid;
        cards.push(mCard(`DEXSeq Bin (${esc(bId)})`, `lfc ${fmtS(bLfc, 2)}`, `padj ${sigBadge(bPadj)}`));
      }
      vBox.innerHTML = `<div class="kcp-sig-top"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;"><strong style="font-size:16px;color:#051c2c;">${esc(sym)}</strong><span class="kcp-pill kcp-p-amb">${esc(fate)} (Δaa=${fmtS(ig?ig.delta_aa:r.delta_aa,0)})</span></div><button type="button" class="kcp-btn" onclick="window._kcpCopyCoord(this, '${esc(coordCopy)}')">Copy ${topSigBin ? esc(topSigBin.fid) : 'Locus'} Coords</button></div><div class="kcp-sig-grid">${cards.join('')}</div>`;

      document.getElementById('kcp-left-title').textContent = `A. ${sym}: Flat Gene Sum vs Internal Isoform Shift & Significance`;
      document.getElementById('kcp-left-svg').innerHTML = renderTab1LeftSvg(ig, r, C);
      document.getElementById('kcp-right-title').textContent = `B. ${sym}: Exon/CDS Isoform Tracks & DEXSeq Bin Significance`;
      document.getElementById('kcp-svg-track').innerHTML = renderTab1RightSvg(ig, r);

      const sigBins = ig ? (ig.dex_bins || []).filter(b => b.padj < 0.05).sort((a, b) => a.padj - b.padj).slice(0, 4) : [];
      exonPills.innerHTML = sigBins.length
        ? `<span style="font-size:10.8px;font-weight:700;color:#475569;">Copy significant bin:</span>` + sigBins.map(b => `<button type="button" class="kcp-btn" style="font-size:10.8px;padding:2px 7px;" onclick="window._kcpCopyCoord(this, '${ig.chr}:${b.start}-${b.end}(${ig.strand})')">${esc(b.fid)} (${fmtQ(b.padj)} ${sigStars(b.padj)})</button>`).join('')
        : '';

      if (ig && ig.splicetrace && ig.splicetrace.length) {
        stBox.innerHTML = `<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:4px;padding:6px 10px;font-size:11.2px;"><strong>SpliceTrace Domain Cross-Check:</strong> ` + ig.splicetrace.map(s => `<code>${esc(s.event_id)}</code> (ΔΨ=${s.dpsi}${s.fdr!=null ? ', FDR='+fmtQ(s.fdr) : ''}) → <strong>${esc(s.nmd_switch)}</strong> [${esc(s.frame)}]`).join(' · ') + `</div>`;
      } else {
        stBox.innerHTML = '';
      }

      if (ig && ig.txs && ig.txs.length) {
        txWrap.style.display = 'block';
        let th = `<table class="kcp-tbl"><thead><tr><th>Transcript</th><th>Biotype (ORF)</th><th>${esc(C.ctrl_label)} IF</th><th>${esc(C.treat_label)} IF</th><th>ΔIF</th><th>tx_q</th><th>TPM (${esc(C.ctrl_label)}→${esc(C.treat_label)})</th></tr></thead><tbody>`;
        ig.txs.forEach(tx => {
          th += `<tr><td><code>${esc(tx.tx_id)}</code> (${esc(tx.role)})</td><td>${esc(shortBt(tx.biotype))} (${tx.aa_len}aa)</td><td>${(tx.if_ctrl*100).toFixed(1)}%</td><td>${(tx.if_treat*100).toFixed(1)}%</td><td style="font-weight:700;color:${tx.dIF>=0?'#0d9488':'#b83a4b'}">${tx.dIF>=0?'+':''}${(tx.dIF*100).toFixed(1)}%</td><td>${sigBadge(tx.tx_q)}</td><td>${tx.tpm_ctrl} → ${tx.tpm_treat}</td></tr>`;
        });
        document.getElementById('kcp-tx-tbl').innerHTML = th + '</tbody></table>';
      } else {
        txWrap.style.display = 'none';
      }
    } else if (activeTab === 'class1') {
      coordCopy = r.dexseq_coord || r.event_id;
      vBox.className = 'kcp-sig-strip';
      const c3Match = findC3(sym, r.gene_base);
      const cards = [
        mCard(`rMATS ${esc(r.event_type)}`, `ΔΨ = ${fmtS(r.rmats_dpsi, 3)}`, `FDR ${sigBadge(r.rmats_fdr)}`),
        mCard(`DEXSeq ${esc(r.dexseq_exon)}`, `lfc = ${fmtS(r.dexseq_lfc, 2)}`, `padj ${sigBadge(r.dexseq_padj)}`),
        mCard('Physical Coordinate Match', `${r.overlap_bp} bp overlap`, `<span class="kcp-pill ${r.concordant_fixed ? 'kcp-p-teal' : 'kcp-p-amb'}">${r.concordant_fixed ? 'Same Direction' : 'Discordant'}</span>`),
      ];
      if (c3Match) {
        cards.push(mCard('DESeq2 DEG (Gene Sum)', `lfc ${fmtS(c3Match.deg_log2fc, 2)}`, `FDR ${sigBadge(c3Match.deg_fdr)}`));
      }
      vBox.innerHTML = `<div class="kcp-sig-top"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;"><strong style="font-size:16px;color:#051c2c;">${esc(sym)} · ${esc(r.dexseq_exon)}</strong><span class="kcp-pill kcp-p-teal">1-bp Dual-Tool Confirmed</span></div><button type="button" class="kcp-btn" onclick="window._kcpCopyCoord(this, '${esc(coordCopy)}')">Copy Exon Coords</button></div><div class="kcp-sig-grid">${cards.join('')}</div>`;
      document.getElementById('kcp-left-title').textContent = `A. Coordinate-Verified Concordance Plot (${rows.length} events · Click dot)`;
      document.getElementById('kcp-left-svg').innerHTML = renderTab2ScatterSvg(rows, selIdx, C);
      document.getElementById('kcp-right-title').textContent = `B. ${sym} (${r.dexseq_exon}): Whole-Gene DEXSeq Profile & rMATS Overlap`;
      document.getElementById('kcp-svg-track').innerHTML = renderTab2TrackSvg(r, C);
      exonPills.innerHTML = `<span style="font-size:11px;color:#475569;">Matched Coordinate: <code>${esc(coordCopy)}</code></span>`;
      stBox.innerHTML = ''; txWrap.style.display = 'none';
      wirePlotPoints(rows);
    } else if (activeTab === 'class2_af_al') {
      coordCopy = r.event_id;
      vBox.className = 'kcp-sig-strip';
      const cards = [
        mCard(`SUPPA2 ${r.etype === 'AF' ? '5′ Alt First (AF)' : '3′ Alt Last (AL)'}`, `ΔΨ = ${fmtS(r.dpsi, 3)}`, `<span class="kcp-pill kcp-p-amb">rMATS-Blind Boundary</span>`),
        mCard('SUPPA2 Nominal P', `|ΔΨ| = ${Math.abs(Number(r.dpsi||0)).toFixed(3)}`, sigBadge(r.pvalue)),
        mCard('SUPPA2 BH-FDR', r.sig_fdr ? 'Passed BH-FDR < 0.05' : 'Nominal p < 0.05 only', sigBadge(r.bh_fdr)),
        mCard('Standard Pipeline Status', r.in_ps_das ? 'In PS-DAS' : 'Missed by rMATS', `<span class="kcp-pill ${r.in_deg ? 'kcp-p-teal' : 'kcp-p-cor'}">${r.in_deg ? 'DEG Sig' : 'Missed by DEG'}</span>`),
      ];
      vBox.innerHTML = `<div class="kcp-sig-top"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;"><strong style="font-size:16px;color:#051c2c;">${esc(sym)} (${esc(r.etype)})</strong></div><button type="button" class="kcp-btn" onclick="window._kcpCopyCoord(this, '${esc(coordCopy)}')">Copy Event ID</button></div><div class="kcp-sig-grid">${cards.join('')}</div>`;
      document.getElementById('kcp-left-title').textContent = `A. SUPPA2 AF/AL Boundary Volcano (${rows.length} events · Click dot)`;
      document.getElementById('kcp-left-svg').innerHTML = renderTab3VolcanoSvg(rows, selIdx, C);
      document.getElementById('kcp-right-title').textContent = `B. ${sym}: ${r.etype} Isoform Architecture & P / BH-FDR Significance Gauge`;
      document.getElementById('kcp-svg-track').innerHTML = renderTab3ArchSvg(r, C);
      exonPills.innerHTML = ''; stBox.innerHTML = ''; txWrap.style.display = 'none';
      wirePlotPoints(rows);
    } else {
      coordCopy = r.coord || r.feature_id;
      vBox.className = 'kcp-sig-strip';
      const cards = [
        mCard(`DEXSeq ${esc(r.feature_id)}`, `log2FC = ${fmtS(r.lfc, 2)}`, `padj ${sigBadge(r.padj)}`),
        mCard('Positional Bin', esc(r.region_bin || 'Exon'), `BaseMean = ${Math.round(Number(r.exonBaseMean||0)).toLocaleString()}`),
        mCard('rMATS Coverage', r.in_tested_rmats ? 'Tested exon' : 'Never tested', `<span class="kcp-pill ${r.in_tested_rmats ? 'kcp-p-sl' : 'kcp-p-vio'}">${r.in_tested_rmats ? 'Tested' : 'DEXSeq-Only'}</span>`),
        mCard('DESeq2 DEG (Gene Sum)', r.gene_in_deg ? 'DEG Significant' : 'Flat gene sum', `<span class="kcp-pill ${r.gene_in_deg ? 'kcp-p-teal' : 'kcp-p-cor'}">${r.gene_in_deg ? 'In DEG' : 'Missed DEG'}</span>`),
      ];
      vBox.innerHTML = `<div class="kcp-sig-top"><div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;"><strong style="font-size:16px;color:#051c2c;">${esc(sym)} · ${esc(r.feature_id)}</strong><span class="kcp-pill kcp-p-vio">DEXSeq Exon Usage</span></div><button type="button" class="kcp-btn" onclick="window._kcpCopyCoord(this, '${esc(coordCopy)}')">Copy Exon Coords</button></div><div class="kcp-sig-grid">${cards.join('')}</div>`;
      document.getElementById('kcp-left-title').textContent = `A. DEXSeq-Only Exon Volcano (${rows.length} exons · Click dot)`;
      document.getElementById('kcp-left-svg').innerHTML = renderTab4VolcanoSvg(rows, selIdx, C);
      document.getElementById('kcp-right-title').textContent = `B. ${sym}: Whole-Gene 5′→3′ DEXSeq Exon Body Profile`;
      document.getElementById('kcp-svg-track').innerHTML = renderTab4ProfileSvg(r, C);
      exonPills.innerHTML = `<span style="font-size:11px;color:#475569;">Coordinate: <code>${esc(coordCopy)}</code></span>`;
      stBox.innerHTML = ''; txWrap.style.display = 'none';
      wirePlotPoints(rows);
    }
  }

  function syncGeneDropdown(rows, selIdx) {
    geneSel.innerHTML = '';
    rows.forEach((r, idx) => {
      const o = document.createElement('option');
      o.value = String(idx);
      if (activeTab === 'class3_switches') {
        o.textContent = `${r.gene_symbol} · ${String(r.fate||'').replace(/\s*\([^)]*\)/g, '').trim()} · q=${fmtQ(r.gene_q)}`;
      } else if (activeTab === 'class1') {
        o.textContent = `${r.gene_symbol} · ${r.dexseq_exon} (${r.event_type}) · ΔΨ=${fmtS(r.rmats_dpsi,2)}`;
      } else if (activeTab === 'class2_af_al') {
        o.textContent = `${r.gene_symbol} (${r.etype}) · ΔΨ=${fmtS(r.dpsi,2)} · FDR=${fmtQ(r.bh_fdr)}`;
      } else {
        o.textContent = `${r.gene_symbol} · ${r.feature_id} · lfc=${fmtS(r.lfc,2)}`;
      }
      geneSel.appendChild(o);
    });
    if (rows.length && selIdx >= 0 && selIdx < rows.length) {
      geneSel.value = String(selIdx);
    }
  }

  function activeTabGeneSymbols(rows) {
    const C = CP.contrasts[activeCid];
    const q = (document.getElementById('kcp-search').value || '').toLowerCase().trim();
    const novel = document.getElementById('kcp-novel-only').checked;
    const strictSig = document.getElementById('kcp-strict-sig').checked;
    let fullList = null;
    if (!q && !novel) {
      const gl = C.gene_lists || {};
      if (activeTab === 'class2_af_al' && strictSig && gl.class2_af_al_fdr) {
        fullList = gl.class2_af_al_fdr;
      } else if (!(activeTab === 'class3_switches' && strictSig) && gl[activeTab]) {
        fullList = gl[activeTab];
      }
    }
    if (!fullList) {
      const seen = new Set();
      fullList = [];
      rows.forEach(r => {
        const s = String(r.gene_symbol || '').trim();
        if (s && !seen.has(s)) { seen.add(s); fullList.push(s); }
      });
    }
    return fullList;
  }

  function renderGeneList(rows, activeSym) {
    const C = CP.contrasts[activeCid];
    const syms = activeTabGeneSymbols(rows);
    const rowIdxBySym = new Map();
    rows.forEach((r, idx) => {
      if (r.gene_symbol && !rowIdxBySym.has(r.gene_symbol)) rowIdxBySym.set(r.gene_symbol, idx);
    });
    const fullName = (C.full_tsv_files && C.full_tsv_files[activeTab]) || '';
    const fullLink = document.getElementById('kcp-full-tsv');
    if (fullLink) {
      fullLink.href = fullName ? (`__TSV_PREFIX__` + fullName) : '#';
      fullLink.style.display = fullName ? 'inline-block' : 'none';
    }
    document.getElementById('kcp-tab-count').textContent = `${tabLabels[activeTab] || activeTab}: ${rows.length} rows in table · ${syms.length} unique genes`;
    document.getElementById('kcp-copy-genes').textContent = `Copy Gene List (${syms.length})`;
    document.getElementById('kcp-genelist-sum').textContent = `Complete Gene List — ${tabLabels[activeTab] || activeTab} (${syms.length} unique genes · click highlighted chip to inspect)`;

    const box = document.getElementById('kcp-genelist-box');
    const shown = syms.slice(0, 250);
    let html = shown.map(sym => {
      if (rowIdxBySym.has(sym)) {
        const idx = rowIdxBySym.get(sym);
        return `<button type="button" class="kcp-gchip clickable ${sym === activeSym ? 'active' : ''}" data-ridx="${idx}">${esc(sym)}</button>`;
      }
      return `<span class="kcp-gchip extra" title="In complete TSV (${esc(fullName)}); use Copy Gene List to export">${esc(sym)}</span>`;
    }).join('');
    if (syms.length > 250) {
      html += `<span style="font-size:11px;color:#64748b;padding:2px 6px;">+ ${syms.length - 250} more genes (use Copy Gene List / Gene List (.txt) / Complete TSV)</span>`;
    }
    box.innerHTML = syms.length ? html : '<span style="font-size:11px;color:#64748b;">No genes match current filter.</span>';
    const tKind = activeTab === 'class1' ? 'c1' : (activeTab === 'class2_af_al' ? 'c2' : (activeTab === 'class3_switches' ? 'c3' : 'c4'));
    box.querySelectorAll('.kcp-gchip.clickable').forEach(btn => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.ridx, 10);
        const r = rows[idx];
        if (!r) return;
        activeRowKey = `${activeTab}_${r.gene_base || ''}_${r.event_id || r.feature_id || idx}`;
        focusObj = { kind: tKind, row: r, ig: findIg(r.gene_symbol, r.gene_base), ridx: idx };
        renderStory();
        renderTable();
      });
    });
  }

  function renderContrast() {
    const C = CP.contrasts[activeCid];
    const cnt = C.counts;
    document.getElementById('kcp-bug-note').innerHTML = C.dexseq_sign_inverted
      ? `<span class="kcp-pill kcp-p-amb">DEXSeq Auto-Calibrated (${cnt.class1_raw_bug_pct}% raw → ${cnt.class1_concordant_pct}%)</span>`
      : `<span class="kcp-pill kcp-p-teal">DEXSeq Verified (${cnt.class1_concordant_pct}% concordant)</span>`;
    activeRowKey = '';
    updateStrictFilterUI();
    focusObj = defaultFocusForTab();
    renderTabs();
    renderStory();
    renderTable();
  }

  function formatCellHtml(k, v) {
    if (v === null || v === undefined) return '<td>—</td>';
    if (k === 'gene_q' || k === 'tx_up_q' || k === 'tx_dn_q' || k === 'deg_fdr' || k === 'rmats_fdr' || k === 'dexseq_padj' || k === 'pvalue' || k === 'bh_fdr' || k === 'padj') {
      return `<td>${sigBadge(v)}</td>`;
    }
    if (k === 'tx_confirm_tier') {
      const [lbl, cls] = tierInfo(v);
      return `<td><span class="kcp-pill ${cls}">${lbl}</span></td>`;
    }
    if (k === 'event_id') {
      const s = String(v);
      const disp = s.length > 28 ? (s.slice(0, 26) + '…') : s;
      return `<td title="${esc(s)}"><code>${esc(disp)}</code></td>`;
    }
    if (k === 'fate' || k === 'region_bin') {
      const s = String(v);
      const disp = s.replace(/\s*\([^)]*\)/g, '').trim();
      return `<td title="${esc(s)}">${esc(disp)}</td>`;
    }
    if (typeof v === 'boolean') return `<td>${v ? '✓' : '—'}</td>`;
    if (typeof v === 'number' && !Number.isInteger(v)) return `<td>${(Math.abs(v) > 0 && Math.abs(v) < 0.001) ? v.toExponential(1) : v.toFixed(3)}</td>`;
    return `<td>${esc(v)}</td>`;
  }

  function renderTable() {
    const C = CP.contrasts[activeCid];
    const { rows, cols } = filteredRowsAndCols();
    const tKind = activeTab === 'class1' ? 'c1' : (activeTab === 'class2_af_al' ? 'c2' : (activeTab === 'class3_switches' ? 'c3' : 'c4'));
    let h = `<table class="kcp-tbl"><thead><tr>${cols.map(c => `<th>${esc(c[1])}</th>`).join('')}</tr></thead><tbody>`;
    let selIdx = -1;
    rows.forEach((r, idx) => {
      const rKey = `${activeTab}_${r.gene_base || ''}_${r.event_id || r.feature_id || idx}`;
      const isSel = activeRowKey ? (activeRowKey === rKey) : (focusObj && focusObj.row && focusObj.row.gene_base === r.gene_base && (focusObj.row.event_id === r.event_id) && (focusObj.row.feature_id === r.feature_id));
      if (isSel && selIdx === -1) selIdx = idx;
      h += `<tr class="kcp-row ${isSel ? 'selected' : ''}" data-ridx="${idx}">` + cols.map(([k]) => formatCellHtml(k, r[k])).join('') + '</tr>';
    });
    const tblBox = document.getElementById('kcp-table-box');
    tblBox.innerHTML = rows.length ? (h + '</tbody></table>') : '<div style="padding:14px;color:#64748b;">No rows match current filter.</div>';
    syncGeneDropdown(rows, selIdx >= 0 ? selIdx : 0);
    const activeSym = (focusObj && focusObj.row && focusObj.row.gene_symbol) || (rows[selIdx >= 0 ? selIdx : 0] && rows[selIdx >= 0 ? selIdx : 0].gene_symbol) || '';
    renderGeneList(rows, activeSym);

    tblBox.querySelectorAll('tr.kcp-row').forEach(tr => tr.addEventListener('click', () => {
      const idx = parseInt(tr.dataset.ridx, 10);
      const r = rows[idx];
      if (!r) return;
      activeRowKey = `${activeTab}_${r.gene_base || ''}_${r.event_id || r.feature_id || idx}`;
      const ig = findIg(r.gene_symbol, r.gene_base);
      focusObj = { kind: tKind, row: r, ig, ridx: idx };
      renderStory();
      renderTable();
      document.getElementById('kcp-story-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }));
  }

  cidSel.addEventListener('change', () => { activeCid = cidSel.value; renderContrast(); });
  geneSel.addEventListener('change', () => {
    const { rows } = filteredRowsAndCols();
    const idx = parseInt(geneSel.value || '0', 10);
    const r = rows[idx];
    if (!r) return;
    activeRowKey = `${activeTab}_${r.gene_base || ''}_${r.event_id || r.feature_id || idx}`;
    const tKind = activeTab === 'class1' ? 'c1' : (activeTab === 'class2_af_al' ? 'c2' : (activeTab === 'class3_switches' ? 'c3' : 'c4'));
    focusObj = { kind: tKind, row: r, ig: findIg(r.gene_symbol, r.gene_base), ridx: idx };
    renderStory();
    renderTable();
  });
  document.getElementById('kcp-search').addEventListener('input', renderTable);
  document.getElementById('kcp-novel-only').addEventListener('change', () => { activeRowKey = ''; focusObj = defaultFocusForTab(); renderStory(); renderTable(); });
  document.getElementById('kcp-strict-sig').addEventListener('change', () => { activeRowKey = ''; focusObj = defaultFocusForTab(); renderStory(); renderTable(); });
  document.getElementById('kcp-copy-genes').addEventListener('click', () => {
    const { rows } = filteredRowsAndCols();
    const syms = activeTabGeneSymbols(rows);
    window._kcpCopyCoord(document.getElementById('kcp-copy-genes'), syms.join('\n'));
  });
  document.getElementById('kcp-dl-genes').addEventListener('click', () => {
    const { rows } = filteredRowsAndCols();
    const syms = activeTabGeneSymbols(rows);
    const blob = new Blob([syms.join('\n') + '\n'], { type: 'text/plain;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `genelist_${activeCid}_${activeTab}.txt`;
    a.click();
  });
  document.getElementById('kcp-export-tsv').addEventListener('click', () => {
    const { rows, cols } = filteredRowsAndCols();
    const lines = [cols.map(c => c[0]).join('\t')];
    rows.forEach(r => lines.push(cols.map(([k]) => String(r[k] ?? '')).join('\t')));
    const blob = new Blob([lines.join('\n')], { type: 'text/tab-separated-values;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `cross_paradigm_${activeCid}_${activeTab}.tsv`;
    a.click();
  });
  renderContrast();
})();
</script>
</section>
"""
    inner_body = template_body.replace("__SAFE_JSON__", safe_json).replace("__STANDALONE_REL__", safe_rel).replace("__TSV_PREFIX__", inline_tsv_prefix)
    standalone_body = template_body.replace("__SAFE_JSON__", safe_json).replace("__STANDALONE_REL__", safe_rel).replace("__TSV_PREFIX__", "")
    standalone_path.parent.mkdir(parents=True, exist_ok=True)
    full_doc = f"<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'><title>Katana-PRISM Cross-Paradigm Visual Explorer</title></head><body style='margin:0;padding:18px;background:#f8fafc;'>{standalone_body}</body></html>"
    standalone_path.write_text(full_doc, encoding="utf-8")
    return inner_body, full_doc


def integrate_cross_paradigm_into_report(
    *,
    cp_result: dict[str, Any],
    outdir: Path,
    panels: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    interactive_blocks: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Mount Cross-Paradigm exhibits into Chapters 04, 07, and 08 (never adds Ch09) and return lookup for Ch08."""
    if not cp_result or not cp_result.get("available"):
        return {}
    cp_out_dir = outdir / "outputs/cross_paradigm"
    fig_dir = outdir / "figures/cross_paradigm"
    cp_out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results_map: dict[str, Any] = cp_result.get("results", {})
    src_out = Path(cp_result["outdir"]) if cp_result.get("outdir") else None
    # Copy precomputed TSVs into report `outputs/cross_paradigm/` for self-contained REPORT.zip archiving
    if src_out is not None and src_out.resolve() != cp_out_dir.resolve() and src_out.is_dir():
        import shutil
        for fp in src_out.glob("*.tsv"):
            shutil.copy2(fp, cp_out_dir / fp.name)
        for fp in src_out.glob("*.json"):
            shutil.copy2(fp, cp_out_dir / fp.name)

    lookup: dict[str, dict[str, Any]] = {}
    for cid, c_res in results_map.items():
        slug = c_res["slug"]
        _render_cp_figs_for_contrast(c_res, cp_out_dir, fig_dir, outdir, panels)

        for aid, fn_s, label, df_obj in (
            ("cross_paradigm.class1", f"{slug}_class1_coord_verified.tsv", f"Class I 1-bp coordinate-verified exons ({cid})", c_res.get("_df_c1")),
            ("cross_paradigm.af_al", f"{slug}_class2_af_al_events.tsv", f"Class II SUPPA2 AF/AL boundary switches ({cid})", c_res.get("_af_al_nom")),
            ("cross_paradigm.isoform_switches", f"{slug}_class3_isoform_switches.tsv", f"Class III stageR full-length isoform switches ({cid})", c_res.get("_df_sw")),
            ("cross_paradigm.dexseq_only", f"{slug}_class4_dexseq_only.tsv", f"Class IV rMATS-blind DEXSeq exons ({cid})", c_res.get("_df_c4")),
        ):
            fp = cp_out_dir / fn_s
            if fp.is_file() and df_obj is not None and not df_obj.empty:
                tables.append({
                    "name": label, "analysis_id": aid, "chapter": 8, "contrast_id": cid,
                    "rows": len(df_obj), "columns": list(df_obj.columns),
                    "source": fp.relative_to(outdir).as_posix(),
                })

        # Build fast lookup per gene for Ch08 Candidate Workbench enrichment
        g_map: dict[str, dict[str, Any]] = {}
        df_c1 = c_res.get("_df_c1")
        if df_c1 is not None and not df_c1.empty:
            for _, r in df_c1.iterrows():
                gb = str(r["gene_base"])
                g_map.setdefault(gb, {})["coord_verified_exon"] = f"{r['dexseq_exon']} ({r['overlap_bp']}bp)"
                g_map.setdefault(str(r["gene_symbol"]), {})["coord_verified_exon"] = f"{r['dexseq_exon']} ({r['overlap_bp']}bp)"
        df_sw = c_res.get("_df_sw")
        if df_sw is not None and not df_sw.empty:
            for _, r in df_sw.iterrows():
                gb = str(r["gene_base"])
                gs = str(r["gene_symbol"])
                info = {"isoform_switch_fate": str(r["fate"]), "hidden_from_deg": bool(r["hidden_from_deg"])}
                g_map.setdefault(gb, {}).update(info)
                g_map.setdefault(gs, {}).update(info)
        lookup[cid] = g_map

    # Multi-contrast uplift figure in Ch04 when >= 2 contrasts
    df_bench = cp_result.get("bench_multi_contrast")
    if isinstance(df_bench, list):
        df_bench = pd.DataFrame(df_bench)
    if df_bench is not None and not df_bench.empty and len(df_bench) >= 2:
        plt.rcParams.update({"font.sans-serif": ["DejaVu Sans", "Arial"], "font.size": 9.5, "axes.titlesize": 11.2, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
        fig, ax = plt.subplots(figsize=(max(9.0, len(df_bench) * 2.2), 5.2), dpi=240)
        fig.patch.set_facecolor("white")
        cx = np.arange(len(df_bench))
        ax.bar(cx, df_bench["baseline_genes"], width=0.52, color=ED_NAVY, label="Baseline: DESeq2 DEG ∪ rMATS PS-DAS")
        ax.bar(cx, df_bench["unlocked_afal"], width=0.52, bottom=df_bench["baseline_genes"], color=ED_AMBER, label="+ SUPPA2 AF/AL (BH-FDR < 0.05)")
        b_dtu = df_bench["baseline_genes"] + df_bench["unlocked_afal"]
        ax.bar(cx, df_bench["unlocked_dtu"], width=0.52, bottom=b_dtu, color=ED_TEAL, label="+ stageR DTU (Isoform Switch)")
        b_dex = b_dtu + df_bench["unlocked_dtu"]
        ax.bar(cx, df_bench["unlocked_dexseq"], width=0.52, bottom=b_dex, color=ED_VIOLET, label="+ DEXSeq-Only Exons")
        ymax_b = max(10.0, float(df_bench["total_union"].max()) * 1.38)
        for i, r in df_bench.iterrows():
            ax.text(i, float(r["total_union"]) + ymax_b * 0.015, f"{int(r['total_union']):,}\n(+{r['uplift_pct']:.0f}%)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=ED_NAVY)
        ax.set_xticks(cx)
        ax.set_xticklabels([str(c).replace("_vs_", "\nvs ") for c in df_bench["contrast"]], fontsize=8.4)
        ax.set_ylim(0, ymax_b)
        ax.set_ylabel("Unique Significant Genes Captured", fontsize=9.5)
        ax.set_title("Multi-Contrast Gene Discovery Uplift Beyond Standard DEG + PS-DAS", loc="left", pad=12)
        ax.legend(loc="upper center", ncol=2, frameon=True, fontsize=8.0)
        png_b = fig_dir / "cp-multi-contrast-uplift.png"
        svg_b = _save_dual(fig, png_b)
        tsv_b = cp_out_dir / "multi_contrast_uplift.tsv"
        df_bench.to_csv(tsv_b, sep="\t", index=False)
        panels.append({
            "id": "cp-multi-contrast-uplift", "chapter": 4, "contrast_id": "all",
            "title": "Cross-paradigm multi-contrast discovery uplift beyond DEG + PS-DAS baseline",
            "caption": "Net new FDR-controlled (FDR < 0.05) genes unlocked per contrast by SUPPA2 AF/AL (BH-FDR < 0.05), stageR full-length DTU switches, and rMATS-blind DEXSeq exons.",
            "image": png_b.relative_to(outdir).as_posix(), "svg": svg_b.relative_to(outdir).as_posix(),
            "data_links": [{"role": "plot_data", "label": "Multi-contrast uplift (TSV)", "path": tsv_b.relative_to(outdir).as_posix(), "unit": "contrast", "contrast_id": "all", "policy": None}],
        })

    # Chapter 08 embedded interactive Cross-Paradigm Explorer
    json_for_ui = {
        "available": True,
        "contrasts": {cid: {k: v for k, v in val.items() if not k.startswith("_")} for cid, val in results_map.items()},
    }
    standalone_p = cp_out_dir / "CROSS_PARADIGM_EXPLORER.html"
    inline_html, _ = _render_ch8_cross_paradigm_explorer(json_for_ui, standalone_p, standalone_p.relative_to(outdir).as_posix())
    tables.append({
        "name": "Cross-Paradigm Isoform Switch & Blind-Spot Interactive Explorer",
        "analysis_id": "cross_paradigm.explorer", "chapter": 8, "contrast_id": "all",
        "rows": sum(len(v.get("interactive_genes", [])) for v in results_map.values()),
        "source": standalone_p.relative_to(outdir).as_posix(),
        "view": standalone_p.relative_to(outdir).as_posix(),
    })
    interactive_blocks.append({"chapter": 8, "id": "interactive-cross-paradigm-isoform-explorer", "html": inline_html})
    return lookup


def load_precomputed_cross_paradigm(cp_dir: Path) -> dict[str, Any]:
    """Load precomputed `KATANA_CROSS_PARADIGM` output directory for instant report fusion."""
    sum_p = cp_dir / "cross_paradigm_summary.json"
    if not sum_p.is_file():
        return {"available": False}
    data = json.loads(sum_p.read_text(encoding="utf-8"))
    if not data.get("available"):
        return {"available": False}
    res_map: dict[str, Any] = {}
    for cid, val in data.get("contrasts", {}).items():
        slug = val["slug"]
        entry = dict(val)
        entry["_df_c1"] = _read_tabular_safe(cp_dir / f"{slug}_class1_coord_verified.tsv", sep="\t")
        entry["_df_sup"] = _read_tabular_safe(cp_dir / f"{slug}_class2_suppa_7class.tsv", sep="\t")
        entry["_af_al_nom"] = _read_tabular_safe(cp_dir / f"{slug}_class2_af_al_events.tsv", sep="\t")
        entry["_df_sw"] = _read_tabular_safe(cp_dir / f"{slug}_class3_isoform_switches.tsv", sep="\t")
        entry["_df_c4"] = _read_tabular_safe(cp_dir / f"{slug}_class4_dexseq_only.tsv", sep="\t")
        res_map[cid] = entry
    bench_p = cp_dir / "multi_contrast_uplift.tsv"
    bench_df = _read_tabular_safe(bench_p, sep="\t") if bench_p.is_file() else pd.DataFrame(data.get("bench_multi_contrast", []))
    return {"available": True, "outdir": str(cp_dir), "results": res_map, "bench_multi_contrast": bench_df}


def cli_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="katana-cross-paradigm", description="Katana-PRISM Cross-Paradigm Resolver (rMATS × DEXSeq × SUPPA2 × stageR DTU × GTF)")
    parser.add_argument("--bundle", type=Path, required=True, help="Completed Nextflow katana_bundle (or results) directory")
    parser.add_argument("--gtf", type=Path, default=None, help="Annotation GTF (uncompressed or .gtf.gz)")
    parser.add_argument("--rmats-mode", choices=["JCEC", "JC"], default="JCEC")
    parser.add_argument("--fdr", type=float, default=0.05)
    parser.add_argument("--splicetrace-dirs", nargs="*", type=str, default=None)
    parser.add_argument("-o", "--outdir", type=Path, required=True)
    parser.add_argument("--emit-versions-yml", type=Path, default=None)
    parser.add_argument("--process-name", default="KATANA_CROSS_PARADIGM")
    args = parser.parse_args(argv)

    res = run_cross_paradigm_analysis(
        outdir=args.outdir,
        bundle_dir=args.bundle,
        gtf_path=args.gtf,
        splicetrace_dirs=args.splicetrace_dirs,
        rmats_mode=args.rmats_mode,
        fdr_cutoff=args.fdr,
    )
    if args.emit_versions_yml is not None:
        vp = Path(args.emit_versions_yml)
        vp.parent.mkdir(parents=True, exist_ok=True)
        vp.write_text(f'"{args.process_name}":\n    katana-cross-paradigm: "1.0.1"\n    python: "{platform.python_version()}"\n', encoding="utf-8")
    print(json.dumps({"available": res.get("available", False), "contrasts": len(res.get("results", {}))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
