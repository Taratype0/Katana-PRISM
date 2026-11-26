"""Automated GTF-driven ORF, Reading-Frame, NMD (50-nt EJC rule), and Curated UniProt/Pfam Domain Consequence Engine (1.0.0).

Key fixes & guarantees in 1.0.0:
  1. Strict `start_codon` + GTF column-8 `CDS phase (0/1/2)` frame alignment; excludes 5'/3'-incomplete
     GENCODE fragments (`cds_start_NF`, `cds_end_NF`, `mRNA_start_NF`) and pre-existing `nonsense_mediated_decay`
     biotypes whenever a complete `protein_coding` host transcript exists.
  2. Splice-junction-aware host transcript selection: ranks candidate `protein_coding` transcripts by exact
     exon-boundary agreement with the splicing event's flanking donor/acceptor sites (`u1`, `d0`), preventing
     accidental mid-exon clipping on multi-isoform giant genes (e.g. `Macf1` 7,355 aa, `Ptbp3` 552 aa, `Stxbp2` 593 aa).
  3. Bundled offline Human + Mouse Reviewed UniProtKB/Swiss-Prot + Pfam domain coordinate catalog
     (`194,738` curated intervals across `37,714` proteins in `data/uniprot_domains_human_mouse.tsv.gz`),
     mapping exact N-/C-anchored residue intervals onto both Inclusion and Exclusion protein isoforms to
     quantify in-frame domain indels AND C-terminal domain loss caused by frameshift/PTC truncation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import gzip
from importlib import resources
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

_ATTR_GENE_ID = re.compile(r'gene_id\s+"?([^";\s]+)')
_ATTR_GENE_NAME = re.compile(r'gene_name\s+"?([^";\s]+)')
_ATTR_TX_ID = re.compile(r'transcript_id\s+"?([^";\s]+)')
_ATTR_TX_TYPE = re.compile(r'transcript_(?:type|biotype)\s+"?([^";\s]+)')
_ATTR_CANON = re.compile(r'tag\s+"?(appris_principal[^";\s]*|Ensembl_canonical|MANE_Select|CCDS)')
_ATTR_INCOMPLETE = re.compile(r'tag\s+"?(cds_start_NF|cds_end_NF|mRNA_start_NF|mRNA_end_NF)')

_DOMAIN_CACHE: dict[str, dict[str, dict]] = {}
_SEQUENCE_CACHE: dict[str, str] | None = None

_PROTEIN_CONSEQUENCE_COLUMNS = [
    "event_id",
    "gene_id",
    "gene_symbol",
    "event_type",
    "dpsi",
    "host_transcript_id",
    "uniprot_acc",
    "coding_region_location",
    "frame_status",
    "nmd_switch_class",
    "aa_len_inclusion",
    "aa_len_exclusion",
    "delta_aa",
    "ptc_to_last_ejc_nt",
    "disrupted_protein_features",
    "functional_consequence_summary",
    "architecture_json",
]


def load_bundled_uniprot_domains(species: str = "auto") -> dict[str, dict]:
    """Load bundled 194,738-interval UniProt/Pfam catalog indexed by lowercase gene_symbol."""
    cache_key = species.lower()
    if cache_key in _DOMAIN_CACHE:
        return _DOMAIN_CACHE[cache_key]

    pkg_root = __name__.split(".")[0]
    data_ref = resources.files(f"{pkg_root}.data").joinpath("uniprot_domains_human_mouse.tsv.gz")
    with resources.as_file(data_ref) as gz_path:
        df = pd.read_csv(gz_path, sep="\t")

    if cache_key in ("mouse", "mus_musculus"):
        df = df[df["species"].eq("mouse")]
    elif cache_key in ("human", "homo_sapiens"):
        df = df[df["species"].eq("human")]

    pfam_col = "pfam_ accessions" if "pfam_ accessions" in df.columns else "pfam_accessions"
    index: dict[str, dict] = {}
    for g_low, acc, ulen, pfam, kind, s_aa, e_aa, lbl in zip(
        df["gene_lower"],
        df["uniprot_acc"],
        df["uniprot_len"],
        df[pfam_col] if pfam_col in df.columns else [""] * len(df),
        df["feature_kind"],
        df["start_aa"],
        df["end_aa"],
        df["feature_label"],
    ):
        gk = str(g_low)
        acc_s = str(acc)
        ent = index.get(gk)
        if ent is None:
            ent = {
                "uniprot_acc": acc_s,
                "uniprot_len": int(ulen),
                "pfam_ids": "" if pd.isna(pfam) else str(pfam),
                "features": [],
            }
            index[gk] = ent
        elif ent["uniprot_acc"] != acc_s:
            # Never merge intervals from a different UniProt accession/species into the same entry
            continue
        ent["features"].append({"kind": str(kind), "start": int(s_aa), "end": int(e_aa), "label": str(lbl)})
    _DOMAIN_CACHE[cache_key] = index
    return index


def load_bundled_uniprot_sequences() -> dict[str, str]:
    """Load bundled canonical Swiss-Prot amino-acid sequences for exact residue-level domain alignment."""
    global _SEQUENCE_CACHE
    if _SEQUENCE_CACHE is not None:
        return _SEQUENCE_CACHE
    pkg_root = __name__.split(".")[0]
    try:
        data_ref = resources.files(f"{pkg_root}.data").joinpath("uniprot_sequences_human_mouse.tsv.gz")
        with resources.as_file(data_ref) as gz_path:
            if not Path(gz_path).is_file():
                _SEQUENCE_CACHE = {}
                return _SEQUENCE_CACHE
            df = pd.read_csv(gz_path, sep="\t")
        _SEQUENCE_CACHE = {
            str(acc).strip(): str(seq).strip()
            for acc, seq in zip(df["uniprot_acc"], df["sequence"])
            if pd.notna(acc) and pd.notna(seq)
        }
    except Exception:
        _SEQUENCE_CACHE = {}
    return _SEQUENCE_CACHE


def _align_isoform_to_canonical(iso_seq: str, canon_seq: str) -> dict[int, int]:
    """Map 1-based canonical UniProt residue positions to 1-based translated isoform positions via sequence alignment."""
    li, lc = len(iso_seq), len(canon_seq)
    if li == 0 or lc == 0:
        return {}
    if iso_seq == canon_seq:
        return {i: i for i in range(1, lc + 1)}

    pre = 0
    min_l = min(li, lc)
    while pre < min_l and iso_seq[pre] == canon_seq[pre]:
        pre += 1
    suf = 0
    while suf < (min_l - pre) and iso_seq[li - 1 - suf] == canon_seq[lc - 1 - suf]:
        suf += 1

    blocks: list[tuple[int, int, int]] = []
    if pre > 0:
        blocks.append((0, 0, pre))
    mid_i = iso_seq[pre : li - suf]
    mid_c = canon_seq[pre : lc - suf]
    if mid_i and mid_c:
        sm = SequenceMatcher(None, mid_i, mid_c, autojunk=False)
        for b in sm.get_matching_blocks():
            if b.size > 0:
                blocks.append((pre + b.a, pre + b.b, b.size))
    if suf > 0:
        blocks.append((li - suf, lc - suf, suf))

    anchors = [b for b in blocks if b[2] >= 6]
    if not anchors and li < 30:
        anchors = [b for b in blocks if b[2] >= 4]

    c2i: dict[int, int] = {}
    for a0, b0, sz in blocks:
        keep = sz >= 6 or (
            sz >= 4
            and anchors
            and any(abs((a0 - aa) - (b0 - ab)) <= 50 and abs(b0 - ab) <= 200 for aa, ab, _ in anchors)
        )
        if keep:
            for off in range(sz):
                c2i[b0 + off + 1] = a0 + off + 1
    return c2i


@dataclass
class HostTranscriptModel:
    transcript_id: str
    gene_id: str
    gene_name: str
    chrom: str
    strand: str
    biotype: str
    exons: list[tuple[int, int]]
    cds_blocks: list[tuple[int, int]]     # sorted ascending genomic (start0, end0)
    cds_phases: dict[tuple[int, int], int]
    start_codon_5p: int | None            # exact 0-based genomic position of first base of start codon
    stop_codon_3p: int | None             # exact 0-based genomic position of 3'-most base of canonical stop codon
    is_canonical: bool
    is_incomplete: bool
    selenocysteine_blocks: list[tuple[int, int]] = field(default_factory=list)


def load_gtf_coding_transcripts(
    gtf_path: Path | str,
    target_genes: set[str] | None = None,
) -> dict[str, list[HostTranscriptModel]]:
    """Streaming GTF/GTF.gz parser honoring `start_codon`, `stop_codon`, `Selenocysteine`, `CDS` column-8 phase, biotypes, and `cds_start_NF` flags."""
    p = Path(gtf_path)
    opener = gzip.open if p.name.endswith(".gz") else open
    clean_targets: set[str] | None = None
    if target_genes:
        clean_targets = {g.split(".")[0].strip() for g in target_genes if g} | {g.lower().strip() for g in target_genes if g}

    tx_meta: dict[str, list] = {}  # [gid, gname, chrom, strand, biotype, is_can, is_inc]
    tx_exons: dict[str, list[tuple[int, int]]] = {}
    tx_cds: dict[str, list[tuple[int, int]]] = {}
    tx_phases: dict[str, dict[tuple[int, int], int]] = {}
    tx_start_bases: dict[str, list[tuple[int, int]]] = {}
    tx_stop_bases: dict[str, list[tuple[int, int]]] = {}
    tx_sec_bases: dict[str, list[tuple[int, int]]] = {}

    with opener(p, "rt", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line or line[0] == "#":
                continue
            parts = line.split("\t", 8)
            if len(parts) < 9:
                continue
            ftype = parts[2]
            if ftype not in ("transcript", "exon", "CDS", "start_codon", "stop_codon", "Selenocysteine"):
                continue
            attrs = parts[8]
            mg = _ATTR_GENE_ID.search(attrs)
            if not mg:
                continue
            gid = mg.group(1).split(".")[0]
            mn = _ATTR_GENE_NAME.search(attrs)
            gname = mn.group(1) if mn else gid

            if clean_targets is not None and (gid not in clean_targets and gname.lower() not in clean_targets):
                continue

            mt = _ATTR_TX_ID.search(attrs)
            if not mt:
                continue
            tid = mt.group(1).split(".")[0]
            start0 = int(parts[3]) - 1
            end0 = int(parts[4])
            if end0 <= start0:
                continue

            is_can = bool(_ATTR_CANON.search(attrs))
            is_inc = bool(_ATTR_INCOMPLETE.search(attrs))
            m_bt = _ATTR_TX_TYPE.search(attrs)
            bt = m_bt.group(1) if m_bt else "protein_coding"

            if tid not in tx_meta:
                tx_meta[tid] = [gid, gname, parts[0], parts[6], bt, is_can, is_inc]
            else:
                if is_can:
                    tx_meta[tid][5] = True
                if is_inc:
                    tx_meta[tid][6] = True

            if ftype == "exon":
                tx_exons.setdefault(tid, []).append((start0, end0))
            elif ftype == "CDS":
                tx_cds.setdefault(tid, []).append((start0, end0))
                ph = int(parts[7]) if parts[7] in ("0", "1", "2") else 0
                tx_phases.setdefault(tid, {})[(start0, end0)] = ph
            elif ftype == "start_codon":
                tx_start_bases.setdefault(tid, []).append((start0, end0))
            elif ftype == "stop_codon":
                tx_stop_bases.setdefault(tid, []).append((start0, end0))
            elif ftype == "Selenocysteine":
                tx_sec_bases.setdefault(tid, []).append((start0, end0))

    by_gene: dict[str, list[HostTranscriptModel]] = {}
    for tid, (gid, gname, chrom, strand, bt, is_can, is_inc) in tx_meta.items():
        ex = sorted(set(tx_exons.get(tid, ())))
        cds = sorted(set(tx_cds.get(tid, ())))
        if not ex:
            continue
        sc_blocks = sorted(set(tx_start_bases.get(tid, ())))
        sc_5p = (sc_blocks[0][0] if strand == "+" else (sc_blocks[-1][1] - 1)) if sc_blocks else None
        st_blocks = sorted(set(tx_stop_bases.get(tid, ())))
        stop_3p = ((st_blocks[-1][1] - 1) if strand == "+" else st_blocks[0][0]) if st_blocks else None
        sec_blocks = sorted(set(tx_sec_bases.get(tid, ())))
        model = HostTranscriptModel(
            transcript_id=tid,
            gene_id=gid,
            gene_name=gname,
            chrom=chrom,
            strand=strand,
            biotype=bt,
            exons=ex,
            cds_blocks=cds,
            cds_phases=tx_phases.get(tid, {}),
            start_codon_5p=sc_5p,
            stop_codon_3p=stop_3p,
            is_canonical=is_can,
            is_incomplete=is_inc,
            selenocysteine_blocks=sec_blocks,
        )
        by_gene.setdefault(gid, []).append(model)
        by_gene.setdefault(gname.lower(), []).append(model)

    for k in by_gene:
        by_gene[k].sort(
            key=lambda m: (
                len(m.cds_blocks) > 0 and m.biotype == "protein_coding" and not m.is_incomplete,
                m.start_codon_5p is not None and not m.is_incomplete,
                len(m.cds_blocks) > 0 and not m.is_incomplete,
                m.is_canonical,
                sum(b - a for a, b in m.cds_blocks),
                sum(b - a for a, b in m.exons),
            ),
            reverse=True,
        )
    return by_gene


def _merge_sorted_blocks(blocks: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(blocks):
        if b <= a:
            continue
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _rank_host_transcripts(candidates: list[HostTranscriptModel], rec: dict) -> list[HostTranscriptModel]:
    if not candidates:
        return []
    rec_strand = str(rec.get("strand") or "").strip()
    rec_chr = str(rec.get("chr") or "").strip()
    if (not rec_strand or not rec_chr) and ":" in str(rec.get("event_id") or ""):
        id_pts = str(rec["event_id"]).split(":")
        if len(id_pts) >= 3:
            rec_chr = rec_chr or id_pts[1].strip()
            rec_strand = rec_strand or id_pts[2].strip()
    if rec_strand in ("+", "-"):
        candidates = [m for m in candidates if m.strand == rec_strand]
        if not candidates:
            return []
    if rec_chr:
        norm_rc = rec_chr.lower().removeprefix("chr")
        candidates = [m for m in candidates if m.chrom.lower().removeprefix("chr") == norm_rc]
        if not candidates:
            return []
    kind = str(rec["event_type"])
    ev_coords = [
        int(rec[c])
        for c in ("exonStart_0base", "exonEnd", "longExonStart_0base", "longExonEnd", "1stExonStart_0base", "2ndExonEnd", "riExonStart_0base", "riExonEnd")
        if c in rec and pd.notna(rec[c])
    ]
    ev_lo, ev_hi = (min(ev_coords), max(ev_coords)) if ev_coords else (0, 0)

    # Flanking splice junctions & intervening-exon topology we want the host transcript to match
    if kind in ("SE", "MXE", "RI"):
        u1, d0 = int(rec["upstreamEE"]), int(rec["downstreamES"])
        anchor_ends = {u1}
        anchor_starts = {d0}
        allowed_mid: set[tuple[int, int]] = set()
        if kind == "SE":
            allowed_mid.add((int(rec["exonStart_0base"]), int(rec["exonEnd"])))
        elif kind == "MXE":
            allowed_mid.add((int(rec["1stExonStart_0base"]), int(rec["1stExonEnd"])))
            allowed_mid.add((int(rec["2ndExonStart_0base"]), int(rec["2ndExonEnd"])))
    else:
        f0, f1 = int(rec["flankingES"]), int(rec["flankingEE"])
        l0, l1 = int(rec["longExonStart_0base"]), int(rec["longExonEnd"])
        s0, s1 = int(rec["shortES"]), int(rec["shortEE"])
        u1, d0 = (min(s1, l1), f0) if l1 <= f0 else (f1, max(l0, s0))
        anchor_starts = {f0, l0, s0}
        anchor_ends = {f1, l1, s1}
        allowed_mid = set()

    def _score_cand(m: HostTranscriptModel) -> tuple:
        tx_lo, tx_hi = m.exons[0][0], m.exons[-1][1]
        overlaps = tx_lo <= ev_hi and tx_hi >= ev_lo
        starts_set = {a for a, _ in m.exons}
        ends_set = {b for _, b in m.exons}
        junc_matches = len(anchor_starts & starts_set) + len(anchor_ends & ends_set)
        mid_ex = [(a, b) for a, b in m.exons if a >= u1 and b <= d0]
        extra_mid = [eb - ea for ea, eb in mid_ex if (ea, eb) not in allowed_mid]
        if kind == "RI":
            spans_ri = any(ea <= u1 and eb >= d0 for ea, eb in m.exons)
            exact_topo = (junc_matches >= 2 and len(mid_ex) == 0) or spans_ri
        else:
            exact_topo = (junc_matches >= 2) and (len(extra_mid) == 0) and (len(mid_ex) <= 1)
        frame_safe_mid = (sum(extra_mid) % 3 == 0)
        clean_coding = len(m.cds_blocks) > 0 and m.biotype == "protein_coding" and not m.is_incomplete
        has_complete_orf = (m.start_codon_5p is not None) and (m.stop_codon_3p is not None) and not m.is_incomplete
        has_sc = m.start_codon_5p is not None and not m.is_incomplete
        return (
            overlaps and clean_coding and has_complete_orf and exact_topo,
            overlaps and clean_coding and exact_topo,
            overlaps and clean_coding and has_complete_orf and (junc_matches >= 2) and frame_safe_mid,
            overlaps and clean_coding and (junc_matches >= 2) and frame_safe_mid,
            overlaps and clean_coding and (junc_matches >= 2),
            overlaps and clean_coding and (junc_matches >= 1),
            has_complete_orf,
            has_sc,
            m.is_canonical,
            sum(b - a for a, b in m.cds_blocks),
        )

    return sorted(candidates, key=_score_cand, reverse=True)


def _build_full_isoform_exons(host_exons: list[tuple[int, int]], row: dict) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Splice local Inclusion vs Exclusion coordinates cleanly into full-length host exons without dropping flank exons."""
    kind = str(row["event_type"])
    strand = str(row["strand"])

    if kind in ("SE", "MXE"):
        u0, u1 = int(row["upstreamES"]), int(row["upstreamEE"])
        d0, d1 = int(row["downstreamES"]), int(row["downstreamEE"])
        left_exons = []
        right_exons = []
        found_u, found_d = False, False
        for a, b in host_exons:
            if b <= u1:
                if b == u1:
                    found_u = True
                left_exons.append((a, b))
            elif a < u1 < b:
                # Host exon extends past u1 -> trim strictly at u1 without expanding across upstream intron
                left_exons.append((a, u1))
                found_u = True
            if a >= d0:
                if a == d0:
                    found_d = True
                right_exons.append((a, b))
            elif a < d0 < b:
                # Host exon starts before d0 -> trim strictly at d0 without expanding across downstream intron
                right_exons.append((d0, b))
                found_d = True
        if not found_u:
            left_exons = [(a, b) for a, b in left_exons if b <= u0] + [(u0, u1)]
        if not found_d:
            right_exons = [(d0, d1)] + [(a, b) for a, b in right_exons if a >= d1]

        if kind == "SE":
            t0, t1 = int(row["exonStart_0base"]), int(row["exonEnd"])
            return _merge_sorted_blocks(left_exons + [(t0, t1)] + right_exons), _merge_sorted_blocks(left_exons + right_exons)
        a0, a1 = int(row["1stExonStart_0base"]), int(row["1stExonEnd"])
        b0, b1 = int(row["2ndExonStart_0base"]), int(row["2ndExonEnd"])
        inc_ex, exc_ex = ((a0, a1), (b0, b1)) if strand == "+" else ((b0, b1), (a0, a1))
        return _merge_sorted_blocks(left_exons + [inc_ex] + right_exons), _merge_sorted_blocks(left_exons + [exc_ex] + right_exons)

    if kind == "RI":
        u0, u1 = int(row["upstreamES"]), int(row["upstreamEE"])
        d0, d1 = int(row["downstreamES"]), int(row["downstreamEE"])
        left_ex = [(a, min(b, u1)) for a, b in host_exons if a < u1]
        right_ex = [(max(a, d0), b) for a, b in host_exons if b > d0]
        if not any(b == u1 for _, b in left_ex):
            left_ex.append((u0, u1))
        if not any(a == d0 for a, _ in right_ex):
            right_ex.append((d0, d1))
        exc = _merge_sorted_blocks(left_ex + right_ex)
        inc = _merge_sorted_blocks(left_ex + [(u1, d0)] + right_ex)
        return inc, exc

    # A3SS / A5SS
    l0, l1 = int(row["longExonStart_0base"]), int(row["longExonEnd"])
    s0, s1 = int(row["shortES"]), int(row["shortEE"])
    f0, f1 = int(row["flankingES"]), int(row["flankingEE"])
    if l1 <= f0:
        # Alternative exon is genomic-left of flanking exon
        # Preserve upstream start of the alternative exon if host exon started earlier than l0
        for a, b in host_exons:
            if a < l0 < b and b <= f0:
                l0, s0 = a, a
                break
        left_base = [(a, b) for a, b in host_exons if b <= l0]
        right_base = [(a, b) for a, b in host_exons if a >= f0]
        if not any(a == f0 for a, _ in right_base):
            right_base = [(f0, f1)] + [(a, b) for a, b in right_base if a >= f1]
        return _merge_sorted_blocks(left_base + [(l0, l1)] + right_base), _merge_sorted_blocks(left_base + [(s0, s1)] + right_base)
    else:
        # Flanking exon is genomic-left of alternative exon
        left_base = [(a, b) for a, b in host_exons if b <= f1]
        if not any(b == f1 for _, b in left_base):
            left_base = [(a, b) for a, b in left_base if b <= f0] + [(f0, f1)]
        for a, b in host_exons:
            if f1 <= a < l1 < b:
                l1, s1 = b, b
                break
        right_base = [(a, b) for a, b in host_exons if a >= l1]
        return _merge_sorted_blocks(left_base + [(l0, l1)] + right_base), _merge_sorted_blocks(left_base + [(s0, s1)] + right_base)


def _resolve_reading_frame_start_gpos(host: HostTranscriptModel) -> tuple[int, int]:
    """Return `(start_gpos, phase_offset)` guaranteeing exact codon-frame alignment."""
    if host.start_codon_5p is not None:
        return host.start_codon_5p, 0
    first_blk = host.cds_blocks[0] if host.strand == "+" else host.cds_blocks[-1]
    ph = host.cds_phases.get(first_blk, 0)
    gpos = first_blk[0] if host.strand == "+" else (first_blk[1] - 1)
    return gpos, ph


def _genomic_pos_to_tx_offset(exons_asc: list[tuple[int, int]], strand: str, gpos: int) -> int | None:
    ordered = exons_asc if strand == "+" else exons_asc[::-1]
    cursor = 0
    for a, b in ordered:
        if a <= gpos < b:
            return cursor + (gpos - a if strand == "+" else (b - 1 - gpos))
        cursor += b - a
    return None


def _selenocysteine_tx_starts(
    exons_asc: list[tuple[int, int]],
    strand: str,
    sec_blocks: list[tuple[int, int]],
) -> set[int]:
    """Project intact 3-nt GTF `Selenocysteine` codon spans onto 0-based spliced transcript offsets."""
    starts: set[int] = set()
    for s0, s1 in sec_blocks:
        if s1 - s0 != 3:
            continue
        b5 = s0 if strand == "+" else (s1 - 1)
        b3 = (s1 - 1) if strand == "+" else s0
        off5 = _genomic_pos_to_tx_offset(exons_asc, strand, b5)
        off3 = _genomic_pos_to_tx_offset(exons_asc, strand, b3)
        if off5 is not None and off3 == off5 + 2:
            starts.add(off5)
    return starts


def _terminal_extension_supported_by_exon(
    pre_exons: list[tuple[int, int]],
    post_exons: list[tuple[int, int]],
    old_tr: dict,
    new_tr: dict,
    candidate_models: list[HostTranscriptModel],
    strand: str,
) -> bool:
    """Require at least one annotated host-gene exon to continuously span from the unextended terminal exon through the healed stop codon."""
    if old_tr["stop_tx_nt"] is not None:
        return True
    if new_tr["stop_tx_nt"] is None:
        return False
    pos = int(new_tr["stop_tx_nt"]) - 1
    for a, b in (post_exons if strand == "+" else post_exons[::-1]):
        if pos < b - a:
            stop = a + pos if strand == "+" else b - 1 - pos
            break
        pos -= b - a
    else:
        return False
    a0, b0 = pre_exons[-1] if strand == "+" else pre_exons[0]
    lo, hi = min(a0, stop), max(b0, stop + 1)
    return any(x <= lo and hi <= y for mod in candidate_models for x, y in mod.exons)


def _translate_and_evaluate_nmd(
    dna_5to3: str,
    exon_lengths_5to3: list[int],
    start_tx_pos: int | None,
    phase_offset: int = 0,
    canonical_stop_tx_end: int | None = None,
    host_baseline_ptc_dist: int = 0,
    sec_codon_tx_starts: set[int] | None = None,
) -> dict:
    total_len = len(dna_5to3)
    sec_starts = sec_codon_tx_starts or set()
    ejc_positions = list(np.cumsum(exon_lengths_5to3)[:-1]) if len(exon_lengths_5to3) >= 2 else []
    last_ejc = int(ejc_positions[-1]) if ejc_positions else None

    if start_tx_pos is not None:
        eff_start = start_tx_pos + int(phase_offset)
    else:
        eff_start = None

    if eff_start is None or eff_start < 0 or eff_start + 3 > total_len:
        best_orf, best_stop_end, best_sec_cnt = "", None, 0
        for m in re.finditer(r"ATG", dna_5to3):
            st = m.start()
            aa_chars, stop_end, sec_cnt = [], None, 0
            for p in range(st, total_len - 2, 3):
                cod = dna_5to3[p : p + 3]
                if p in sec_starts and cod == "TGA":
                    aa_chars.append("U")
                    sec_cnt += 1
                    continue
                aa = CODON_TABLE.get(cod, "X")
                if aa == "*":
                    stop_end = p + 3
                    break
                aa_chars.append(aa)
            if len(aa_chars) > len(best_orf):
                best_orf, best_stop_end, best_sec_cnt = "".join(aa_chars), stop_end, sec_cnt
        dist_ejc = int(last_ejc - best_stop_end) if (best_stop_end is not None and last_ejc is not None) else 0
        is_native_stop = canonical_stop_tx_end is not None and best_stop_end == canonical_stop_tx_end and dist_ejc <= max(host_baseline_ptc_dist + 6, 49)
        nmd = bool(best_stop_end is not None and last_ejc is not None and dist_ejc >= 50 and not is_native_stop)
        return {
            "protein": best_orf,
            "aa_len": len(best_orf),
            "stop_tx_nt": best_stop_end,
            "last_ejc_tx_nt": last_ejc,
            "mrna_len_nt": total_len,
            "nmd_sensitive": nmd,
            "ptc_to_last_ejc_nt": max(0, dist_ejc),
            "start_preserved": False,
            "sec_translated": best_sec_cnt,
        }

    aa_list: list[str] = []
    stop_end: int | None = None
    sec_cnt = 0
    for p in range(eff_start, total_len - 2, 3):
        cod = dna_5to3[p : p + 3]
        if p in sec_starts and cod == "TGA":
            aa_list.append("U")
            sec_cnt += 1
            continue
        aa = CODON_TABLE.get(cod, "X")
        if aa == "*":
            stop_end = p + 3
            break
        aa_list.append(aa)
    prot = "".join(aa_list)
    dist_ejc = int(last_ejc - stop_end) if (stop_end is not None and last_ejc is not None) else 0
    is_native_stop = canonical_stop_tx_end is not None and stop_end == canonical_stop_tx_end and dist_ejc <= max(host_baseline_ptc_dist + 6, 49)
    nmd = bool(stop_end is not None and last_ejc is not None and dist_ejc >= 50 and not is_native_stop)
    return {
        "protein": prot,
        "aa_len": len(prot),
        "stop_tx_nt": stop_end,
        "last_ejc_tx_nt": last_ejc,
        "mrna_len_nt": total_len,
        "nmd_sensitive": nmd,
        "ptc_to_last_ejc_nt": max(0, dist_ejc),
        "start_preserved": True,
        "sec_translated": sec_cnt,
    }


def _map_uniprot_domains_to_isoforms(
    gene_symbol: str,
    prot_inc: str,
    prot_exc: str,
    nmd_inc: bool,
    nmd_exc: bool,
    domain_db: dict[str, dict],
) -> tuple[str, str, list[dict]]:
    """Map curated UniProt/Pfam intervals onto Inclusion vs Exclusion ORFs and classify intact vs spliced vs PTC-lost."""
    entry = domain_db.get(gene_symbol.lower())
    len_inc, len_exc = len(prot_inc), len(prot_exc)
    ref_len = max(len_inc, len_exc)
    if ref_len == 0:
        return "NA", "none", []

    # Exact shared N-terminal prefix & C-terminal suffix between Inclusion and Exclusion proteins
    pre = 0
    min_l = min(len_inc, len_exc)
    while pre < min_l and prot_inc[pre] == prot_exc[pre]:
        pre += 1
    suf = 0
    while suf < (min_l - pre) and prot_inc[len_inc - 1 - suf] == prot_exc[len_exc - 1 - suf]:
        suf += 1

    # Altered window on the longer reference protein (1-based amino acid coordinates)
    alt_start_aa = pre + 1
    alt_end_aa = max(alt_start_aa, ref_len - suf)

    mapped_arch: list[dict] = []
    disrupted_labels: list[str] = []
    uniprot_acc = "predicted"

    if entry is not None and entry["features"]:
        uniprot_acc = entry["uniprot_acc"]
        u_len = max(entry["uniprot_len"], 1)
        u_seq = entry.get("uniprot_seq")
        if u_seq is None:
            cand_seq = load_bundled_uniprot_sequences().get(uniprot_acc)
            if cand_seq is not None and abs(len(cand_seq) - u_len) <= 2:
                u_seq = cand_seq

        if u_seq is not None and len(u_seq) > 0:
            c2i_inc = _align_isoform_to_canonical(prot_inc, u_seq)
            c2i_exc = _align_isoform_to_canonical(prot_exc, u_seq)
            best_aligned = max(len(c2i_inc), len(c2i_exc))
            min_anchor_need = min(20, max(6, int(0.25 * min(ref_len, len(u_seq)))))
            if best_aligned >= min_anchor_need:
                canon_is_inc = len(c2i_inc) >= len(c2i_exc)
                for feat in entry["features"]:
                    u_s, u_e = int(feat["start"]), int(feat["end"])
                    dom_span = max(1, u_e - u_s + 1)
                    pos_inc = [u for u in range(u_s, u_e + 1) if u in c2i_inc]
                    pos_exc = [u for u in range(u_s, u_e + 1) if u in c2i_exc]
                    cov_inc = len(pos_inc)
                    cov_exc = len(pos_exc)
                    cov_max = max(cov_inc, cov_exc)
                    cov_min = min(cov_inc, cov_exc)

                    # Guard against isoforms that completely lack an N-terminal/alternative canonical region (e.g., F7 382-aa vs Gla [61-105])
                    if cov_max == 0 or (cov_max < 4 and (cov_max / dom_span) < 0.15) or (cov_max / dom_span) < 0.10:
                        continue

                    if canon_is_inc and pos_inc:
                        s_map, e_map = c2i_inc[pos_inc[0]], c2i_inc[pos_inc[-1]]
                    elif (not canon_is_inc) and pos_exc:
                        s_map, e_map = c2i_exc[pos_exc[0]], c2i_exc[pos_exc[-1]]
                    elif pos_inc:
                        s_map, e_map = c2i_inc[pos_inc[0]], c2i_inc[pos_inc[-1]]
                    else:
                        s_map, e_map = c2i_exc[pos_exc[0]], c2i_exc[pos_exc[-1]]
                    if s_map > e_map:
                        s_map, e_map = e_map, s_map

                    pos_both = [u for u in pos_inc if u in c2i_exc]
                    is_internal_ins = False
                    if len(pos_both) >= 2 and cov_inc == cov_exc and len_inc != len_exc:
                        span_inc = abs(c2i_inc[pos_both[-1]] - c2i_inc[pos_both[0]])
                        span_exc = abs(c2i_exc[pos_both[-1]] - c2i_exc[pos_both[0]])
                        is_internal_ins = (span_inc != span_exc)

                    changed_cov = cov_inc != cov_exc
                    if prot_inc != prot_exc and (changed_cov or is_internal_ins) and not (nmd_inc and nmd_exc):
                        if (nmd_inc or nmd_exc) and cov_min < cov_max:
                            prod_cov, nmd_cov = (cov_exc, cov_inc) if nmd_inc else (cov_inc, cov_exc)
                            if nmd_cov < prod_cov:
                                state = "lost_by_PTC_truncation" if nmd_cov == 0 else "truncated_by_PTC"
                                pct = min(100, max(1, int(round(100.0 * (prod_cov - nmd_cov) / prod_cov))))
                                tag_dir = f"{pct}% PTC"
                            else:
                                state = "exclusive_to_NMD_isoform"
                                tag_dir = "NMD-isoform"
                        elif is_internal_ins and not changed_cov:
                            ins_len = abs(len_inc - len_exc)
                            state = "inserted_by_inclusion" if len_inc > len_exc else "inserted_by_exclusion"
                            tag_dir = f"+{ins_len} aa ins"
                        elif cov_inc > cov_exc and suf > 0:
                            state = "inserted_by_inclusion"
                            pct = min(100, max(1, int(round(100.0 * (cov_inc - cov_exc) / dom_span))))
                            tag_dir = f"{pct}%"
                        elif suf == 0 and cov_min < cov_max:
                            state = (
                                "canonical_C_tail_replaced_by_inclusion"
                                if cov_inc < cov_exc
                                else "canonical_C_tail_replaced_by_exclusion"
                            )
                            tag_dir = "replaced_C_tail"
                        else:
                            state = "spliced_indel_overlap"
                            pct = min(100, max(1, int(round(100.0 * abs(cov_inc - cov_exc) / dom_span))))
                            tag_dir = f"{pct}%"
                        if feat["kind"] != "REGION" or "Disordered" not in feat["label"]:
                            disrupted_labels.append(f"{feat['label']} [{s_map}-{e_map} aa, {tag_dir}]")
                        elif len(disrupted_labels) == 0:
                            disrupted_labels.append(f"{feat['label']} [{s_map}-{e_map} aa]")
                    else:
                        state = "intact"
                    mapped_arch.append({
                        "kind": feat["kind"],
                        "label": feat["label"],
                        "start": int(s_map),
                        "end": int(e_map),
                        "uniprot_start": u_s,
                        "uniprot_end": u_e,
                        "cov_inc_aa": cov_inc,
                        "cov_exc_aa": cov_exc,
                        "alignment_method": "residue_sequence_alignment",
                        "state": state,
                    })
        else:
            # Fallback only for synthetic test/mock entries or unbundled entries with exact/near-exact length agreement (no length scaling)
            canon_is_inc = abs(len_inc - u_len) <= abs(len_exc - u_len)
            canon_prot = prot_inc if canon_is_inc else prot_exc
            canon_len = max(len(canon_prot), 1)
            c_alt_start = pre + 1
            c_alt_end = canon_len - suf
            if abs(canon_len - u_len) <= 10:
                for feat in entry["features"]:
                    s_map = max(1, min(canon_len, feat["start"]))
                    e_map = max(s_map, min(canon_len, feat["end"]))
                    ov = max(0, min(e_map, c_alt_end) - max(s_map, c_alt_start) + 1) if c_alt_end >= c_alt_start else 0
                    dom_span = max(1, e_map - s_map + 1)
                    is_internal_ins = (c_alt_end < c_alt_start) and (s_map <= pre) and (e_map >= pre + 1)
                    if prot_inc != prot_exc and (ov >= 1 or is_internal_ins) and not (nmd_inc and nmd_exc):
                        if (nmd_inc or nmd_exc) and e_map > min_l:
                            state = "lost_by_PTC_truncation" if s_map > min_l else "truncated_by_PTC"
                            pct = min(100, int(round(100.0 * max(1, e_map - max(s_map - 1, min_l)) / dom_span)))
                            tag_dir = f"{pct}% PTC"
                        elif is_internal_ins:
                            ins_len = abs(len_inc - len_exc)
                            state = "inserted_by_inclusion" if len_inc > len_exc else "inserted_by_exclusion"
                            tag_dir = f"+{ins_len} aa ins"
                        elif canon_is_inc and len_inc > len_exc and suf > 0:
                            state = "inserted_by_inclusion"
                            pct = min(100, int(round(100.0 * ov / dom_span)))
                            tag_dir = f"{pct}%"
                        elif (not canon_is_inc) and suf == 0:
                            state = "canonical_C_tail_replaced_by_inclusion"
                            tag_dir = "replaced_C_tail"
                        else:
                            state = "spliced_indel_overlap"
                            pct = min(100, int(round(100.0 * ov / dom_span)))
                            tag_dir = f"{pct}%"
                        if feat["kind"] != "REGION" or "Disordered" not in feat["label"]:
                            disrupted_labels.append(f"{feat['label']} [{s_map}-{e_map} aa, {tag_dir}]")
                        elif len(disrupted_labels) == 0:
                            disrupted_labels.append(f"{feat['label']} [{s_map}-{e_map} aa]")
                    else:
                        state = "intact"
                    mapped_arch.append({
                        "kind": feat["kind"],
                        "label": feat["label"],
                        "start": s_map,
                        "end": e_map,
                        "state": state,
                    })

    dedup_disrupted = list(dict.fromkeys(disrupted_labels))[:5]
    feat_str = "; ".join(dedup_disrupted) if dedup_disrupted else "none"
    return uniprot_acc, feat_str, mapped_arch


def _strict_int_coord(rec: dict, col: str) -> int:
    if col not in rec:
        raise ValueError(f"Missing required coordinate column '{col}'.")
    val = rec[col]
    try:
        if val is None or pd.isna(val):
            raise ValueError(f"Missing coordinate in column '{col}'.")
    except (TypeError, ValueError):
        if isinstance(val, (list, tuple, dict)):
            raise ValueError(f"Invalid coordinate in column '{col}'.")
    if isinstance(val, bool):
        raise ValueError(f"Boolean coordinate not allowed in '{col}'.")
    try:
        fv = float(str(val).strip()) if isinstance(val, str) else float(val)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric coordinate {val!r} in column '{col}'.") from exc
    if (not np.isfinite(fv)) or fv != np.floor(fv) or fv < 0:
        raise ValueError(f"Non-integer or negative coordinate {val!r} in column '{col}'.")
    return int(round(fv))


def _validate_event_geometry_and_bounds(rec: dict, reader, stream) -> dict:
    """Validate event coordinates/geometry strictly and verify FASTA contig presence, bounds, and .fai integrity."""
    rec_out = dict(rec)
    eid = str(rec_out.get("event_id") or "")
    etype = str(rec_out.get("event_type") or "").strip()
    strand = str(rec_out.get("strand") or "").strip()
    chrom = str(rec_out.get("chr") or "").strip()
    if (not chrom or not strand) and ":" in eid:
        id_pts = eid.split(":")
        if len(id_pts) >= 3:
            chrom = chrom or id_pts[1].strip()
            strand = strand or id_pts[2].strip()
    if etype not in ("SE", "A3SS", "A5SS", "MXE", "RI") or strand not in ("+", "-"):
        raise ValueError(f"Unsupported event_type={etype!r} or strand={strand!r} for event {eid!r}.")
    if not chrom:
        raise ValueError(f"Missing chromosome for event {eid!r}.")

    coord_map: dict[str, int] = {}
    if etype == "SE":
        for c in ("upstreamES", "upstreamEE", "exonStart_0base", "exonEnd", "downstreamES", "downstreamEE"):
            coord_map[c] = _strict_int_coord(rec_out, c)
        u0, u1 = coord_map["upstreamES"], coord_map["upstreamEE"]
        t0, t1 = coord_map["exonStart_0base"], coord_map["exonEnd"]
        d0, d1 = coord_map["downstreamES"], coord_map["downstreamEE"]
        if not (u0 < u1 < t0 < t1 < d0 < d1):
            raise ValueError(
                f"Invalid SE geometry for {eid}: expected u0 < u1 < exonStart < exonEnd < d0 < d1, "
                f"got ({u0},{u1}), ({t0},{t1}), ({d0},{d1})."
            )
    elif etype == "MXE":
        for c in (
            "upstreamES", "upstreamEE", "1stExonStart_0base", "1stExonEnd",
            "2ndExonStart_0base", "2ndExonEnd", "downstreamES", "downstreamEE",
        ):
            coord_map[c] = _strict_int_coord(rec_out, c)
        u0, u1 = coord_map["upstreamES"], coord_map["upstreamEE"]
        a0, a1 = coord_map["1stExonStart_0base"], coord_map["1stExonEnd"]
        b0, b1 = coord_map["2ndExonStart_0base"], coord_map["2ndExonEnd"]
        d0, d1 = coord_map["downstreamES"], coord_map["downstreamEE"]
        if not (u0 < u1 < a0 < a1 <= b0 < b1 < d0 < d1):
            raise ValueError(
                f"Invalid MXE exon ordering for {eid}: expected u0 < u1 < 1st < 2nd < d0 < d1."
            )
    elif etype == "RI":
        for c in ("upstreamES", "upstreamEE", "downstreamES", "downstreamEE", "riExonStart_0base", "riExonEnd"):
            coord_map[c] = _strict_int_coord(rec_out, c)
        u0, u1 = coord_map["upstreamES"], coord_map["upstreamEE"]
        d0, d1 = coord_map["downstreamES"], coord_map["downstreamEE"]
        ri0, ri1 = coord_map["riExonStart_0base"], coord_map["riExonEnd"]
        if not (u0 < u1 < d0 < d1) or (ri0, ri1) != (u0, d1):
            raise ValueError(
                f"Invalid RI geometry for {eid}: riExon ({ri0},{ri1}) disagrees with flanks ({u0},{u1})-({d0},{d1})."
            )
    else:
        for c in ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"):
            coord_map[c] = _strict_int_coord(rec_out, c)
        l0, l1 = coord_map["longExonStart_0base"], coord_map["longExonEnd"]
        s0, s1 = coord_map["shortES"], coord_map["shortEE"]
        f0, f1 = coord_map["flankingES"], coord_map["flankingEE"]
        if not (l0 < l1 and s0 < s1 and f0 < f1):
            raise ValueError(f"Empty/reversed A3SS/A5SS exon interval for {eid}.")

    rec_out.update(coord_map)
    contig = reader.resolve_chrom(chrom)
    if hasattr(reader, "_validate_contig_index"):
        reader._validate_contig_index(stream, contig)
    seq_len = int(reader.index[contig][0])
    max_coord = max(coord_map.values())
    if max_coord > seq_len:
        raise ValueError(
            f"Event {eid} coordinate {max_coord} exceeds FASTA contig {contig!r} length {seq_len}."
        )
    return rec_out


def _classify_nmd_pair(t_inc: dict, t_exc: dict) -> tuple[str, int]:
    if t_inc["stop_tx_nt"] is None or t_exc["stop_tx_nt"] is None:
        return "incomplete_3prime_CDS_or_nonstop", 0
    if t_inc["nmd_sensitive"] and not t_exc["nmd_sensitive"]:
        return "inclusion_triggers_NMD_poison_exon", int(t_inc["ptc_to_last_ejc_nt"])
    if t_exc["nmd_sensitive"] and not t_inc["nmd_sensitive"]:
        return "skipping_triggers_NMD_decay", int(t_exc["ptc_to_last_ejc_nt"])
    if not t_inc["nmd_sensitive"] and not t_exc["nmd_sensitive"]:
        return ("productive_protein_isoform_switch" if t_inc["protein"] != t_exc["protein"] else "coding_silent_utr_switch"), 0
    return "dual_NMD_background", int(max(t_inc["ptc_to_last_ejc_nt"], t_exc["ptc_to_last_ejc_nt"]))


def trace_protein_and_nmd_consequences(
    events_df: pd.DataFrame,
    gtf_path: Path | str,
    fasta_path: Path | str,
    *,
    indexed_fasta_cls: type,
    species: str = "auto",
    require_annotated_a3_a5_pair: bool = True,
    selenocysteine_mode: str = "conditional",
) -> pd.DataFrame:
    """Deterministic GTF + FASTA + Curated UniProt/Pfam ORF, reading-frame, NMD, and protein-domain consequence engine."""
    if selenocysteine_mode not in ("conditional", "recode", "ignore"):
        raise ValueError(f"Unsupported selenocysteine_mode={selenocysteine_mode!r}; expected 'conditional', 'recode', or 'ignore'.")
    if events_df.empty:
        return pd.DataFrame(columns=_PROTEIN_CONSEQUENCE_COLUMNS)

    target_genes = (
        set(events_df["gene_id"].dropna().astype(str))
        | set(events_df.get("gene_symbol", pd.Series(dtype=str)).dropna().astype(str))
        | set(events_df.get("gene_name", pd.Series(dtype=str)).dropna().astype(str))
    )
    tx_index = load_gtf_coding_transcripts(gtf_path, target_genes=target_genes)
    eff_species = str(species or "auto").lower()
    if eff_species == "auto":
        gids_str = events_df["gene_id"].dropna().astype(str) if "gene_id" in events_df.columns else pd.Series(dtype=str)
        eff_species = "mouse" if gids_str.str.upper().str.startswith("ENSMUSG").any() else "human"
    domain_db = load_bundled_uniprot_domains(species=eff_species)
    reader = indexed_fasta_cls(Path(fasta_path))

    rows: list[dict] = []
    with reader.open_stream() as stream:
        for raw_rec in events_df.to_dict("records"):
            rec = _validate_event_geometry_and_bounds(raw_rec, reader, stream)
            eid = str(rec["event_id"])
            gid = str(rec.get("gene_id", "")).split(".")[0]
            raw_sym = rec.get("gene_symbol")
            if raw_sym is None or pd.isna(raw_sym) or str(raw_sym).strip() == "":
                raw_sym = rec.get("gene_name")
            gsym = str(raw_sym).strip() if (raw_sym is not None and pd.notna(raw_sym) and str(raw_sym).strip() != "") else gid
            dpsi = float(rec.get("dpsi", 0.0))
            etype = str(rec["event_type"])
            strand = str(rec["strand"])

            candidates = tx_index.get(gid) or tx_index.get(gsym.lower()) or []
            ranked_hosts = _rank_host_transcripts(candidates, rec)
            host = ranked_hosts[0] if ranked_hosts else None
            if host is not None and (gsym == gid or gsym.startswith(("ENSG", "ENSMUSG"))) and host.gene_name:
                gsym = host.gene_name

            conditional_novel_a3_a5 = False
            if etype in ("A3SS", "A5SS"):
                try:
                    l0_a, l1_a = int(rec["longExonStart_0base"]), int(rec["longExonEnd"])
                    s0_a, s1_a = int(rec["shortES"]), int(rec["shortEE"])
                    f0_a, f1_a = int(rec["flankingES"]), int(rec["flankingEE"])
                    if (etype == "A3SS" and strand == "+") or (etype == "A5SS" and strand == "-"):
                        dir_ok = (l0_a < s0_a < s1_a) and (l1_a == s1_a) and (f0_a < f1_a < l0_a)
                        pair_l = ((f0_a, f1_a), (l0_a, l1_a))
                        pair_s = ((f0_a, f1_a), (s0_a, s1_a))
                    elif (etype == "A3SS" and strand == "-") or (etype == "A5SS" and strand == "+"):
                        dir_ok = (l0_a == s0_a) and (l0_a < s1_a < l1_a) and (l1_a < f0_a < f1_a)
                        pair_l = ((l0_a, l1_a), (f0_a, f1_a))
                        pair_s = ((s0_a, s1_a), (f0_a, f1_a))
                    else:
                        dir_ok = False
                        pair_l = pair_s = None
                except (KeyError, TypeError, ValueError):
                    dir_ok = False
                    pair_l = pair_s = None

                if (not dir_ok) or (not ranked_hosts):
                    rows.append({
                        "event_id": eid,
                        "gene_id": gid,
                        "gene_symbol": gsym,
                        "event_type": etype,
                        "dpsi": dpsi,
                        "host_transcript_id": host.transcript_id if host else "unmatched",
                        "uniprot_acc": "NA",
                        "coding_region_location": "complex_junction_usage",
                        "frame_status": "complex_junction_usage",
                        "nmd_switch_class": "complex_junction_usage",
                        "aa_len_inclusion": 0,
                        "aa_len_exclusion": 0,
                        "delta_aa": 0,
                        "ptc_to_last_ejc_nt": 0,
                        "disrupted_protein_features": "none",
                        "functional_consequence_summary": (
                            "Excluded: A3SS/A5SS event fails biological contig, strand, or splice-direction geometry"
                        ),
                        "architecture_json": "{}",
                    })
                    continue

                adj_by_gene: dict[str, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
                for m in ranked_hosts:
                    adj_by_gene.setdefault(m.gene_id, set()).update(zip(m.exons, m.exons[1:]))
                valid_pair_gids = {g for g, pairs in adj_by_gene.items() if (pair_l in pairs) and (pair_s in pairs)}
                has_annotated_pair = bool(valid_pair_gids)
                if has_annotated_pair:
                    ranked_hosts = [m for m in ranked_hosts if m.gene_id in valid_pair_gids]
                    if host is None or host.gene_id not in valid_pair_gids:
                        host = ranked_hosts[0]
                if not has_annotated_pair:
                    if require_annotated_a3_a5_pair:
                        rows.append({
                            "event_id": eid,
                            "gene_id": gid,
                            "gene_symbol": gsym,
                            "event_type": etype,
                            "dpsi": dpsi,
                            "host_transcript_id": host.transcript_id if host else "unmatched",
                            "uniprot_acc": "NA",
                            "coding_region_location": "complex_junction_usage",
                            "frame_status": "complex_junction_usage",
                            "nmd_switch_class": "complex_junction_usage",
                            "aa_len_inclusion": 0,
                            "aa_len_exclusion": 0,
                            "delta_aa": 0,
                            "ptc_to_last_ejc_nt": 0,
                            "disrupted_protein_features": "none",
                            "functional_consequence_summary": (
                                "Excluded: A3SS/A5SS arms lack an annotated adjacent flank-exon pair in GTF "
                                "(pass require_annotated_a3_a5_pair=False for conditional novel-site modeling)"
                            ),
                            "architecture_json": "{}",
                        })
                        continue
                    conditional_novel_a3_a5 = True

            if host is None:
                rows.append({
                    "event_id": eid,
                    "gene_id": gid,
                    "gene_symbol": gsym,
                    "event_type": etype,
                    "dpsi": dpsi,
                    "host_transcript_id": "unmatched",
                    "uniprot_acc": "NA",
                    "coding_region_location": "unannotated_host",
                    "frame_status": "annotation_unmatched",
                    "nmd_switch_class": "unannotated",
                    "aa_len_inclusion": 0,
                    "aa_len_exclusion": 0,
                    "delta_aa": 0,
                    "ptc_to_last_ejc_nt": 0,
                    "disrupted_protein_features": "none",
                    "functional_consequence_summary": "Unmatched host gene/transcript in reference GTF annotation",
                    "architecture_json": "{}",
                })
                continue

            if not host.cds_blocks:
                rows.append({
                    "event_id": eid,
                    "gene_id": gid,
                    "gene_symbol": gsym,
                    "event_type": etype,
                    "dpsi": dpsi,
                    "host_transcript_id": host.transcript_id,
                    "uniprot_acc": "NA",
                    "coding_region_location": "noncoding_host",
                    "frame_status": "noncoding_transcript",
                    "nmd_switch_class": "noncoding",
                    "aa_len_inclusion": 0,
                    "aa_len_exclusion": 0,
                    "delta_aa": 0,
                    "ptc_to_last_ejc_nt": 0,
                    "disrupted_protein_features": "none",
                    "functional_consequence_summary": "Non-coding transcript isoform change",
                    "architecture_json": "{}",
                })
                continue

            coding_pool = [m for m in ranked_hosts if m.cds_blocks and m.gene_id == host.gene_id][:6]
            # Gene-wide furthest 3' coding/UTR boundary to heal clipped 3'-terminal exon annotations (e.g. ENST00000538290 vs ENST00000302035)
            gene_3p_min = min(m.exons[0][0] for m in coding_pool)
            gene_3p_max = max(m.exons[-1][1] for m in coding_pool)
            chosen_bundle = None
            best_bundle_rank = (-1, -1)
            for cand_m in coding_pool:
                c_inc_ex, c_exc_ex = _build_full_isoform_exons(cand_m.exons, rec)
                c_ord_inc = c_inc_ex if strand == "+" else c_inc_ex[::-1]
                c_ord_exc = c_exc_ex if strand == "+" else c_exc_ex[::-1]
                c_dna_inc = reader.fetch_blocks(stream, cand_m.chrom, strand, c_ord_inc)
                c_dna_exc = reader.fetch_blocks(stream, cand_m.chrom, strand, c_ord_exc)
                c_start_gpos, c_phase_off = _resolve_reading_frame_start_gpos(cand_m)
                c_st_inc = _genomic_pos_to_tx_offset(c_inc_ex, strand, c_start_gpos)
                c_st_exc = _genomic_pos_to_tx_offset(c_exc_ex, strand, c_start_gpos)

                host_ord = cand_m.exons if strand == "+" else cand_m.exons[::-1]
                host_ejcs = list(np.cumsum([b - a for a, b in host_ord])[:-1]) if len(host_ord) >= 2 else []
                host_last_ejc = int(host_ejcs[-1]) if host_ejcs else 0
                if cand_m.stop_codon_3p is not None:
                    h_stop_off = _genomic_pos_to_tx_offset(cand_m.exons, strand, cand_m.stop_codon_3p)
                    host_base_ptc = max(0, host_last_ejc - (h_stop_off + 1)) if h_stop_off is not None else 0
                    inc_stop_off = _genomic_pos_to_tx_offset(c_inc_ex, strand, cand_m.stop_codon_3p)
                    exc_stop_off = _genomic_pos_to_tx_offset(c_exc_ex, strand, cand_m.stop_codon_3p)
                    can_end_inc = (inc_stop_off + 1) if inc_stop_off is not None else None
                    can_end_exc = (exc_stop_off + 1) if exc_stop_off is not None else None
                else:
                    host_base_ptc, can_end_inc, can_end_exc = 0, None, None

                ct_inc = _translate_and_evaluate_nmd(
                    c_dna_inc, [b - a for a, b in c_ord_inc], c_st_inc,
                    phase_offset=c_phase_off, canonical_stop_tx_end=can_end_inc, host_baseline_ptc_dist=host_base_ptc,
                )
                ct_exc = _translate_and_evaluate_nmd(
                    c_dna_exc, [b - a for a, b in c_ord_exc], c_st_exc,
                    phase_offset=c_phase_off, canonical_stop_tx_end=can_end_exc, host_baseline_ptc_dist=host_base_ptc,
                )

                # If an alternative C-terminal reading frame ran off a clipped 3'-UTR end while sibling isoforms extend further 3', heal terminal exon
                if (ct_inc["stop_tx_nt"] is None or ct_exc["stop_tx_nt"] is None):
                    cand_seq_len = int(reader.index[reader.resolve_chrom(cand_m.chrom)][0])
                    ext_inc, ext_exc = list(c_inc_ex), list(c_exc_ex)
                    healed = False
                    if strand == "+" and gene_3p_max > ext_inc[-1][1] and (gene_3p_max - ext_inc[-1][1]) <= 2500 and gene_3p_max <= cand_seq_len:
                        ext_inc[-1] = (ext_inc[-1][0], gene_3p_max)
                        ext_exc[-1] = (ext_exc[-1][0], max(ext_exc[-1][1], gene_3p_max))
                        healed = True
                    elif strand == "-" and gene_3p_min < ext_inc[0][0] and (ext_inc[0][0] - gene_3p_min) <= 2500 and gene_3p_min >= 0:
                        ext_inc[0] = (gene_3p_min, ext_inc[0][1])
                        ext_exc[0] = (min(ext_exc[0][0], gene_3p_min), ext_exc[0][1])
                        healed = True
                    if healed:
                        o_i = ext_inc if strand == "+" else ext_inc[::-1]
                        o_e = ext_exc if strand == "+" else ext_exc[::-1]
                        d_i = reader.fetch_blocks(stream, cand_m.chrom, strand, o_i)
                        d_e = reader.fetch_blocks(stream, cand_m.chrom, strand, o_e)
                        s_i = _genomic_pos_to_tx_offset(ext_inc, strand, c_start_gpos)
                        s_e = _genomic_pos_to_tx_offset(ext_exc, strand, c_start_gpos)
                        t_i2 = _translate_and_evaluate_nmd(d_i, [b - a for a, b in o_i], s_i, phase_offset=c_phase_off, canonical_stop_tx_end=can_end_inc, host_baseline_ptc_dist=host_base_ptc)
                        t_e2 = _translate_and_evaluate_nmd(d_e, [b - a for a, b in o_e], s_e, phase_offset=c_phase_off, canonical_stop_tx_end=can_end_exc, host_baseline_ptc_dist=host_base_ptc)
                        if (
                            _terminal_extension_supported_by_exon(c_inc_ex, ext_inc, ct_inc, t_i2, candidates, strand)
                            and _terminal_extension_supported_by_exon(c_exc_ex, ext_exc, ct_exc, t_e2, candidates, strand)
                            and (t_i2["stop_tx_nt"] is not None)
                            and (t_e2["stop_tx_nt"] is not None)
                        ):
                            c_inc_ex, c_exc_ex, c_ord_inc, c_ord_exc, c_dna_inc, c_dna_exc, ct_inc, ct_exc = ext_inc, ext_exc, o_i, o_e, d_i, d_e, t_i2, t_e2

                both_closed = int((ct_inc["stop_tx_nt"] is not None) and (ct_exc["stop_tx_nt"] is not None))
                not_dual_nmd = int(not (ct_inc["nmd_sensitive"] and ct_exc["nmd_sensitive"]))
                b_rank = (both_closed, not_dual_nmd)
                if b_rank > best_bundle_rank:
                    best_bundle_rank = b_rank
                    chosen_bundle = (cand_m, c_inc_ex, c_exc_ex, c_ord_inc, c_ord_exc, c_dna_inc, c_dna_exc, ct_inc, ct_exc)
                if b_rank == (1, 1):
                    break

            if chosen_bundle is None:
                raise ValueError(f"Unable to assemble coding isoform bundle for annotated event {eid}.")
            host, inc_exons, exc_exons, ord_inc, ord_exc, dna_inc, dna_exc, t_inc, t_exc = chosen_bundle

            sec_cond_info: dict | None = None
            if host.selenocysteine_blocks and selenocysteine_mode != "ignore":
                h_start_gpos, h_phase_off = _resolve_reading_frame_start_gpos(host)
                h_ord = host.exons if strand == "+" else host.exons[::-1]
                h_ejcs = list(np.cumsum([b - a for a, b in h_ord])[:-1]) if len(h_ord) >= 2 else []
                h_last_ejc = int(h_ejcs[-1]) if h_ejcs else 0
                if host.stop_codon_3p is not None:
                    hs_off = _genomic_pos_to_tx_offset(host.exons, strand, host.stop_codon_3p)
                    h_base_ptc = max(0, h_last_ejc - (hs_off + 1)) if hs_off is not None else 0
                    i_st_off = _genomic_pos_to_tx_offset(inc_exons, strand, host.stop_codon_3p)
                    e_st_off = _genomic_pos_to_tx_offset(exc_exons, strand, host.stop_codon_3p)
                    c_end_i = (i_st_off + 1) if i_st_off is not None else None
                    c_end_e = (e_st_off + 1) if e_st_off is not None else None
                else:
                    h_base_ptc, c_end_i, c_end_e = 0, None, None
                st_i_sec = _genomic_pos_to_tx_offset(inc_exons, strand, h_start_gpos)
                st_e_sec = _genomic_pos_to_tx_offset(exc_exons, strand, h_start_gpos)
                sec_i = _translate_and_evaluate_nmd(
                    dna_inc, [b - a for a, b in ord_inc], st_i_sec,
                    phase_offset=h_phase_off, canonical_stop_tx_end=c_end_i, host_baseline_ptc_dist=h_base_ptc,
                    sec_codon_tx_starts=_selenocysteine_tx_starts(inc_exons, strand, host.selenocysteine_blocks),
                )
                sec_e = _translate_and_evaluate_nmd(
                    dna_exc, [b - a for a, b in ord_exc], st_e_sec,
                    phase_offset=h_phase_off, canonical_stop_tx_end=c_end_e, host_baseline_ptc_dist=h_base_ptc,
                    sec_codon_tx_starts=_selenocysteine_tx_starts(exc_exons, strand, host.selenocysteine_blocks),
                )
                sec_cls, _ = _classify_nmd_pair(sec_i, sec_e)
                sec_cond_info = {
                    "gtf_sec_sites": len(host.selenocysteine_blocks),
                    "aa_len_inclusion": int(sec_i["aa_len"]),
                    "aa_len_exclusion": int(sec_e["aa_len"]),
                    "delta_aa": int(sec_i["aa_len"] - sec_e["aa_len"]),
                    "nmd_switch_class": sec_cls,
                    "inc_sec_translated": int(sec_i.get("sec_translated", 0)),
                    "exc_sec_translated": int(sec_e.get("sec_translated", 0)),
                }
                if selenocysteine_mode == "recode":
                    t_inc, t_exc = sec_i, sec_e

            cds_lo, cds_hi = host.cds_blocks[0][0], host.cds_blocks[-1][1]
            ev_coords = [
                int(rec[c])
                for c in ("exonStart_0base", "exonEnd", "longExonStart_0base", "longExonEnd", "1stExonStart_0base", "2ndExonEnd", "riExonStart_0base", "riExonEnd")
                if c in rec and pd.notna(rec[c])
            ]
            ev_lo, ev_hi = (min(ev_coords), max(ev_coords)) if ev_coords else (0, 0)
            overlaps_cds = bool(ev_lo < cds_hi and ev_hi > cds_lo)
            if not overlaps_cds:
                is_5utr = (ev_hi <= cds_lo) if strand == "+" else (ev_lo >= cds_hi)
                loc = "5prime_UTR" if is_5utr else "3prime_UTR"
            else:
                loc = "CDS_coding"

            delta_nt = len(dna_inc) - len(dna_exc)
            if t_inc["protein"] == t_exc["protein"]:
                f_status = "utr_cis_regulatory_only"
            elif not (t_inc["start_preserved"] and t_exc["start_preserved"]):
                f_status = "translation_start_altered"
            elif abs(delta_nt) % 3 == 0:
                f_status = "in_frame_PTC" if (t_inc["nmd_sensitive"] != t_exc["nmd_sensitive"]) else "in_frame_indel"
            else:
                f_status = "frameshift"

            nmd_cls, ptc_dist = _classify_nmd_pair(t_inc, t_exc)

            u_acc, feat_str, arch_list = _map_uniprot_domains_to_isoforms(
                gsym, t_inc["protein"], t_exc["protein"], t_inc["nmd_sensitive"], t_exc["nmd_sensitive"], domain_db
            )
            delta_aa = int(t_inc["aa_len"] - t_exc["aa_len"])

            # Exact shared prefix residue for visualization
            pre_aa = 0
            p1, p2 = t_inc["protein"], t_exc["protein"]
            while pre_aa < min(len(p1), len(p2)) and p1[pre_aa] == p2[pre_aa]:
                pre_aa += 1

            arch_payload = {
                "uniprot_acc": u_acc,
                "mrna_nt_inc": t_inc["mrna_len_nt"],
                "mrna_nt_exc": t_exc["mrna_len_nt"],
                "last_ejc_inc": t_inc["last_ejc_tx_nt"],
                "last_ejc_exc": t_exc["last_ejc_tx_nt"],
                "stop_nt_inc": t_inc["stop_tx_nt"],
                "stop_nt_exc": t_exc["stop_tx_nt"],
                "n_exons_inc": len(ord_inc),
                "n_exons_exc": len(ord_exc),
                "splice_junction_aa": pre_aa + 1,
                "domains": arch_list[:18],
            }

            if sec_cond_info is not None:
                arch_payload["selenocysteine_conditional"] = sec_cond_info

            if nmd_cls == "inclusion_triggers_NMD_poison_exon":
                fav = "Promotes NMD decay" if dpsi > 0 else "Skips poison exon -> rescues protein"
                dom_lost = f" (truncates {feat_str})" if feat_str != "none" else ""
                summary = f"Predicted NMD poison exon ({f_status}): inclusion PTC {ptc_dist} nt upstream of last EJC -> {fav}{dom_lost}"
            elif nmd_cls == "skipping_triggers_NMD_decay":
                fav = "Triggers NMD decay" if dpsi < 0 else "Maintains full-length ORF"
                dom_lost = f" (truncates {feat_str})" if feat_str != "none" else ""
                summary = f"Predicted NMD-inducing skipping ({f_status}): exclusion PTC {ptc_dist} nt upstream of last EJC -> {fav}{dom_lost}"
            elif nmd_cls == "incomplete_3prime_CDS_or_nonstop":
                summary = f"Unresolved 3'-end / potential non-stop transcript ({t_inc['aa_len']} vs {t_exc['aa_len']} aa)"
            elif nmd_cls == "dual_NMD_background":
                summary = (
                    f"Dual-PTC NMD-sensitive background ({f_status}): both isoforms terminate >=50 nt upstream of last EJC "
                    f"(max {ptc_dist} nt; {t_inc['aa_len']} vs {t_exc['aa_len']} aa)"
                )
            elif f_status == "utr_cis_regulatory_only":
                summary = f"ORF intact ({t_inc['aa_len']} aa); remodels {loc} ({delta_nt:+d} nt)"
            else:
                has_c_repl = any(d["state"] == "canonical_C_tail_replaced_by_inclusion" for d in arch_list)
                if has_c_repl and feat_str != "none":
                    summary = f"Productive alternative C-tail ({f_status}): {t_inc['aa_len']} vs {t_exc['aa_len']} aa ({delta_aa:+d} aa); inclusion replaces canonical {feat_str}"
                else:
                    dom_note = f"; remodels {feat_str}" if feat_str != "none" else ""
                    summary = f"Productive {f_status}: {t_inc['aa_len']} vs {t_exc['aa_len']} aa ({delta_aa:+d} aa){dom_note}"
            if conditional_novel_a3_a5:
                summary = f"[Conditional novel-site model] {summary}"
            if sec_cond_info is not None and selenocysteine_mode == "conditional":
                summary = (
                    f"[Conditional selenocysteine (UGA->Sec) host ({sec_cond_info['gtf_sec_sites']} GTF Sec sites; "
                    f"full-recoding model: {sec_cond_info['aa_len_inclusion']} vs {sec_cond_info['aa_len_exclusion']} aa, "
                    f"{sec_cond_info['nmd_switch_class']})] {summary}"
                )

            rows.append({
                "event_id": eid,
                "gene_id": gid,
                "gene_symbol": gsym,
                "event_type": etype,
                "dpsi": dpsi,
                "host_transcript_id": host.transcript_id,
                "uniprot_acc": u_acc,
                "coding_region_location": loc,
                "frame_status": f_status,
                "nmd_switch_class": nmd_cls,
                "aa_len_inclusion": int(t_inc["aa_len"]),
                "aa_len_exclusion": int(t_exc["aa_len"]),
                "delta_aa": delta_aa,
                "ptc_to_last_ejc_nt": int(ptc_dist),
                "disrupted_protein_features": feat_str,
                "functional_consequence_summary": summary,
                "architecture_json": json.dumps(arch_payload),
            })
    return pd.DataFrame(rows, columns=_PROTEIN_CONSEQUENCE_COLUMNS)
