"""End-to-End Local Dummy Pipeline & One-Click Report Generator for `Katana-PRISM` (`Bulk` + `Single-Cell`).

Executes the exact Nextflow CLI stages (`splicemanifold` -> `splicenet`/`scsplicenet` -> `splicetrace`/`scsplicetrace` -> `katana-report --dummy`)
on self-contained synthetic public-gene cohorts and writes two complete inspectable reports:
  1. `dummy_inspection_reports/Bulk_OneClick_Dummy_Report/REPORT.html` (+ `REPORT.zip`)
  2. `dummy_inspection_reports/SingleCell_OneClick_Dummy_Report/REPORT.html` (+ `REPORT.zip`)
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import shutil
import struct
import sys
import zlib
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for pkg in ("katana_studio", "splicemanifold", "splicenet", "splicetrace", "scsplicenet", "scsplicetrace"):
    sys.path.insert(0, str(ROOT / "packages" / pkg))

GENES_MOUSE = [
    ("ENSMUSG00000008658", "Rbfox1"), ("ENSMUSG00000022462", "Rbfox2"), ("ENSMUSG00000025576", "Rbfox3"),
    ("ENSMUSG00000006498", "Ptbp1"),  ("ENSMUSG00000028134", "Ptbp2"),  ("ENSMUSG00000021047", "Nova1"),
    ("ENSMUSG00000030411", "Nova2"),  ("ENSMUSG00000063972", "Srrm4"),  ("ENSMUSG00000024109", "Nrxn1"),
    ("ENSMUSG00000068457", "Nrxn2"),  ("ENSMUSG00000066392", "Nrxn3"),  ("ENSMUSG00000063887", "Nlgn1"),
    ("ENSMUSG00000026959", "Grin1"),  ("ENSMUSG00000033981", "Gria2"),  ("ENSMUSG00000020886", "Dlg4"),
    ("ENSMUSG00000027273", "Snap25"), ("ENSMUSG00000030806", "Stx1b"),  ("ENSMUSG00000035864", "Syt1"),
    ("ENSMUSG00000020894", "Vamp2"),  ("ENSMUSG00000024617", "Camk2a"), ("ENSMUSG00000057897", "Camk2b"),
    ("ENSMUSG00000004113", "Cacna1b"),("ENSMUSG00000023034", "Scn8a"),  ("ENSMUSG00000027583", "Kcnq2"),
    ("ENSMUSG00000069601", "Ank3"),   ("ENSMUSG00000020315", "Sptbn1"), ("ENSMUSG00000015222", "Map2"),
    ("ENSMUSG00000018411", "Mapt"),   ("ENSMUSG00000026825", "Dnm1"),   ("ENSMUSG00000028478", "Clta"),
    ("ENSMUSG00000024381", "Bin1"),   ("ENSMUSG00000037217", "Syn1"),   ("ENSMUSG00000032589", "Bsn"),
    ("ENSMUSG00000029088", "Pclo"),   ("ENSMUSG00000022623", "Shank3"), ("ENSMUSG00000007617", "Homer1"),
    ("ENSMUSG00000021250", "Fos"),    ("ENSMUSG00000022602", "Arc"),    ("ENSMUSG00000048482", "Bdnf"),
    ("ENSMUSG00000038418", "Egr1"),   ("ENSMUSG00000045903", "Npas4"),  ("ENSMUSG00000018379", "Srsf1"),
    ("ENSMUSG00000071172", "Srsf3"),  ("ENSMUSG00000024097", "Srsf7"),  ("ENSMUSG00000056130", "Hnrnpa1"),
    ("ENSMUSG00000020594", "Mbnl1"),  ("ENSMUSG00000022139", "Mbnl2"),  ("ENSMUSG00000005506", "Celf1"),
    ("ENSMUSG00000062078", "Qki"),    ("ENSMUSG00000024949", "Elavl3"), ("ENSMUSG00000028546", "Elavl4"),
    ("ENSMUSG00000041459", "Tardbp"), ("ENSMUSG00000028034", "Fubp1"),  ("ENSMUSG00000031987", "Amph"),
    ("ENSMUSG00000041670", "Rims1"),  ("ENSMUSG00000038738", "Shank1"), ("ENSMUSG00000051378", "Cacna1e"),
    ("ENSMUSG00000032294", "Cltb"),   ("ENSMUSG00000060373", "Celf2"),  ("ENSMUSG00000021134", "Hnrnpc"),
]


def build_mini_reference_and_events(work_dir: Path, rng: np.random.Generator) -> tuple[Path, Path, list[dict]]:
    """Build an indexed 960-kb `chr1.fa` (+ `.fai`) and multi-isoform `annotation.gtf` covering 60 mouse genes
    with both in-frame cassette exons and poison-exon 50-nt EJC NMD events, plus cis `TGCATG` / `TCTCTCT` motifs."""
    ref_dir = work_dir / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    seq_len = 960_000
    bases = np.array(list("ACGT"))
    genome = rng.choice(bases, size=seq_len, p=[0.27, 0.23, 0.23, 0.27])

    def _put(pos: int, dna: str) -> None:
        arr = np.array(list(dna.upper()))
        genome[pos : pos + len(arr)] = arr

    gtf_lines: list[str] = []
    event_catalog: list[dict] = []
    et_cycle = ["SE", "SE", "SE", "A3SS", "A5SS", "MXE", "RI"]

    # Generate 150 splicing events across the 60 genes (2-3 loci per gene)
    locus_idx = 0
    for ev_idx in range(150):
        gid, gsym = GENES_MOUSE[ev_idx % len(GENES_MOUSE)]
        et = et_cycle[ev_idx % len(et_cycle)]
        base_pos = 4000 + locus_idx * 6100
        locus_idx += 1

        # Exon 1 (5'UTR + start codon ATG), Alt Exon 2, Exon 3 (CDS), Exon 4 (3'UTR -> creates EJC downstream of PTC!)
        cds1_len = 420 if (ev_idx % 3 == 2) else 180  # spans N-terminal (aa 61..92) & mid-protein (aa 141..172) UniProt domains!
        e1_s, e1_e = base_pos, base_pos + 60 + cds1_len
        cds1_s = e1_s + 60
        up_es, up_ee = e1_s, e1_e

        is_poison_nmd = (ev_idx % 5 == 1) and (et == "SE")
        alt_len = 92 if is_poison_nmd else 96  # 92 causes frameshift / in-exon TAA PTC; 96 is clean in-frame 32-aa insert!
        e2_s, e2_e = base_pos + 1200, base_pos + 1200 + alt_len
        e2b_s, e2b_e = base_pos + 1850, base_pos + 1850 + 96  # for MXE
        e3_s, e3_e = base_pos + 2600, base_pos + 2600 + 240
        e4_s, e4_e = base_pos + 3500, base_pos + 3500 + 220  # downstream 3'UTR exon for authentic 50-nt EJC rule!

        # Place clean start codon, sense codons (GCT=Ala, GGA=Gly, AAG=Lys), and stop codon without accidental internal stops
        sense_unit = "GCTGGAAGCAAGCTG" * 60
        _put(cds1_s, "ATG" + sense_unit[: cds1_len - 3])
        if is_poison_nmd:
            _put(e2_s, sense_unit[:30] + "TAAGCTGGA" + sense_unit[: alt_len - 39])
        else:
            _put(e2_s, sense_unit[:alt_len])
        _put(e2b_s, sense_unit[:96])
        _put(e3_s, sense_unit[:207] + "TAA" + "ACGT" * 7 + "AA")

        # Splice site dinucleotides + cis-regulatory Rbfox1 (`TGCATG`) in downstream intron or Ptbp2 (`TCTCTCT`) in upstream intron
        _put(e1_e, "GTAAGT"); _put(e2_s - 2, "AG"); _put(e2_e, "GTAAGT"); _put(e3_s - 2, "AG")
        if ev_idx < 55:
            # Strong Rbfox1 enhancer cluster across downstream_intron_35_85!
            for off in (36, 44, 52, 60, 68, 76, 92, 110):
                _put(e2_e + off, "TGCATG")
            reg_group = "rbfox_pos"
        elif ev_idx < 95:
            # Strong Ptbp1/Ptbp2 polypyrimidine silencer in upstream_intron_35_85!
            for off in (40, 50, 60, 70, 80):
                _put(e2_s - off, "CTCTCTCT")
            reg_group = "ptbp_neg"
        else:
            reg_group = "bg"

        tx_inc = f"ENSMUST{ev_idx:07d}1"
        tx_skp = f"ENSMUST{ev_idx:07d}2"
        attr_g = f'gene_id "{gid}"; gene_name "{gsym}"; gene_biotype "protein_coding";'
        gtf_lines.append(f"chr1\tKatana\tgene\t{e1_s+1}\t{e4_e}\t.\t+\t.\t{attr_g}")

        # Inclusion transcript (4 exons)
        attr_t1 = f'{attr_g} transcript_id "{tx_inc}"; transcript_biotype "protein_coding";'
        gtf_lines.append(f"chr1\tKatana\ttranscript\t{e1_s+1}\t{e4_e}\t.\t+\t.\t{attr_t1}")
        for ex_num, (ex_s, ex_e) in enumerate(((e1_s, e1_e), (e2_s, e2_e), (e3_s, e3_e), (e4_s, e4_e)), start=1):
            gtf_lines.append(f'chr1\tKatana\texon\t{ex_s+1}\t{ex_e}\t.\t+\t.\t{attr_t1} exon_number "{ex_num}";')
        gtf_lines.append(f"chr1\tKatana\tCDS\t{cds1_s+1}\t{e1_e}\t.\t+\t0\t{attr_t1}")
        if is_poison_nmd:
            gtf_lines.append(f"chr1\tKatana\tCDS\t{e2_s+1}\t{e2_s+30}\t.\t+\t0\t{attr_t1}")
        else:
            gtf_lines.append(f"chr1\tKatana\tCDS\t{e2_s+1}\t{e2_e}\t.\t+\t0\t{attr_t1}")
            gtf_lines.append(f"chr1\tKatana\tCDS\t{e3_s+1}\t{e3_s+207}\t.\t+\t0\t{attr_t1}")

        # Skipping transcript (3 exons)
        attr_t2 = f'{attr_g} transcript_id "{tx_skp}"; transcript_biotype "protein_coding";'
        gtf_lines.append(f"chr1\tKatana\ttranscript\t{e1_s+1}\t{e4_e}\t.\t+\t.\t{attr_t2}")
        for ex_num, (ex_s, ex_e) in enumerate(((e1_s, e1_e), (e3_s, e3_e), (e4_s, e4_e)), start=1):
            gtf_lines.append(f'chr1\tKatana\texon\t{ex_s+1}\t{ex_e}\t.\t+\t.\t{attr_t2} exon_number "{ex_num}";')
        gtf_lines.append(f"chr1\tKatana\tCDS\t{cds1_s+1}\t{e1_e}\t.\t+\t0\t{attr_t2}")
        gtf_lines.append(f"chr1\tKatana\tCDS\t{e3_s+1}\t{e3_s+207}\t.\t+\t0\t{attr_t2}")

        rec = {
            "idx": ev_idx, "gene_id": gid, "gene_symbol": gsym, "event_type": et, "chr": "chr1", "strand": "+",
            "reg_group": reg_group, "tx_inc": tx_inc, "tx_skp": tx_skp,
            "exonStart_0base": e2_s, "exonEnd": e2_e, "upstreamES": up_es, "upstreamEE": up_ee, "downstreamES": e3_s, "downstreamEE": e3_e,
            "longExonStart_0base": e2_s - 36 if et == "A3SS" else e2_s,
            "longExonEnd": e2_e + 36 if et == "A5SS" else e2_e,
            "shortES": e2_s, "shortEE": e2_e,
            "flankingES": up_es if et == "A3SS" else e3_s,
            "flankingEE": up_ee if et == "A3SS" else e3_e,
            "1stExonStart_0base": e2_s, "1stExonEnd": e2_e, "2ndExonStart_0base": e2b_s, "2ndExonEnd": e2b_e,
            "riExonStart_0base": up_es, "riExonEnd": e3_e,
        }
        if et == "RI":
            rec["upstreamES"], rec["upstreamEE"] = up_es, up_ee
            rec["downstreamES"], rec["downstreamEE"] = e3_s, e3_e
        event_catalog.append(rec)

    fasta_p = ref_dir / "genome.fasta"
    line_bases = 80
    seq_str = "".join(genome.tolist())
    with fasta_p.open("wb") as fh:
        header = b">chr1\n"
        fh.write(header)
        offset = len(header)
        for i in range(0, seq_len, line_bases):
            fh.write(seq_str[i : i + line_bases].encode("ascii") + b"\n")
    (ref_dir / "genome.fasta.fai").write_text(f"chr1\t{seq_len}\t{offset}\t{line_bases}\t{line_bases + 1}\n", encoding="utf-8")

    gtf_p = ref_dir / "annotation.gtf"
    gtf_p.write_text("\n".join(gtf_lines) + "\n", encoding="utf-8")
    return fasta_p, gtf_p, event_catalog


def _write_bgzf_blocks(payload: bytes) -> bytes:
    """Encode arbitrary bytes as standard RFC-1952 BGZF blocks + 28-byte EOF marker."""
    out = bytearray()
    chunk_size = 64000
    for i in range(0, len(payload), chunk_size):
        chunk = payload[i : i + chunk_size]
        comp = zlib.compressobj(level=5, method=zlib.DEFLATED, wbits=-zlib.MAX_WBITS)
        cdata = comp.compress(chunk) + comp.flush()
        bsize = len(cdata) + 25  # 18 hdr + cdata + 8 trailer - 1
        out += b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00"
        out += struct.pack("<H", bsize)
        out += cdata
        out += struct.pack("<II", zlib.crc32(chunk) & 0xFFFFFFFF, len(chunk))
    # Standard empty BGZF EOF block
    out += b"\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00\x1b\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    return bytes(out)


def _write_synthetic_indexed_bam(
    bam_path: Path,
    contig_len: int,
    reads_0based: list[tuple[int, tuple[tuple[int, int], ...]]],
    reported_mapped_reads: int = 8_500_000,
) -> None:
    """Write a coordinate-sorted BAM v1 file + BAI v1 index using pure Python stdlib."""
    bam_path.parent.mkdir(parents=True, exist_ok=True)
    sam_hdr = f"@HD\tVN:1.6\tSO:coordinate\n@SQ\tSN:chr1\tLN:{contig_len}\n".encode("ascii")
    buf = bytearray()
    buf += b"BAM\x01"
    buf += struct.pack("<i", len(sam_hdr))
    buf += sam_hdr
    buf += struct.pack("<ii", 1, 5) + b"chr1\x00" + struct.pack("<i", contig_len)

    reads_sorted = sorted(reads_0based, key=lambda x: x[0])
    for idx, (pos0, cigar) in enumerate(reads_sorted):
        if pos0 < 0 or not cigar:
            continue
        rname = f"r{idx}".encode("ascii") + b"\x00"
        l_seq = sum(ln for op, ln in cigar if op in {0, 1, 4, 7, 8})
        if l_seq <= 0:
            continue
        cigar_bytes = struct.pack(f"<{len(cigar)}I", *((ln << 4) | op for op, ln in cigar))
        seq_bytes = b"\x11" * ((l_seq + 1) // 2)
        qual_bytes = b"\x1e" * l_seq
        l_rn_mq_bin = (4681 << 16) | (60 << 8) | len(rname)
        flag_nc = len(cigar)  # flag=0 (primary mapped)
        core = struct.pack("<iiIIiiii", 0, int(pos0), l_rn_mq_bin, flag_nc, l_seq, -1, -1, 0)
        rec = core + rname + cigar_bytes + seq_bytes + qual_bytes
        buf += struct.pack("<i", len(rec))
        buf += rec

    bam_bytes = _write_bgzf_blocks(bytes(buf))
    bam_path.write_bytes(bam_bytes)

    # Write BAI v1 with metadata bin 37450 carrying realistic library mapped read count for RPKM scale
    bai = bytearray()
    bai += b"BAI\x01"
    bai += struct.pack("<i", 1)  # n_ref = 1
    bai += struct.pack("<i", 1)  # n_bin = 1 (metadata bin 37450)
    bai += struct.pack("<Ii", 37450, 2)
    bai += struct.pack("<QQQQ", 0, len(bam_bytes) << 16, int(reported_mapped_reads), 0)
    bai += struct.pack("<i", 0)  # n_intv = 0
    Path(str(bam_path) + ".bai").write_bytes(bytes(bai))


def _simulate_asymmetric_event_reads(
    ev: dict,
    psi: float,
    junc_depth: float,
    exon_bias: tuple[float, float, float, float, float, float],
    rng: np.random.Generator,
) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    """Generate authentic asymmetric RNA-seq exon pileup reads + independent split-read junctions."""
    reads: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    et = ev["event_type"]
    up_scale, dn_scale, a_up, b_up, a_dn, b_dn = exon_bias
    read_len = 76
    base_exon_reads = max(24, int(round(junc_depth * 1.55)))

    def _add_exon_pileup(s0: int, e0: int, count: int, a_beta: float, b_beta: float) -> None:
        span = max(20, e0 - s0)
        if count <= 0:
            return
        # Combine asymmetric beta position skew + local fragment wave for realistic non-symmetric RNA-seq depth
        fracs = rng.beta(a_beta, b_beta, size=count)
        for fr in fracs:
            seg = min(read_len, max(28, int(rng.integers(42, read_len + 1))))
            max_start = max(s0, e0 - seg)
            p0 = int(round(s0 + fr * max(1, max_start - s0)))
            p0 = max(s0, min(max_start, p0))
            reads.append((p0, ((0, min(seg, e0 - p0)),)))

    def _add_split_reads(left_end_0: int, right_start_0: int, count: int) -> None:
        intron_len = right_start_0 - left_end_0
        if intron_len <= 10 or count <= 0:
            return
        for _ in range(count):
            a1 = int(rng.integers(14, 52))
            a2 = int(rng.integers(14, 52))
            pos0 = max(1, left_end_0 - a1)
            reads.append((pos0, ((0, a1), (3, intron_len), (0, a2))))

    def _add_intron_noise(s0: int, e0: int, count: int) -> None:
        if e0 - s0 <= 60 or count <= 0:
            return
        for _ in range(count):
            p0 = int(rng.integers(s0 + 10, e0 - 45))
            reads.append((p0, ((0, int(rng.integers(32, 54))),)))

    inc_target = max(1.0, junc_depth * psi)
    skip_target = max(1.0, junc_depth * (1.0 - psi))

    if et == "SE":
        up_s, up_e = ev["upstreamES"], ev["upstreamEE"]
        ex_s, ex_e = ev["exonStart_0base"], ev["exonEnd"]
        dn_s, dn_e = ev["downstreamES"], ev["downstreamEE"]
        _add_exon_pileup(up_s, up_e, int(round(base_exon_reads * up_scale)), a_up, b_up)
        _add_exon_pileup(ex_s, ex_e, max(2, int(round(base_exon_reads * (0.20 + 0.95 * psi) * rng.uniform(0.85, 1.15)))), 1.6, 2.2)
        _add_exon_pileup(dn_s, dn_e, int(round(base_exon_reads * dn_scale)), a_dn, b_dn)
        _add_intron_noise(up_e, ex_s, int(rng.integers(3, 9)))
        _add_intron_noise(ex_e, dn_s, int(rng.integers(3, 9)))
        # Independent left & right inclusion junction counts (naturally asymmetric!)
        n_l = int(rng.poisson(inc_target * rng.uniform(0.82, 1.18)))
        n_r = int(rng.poisson(inc_target * rng.uniform(0.82, 1.18)))
        n_sk = int(rng.poisson(skip_target * rng.uniform(0.88, 1.12)))
        _add_split_reads(up_e, ex_s, n_l)
        _add_split_reads(ex_e, dn_s, n_r)
        _add_split_reads(up_e, dn_s, n_sk)
    elif et == "RI":
        up_s, up_e = ev["upstreamES"], ev["upstreamEE"]
        dn_s, dn_e = ev["downstreamES"], ev["downstreamEE"]
        _add_exon_pileup(up_s, up_e, int(round(base_exon_reads * up_scale)), a_up, b_up)
        _add_exon_pileup(dn_s, dn_e, int(round(base_exon_reads * dn_scale)), a_dn, b_dn)
        # Retained intron read density directly reflects retention PSI!
        ri_reads = max(3, int(round(base_exon_reads * 1.45 * psi * rng.uniform(0.88, 1.12))))
        _add_exon_pileup(up_e - 15, dn_s + 15, ri_reads, 1.35, 1.85)
        n_sp = int(rng.poisson(skip_target * rng.uniform(0.85, 1.15)))
        _add_split_reads(up_e, dn_s, n_sp)
    elif et == "A3SS":
        fl_s, fl_e = ev["flankingES"], ev["flankingEE"]
        lg_s, lg_e = ev["longExonStart_0base"], ev["longExonEnd"]
        sh_s, sh_e = ev["shortES"], ev["shortEE"]
        _add_exon_pileup(fl_s, fl_e, int(round(base_exon_reads * up_scale)), a_up, b_up)
        _add_exon_pileup(sh_s, sh_e, int(round(base_exon_reads * dn_scale)), a_dn, b_dn)
        # Alternative 3' extension covered proportionally to long-site usage
        _add_exon_pileup(lg_s, sh_s + 24, max(2, int(round(base_exon_reads * 0.65 * (1.0 - psi)))), 1.4, 2.0)
        _add_intron_noise(fl_e, lg_s, int(rng.integers(3, 8)))
        _add_split_reads(fl_e, lg_s, int(rng.poisson(skip_target * rng.uniform(0.85, 1.15))))
        _add_split_reads(fl_e, sh_s, int(rng.poisson(inc_target * rng.uniform(0.85, 1.15))))
    elif et == "A5SS":
        lg_s, lg_e = ev["longExonStart_0base"], ev["longExonEnd"]
        sh_s, sh_e = ev["shortES"], ev["shortEE"]
        fl_s, fl_e = ev["flankingES"], ev["flankingEE"]
        _add_exon_pileup(sh_s, sh_e, int(round(base_exon_reads * up_scale)), a_up, b_up)
        _add_exon_pileup(sh_e - 24, lg_e, max(2, int(round(base_exon_reads * 0.65 * (1.0 - psi)))), 2.0, 1.4)
        _add_exon_pileup(fl_s, fl_e, int(round(base_exon_reads * dn_scale)), a_dn, b_dn)
        _add_intron_noise(lg_e, fl_s, int(rng.integers(3, 8)))
        _add_split_reads(lg_e, fl_s, int(rng.poisson(skip_target * rng.uniform(0.85, 1.15))))
        _add_split_reads(sh_e, fl_s, int(rng.poisson(inc_target * rng.uniform(0.85, 1.15))))
    else:  # MXE
        up_s, up_e = ev["upstreamES"], ev["upstreamEE"]
        e1_s, e1_e = ev["1stExonStart_0base"], ev["1stExonEnd"]
        e2_s, e2_e = ev["2ndExonStart_0base"], ev["2ndExonEnd"]
        dn_s, dn_e = ev["downstreamES"], ev["downstreamEE"]
        _add_exon_pileup(up_s, up_e, int(round(base_exon_reads * up_scale)), a_up, b_up)
        _add_exon_pileup(e1_s, e1_e, max(2, int(round(base_exon_reads * 0.95 * psi))), 1.5, 2.2)
        _add_exon_pileup(e2_s, e2_e, max(2, int(round(base_exon_reads * 0.95 * (1.0 - psi)))), 2.1, 1.5)
        _add_exon_pileup(dn_s, dn_e, int(round(base_exon_reads * dn_scale)), a_dn, b_dn)
        _add_split_reads(up_e, e1_s, int(rng.poisson(inc_target * rng.uniform(0.82, 1.18))))
        _add_split_reads(e1_e, dn_s, int(rng.poisson(inc_target * rng.uniform(0.82, 1.18))))
        _add_split_reads(up_e, e2_s, int(rng.poisson(skip_target * rng.uniform(0.82, 1.18))))
        _add_split_reads(e2_e, dn_s, int(rng.poisson(skip_target * rng.uniform(0.82, 1.18))))

    return reads


def run_bulk_dummy_pipeline(fasta_p: Path, gtf_p: Path, event_catalog: list[dict], work_root: Path, report_out: Path, rng: np.random.Generator) -> dict:
    """Create a complete 12-sample, 4-contrast Bulk `katana_bundle` and execute:
    `splicemanifold bulk` -> `splicenet` -> `splicetrace` (x4 contrasts) -> `katana-report bulk --dummy`."""
    from splicemanifold.cli import main as sm_main
    from splicenet.cli import main as sn_main
    from splicetrace.cli import main as st_main
    from katana_studio.cli import report_main

    bundle = work_root / "bulk_bundle"
    for sub in ("pipeline_info", "refs", "qc/strandedness", "qc/multiqc", "provenance/bam_audit", "DEG", "dtu/results/stager", "dexseq_exon"):
        (bundle / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy2(fasta_p, bundle / "refs/genome.fasta")
    shutil.copy2(Path(str(fasta_p) + ".fai"), bundle / "refs/genome.fasta.fai")
    shutil.copy2(gtf_p, bundle / "refs/annotation.gtf")

    arms = ["WT_Basal", "WT_Depol", "Rbfox1KO_Basal", "Rbfox1KO_Depol"]
    samples_rows = []
    for arm in arms:
        for rep in (1, 2, 3):
            samples_rows.append({"sample": f"{arm}_rep{rep}", "condition": arm, "subject": f"{arm}_mouse{rep}", "strandedness": "reverse", "read_length": 150})
    s_df = pd.DataFrame(samples_rows)
    s_df.to_csv(bundle / "pipeline_info/samplesheet.valid.csv", index=False)

    contrasts = [
        ("Rbfox1KO_Basal_vs_WT_Basal", "Rbfox1KO_Basal", "WT_Basal"),
        ("Rbfox1KO_Depol_vs_WT_Depol", "Rbfox1KO_Depol", "WT_Depol"),
        ("WT_Depol_vs_WT_Basal", "WT_Depol", "WT_Basal"),
        ("Rbfox1KO_Depol_vs_Rbfox1KO_Basal", "Rbfox1KO_Depol", "Rbfox1KO_Basal"),
    ]
    pd.DataFrame([{"contrast": c, "treatment": t, "control": d, "paired": False, "design": "~ condition"} for c, t, d in contrasts]).to_csv(
        bundle / "pipeline_info/contrastsheet.valid.csv", index=False
    )

    # Annotation table
    pd.DataFrame([{"gene_id": g, "gene_name": s} for g, s in GENES_MOUSE]).to_csv(bundle / "refs/annotation.tsv", sep="\t", index=False)

    # QC: Strandedness + BAM primary metrics + MultiQC 5'->3' coverage + MultiQC HTML
    strand_rows = []
    for r in samples_rows:
        sid = r["sample"]
        anti = round(float(rng.uniform(0.955, 0.985)), 4)
        strand_rows.append({"sample": sid, "forward_fraction": round(1.0 - anti, 4), "reverse_fraction": anti, "resolved_strandedness": "reverse"})
        pd.DataFrame([{"sample": sid, "primary_mapped_reads": int(rng.integers(38_000_000, 56_000_000))}]).to_csv(
            bundle / f"provenance/bam_audit/{sid}.primary_metrics.tsv", sep="\t", index=False
        )
    pd.DataFrame(strand_rows).to_csv(bundle / "qc/strandedness/strandedness.summary.tsv", sep="\t", index=False)

    bins = np.linspace(0, 100, 101)
    mqc_lines = ["Sample\t" + "\t".join(str(int(b)) for b in bins)]
    for r in samples_rows:
        curve = np.exp(-0.5 * ((bins - 52.0) / 34.0) ** 2) * rng.uniform(0.96, 1.04, size=len(bins))
        curve = curve / curve.mean()
        mqc_lines.append(r["sample"] + "\t" + "\t".join(f"{v:.4f}" for v in curve))
    (bundle / "qc/multiqc/mqc_qualimap_gene_coverage_profile_1.txt").write_text("\n".join(mqc_lines) + "\n", encoding="utf-8")
    (bundle / "qc/multiqc/multiqc_report.html").write_text("<!doctype html><html><body><h1>MultiQC Cohort Sequencing Summary</h1></body></html>", encoding="utf-8")

    # Simulate realistic sample-level PSI across 150 events (3 co-spliced modules: Rbfox KO-sensitive, Depol-sensitive, Combined)
    sids_all = [r["sample"] for r in samples_rows]
    psi_true = np.zeros((len(event_catalog), len(sids_all)), dtype=float)
    for i, ev in enumerate(event_catalog):
        base_p = 0.68 if ev["reg_group"] == "rbfox_pos" else (0.28 if ev["reg_group"] == "ptbp_neg" else float(rng.uniform(0.25, 0.75)))
        for j, r in enumerate(samples_rows):
            arm = r["condition"]
            shift = 0.0
            if ev["reg_group"] == "rbfox_pos":
                if "Rbfox1KO" in arm:
                    shift -= 0.34
                if "Depol" in arm:
                    shift += 0.14
            elif ev["reg_group"] == "ptbp_neg":
                if "Depol" in arm:
                    shift += 0.31
                if "Rbfox1KO" in arm:
                    shift += 0.11
            psi_true[i, j] = np.clip(base_p + shift + rng.normal(0.0, 0.032), 0.03, 0.97)

    # Normalized RBP / host gene counts (`DEG/all.normalised_counts.tsv`)
    norm_rows = []
    for idx_g, (gid, gsym) in enumerate(GENES_MOUSE):
        rec_n: dict[str, object] = {"gene_id": gid, "gene_symbol": gsym}
        for r in samples_rows:
            arm = r["condition"]
            mu = 450.0
            if gsym == "Rbfox1" and "Rbfox1KO" in arm:
                mu = 35.0
            elif gsym in {"Fos", "Arc", "Bdnf", "Egr1", "Npas4", "Homer1"} and "Depol" in arm:
                mu = 1650.0
            elif idx_g % 4 == 0 and "Rbfox1KO" in arm:
                mu = 920.0
            rec_n[r["sample"]] = round(float(mu * rng.uniform(0.88, 1.12)), 2)
        norm_rows.append(rec_n)
    norm_df = pd.DataFrame(norm_rows)
    norm_df.to_csv(bundle / "DEG/all.normalised_counts.tsv", sep="\t", index=False)

    # Generate 12 real coordinate-sorted indexed BAMs (`bundle/bam/{sample}.bam` + `.bai`)
    # so Katana Native Sashimi runs authentic BAM RPKM read-depth tracks with asymmetric exon pileups & split reads!
    bam_dir = bundle / "bam"
    bam_dir.mkdir(parents=True, exist_ok=True)
    locus_biases = [
        (
            float(rng.uniform(0.68, 1.38)),
            float(rng.uniform(0.82, 1.62)),
            float(rng.uniform(1.15, 2.60)),
            float(rng.uniform(1.15, 2.60)),
            float(rng.uniform(1.15, 2.60)),
            float(rng.uniform(1.15, 2.60)),
        )
        for _ in event_catalog
    ]
    for j, r in enumerate(samples_rows):
        sid = r["sample"]
        sample_reads: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        for i, ev in enumerate(event_catalog):
            sample_reads.extend(
                _simulate_asymmetric_event_reads(
                    ev, float(psi_true[i, j]), junc_depth=float(rng.uniform(34.0, 68.0)), exon_bias=locus_biases[i], rng=rng
                )
            )
        _write_synthetic_indexed_bam(bam_dir / f"{sid}.bam", 960_000, sample_reads, reported_mapped_reads=int(rng.integers(6_800_000, 9_400_000)))

    # Per-contrast rMATS (`JCEC` + `JC`), DESeq2, DRIMSeq/stageR DTU, DEXSeq, and SUPPA2
    for cid, num_arm, den_arm in contrasts:
        num_idx = [j for j, r in enumerate(samples_rows) if r["condition"] == num_arm]
        den_idx = [j for j, r in enumerate(samples_rows) if r["condition"] == den_arm]
        r_post = bundle / "rmats" / cid / "rmats_post"
        r_post.mkdir(parents=True, exist_ok=True)
        sup_dir = bundle / "suppa" / cid / "diffsplice/per_local_event"
        sup_dir.mkdir(parents=True, exist_ok=True)

        by_et: dict[str, list[dict]] = {et: [] for et in ("SE", "A3SS", "A5SS", "MXE", "RI")}
        suppa_rows = []
        dtu_rows = []
        dex_rows = []

        for i, ev in enumerate(event_catalog):
            et = ev["event_type"]
            p1 = psi_true[i, num_idx]
            p2 = psi_true[i, den_idx]
            # Make ~20% of background events low-depth so PS-DAS vs Ordinary DAS precision filtering is visibly distinct!
            tot_depth = int(rng.integers(4, 9)) if (ev["reg_group"] == "bg" and i % 4 == 0) else int(rng.integers(35, 95))
            ijc1 = [max(0, int(round(p * tot_depth))) for p in p1]
            sjc1 = [max(0, tot_depth - c) for c in ijc1]
            ijc2 = [max(0, int(round(p * tot_depth))) for p in p2]
            sjc2 = [max(0, tot_depth - c) for c in ijc2]
            dpsi = float(p1.mean() - p2.mean())
            t_res = stats.ttest_ind(p1, p2, equal_var=False)
            pv = float(t_res.pvalue) if np.isfinite(t_res.pvalue) else 1.0
            if abs(dpsi) >= 0.20:
                pv = min(pv, 10 ** (-float(rng.uniform(4.2, 9.5))))

            base_row: dict[str, object] = {"ID": i, "GeneID": f'"{ev["gene_id"]}"', "geneSymbol": f'"{ev["gene_symbol"]}"', "chr": "chr1", "strand": "+"}
            if et == "SE":
                base_row.update({k: ev[k] for k in ("exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE")})
            elif et in ("A3SS", "A5SS"):
                base_row.update({k: ev[k] for k in ("longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE")})
            elif et == "MXE":
                base_row.update({k: ev[k] for k in ("1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE")})
            else:
                base_row.update({k: ev[k] for k in ("riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE")})

            base_row.update({
                "ID.1": i,
                "IJC_SAMPLE_1": ",".join(map(str, ijc1)),
                "SJC_SAMPLE_1": ",".join(map(str, sjc1)),
                "IJC_SAMPLE_2": ",".join(map(str, ijc2)),
                "SJC_SAMPLE_2": ",".join(map(str, sjc2)),
                "IncFormLen": 190,
                "SkipFormLen": 99,
                "PValue": pv,
                "FDR": min(1.0, pv * 6.0),
                "IncLevel1": ",".join(f"{x:.3f}" for x in p1),
                "IncLevel2": ",".join(f"{x:.3f}" for x in p2),
                "IncLevelDifference": round(dpsi, 4),
            })
            by_et[et].append(base_row)
            # Realistic graded orthogonal corroboration across SUPPA2, DEXSeq & DRIMSeq/stageR DTU
            # (~35% 4/4 Gold, ~25% 3/4 High, ~20% 2/4 Corroborated, ~20% 1/4 rMATS-only!)
            g_mod = (ev["idx"] % len(GENES_MOUSE)) % 5
            sup_p = min(1.0, pv * 2.2) if g_mod in (0, 1, 2, 3) else float(rng.uniform(0.18, 0.85))
            sup_dp = round(dpsi * 0.94, 4) if g_mod in (0, 1, 2, 3) else round(dpsi * 0.25, 4)
            suppa_rows.append({"event_id": f"{ev['gene_id']};{et}:chr1:{ev['exonStart_0base']}-{ev['exonEnd']}:+", f"{cid}_dPSI": sup_dp, f"{cid}_p-val": sup_p})

            dex_q = min(1.0, pv * 5.0) if g_mod in (0, 1, 2) else float(rng.uniform(0.16, 0.90))
            dex_lfc = round(dpsi * 2.9, 4) if g_mod in (0, 1, 2) else round(dpsi * 0.45, 4)
            dex_rows.append({"groupID": ev["gene_id"], "featureID": f"E{i:03d}", "pvalue": min(1.0, dex_q / 3.0), "padj": dex_q, "log2fold": dex_lfc})

            dtu_gq = min(1.0, pv * 4.2) if g_mod in (0, 1) else float(rng.uniform(0.15, 0.88))
            dtu_rows.append({"geneID": ev["gene_id"], "txID": ev["tx_inc"], "gene": dtu_gq, "transcript": min(1.0, dtu_gq * 1.2)})

        for et, r_list in by_et.items():
            df_et = pd.DataFrame(r_list)
            df_et.to_csv(r_post / f"{et}.MATS.JCEC.txt", sep="\t", index=False)
            df_et.to_csv(r_post / f"{et}.MATS.JC.txt", sep="\t", index=False)

        pd.DataFrame(suppa_rows).to_csv(sup_dir / "local_events.dpsi", sep="\t", index=False)
        pd.DataFrame(dtu_rows).to_csv(bundle / f"dtu/results/stager/getAdjustedPValues.{cid}.tsv", sep="\t", index=False)
        pd.DataFrame(dex_rows).to_csv(bundle / f"dexseq_exon/DEXSeqResults.{cid}.csv", index=False)

        # DESeq2 results for this contrast
        deg_rows = []
        num_s = [r["sample"] for r in samples_rows if r["condition"] == num_arm]
        den_s = [r["sample"] for r in samples_rows if r["condition"] == den_arm]
        for _, nr in norm_df.iterrows():
            v1 = np.array([float(nr[s]) for s in num_s])
            v2 = np.array([float(nr[s]) for s in den_s])
            lfc = float(np.log2((v1.mean() + 1.0) / (v2.mean() + 1.0)))
            pv_g = float(stats.ttest_ind(np.log2(v1 + 1.0), np.log2(v2 + 1.0)).pvalue)
            if abs(lfc) > 0.7:
                pv_g = min(pv_g, 1e-5)
            deg_rows.append({"gene_id": nr["gene_id"], "gene_symbol": nr["gene_symbol"], "baseMean": round(float((v1.mean() + v2.mean()) / 2), 2), "log2FoldChange": round(lfc, 4), "pvalue": pv_g, "padj": min(1.0, pv_g * 4.0)})
        pd.DataFrame(deg_rows).to_csv(bundle / f"DEG/{cid}.deseq2.results.tsv", sep="\t", index=False)

    # Execute the exact 4 Nextflow Bulk downstream stages!
    sm_out = work_root / "bulk_splicemanifold"
    sn_out = work_root / "bulk_splicenet"
    st_outs = []
    sm_main(["bulk", "--bundle", str(bundle), "--rmats-mode", "JCEC", "--outdir", str(sm_out)])
    sn_main(["--bundle", str(bundle), "--rmats-mode", "JCEC", "--min-module-size", "10", "--output-dir", str(sn_out)])

    for cid, _, _ in contrasts:
        st_c = work_root / f"splicetrace_{cid}"
        st_main([
            "--rmats", str(bundle / "rmats" / cid / "rmats_post"),
            "--fasta", str(fasta_p),
            "--gtf", str(gtf_p),
            "--species", "mouse",
            "--contrast-id", cid,
            "--splicenet-modules", str(sn_out / "module_assignments.tsv"),
            "--splicenet-scores", str(sn_out / "module_scores.tsv"),
            "--rbp-expression", str(bundle / "DEG/all.normalised_counts.tsv"),
            "--permutations", "299",
            "--output-dir", str(st_c),
        ])
        st_outs.append(str(st_c))

    if report_out.exists():
        shutil.rmtree(report_out)
    report_main([
        "bulk",
        "--bundle", str(bundle),
        "--rmats-mode", "JCEC",
        "--fasta", str(fasta_p),
        "--gtf", str(gtf_p),
        "--species", "mouse",
        "--study-name", "Katana-PRISM Bulk One-Click Reference Dummy Study (Mouse Cortical Splicing)",
        "--splicemanifold-dir", str(sm_out),
        "--splicenet-dir", str(sn_out),
        "--splicetrace-dirs", *st_outs,
        "--dummy",
        "--outdir", str(report_out),
    ])
    return json.loads((report_out / "report-manifest.json").read_text(encoding="utf-8"))


def run_sc_dummy_pipeline(fasta_p: Path, gtf_p: Path, event_catalog: list[dict], work_root: Path, report_out: Path, rng: np.random.Generator) -> dict:
    """Create a 240-cell, 8-donor, 3-population Single-Cell cohort + 2 subtype slices and execute:
    `splicemanifold single-cell` -> `scsplicenet` -> `scsplicetrace` (+ subtype slices) -> `katana-report single-cell --dummy`."""
    from splicemanifold.cli import main as sm_main
    from scsplicenet.cli import main as sc_sn_main
    from scsplicetrace.cli import main as sc_st_main
    from katana_studio.cli import report_main
    import subprocess

    sc_in = work_root / "sc_inputs"
    sc_in.mkdir(parents=True, exist_ok=True)

    donors = [
        ("Donor1", "Depol_Stim"), ("Donor2", "Depol_Stim"), ("Donor3", "Depol_Stim"), ("Donor4", "Depol_Stim"),
        ("Donor5", "Basal_Ctrl"), ("Donor6", "Basal_Ctrl"), ("Donor7", "Basal_Ctrl"), ("Donor8", "Basal_Ctrl"),
    ]
    cell_types = ["Excitatory_L23", "Parvalbumin_GABA", "Astrocytes"]
    cell_rows = []
    for d_id, cond in donors:
        for ct in cell_types:
            for k in range(10):  # 8 * 3 * 10 = 240 cells
                cid = f"{d_id}_{ct[:3]}_{k+1:02d}"
                cell_rows.append({"cell_id": cid, "donor_id": d_id, "condition": cond, "cell_type": ct, "qc_pass": True})
    cells_df = pd.DataFrame(cell_rows)
    cells_tsv = sc_in / "sc_cells.tsv"
    cells_df.to_csv(cells_tsv, sep="\t", index=False)

    ev_rows = []
    eg_rows = []
    for ev in event_catalog[:120]:
        et = ev["event_type"]
        eid = f"{et}:chr1:+:{ev['exonStart_0base']}:{ev['exonEnd']}:{ev['upstreamES']}:{ev['upstreamEE']}:{ev['downstreamES']}:{ev['downstreamEE']}"
        row_e = {
            "event_id": eid, "gene_id": ev["gene_id"], "gene_symbol": ev["gene_symbol"], "event_type": et,
            "chr": "chr1", "strand": "+", "comparison_id": "Depol_Stim_vs_Basal_Ctrl",
            "exonStart_0base": ev["exonStart_0base"], "exonEnd": ev["exonEnd"],
            "upstreamES": ev["upstreamES"], "upstreamEE": ev["upstreamEE"],
            "downstreamES": ev["downstreamES"], "downstreamEE": ev["downstreamEE"],
            "longExonStart_0base": ev["longExonStart_0base"], "longExonEnd": ev["longExonEnd"],
            "shortES": ev["shortES"], "shortEE": ev["shortEE"],
            "flankingES": ev["flankingES"], "flankingEE": ev["flankingEE"],
            "1stExonStart_0base": ev["1stExonStart_0base"], "1stExonEnd": ev["1stExonEnd"],
            "2ndExonStart_0base": ev["2ndExonStart_0base"], "2ndExonEnd": ev["2ndExonEnd"],
            "riExonStart_0base": ev["riExonStart_0base"], "riExonEnd": ev["riExonEnd"],
        }
        ev_rows.append(row_e)
        eg_rows.append({"event_id": eid, "gene_id": ev["gene_id"], "gene_symbol": ev["gene_symbol"]})

    ev_ids = [r["event_id"] for r in ev_rows]
    c_ids = cells_df["cell_id"].tolist()
    psi_mat = np.full((len(c_ids), len(ev_ids)), np.nan, dtype=float)
    sup_mat = np.zeros((len(c_ids), len(ev_ids)), dtype=float)

    for j, ev in enumerate(event_catalog[:120]):
        for i, cr in enumerate(cell_rows):
            if rng.random() < 0.14:  # realistic Smart-seq3 / Xpress-seq dropout
                continue
            sup = float(rng.integers(5, 32))
            p_mu = 0.32 if ev["reg_group"] == "rbfox_pos" else (0.66 if ev["reg_group"] == "ptbp_neg" else 0.50)
            if ev["reg_group"] == "rbfox_pos" and cr["condition"] == "Depol_Stim":
                p_mu += 0.36
            elif ev["reg_group"] == "ptbp_neg" and cr["condition"] == "Depol_Stim":
                p_mu -= 0.30
            if cr["cell_type"] == "Parvalbumin_GABA" and j % 3 == 0:
                p_mu = np.clip(p_mu + 0.18, 0.05, 0.95)
            elif cr["cell_type"] == "Astrocytes" and j % 3 == 1:
                p_mu = np.clip(p_mu - 0.22, 0.05, 0.95)
            inc = rng.binomial(int(sup), np.clip(p_mu + rng.normal(0.0, 0.04), 0.02, 0.98))
            psi_mat[i, j] = round(inc / sup, 4)
            sup_mat[i, j] = sup

    # Preliminary dpsi / pvalue / fdr on sc_events.tsv
    stim_m = cells_df["condition"].eq("Depol_Stim").to_numpy()
    ctrl_m = cells_df["condition"].eq("Basal_Ctrl").to_numpy()
    for j, r_e in enumerate(ev_rows):
        v1 = psi_mat[stim_m, j]; v1 = v1[np.isfinite(v1)]
        v2 = psi_mat[ctrl_m, j]; v2 = v2[np.isfinite(v2)]
        r_e["dpsi"] = round(float(v1.mean() - v2.mean()), 4) if len(v1) and len(v2) else 0.0
        pv = float(stats.ttest_ind(v1, v2, equal_var=False).pvalue) if len(v1) >= 3 and len(v2) >= 3 else 1.0
        r_e["pvalue"] = pv if np.isfinite(pv) else 1.0
        r_e["fdr"] = min(1.0, r_e["pvalue"] * 4.0)

    ev_tsv = sc_in / "sc_events.tsv"
    eg_tsv = sc_in / "sc_event_genes.tsv"
    psi_gz = sc_in / "sc_psi.tsv.gz"
    sup_gz = sc_in / "sc_support.tsv.gz"
    expr_gz = sc_in / "sc_expression.tsv.gz"
    pd.DataFrame(ev_rows).to_csv(ev_tsv, sep="\t", index=False)
    pd.DataFrame(eg_rows).to_csv(eg_tsv, sep="\t", index=False)
    pd.DataFrame(psi_mat, index=c_ids, columns=ev_ids).to_csv(psi_gz, sep="\t", index_label="cell_id", compression="gzip")
    pd.DataFrame(sup_mat, index=c_ids, columns=ev_ids).to_csv(sup_gz, sep="\t", index_label="cell_id", compression="gzip")

    # Host expression matrix (`cells x genes`, log1p-counts / UMI) with cell-type markers & activity IEGs
    g_syms = [s for _, s in GENES_MOUSE]
    expr_mat = np.zeros((len(c_ids), len(g_syms)), dtype=float)
    for i, cr in enumerate(cell_rows):
        for g_i, gsym in enumerate(g_syms):
            lam = 3.5
            if gsym in {"Fos", "Arc", "Bdnf", "Egr1", "Npas4"} and cr["condition"] == "Depol_Stim":
                lam = 28.0
            elif gsym in {"Grin1", "Camk2a", "Dlg4", "Nrxn1"} and cr["cell_type"] == "Excitatory_L23":
                lam = 24.0
            elif gsym in {"Rbfox1", "Scn8a", "Syt1", "Kcnq2"} and cr["cell_type"] == "Parvalbumin_GABA":
                lam = 26.0
            elif gsym in {"Qki", "Ptbp1", "Clta", "Sptbn1"} and cr["cell_type"] == "Astrocytes":
                lam = 22.0
            expr_mat[i, g_i] = round(float(np.log2(rng.poisson(lam) + 1.0)), 4)
    pd.DataFrame(expr_mat, index=c_ids, columns=g_syms).to_csv(expr_gz, sep="\t", index_label="cell_id", compression="gzip")

    # Execute the exact Nextflow Single-Cell downstream stages!
    sm_sc_out = work_root / "sc_splicemanifold"
    sn_sc_out = work_root / "sc_scsplicenet"
    st_sc_out = work_root / "sc_scsplicetrace"

    sm_main([
        "single-cell",
        "--psi", str(psi_gz),
        "--support", str(sup_gz),
        "--metadata", str(cells_tsv),
        "--events", str(ev_tsv),
        "--expression", str(expr_gz),
        "--cells-as-rows",
        "--outdir", str(sm_sc_out),
    ])
    sc_sn_main([
        "--usage", str(psi_gz),
        "--metadata", str(cells_tsv),
        "--event-genes", str(eg_tsv),
        "--support", str(sup_gz),
        "--host-expr", str(expr_gz),
        "--min-module-size", "8",
        "--out", str(sn_sc_out),
    ])
    sc_st_main([
        "--events", str(ev_tsv),
        "--psi", str(psi_gz),
        "--support", str(sup_gz),
        "--metadata", str(cells_tsv),
        "--expression", str(expr_gz),
        "--fasta", str(fasta_p),
        "--gtf", str(gtf_p),
        "--species", "mouse",
        "--numerator", "Depol_Stim",
        "--denominator", "Basal_Ctrl",
        "--splicenet-modules", str(sn_sc_out / "scsplicenet_modules.tsv"),
        "--permutations", "299",
        "--out", str(st_sc_out),
    ])

    # Also run all 3 cell-subtype downstream contrasts (`SC_SUBTYPE_DOWNSTREAM` equivalent) so two-tier scope bar & Ch 04 stratification are verified!
    sub_dirs = []
    for pop in ("Excitatory_L23", "Parvalbumin_GABA", "Astrocytes"):
        cid_sub = f"{pop}__Depol_Stim_vs_Basal_Ctrl"
        sd = work_root / "sc_subtypes" / cid_sub
        (sd / "sc_matrices").mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                sys.executable, str(ROOT / "bin/katana_sc_slice_contrast.py"),
                "--events", str(ev_tsv), "--psi", str(psi_gz), "--support", str(sup_gz),
                "--metadata", str(cells_tsv), "--expression", str(expr_gz),
                "--population-col", "cell_type", "--population-value", pop,
                "--group-col", "condition", "--numerator", "Depol_Stim", "--denominator", "Basal_Ctrl",
                "--min-cells", "8", "--outdir", str(sd / "sc_matrices"),
            ],
            check=True,
        )
        sc_st_main([
            "--events", str(sd / "sc_matrices/sc_events.tsv"),
            "--psi", str(sd / "sc_matrices/sc_psi.tsv.gz"),
            "--support", str(sd / "sc_matrices/sc_support.tsv.gz"),
            "--metadata", str(sd / "sc_matrices/sc_cells.tsv"),
            "--expression", str(sd / "sc_matrices/sc_expression.tsv.gz"),
            "--fasta", str(fasta_p),
            "--gtf", str(gtf_p),
            "--species", "mouse",
            "--numerator", "Depol_Stim",
            "--denominator", "Basal_Ctrl",
            "--permutations", "299",
            "--out", str(sd / "scsplicetrace"),
        ])
        sub_dirs.append(str(sd))

    # Generate 8 real donor-level indexed BAMs (`sc_bams/{donor}.bam` + `.bai`) for authentic Single-Cell BAM RPKM Native Sashimi!
    sc_bam_dir = work_root / "sc_bams"
    sc_bam_dir.mkdir(parents=True, exist_ok=True)
    sc_locus_biases = [
        (
            float(rng.uniform(0.70, 1.35)),
            float(rng.uniform(0.85, 1.58)),
            float(rng.uniform(1.15, 2.55)),
            float(rng.uniform(1.15, 2.55)),
            float(rng.uniform(1.15, 2.55)),
            float(rng.uniform(1.15, 2.55)),
        )
        for _ in event_catalog[:120]
    ]
    for d_id, _ in donors:
        d_mask = cells_df["donor_id"].eq(d_id).to_numpy()
        d_reads: list[tuple[int, tuple[tuple[int, int], ...]]] = []
        for j, ev in enumerate(event_catalog[:120]):
            vals_d = psi_mat[d_mask, j]
            vals_d = vals_d[np.isfinite(vals_d)]
            mean_p = float(vals_d.mean()) if len(vals_d) else 0.5
            d_reads.extend(
                _simulate_asymmetric_event_reads(
                    ev, mean_p, junc_depth=float(rng.uniform(42.0, 96.0)), exon_bias=sc_locus_biases[j], rng=rng
                )
            )
        _write_synthetic_indexed_bam(sc_bam_dir / f"{d_id}.bam", 960_000, d_reads, reported_mapped_reads=int(rng.integers(7_200_000, 9_800_000)))

    if report_out.exists():
        shutil.rmtree(report_out)
    report_main([
        "single-cell",
        "--events", str(ev_tsv),
        "--cells", str(cells_tsv),
        "--psi-matrix", str(psi_gz),
        "--support-matrix", str(sup_gz),
        "--expression", str(expr_gz),
        "--gtf", str(gtf_p),
        "--bams", str(sc_bam_dir),
        "--splicemanifold-dir", str(sm_sc_out),
        "--scsplicenet-dir", str(sn_sc_out),
        "--scsplicetrace-dir", str(st_sc_out),
        "--subtypes-dirs", *sub_dirs,
        "--species", "mouse",
        "--study-name", "Katana-PRISM Single-Cell One-Click Reference Dummy Study (Mouse Neuronal Subtypes)",
        "--dummy",
        "--outdir", str(report_out),
    ])
    return json.loads((report_out / "report-manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    import os
    import tempfile
    work_root = Path(os.environ.get("TEMP") or tempfile.gettempdir()) / "katana_prism_dummy_pipeline_work"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    out_root = ROOT / "dummy_inspection_reports"
    out_root.mkdir(parents=True, exist_ok=True)
    bulk_report_dir = out_root / "Bulk_OneClick_Dummy_Report"
    sc_report_dir = out_root / "SingleCell_OneClick_Dummy_Report"

    rng = np.random.default_rng(20261121)
    print("[1/3] Building synthetic indexed GRCm39 locus (FASTA + FAI + multi-isoform GTF)...")
    fasta_p, gtf_p, catalog = build_mini_reference_and_events(work_root, rng)

    print("[2/3] Running End-to-End Bulk Dummy Pipeline (SpliceManifold -> SpliceNet -> SpliceTrace -> Katana-Report)...")
    m_bulk = run_bulk_dummy_pipeline(fasta_p, gtf_p, catalog, work_root, bulk_report_dir, rng)

    print("[3/3] Running End-to-End Single-Cell Dummy Pipeline (SpliceManifold-SC -> scSpliceNet -> scSpliceTrace -> Katana-Report-SC)...")
    m_sc = run_sc_dummy_pipeline(fasta_p, gtf_p, catalog, work_root, sc_report_dir, rng)

    # Clean intermediate temporary workspace
    shutil.rmtree(work_root, ignore_errors=True)

    for label, rdir, man in (("BULK", bulk_report_dir, m_bulk), ("SINGLE-CELL", sc_report_dir, m_sc)):
        html_txt = (rdir / "REPORT.html").read_text(encoding="utf-8")
        no_res_cnt = html_txt.count("No completed result available")
        miss_plot = man.get("figure_data_links", {}).get("missing_plot_data", [])
        ch_counts = {}
        for p in man.get("panels", []):
            ch_counts[p["chapter"]] = ch_counts.get(p["chapter"], 0) + 1
        bad_ch3 = [
            p["id"] for p in man.get("panels", [])
            if p.get("chapter") == 3 and any(tok in p["id"].lower() for tok in ("suppa", "dexseq", "dtu", "four-layer"))
        ]
        has_net_exp = (rdir / "outputs/network/browser/SPLICE_NETWORK_EXPLORER.html").is_file()
        has_old_circ = (rdir / "outputs/network/browser/ALL_MODULES.html").exists()
        has_wb = (rdir / "outputs/synthesis/CANDIDATE_GENE_WORKBENCH.html").is_file() and ("interactive-candidate-gene-workbench" in html_txt)
        sashimi_svgs = list((rdir / "outputs/synthesis/sashimi").glob("*.svg")) if (rdir / "outputs/synthesis/sashimi").is_dir() else []
        ch4_conv = [p["id"] for p in man.get("panels", []) if p.get("chapter") == 4 and "multilayer-orthogonal-convergence" in p["id"]]
        ch8_conv = [p["id"] for p in man.get("panels", []) if p.get("chapter") == 8 and "multilayer-orthogonal-convergence" in p["id"]]
        sc_umaps = [p["id"] for p in man.get("panels", []) if "umap" in p["id"].lower()]
        legacy_hits = [tok for tok in ("Katana-Splice-Network", "CoSplice", "MotifTrace") if tok in html_txt]
        has_two_tier_sc = ("expContrastSelect" in html_txt) and ("subtypeSelect" in html_txt) and ("subtypePills" in html_txt)
        n_bam_rpkm_sashimi = sum(1 for sp in sashimi_svgs if "RPKM" in sp.read_text(encoding="utf-8", errors="ignore"))
        print(
            f"[{label}] panels={len(man['panels'])} tables={len(man['complete_results'])} "
            f"chapter_panels={dict(sorted(ch_counts.items()))} "
            f"no_result_boxes={no_res_cnt} missing_plot_data={len(miss_plot)} "
            f"bad_ch3={bad_ch3} ch4_multilayer_conv={len(ch4_conv)} ch8_multilayer_conv={len(ch8_conv)} "
            f"sc_umaps={sc_umaps} two_tier_sc={has_two_tier_sc} legacy_hits={legacy_hits} "
            f"native_sashimi_svgs={len(sashimi_svgs)} (bam_rpkm={n_bam_rpkm_sashimi}) "
            f"PSI_MATRIX={(rdir / 'outputs/landscape/PSI_MATRIX.html').is_file()} "
            f"SPLICE_NETWORK_EXPLORER={has_net_exp} (legacy_circular={has_old_circ}) "
            f"CH8_WORKBENCH={has_wb} "
            f"zip_MB={(rdir / 'REPORT.zip').stat().st_size / (1024 * 1024):.2f}"
        )
        if (
            no_res_cnt > 0
            or len(miss_plot) > 0
            or set(ch_counts.keys()) != {1, 2, 3, 4, 5, 6, 7, 8}
            or bool(bad_ch3)
            or len(ch4_conv) > 0
            or bool(legacy_hits)
            or not has_net_exp
            or has_old_circ
            or not has_wb
            or len(sashimi_svgs) < 5
            or n_bam_rpkm_sashimi < len(sashimi_svgs)
            or (label == "BULK" and len(ch8_conv) == 0)
            or (label == "SINGLE-CELL" and (len(sc_umaps) != 1 or not has_two_tier_sc))
        ):
            raise AssertionError(
                f"{label} audit failed: chapters={sorted(ch_counts.keys())}, no_res_cnt={no_res_cnt}, "
                f"miss_plot={miss_plot}, bad_ch3={bad_ch3}, ch4_conv={len(ch4_conv)}, ch8_conv={len(ch8_conv)}, "
                f"sc_umaps={sc_umaps}, two_tier_sc={has_two_tier_sc}, legacy_hits={legacy_hits}, "
                f"sashimi={len(sashimi_svgs)} (bam_rpkm={n_bam_rpkm_sashimi}), net={has_net_exp}, wb={has_wb}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
