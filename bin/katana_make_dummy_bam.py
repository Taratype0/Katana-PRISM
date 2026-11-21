#!/usr/bin/env python3
"""Write a minimal spec-compliant 1-record coordinate-sorted BGZF BAM file from GTF contigs
so `rMATS --b1 ... --statoff` extracts `fromGTF.{SE,MXE,RI,A5SS,A3SS}.txt` without requiring STAR indexes.
"""
from __future__ import annotations

import struct
import sys
import zlib

BGZF_EOF = bytes.fromhex("1f8b08040000000000ff0600424302001b0003000000000000000000")


def make_bgzf_block(payload: bytes) -> bytes:
    cobj = zlib.compressobj(6, zlib.DEFLATED, -15)
    cdata = cobj.compress(payload) + cobj.flush()
    bsize = 18 + len(cdata) + 8 - 1
    hdr = struct.pack("<BBBBIBBHBBHH", 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, bsize)
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    ftr = struct.pack("<II", crc, len(payload))
    return hdr + cdata + ftr


def main() -> int:
    gtf_path, out_bam = sys.argv[1], sys.argv[2]
    contigs: dict[str, int] = {}
    with open(gtf_path, "r", encoding="utf-8", errors="replace") as fh:
        for ln in fh:
            if ln and not ln.startswith("#"):
                pts = ln.split("\t")
                if len(pts) >= 5:
                    c, end1 = pts[0], int(pts[4])
                    if end1 > contigs.get(c, 0):
                        contigs[c] = end1
    clist = list(contigs.items())
    sam_hdr = (
        "@HD\tVN:1.6\tSO:coordinate\n"
        + "".join(f"@SQ\tSN:{c}\tLN:{mlen + 10000}\n" for c, mlen in clist)
    ).encode("ascii")

    bam_hdr = bytearray(b"BAM\x01")
    bam_hdr += struct.pack("<i", len(sam_hdr)) + sam_hdr + struct.pack("<i", len(clist))
    for c, mlen in clist:
        cb = c.encode("ascii") + b"\x00"
        bam_hdr += struct.pack("<i", len(cb)) + cb + struct.pack("<i", mlen + 10000)

    qname = b"r1\x00"
    cigar = struct.pack("<I", (50 << 4) | 0)
    seq_enc = bytes([0x11] * 25)
    qual = bytes([40] * 50)
    rec_body = (
        struct.pack("<iiBBHHHiiii", 0, 0, len(qname), 60, 4680, 1, 0, 50, -1, -1, 0)
        + qname
        + cigar
        + seq_enc
        + qual
    )
    rec = struct.pack("<i", len(rec_body)) + rec_body

    with open(out_bam, "wb") as out:
        out.write(make_bgzf_block(bytes(bam_hdr) + rec) + BGZF_EOF)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
