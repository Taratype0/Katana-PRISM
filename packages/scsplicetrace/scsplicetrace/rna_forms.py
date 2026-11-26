"""Standalone 5-event mature RNA reconstruction, flank extraction, and junction consequence tracing."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

EVENT_TYPES = ("SE", "A3SS", "A5SS", "MXE", "RI")
_COMPLEMENT = bytes.maketrans(b"ACGTRYKMSWBDHVNacgtrykmswbdhvn", b"TGCAYRMKSWVHDBNtgcayrmkswvhdbn")


import contextlib
import mmap

class IndexedFasta:
    """Zero-dependency mmap-accelerated indexed FASTA reader supporting .fai random access."""

    def __init__(self, fasta_path: Path) -> None:
        self.fasta = Path(fasta_path)
        fai = Path(str(self.fasta) + ".fai")
        if not self.fasta.is_file() or not fai.is_file():
            raise FileNotFoundError(f"Indexed FASTA and .fai required: {self.fasta}")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        self.aliases: dict[str, str] = {}
        self._validated_contigs: set[str] = set()
        self._file_size: int = int(self.fasta.stat().st_size)
        for line in fai.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                raise ValueError(f"Malformed .fai entry in {fai}: {line!r}")
            contig = parts[0]
            self.index[contig] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))
            self.aliases[contig] = contig
            m = re.search(r"(GL\d+|KI\d+)v(\d+)", contig)
            if m:
                alias = f"chr{m.group(1)}.{m.group(2)}"
                if alias not in self.aliases:
                    self.aliases[alias] = contig

    @staticmethod
    def _read_slice(stream, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if isinstance(stream, (mmap.mmap, bytes, bytearray, memoryview)):
            return bytes(stream[start : start + length])
        stream.seek(start)
        return stream.read(length)

    def _validate_contig_index(self, stream, contig: str) -> None:
        if contig in self._validated_contigs:
            return
        seq_len, offset, line_bases, line_width = self.index[contig]
        if seq_len <= 0 or offset <= 0 or line_bases <= 0 or line_width < line_bases:
            raise ValueError(f"Invalid .fai geometry for {contig} in {self.fasta}: {self.index[contig]}")
        eol_len = line_width - line_bases
        if eol_len not in (0, 1, 2):
            raise ValueError(f"Unsupported .fai line_width={line_width} vs line_bases={line_bases} for {contig}")
        last_base_byte = offset + ((seq_len - 1) // line_bases) * line_width + ((seq_len - 1) % line_bases)
        if offset >= self._file_size or last_base_byte >= self._file_size:
            raise ValueError(
                f"Stale or truncated .fai index for {contig} in {self.fasta}: "
                f"last_base_byte={last_base_byte} >= file_size={self._file_size}"
            )
        hdr_win = min(offset, 4096)
        pre = self._read_slice(stream, offset - hdr_win, hdr_win)
        if not pre.endswith(b"\n"):
            raise ValueError(f"Stale .fai offset={offset} for {contig} in {self.fasta} (not preceded by header newline)")
        last_nl = pre.rfind(b"\n", 0, len(pre) - 1)
        hdr_line = pre[last_nl + 1 : -1].rstrip(b"\r")
        if not hdr_line.startswith(b">"):
            raise ValueError(f"Stale .fai offset={offset} for {contig} in {self.fasta} (missing FASTA '>' header)")
        hdr_tok = hdr_line[1:].strip().split(None, 1)[0] if hdr_line[1:].strip() else b""
        if hdr_tok != contig.encode("ascii"):
            raise ValueError(
                f"Stale .fai header mismatch for {contig} in {self.fasta}: found {hdr_tok!r}"
            )
        if seq_len > line_bases:
            if eol_len not in (1, 2):
                raise ValueError(f"Multi-line FASTA {contig} requires EOL length 1 or 2 in .fai (got {eol_len})")
            first_line = self._read_slice(stream, offset, line_width)
            if len(first_line) != line_width:
                raise ValueError(f"Truncated sequence line at {contig} in {self.fasta}")
            seq_part = first_line[:line_bases]
            if b"\r" in seq_part or b"\n" in seq_part or b">" in seq_part:
                raise ValueError(
                    f"Stale .fai line wrapping for {contig} in {self.fasta} (EOL or header encountered inside sequence bases)"
                )
            exp_eol = b"\n" if eol_len == 1 else b"\r\n"
            if first_line[line_bases:] != exp_eol:
                raise ValueError(
                    f"Stale .fai EOL mismatch for {contig} in {self.fasta}: expected {exp_eol!r}, got {first_line[line_bases:]!r}"
                )
        last_base = self._read_slice(stream, last_base_byte, 1)
        if last_base in (b"\r", b"\n", b">", b""):
            raise ValueError(f"Stale .fai end position for {contig} in {self.fasta}: unexpected byte {last_base!r}")
        tail = self._read_slice(stream, last_base_byte + 1, min(4, self._file_size - (last_base_byte + 1)))
        if tail:
            if tail.startswith(b"\r\n"):
                after_eol = tail[2:]
            elif tail.startswith(b"\n"):
                after_eol = tail[1:]
            else:
                raise ValueError(f"Stale .fai line/length alignment at end of {contig} in {self.fasta}: {tail!r}")
            if after_eol and not after_eol.startswith((b">", b"\r", b"\n")):
                raise ValueError(f"Unexpected sequence continuation past indexed length of {contig} in {self.fasta}")
        self._validated_contigs.add(contig)

    @contextlib.contextmanager
    def open_stream(self):
        with self.fasta.open("rb") as fobj:
            try:
                mm = mmap.mmap(fobj.fileno(), 0, access=mmap.ACCESS_READ)
            except Exception:
                mm = None
            try:
                yield mm if mm is not None else fobj
            finally:
                if mm is not None:
                    mm.close()

    def resolve_chrom(self, chrom: str) -> str:
        if chrom in self.index:
            return chrom
        if chrom in self.aliases:
            return self.aliases[chrom]
        alt = chrom[3:] if chrom.startswith("chr") else "chr" + chrom
        if alt in self.index:
            return alt
        if alt in self.aliases:
            return self.aliases[alt]
        raise KeyError(f"Chromosome {chrom!r} absent from {self.fasta}")

    def fetch_blocks(self, stream, chrom: str, strand: str, blocks: list[tuple[int, int]]) -> str:
        contig = self.resolve_chrom(chrom)
        self._validate_contig_index(stream, contig)
        seq_len, offset, line_bases, line_width = self.index[contig]
        pieces: list[bytes] = []
        is_buf = isinstance(stream, (mmap.mmap, bytes, bytearray, memoryview))
        for a, b in blocks:
            if a < 0 or b <= a or b > seq_len:
                raise ValueError(f"Out-of-bounds block ({a}, {b}) on {contig} (len={seq_len})")
            start_byte = offset + (a // line_bases) * line_width + (a % line_bases)
            end_byte = offset + ((b - 1) // line_bases) * line_width + ((b - 1) % line_bases) + 1
            if is_buf:
                raw_slice = bytes(stream[start_byte:end_byte])
            else:
                stream.seek(start_byte)
                raw_slice = stream.read(end_byte - start_byte)
            raw = raw_slice.replace(b"\r", b"").replace(b"\n", b"")
            if b">" in raw or len(raw) != (b - a):
                raise ValueError(
                    f"Corrupt or stale .fai sequence read for {contig}:({a},{b}) in {self.fasta} "
                    f"(expected {b - a} bases, got {len(raw)})"
                )
            if strand == "-":
                raw = raw.translate(_COMPLEMENT)[::-1]
            pieces.append(raw.upper())
        return b"".join(pieces).decode("ascii")


def tx_slice(block: tuple[int, int], strand: str, start0: int, end0: int) -> tuple[int, int]:
    a, b = block
    if strand == "+":
        return (a + start0, a + end0)
    return (b - end0, b - start0)


def coalesce(blocks: list[tuple[int, int]], strand: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for a, b in sorted(blocks):
        if result and a < result[-1][1]:
            raise ValueError("Overlapping blocks within a local RNA form")
        if result and a == result[-1][1]:
            result[-1] = (result[-1][0], b)
        else:
            result.append((a, b))
    return result if strand == "+" else result[::-1]


def subtract_blocks(first: list[tuple[int, int]], second: list[tuple[int, int]], strand: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for a, b in first:
        pieces = [(a, b)]
        for c, d in second:
            pieces = (
                [(x, min(y, c)) for x, y in pieces if x < min(y, c)]
                + [(max(x, d), y) for x, y in pieces if max(x, d) < y]
            )
        output.extend(pieces)
    return coalesce(output, strand)


def _interval(row: dict, left: str, right: str) -> tuple[int, int]:
    vals = np.asarray([row[left], row[right]], dtype=float)
    if not np.isfinite(vals).all() or (vals != np.floor(vals)).any():
        raise ValueError(f"Non-integer/missing coordinates: {left}/{right}")
    a, b = int(vals[0]), int(vals[1])
    if a < 0 or b <= a:
        raise ValueError(f"Empty/reversed interval: {left}/{right}")
    return a, b


def event_forms(row: dict, anchor_nt: int = 100) -> dict[str, list[tuple[int, int]]]:
    kind, strand = str(row["event_type"]), str(row["strand"])
    if kind not in EVENT_TYPES or strand not in {"+", "-"}:
        raise ValueError(f"Unsupported event_type={kind!r} or strand={strand!r}")
    get = lambda l, r: _interval(row, l, r)
    if kind in {"SE", "MXE", "RI"}:
        up, down = get("upstreamES", "upstreamEE"), get("downstreamES", "downstreamEE")
        if up[1] >= down[0]:
            raise ValueError("Flanking exons must have a non-empty intervening interval")
        u = (max(up[0], up[1] - anchor_nt), up[1])
        d = (down[0], min(down[1], down[0] + anchor_nt))
        if kind == "SE":
            target = get("exonStart_0base", "exonEnd")
            if not (up[1] < target[0] < target[1] < down[0]):
                raise ValueError("SE target exon must lie strictly between flanking exons")
            first, second = [u, target, d], [u, d]
        elif kind == "MXE":
            a, b = get("1stExonStart_0base", "1stExonEnd"), get("2ndExonStart_0base", "2ndExonEnd")
            if not (up[1] < a[0] < a[1] <= b[0] < b[1] < down[0]):
                raise ValueError("Invalid MXE exon ordering")
            inc, exc = (a, b) if strand == "+" else (b, a)
            first, second = [u, inc, d], [u, exc, d]
        else:
            retained = get("riExonStart_0base", "riExonEnd")
            if retained != (up[0], down[1]):
                raise ValueError("RI retained-exon coordinates disagree with flanks")
            first, second = [u, (up[1], down[0]), d], [u, d]
    else:
        long_ex = get("longExonStart_0base", "longExonEnd")
        short_ex = get("shortES", "shortEE")
        flank = get("flankingES", "flankingEE")
        suffix = (kind == "A5SS") == (strand == "+")
        if suffix:
            if not (long_ex[0] == short_ex[0] < short_ex[1] < long_ex[1] < flank[0]):
                raise ValueError("Alternative splice site geometry conflicts with type/strand")
            flank = (flank[0], min(flank[1], flank[0] + anchor_nt))
        else:
            if not (flank[1] < long_ex[0] < short_ex[0] < short_ex[1] == long_ex[1]):
                raise ValueError("Alternative splice site geometry conflicts with type/strand")
            flank = (max(flank[0], flank[1] - anchor_nt), flank[1])
        first, second = [long_ex, flank], [short_ex, flank]
    first, second = coalesce(first, strand), coalesce(second, strand)
    return {
        "inclusion": first,
        "exclusion": second,
        "affected_inclusion": subtract_blocks(first, second, strand),
        "affected_exclusion": subtract_blocks(second, first, strand),
    }


def regulatory_intron_specs(
    forms: dict[str, list[tuple[int, int]]],
    strand: str,
    intron_nt: int = 200,
    distal_intron_nt: int = 0,
    acceptor_core_nt: int = 20,
) -> list[tuple[str, list[tuple[int, int]]]]:
    """Return canonical donor/acceptor regulatory pre-mRNA intervals (plus optional separate [intron_nt..distal] regions).

    Setting `acceptor_core_nt=3` masks only the invariant 3'YAG consensus and preserves the [-20..-3] nt
    polypyrimidine tract (critical for U2AF2, PTBP1/2, PCBP2, TIA1/TIAL1, and HNRNPC).
    """
    acc_off = max(int(acceptor_core_nt), 3)
    specs: list[tuple[str, list[tuple[int, int]]]] = []
    for form in ("inclusion", "exclusion"):
        for i, (a, b) in enumerate(zip(forms[form], forms[form][1:]), 1):
            gap = (a[1], b[0]) if strand == "+" else (b[1], a[0])
            size = gap[1] - gap[0]
            if size > 6:
                specs.append((f"{form}_junction_{i}_donor_intron", [tx_slice(gap, strand, 6, min(size, intron_nt))]))
            if size > acc_off:
                specs.append((f"{form}_junction_{i}_acceptor_intron", [tx_slice(gap, strand, max(0, size - intron_nt), size - acc_off)]))
            if distal_intron_nt > intron_nt and size > intron_nt + acc_off + 5:
                d_end = min(size - acc_off, distal_intron_nt)
                if d_end > intron_nt:
                    specs.append((f"{form}_junction_{i}_donor_intron_distal", [tx_slice(gap, strand, intron_nt, d_end)]))
                a_start = max(6, size - distal_intron_nt)
                if size - intron_nt > a_start:
                    specs.append((f"{form}_junction_{i}_acceptor_intron_distal", [tx_slice(gap, strand, a_start, size - intron_nt)]))
    return specs


def extract_event_rna(
    events: pd.DataFrame,
    fasta_path: Path,
    *,
    anchor_nt: int = 100,
    intron_nt: int = 200,
    distal_intron_nt: int = 0,
    acceptor_core_nt: int = 20,
    feature_only: bool = False,
    compute_sha256: bool = True,
) -> pd.DataFrame:
    """Reconstruct mature, affected, and regulatory pre-mRNA sequences for all 5 event types."""
    if events.empty:
        return pd.DataFrame()
    if events["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id in events table")
    fasta = IndexedFasta(Path(fasta_path))
    rows: list[dict] = []
    with fasta.open_stream() as stream:
        for row in events.sort_values("event_id").to_dict("records"):
            forms = event_forms(row, anchor_nt=anchor_nt)
            chrom = fasta.resolve_chrom(str(row["chr"]))
            strand = str(row["strand"])
            specs: list[tuple[str, list[tuple[int, int]], str]] = [
                (name, blks, "mature_local_form" if name in {"inclusion", "exclusion"} else "affected_sequence")
                for name, blks in forms.items()
            ]
            if feature_only:
                specs = [s for s in specs if s[2] == "affected_sequence"]
            else:
                for form in ("inclusion", "exclusion"):
                    for i, (a, b) in enumerate(zip(forms[form], forms[form][1:]), 1):
                        alen, blen = a[1] - a[0], b[1] - b[0]
                        specs.append((
                            f"{form}_junction_{i}",
                            [tx_slice(a, strand, max(0, alen - anchor_nt), alen), tx_slice(b, strand, 0, min(anchor_nt, blen))],
                            "splice_junction",
                        ))
                for reg_name, reg_blocks in regulatory_intron_specs(
                    forms,
                    strand,
                    intron_nt=intron_nt,
                    distal_intron_nt=distal_intron_nt,
                    acceptor_core_nt=acceptor_core_nt,
                ):
                    specs.append((reg_name, reg_blocks, "regulatory_pre_mRNA"))

            for region, blocks, layer in specs:
                if not blocks:
                    continue
                seq = fasta.fetch_blocks(stream, chrom, strand, blocks)
                joins = np.cumsum([b - a for a, b in blocks])[:-1].tolist()
                rows.append({
                    "event_id": str(row["event_id"]),
                    "gene_id": str(row["gene_id"]),
                    "gene_symbol": str(row.get("gene_symbol", "")),
                    "comparison_id": str(row.get("comparison_id", "")),
                    "event_type": str(row["event_type"]),
                    "chr": chrom,
                    "strand": strand,
                    "region": region,
                    "layer": layer,
                    "blocks_json": json.dumps(blocks),
                    "junctions_json": json.dumps(joins),
                    "sequence": seq,
                    "sequence_RNA": seq.replace("T", "U"),
                    "sequence_sha256": hashlib.sha256(seq.encode("ascii")).hexdigest() if compute_sha256 else "",
                    "length": len(seq),
                })
    return pd.DataFrame(rows)


def compute_sequence_gc(seq: str) -> tuple[float, float]:
    s = seq.upper().replace("U", "T")
    if not s:
        return np.nan, 1.0
    valid = s.count("A") + s.count("C") + s.count("G") + s.count("T")
    ambig = 1.0 - valid / len(s)
    gc = (s.count("G") + s.count("C")) / valid if valid > 0 else np.nan
    return gc, ambig


def compute_flank_gc_fast(
    events: pd.DataFrame, fasta_path: Path, *, anchor_nt: int = 100, intron_nt: int = 200
) -> pd.Series:
    """Fast sequential-IO computation of proximal intronic flank GC across large event pools."""
    fasta = IndexedFasta(Path(fasta_path))
    tasks: list[tuple[str, int, int, str]] = []
    for row in events.to_dict("records"):
        eid = str(row["event_id"])
        strand = str(row["strand"])
        chrom = fasta.resolve_chrom(str(row["chr"]))
        forms = event_forms(row, anchor_nt=anchor_nt)
        seen_intervals: set[tuple[int, int]] = set()
        for _, blks in regulatory_intron_specs(forms, strand, intron_nt=intron_nt):
            for a, b in blks:
                # GC content is strand-invariant (G+C == C+G), avoiding reverse-complement overhead!
                if (a, b) not in seen_intervals:
                    seen_intervals.add((a, b))
                    tasks.append((chrom, a, b, eid))

    tasks.sort(key=lambda t: (t[0], t[1]))
    gc_num: dict[str, int] = {}
    gc_den: dict[str, int] = {}
    with fasta.open_stream() as stream:
        for chrom, a, b, eid in tasks:
            seq = fasta.fetch_blocks(stream, chrom, "+", [(a, b)])
            g_c = seq.count("G") + seq.count("C")
            tot = g_c + seq.count("A") + seq.count("T")
            gc_num[eid] = gc_num.get(eid, 0) + g_c
            gc_den[eid] = gc_den.get(eid, 0) + tot

    out = {
        str(eid): (gc_num[str(eid)] / gc_den[str(eid)] if gc_den.get(str(eid), 0) > 0 else np.nan)
        for eid in events["event_id"].astype(str)
    }
    return pd.Series(out, name="flank_GC")


def compute_flank_gc_from_rna(rna_df: pd.DataFrame) -> pd.Series:
    """Compute mean intronic flank GC per event from deduplicated regulatory_pre_mRNA blocks."""
    flanks = rna_df[rna_df["layer"].eq("regulatory_pre_mRNA")]
    if flanks.empty:
        return pd.Series(dtype=float)
    out: dict[str, float] = {}
    for eid, grp in flanks.groupby("event_id", sort=False):
        if "blocks_json" in grp.columns:
            grp = grp.drop_duplicates(subset=["blocks_json"])
        joined = "".join(grp["sequence"].astype(str))
        gc, _ = compute_sequence_gc(joined)
        out[str(eid)] = gc
    return pd.Series(out, name="flank_GC")


def _mapped_genomic_blocks(
    blocks: list[tuple[int, int]], strand: str, rna_start0: int, rna_end0: int
) -> list[tuple[int, int]]:
    """Map mature-RNA coordinates [rna_start0, rna_end0) back to genomic 0-based blocks."""
    cursor = 0
    mapped: list[tuple[int, int]] = []
    for a, b in blocks:
        seg_len = b - a
        ov_s = max(rna_start0, cursor) - cursor
        ov_e = min(rna_end0, cursor + seg_len) - cursor
        if ov_e > ov_s:
            mapped.append(tx_slice((a, b), strand, ov_s, ov_e))
        cursor += seg_len
    return coalesce(mapped, strand)


def trace_motif_instances(
    rna_df: pd.DataFrame,
    events_df: pd.DataFrame,
    models: dict[str, dict],
    *,
    relative_likelihood_cutoff: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trace per-event motif loci across mature inclusion/exclusion forms (including junction-spanning sites)."""
    mature = rna_df[rna_df["layer"].eq("mature_local_form")]
    if mature.empty:
        return pd.DataFrame(), pd.DataFrame()

    ev_lookup = events_df.set_index("event_id")
    encode_tbl = np.full(256, 4, dtype=np.uint8)
    encode_tbl[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)

    precomp = {}
    for mid, m in models.items():
        mat = (20.0 * np.asarray(m["matrix"], dtype=float) + 0.025) / 20.1
        odds = mat / 0.25
        rel = mat / mat.max(axis=1, keepdims=True)
        raw_r = m.get("RBPs", "")
        rbp_clean = ";".join(str(x) for x in raw_r) if isinstance(raw_r, (list, tuple)) else str(raw_r)
        precomp[mid] = (len(mat), odds, rel, rbp_clean)

    hit_rows: list[dict] = []
    change_rows: list[dict] = []

    for eid, grp in mature.groupby("event_id", sort=False):
        if eid not in ev_lookup.index:
            continue
        dpsi = float(ev_lookup.loc[eid, "dpsi"]) if "dpsi" in ev_lookup.columns else np.nan
        forms_list = grp.to_dict("records")
        for mid, (w, odds, rel, rbp_str) in precomp.items():
            loci: dict[str, dict[str, dict]] = {}
            for form in forms_list:
                seq = str(form["sequence"]).upper().replace("U", "T")
                n = max(0, len(seq) - w + 1)
                if not n:
                    continue
                enc = encode_tbl[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
                inv = np.r_[0, np.cumsum(enc == 4)]
                starts = np.flatnonzero((inv[w:] - inv[:-w]) == 0)
                if not len(starts):
                    continue
                v_odds = np.ones(len(starts), dtype=float)
                v_rel = np.ones(len(starts), dtype=float)
                for k in range(w):
                    v_odds *= odds[k, enc[starts + k]]
                    v_rel *= rel[k, enc[starts + k]]
                cand_idx = np.flatnonzero(v_rel >= relative_likelihood_cutoff)
                if not len(cand_idx):
                    continue
                blks = [tuple(x) for x in json.loads(form["blocks_json"])]
                joins = json.loads(form["junctions_json"])
                for ci in cand_idx:
                    st = int(starts[ci])
                    g_blocks = _mapped_genomic_blocks(blks, str(form["strand"]), st, st + w)
                    g_key = json.dumps(g_blocks)
                    hit = {
                        "event_id": str(eid),
                        "event_type": str(form["event_type"]),
                        "gene_id": str(form["gene_id"]),
                        "model_id": mid,
                        "RBP": rbp_str,
                        "form": str(form["region"]),
                        "RNA_start0": st,
                        "RNA_end0": st + w,
                        "hit_RNA": str(form["sequence_RNA"])[st : st + w],
                        "chr": str(form["chr"]),
                        "strand": str(form["strand"]),
                        "genomic_blocks_json": g_key,
                        "odds_score": float(v_odds[ci]),
                        "crosses_splice_join": any(st < j < st + w for j in joins),
                    }
                    hit_rows.append(hit)
                    loci.setdefault(g_key, {})[str(form["region"])] = hit

            for g_key, f_dict in loci.items():
                in_inc = "inclusion" in f_dict
                in_exc = "exclusion" in f_dict
                state = "retained_both" if (in_inc and in_exc) else ("inclusion_only" if in_inc else "exclusion_only")
                if state == "retained_both":
                    continue
                first = next(iter(f_dict.values()))
                delta_opp = dpsi * (int(in_inc) - int(in_exc)) if np.isfinite(dpsi) else np.nan
                change_rows.append({
                    "event_id": first["event_id"],
                    "event_type": first["event_type"],
                    "gene_id": first["gene_id"],
                    "model_id": mid,
                    "RBP": rbp_str,
                    "hit_RNA": first["hit_RNA"],
                    "genomic_blocks_json": g_key,
                    "crosses_splice_join": bool(first["crosses_splice_join"]),
                    "instance_state": state,
                    "delta_sequence_opportunity": delta_opp,
                    "consequence_direction": (
                        "gained_in_condition" if delta_opp > 0 else ("skipped_or_lost_in_condition" if delta_opp < 0 else "zero")
                    ),
                })

    return pd.DataFrame(hit_rows), pd.DataFrame(change_rows)
