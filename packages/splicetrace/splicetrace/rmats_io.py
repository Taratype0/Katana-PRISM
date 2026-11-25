"""Direct rMATS directory, .tar.gz archive, TSV, and Parquet loader for splicetrace."""
from __future__ import annotations

import json
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd

EVENT_COORD_COLS = {
    "SE": ["exonStart_0base", "exonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"],
    "A3SS": ["longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"],
    "A5SS": ["longExonStart_0base", "longExonEnd", "shortES", "shortEE", "flankingES", "flankingEE"],
    "MXE": ["1stExonStart_0base", "1stExonEnd", "2ndExonStart_0base", "2ndExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"],
    "RI": ["riExonStart_0base", "riExonEnd", "upstreamES", "upstreamEE", "downstreamES", "downstreamEE"],
}


def _mode_file_order(rmats_mode: str) -> tuple[str, ...]:
    mode = str(rmats_mode or "JCEC").upper()
    if mode not in {"JCEC", "JC"}:
        raise ValueError(f"Unsupported rmats_mode: {rmats_mode}")
    if mode == "JC":
        return (".MATS.JC.txt", ".MATS.JCEC.txt", ".MATS.JunctionCountOnly.txt")
    return (".MATS.JCEC.txt", ".MATS.JC.txt", ".MATS.JunctionCountOnly.txt")


def _has_rmats_files(d: Path, mode: str = "JCEC") -> bool:
    if not d.is_dir():
        return False
    sufs = _mode_file_order(mode)
    return any((d / f"{k}{s}").is_file() for k in EVENT_COORD_COLS for s in sufs)


def _find_enclosing_bundle(src: Path) -> Path | None:
    """Locate an enclosing Katana bundle root (direct path or up to 4 ancestors) if present."""
    candidates = [src, *list(src.parents)[:4]]
    for cand in candidates:
        if (cand / "manifest.json").is_file() or any(
            (cand / rel).is_file()
            for rel in (
                "pipeline_info/contrastsheet.valid.csv",
                "inputs/contrastsheet.csv",
                "contrastsheet.csv",
            )
        ):
            return cand
    return None


def _resolve_bundle_context(
    bdir: Path,
    *,
    src_dir: Path | None,
    contrast_id: str | None,
    rmats_mode: str,
) -> tuple[Path | None, bool, str | None]:
    """Resolve contrast rMATS dir and biological orientation from bundle manifest/contrastsheet."""
    c_path = next(
        (
            bdir / rel
            for rel in (
                "pipeline_info/contrastsheet.valid.csv",
                "inputs/contrastsheet.csv",
                "contrastsheet.csv",
            )
            if (bdir / rel).is_file()
        ),
        None,
    )
    contrasts: list[tuple[str, str, str]] = []
    if c_path is not None:
        sep = "\t" if c_path.suffix.lower() in {".tsv", ".txt"} else ","
        c_df = pd.read_csv(c_path, sep=sep)
        for _, r in c_df.iterrows():
            cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id") or "").strip()
            num_l = str(r.get("treatment") or r.get("numerator") or r.get("group_a") or "").strip()
            den_l = str(r.get("control") or r.get("denominator") or r.get("group_b") or "").strip()
            if cid:
                contrasts.append((cid, num_l, den_l))

    manifest_comps: dict[str, dict[str, str]] = {}
    if (bdir / "manifest.json").is_file():
        try:
            bm = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
            for cm in bm.get("comparisons", []):
                if isinstance(cm, dict) and cm.get("comparison_id"):
                    c_key = str(cm["comparison_id"]).strip()
                    rec = {
                        "rmats_group1": str(cm.get("rmats_group1") or "").strip(),
                        "rmats_group2": str(cm.get("rmats_group2") or "").strip(),
                        "rmats_dir": str(cm.get("rmats_dir") or "").strip(),
                    }
                    manifest_comps[c_key] = rec
                    manifest_comps.setdefault(c_key.lower(), rec)
        except Exception:
            pass

    target_cid = contrast_id.strip() if contrast_id else None
    matched_c: tuple[str, str, str] | None = None
    if target_cid:
        matched_c = next((c for c in contrasts if c[0] == target_cid or c[0].lower() == target_cid.lower()), None)
    if matched_c is None and len(contrasts) == 1:
        matched_c = contrasts[0]

    eff_cid = matched_c[0] if matched_c is not None else target_cid
    num_lbl, den_lbl = (matched_c[1], matched_c[2]) if matched_c is not None else ("", "")

    mc = (manifest_comps.get(eff_cid) or manifest_comps.get(eff_cid.lower())) if eff_cid else None
    if mc is None and len(manifest_comps) == 1 and target_cid is None:
        mc = next(iter(manifest_comps.values()))

    resolved_dir: Path | None = None
    path_inverted = False
    if src_dir is not None and _has_rmats_files(src_dir, rmats_mode):
        resolved_dir = src_dir
        if num_lbl and den_lbl:
            dname = src_dir.parent.name if src_dir.name == "rmats_post" else src_dir.name
            if dname.lower() in {f"{den_lbl}-{num_lbl}".lower(), f"{den_lbl}_vs_{num_lbl}".lower()}:
                path_inverted = True
    elif eff_cid is not None:
        if mc and mc.get("rmats_dir"):
            raw_d = Path(mc["rmats_dir"])
            cand_m = raw_d if raw_d.is_absolute() else (bdir / raw_d)
            if _has_rmats_files(cand_m, rmats_mode):
                resolved_dir = cand_m
        if resolved_dir is None:
            roots = [bdir / "rmats", bdir / "rnasplicing/rmats"]
            fwd = [k for k in (eff_cid, f"{num_lbl}-{den_lbl}", f"{num_lbl}_vs_{den_lbl}") if k and k != "-"]
            rev = [k for k in (f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}") if k and k != "-"]
            for root in roots:
                if not root.is_dir():
                    continue
                for k in fwd:
                    for p in (root / k / "rmats_post", root / k):
                        if _has_rmats_files(p, rmats_mode):
                            resolved_dir = p
                            break
                    if resolved_dir is not None:
                        break
                if resolved_dir is not None:
                    break
                for k in rev:
                    for p in (root / k / "rmats_post", root / k):
                        if _has_rmats_files(p, rmats_mode):
                            resolved_dir = p
                            path_inverted = True
                            break
                    if resolved_dir is not None:
                        break
                if resolved_dir is not None:
                    break
        if resolved_dir is None and len(contrasts) > 1:
            raise FileNotFoundError(f"No rMATS contrast directory matched '{eff_cid}' under {bdir}")

    rev_dir = path_inverted
    if mc and mc.get("rmats_group1") and mc.get("rmats_group2") and num_lbl and den_lbl:
        rev_dir = (mc["rmats_group1"] == den_lbl and mc["rmats_group2"] == num_lbl)
    return resolved_dir, rev_dir, eff_cid


def _standardize_rmats_df(
    d: pd.DataFrame,
    kind: str,
    valid_contigs: set[str] | None,
    comparison_id: str,
    *,
    reverse_direction: bool = False,
) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    if valid_contigs is not None:
        d = d[d["chr"].astype(str).isin(valid_contigs) & d["strand"].isin(["+", "-"])].copy()
    else:
        d = d[d["strand"].isin(["+", "-"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["event_type"] = kind
    d["gene_id"] = d["GeneID"].astype(str).str.strip('"').str.split(".").str[0]
    d["gene_symbol"] = d.get("geneSymbol", d["gene_id"]).astype(str).str.strip('"')
    coord_cols = EVENT_COORD_COLS[kind]
    d["event_id"] = (
        kind + ":" + d["chr"].astype(str) + ":" + d["strand"].astype(str) + ":"
        + d[coord_cols].astype(str).agg(":".join, axis=1)
    )
    d = d.drop_duplicates(subset=["event_id"]).copy()
    d["fdr"] = pd.to_numeric(d["FDR"], errors="coerce")
    d["PValue"] = pd.to_numeric(d.get("PValue", d["FDR"]), errors="coerce")
    raw_dpsi = pd.to_numeric(d["IncLevelDifference"], errors="coerce")
    d["dpsi"] = -raw_dpsi if reverse_direction else raw_dpsi
    d["reverse_direction"] = bool(reverse_direction)

    base_col = "IncLevel1" if reverse_direction else "IncLevel2"
    r1_m, r2_m, base_p = [], [], []
    for a1, b1, a2, b2, inc_base in zip(
        d["IJC_SAMPLE_1"], d["SJC_SAMPLE_1"], d["IJC_SAMPLE_2"], d["SJC_SAMPLE_2"], d[base_col]
    ):
        x1 = np.fromstring(str(a1), sep=",")
        y1 = np.fromstring(str(b1), sep=",")
        x2 = np.fromstring(str(a2), sep=",")
        y2 = np.fromstring(str(b2), sep=",")
        p_b = np.fromstring(str(inc_base).replace("NA", "nan"), sep=",")
        if len(x1) >= 1 and len(y1) >= 1 and len(x2) >= 1 and len(y2) >= 1:
            r1_m.append(float(np.mean(x1 + y1)))
            r2_m.append(float(np.mean(x2 + y2)))
        else:
            r1_m.append(np.nan)
            r2_m.append(np.nan)
        fin_p = p_b[np.isfinite(p_b)]
        base_p.append(float(np.mean(fin_p)) if len(fin_p) >= 1 else np.nan)

    d["depth"] = np.minimum(r1_m, r2_m)
    d["baseline_PSI"] = base_p
    d["comparison_id"] = comparison_id
    return d


def load_rmats_measurable_events(
    rmats_source: Path | str,
    fasta_path: Path | str | None = None,
    *,
    min_depth: float = 10.0,
    comparison_id: str = "rmats_contrast",
    rmats_mode: str = "JCEC",
    reverse_direction: bool | None = None,
    bundle_dir: Path | str | None = None,
    contrast_id: str | None = None,
) -> pd.DataFrame:
    """Load measurable splicing events from an rMATS output directory, `.tar.gz` archive, or `.tsv`/`.parquet` file."""
    src = Path(rmats_source)
    prio_sufs = _mode_file_order(rmats_mode)
    valid_contigs: set[str] | None = None
    if fasta_path is not None:
        fai = Path(str(fasta_path) + ".fai")
        if fai.is_file():
            raw_contigs = {ln.split("\t")[0] for ln in fai.read_text(encoding="utf-8").splitlines() if ln.strip()}
            valid_contigs = set(raw_contigs)
            for c in raw_contigs:
                valid_contigs.add(c[3:] if c.startswith("chr") else f"chr{c}")

    eff_cid_query = contrast_id if contrast_id is not None else (comparison_id if comparison_id != "rmats_contrast" else None)
    eff_comp_id = contrast_id or comparison_id
    eff_rev = bool(reverse_direction) if reverse_direction is not None else False

    bdir_resolved = Path(bundle_dir) if bundle_dir is not None else (_find_enclosing_bundle(src) if src.exists() else None)
    if bdir_resolved is not None and (src.is_dir() or bundle_dir is not None):
        rdir_res, bundle_rev, bundle_cid = _resolve_bundle_context(
            bdir_resolved,
            src_dir=src if src.is_dir() else None,
            contrast_id=eff_cid_query,
            rmats_mode=rmats_mode,
        )
        if rdir_res is not None:
            src = rdir_res
        if reverse_direction is None:
            eff_rev = bundle_rev
        if comparison_id == "rmats_contrast" and bundle_cid:
            eff_comp_id = bundle_cid

    frames: list[pd.DataFrame] = []
    found_any_rmats_file = False
    if src.is_dir():
        eff_suf = next(
            (s for s in prio_sufs if any((src / f"{kind}{s}").is_file() for kind in EVENT_COORD_COLS)),
            prio_sufs[0],
        )
        for kind in EVENT_COORD_COLS:
            cand = src / f"{kind}{eff_suf}"
            if not cand.is_file():
                cand = next((src / f"{kind}{s}" for s in prio_sufs if (src / f"{kind}{s}").is_file()), None)
            if cand is not None and cand.is_file():
                found_any_rmats_file = True
                raw = pd.read_csv(cand, sep="\t", low_memory=False)
                frames.append(_standardize_rmats_df(raw, kind, valid_contigs, eff_comp_id, reverse_direction=eff_rev))
    elif str(src).endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(src, "r:*") as tar:
            best_by_kind: dict[str, tuple[int, tarfile.TarInfo]] = {}
            for member in tar:
                base = Path(member.name).name
                if not member.isfile() or base.startswith("._"):
                    continue
                for kind in EVENT_COORD_COLS:
                    for rank_p, suf in enumerate(prio_sufs):
                        if base == f"{kind}{suf}":
                            prev = best_by_kind.get(kind)
                            if prev is None or rank_p < prev[0]:
                                best_by_kind[kind] = (rank_p, member)
            for kind, (_, member) in best_by_kind.items():
                fobj = tar.extractfile(member)
                if fobj is not None:
                    found_any_rmats_file = True
                    raw = pd.read_csv(fobj, sep="\t", low_memory=False)
                    frames.append(_standardize_rmats_df(raw, kind, valid_contigs, eff_comp_id, reverse_direction=eff_rev))
    elif src.suffix == ".parquet":
        raw_p = pd.read_parquet(src)
        if "IncLevelDifference" in raw_p.columns and "dpsi" not in raw_p.columns and "event_type" in raw_p.columns:
            for kind, grp in raw_p.groupby("event_type"):
                if str(kind) in EVENT_COORD_COLS:
                    found_any_rmats_file = True
                    frames.append(_standardize_rmats_df(grp.copy(), str(kind), valid_contigs, eff_comp_id, reverse_direction=eff_rev))
        else:
            return raw_p
    else:
        raw_t = pd.read_csv(src, sep="\t", low_memory=False)
        if "IncLevelDifference" in raw_t.columns and "dpsi" not in raw_t.columns and "event_type" in raw_t.columns:
            for kind, grp in raw_t.groupby("event_type"):
                if str(kind) in EVENT_COORD_COLS:
                    found_any_rmats_file = True
                    frames.append(_standardize_rmats_df(grp.copy(), str(kind), valid_contigs, eff_comp_id, reverse_direction=eff_rev))
        else:
            return raw_t

    frames = [f for f in frames if not f.empty]
    if not frames:
        if found_any_rmats_file:
            return pd.DataFrame(columns=["event_id", "gene_id", "gene_symbol", "event_type", "chr", "strand", "fdr", "PValue", "dpsi", "depth", "baseline_PSI", "comparison_id", "reverse_direction"])
        raise ValueError(f"No valid rMATS {rmats_mode.upper()} event tables found in {src}")
    all_ev = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["event_id"])
    ok = (
        all_ev["fdr"].notna()
        & all_ev["dpsi"].notna()
        & all_ev["baseline_PSI"].notna()
        & (all_ev["depth"] >= float(min_depth))
        & (all_ev["gene_id"] != "")
    )
    return all_ev.loc[ok].reset_index(drop=True)
