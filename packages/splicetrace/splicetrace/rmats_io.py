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
    return (f".MATS.{mode}.txt",)


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


def _find_expected_dir_for_contrast(
    bdir: Path,
    eff_cid: str,
    num_lbl: str,
    den_lbl: str,
    mc: dict[str, str] | None,
    rmats_mode: str,
) -> tuple[Path | None, bool]:
    if mc and mc.get("rmats_dir"):
        raw_d = Path(mc["rmats_dir"])
        cand_m = raw_d if raw_d.is_absolute() else (bdir / raw_d)
        if _has_rmats_files(cand_m, rmats_mode):
            return cand_m, False
        raise FileNotFoundError(f"Manifest rmats_dir for contrast '{eff_cid}' lacks *.MATS.{rmats_mode.upper()}.txt files: {cand_m}")
    roots = [bdir / "rmats", bdir / "rnasplicing/rmats"]
    fwd = [k for k in (eff_cid, f"{num_lbl}-{den_lbl}", f"{num_lbl}_vs_{den_lbl}") if k and k != "-"]
    rev = [k for k in (f"{den_lbl}-{num_lbl}", f"{den_lbl}_vs_{num_lbl}") if k and k != "-"]
    for root in roots:
        if not root.is_dir():
            continue
        for k in fwd:
            for p in (root / k / "rmats_post", root / k):
                if _has_rmats_files(p, rmats_mode):
                    return p, False
        for k in rev:
            for p in (root / k / "rmats_post", root / k):
                if _has_rmats_files(p, rmats_mode):
                    return p, True
    return None, False


def _resolve_bundle_context(
    bdir: Path,
    *,
    src_dir: Path | None,
    contrast_id: str | None,
    rmats_mode: str,
    explicit_reverse_direction: bool | None = None,
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
        c_df = pd.read_csv(c_path, sep=sep, dtype=str, keep_default_na=False)
        for _, r in c_df.iterrows():
            cid = str(r.get("contrast") or r.get("contrast_id") or r.get("comparison_id") or "").strip()
            num_l = str(r.get("treatment") or r.get("numerator") or r.get("group_a") or "").strip()
            den_l = str(r.get("control") or r.get("denominator") or r.get("group_b") or "").strip()
            if cid:
                contrasts.append((cid, num_l, den_l))

    manifest_comps: dict[str, dict[str, str]] = {}
    mpath = bdir / "manifest.json"
    if mpath.is_file():
        try:
            bm = json.loads(mpath.read_text(encoding="utf-8"))
            if not isinstance(bm, dict):
                raise ValueError("top-level JSON value must be an object")
            comps_raw = bm.get("comparisons", [])
            if comps_raw is not None and not isinstance(comps_raw, list):
                raise ValueError("'comparisons' must be a list")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise ValueError(f"Malformed manifest.json in {mpath}: {exc}") from exc
        for cm in comps_raw or []:
            if not isinstance(cm, dict) or not cm.get("comparison_id"):
                continue
            c_key = str(cm["comparison_id"]).strip()
            rec = {
                "rmats_group1": str(cm.get("rmats_group1") or "").strip(),
                "rmats_group2": str(cm.get("rmats_group2") or "").strip(),
                "rmats_dir": str(cm.get("rmats_dir") or "").strip(),
            }
            manifest_comps[c_key] = rec
            manifest_comps.setdefault(c_key.lower(), rec)

    target_cid = contrast_id.strip() if (contrast_id is not None and contrast_id.strip()) else None
    matched_c: tuple[str, str, str] | None = None
    if target_cid is not None:
        matched_c = next((c for c in contrasts if c[0] == target_cid or c[0].lower() == target_cid.lower()), None)
        if matched_c is None and contrasts:
            raise ValueError(
                f"Requested contrast_id {target_cid!r} not found in bundle contrasts: {[c[0] for c in contrasts]}."
            )
        if matched_c is None and manifest_comps and target_cid not in manifest_comps and target_cid.lower() not in manifest_comps:
            raise ValueError(
                f"Requested contrast_id {target_cid!r} not found in bundle manifest comparisons."
            )
    elif len(contrasts) == 1:
        matched_c = contrasts[0]
    elif len(contrasts) >= 2 and src_dir is not None and src_dir.resolve() != bdir.resolve():
        # Match explicit src_dir back to the owning contrast when caller passed --rmats <subdir> without --contrast-id
        src_res = src_dir.resolve()
        for c_cand in contrasts:
            mc_cand = manifest_comps.get(c_cand[0]) or manifest_comps.get(c_cand[0].lower())
            try:
                exp_c_dir, _ = _find_expected_dir_for_contrast(bdir, c_cand[0], c_cand[1], c_cand[2], mc_cand, rmats_mode)
            except FileNotFoundError:
                exp_c_dir = None
            if exp_c_dir is not None and exp_c_dir.resolve() == src_res:
                matched_c = c_cand
                break

    eff_cid = matched_c[0] if matched_c is not None else target_cid
    num_lbl, den_lbl = (matched_c[1], matched_c[2]) if matched_c is not None else ("", "")

    mc = (manifest_comps.get(eff_cid) or manifest_comps.get(eff_cid.lower())) if eff_cid else None
    if mc is None and len(manifest_comps) == 1 and target_cid is None:
        mc = next(iter(manifest_comps.values()))

    has_manifest_arms = bool(mc and mc.get("rmats_group1") and mc.get("rmats_group2"))
    if has_manifest_arms and num_lbl and den_lbl and {mc["rmats_group1"], mc["rmats_group2"]} != {num_lbl, den_lbl}:
        raise ValueError(
            f"manifest.json rmats_group1/2 ({mc['rmats_group1']}, {mc['rmats_group2']}) do not match contrast '{eff_cid}' ({num_lbl}, {den_lbl})."
        )
    if (
        explicit_reverse_direction is None
        and (not has_manifest_arms)
        and eff_cid
        and num_lbl
        and den_lbl
        and eff_cid.lower() in {f"{den_lbl}-{num_lbl}".lower(), f"{den_lbl}_vs_{num_lbl}".lower(), f"{den_lbl}_{num_lbl}".lower()}
    ):
        raise ValueError(
            f"Ambiguous contrast_id {eff_cid!r} matches reversed condition order '{den_lbl}-{num_lbl}' "
            f"for numerator={num_lbl!r}, denominator={den_lbl!r} without explicit manifest rmats_group1/rmats_group2 or reverse_direction."
        )

    src_is_explicit = src_dir is not None and src_dir.resolve() != bdir.resolve()
    resolved_dir: Path | None = None
    path_inverted = False

    if eff_cid is not None:
        expected_dir, expected_inv = _find_expected_dir_for_contrast(bdir, eff_cid, num_lbl, den_lbl, mc, rmats_mode)
        if src_is_explicit and _has_rmats_files(src_dir, rmats_mode):
            src_res = src_dir.resolve()
            if expected_dir is not None and src_res != expected_dir.resolve():
                raise ValueError(
                    f"Explicit rmats_source {src_dir} does not match expected contrast directory {expected_dir} for contrast '{eff_cid}'."
                )
            # Also reject if src_dir belongs to a different declared contrast in the bundle
            for other_cid, other_num, other_den in contrasts:
                if other_cid == eff_cid:
                    continue
                other_mc = manifest_comps.get(other_cid) or manifest_comps.get(other_cid.lower())
                try:
                    other_dir, _ = _find_expected_dir_for_contrast(bdir, other_cid, other_num, other_den, other_mc, rmats_mode)
                except FileNotFoundError:
                    other_dir = None
                if other_dir is not None and other_dir.resolve() == src_res:
                    raise ValueError(
                        f"Explicit rmats_source {src_dir} belongs to contrast '{other_cid}', conflicting with requested contrast '{eff_cid}'."
                    )
            resolved_dir = src_dir
            if num_lbl and den_lbl:
                dname = src_dir.parent.name if src_dir.name == "rmats_post" else src_dir.name
                if dname.lower() in {f"{den_lbl}-{num_lbl}".lower(), f"{den_lbl}_vs_{num_lbl}".lower()}:
                    path_inverted = True
        else:
            resolved_dir = expected_dir
            path_inverted = expected_inv
            if resolved_dir is None and len(contrasts) >= 1:
                raise FileNotFoundError(f"No rMATS contrast directory matched '{eff_cid}' (mode={rmats_mode.upper()}) under {bdir}")
    elif src_is_explicit and _has_rmats_files(src_dir, rmats_mode):
        resolved_dir = src_dir

    rev_dir = path_inverted
    if has_manifest_arms and num_lbl and den_lbl:
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


def _select_tar_contrast_members(
    tar: tarfile.TarFile,
    prio_sufs: tuple[str, ...],
    eff_cid_query: str | None,
    src: Path,
) -> dict[str, tarfile.TarInfo]:
    """Select rMATS files from a single unambiguous contrast directory inside a `.tar`/`.tar.gz` archive."""
    groups: dict[str, dict[str, tuple[int, tarfile.TarInfo]]] = {}
    for member in tar:
        norm_name = member.name.replace("\\", "/").lstrip("./")
        base = norm_name.rsplit("/", 1)[-1]
        if not member.isfile() or base.startswith("._"):
            continue
        parent_parts = [p for p in norm_name.split("/")[:-1] if p and p != "."]
        if parent_parts and parent_parts[-1] == "rmats_post":
            parent_parts = parent_parts[:-1]
        group_key = "/".join(parent_parts)
        for kind in EVENT_COORD_COLS:
            for rank_p, suf in enumerate(prio_sufs):
                if base == f"{kind}{suf}":
                    g_bucket = groups.setdefault(group_key, {})
                    prev = g_bucket.get(kind)
                    if prev is not None and rank_p == prev[0]:
                        raise ValueError(
                            f"Duplicate rMATS '{kind}{suf}' entries in archive contrast directory {group_key or '.'!r} of {src}."
                        )
                    if prev is None or rank_p < prev[0]:
                        g_bucket[kind] = (rank_p, member)

    if not groups:
        return {}
    if len(groups) > 1:
        if not eff_cid_query:
            raise ValueError(
                f"Archive {src} contains multiple rMATS contrast directories ({sorted(groups)}); "
                "pass explicit --contrast-id / contrast_id= to select one unambiguously."
            )
        q_exact = eff_cid_query.strip()
        q_low = q_exact.lower()
        matches = [k for k in groups if k == q_exact or Path(k).name == q_exact]
        if not matches:
            matches = [k for k in groups if k.lower() == q_low or Path(k).name.lower() == q_low]
        if not matches:
            raise ValueError(
                f"Requested contrast_id {eff_cid_query!r} not found among archive contrast directories {sorted(groups)} in {src}."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous contrast_id {eff_cid_query!r} matched multiple archive directories {sorted(matches)} in {src}."
            )
        chosen = matches[0]
    else:
        chosen = next(iter(groups))
        if eff_cid_query and chosen:
            leaf = Path(chosen).name
            if ("_vs_" in leaf or "-" in leaf) and leaf.lower() != eff_cid_query.strip().lower():
                raise ValueError(
                    f"Requested contrast_id {eff_cid_query!r} does not match archive contrast directory {chosen!r} in {src}."
                )
    return {kind: pair[1] for kind, pair in groups[chosen].items()}


def _filter_precomputed_contrast(
    df: pd.DataFrame,
    *,
    eff_cid_query: str | None,
    src: Path,
) -> pd.DataFrame:
    """Validate and filter precomputed `.tsv`/`.parquet` splicing tables by requested `contrast_id`."""
    c_col = next((c for c in ("comparison_id", "contrast_id", "contrast") if c in df.columns), None)
    if c_col is None or df.empty:
        return df
    vals = df[c_col].astype(str).str.strip()
    uniq_c = [v for v in dict.fromkeys(vals) if v and v.lower() != "nan"]
    if eff_cid_query is not None:
        q_s = eff_cid_query.strip()
        mask = vals == q_s
        if not mask.any():
            mask = vals.str.lower() == q_s.lower()
        if not mask.any():
            raise ValueError(
                f"Requested contrast_id {eff_cid_query!r} not found in {src} column '{c_col}' (available: {uniq_c})."
            )
        return df.loc[mask].reset_index(drop=True)
    if len(uniq_c) > 1:
        raise ValueError(
            f"Precomputed event table {src} contains multiple contrasts in '{c_col}': {uniq_c}; "
            "specify explicit --contrast-id / contrast_id=."
        )
    return df


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
            explicit_reverse_direction=reverse_direction,
        )
        if src.is_dir() and rdir_res is not None:
            src = rdir_res
        if reverse_direction is None:
            eff_rev = bundle_rev
        if comparison_id == "rmats_contrast" and bundle_cid:
            eff_comp_id = bundle_cid

    frames: list[pd.DataFrame] = []
    found_any_rmats_file = False
    if src.is_dir():
        eff_suf = prio_sufs[0]
        for kind in EVENT_COORD_COLS:
            cand = src / f"{kind}{eff_suf}"
            if cand.is_file():
                found_any_rmats_file = True
                raw = pd.read_csv(cand, sep="\t", low_memory=False)
                frames.append(_standardize_rmats_df(raw, kind, valid_contigs, eff_comp_id, reverse_direction=eff_rev))
    elif str(src).endswith((".tar.gz", ".tgz", ".tar")):
        with tarfile.open(src, "r:*") as tar:
            selected_members = _select_tar_contrast_members(tar, prio_sufs, eff_cid_query, src)
            for kind, member in selected_members.items():
                fobj = tar.extractfile(member)
                if fobj is not None:
                    found_any_rmats_file = True
                    raw = pd.read_csv(fobj, sep="\t", low_memory=False)
                    frames.append(_standardize_rmats_df(raw, kind, valid_contigs, eff_comp_id, reverse_direction=eff_rev))
    elif src.suffix == ".parquet":
        raw_p = _filter_precomputed_contrast(pd.read_parquet(src), eff_cid_query=eff_cid_query, src=src)
        if "IncLevelDifference" in raw_p.columns and "dpsi" not in raw_p.columns and "event_type" in raw_p.columns:
            for kind, grp in raw_p.groupby("event_type"):
                if str(kind) in EVENT_COORD_COLS:
                    found_any_rmats_file = True
                    frames.append(_standardize_rmats_df(grp.copy(), str(kind), valid_contigs, eff_comp_id, reverse_direction=eff_rev))
        else:
            return raw_p
    else:
        raw_t = _filter_precomputed_contrast(pd.read_csv(src, sep="\t", low_memory=False), eff_cid_query=eff_cid_query, src=src)
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
