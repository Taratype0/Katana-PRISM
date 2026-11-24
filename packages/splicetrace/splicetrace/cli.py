"""Standalone CLI for splicetrace (1.0.2) — Package & nf-core unified interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time
import pandas as pd

from splicetrace.continuous_rescue import run_bulk_continuous_rescue
from splicetrace.fast_accel import configure_server_threads, fast_score_mean_log_odds
from splicetrace.matching import match_events
from splicetrace.module_bridge import annotate_rbp_expression_disambiguation, bridge_splicenet_modules_to_rbps
from splicetrace.protein_consequences import trace_protein_and_nmd_consequences
from splicetrace.rmats_io import load_rmats_measurable_events
from splicetrace.rna_forms import IndexedFasta, compute_flank_gc_fast, extract_event_rna, trace_motif_instances
from splicetrace.scanner import run_splicetrace_scan

__version__ = "1.0.2"


def write_nfcore_versions_yml(path: Path | str, process_name: str = "KATANA_SPLICETRACE") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(f'"{process_name}":\n    splicetrace: "{__version__}"\n    python: "{platform.python_version()}"\n', encoding="utf-8")


def add_extensible_library_args(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("Motif Library & Extensible Database Options")
    grp.add_argument("--species", choices=["human", "mouse"], default="human", help="RBP & UniProt species namespace (`human` or `mouse`)")
    grp.add_argument("--extra-motifs", nargs="+", type=Path, default=None, help="External motif files (.meme/.motif/.jaspar/.pfm/.tsv/.csv/.json[.gz])")
    grp.add_argument("--no-extended-motifs", action="store_true", help="Disable curated m6A & ENCODE RBNS/eCLIP expansion")
    grp.add_argument("--no-user-packs", action="store_true", help="Ignore user-installed motif packs in ~/.katana/motif_packs")
    grp.add_argument("--list-rbps", action="store_true", help="Print full RBP catalog summary and exit")
    grp.add_argument("--filter-rbp", default=None, help="Substring/symbol filter when using --list-rbps")
    grp.add_argument("--import-motifs", type=Path, default=None, help="Convert external motif file into a portable Katana Motif Pack")
    grp.add_argument("--export-pack", type=Path, default=None, help="Destination .json[.gz] path for --import-motifs")
    grp.add_argument("--install-pack", type=Path, default=None, help="Install external motif pack into ~/.katana/motif_packs")
    grp.add_argument("--pack-name", default="custom_user_pack", help="Pack name for --install-pack or --remove-pack")
    grp.add_argument("--remove-pack", default=None, help="Remove an installed user motif pack by name and exit")


def handle_library_management_commands(args: argparse.Namespace) -> tuple[bool, object | None]:
    from splicetrace.library import (
        export_katana_motif_pack,
        get_user_motif_packs_dir,
        load_bundled_human_rbp_library,
        load_curated_human_rbp_library,
        load_external_motif_entries,
    )

    if args.remove_pack:
        pdir = get_user_motif_packs_dir()
        removed = []
        for cand in (pdir / f"{args.remove_pack}.json.gz", pdir / f"{args.remove_pack}.json"):
            if cand.is_file():
                cand.unlink()
                removed.append(str(cand))
        print(json.dumps({"action": "remove_pack", "removed": removed}))
        return True, None

    if args.import_motifs is not None:
        if args.export_pack is None:
            raise ValueError("--export-pack <out.json.gz> is required with --import-motifs")
        ents = load_external_motif_entries(args.import_motifs, default_source=args.pack_name)
        info = export_katana_motif_pack(ents, args.export_pack, pack_name=args.pack_name)
        print(json.dumps({"action": "import_motifs", **info}, indent=2))
        return True, None

    if args.install_pack is not None:
        pdir = get_user_motif_packs_dir()
        pdir.mkdir(parents=True, exist_ok=True)
        dst = pdir / f"{args.pack_name}.json.gz"
        ents = load_external_motif_entries(args.install_pack, default_source=args.pack_name)
        info = export_katana_motif_pack(ents, dst, pack_name=args.pack_name)
        print(json.dumps({"action": "install_pack", **info}, indent=2))
        return True, None

    if getattr(args, "models_json", None) is not None and getattr(args, "assignments_tsv", None) is not None:
        lib = load_curated_human_rbp_library(
            args.models_json, args.assignments_tsv,
            include_extended_motifs=not args.no_extended_motifs, extra_motif_files=args.extra_motifs,
            include_installed_user_packs=not args.no_user_packs, species=args.species,
        )
    else:
        lib = load_bundled_human_rbp_library(
            include_extended_motifs=not args.no_extended_motifs, extra_motif_files=args.extra_motifs,
            include_installed_user_packs=not args.no_user_packs, species=args.species,
        )

    if args.list_rbps:
        df_sum = lib.to_summary_dataframe()
        if args.filter_rbp:
            q = str(args.filter_rbp).upper()
            df_sum = df_sum[
                df_sum["RBP"].str.upper().str.contains(q)
                | df_sum["mouse_symbol"].str.upper().str.contains(q)
                | df_sum["category"].str.upper().str.contains(q)
            ]
        print(df_sum.to_string(index=False))
        return True, lib

    return False, lib


def _read_table(path: Path) -> pd.DataFrame:
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith(".csv") or name.endswith(".csv.gz"):
        return pd.read_csv(path)
    return pd.read_csv(path, sep="\t")


def _run_gtf_and_5plots(
    ev_df: pd.DataFrame,
    conseq_df: pd.DataFrame,
    gtf_path: Path,
    fasta_path: Path,
    out_dir: Path,
    species: str,
    contrast_label: str = "contrast",
) -> dict:
    prot_df = trace_protein_and_nmd_consequences(
        ev_df.head(500), gtf_path, fasta_path, indexed_fasta_cls=IndexedFasta, species=species
    )
    prot_df.to_csv(out_dir / "protein_and_nmd_consequences.tsv", sep="\t", index=False)
    info = {
        "protein_consequences_evaluated": int(len(prot_df)),
        "nmd_switch_events": int(prot_df["nmd_switch_class"].isin(["inclusion_triggers_NMD_poison_exon", "skipping_triggers_NMD_decay"]).sum()) if not prot_df.empty else 0,
        "uniprot_domain_hits": int(prot_df["disrupted_protein_features"].ne("none").sum()) if not prot_df.empty else 0,
    }
    try:
        from splicetrace.protein_viz import (
            _clean_rbp_label,
            plot_causal_alluvial_flow,
            plot_consequence_landscape,
            plot_dual_track_nmd_architecture,
            plot_lollipop_protein_splicing_map,
            plot_side_by_side_inframe_domains,
        )
        if not prot_df.empty:
            fig_dir = out_dir / "figures"
            m_map = {}
            if not conseq_df.empty and "event_id" in conseq_df.columns:
                for eid_m, grp_m in conseq_df.groupby("event_id"):
                    m_map[str(eid_m)] = ", ".join(list(dict.fromkeys(f"{_clean_rbp_label(r.RBP)}({r.hit_RNA})" for r in grp_m.itertuples()))[:3])
            plot_consequence_landscape({contrast_label: prot_df}, contrast_label, fig_dir / "style1_proteome_consequence_landscape")
            plot_dual_track_nmd_architecture(prot_df, m_map, fig_dir / "style2_nmd_dual_track_architecture", contrast_label)
            plot_side_by_side_inframe_domains(prot_df, fig_dir / "style3_inframe_domain_remodeling", contrast_label)
            plot_lollipop_protein_splicing_map(prot_df, fig_dir / "style4_lollipop_protein_map", contrast_label)
            plot_causal_alluvial_flow(prot_df, conseq_df, fig_dir / "style5_causal_rbp_to_proteome_alluvial", contrast_label)
    except Exception:
        pass
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="splicetrace",
        description="SpliceTrace 1.0.2: Bulk AS RBP discovery, mature-mRNA motif, SpliceNet bridge & GTF/UniProt ORF/NMD suite.",
    )
    parser.add_argument("--version", action="version", version=f"splicetrace {__version__}")
    parser.add_argument("--rmats", type=Path, default=None, help="rMATS directory, .tar.gz archive, or measurable events table")
    parser.add_argument("--foreground", type=Path, default=None, help="Discrete foreground event table (.tsv[.gz]/.csv/.parquet)")
    parser.add_argument("--background", type=Path, default=None, help="Discrete background pool event table (.tsv[.gz]/.csv/.parquet)")
    parser.add_argument("--events", type=Path, default=None, help="Standalone event coordinate table for --gtf or --splicenet-modules runs")
    parser.add_argument("--fasta", type=Path, required=False, help="Indexed reference genome FASTA (.fa with .fa.fai)")
    parser.add_argument("--gtf", type=Path, default=None, help="Optional annotation GTF (.gtf[.gz]) for ORF, frameshift, 50-nt EJC NMD & UniProt consequences + 5 plots")
    parser.add_argument("--splicenet-modules", type=Path, default=None, help="Optional SpliceNet module_assignments.tsv for Cross-Module RBP motif bridge")
    parser.add_argument("--splicenet-scores", type=Path, default=None, help="Optional SpliceNet module_scores.tsv for eigengene x RBP expression correlation")
    parser.add_argument("--rbp-expression", type=Path, default=None, help="Optional cohort gene expression table/matrix for paralog expression disambiguation")
    parser.add_argument("--min-rbp-expr", type=float, default=1.0, help="Minimum mean expression (TPM/CPM) to mark an RBP as expressed (default: 1.0)")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for Numba acceleration (0 = auto-detect SLURM_CPUS_PER_TASK / all cores)")
    parser.add_argument("-o", "--output-dir", "--out", dest="output_dir", type=Path, required=False, help="Output directory")
    parser.add_argument("--contrast-id", default=None, help="Comparison identifier for plot headers (defaults to output directory name)")
    parser.add_argument("--mode", choices=["auto", "discrete", "rescue"], default="auto")
    parser.add_argument("--models-json", type=Path, default=None)
    parser.add_argument("--assignments-tsv", type=Path, default=None)
    parser.add_argument("--flank-gc-caliper", type=float, default=0.08)
    parser.add_argument("--acceptor-core-nt", type=int, default=3)
    parser.add_argument("--distal-intron-nt", type=int, default=500, help="Distal intronic cis-regulatory window outer bound in nt (default: 500; 0 for proximal 200-nt only)")
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--emit-versions-yml", type=Path, default=None)
    parser.add_argument("--process-name", default="KATANA_SPLICETRACE")
    add_extensible_library_args(parser)
    args = parser.parse_args(argv)

    handled, lib = handle_library_management_commands(args)
    if handled:
        return 0

    if args.emit_versions_yml is not None and args.output_dir is None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
        return 0
    if args.fasta is None or args.output_dir is None:
        parser.error("--fasta and --output-dir (or --out) are required for analysis runs.")

    active_threads = configure_server_threads(args.threads)
    t0 = time.monotonic()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bridge_pool_df: pd.DataFrame | None = None
    rbp_expr_df = _read_table(args.rbp_expression) if args.rbp_expression is not None else None
    mod_scores_df = _read_table(args.splicenet_scores) if args.splicenet_scores is not None else None

    if args.rmats is not None and (args.mode in {"auto", "rescue"} and args.foreground is None):
        meas_df = load_rmats_measurable_events(args.rmats, args.fasta)
        bridge_pool_df = meas_df
        if len(meas_df) < 10:
            pd.DataFrame(columns=["RBP", "best_channel", "stat_Z", "rbp_P_FWER"]).to_csv(out_dir / "rbp_ranking.tsv", sep="\t", index=False)
            pd.DataFrame(columns=["RBP", "region", "stat_Z", "P_FWER"]).to_csv(out_dir / "region_tests.tsv", sep="\t", index=False)
            meas_df.to_csv(out_dir / "evaluated_events.tsv", sep="\t", index=False)
            receipt = {"version": __version__, "library": lib.name, "status": "insufficient_events_for_rescue", "evaluated_events": int(len(meas_df)), "active_threads": int(active_threads), "elapsed_seconds": round(time.monotonic() - t0, 2)}
            (out_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
            if args.emit_versions_yml is not None:
                write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
            print(json.dumps(receipt))
            return 0
        rescue_res = run_bulk_continuous_rescue(
            meas_df,
            args.fasta,
            lib,
            distal_intron_nt=args.distal_intron_nt,
            rbp_expr_df=rbp_expr_df,
            min_rbp_expr=args.min_rbp_expr,
            n_permutations=args.permutations,
            seed=args.seed,
            threads=active_threads,
        )
        rescue_res.rbp_ranking.to_csv(out_dir / "rbp_ranking.tsv", sep="\t", index=False)
        rescue_res.region_tests.to_csv(out_dir / "region_tests.tsv", sep="\t", index=False)
        rescue_res.event_contrasts.to_csv(out_dir / "evaluated_events.tsv", sep="\t", index=False)
        if not rescue_res.mature_consequences.empty:
            rescue_res.mature_consequences.to_csv(out_dir / "motif_consequence_changes.tsv", sep="\t", index=False)
        receipt = {"version": __version__, "library": lib.name, "n_rbps": lib.n_rbps, "n_models": lib.n_models, "mode": "continuous_rescue", "active_threads": int(active_threads), **rescue_res.diagnostics}
        if args.gtf is not None:
            fg_prot_ev = rescue_res.event_contrasts[rescue_res.event_contrasts["_is_fg"]]
            c_lbl = str(args.contrast_id or out_dir.name or "contrast")
            receipt.update(_run_gtf_and_5plots(fg_prot_ev, rescue_res.mature_consequences, args.gtf, args.fasta, out_dir, args.species, contrast_label=c_lbl))

    elif args.foreground is not None and args.background is not None:
        fg = _read_table(args.foreground).copy()
        bg = _read_table(args.background).copy()
        comb_all = pd.concat([fg, bg], ignore_index=True).drop_duplicates("event_id")
        bridge_pool_df = comb_all
        if args.flank_gc_caliper > 0:
            if "flank_GC" not in fg.columns:
                fg["flank_GC"] = fg["event_id"].astype(str).map(compute_flank_gc_fast(fg, args.fasta))
            if "flank_GC" not in bg.columns:
                bg["flank_GC"] = bg["event_id"].astype(str).map(compute_flank_gc_fast(bg, args.fasta))
            caliper_val: float | None = args.flank_gc_caliper
        else:
            caliper_val = None
        matched, _, _ = match_events(fg, bg, flank_gc_caliper=caliper_val)
        m_ok = matched[matched["status"].eq("matched")]
        conseq_fg = pd.DataFrame()
        if args.mode == "auto" and len(m_ok) < 250 and "dpsi" in comb_all.columns and "baseline_PSI" in comb_all.columns:
            rescue_res = run_bulk_continuous_rescue(
                comb_all,
                args.fasta,
                lib,
                distal_intron_nt=args.distal_intron_nt,
                rbp_expr_df=rbp_expr_df,
                min_rbp_expr=args.min_rbp_expr,
                n_permutations=args.permutations,
                seed=args.seed,
                threads=active_threads,
            )
            rescue_res.rbp_ranking.to_csv(out_dir / "rbp_ranking.tsv", sep="\t", index=False)
            rescue_res.region_tests.to_csv(out_dir / "region_tests.tsv", sep="\t", index=False)
            conseq_fg = rescue_res.mature_consequences
            receipt = {"version": __version__, "library": lib.name, "n_rbps": lib.n_rbps, "mode": "auto_small_sample_rescue", "active_threads": int(active_threads), **rescue_res.diagnostics}
        else:
            matched.to_csv(out_dir / "matched_events.tsv", sep="\t", index=False)
            needed = set(m_ok["foreground_event"].astype(str)) | set(m_ok["background_event"].astype(str))
            comb_ev = pd.concat([fg[fg["event_id"].isin(needed)], bg[bg["event_id"].isin(needed)]], ignore_index=True)
            rna_df = extract_event_rna(
                comb_ev,
                args.fasta,
                intron_nt=200,
                distal_intron_nt=args.distal_intron_nt,
                acceptor_core_nt=args.acceptor_core_nt,
            )
            res = run_splicetrace_scan(matched, rna_df, lib, permutations=args.permutations, seed=args.seed)
            res["whole_regions"].to_csv(out_dir / "region_tests.tsv", sep="\t", index=False)
            res["clusters"].to_csv(out_dir / "motif_clusters.tsv", sep="\t", index=False)
            rank_all = res["rbp_ranking_all"]
            if rbp_expr_df is not None and not rank_all.empty:
                z_col_disc = "rbp_pivotal_Z" if "rbp_pivotal_Z" in rank_all.columns else rank_all.columns[1]
                rank_all = annotate_rbp_expression_disambiguation(
                    rank_all,
                    rbp_expr_df,
                    fwer_col="rbp_P_FWER",
                    z_col=z_col_disc,
                    min_rbp_expr=args.min_rbp_expr,
                )
            rank_all.to_csv(out_dir / "rbp_ranking.tsv", sep="\t", index=False)
            receipt = {"version": __version__, "library": lib.name, "n_rbps": lib.n_rbps, "mode": "discrete_matched", "active_threads": int(active_threads), "matched_events": int(len(m_ok)), "significant_rbps_fwer05": int((rank_all["rbp_P_FWER"] <= 0.05).sum())}
        if args.gtf is not None:
            if "dpsi" not in fg.columns:
                fg["dpsi"] = 0.0
            c_lbl = str(args.contrast_id or out_dir.name or "contrast")
            receipt.update(_run_gtf_and_5plots(fg, conseq_fg, args.gtf, args.fasta, out_dir, args.species, contrast_label=c_lbl))

    elif args.events is not None and (args.gtf is not None or args.splicenet_modules is not None):
        ev_only = _read_table(args.events).copy()
        if "dpsi" not in ev_only.columns:
            ev_only["dpsi"] = 0.0
        bridge_pool_df = ev_only
        receipt = {"version": __version__, "library": lib.name, "mode": "consequences_or_bridge_only", "active_threads": int(active_threads)}
        if args.gtf is not None:
            c_lbl = str(args.contrast_id or out_dir.name or "contrast")
            receipt.update(_run_gtf_and_5plots(ev_only, pd.DataFrame(), args.gtf, args.fasta, out_dir, args.species, contrast_label=c_lbl))
    else:
        parser.error("Provide --rmats, --foreground/--background, --events (with --gtf or --splicenet-modules), or a library flag (--list-rbps).")

    if args.splicenet_modules is not None:
        ev_bridge = _read_table(args.events) if args.events is not None else bridge_pool_df
        if ev_bridge is not None and not ev_bridge.empty:
            mod_in = _read_table(args.splicenet_modules)
            br_df = bridge_splicenet_modules_to_rbps(
                mod_in, ev_bridge, args.fasta, lib,
                extract_rna_fn=extract_event_rna, score_odds_fn=fast_score_mean_log_odds,
                distal_intron_nt=args.distal_intron_nt,
                rbp_expr_df=rbp_expr_df,
                module_scores_df=mod_scores_df,
                min_rbp_expr=args.min_rbp_expr,
                n_permutations=args.permutations, seed=args.seed,
            )
            br_df.to_csv(out_dir / "module_rbp_bridge.tsv", sep="\t", index=False)
            receipt["bridged_module_rbp_rows"] = int(len(br_df))

    receipt["elapsed_seconds"] = round(time.monotonic() - t0, 2)
    (out_dir / "run_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    if args.emit_versions_yml is not None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
    print(json.dumps(receipt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
