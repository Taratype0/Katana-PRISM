"""CLIs for `katana-report` (self-contained HTML/ZIP synthesis) and `katana-studio` (interactive local workbench)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform

from katana_studio import __version__, open_workspace, run_bulk, run_report_from_nextflow, run_single_cell


def write_nfcore_versions_yml(path: Path | str, process_name: str = "KATANA_REPORT") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        f'"{process_name}":\n    katana-studio: "{__version__}"\n    python: "{platform.python_version()}"\n',
        encoding="utf-8",
    )


def report_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="katana-report",
        description="Katana Report 1.0.1: Build SHA256-verified self-contained REPORT.html and REPORT.zip for bulk or single-cell splicing studies.",
    )
    parser.add_argument("--version", action="version", version=f"katana-report {__version__}")
    parser.add_argument("--emit-versions-yml", type=Path, default=None)
    parser.add_argument("--process-name", default="KATANA_REPORT")
    sub = parser.add_subparsers(dest="command")

    pb = sub.add_parser("bulk", help="Build executive bulk splicing report from Nextflow bundle or standalone rMATS/DESeq2 outputs")
    pb.add_argument("--bundle", type=Path, default=None, help="Completed Nextflow katana_bundle directory")
    pb.add_argument("--rmats", type=str, default=None, help="Standalone rMATS directory or contrast=dir mapping")
    pb.add_argument("--rmats-mode", choices=["JCEC", "JC"], default="JCEC", help="Preferred rMATS count mode (default: JCEC)")
    pb.add_argument("--samples", type=Path, default=None, help="Sample metadata CSV/TSV")
    pb.add_argument("--contrasts", type=Path, default=None, help="Optional contrastsheet CSV/TSV")
    pb.add_argument("--deseq2", type=Path, default=None, help="Optional DESeq2 results directory or TSV")
    pb.add_argument("--multiqc", type=Path, default=None, help="Optional MultiQC directory for 5'-3' gene-body coverage")
    pb.add_argument("--fasta", type=Path, default=None, help="Optional genome FASTA")
    pb.add_argument("--gtf", type=Path, default=None, help="Optional annotation GTF")
    pb.add_argument("--splicemanifold-dir", type=Path, default=None, help="Optional precomputed SpliceManifold bulk directory")
    pb.add_argument("--splicenet-dir", type=Path, default=None, help="Optional precomputed SpliceNet output directory")
    pb.add_argument("--splicetrace-dirs", nargs="*", type=str, default=None, help="Optional precomputed per-contrast SpliceTrace directories (`dir` or `contrast_id=dir`)")
    pb.add_argument("--cross-paradigm-dir", type=Path, default=None, help="Optional precomputed KATANA_CROSS_PARADIGM output directory")
    pb.add_argument("--study-name", default="Katana Bulk Splicing Study")
    pb.add_argument("--species", choices=["auto", "human", "mouse"], default="auto")
    pb.add_argument("--fdr", type=float, default=0.05)
    pb.add_argument("--dpsi", type=float, default=0.10)
    pb.add_argument("--dummy", action="store_true", help="Mark report header as SYNTHETIC DISPLAY REVIEW (`dummy=true`)")
    pb.add_argument("-o", "--outdir", type=Path, required=True)
    pb.add_argument("--emit-versions-yml", type=Path, default=None)
    pb.add_argument("--process-name", default="KATANA_REPORT")

    psc = sub.add_parser("single-cell", help="Build executive single-cell splicing report from MARVEL / Smart-seq3 / 10x matrices")
    psc.add_argument("--events", type=Path, required=True)
    psc.add_argument("--cells", type=Path, required=True)
    psc.add_argument("--psi-matrix", type=Path, default=None)
    psc.add_argument("--support-matrix", type=Path, default=None)
    psc.add_argument("--expression", type=Path, default=None)
    psc.add_argument("--gtf", type=Path, default=None, help="Optional annotation GTF for Katana Native Sashimi exon labels")
    psc.add_argument("--bams", type=Path, default=None, help="Optional directory of donor-level indexed BAMs for BAM RPKM Katana Native Sashimi")
    psc.add_argument("--sc-bam", type=Path, default=None, help="Optional multiplexed single-cell BAM for localized per-condition BAM RPKM Katana Native Sashimi")
    psc.add_argument("--sc-bc-tag", default="BC", help="BAM cell-barcode auxiliary tag (default: BC)")
    psc.add_argument("--sc-barcode-shares", type=Path, default=None, help="Optional barcode_shares.txt mapping aliased cell barcodes to QC-passed cells")
    psc.add_argument("--splicemanifold-dir", type=Path, default=None, help="Optional precomputed splicemanifold single-cell directory")
    psc.add_argument("--scsplicenet-dir", type=Path, default=None, help="Optional precomputed scsplicenet output directory")
    psc.add_argument("--scsplicetrace-dir", type=Path, default=None, help="Optional precomputed scsplicetrace output directory")
    psc.add_argument("--subtypes-dirs", nargs="*", type=Path, default=None, help="Optional per-subtype downstream contrast directories")
    psc.add_argument("--study-name", default="Katana Single-Cell Splicing Study")
    psc.add_argument("--species", choices=["human", "mouse"], default="human")
    psc.add_argument("--fdr", type=float, default=0.05)
    psc.add_argument("--dpsi", type=float, default=0.10)
    psc.add_argument("--dummy", action="store_true", help="Mark report header as SYNTHETIC DISPLAY REVIEW (`dummy=true`)")
    psc.add_argument("-o", "--outdir", type=Path, required=True)
    psc.add_argument("--emit-versions-yml", type=Path, default=None)
    psc.add_argument("--process-name", default="KATANA_REPORT_SC")

    args = parser.parse_args(argv)
    if args.command is None:
        if args.emit_versions_yml is not None:
            write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
            return 0
        parser.print_help()
        return 0

    if args.command == "bulk":
        if args.bundle is not None and (
            args.splicemanifold_dir is not None
            or args.splicenet_dir is not None
            or args.splicetrace_dirs
            or args.cross_paradigm_dir is not None
        ):
            receipt = run_report_from_nextflow(
                outdir=args.outdir,
                bundle=args.bundle,
                fasta=args.fasta,
                gtf=args.gtf,
                study_name=args.study_name,
                species=args.species,
                fdr_cutoff=args.fdr,
                dpsi_cutoff=args.dpsi,
                rmats_mode=args.rmats_mode,
                splicemanifold_dir=args.splicemanifold_dir,
                splicenet_dir=args.splicenet_dir,
                splicetrace_dirs=args.splicetrace_dirs,
                cross_paradigm_dir=args.cross_paradigm_dir,
                dummy=args.dummy,
            )
        else:
            receipt = run_bulk(
                outdir=args.outdir,
                bundle=args.bundle,
                rmats=args.rmats,
                rmats_mode=args.rmats_mode,
                samples=args.samples,
                contrasts=args.contrasts,
                deseq2=args.deseq2,
                multiqc=args.multiqc,
                fasta=args.fasta,
                gtf=args.gtf,
                splicemanifold_dir=args.splicemanifold_dir,
                splicenet_dir=args.splicenet_dir,
                splicetrace_dirs=args.splicetrace_dirs,
                cross_paradigm_dir=args.cross_paradigm_dir,
                study_name=args.study_name,
                species=args.species,
                fdr_cutoff=args.fdr,
                dpsi_cutoff=args.dpsi,
                dummy=args.dummy,
            )
    else:
        receipt = run_single_cell(
            outdir=args.outdir,
            events=args.events,
            cells=args.cells,
            psi_matrix=args.psi_matrix,
            support_matrix=args.support_matrix,
            expression=args.expression,
            gtf=args.gtf,
            bams=args.bams,
            sc_bam=args.sc_bam,
            sc_bc_tag=args.sc_bc_tag,
            sc_barcode_shares=args.sc_barcode_shares,
            splicemanifold_dir=args.splicemanifold_dir,
            scsplicenet_dir=args.scsplicenet_dir,
            scsplicetrace_dir=args.scsplicetrace_dir,
            subtypes_dirs=args.subtypes_dirs,
            study_name=args.study_name,
            species=args.species,
            fdr_cutoff=args.fdr,
            dpsi_cutoff=args.dpsi,
            dummy=args.dummy,
        )

    if args.emit_versions_yml is not None:
        write_nfcore_versions_yml(args.emit_versions_yml, args.process_name)
    print(json.dumps(receipt, indent=2))
    return 0


def studio_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="katana-studio",
        description="Katana Studio 1.0.1: Launch the interactive local web workbench on a REPORT.zip or report directory.",
    )
    parser.add_argument("--version", action="version", version=f"katana-studio {__version__}")
    parser.add_argument("report", nargs="?", default=None, help="Optional REPORT.zip or report directory to open immediately")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=18775, help="Bind port (default: 18775)")
    parser.add_argument("--no-browser", action="store_true", help="Do not launch browser automatically")
    args = parser.parse_args(argv)
    open_workspace(args.report, host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(studio_main())
