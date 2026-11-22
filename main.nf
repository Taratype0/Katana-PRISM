#!/usr/bin/env nextflow
nextflow.enable.dsl = 2

/*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Katana (nf-core style Alternative Splicing, Isoform & Cis-Regulatory Pipeline)
    Repository: Katana-PRISM | Version: 1.0.1
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Two decoupled yet convergent execution paradigms:
      1. End-to-End Scheduled Execution via Nextflow (`main.nf`):
         - Bulk RNA-seq (`--mode complete | upstream-only | downstream-only`):
           FASTQ / BAM -> QC, STAR, Salmon, MultiQC, rMATS, DEXSeq, SUPPA ->
           SpliceManifold (PS-DAS) + SpliceNet + SpliceTrace + Katana Report
         - Single-Cell Smart-seq3 (`--mode single-cell` or `-entry SINGLE_CELL`):
           SpliceManifold + scSpliceNet + scSpliceTrace + Katana Report
      2. Standalone Lightweight Python Packages (`packages/*`, pip-installable):
         - `splicemanifold`, `splicenet`, `splicetrace`, `scsplicenet`, `scsplicetrace`, `katana-studio`
----------------------------------------------------------------------------------------
*/

include { KATANA_PIPELINE }           from './workflows/katana_core'
include { KATANA_MODULAR_DOWNSTREAM } from './workflows/katana_modular_downstream'
include { KATANA_SINGLE_CELL; KATANA_SINGLE_CELL_FROM_FASTQ } from './workflows/katana_singlecell'
include { KATANA_TARGETS }            from './workflows/katana_targets'

def requireParameters(names) {
    names.each { name -> if (!params[name]) error("Required parameter missing: --${name}") }
}

def validateStrandednessParameters() {
    def stranded = Double.parseDouble(params.stranded_threshold.toString())
    def unstranded = Double.parseDouble(params.unstranded_threshold.toString())
    if (!Double.isFinite(stranded) || stranded <= 0.5 || stranded > 1.0 ||
        !Double.isFinite(unstranded) || unstranded < 0.0 || unstranded >= 2.0 * stranded - 1.0) {
        error('Require 0.5 < stranded_threshold <= 1 and 0 <= unstranded_threshold < 2*stranded_threshold-1')
    }
}

workflow SINGLE_CELL {
    if (params.sc_fastq_manifest || params.sc_10x_fastq_sheet || params.sc_sj_dir || params.sc_bam) {
        requireParameters(['sc_metadata', 'fasta', 'gtf'])
        ch_meta  = Channel.value(file(params.sc_metadata, checkIfExists: true))
        ch_expr  = params.sc_expression ? Channel.value(file(params.sc_expression, checkIfExists: true)) : Channel.value(file('NO_EXPRESSION'))
        ch_fasta = Channel.value(file(params.fasta, checkIfExists: true))
        ch_gtf   = Channel.value(file(params.gtf, checkIfExists: true))
        KATANA_SINGLE_CELL_FROM_FASTQ(ch_meta, ch_expr, ch_fasta, ch_gtf)
    } else {
        requireParameters(['sc_events', 'sc_psi', 'sc_metadata', 'sc_event_genes', 'fasta', 'gtf'])
        ch_events = Channel.value(file(params.sc_events, checkIfExists: true))
        ch_psi    = Channel.value(file(params.sc_psi, checkIfExists: true))
        ch_sup    = params.sc_support ? Channel.value(file(params.sc_support, checkIfExists: true)) : Channel.value(file('NO_SUPPORT'))
        ch_meta   = Channel.value(file(params.sc_metadata, checkIfExists: true))
        ch_eg     = Channel.value(file(params.sc_event_genes, checkIfExists: true))
        ch_expr   = params.sc_expression ? Channel.value(file(params.sc_expression, checkIfExists: true)) : Channel.value(file('NO_EXPRESSION'))
        ch_fasta  = Channel.value(file(params.fasta, checkIfExists: true))
        ch_gtf    = Channel.value(file(params.gtf, checkIfExists: true))

        KATANA_SINGLE_CELL(ch_events, ch_psi, ch_sup, ch_meta, ch_eg, ch_expr, ch_fasta, ch_gtf)
    }
}

workflow {
    if (params.help) {
        log.info """
        ================================================================================
         Katana-PRISM — Alternative Splicing & Cis-Regulatory Discovery Pipeline (1.0.0)
        ================================================================================
         Usage:
           1) Full Bulk End-to-End (FASTQ or BAM -> MultiQC/STAR/Salmon/rMATS/DEXSeq -> SpliceManifold + SpliceNet + SpliceTrace + Katana Report):
              nextflow run main.nf \\
                --mode complete \\
                --input samplesheet.csv \\
                --contrasts contrasts.csv \\
                --fasta genome.fa --gtf annotation.gtf \\
                --species mouse --reference_build GRCm39 \\
                --outdir results -profile docker

           2) Upstream Alignment & Splicing Callers Only:
              nextflow run main.nf --mode upstream-only --input samplesheet.csv --contrasts contrasts.csv --fasta genome.fa --gtf annotation.gtf --outdir results

           3) Downstream Modular Discovery Only (on existing katana_bundle):
              nextflow run main.nf --mode downstream-only --bundle results/katana_bundle --contrasts contrasts.csv --fasta genome.fa --gtf annotation.gtf --outdir results_downstream

           4) Single-Cell Smart-seq3 End-to-End (SpliceManifold + scSpliceNet + scSpliceTrace + Katana Report):
              nextflow run main.nf --mode single-cell --sc_events events.tsv --sc_psi psi.tsv.gz --sc_metadata cells.tsv --sc_event_genes event_genes.tsv --fasta genome.fa --gtf annotation.gtf --outdir results_sc

           5) Selected-Target 4-Track GeneCard / EventCard + Native Sashimi Follow-up:
              nextflow run main.nf -entry TARGETS --target_request request.zip --bams_dir /path/to/bams --gtf annotation.gtf --outdir results_targets
        ================================================================================
        """.stripIndent()
    } else {
        if (!(params.mode in ['complete', 'upstream-only', 'downstream-only', 'single-cell', 'targets'])) {
            error("Invalid --mode '${params.mode}'; choose complete, upstream-only, downstream-only, single-cell, or targets")
        }

        if (params.mode == 'targets') {
            TARGETS()
        } else if (params.mode == 'single-cell') {
            SINGLE_CELL()
        } else if (params.mode == 'downstream-only') {
            requireParameters(['bundle', 'contrasts', 'fasta', 'gtf'])
            ch_bundle    = Channel.value(file(params.bundle, checkIfExists: true))
            ch_contrasts = Channel.value(file(params.contrasts, checkIfExists: true))
            ch_fasta     = Channel.value(file(params.fasta, checkIfExists: true))
            ch_gtf       = Channel.value(file(params.gtf, checkIfExists: true))
            KATANA_MODULAR_DOWNSTREAM(ch_bundle, ch_contrasts, ch_fasta, ch_gtf)
        } else {
            validateStrandednessParameters()
            requireParameters(['input', 'contrasts', 'fasta', 'gtf'])
            ch_samples   = Channel.value(file(params.input, checkIfExists: true))
            ch_contrasts = Channel.value(file(params.contrasts, checkIfExists: true))
            ch_fasta     = Channel.value(file(params.fasta, checkIfExists: true))
            ch_gtf       = Channel.value(file(params.gtf, checkIfExists: true))

            KATANA_PIPELINE(ch_samples, ch_contrasts, ch_fasta, ch_gtf)

            if (params.mode == 'complete') {
                KATANA_MODULAR_DOWNSTREAM(
                    KATANA_PIPELINE.out.bundle,
                    KATANA_PIPELINE.out.contrastsheet,
                    KATANA_PIPELINE.out.fasta,
                    KATANA_PIPELINE.out.gtf
                )
            }
        }
    }
}

// A new follow-up run from a portable request exported by the browser.
workflow TARGETS {
    if (params.help) { log.info "Katana-PRISM -entry TARGETS --target_request request.zip [--bams_dir DIR --gtf FILE | --target_bindings bindings.json] --outdir results"; return }
    if (!params.target_request) error('TARGETS requires --target_request')
    def bindings = params.target_bindings ? new groovy.json.JsonSlurper().parse(file(params.target_bindings).toFile()) : [resources:[:]]
    if (bindings.schema && bindings.schema != 'katana.target_bindings/v1') error('Unsupported target bindings schema')
    def base = params.target_bindings ? file(params.target_bindings).toAbsolutePath().parent : launchDir
    def resourceMap = bindings.resources ?: [:]
    def resourceIds = resourceMap.keySet().findAll { k -> resourceMap[k] != null && resourceMap[k].toString().trim() != '' }.toList().sort()
    def resourcePaths = resourceIds.collect { key ->
        def location = java.nio.file.Paths.get(resourceMap[key].toString())
        file(location.isAbsolute() ? location.toString() : base.resolve(location).toString(), checkIfExists:true)
    }
    def ch_bams   = params.bams_dir ? Channel.value(file(params.bams_dir, checkIfExists: true)) : Channel.value(file('NO_BAMS_DIR'))
    def ch_scbam  = params.sc_bam ? Channel.value(file(params.sc_bam, checkIfExists: true)) : Channel.value(file('NO_SC_BAM'))
    def ch_shares = params.sc_barcode_shares ? Channel.value(file(params.sc_barcode_shares, checkIfExists: true)) : Channel.value(file('NO_SC_SHARES'))
    def ch_gtf    = params.gtf ? Channel.value(file(params.gtf, checkIfExists: true)) : Channel.value(file('NO_GTF'))
    KATANA_TARGETS(
        Channel.value(file(params.target_request, checkIfExists: true)),
        Channel.value(resourceIds),
        Channel.value(resourcePaths),
        ch_bams,
        ch_scbam,
        ch_shares,
        ch_gtf
    )
}
