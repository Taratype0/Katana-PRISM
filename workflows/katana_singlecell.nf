// Native nf-core Single-Cell splicing workflow supporting BOTH:
//   1. True One-Step FASTQ -> Report (`KATANA_SINGLE_CELL_FROM_FASTQ`: Smart-seq2/3 manifest OR 10x barcoded sheet)
//   2. Precomputed Matrix -> Report (`KATANA_SINGLE_CELL`)
// Orchestrates:
//   - PREPARE_REFERENCES + STAR_GENOMEGENERATE (when --star_index is not supplied)
//   - SC_STARSOLO_SMARTSEQ / SC_STARSOLO_DROPLET + SC_BUILD_SPLICE_MATRICES
//   - SPLICEMANIFOLD_SC (`splicemanifold single-cell`)
//   - SCSPLICENET       (`scsplicenet`)
//   - SCSPLICETRACE     (`scsplicetrace`, with host expression RBP gating)
//   - KATANA_REPORT_SC  (`katana-report single-cell`)

include { PREPARE_REFERENCES }   from '../modules/local/prepare_references'
include { STAR_GENOMEGENERATE }  from '../modules/nf-core/star/genomegenerate/main'
include {
    SC_STARSOLO_SMARTSEQ;
    SC_STARSOLO_DROPLET;
    SC_BUILD_SPLICE_MATRICES;
    SPLICEMANIFOLD_SC;
    SCSPLICENET;
    SCSPLICETRACE;
    KATANA_REPORT_SC
} from '../modules/local/katana_prism_modules'

workflow KATANA_SINGLE_CELL {
    take:
    ch_events
    ch_psi
    ch_support
    ch_metadata
    ch_event_genes
    ch_expression
    ch_fasta
    ch_gtf

    main:
    ch_versions = Channel.empty()

    SPLICEMANIFOLD_SC(
        params.study_name,
        ch_psi,
        ch_support,
        ch_metadata,
        ch_expression
    )
    ch_versions = ch_versions.mix(SPLICEMANIFOLD_SC.out.versions)

    if (!params.skip_network) {
        SCSPLICENET(
            params.study_name,
            ch_psi,
            ch_support,
            ch_metadata,
            ch_event_genes,
            ch_expression
        )
        ch_sc_net   = SCSPLICENET.out.results.first()
        ch_versions = ch_versions.mix(SCSPLICENET.out.versions)
    } else {
        ch_sc_net = Channel.value(file('NO_SCSPLICENET'))
    }

    if (!params.skip_splicetrace && params.sc_numerator && params.sc_denominator) {
        SCSPLICETRACE(
            ch_events,
            ch_psi,
            ch_support,
            ch_metadata,
            ch_expression,
            ch_fasta,
            ch_gtf,
            params.species ?: 'human',
            params.sc_numerator,
            params.sc_denominator,
            ch_sc_net
        )
        ch_versions = ch_versions.mix(SCSPLICETRACE.out.versions)
    }

    KATANA_REPORT_SC(
        params.study_name,
        ch_events,
        ch_metadata,
        ch_psi,
        ch_support,
        ch_expression,
        params.species ?: 'human',
        params.das_fdr,
        params.das_dpsi
    )
    ch_versions = ch_versions.mix(KATANA_REPORT_SC.out.versions)

    emit:
    report_html = KATANA_REPORT_SC.out.html
    report_zip  = KATANA_REPORT_SC.out.archive
    versions    = ch_versions
}

workflow KATANA_SINGLE_CELL_FROM_FASTQ {
    take:
    ch_metadata
    ch_expression
    ch_fasta_in
    ch_gtf_in

    main:
    ch_versions = Channel.empty()

    PREPARE_REFERENCES(ch_fasta_in, ch_gtf_in)
    ch_fasta = PREPARE_REFERENCES.out.fasta
    ch_gtf   = PREPARE_REFERENCES.out.gtf
    ch_versions = ch_versions.mix(PREPARE_REFERENCES.out.versions)

    if (params.star_index) {
        ch_star_idx = Channel.value(file(params.star_index, checkIfExists: true))
    } else {
        STAR_GENOMEGENERATE(ch_fasta, ch_gtf)
        ch_star_idx = STAR_GENOMEGENERATE.out.index
        ch_versions = ch_versions.mix(STAR_GENOMEGENERATE.out.versions)
    }

    if (params.sc_fastq_manifest) {
        def sep_m = params.sc_fastq_manifest.toString().toLowerCase().endsWith('.csv') ? ',' : '\t'
        ch_rows = Channel
            .fromPath(params.sc_fastq_manifest, checkIfExists: true)
            .splitCsv(header: true, sep: sep_m)
            .map { row ->
                def cid = (row.cell_id ?: row.sample_id ?: row.sample ?: row.id)?.toString()?.trim()
                def fq1 = (row.fastq_1 ?: row.read1 ?: row.r1)?.toString()?.trim()
                def fq2 = (row.fastq_2 ?: row.read2 ?: row.r2 ?: '')?.toString()?.trim()
                if (!cid || !fq1) {
                    error("Each row in --sc_fastq_manifest must provide cell_id and fastq_1 (got: ${row})")
                }
                [
                    meta: [cell_id: cid],
                    r1: file(fq1, checkIfExists: true),
                    r2: fq2 ? file(fq2, checkIfExists: true) : null
                ]
            }
            .toList()

        ch_meta_list = ch_rows.map { list -> list.collect { it.meta } }
        ch_r1_list   = ch_rows.map { list -> list.collect { it.r1 } }
        ch_r2_list   = ch_rows.map { list ->
            list.every { it.r2 != null } ? list.collect { it.r2 } : file('NO_R2')
        }

        SC_STARSOLO_SMARTSEQ(
            params.study_name,
            ch_meta_list,
            ch_r1_list,
            ch_r2_list,
            ch_star_idx,
            ch_gtf
        )
        ch_sj_prefixes = Channel.value([''])
        ch_sj_dirs     = SC_STARSOLO_SMARTSEQ.out.sj_raw
        ch_versions    = ch_versions.mix(SC_STARSOLO_SMARTSEQ.out.versions)
    } else {
        def sep_d = params.sc_10x_fastq_sheet.toString().toLowerCase().endsWith('.csv') ? ',' : '\t'
        ch_libs = Channel
            .fromPath(params.sc_10x_fastq_sheet, checkIfExists: true)
            .splitCsv(header: true, sep: sep_d)
            .map { row ->
                def sid = (row.sample_id ?: row.sample ?: row.library_id ?: '')?.toString()?.trim()
                def r1  = (row.fastq_r1 ?: row.fastq_1 ?: row.r1)?.toString()?.trim()
                def r2  = (row.fastq_r2 ?: row.fastq_2 ?: row.r2)?.toString()?.trim()
                if (!sid || !r1 || !r2) {
                    error("Each row in --sc_10x_fastq_sheet must provide sample_id, fastq_r1, and fastq_r2 (got: ${row})")
                }
                tuple(sid, file(r1, checkIfExists: true), file(r2, checkIfExists: true))
            }

        ch_wl = params.sc_cb_whitelist ? Channel.value(file(params.sc_cb_whitelist, checkIfExists: true)) : Channel.value(file('NO_WHITELIST'))

        SC_STARSOLO_DROPLET(
            ch_libs,
            ch_wl,
            ch_star_idx,
            ch_gtf,
            params.sc_cb_len,
            params.sc_umi_len
        )
        ch_collected   = SC_STARSOLO_DROPLET.out.sj_raw.toList()
        ch_sj_prefixes = ch_collected.map { items -> items.collect { it[0] } }
        ch_sj_dirs     = ch_collected.map { items -> items.collect { it[1] } }
        ch_versions    = ch_versions.mix(SC_STARSOLO_DROPLET.out.versions.first())
    }

    SC_BUILD_SPLICE_MATRICES(
        params.study_name,
        ch_sj_prefixes,
        ch_sj_dirs,
        ch_metadata,
        ch_gtf,
        params.sc_group_col ?: 'condition',
        params.sc_numerator ?: '',
        params.sc_denominator ?: '',
        params.sc_min_cells,
        params.sc_min_cell_support,
        params.sc_pseudobulk_by ?: ''
    )
    ch_versions = ch_versions.mix(SC_BUILD_SPLICE_MATRICES.out.versions)

    KATANA_SINGLE_CELL(
        SC_BUILD_SPLICE_MATRICES.out.events,
        SC_BUILD_SPLICE_MATRICES.out.psi,
        SC_BUILD_SPLICE_MATRICES.out.support,
        SC_BUILD_SPLICE_MATRICES.out.metadata,
        SC_BUILD_SPLICE_MATRICES.out.event_genes,
        ch_expression,
        ch_fasta,
        ch_gtf
    )
    ch_versions = ch_versions.mix(KATANA_SINGLE_CELL.out.versions)

    emit:
    report_html = KATANA_SINGLE_CELL.out.report_html
    report_zip  = KATANA_SINGLE_CELL.out.report_zip
    versions    = ch_versions
}
