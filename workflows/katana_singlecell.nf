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
    SC_EXTRACT_MULTIPLEXED_BAM;
    SC_RMATS_FROM_GTF;
    SC_PREPARE_MARVEL_INPUTS;
    SC_MARVEL_QUANT;
    SC_BUILD_SPLICE_MATRICES;
    SC_ENUMERATE_TESTABLE_SUBTYPES;
    SC_SUBTYPE_DOWNSTREAM;
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
    ch_ev_val   = ch_events.first()
    ch_psi_val  = ch_psi.first()
    ch_sup_val  = ch_support.first()
    ch_meta_val = ch_metadata.first()
    ch_eg_val   = ch_event_genes.first()
    ch_expr_val = ch_expression.first()
    ch_fa_val   = ch_fasta.first()
    ch_gtf_val  = ch_gtf.first()

    SPLICEMANIFOLD_SC(
        params.study_name,
        ch_ev_val,
        ch_psi_val,
        ch_sup_val,
        ch_meta_val,
        ch_expr_val
    )
    ch_versions = ch_versions.mix(SPLICEMANIFOLD_SC.out.versions)

    if (!params.skip_network) {
        SCSPLICENET(
            params.study_name,
            ch_psi_val,
            ch_sup_val,
            ch_meta_val,
            ch_eg_val,
            ch_expr_val
        )
        ch_sc_net   = SCSPLICENET.out.results.first()
        ch_versions = ch_versions.mix(SCSPLICENET.out.versions)
    } else {
        ch_sc_net = Channel.value(file('NO_SCSPLICENET'))
    }

    if (!params.skip_splicetrace && params.sc_numerator && params.sc_denominator) {
        SCSPLICETRACE(
            ch_ev_val,
            ch_psi_val,
            ch_sup_val,
            ch_meta_val,
            ch_expr_val,
            ch_fa_val,
            ch_gtf_val,
            params.species ?: 'human',
            params.sc_group_col ?: 'condition',
            params.sc_numerator,
            params.sc_denominator,
            ch_sc_net
        )
        ch_sc_trace = SCSPLICETRACE.out.results.first()
        ch_versions = ch_versions.mix(SCSPLICETRACE.out.versions)
    } else {
        ch_sc_trace = Channel.value(file('NO_SCSPLICETRACE'))
    }

    if (params.sc_contrasts || params.sc_celltype_cols) {
        if (params.sc_contrasts) {
            def sep_c = params.sc_contrasts.toString().toLowerCase().endsWith('.csv') ? ',' : '\t'
            ch_sc_contrast_rows = Channel
                .fromPath(params.sc_contrasts, checkIfExists: true)
                .splitCsv(header: true, sep: sep_c)
                .map { row ->
                    def cid  = (row.contrast_id ?: row.id)?.toString()?.trim()
                    def pcol = (row.population_col ?: 'ALL')?.toString()?.trim()
                    def pval = (row.population_value ?: 'ALL')?.toString()?.trim()
                    def num  = (row.numerator ?: row.treatment)?.toString()?.trim()
                    def den  = (row.denominator ?: row.control)?.toString()?.trim()
                    if (!cid || !num || !den) {
                        error("Each row in --sc_contrasts requires contrast_id, numerator, and denominator (got: ${row})")
                    }
                    tuple(cid, pcol, pval, num, den)
                }
        } else {
            SC_ENUMERATE_TESTABLE_SUBTYPES(
                params.study_name,
                ch_meta_val,
                params.sc_celltype_cols,
                params.sc_group_col ?: 'condition',
                params.sc_numerator ?: '',
                params.sc_denominator ?: '',
                params.sc_condition_pairs ?: '',
                params.sc_min_donors_per_arm ?: 2
            )
            ch_versions = ch_versions.mix(SC_ENUMERATE_TESTABLE_SUBTYPES.out.versions)
            ch_sc_contrast_rows = SC_ENUMERATE_TESTABLE_SUBTYPES.out.contrasts_tsv
                .splitCsv(header: true, sep: '\t')
                .map { row ->
                    tuple(
                        row.contrast_id.toString().trim(),
                        row.population_col.toString().trim(),
                        row.population_value.toString().trim(),
                        row.numerator.toString().trim(),
                        row.denominator.toString().trim()
                    )
                }
        }

        SC_SUBTYPE_DOWNSTREAM(
            ch_sc_contrast_rows,
            ch_ev_val,
            ch_psi_val,
            ch_sup_val,
            ch_meta_val,
            ch_expr_val,
            ch_fa_val,
            ch_gtf_val,
            params.species ?: 'human',
            params.sc_group_col ?: 'condition',
            params.sc_min_cells,
            params.das_fdr,
            params.das_dpsi,
            params.skip_network,
            params.skip_splicetrace
        )
        ch_subtypes_all = SC_SUBTYPE_DOWNSTREAM.out.contrast_dir.collect().ifEmpty([file('NO_SUBTYPES')])
        ch_versions     = ch_versions.mix(SC_SUBTYPE_DOWNSTREAM.out.versions.first())
    } else {
        ch_subtypes_all = Channel.value(file('NO_SUBTYPES'))
    }

    ch_sc_bam_val = params.sc_bam ? Channel.value(file(params.sc_bam, checkIfExists: true)) : Channel.value(file('NO_SC_BAM'))
    ch_sc_bai_val = (params.sc_bam && file("${params.sc_bam}.bai").exists()) ? Channel.value(file("${params.sc_bam}.bai")) : Channel.value(file('NO_SC_BAM_BAI'))
    ch_sc_bcs_val = params.sc_barcode_shares ? Channel.value(file(params.sc_barcode_shares, checkIfExists: true)) : Channel.value(file('NO_BARCODE_SHARES'))

    KATANA_REPORT_SC(
        params.study_name,
        ch_ev_val,
        ch_meta_val,
        ch_psi_val,
        ch_sup_val,
        ch_expr_val,
        ch_gtf_val,
        SPLICEMANIFOLD_SC.out.results.first(),
        ch_sc_net,
        ch_sc_trace,
        ch_subtypes_all,
        params.species ?: 'human',
        params.das_fdr,
        params.das_dpsi,
        ch_sc_bam_val,
        ch_sc_bai_val,
        ch_sc_bcs_val,
        params.sc_bc_tag ?: 'BC'
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
    ch_expr_eff = ch_expression

    PREPARE_REFERENCES(ch_fasta_in, ch_gtf_in)
    ch_fasta = PREPARE_REFERENCES.out.fasta
    ch_gtf   = PREPARE_REFERENCES.out.gtf
    ch_versions = ch_versions.mix(PREPARE_REFERENCES.out.versions)

    if (params.sc_sj_dir) {
        ch_sj_prefixes = Channel.value([''])
        ch_sj_dirs     = Channel.value(file(params.sc_sj_dir, checkIfExists: true))
    } else if (params.sc_bam && !params.sc_fastq_manifest && !params.sc_10x_fastq_sheet) {
        ch_bam_in = Channel.value(file(params.sc_bam, checkIfExists: true))
        ch_bai_in = file("${params.sc_bam}.bai").exists() ? Channel.value(file("${params.sc_bam}.bai")) : Channel.value(file('NO_SC_BAM_BAI'))
        ch_bcs_in = params.sc_barcode_shares ? Channel.value(file(params.sc_barcode_shares, checkIfExists: true)) : Channel.value(file('NO_BARCODE_SHARES'))
        ch_umi_in = params.sc_umi_matrix ? Channel.value(file(params.sc_umi_matrix, checkIfExists: true)) : Channel.value(file('NO_UMI_MATRIX'))
        ch_gn_in  = params.sc_gene_names ? Channel.value(file(params.sc_gene_names, checkIfExists: true)) : Channel.value(file('NO_GENE_NAMES'))

        SC_EXTRACT_MULTIPLEXED_BAM(
            params.study_name,
            ch_bam_in,
            ch_bai_in,
            ch_metadata,
            ch_bcs_in,
            ch_umi_in,
            ch_gn_in,
            params.sc_bc_tag ?: 'BC'
        )
        ch_sj_prefixes = Channel.value([''])
        ch_sj_dirs     = SC_EXTRACT_MULTIPLEXED_BAM.out.sj_raw
        ch_versions    = ch_versions.mix(SC_EXTRACT_MULTIPLEXED_BAM.out.versions)
        if (!params.sc_expression && params.sc_umi_matrix) {
            ch_expr_eff = SC_EXTRACT_MULTIPLEXED_BAM.out.expr_gz
        }
    } else {
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
    }

    if ((params.sc_quantifier ?: 'marvel').toString().toLowerCase() == 'marvel') {
        SC_RMATS_FROM_GTF(ch_gtf)
        ch_versions = ch_versions.mix(SC_RMATS_FROM_GTF.out.versions)

        ch_sc_bam = params.sc_bam ? Channel.value(file(params.sc_bam, checkIfExists: true)) : Channel.value(file('NO_SC_BAM'))
        ch_sc_bai = (params.sc_bam && file("${params.sc_bam}.bai").exists()) ? Channel.value(file("${params.sc_bam}.bai")) : Channel.value(file('NO_SC_BAM_BAI'))
        ch_bcs    = params.sc_barcode_shares ? Channel.value(file(params.sc_barcode_shares, checkIfExists: true)) : Channel.value(file('NO_BARCODE_SHARES'))
        ch_mtar   = params.marvel_tarball ? Channel.value(file(params.marvel_tarball, checkIfExists: true)) : Channel.value(file('NO_MARVEL_TARBALL'))

        SC_PREPARE_MARVEL_INPUTS(
            params.study_name,
            ch_sj_prefixes,
            ch_sj_dirs,
            SC_RMATS_FROM_GTF.out.fromgtf_dir,
            ch_metadata,
            ch_gtf,
            ch_sc_bam,
            ch_sc_bai,
            ch_bcs,
            params.sc_bc_tag ?: 'BC'
        )
        ch_versions = ch_versions.mix(SC_PREPARE_MARVEL_INPUTS.out.versions)

        SC_MARVEL_QUANT(
            params.study_name,
            SC_PREPARE_MARVEL_INPUTS.out.prep_dir,
            SC_RMATS_FROM_GTF.out.fromgtf_dir,
            ch_mtar,
            params.sc_group_col ?: 'condition',
            params.sc_numerator ?: '',
            params.sc_denominator ?: '',
            params.sc_min_cells,
            params.sc_min_cell_support
        )
        ch_versions   = ch_versions.mix(SC_MARVEL_QUANT.out.versions)
        ch_ev_final   = SC_MARVEL_QUANT.out.events
        ch_psi_final  = SC_MARVEL_QUANT.out.psi
        ch_sup_final  = SC_MARVEL_QUANT.out.support
        ch_meta_final = SC_MARVEL_QUANT.out.metadata
        ch_eg_final   = SC_MARVEL_QUANT.out.event_genes
    } else {
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
        ch_versions   = ch_versions.mix(SC_BUILD_SPLICE_MATRICES.out.versions)
        ch_ev_final   = SC_BUILD_SPLICE_MATRICES.out.events
        ch_psi_final  = SC_BUILD_SPLICE_MATRICES.out.psi
        ch_sup_final  = SC_BUILD_SPLICE_MATRICES.out.support
        ch_meta_final = SC_BUILD_SPLICE_MATRICES.out.metadata
        ch_eg_final   = SC_BUILD_SPLICE_MATRICES.out.event_genes
    }

    KATANA_SINGLE_CELL(
        ch_ev_final,
        ch_psi_final,
        ch_sup_final,
        ch_meta_final,
        ch_eg_final,
        ch_expr_eff,
        ch_fasta,
        ch_gtf
    )
    ch_versions = ch_versions.mix(KATANA_SINGLE_CELL.out.versions)

    emit:
    report_html = KATANA_SINGLE_CELL.out.report_html
    report_zip  = KATANA_SINGLE_CELL.out.report_zip
    versions    = ch_versions
}
