// Native nf-core Single-Cell Smart-seq3 splicing workflow orchestrating:
//   - SPLICEMANIFOLD_SC (`splicemanifold single-cell`)
//   - SCSPLICENET       (`scsplicenet`)
//   - SCSPLICETRACE     (`scsplicetrace`)
//   - KATANA_REPORT_SC  (`katana-report single-cell`)

include { SPLICEMANIFOLD_SC; SCSPLICENET; SCSPLICETRACE; KATANA_REPORT_SC } from '../modules/local/katana_prism_modules'

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
