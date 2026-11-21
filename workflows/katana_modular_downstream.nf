// Native nf-core scatter-gather downstream workflow orchestrating Katana-PRISM 1.0.0 packages:
//   1. SPLICEMANIFOLD (study-wide PS-DAS qualification + Beyond-Jutils PSI Clustermap/PCA/UMAP)
//   2. SPLICENET      (study-wide Excess-BioVar Signed TOM co-splicing network)
//   3. SPLICETRACE    (per-contrast parallel 170-RBP splicing map, module bridge & NMD/UniProt consequences)
//   4. KATANA_REPORT  (unified 4-facet GO/KEGG ORA + MultiQC 5'->3' synthesis + REPORT.html & REPORT.zip)

include { SPLICEMANIFOLD; SPLICENET; SPLICETRACE; KATANA_REPORT } from '../modules/local/katana_prism_modules'

workflow KATANA_MODULAR_DOWNSTREAM {
    take:
    ch_bundle        // path: katana_bundle
    ch_contrasts     // path: contrastsheet.csv
    ch_fasta         // path: genome.fasta
    ch_gtf           // path: annotation.gtf

    main:
    ch_versions = Channel.empty()

    // 1. Study-wide PS-DAS qualification & Splicing Manifold (SpliceManifold)
    SPLICEMANIFOLD(
        params.study_name,
        ch_bundle,
        params.rmats_mode,
        params.das_fdr,
        params.das_dpsi
    )
    ch_manifold = SPLICEMANIFOLD.out.results.first()
    ch_versions = ch_versions.mix(SPLICEMANIFOLD.out.versions)

    // 2. Study-wide Co-Splicing Network (SpliceNet)
    if (!params.skip_network) {
        SPLICENET(
            params.study_name,
            ch_bundle,
            params.rmats_mode,
            params.network_min_module_size,
            params.network_merge_cut_height,
            params.with_network_bootstrap ? 30 : 0
        )
        ch_splicenet = SPLICENET.out.results.first()
        ch_versions  = ch_versions.mix(SPLICENET.out.versions)
    } else {
        ch_splicenet = Channel.value(file('NO_SPLICENET'))
    }

    // 3. Per-Contrast Parallel Cis-Regulatory RBP Map & Isoform Protein Consequences (SpliceTrace)
    if (!params.skip_splicetrace) {
        ch_contrast_rmats = ch_contrasts
            .splitCsv(header: true)
            .combine(ch_bundle)
            .map { row, bundle ->
                def cid = row.contrast ?: row.contrast_id
                tuple(cid, bundle.resolve("rmats/${cid}/rmats_post"))
            }

        SPLICETRACE(
            ch_contrast_rmats,
            ch_fasta,
            ch_gtf,
            params.species ?: 'mouse',
            ch_splicenet,
            ch_bundle
        )
        ch_splicetrace_all = SPLICETRACE.out.results.map { cid, dir -> dir }.collect()
        ch_versions        = ch_versions.mix(SPLICETRACE.out.versions.first())
    } else {
        ch_splicetrace_all = Channel.value(file('NO_SPLICETRACE'))
    }

    // 4. Unified Executive Report & Portable Archive (Katana Report)
    KATANA_REPORT(
        params.study_name,
        ch_bundle,
        ch_fasta,
        ch_gtf,
        params.species ?: 'auto',
        params.rmats_mode ?: 'JCEC',
        params.das_fdr,
        params.das_dpsi,
        ch_manifold,
        ch_splicenet,
        ch_splicetrace_all
    )
    ch_versions = ch_versions.mix(KATANA_REPORT.out.versions)

    emit:
    report_html    = KATANA_REPORT.out.html
    report_zip     = KATANA_REPORT.out.archive
    report_dir     = KATANA_REPORT.out.report_dir
    splicemanifold = ch_manifold
    splicenet      = ch_splicenet
    versions       = ch_versions
}
