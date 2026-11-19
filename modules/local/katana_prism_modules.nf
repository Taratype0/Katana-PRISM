// Native nf-core DSL2 processes invoking Katana-PRISM's independent 1.0.0 analytical packages:
// Bulk RNA-seq:
//   - SPLICEMANIFOLD : PS-DAS qualification + Empirical-Bayes PSI Clustermap / PERMANOVA PCA / Event-PSI UMAP (`splicemanifold`)
//   - SPLICENET      : Excess-BioVar debiased signed TOM co-splicing network & hub modules (`splicenet`)
//   - SPLICETRACE    : Per-contrast 170-RBP PWM 11-channel splicing map, module bridge & 50-nt EJC NMD / UniProt consequences (`splicetrace`)
//   - KATANA_REPORT  : 4-facet GO/KEGG ORA, MultiQC 5'->3' synthesis & SHA256-verified REPORT.html + REPORT.zip (`katana-report`)
// Single-Cell Smart-seq3:
//   - SPLICEMANIFOLD_SC, SCSPLICENET, SCSPLICETRACE, KATANA_REPORT_SC

process SPLICEMANIFOLD {
    tag "${study_name}"
    label 'process_low'

    input:
    val  study_name
    path bundle
    val  rmats_mode
    val  das_fdr
    val  das_dpsi

    output:
    path "splicemanifold_out", emit: results
    path "versions.yml"      , emit: versions

    script:
    """
    splicemanifold bulk \
        --bundle "${bundle}" \
        --rmats-mode "${rmats_mode}" \
        --fdr-cutoff ${das_fdr} \
        --dpsi-cutoff ${das_dpsi} \
        --outdir splicemanifold_out \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SPLICENET {
    tag "${study_name}"
    label 'process_medium'

    input:
    val  study_name
    path bundle
    val  rmats_mode
    val  min_module_size
    val  merge_cut_height
    val  subsamples

    output:
    path "splicenet_out", emit: results
    path "versions.yml" , emit: versions

    script:
    """
    splicenet \
        --bundle "${bundle}" \
        --rmats-mode "${rmats_mode}" \
        --min-module-size ${min_module_size} \
        --merge-cut-height ${merge_cut_height} \
        --subsamples ${subsamples} \
        --threads ${task.cpus} \
        --output-dir splicenet_out \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SPLICETRACE {
    tag "${contrast_id}"
    label 'process_medium'

    input:
    tuple val(contrast_id), path(rmats_dir)
    path fasta
    path gtf
    val  species
    path splicenet_dir

    output:
    tuple val(contrast_id), path("splicetrace_${contrast_id}"), emit: results
    path "versions.yml"                                        , emit: versions

    script:
    def net_arg = splicenet_dir.name != 'NO_SPLICENET' ? "--splicenet-modules ${splicenet_dir}/module_assignments.tsv" : ''
    def sp_val  = (species in ['human', 'mouse']) ? species : 'mouse'
    """
    fasta_ref="${fasta}"
    if [[ "\$fasta_ref" == *.gz ]]; then gzip -dc "\$fasta_ref" > genome_unpacked.fa; fasta_ref="genome_unpacked.fa"; fi
    if [[ ! -f "\${fasta_ref}.fai" ]]; then cp -L "\$fasta_ref" genome_local.fa && fasta_ref="genome_local.fa" && samtools faidx "\$fasta_ref"; fi
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
    splicetrace \
        --rmats "${rmats_dir}" \
        --fasta "\${fasta_ref}" \
        --gtf "\${gtf_ref}" \
        --species "${sp_val}" \
        ${net_arg} \
        --threads ${task.cpus} \
        --output-dir "splicetrace_${contrast_id}" \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process KATANA_REPORT {
    tag "${study_name}"
    label 'process_low'

    input:
    val  study_name
    path bundle
    path fasta
    path gtf
    val  species
    val  das_fdr
    val  das_dpsi
    path splicemanifold_out
    path splicenet_out
    path splicetrace_outs

    output:
    path "report/REPORT.html"         , emit: html
    path "report/REPORT.zip"          , emit: archive
    path "report/report-manifest.json", emit: manifest
    path "report"                     , emit: report_dir
    path "versions.yml"               , emit: versions

    script:
    def net_arg   = splicenet_out.name != 'NO_SPLICENET' ? "--splicenet-dir ${splicenet_out}" : ''
    def trace_arg = (splicetrace_outs && splicetrace_outs.toString() != 'NO_SPLICETRACE') ? "--splicetrace-dirs ${splicetrace_outs}" : ''
    def sp_val    = (species in ['auto', 'human', 'mouse']) ? species : 'auto'
    """
    fasta_ref="${fasta}"
    if [[ "\$fasta_ref" == *.gz ]]; then gzip -dc "\$fasta_ref" > genome_unpacked.fa; fasta_ref="genome_unpacked.fa"; fi
    if [[ ! -f "\${fasta_ref}.fai" ]]; then cp -L "\$fasta_ref" genome_local.fa && fasta_ref="genome_local.fa" && samtools faidx "\$fasta_ref"; fi
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
    katana-report bulk \
        --bundle "${bundle}" \
        --fasta "\${fasta_ref}" \
        --gtf "\${gtf_ref}" \
        --study-name "${study_name}" \
        --species "${sp_val}" \
        --fdr ${das_fdr} \
        --dpsi ${das_dpsi} \
        ${net_arg} \
        ${trace_arg} \
        --outdir report \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SPLICEMANIFOLD_SC {
    tag "${study_name}"
    label 'process_medium'

    input:
    val  study_name
    path psi_matrix
    path support_matrix
    path cell_metadata
    path host_expression

    output:
    path "sc_manifold_out", emit: results
    path "versions.yml"   , emit: versions

    script:
    def sup_arg  = support_matrix.name != 'NO_SUPPORT' ? "--support ${support_matrix}" : ''
    def expr_arg = host_expression.name != 'NO_EXPRESSION' ? "--expression ${host_expression}" : ''
    """
    splicemanifold single-cell \
        --psi "${psi_matrix}" \
        --metadata "${cell_metadata}" \
        ${sup_arg} \
        ${expr_arg} \
        --cells-as-rows \
        --outdir sc_manifold_out \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SCSPLICENET {
    tag "${study_name}"
    label 'process_medium'

    input:
    val  study_name
    path psi_matrix
    path support_matrix
    path cell_metadata
    path event_genes
    path host_expression

    output:
    path "scsplicenet_out", emit: results
    path "versions.yml"   , emit: versions

    script:
    def sup_arg  = support_matrix.name != 'NO_SUPPORT' ? "--support ${support_matrix}" : ''
    def expr_arg = host_expression.name != 'NO_EXPRESSION' ? "--host-expr ${host_expression}" : ''
    """
    scsplicenet \
        --usage "${psi_matrix}" \
        --metadata "${cell_metadata}" \
        --event-genes "${event_genes}" \
        ${sup_arg} \
        ${expr_arg} \
        --threads ${task.cpus} \
        --out scsplicenet_out \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SCSPLICETRACE {
    tag "${numerator}_vs_${denominator}"
    label 'process_medium'

    input:
    path events_table
    path psi_matrix
    path support_matrix
    path cell_metadata
    path fasta
    path gtf
    val  species
    val  numerator
    val  denominator
    path scsplicenet_dir

    output:
    path "scsplicetrace_out", emit: results
    path "versions.yml"     , emit: versions

    script:
    def sup_arg = support_matrix.name != 'NO_SUPPORT' ? "--support ${support_matrix}" : ''
    def net_arg = scsplicenet_dir.name != 'NO_SCSPLICENET' ? "--splicenet-modules ${scsplicenet_dir}/scsplicenet_modules.tsv" : ''
    def sp_val  = (species in ['human', 'mouse']) ? species : 'human'
    """
    fasta_ref="${fasta}"
    if [[ "\$fasta_ref" == *.gz ]]; then gzip -dc "\$fasta_ref" > genome_unpacked.fa; fasta_ref="genome_unpacked.fa"; fi
    if [[ ! -f "\${fasta_ref}.fai" ]]; then cp -L "\$fasta_ref" genome_local.fa && fasta_ref="genome_local.fa" && samtools faidx "\$fasta_ref"; fi
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
    scsplicetrace \
        --events "${events_table}" \
        --psi "${psi_matrix}" \
        --metadata "${cell_metadata}" \
        --fasta "\${fasta_ref}" \
        --gtf "\${gtf_ref}" \
        --species "${sp_val}" \
        --numerator "${numerator}" \
        --denominator "${denominator}" \
        ${sup_arg} \
        ${net_arg} \
        --threads ${task.cpus} \
        --out scsplicetrace_out \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process KATANA_REPORT_SC {
    tag "${study_name}"
    label 'process_low'

    input:
    val  study_name
    path events_table
    path cell_metadata
    path psi_matrix
    path support_matrix
    path host_expression
    val  species
    val  das_fdr
    val  das_dpsi

    output:
    path "report/REPORT.html", emit: html
    path "report/REPORT.zip" , emit: archive
    path "report"            , emit: report_dir
    path "versions.yml"      , emit: versions

    script:
    def sup_arg  = support_matrix.name != 'NO_SUPPORT' ? "--support-matrix ${support_matrix}" : ''
    def expr_arg = host_expression.name != 'NO_EXPRESSION' ? "--expression ${host_expression}" : ''
    def sp_val   = (species in ['human', 'mouse']) ? species : 'human'
    """
    katana-report single-cell \
        --events "${events_table}" \
        --cells "${cell_metadata}" \
        --psi-matrix "${psi_matrix}" \
        ${sup_arg} \
        ${expr_arg} \
        --study-name "${study_name}" \
        --species "${sp_val}" \
        --fdr ${das_fdr} \
        --dpsi ${das_dpsi} \
        --outdir report \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}
