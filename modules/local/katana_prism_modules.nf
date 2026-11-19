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
    path host_expression
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
    def sup_arg  = support_matrix.name != 'NO_SUPPORT' ? "--support ${support_matrix}" : ''
    def expr_arg = host_expression.name != 'NO_EXPRESSION' ? "--expression ${host_expression}" : ''
    def net_arg  = scsplicenet_dir.name != 'NO_SCSPLICENET' ? "--splicenet-modules ${scsplicenet_dir}/scsplicenet_modules.tsv" : ''
    def sp_val   = (species in ['human', 'mouse']) ? species : 'human'
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
        ${expr_arg} \
        ${net_arg} \
        --threads ${task.cpus} \
        --out scsplicetrace_out \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SC_STARSOLO_SMARTSEQ {
    tag "${study_name}"
    label 'process_high'

    conda "bioconda::star=2.7.10a bioconda::samtools=1.16.1 conda-forge::gawk=5.1.0"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/mulled-v2-1fa26d1ce03c295fe2fdcf85831a92fbcbd7e8c2:1df389393721fc66f3fd8778ad938ac711951107-0' :
        'quay.io/biocontainers/mulled-v2-1fa26d1ce03c295fe2fdcf85831a92fbcbd7e8c2:1df389393721fc66f3fd8778ad938ac711951107-0' }"

    input:
    val  study_name
    val  manifest_rows
    path staged_r1, stageAs: 'fq_r1/*'
    path staged_r2, stageAs: 'fq_r2/*'
    path star_index
    path gtf

    output:
    path "Solo.out/SJ/raw", emit: sj_raw
    path "versions.yml"   , emit: versions

    script:
    def r1_list = [staged_r1].flatten()
    def r2_list = staged_r2.name == 'NO_R2' ? [] : [staged_r2].flatten()
    def is_gz   = r1_list[0].name.endsWith('.gz') ? '--readFilesCommand zcat' : ''
    def lines   = []
    manifest_rows.eachWithIndex { row, idx ->
        def f1 = r1_list[idx]
        def f2 = r2_list ? r2_list[idx] : '-'
        lines << "${f1}\t${f2}\t${row.cell_id}"
    }
    def manifest_body = lines.join('\n')
    """
    cat << 'EOF_MANIFEST' > starsolo_manifest.tsv
${manifest_body}
EOF_MANIFEST

    STAR \
        --genomeDir "${star_index}" \
        --sjdbGTFfile "${gtf}" \
        --runThreadN ${task.cpus} \
        --soloType SmartSeq \
        --readFilesManifest starsolo_manifest.tsv \
        ${is_gz} \
        --soloUMIdedup NoDedup \
        --soloStrand Unstranded \
        --soloFeatures SJ \
        --outSAMtype None

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        star: \$(STAR --version | sed -e "s/STAR_//g")
    END_VERSIONS
    """
}

process SC_STARSOLO_DROPLET {
    tag "${sample_id}"
    label 'process_high'

    conda "bioconda::star=2.7.10a bioconda::samtools=1.16.1 conda-forge::gawk=5.1.0"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/mulled-v2-1fa26d1ce03c295fe2fdcf85831a92fbcbd7e8c2:1df389393721fc66f3fd8778ad938ac711951107-0' :
        'quay.io/biocontainers/mulled-v2-1fa26d1ce03c295fe2fdcf85831a92fbcbd7e8c2:1df389393721fc66f3fd8778ad938ac711951107-0' }"

    input:
    tuple val(sample_id), path(fastq_r1, stageAs: 'r1/*'), path(fastq_r2, stageAs: 'r2/*')
    path cb_whitelist
    path star_index
    path gtf
    val  cb_len
    val  umi_len

    output:
    tuple val(sample_id), path("${sample_id}_SJ_raw"), emit: sj_raw
    path "versions.yml"                              , emit: versions

    script:
    def is_gz = fastq_r2.name.endsWith('.gz') ? '--readFilesCommand zcat' : ''
    def wl_arg = cb_whitelist.name != 'NO_WHITELIST' ? "${cb_whitelist}" : 'None'
    """
    wl_file="${wl_arg}"
    if [[ "\$wl_file" == *.gz ]]; then gzip -dc "\$wl_file" > whitelist_unpacked.txt; wl_file="whitelist_unpacked.txt"; fi

    STAR \
        --genomeDir "${star_index}" \
        --sjdbGTFfile "${gtf}" \
        --runThreadN ${task.cpus} \
        --soloType CB_UMI_Simple \
        --readFilesIn "${fastq_r2}" "${fastq_r1}" \
        ${is_gz} \
        --soloCBwhitelist "\${wl_file}" \
        --soloCBlen ${cb_len} \
        --soloUMIstart ${cb_len + 1} \
        --soloUMIlen ${umi_len} \
        --soloFeatures SJ \
        --outSAMtype None

    cp -r Solo.out/SJ/raw "${sample_id}_SJ_raw"

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        star: \$(STAR --version | sed -e "s/STAR_//g")
    END_VERSIONS
    """
}

process SC_BUILD_SPLICE_MATRICES {
    tag "${study_name}"
    label 'process_medium'

    input:
    val  study_name
    val  sample_prefixes
    path sj_dirs, stageAs: 'sj_in_*'
    path cell_metadata
    path gtf
    val  group_col
    val  numerator
    val  denominator
    val  min_cells
    val  min_cell_support
    val  pseudobulk_by

    output:
    path "sc_matrices/sc_events.tsv"     , emit: events
    path "sc_matrices/sc_event_genes.tsv", emit: event_genes
    path "sc_matrices/sc_psi.tsv.gz"     , emit: psi
    path "sc_matrices/sc_support.tsv.gz" , emit: support
    path "sc_matrices/sc_cells.tsv"      , emit: metadata
    path "versions.yml"                  , emit: versions

    script:
    def dir_list = [sj_dirs].flatten()
    def specs = []
    dir_list.eachWithIndex { d, i ->
        def pref = (sample_prefixes && sample_prefixes[i]) ? "${sample_prefixes[i]}::" : ''
        specs << "${pref}${d}"
    }
    def sj_arg  = specs.collect { "\"${it}\"" }.join(' ')
    def num_arg = numerator ? "--numerator \"${numerator}\"" : ''
    def den_arg = denominator ? "--denominator \"${denominator}\"" : ''
    def pb_arg  = pseudobulk_by ? "--pseudobulk-by \"${pseudobulk_by}\"" : ''
    """
    katana_sc_build_splice_matrices.py \
        --gtf "${gtf}" \
        --sj-dirs ${sj_arg} \
        --metadata "${cell_metadata}" \
        --group-col "${group_col}" \
        ${num_arg} \
        ${den_arg} \
        --min-cells ${min_cells} \
        --min-cell-support ${min_cell_support} \
        ${pb_arg} \
        --outdir sc_matrices

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
    END_VERSIONS
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
