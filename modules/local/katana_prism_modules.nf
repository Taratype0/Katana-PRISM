// Native nf-core DSL2 processes invoking Katana-PRISM's independent 1.0.0 analytical packages:
// Bulk RNA-seq:
//   - SPLICEMANIFOLD : PS-DAS qualification + Empirical-Bayes PSI Clustermap / PERMANOVA PCA / Event-PSI UMAP (`splicemanifold`)
//   - SPLICENET      : Excess-BioVar debiased signed TOM co-splicing network & hub modules (`splicenet`)
//   - SPLICETRACE         : Per-contrast 170-RBP PWM 11-channel splicing map, module bridge & 50-nt EJC NMD / UniProt consequences (`splicetrace`)
//   - KATANA_CROSS_PARADIGM: Multi-paradigm resolver (rMATS × DEXSeq × SUPPA2 AF/AL × stageR DTU × GTF) (`katana-cross-paradigm`)
//   - KATANA_REPORT       : 4-facet GO/KEGG ORA, MultiQC 5'->3' synthesis & SHA256-verified REPORT.html + REPORT.zip (`katana-report`)
// Single-Cell Smart-seq3:
//   - SPLICEMANIFOLD_SC, SCSPLICENET, SCSPLICETRACE, KATANA_REPORT_SC

def prismPythonPathBlock(projDir) {
    return "if [[ -d \"${projDir}/packages/katana_studio\" ]]; then export PYTHONPATH=\"${projDir}/packages/katana_studio:${projDir}/packages/splicemanifold:${projDir}/packages/splicenet:${projDir}/packages/splicetrace:${projDir}/packages/scsplicenet:${projDir}/packages/scsplicetrace:\${PYTHONPATH:-}\"; fi"
}

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
    ${prismPythonPathBlock(projectDir)}
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
    ${prismPythonPathBlock(projectDir)}
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
    val  rmats_mode
    path splicenet_dir
    path bundle

    output:
    tuple val(contrast_id), path("splicetrace_${contrast_id}"), emit: results
    path "versions.yml"                                        , emit: versions

    script:
    def net_arg = splicenet_dir.name != 'NO_SPLICENET' ? "--splicenet-modules \"${splicenet_dir}/module_assignments.tsv\" --splicenet-scores \"${splicenet_dir}/module_scores.tsv\"" : ''
    def sp_val  = (species in ['human', 'mouse']) ? species : 'mouse'
    def mode_val = (rmats_mode ?: 'JCEC').toString().toUpperCase()
    def rmats_arg = rmats_dir.name != 'NO_RMATS_DIR' ? "--rmats \"${rmats_dir}\"" : ''
    """
    ${prismPythonPathBlock(projectDir)}
    fasta_ref="${fasta}"
    if [[ "\$fasta_ref" == *.gz ]]; then gzip -dc "\$fasta_ref" > genome_unpacked.fa; fasta_ref="genome_unpacked.fa"; fi
    if [[ ! -f "\${fasta_ref}.fai" ]]; then cp -L "\$fasta_ref" genome_local.fa && fasta_ref="genome_local.fa" && samtools faidx "\$fasta_ref"; fi
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
    EXPR_ARG=""
    if [[ -s "${bundle}/DEG/all.normalised_counts.tsv" ]]; then
        EXPR_ARG="--rbp-expression ${bundle}/DEG/all.normalised_counts.tsv"
    fi
    splicetrace \
        ${rmats_arg} \
        --bundle "${bundle}" \
        --rmats-mode "${mode_val}" \
        --fasta "\${fasta_ref}" \
        --gtf "\${gtf_ref}" \
        --species "${sp_val}" \
        --contrast-id "${contrast_id}" \
        ${net_arg} \
        \${EXPR_ARG} \
        --threads ${task.cpus} \
        --output-dir "splicetrace_${contrast_id}" \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process KATANA_CROSS_PARADIGM {
    tag "${study_name}"
    label 'process_low'

    input:
    val  study_name
    path bundle
    path gtf
    val  rmats_mode
    val  das_fdr
    path splicetrace_outs

    output:
    path "cross_paradigm_out", emit: results
    path "versions.yml"      , emit: versions

    script:
    def trace_list = [splicetrace_outs].flatten().findAll { it.name != 'NO_SPLICETRACE' }
    def trace_arg  = trace_list ? "--splicetrace-dirs " + trace_list.collect { "\"${it}\"" }.join(' ') : ''
    """
    ${prismPythonPathBlock(projectDir)}
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
    python3 -m katana_studio.cross_paradigm \
        --bundle "${bundle}" \
        --gtf "\${gtf_ref}" \
        --rmats-mode "${rmats_mode}" \
        --fdr ${das_fdr} \
        ${trace_arg} \
        --outdir cross_paradigm_out \
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
    val  rmats_mode
    val  das_fdr
    val  das_dpsi
    path splicemanifold_out
    path splicenet_out
    path splicetrace_outs
    path cross_paradigm_out

    output:
    path "report/REPORT.html"         , emit: html
    path "report/REPORT.zip"          , emit: archive
    path "report/report-manifest.json", emit: manifest
    path "report"                     , emit: report_dir
    path "versions.yml"               , emit: versions

    script:
    def sm_arg     = splicemanifold_out.name != 'NO_SPLICEMANIFOLD' ? "--splicemanifold-dir \"${splicemanifold_out}\"" : ''
    def net_arg    = splicenet_out.name != 'NO_SPLICENET' ? "--splicenet-dir \"${splicenet_out}\"" : ''
    def trace_list = [splicetrace_outs].flatten().findAll { it.name != 'NO_SPLICETRACE' }
    def trace_arg  = trace_list ? "--splicetrace-dirs " + trace_list.collect { "\"${it}\"" }.join(' ') : ''
    def cp_arg     = cross_paradigm_out.name != 'NO_CROSS_PARADIGM' ? "--cross-paradigm-dir \"${cross_paradigm_out}\"" : ''
    def sp_val     = (species in ['auto', 'human', 'mouse']) ? species : 'auto'
    """
    ${prismPythonPathBlock(projectDir)}
    fasta_ref="${fasta}"
    if [[ "\$fasta_ref" == *.gz ]]; then gzip -dc "\$fasta_ref" > genome_unpacked.fa; fasta_ref="genome_unpacked.fa"; fi
    if [[ ! -f "\${fasta_ref}.fai" ]]; then cp -L "\$fasta_ref" genome_local.fa && fasta_ref="genome_local.fa" && samtools faidx "\$fasta_ref"; fi
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
    katana-report bulk \
        --bundle "${bundle}" \
        --rmats-mode "${rmats_mode}" \
        --fasta "\${fasta_ref}" \
        --gtf "\${gtf_ref}" \
        --study-name "${study_name}" \
        --species "${sp_val}" \
        --fdr ${das_fdr} \
        --dpsi ${das_dpsi} \
        ${sm_arg} \
        ${net_arg} \
        ${trace_arg} \
        ${cp_arg} \
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
    path events_table
    path psi_matrix
    path support_matrix
    path cell_metadata
    path host_expression

    output:
    path "sc_manifold_out", emit: results
    path "versions.yml"   , emit: versions

    script:
    def ev_arg   = events_table.name != 'NO_EVENTS' ? "--events ${events_table}" : ''
    def sup_arg  = support_matrix.name != 'NO_SUPPORT' ? "--support ${support_matrix}" : ''
    def expr_arg = host_expression.name != 'NO_EXPRESSION' ? "--expression ${host_expression}" : ''
    """
    ${prismPythonPathBlock(projectDir)}
    splicemanifold single-cell \
        --psi "${psi_matrix}" \
        --metadata "${cell_metadata}" \
        ${ev_arg} \
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
    ${prismPythonPathBlock(projectDir)}
    export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
    scsplicenet \
        --usage "${psi_matrix}" \
        --metadata "${cell_metadata}" \
        --event-genes "${event_genes}" \
        ${sup_arg} \
        ${expr_arg} \
        --select-budget 1500 \
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
    val  group_col
    val  donor_col
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
    def don_arg  = donor_col ? "--donor-col \"${donor_col}\"" : ''
    def sp_val   = (species in ['human', 'mouse']) ? species : 'human'
    """
    ${prismPythonPathBlock(projectDir)}
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
        --group-col "${group_col}" \
        ${don_arg} \
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
    val  donor_col
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
    def don_arg = donor_col ? "--donor-col \"${donor_col}\"" : ''
    def pb_arg  = pseudobulk_by ? "--pseudobulk-by \"${pseudobulk_by}\"" : ''
    """
    katana_sc_build_splice_matrices.py \
        --gtf "${gtf}" \
        --sj-dirs ${sj_arg} \
        --metadata "${cell_metadata}" \
        --group-col "${group_col}" \
        ${don_arg} \
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
    path gtf
    path splicemanifold_dir
    path scsplicenet_dir
    path scsplicetrace_dir
    path subtypes_dirs
    val  species
    val  das_fdr
    val  das_dpsi
    path sc_bam
    path sc_bam_bai
    path sc_barcode_shares
    val  sc_bc_tag

    output:
    path "report/REPORT.html"         , emit: html
    path "report/REPORT.zip"          , emit: archive
    path "report/report-manifest.json", emit: manifest
    path "report"                     , emit: report_dir
    path "versions.yml"               , emit: versions

    script:
    def sup_arg  = support_matrix.name != 'NO_SUPPORT' ? "--support-matrix \"${support_matrix}\"" : ''
    def expr_arg = host_expression.name != 'NO_EXPRESSION' ? "--expression \"${host_expression}\"" : ''
    def sm_arg   = splicemanifold_dir.name != 'NO_SPLICEMANIFOLD' ? "--splicemanifold-dir \"${splicemanifold_dir}\"" : ''
    def sn_arg   = scsplicenet_dir.name != 'NO_SCSPLICENET' ? "--scsplicenet-dir \"${scsplicenet_dir}\"" : ''
    def st_arg   = scsplicetrace_dir.name != 'NO_SCSPLICETRACE' ? "--scsplicetrace-dir \"${scsplicetrace_dir}\"" : ''
    def sub_list = [subtypes_dirs].flatten().findAll { it.name != 'NO_SUBTYPES' }
    def sub_arg  = sub_list ? "--subtypes-dirs " + sub_list.collect { "\"${it}\"" }.join(' ') : ''
    def bam_arg  = sc_bam.name != 'NO_SC_BAM' ? "--sc-bam \"${sc_bam}\" --sc-bc-tag \"${sc_bc_tag}\"" : ''
    def bcs_arg  = (sc_bam.name != 'NO_SC_BAM' && sc_barcode_shares.name != 'NO_BARCODE_SHARES') ? "--sc-barcode-shares \"${sc_barcode_shares}\"" : ''
    def sp_val   = (species in ['human', 'mouse']) ? species : 'human'
    """
    ${prismPythonPathBlock(projectDir)}
    gtf_ref="${gtf}"
    GTF_ARG=""
    if [[ "\$gtf_ref" != "NO_GTF" && -s "\$gtf_ref" ]]; then
        if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi
        GTF_ARG="--gtf \${gtf_ref}"
    fi
    if [[ "${sc_bam}" != "NO_SC_BAM" && -s "${sc_bam}" && ! -f "${sc_bam}.bai" ]]; then
        samtools index -@ ${task.cpus} "${sc_bam}"
    fi
    katana-report single-cell \
        --events "${events_table}" \
        --cells "${cell_metadata}" \
        --psi-matrix "${psi_matrix}" \
        ${sup_arg} \
        ${expr_arg} \
        \${GTF_ARG} \
        ${bam_arg} \
        ${bcs_arg} \
        ${sm_arg} \
        ${sn_arg} \
        ${st_arg} \
        ${sub_arg} \
        --study-name "${study_name}" \
        --species "${sp_val}" \
        --fdr ${das_fdr} \
        --dpsi ${das_dpsi} \
        --outdir report \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SC_RMATS_FROM_GTF {
    tag "${gtf.name}"
    label 'process_low'

    conda 'bioconda::r-pairadise=1.0.0 bioconda::rmats=4.1.2'
    container "${workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/mulled-v2-8ea76ff0a6a4c7e5c818fd4281abf918f92eeeae:121e48ab4817ec619c157a346458efca1ccf3c0a-0' :
        'quay.io/biocontainers/mulled-v2-8ea76ff0a6a4c7e5c818fd4281abf918f92eeeae:121e48ab4817ec619c157a346458efca1ccf3c0a-0'}"

    input:
    path gtf

    output:
    path "rmats_fromgtf", emit: fromgtf_dir
    path "versions.yml" , emit: versions

    script:
    """
    gtf_in="${gtf}"
    if [[ "\$gtf_in" == *.gz ]]; then gzip -dc "\$gtf_in" > unpacked.gtf; gtf_in="unpacked.gtf"; fi

    python3 "${projectDir}/bin/katana_make_dummy_bam.py" "\$gtf_in" "dummy.bam"
    echo "\$PWD/dummy.bam" > b1.txt

    rmats.py \
        --b1 b1.txt \
        --gtf "\$gtf_in" \
        -t single \
        --libType fr-unstranded \
        --readLength 50 \
        --variable-read-length \
        --allow-clipping \
        --od rmats_fromgtf \
        --tmp rmats_tmp \
        --statoff \
        --nthread ${task.cpus}

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        rmats: \$(rmats.py --version 2>&1 | sed 's/^v//g')
    END_VERSIONS
    """
}

process SC_PREPARE_MARVEL_INPUTS {
    tag "${study_name}"
    label 'process_medium'

    input:
    val  study_name
    val  sample_prefixes
    path sj_dirs, stageAs: 'sj_in_*'
    path fromgtf_dir
    path cell_metadata
    path gtf
    path sc_bam
    path sc_bam_bai
    path barcode_shares
    val  bc_tag

    output:
    path "marvel_prep" , emit: prep_dir
    path "versions.yml", emit: versions

    script:
    def dir_list = [sj_dirs].flatten()
    def specs = []
    dir_list.eachWithIndex { d, i ->
        def pref = (sample_prefixes && sample_prefixes[i]) ? "${sample_prefixes[i]}::" : ''
        specs << "${pref}${d}"
    }
    def sj_arg  = specs.collect { "\"${it}\"" }.join(' ')
    def bam_arg = sc_bam.name != 'NO_SC_BAM' ? "--bam \"${sc_bam}\"" : ''
    def bcs_arg = barcode_shares.name != 'NO_BARCODE_SHARES' ? "--barcode-shares \"${barcode_shares}\"" : ''
    """
    python3 "${projectDir}/bin/katana_sc_prepare_marvel_inputs.py" \
        --gtf "${gtf}" \
        --fromgtf-dir "${fromgtf_dir}" \
        --sj-dirs ${sj_arg} \
        --metadata "${cell_metadata}" \
        ${bam_arg} \
        ${bcs_arg} \
        --bc-tag "${bc_tag}" \
        --threads ${task.cpus} \
        --outdir marvel_prep

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
    END_VERSIONS
    """
}

process SC_MARVEL_QUANT {
    tag "${study_name}"
    label 'process_medium'

    conda "bioconda::bioconductor-drimseq=1.18.0"
    container "${ workflow.containerEngine in ['singularity', 'apptainer'] && !task.ext.singularity_pull_docker_container ?
        'https://depot.galaxyproject.org/singularity/bioconductor-drimseq:1.18.0--r40_0' :
        'quay.io/biocontainers/bioconductor-drimseq:1.18.0--r40_0' }"

    input:
    val  study_name
    path prep_dir
    path fromgtf_dir
    path marvel_tarball
    val  group_col
    val  numerator
    val  denominator
    val  min_cells
    val  min_cell_support

    output:
    path "sc_matrices/MarvelObject.rds"  , emit: marvel_rds
    path "sc_matrices/sc_events.tsv"     , emit: events
    path "sc_matrices/sc_event_genes.tsv", emit: event_genes
    path "sc_matrices/sc_psi.tsv.gz"     , emit: psi
    path "sc_matrices/sc_support.tsv.gz" , emit: support
    path "sc_matrices/sc_cells.tsv"      , emit: metadata
    path "versions.yml"                  , emit: versions

    script:
    def tar_arg = marvel_tarball.name != 'NO_MARVEL_TARBALL' ? "--marvel-tarball \"${marvel_tarball}\"" : ''
    def num_arg = numerator ? "--numerator \"${numerator}\"" : ''
    def den_arg = denominator ? "--denominator \"${denominator}\"" : ''
    """
    Rscript "${projectDir}/bin/katana_sc_marvel_quant.R" \
        --prep-dir "${prep_dir}" \
        --fromgtf-dir "${fromgtf_dir}" \
        ${tar_arg} \
        --group-col "${group_col}" \
        ${num_arg} \
        ${den_arg} \
        --min-cells ${min_cells} \
        --coverage-threshold ${min_cell_support} \
        --threads ${task.cpus} \
        --outdir sc_matrices

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        r-base: \$(R --version | head -n1 | sed 's/R version \\([^ ]*\\).*/\\1/')
        MARVEL: "2.0.5"
    END_VERSIONS
    """
}

process SC_SUBTYPE_DOWNSTREAM {
    tag "${contrast_id}"
    label 'process_medium'

    input:
    tuple val(contrast_id), val(pop_col), val(pop_val), val(numerator), val(denominator)
    path events_table
    path psi_matrix
    path support_matrix
    path cell_metadata
    path host_expression
    path fasta
    path gtf
    val  species
    val  group_col
    val  donor_col
    val  min_cells
    val  das_fdr
    val  das_dpsi
    val  skip_network
    val  skip_splicetrace

    output:
    path "${contrast_id}"                      , emit: contrast_dir
    path "${contrast_id}/report/REPORT.html"   , emit: html
    path "${contrast_id}/report/REPORT.zip"    , emit: archive
    path "versions.yml"                        , emit: versions

    script:
    def sup_in  = support_matrix.name != 'NO_SUPPORT' ? "--support \"${support_matrix}\"" : ''
    def expr_in = host_expression.name != 'NO_EXPRESSION' ? "--expression \"${host_expression}\"" : ''
    def sp_val  = (species in ['human', 'mouse']) ? species : 'human'
    def do_net  = skip_network ? '0' : '1'
    def do_trc  = skip_splicetrace ? '0' : '1'
    def don_arg = donor_col ? "--donor-col \"${donor_col}\"" : ''
    """
    ${prismPythonPathBlock(projectDir)}
    mkdir -p "${contrast_id}"
    python3 "${projectDir}/bin/katana_sc_slice_contrast.py" \
        --events "${events_table}" \
        --psi "${psi_matrix}" \
        ${sup_in} \
        --metadata "${cell_metadata}" \
        ${expr_in} \
        --population-col "${pop_col}" \
        --population-value "${pop_val}" \
        --group-col "${group_col}" \
        --numerator "${numerator}" \
        --denominator "${denominator}" \
        --min-cells ${min_cells} \
        --outdir "${contrast_id}/sc_matrices"

    SM_DIR="${contrast_id}/sc_matrices"
    EXPR_SLICE=""
    EXPR_NET=""
    if [[ -s "\${SM_DIR}/sc_expression.tsv.gz" ]]; then
        EXPR_SLICE="--expression \${SM_DIR}/sc_expression.tsv.gz"
        EXPR_NET="--host-expr \${SM_DIR}/sc_expression.tsv.gz"
    fi

    splicemanifold single-cell \
        --psi "\${SM_DIR}/sc_psi.tsv.gz" \
        --metadata "\${SM_DIR}/sc_cells.tsv" \
        --events "\${SM_DIR}/sc_events.tsv" \
        --support "\${SM_DIR}/sc_support.tsv.gz" \
        \${EXPR_SLICE} \
        --cells-as-rows \
        --outdir "${contrast_id}/splicemanifold" \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}" || echo "[WARN] splicemanifold skipped on sparse subtype ${contrast_id}"

    NET_MOD_ARG=""
    N_CELLS=\$(awk 'NR>1{c++} END{print c+0}' "\${SM_DIR}/sc_cells.tsv")
    N_EV=\$(awk 'NR>1{c++} END{print c+0}' "\${SM_DIR}/sc_events.tsv")
    if [[ "${do_net}" == "1" && "\${N_CELLS}" -ge 20 && "\${N_EV}" -ge 15 ]]; then
        export OMP_NUM_THREADS=${task.cpus} OPENBLAS_NUM_THREADS=${task.cpus} MKL_NUM_THREADS=${task.cpus}
        if scsplicenet \
            --usage "\${SM_DIR}/sc_psi.tsv.gz" \
            --metadata "\${SM_DIR}/sc_cells.tsv" \
            --event-genes "\${SM_DIR}/sc_event_genes.tsv" \
            --support "\${SM_DIR}/sc_support.tsv.gz" \
            \${EXPR_NET} \
            --select-budget 1500 \
            --threads ${task.cpus} \
            --out "${contrast_id}/scsplicenet"; then
            if [[ -s "${contrast_id}/scsplicenet/scsplicenet_modules.tsv" ]]; then
                NET_MOD_ARG="--splicenet-modules ${contrast_id}/scsplicenet/scsplicenet_modules.tsv"
            fi
        fi
    fi

    fasta_ref="${fasta}"
    if [[ "\$fasta_ref" == *.gz ]]; then gzip -dc "\$fasta_ref" > genome_unpacked.fa; fasta_ref="genome_unpacked.fa"; fi
    if [[ ! -f "\${fasta_ref}.fai" ]]; then cp -L "\$fasta_ref" genome_local.fa && fasta_ref="genome_local.fa" && samtools faidx "\$fasta_ref"; fi
    gtf_ref="${gtf}"
    if [[ "\$gtf_ref" == *.gz ]]; then gzip -dc "\$gtf_ref" > annotation_unpacked.gtf; gtf_ref="annotation_unpacked.gtf"; fi

    if [[ "${do_trc}" == "1" && "\${N_EV}" -ge 5 ]]; then
        scsplicetrace \
            --events "\${SM_DIR}/sc_events.tsv" \
            --psi "\${SM_DIR}/sc_psi.tsv.gz" \
            --metadata "\${SM_DIR}/sc_cells.tsv" \
            --fasta "\${fasta_ref}" \
            --gtf "\${gtf_ref}" \
            --species "${sp_val}" \
            --group-col "${group_col}" \
            ${don_arg} \
            --numerator "${numerator}" \
            --denominator "${denominator}" \
            --support "\${SM_DIR}/sc_support.tsv.gz" \
            \${EXPR_SLICE} \
            \${NET_MOD_ARG} \
            --threads ${task.cpus} \
            --out "${contrast_id}/scsplicetrace" || echo "[WARN] scsplicetrace skipped on sparse subtype ${contrast_id}"
    fi

    SM_REPORT_ARG=""
    SN_REPORT_ARG=""
    ST_REPORT_ARG=""
    if [[ -d "${contrast_id}/splicemanifold" ]]; then SM_REPORT_ARG="--splicemanifold-dir ${contrast_id}/splicemanifold"; fi
    if [[ -d "${contrast_id}/scsplicenet" ]]; then SN_REPORT_ARG="--scsplicenet-dir ${contrast_id}/scsplicenet"; fi
    if [[ -d "${contrast_id}/scsplicetrace" ]]; then ST_REPORT_ARG="--scsplicetrace-dir ${contrast_id}/scsplicetrace"; fi

    katana-report single-cell \
        --events "\${SM_DIR}/sc_events.tsv" \
        --cells "\${SM_DIR}/sc_cells.tsv" \
        --psi-matrix "\${SM_DIR}/sc_psi.tsv.gz" \
        --support-matrix "\${SM_DIR}/sc_support.tsv.gz" \
        --gtf "\${gtf_ref}" \
        \${EXPR_SLICE} \
        \${SM_REPORT_ARG} \
        \${SN_REPORT_ARG} \
        \${ST_REPORT_ARG} \
        --study-name "${contrast_id}" \
        --species "${sp_val}" \
        --fdr ${das_fdr} \
        --dpsi ${das_dpsi} \
        --outdir "${contrast_id}/report" \
        --emit-versions-yml versions.yml \
        --process-name "${task.process}"
    """
}

process SC_EXTRACT_MULTIPLEXED_BAM {
    tag "${study_name}"
    label 'process_high'

    input:
    val  study_name
    path sc_bam
    path sc_bam_bai
    path cell_metadata
    path barcode_shares
    path umi_matrix
    path gene_names
    val  bc_tag

    output:
    path "bam_extracted/Solo.out/SJ/raw"     , emit: sj_raw
    path "bam_extracted/sc_expression.tsv.gz", optional: true, emit: expr_gz
    path "versions.yml"                      , emit: versions

    script:
    def bcs_arg = barcode_shares.name != 'NO_BARCODE_SHARES' ? "--barcode-shares \"${barcode_shares}\"" : ''
    def umi_arg = umi_matrix.name != 'NO_UMI_MATRIX' ? "--umi-matrix \"${umi_matrix}\"" : ''
    def gn_arg  = gene_names.name != 'NO_GENE_NAMES' ? "--gene-names \"${gene_names}\"" : ''
    """
    if [[ ! -f "${sc_bam}.bai" ]]; then samtools index -@ ${task.cpus} "${sc_bam}"; fi
    python3 "${projectDir}/bin/katana_sc_extract_multiplexed_bam.py" \
        --bam "${sc_bam}" \
        --metadata "${cell_metadata}" \
        ${bcs_arg} \
        ${umi_arg} \
        ${gn_arg} \
        --bc-tag "${bc_tag}" \
        --threads ${task.cpus} \
        --outdir bam_extracted

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
        samtools: \$(samtools --version | head -n1 | sed 's/samtools //')
    END_VERSIONS
    """
}

process SC_ENUMERATE_TESTABLE_SUBTYPES {
    tag "${study_name}"
    label 'process_single'

    input:
    val  study_name
    path cell_metadata
    val  celltype_cols
    val  group_col
    val  donor_col
    val  numerator
    val  denominator
    val  condition_pairs
    val  min_donors_per_arm

    output:
    path "subtype_audit/testable_subtype_contrasts.tsv", emit: contrasts_tsv
    path "subtype_audit/sc_subtype_donor_audit.tsv"    , emit: audit_tsv
    path "versions.yml"                                , emit: versions

    script:
    def num_arg  = numerator ? "--numerator \"${numerator}\"" : ''
    def den_arg  = denominator ? "--denominator \"${denominator}\"" : ''
    def don_arg  = donor_col ? "--donor-col \"${donor_col}\"" : ''
    def pair_arg = condition_pairs ? "--condition-pairs \"${condition_pairs}\"" : ''
    """
    python3 "${projectDir}/bin/katana_sc_enumerate_testable_subtypes.py" \
        --metadata "${cell_metadata}" \
        --celltype-cols "${celltype_cols}" \
        --group-col "${group_col}" \
        ${don_arg} \
        ${num_arg} \
        ${den_arg} \
        ${pair_arg} \
        --min-donors-per-arm ${min_donors_per_arm} \
        --outdir subtype_audit

    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        python: \$(python3 --version | sed 's/Python //')
    END_VERSIONS
    """
}
