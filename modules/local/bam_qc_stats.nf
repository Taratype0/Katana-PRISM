// Lightweight per-sample BAM alignment & chromosome summary metrics (ensures BAM-entry runs still populate MultiQC)
process SAMTOOLS_BAM_STATS {
    tag "${meta.id}"
    label 'process_low'

    conda 'bioconda::samtools=1.17'
    container "${workflow.containerEngine in ['singularity', 'apptainer'] ? 'https://depot.galaxyproject.org/singularity/samtools:1.17--h00cdaf9_0' : 'quay.io/biocontainers/samtools:1.17--h00cdaf9_0'}"

    input:
    tuple val(meta), path(bam), path(bai)

    output:
    tuple val(meta), path("${meta.id}.flagstat"), emit: flagstat
    tuple val(meta), path("${meta.id}.idxstats"), emit: idxstats
    path 'versions.yml', emit: versions

    script:
    """
    samtools flagstat -@ ${task.cpus} "$bam" > "${meta.id}.flagstat"
    samtools idxstats "$bam" > "${meta.id}.idxstats"
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        samtools: \$(samtools --version | head -n 1 | sed 's/^samtools //')
    END_VERSIONS
    """
}

// Fast parallel upstream 5'-3' transcript coverage & RNA-seq QC via Qualimap (replaces slow 12-h serial RSeQC BAM scans)
process QUALIMAP_GENEBODY {
    tag "${meta.id}"
    label 'process_low'

    conda 'bioconda::qualimap=2.3'
    container "${workflow.containerEngine in ['singularity', 'apptainer'] ?
        'https://depot.galaxyproject.org/singularity/qualimap:2.3--hdfd78af_0' :
        'quay.io/biocontainers/qualimap:2.3--hdfd78af_0'}"

    input:
    tuple val(meta), path(bam), path(bai)
    path gtf

    output:
    tuple val(meta), path("${meta.id}"), emit: results
    path 'versions.yml', emit: versions

    script:
    def strand_map = [forward: 'strand-specific-forward', reverse: 'strand-specific-reverse', unstranded: 'non-strand-specific']
    def protocol = strand_map[meta.strandedness] ?: 'non-strand-specific'
    // Reserve 80% of allocated task memory for JVM heap (prevents container OOM)
    def java_mem_mb = Math.max(2048, (int)(task.memory.toMega() * 0.80))
    def paired_arg = meta.single_end ? '' : '-pe'
    """
    unset DISPLAY
    mkdir -p tmp_java
    trap 'rm -rf tmp_java' EXIT
    export _JAVA_OPTIONS="-Djava.awt.headless=true -Djava.io.tmpdir=\$PWD/tmp_java"
    qualimap --java-mem-size=${java_mem_mb}M rnaseq \\
        -bam "$bam" \\
        -gtf "$gtf" \\
        -p $protocol \\
        $paired_arg \\
        -outdir "${meta.id}"
    rm -rf tmp_java
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        qualimap: \$(qualimap --version 2>&1 | grep -i 'QualiMap v' | sed 's/.*QualiMap v.//; s/ .*//' || echo '2.3')
    END_VERSIONS
    """
}
