// Unified MultiQC aggregation across FastQC, TrimGalore, STAR, Samtools, RSeQC, Qualimap 5'-3' coverage, Salmon & rMATS
process KATANA_MULTIQC {
    label 'process_single'
    cpus 1
    memory { 6.GB * task.attempt }
    time { 2.h * task.attempt }
    conda 'bioconda::multiqc=1.35'
    container "${workflow.containerEngine in ['singularity', 'apptainer']
        ? 'https://community-cr-prod.seqera.io/docker/registry/v2/blobs/sha256/c8/c8e346f4f6080eadf1253505e6ff09ef004454fc18e8d672006fd7b222cc412e/data'
        : 'community.wave.seqera.io/library/multiqc:1.35--c17fb751507e9dfc'}"

    input:
    path qc_files, stageAs: 'inputs/??????/*'

    output:
    path 'multiqc_report.html', emit: report
    path 'multiqc_report_data/multiqc_data.json', emit: data
    path 'multiqc_report_data/*', emit: data_files
    path 'versions.yml', emit: versions

    script:
    """
    multiqc inputs \\
        --filename multiqc_report.html \\
        --outdir . \\
        --data-format tsv \\
        --force
    # Ensure JSON data summary is also written alongside flat TSV curve tables (mqc_qualimap_*, mqc_rseqc_*)
    if [ ! -s multiqc_report_data/multiqc_data.json ]; then
        multiqc inputs --filename multiqc_report.html --outdir . --data-format json --force
    fi
    test -s multiqc_report.html
    test -s multiqc_report_data/multiqc_data.json
    cat <<-END_VERSIONS > versions.yml
    "${task.process}":
        multiqc: \$(multiqc --version | sed 's/.*version //')
    END_VERSIONS
    """
}
