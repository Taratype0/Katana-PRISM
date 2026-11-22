// Independent selected-target Nextflow DAG (`-entry TARGETS` or `--mode targets`).
// Reuses verified Studio target requests (`request.zip` + `targets.tsv`) with zero FASTQ re-alignment or statistical refit.
process TARGET_REQUEST_INSPECT {
    tag 'inspect target request'
    cpus 1
    container { params.katana_container ?: null }

    input:
    path request

    output:
    path 'targets.json', emit: tasks_json
    path 'targets.tsv',  emit: tasks_tsv

    script:
    """
    python -s -m katana_next.targets tasks --request '${request}' --format json > targets.json
    python -s -m katana_next.targets tasks --request '${request}' --format tsv  > targets.tsv
    """
}

process TARGET_DETAIL {
    tag { target_id }
    cpus 1
    container { params.katana_container ?: null }
    publishDir "${params.outdir}/targets", mode: 'copy'

    input:
    val target_id
    path request
    val resource_ids
    path resources, stageAs: 'resources/resource??/*'
    path bams_dir
    path sc_bam
    path sc_barcode_shares
    path gtf

    output:
    path "part_${target_id}", emit: target_part

    script:
    def staged = resources instanceof List ? resources : [resources]
    def hasBindings = resource_ids && resource_ids.size() > 0 && staged.size() == resource_ids.size()
    def bindings = hasBindings ? groovy.json.JsonOutput.toJson([schema:'katana.target_bindings/v1', resources:
        [resource_ids, staged.collect { it.toString() }].transpose().collectEntries { [(it[0]): it[1]] }]) : ''
    def encoded = bindings ? bindings.bytes.encodeBase64().toString() : ''
    def bamsFlag = (bams_dir.name != 'NO_BAMS_DIR') ? "--bams-dir '${bams_dir}'" : ''
    def scBamFlag = (sc_bam.name != 'NO_SC_BAM') ? "--sc-bam '${sc_bam}' --sc-bc-tag '${params.sc_bc_tag}'" : ''
    def scSharesFlag = (sc_barcode_shares.name != 'NO_SC_SHARES') ? "--sc-barcode-shares '${sc_barcode_shares}'" : ''
    def gtfFlag = (gtf.name != 'NO_GTF') ? "--gtf '${gtf}'" : ''
    def bindFlag = encoded ? "--bindings bindings.json" : ''
    """
    if [ -n "${encoded}" ]; then
    python -s - <<'PY'
import base64
from pathlib import Path
Path('bindings.json').write_bytes(base64.b64decode('${encoded}'))
PY
    fi
    set +e
    python -s -m katana_next.targets run \
      --request '${request}' \
      ${bindFlag} \
      ${bamsFlag} \
      ${scBamFlag} \
      ${scSharesFlag} \
      ${gtfFlag} \
      --outdir 'part_${target_id}' \
      --target-id '${target_id}' \
      --workers ${task.cpus} > execution.json
    code=\$?
    set -e
    if [ "\$code" -ne 0 ] && [ "\$code" -ne 2 ]; then cat execution.json >&2; exit "\$code"; fi
    cp execution.json 'part_${target_id}/pipeline-execution.json'
    """
}

process TARGET_AGGREGATE {
    tag 'aggregate selected-target batch'
    cpus 1
    container { params.katana_container ?: null }
    publishDir "${params.outdir}/targets_report", mode: 'copy'

    input:
    path request
    path 'parts/*'

    output:
    path 'targets_report', emit: report_dir

    script:
    """
    part_flags=""
    for p in parts/*; do
      if [ -d "\$p" ]; then
        part_flags="\${part_flags} --part \${p}"
      fi
    done
    python -s -m katana_next.targets aggregate --request '${request}' \${part_flags} --outdir targets_report > aggregate.json
    """
}

workflow KATANA_TARGETS {
    take:
    request
    resource_ids
    resources
    bams_dir
    sc_bam
    sc_barcode_shares
    gtf

    main:
    TARGET_REQUEST_INSPECT(request)
    ids = TARGET_REQUEST_INSPECT.out.tasks_tsv.splitCsv(header:true, sep:'\t').map { row ->
        if (!(row.target_id ==~ /(?:gene|event)-[a-f0-9]+/)) error("Invalid exact target ID: ${row.target_id}")
        row.target_id
    }
    TARGET_DETAIL(ids, request, resource_ids, resources, bams_dir, sc_bam, sc_barcode_shares, gtf)
    TARGET_AGGREGATE(request, TARGET_DETAIL.out.target_part.collect())

    emit:
    parts  = TARGET_DETAIL.out.target_part
    report = TARGET_AGGREGATE.out.report_dir
}
