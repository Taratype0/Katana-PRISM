"""Prepare a portable native Nextflow task, separate from the original run."""
from pathlib import Path
import json
import shutil
import zipfile
from .bundle import open_bundle, write_targets_tsv


def prepare_task(request, output_dir, resources=None):
    output = Path(output_dir).resolve()
    if output.exists():
        raise ValueError('Use a new Nextflow task directory')
    with open_bundle(request) as bundle:
        request_id = bundle.request['request_id']
        requirements = bundle.request['resources']
        resource_ids = {r['resource_id'] for r in requirements}
        if set(resources or {}) - resource_ids:
            raise ValueError('Unknown resource binding')
        output.mkdir(parents=True)
        shutil.copytree(bundle.root, output / 'request')
        write_targets_tsv(bundle.request, output / 'targets.tsv')
    with zipfile.ZipFile(output / 'request.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((output / 'request').rglob('*')):
            if path.is_file():
                archive.write(path, path.relative_to(output / 'request').as_posix())
    # Resolve the very same module used by main.nf -entry TARGETS (repo checkout or packaged fallback).
    repo_root = Path(__file__).resolve().parents[4]
    module = next((p for p in (
        repo_root / 'workflows/katana_targets.nf',
        Path(__file__).resolve().parent / '_workflows/katana_targets.nf',
        Path(__file__).resolve().parents[2] / 'pipeline/workflows/katana_targets.nf',
        Path(__file__).resolve().parents[1] / 'flow/_pipeline/workflows/katana_targets.nf',
    ) if p.is_file()), None)
    if module is None:
        raise FileNotFoundError('Could not locate katana_targets.nf workflow definition')
    shutil.copy2(module, output / 'katana_targets.nf')
    main = next((p for p in (
        repo_root / 'main.nf',
        Path(__file__).resolve().parents[2] / 'pipeline/main.nf',
        Path(__file__).resolve().parents[1] / 'flow/_pipeline/main.nf',
    ) if p.is_file()), None)
    if main is not None and '// A new follow-up run from a portable request exported by the browser.' in main.read_text(encoding='utf-8'):
        raw_entry = main.read_text(encoding='utf-8').split('// A new follow-up run from a portable request exported by the browser.')[1]
        raw_entry = raw_entry.replace("'./workflows/katana_targets'", "'./katana_targets'")
        entry = "nextflow.enable.dsl = 2\ninclude { KATANA_TARGETS } from './katana_targets'\n" + raw_entry
    else:
        entry = (
            "nextflow.enable.dsl = 2\n"
            "include { KATANA_TARGETS } from './katana_targets'\n"
            "workflow TARGETS {\n"
            "    if (!params.target_request) error('TARGETS requires --target_request')\n"
            "    KATANA_TARGETS(Channel.value(file(params.target_request, checkIfExists:true)), "
            "params.target_bindings ? file(params.target_bindings, checkIfExists:true) : file('NO_BINDINGS'), "
            "params.bams_dir ?: '', params.sc_bam ?: '', params.sc_barcode_shares ?: '', params.gtf ?: '')\n"
            "}\n"
        )
    (output / 'main.nf').write_text(entry, encoding='utf-8')
    (output / 'nextflow.config').write_text(
        "nextflow.enable.dsl = 2\n"
        "params.katana_container = null\n"
        "params.target_request = 'request.zip'\n"
        "params.target_bindings = null\n"
        "params.bams_dir = null\n"
        "params.sc_bam = null\n"
        "params.sc_barcode_shares = null\n"
        "params.sc_bc_tag = 'BC'\n"
        "params.gtf = null\n"
        "params.outdir = 'results'\n"
        "trace.enabled = true\n"
        "trace.file = 'trace.txt'\n",
        encoding='utf-8',
    )
    template = {'schema':'katana.target_bindings/v1', 'resources':{r['resource_id']: '' for r in requirements}}
    (output / 'bindings.template.json').write_text(json.dumps(template, indent=2), encoding='utf-8')
    (output / 'resource-requirements.json').write_text(json.dumps(requirements, indent=2), encoding='utf-8')
    # Execution-machine paths stay outside the portable task archive.
    if resources:
        (output / 'bindings.local.json').write_text(json.dumps({'schema':template['schema'],'resources':resources}, indent=2), encoding='utf-8')
    command = 'nextflow run main.nf -entry TARGETS --target_request request.zip --bams_dir /path/to/bams --gtf /path/to/annotation.gtf --outdir results -work-dir work'
    (output / 'README.txt').write_text(
        'Katana selected-target follow-up task\n\n'
        'Included pipeline-readable target table: targets.tsv\n'
        'Option A (explicit bindings JSON): fill bindings.template.json and save as bindings.json, then run:\n'
        '  nextflow run main.nf -entry TARGETS --target_request request.zip --target_bindings bindings.json --outdir results -work-dir work\n\n'
        'Option B (Bulk server auto-binding via --bams_dir + --gtf):\n  ' + command + '\n\n'
        'Option C (Single-Cell multiplexed BAM via --sc_bam + --sc_barcode_shares + --gtf):\n'
        '  nextflow run main.nf -entry TARGETS --target_request request.zip --sc_bam /path/to/sc.bam --sc_barcode_shares /path/to/shares.tsv --gtf /path/to/annotation.gtf --outdir results -work-dir work\n\n'
        'Use -c site.config (e.g. -profile apptainer --katana_container /path/to/katana-prism-1.0.1.sif) and -resume to resume. '
        'No upstream alignment or statistical refit is launched. Reopen results/targets_report/REPORT.zip (or target_report/result-manifest.json) in Katana-Studio.\n',
        encoding='utf-8',
    )
    archive_path = output / 'NEXTFLOW_TASK.zip'
    with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name in ('request.zip', 'targets.tsv', 'main.nf', 'katana_targets.nf', 'nextflow.config', 'bindings.template.json', 'resource-requirements.json', 'README.txt'):
            archive.write(output / name, name)
    result = {'status':'prepared', 'execution':'native_nextflow', 'request_id':request_id,
              'directory':str(output), 'archive':str(archive_path),
              'targets_tsv':str(output / 'targets.tsv'), 'command':command,
              'submitted':False}
    (output / 'task.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result
