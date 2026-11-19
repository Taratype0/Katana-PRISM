#!/bin/bash -l
#SBATCH --job-name=prism_sc_e2e
#SBATCH --account=naiss2026-4-301
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=prism_sc_controller-%j.out
#SBATCH --error=prism_sc_controller-%j.err

set -euo pipefail

# ==============================================================================
# 1. EDIT ONLY THESE STUDY VARIABLES BEFORE RUNNING `sbatch` ON DARDEL
# ==============================================================================
export SLURM_ACCOUNT="naiss2026-4-301"
export ROOT="/cfs/klemming/projects/supr/naiss2025-23-363"
export PRISM="${ROOT}/Katana-PRISM"
export RUN_DIR="${PRISM}/runs/my_singlecell_study"

# Reference genome & GENCODE/Ensembl GTF (supports .fa/.fa.gz and .gtf/.gtf.gz)
export FASTA="/path/to/genome.fa"
export GTF="/path/to/annotation.gtf"
export SPECIES="human"                 # "human" or "mouse"
export STUDY_NAME="SingleCell Splicing Study"

# Optional: existing STAR index directory (leave empty "" to build automatically)
export STAR_INDEX=""

# Pre-annotated cell metadata & gene expression matrix:
#   - SC_METADATA: TSV/CSV with columns `cell_id`, `cell_type`, `condition`, `donor_id`
#   - SC_EXPRESSION: TSV/TSV.GZ (cells x genes or genes x cells log1p/TPM/counts)
export SC_METADATA="${RUN_DIR}/inputs/cell_metadata.tsv"
export SC_EXPRESSION="${RUN_DIR}/inputs/expression.tsv.gz"

# Contrast to test in scSpliceTrace (can be two `condition` labels OR two `cell_type` labels!)
export SC_GROUP_COL="condition"        # "condition" or "cell_type"
export SC_NUMERATOR="Case"             # e.g. "Activated" or "CellTypeA"
export SC_DENOMINATOR="Control"        # e.g. "Resting"   or "CellTypeB"

# Choose ONE of the two FASTQ input modes below:
#   Mode A (Full-length plate Smart-seq2 / Smart-seq3):
#     TSV/CSV with header: `cell_id,fastq_1,fastq_2`
export SC_FASTQ_MANIFEST="${RUN_DIR}/inputs/sc_fastq_manifest.tsv"

#   Mode B (10x Genomics / droplet CB+UMI barcoded FASTQs — leave empty if using Mode A):
#     TSV/CSV with header: `sample_id,fastq_r1,fastq_r2`
#     Note: for 10x 3'/5' libraries, set SC_PSEUDOBULK_BY="donor_id,cell_type"
export SC_10X_FASTQ_SHEET=""
export SC_CB_WHITELIST=""
export SC_PSEUDOBULK_BY=""
# ==============================================================================

module load PDC java apptainer nextflow/25.10.4

export NXF_HOME="${PRISM}/.cache/nextflow_sc"
export NXF_SINGULARITY_CACHEDIR="${ROOT}/.nextflow/singularity"
export SINGULARITY_CACHEDIR="${ROOT}/cache/katana/singularity"
export APPTAINER_CACHEDIR="${ROOT}/cache/katana/apptainer"
export SINGULARITYENV_NUMBA_CACHE_DIR="${PRISM}/.cache/numba"
export APPTAINERENV_NUMBA_CACHE_DIR="${PRISM}/.cache/numba"
export NXF_OPTS="-Xms256m -Xmx2g"

mkdir -p "${RUN_DIR}/results/pipeline_info" "${RUN_DIR}/work" "${NXF_HOME}" "${PRISM}/.cache/numba" "${PRISM}/containers"

# Ensure Katana-PRISM 1.0.1 SIF image is cached on Klemming
SIF_PATH="${PRISM}/containers/katana-prism-1.0.1.sif"
if [[ ! -s "${SIF_PATH}" ]]; then
    apptainer pull "${SIF_PATH}" docker://ghcr.io/taratype0/katana-prism:1.0.1
fi

EXTRA_ARGS=()
if [[ -n "${STAR_INDEX}" ]]; then
    EXTRA_ARGS+=(--star_index "${STAR_INDEX}")
fi
if [[ -n "${SC_FASTQ_MANIFEST}" ]]; then
    EXTRA_ARGS+=(--sc_fastq_manifest "${SC_FASTQ_MANIFEST}")
elif [[ -n "${SC_10X_FASTQ_SHEET}" ]]; then
    EXTRA_ARGS+=(--sc_10x_fastq_sheet "${SC_10X_FASTQ_SHEET}")
    if [[ -n "${SC_CB_WHITELIST}" ]]; then
        EXTRA_ARGS+=(--sc_cb_whitelist "${SC_CB_WHITELIST}")
    fi
fi
if [[ -n "${SC_PSEUDOBULK_BY}" ]]; then
    EXTRA_ARGS+=(--sc_pseudobulk_by "${SC_PSEUDOBULK_BY}")
fi

cd "${RUN_DIR}"

exec nextflow run "${PRISM}/source/main.nf" \
  -profile singularity,pdc_dardel \
  -c "${PRISM}/config/dardel_prism.config" \
  --project "${SLURM_ACCOUNT}" \
  --mode single-cell \
  --study_name "${STUDY_NAME}" \
  --species "${SPECIES}" \
  --fasta "${FASTA}" \
  --gtf "${GTF}" \
  --sc_metadata "${SC_METADATA}" \
  --sc_expression "${SC_EXPRESSION}" \
  --sc_group_col "${SC_GROUP_COL}" \
  --sc_numerator "${SC_NUMERATOR}" \
  --sc_denominator "${SC_DENOMINATOR}" \
  --katana_container "${SIF_PATH}" \
  "${EXTRA_ARGS[@]}" \
  --outdir "${RUN_DIR}/results" \
  -work-dir "${RUN_DIR}/work" \
  -with-trace "${RUN_DIR}/results/pipeline_info/execution_trace.txt" \
  -with-report "${RUN_DIR}/results/pipeline_info/execution_report.html" \
  -resume
