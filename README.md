# Katana-PRISM (`1.0.1`)

**Katana-PRISM** (*Precision Regulatory Isoform & Splicing Mechanisms*) is a decoupled, publication-grade computational ecosystem for **alternative splicing qualification, depth-stabilized splicing manifold geometry, co-splicing network topology, cis-regulatory RBP splicing maps, and isoform protein/NMD consequence modeling** across bulk RNA-seq and full-length single-cell RNA-seq (`Smart-seq3`).

---

## Architecture: Two Independent Yet Convergent Entry Paths

Katana-PRISM cleanly separates **cluster/container workflow scheduling** from **modular statistical & mechanistic Python engines**:

1. **Full End-to-End Execution (`FASTQ` / `BAM` $\rightarrow$ `Katana Report`) via Nextflow (`main.nf`)**:
   - Built on `nf-core` DSL2 conventions with SLURM / Docker / Singularity / Apptainer / Conda isolation and dynamic per-task resource scaling.
   - Performs single-pass coordinate-sorted `STAR` alignment (or header-inspected zero-resort `BAM` ingestion), parallel `Qualimap` $5'\rightarrow 3'$ gene-body coverage, unified `MultiQC`, `Salmon` + `DESeq2`, `DEXSeq` / `DRIMSeq` / `stageR`, `SUPPA2`, and parallelized `rMATS-turbo`, followed by scatter-gather execution of Katana's downstream analytical modules.
2. **Standalone Python Packages (`packages/*`) for External Caller Outputs**:
   - Researchers who have already generated `rMATS` tables or `Smart-seq3` splicing matrices in another pipeline can `pip install` **only** the lightweight Python module(s) they need—without installing Nextflow or upstream aligners.

---

## Official Module & Package Catalog (`1.0.1`)

| Component | Package / Directory | CLI Command | Scope & Methodological Innovation |
| :--- | :--- | :--- | :--- |
| **`Katana`** | Root (`main.nf`, `workflows/`, `modules/`) | `nextflow run main.nf` | `nf-core` DSL2 end-to-end bulk (`FASTQ`/`BAM`) & single-cell (`Smart-seq3`) orchestrator |
| **`SpliceManifold`** | `packages/splicemanifold` | `splicemanifold` | **`PS-DAS`** (*Posterior-Stability / Precision-Supported DAS*, `katana-ps-das`) qualification, Empirical-Bayes shrunk PSI Ward Clustermap, PERMANOVA PCA, and precision-weighted **Isoform/Event-PSI UMAP** ($\Delta_{\text{kNN}}$ divergence) |
| **`SpliceNet`** | `packages/splicenet` | `splicenet` | Bulk RNA-seq `ExcessBioVar` binomial-debiased signed Topological Overlap (TOM) co-splicing network & hub discovery |
| **`SpliceTrace`** | `packages/splicetrace` | `splicetrace` | Bulk RNA-seq 170-RBP (621 PWM) 13-channel (mature-RNA path + 12 spatial windows) FWL splicing map, gene-block FWER inference, module bridge, `50-nt EJC` NMD & UniProt domain consequences |
| **`scSpliceNet`** | `packages/scsplicenet` | `scsplicenet` | Independent `Smart-seq2/3` single-cell donor-aware support-precision co-splicing network & module discovery |
| **`scSpliceTrace`** | `packages/scsplicetrace` | `scsplicetrace` | Independent `Smart-seq2/3` single-cell donor-stratified cis-regulatory RBP program & isoform protein consequence engine |
| **`Katana Studio`** | `packages/katana_studio` | `katana-report`, `katana-studio` | SHA256-verified self-contained `REPORT.html` / `REPORT.zip` synthesis and interactive local web workbench |

---

## Quick Start

### Path A: End-to-End from `FASTQ` or `BAM` (Nextflow DSL2)

```bash
nextflow run main.nf \
  --mode complete \
  --input samplesheet.csv \
  --contrasts contrasts.csv \
  --fasta GRCm39.primary_assembly.genome.fa \
  --gtf gencode.vM33.annotation.gtf \
  --species mouse \
  --reference_build GRCm39 \
  --outdir results \
  -profile docker
```

### Path B: Standalone Python Packages (Direct from `rMATS` or `Smart-seq3`)

Install any module independently:

```bash
pip install ./packages/splicemanifold
pip install ./packages/splicenet
pip install ./packages/splicetrace
pip install ./packages/scsplicenet
pip install ./packages/scsplicetrace
pip install ./packages/katana_studio
```

Run individual bulk discovery modules directly on existing `rMATS` outputs:

```bash
# 1. PS-DAS event qualification + Empirical-Bayes PSI Clustermap, PCA & Event-PSI UMAP
splicemanifold bulk --rmats-dir path/to/rmats_post --samples samples.csv --outdir out_manifold

# 2. Excess-BioVar Signed TOM co-splicing network & hub discovery
splicenet --rmats-dir path/to/rmats_post --samples samples.csv --output-dir out_splicenet

# 3. 170-RBP 13-channel (mature-RNA path + 12 spatial windows) splicing map + 50-nt EJC NMD & UniProt domain consequences
splicetrace --rmats path/to/rmats_post --fasta genome.fa --gtf annotation.gtf \
  --species mouse --splicenet-modules out_splicenet/module_assignments.tsv --output-dir out_splicetrace

# 4. Synthesize self-contained REPORT.html / REPORT.zip & open interactive Katana Studio
katana-report bulk --rmats path/to/rmats_post --samples samples.csv --fasta genome.fa --gtf annotation.gtf --outdir out_report
katana-studio out_report/REPORT.zip --port 18775
```

Run independent single-cell (`Smart-seq2/3`) discovery modules:

```bash
# Single-cell Precision Isoform/Event-PSI UMAP & expression modality divergence
splicemanifold single-cell --psi cell_psi.tsv.gz --support cell_depth.tsv.gz --metadata cells.tsv --expression log_expr.tsv.gz --cells-as-rows --outdir sc_manifold

# Single-cell support-precision co-splicing network
scsplicenet --usage cell_psi.tsv.gz --support cell_depth.tsv.gz --metadata cells.tsv --event-genes event_genes.tsv --out sc_net

# Single-cell donor-stratified RBP discovery & isoform protein consequences
scsplicetrace --events sc_events.tsv --psi cell_psi.tsv.gz --metadata cells.tsv --fasta genome.fa --gtf annotation.gtf \
  --numerator Activated --denominator Resting --splicenet-modules sc_net/scsplicenet_modules.tsv --out sc_trace
```

---

## Authors & Citation

- **Dhruba Tara Maharjan** — Lead Developer & Co-Corresponding Author (`@Taratype0`, Department of Medical Biochemistry and Biophysics, Karolinska Institutet, Stockholm, Sweden; `dhruba.tara.maharjan@ki.se`)
- **Onur Dagliyan** — Principal Investigator & Co-Corresponding Author (Department of Medical Biochemistry and Biophysics, Karolinska Institutet, Stockholm, Sweden; `onur.dagliyan@ki.se`)

## License

Released under the [MIT License](LICENSE) (`Copyright (c) 2026 Dhruba Tara Maharjan`).
