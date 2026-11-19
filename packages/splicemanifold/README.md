# SpliceManifold (`1.0.1`)

**SpliceManifold** is an independent Python package for **PS-DAS** (*Precision-Supported Differential Alternative Splicing*) qualification and depth-stabilized splicing manifold geometry across bulk RNA-seq (`rMATS`) and full-length single-cell RNA-seq (`Smart-seq3`).

## Key Capabilities
1. **PS-DAS Qualification (`katana-ps-das`)**: Isoform-length-corrected effective junction support, replicate quantile criterion (`>=75%`), pooled Jeffreys posterior `SE(dPSI)` gate, and biological AS-NMD asymmetry rescue without altering native `PValue`/`FDR`.
2. **Beyond-Jutils Splicing Manifold**: Empirical-Bayes binomial prior shrinkage (`kappa = 10`), `ExcessBioVar` noise subtraction, Ward Z-score clustermap with non-colliding group/subtype legends, and permutation PERMANOVA PCA.
3. **Fisher-Precision Kernel Event-PSI UMAP**: Pairwise co-measured Fisher information kernel avoiding naive mean-imputation dropout artifacts, plus Procrustes expression alignment and cross-modality divergence `Delta_kNN`.

## Quick Start
```bash
pip install ./packages/splicemanifold

# Standalone Bulk rMATS run
splicemanifold bulk --rmats-dir path/to/rmats_post --samples samples.csv --outdir manifold_out

# Standalone Single-Cell Smart-seq3 run
splicemanifold single-cell --psi cell_psi.tsv.gz --support cell_depth.tsv.gz --metadata cell_meta.tsv --expression log_expr.tsv.gz --cells-as-rows --outdir sc_manifold_out
```
