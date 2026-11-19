# Katana-PRISM (`v1.0.1`) — Public Benchmark Reproducibility Suite

This directory provides the standalone benchmark driver (`reproduce_nar_benchmarks.py`) and verified summary tables (`tables/*.tsv`) accompanying the *Nucleic Acids Research* manuscript:

1. **Foreground-Block & Continuous-Score `SpliceTrace` Calibration (`tables/fig2c_encode_deep5_splicetrace_ablation.tsv`)** — Separates exact-target `FWER <= 0.05` recovery (`4/5` on high-block perturbations with `608–2,731` matched events; `0/147` false positives across `NegA`, `SUPV3L1`, and `PNPT1` negative controls) from `PTBP1/PTBP2` paralog PWM ties (`--rbp-expression` disambiguation).
2. **`HepG2` `eCLIP` Physical Binding Validation (`tables/fig2e_hepg2_eclip_locus_and_event_validation.tsv`)** — Exact `2x2` conditional Fisher odds ratios and Haldane-Anscombe 95% confidence intervals across both whole-event regions and `+-35 nt` cis-motif windows (`ENCFF154DRN` `RBFOX2`, `ENCFF170SCU` `PTBP1`).
3. **`12`-Channel (`distal_intron_nt = 500`) `SpliceNet <-> SpliceTrace` Regulatory Bridge (`tables/fig4a_encode_module_rbp_bridge_distal500.tsv`, `fig4c_bridged_vs_orphan_coding_flows.tsv`, `fig4d_deconvolution_vs_competitors.tsv`)** — Explicitly separates `286` `module_P_FWER <= 0.05` bridged coding pathways (`M1`, `M3`, `M9`, `M12`) from `325` unattributed (`FWER > 0.05`) orphan-module coding switches (`precision = 40.0%`, `F1 = 0.444`).
4. **Single-Cell `Smart-seq3` Benchmarks (`GSE270928`, `tables/fig5a_...`, `fig5b_...`, `fig5c_...`, `figS4a_...`)** — Dual-denominator `Jutils` retention (`0/1,000` shared HVEs; `327/66,391` candidate transcripts), donor-stratified `Stouffer BH q` across `167` protein-altering switches, `SLAMF1` (`Q13291-4` replacing `Q13291-1`) per-cell `83-nt` cassette-exon `PSI` (`76/192` cells), and within-`donor x state` cell-pooling sensitivity (`5–48` cells).

## Quick Verification

```bash
python benchmarks/nar_reproducibility/reproduce_nar_benchmarks.py --verify-bundled
```

## Full Re-Execution from Raw Public ENCODE rMATS & Reference FASTA

```bash
python benchmarks/nar_reproducibility/reproduce_nar_benchmarks.py \
  --splicetools-dir /path/to/public_encode_rmats \
  --hg38-fasta /path/to/hg38.fa \
  --out-dir benchmarks/nar_reproducibility/tables
```
