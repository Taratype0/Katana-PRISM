# Katana-PRISM (`v1.0.1`) — Public Benchmark Reproducibility Suite

This directory provides the standalone benchmark driver (`reproduce_nar_benchmarks.py`) and verified summary tables (`tables/*.tsv`) accompanying the *Nucleic Acids Research* manuscript:

1. **Foreground-Block & Single-Variable `FWL` `SpliceTrace` Calibration (`tables/fig2c_splicetrace_exact_and_ablations.tsv`)** — Reports per-protein exact `P_FWER` and ranks across `169` human RBPs × `13` channels for discrete matching (`999` gene-block sign flips), continuous `FWL ON + maxT` (`1,999` between-gene response permutations), single-variable `No-FWL` (`FWL OFF + maxT`: `0, 0, 0` vs `20, 27, 3` null FPs; `4/5` vs `3/5` high-block; `1/10` vs `5/10` low-block across `0–153` calls), `5`-fold OOF `No-FWL GLM`, and `4,394`-test `Unmatched MWU`.
2. **`HepG2` `eCLIP` Physical Binding Validation (`tables/fig2e_hepg2_eclip_locus_and_event_validation.tsv`)** — Exact `2x2` conditional Fisher odds ratios and exact conditional 95% confidence intervals across both whole-event regions and `+-35 nt` cis-motif windows (`ENCFF154DRN` `RBFOX2`, `ENCFF170SCU` `PTBP1`).
3. **`12`-Channel (`distal_intron_nt = 500`) `SpliceNet <-> SpliceTrace` Regulatory Bridge (`tables/fig4a_encode_module_rbp_bridge_distal500.tsv`, `fig4c_bridged_vs_orphan_coding_flows.tsv`, `fig4d_deconvolution_vs_competitors.tsv`)** — Explicitly separates `286` `module_P_FWER <= 0.05` bridged coding pathways (`M1`, `M3`, `M9`, `M12`) from `325` unattributed (`FWER > 0.05`) orphan-module coding switches (`exact F1 = 0.400 [2/6]`, `paralog-adjusted F1 = 0.444 [2/5]`).
4. **Single-Cell `Smart-seq2/3` Benchmarks (`GSE270928` & `Tasic`, `tables/fig5a_...` through `fig5f_...`, `figS4a_...`)** — Same-feature `1,000`-HVE comparison vs `MARVEL` (`3.9x`) and `Jutils` (`0/1,000` shared HVEs; `327/66,391` candidate transcripts), donor-evaluability across `167` coding candidates (`60/78` two-donor sign-concordant, `89` single-donor), `SLAMF1` (`Q13291-4` replacing `Q13291-1`) per-cell `83-nt` cassette-exon `PSI` (`18/96` & `90/96` `>=1` junction read; `7/96` & `69/96` computable `PSI`), `scSpliceNet <-> scSpliceTrace` module & RBP bridge tables (`RBM10/HNRNPAB -> M3 [SLAMF1, SRSF11]`, `SF1 -> M2 [PTPRC]`), `Tasic` `71`-mouse 5-fold cross-donor correlation (`0.0201 vs 0.0203 & 0.0032` all types; `0.0064 vs 0.0056 & 0.0022` within animal × cell type), `151`/`53`-gene `MSigDB v2025.1` pathway partitioning, and within-`donor x state` cell-pooling (`5–48` cells).

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
