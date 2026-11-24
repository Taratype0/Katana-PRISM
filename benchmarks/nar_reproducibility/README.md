# Katana-PRISM (`v1.0.1`) — Public Benchmark Reproducibility Suite

This directory provides the standalone benchmark driver (`reproduce_nar_benchmarks.py`) and verified summary tables (`tables/*.tsv`) accompanying the *Nucleic Acids Research* manuscript:

1. **Foreground-Block & Single-Variable `FWL` `SpliceTrace` Calibration (`tables/fig2c_splicetrace_exact_and_ablations.tsv`, `tableS2a_native_rmaps3_all18_cohorts.tsv`, `tableS2a_untruncated_rmaps2_169rbp_all18.tsv`)** — Reports per-protein exact `P_FWER` and ranks across `169` human (`168` mouse) RBPs × `13` channels for unified discrete matching (`999` gene-block sign flips), continuous `FWL ON + maxT` (`1,999` between-gene response permutations), single-variable `No-FWL` (`FWL OFF + maxT`: `0, 0, 0` vs `21, 27, 3` null FPs; `4/5` vs `3/5` high-block; `1/10` vs `5/10` low-block across `0–153` calls), `5`-fold OOF `No-FWL GLM`, `4,394`-test `Unmatched MWU`, official unmodified native `rMAPS3` (`10, 26, 29` null Bonf FPs), and untruncated `169`-RBP `rMAPS2`-style proxy (`4, 1, 0` Bonf FPs).
2. **`GRCh38` Physical `HepG2` IDR `eCLIP` & `PTBP2` `CLIP` Validation (`tables/fig2e_hepg2_eclip_locus_and_event_validation.tsv`, `fig1_ps_das_rescued_eclip_validation.tsv`)** — Exact `2x2` conditional Fisher odds ratios and exact conditional 95% confidence intervals across whole events, `+-35 nt` cis-motif windows, and `PS-DAS_rescued_only` subsets.
3. **Real `GSE122380` `12`-Channel (`distal_intron_nt = 500`) `SpliceNet <-> SpliceTrace` Regulatory Bridge (`tables/fig4a_gse122380_module_rbp_bridge_distal500.tsv`, `fig4_causal_alluvial_flows.tsv`, `fig4d_deconvolution_vs_competitors.tsv`)** — Evaluates `4` modules (`M1–M4` across `N = 297` libraries, `19` donors) × `169` human RBPs (`676` rows: `9` `study_P_FWER <= 0.05` pairs across `8` developmental RBPs / `3/3` cardiac families in `M1/M2`, `0` FPs in `M3/M4`), separates `978` coding alluvial flows (`612` bridged + `366` `M3/M4` orphan), and benchmarks `6` methods across `50` null trials under matched study-wide `Holm <= 0.05` (`Katana-PRISM 4.0%` vs `34.0%` / `48.0%`).
4. **Single-Cell `Smart-seq2/3` Benchmarks (`GSE270928`, `Tabula Muris`, `Tasic`, `Longcell`; `tables/fig5a_...` through `fig5f_...`, `figS4a_...`, `gse270928_smartseq3_usage_prediction.tsv`, `suite3_longcell_task1_blind_rbp_recovery.tsv`, `tableS7d_manifold_and_treecut_sensitivity.tsv`)** — Same-feature `1,000`-HVI state separation vs `MARVEL` (`3.9x`) and `Jutils` (`0/1,000` shared HVIs; `327/66,391` candidate transcripts; `k_neighbors = 12 [default]` `Delta_kNN = 0.9748` / `LISI = 1.385`), donor-evaluability across `167` coding candidates (`60/78` two-donor sign-concordant, `89` single-donor), `SLAMF1` (`Q13291-4` replacing `Q13291-1`) per-cell `83-nt` `PSI` (`18/96` & `90/96`; `7/96` & `69/96`), `scSpliceNet <-> scSpliceTrace` module bridge (`RBM10`/`HNRNPAB -> M3`, `SF1 -> M2`, `FMR1 -> M4`), `Tasic` `71`-mouse `5`-fold correlation (`0.0201 vs 0.0203 & 0.0032` all types; `0.0064 vs 0.0056 & 0.0022` within animal × cell type), `GSE270928` usage `MSE` (`0.00968` virtual tie vs `acorde`), `Longcell` `N=1` blind CRISPRi stress test (`0/12` `FWER`), and dual-background `MSigDB v2025.1` pathway partitioning.

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
