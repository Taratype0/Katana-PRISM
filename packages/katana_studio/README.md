# Katana Studio & Katana Report (`1.0.1`)

**`katana-studio`** provides two CLI entrypoints:
1. **`katana-report`**: Synthesizes upstream `Katana` Nextflow bundles or standalone `SpliceManifold` / `SpliceNet` / `SpliceTrace` outputs into a self-contained, SHA256-audited `REPORT.html` and portable `REPORT.zip` archive.
2. **`katana-studio`**: Launches the local interactive web workbench (`http://127.0.0.1:18775`) to explore any `REPORT.zip`, adjust `PS-DAS` vs `Ordinary DAS` filters live, inspect co-splicing hub networks, run live RBP motif & GO/KEGG enrichment queries, and render on-demand Sashimi plots.

## Usage
```bash
pip install ./packages/katana_studio

# Build self-contained REPORT.html + REPORT.zip
katana-report bulk --bundle results/katana_bundle --outdir results/report

# Open interactive local workbench
katana-studio results/report/REPORT.zip --port 18775
```
