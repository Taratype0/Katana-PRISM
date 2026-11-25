import {render, screen, fireEvent} from "@testing-library/react";
import {QueryClient, QueryClientProvider} from "@tanstack/react-query";
import {vi, test, expect} from "vitest";
import {ReportWorkspace} from "./ReportWorkspace";
import {ApplicationTree, applicationRoutes} from "./ApplicationTree";
import {CrossParadigmExplorer} from "./CrossParadigmExplorer";
import {api} from "../api";
import type {ProjectPayload, Artifact} from "../types";

vi.mock("./ReportFigures", () => ({
  reportGet: vi.fn(async () => ({
    panels: [
      {id: "p", title: "Sample PSI PCA", chapter: 2, contrast_id: "all"},
      {id: "m", title: "Native MAPP activity", chapter: 7, contrast_id: "C1"},
    ],
  })),
  reportUrl: (pid: string, rel: string) => `/api/projects/${pid}/report-source/${rel}`,
  ReportFigures: ({panels}: {panels: {title: string}[]}) => (
    <div>{panels.map(p => <span key={p.title}>{p.title}</span>)}</div>
  ),
}));
vi.mock("./ReportExplorer", () => ({
  ReportExplorer: ({artifacts, eventType}: {artifacts: Artifact[]; eventType?: string}) => (
    <div>
      {artifacts.map(a => <span key={a.artifact_id}>{a.name}</span>)}
      {eventType}
    </div>
  ),
}));
vi.mock("./ReportNetwork", () => ({ReportNetwork: () => <div>Inline interactive network</div>}));
vi.mock("./PsiLandscapeLab", () => ({PsiLandscapeLab: () => <div>Live PSI Landscape Lab</div>}));
vi.mock("./LiveOraWorkbench", () => ({LiveOraWorkbench: () => <div>Live GO KEGG ORA Lab</div>}));

const project = {
  snapshot: {project_id: "P", genome_build: "GRCm39"},
  manifest: {samples: [], contrasts: [{contrast_id: "C1"}, {contrast_id: "C2"}]},
} as unknown as ProjectPayload;

function show(route: string, artifacts: Artifact[] = []) {
  return render(
    <QueryClientProvider client={new QueryClient({defaultOptions: {queries: {retry: false}}})}>
      <ReportWorkspace
        project={project}
        route={applicationRoutes.find(r => r.id === route)!}
        artifacts={artifacts}
        onAdd={vi.fn()}
        onNavigate={vi.fn()}
        onSetup={vi.fn()}
      />
    </QueryClientProvider>
  );
}

test("ordered 01-09 Consulting navigation includes QC, PSI Lab, DAS subtypes, SpliceNet, SpliceTrace and GeneCard", () => {
  const select = vi.fn();
  render(<ApplicationTree active="das" onSelect={select} />);
  expect(screen.getByRole("button", {name: /03 · Differential AS/})).toHaveAttribute("aria-current", "page");
  fireEvent.click(screen.getByRole("button", {name: /Event-type subanalyses/}));
  fireEvent.click(screen.getByRole("button", {name: /03\.SE · SE Subtype Analysis/}));
  expect(select).toHaveBeenCalledWith("event-SE");
  expect(screen.getByRole("button", {name: /02 · PSI Landscape/})).toBeInTheDocument();
  expect(screen.getByRole("button", {name: /05 · GO & KEGG Live Enrichment Lab/})).toBeInTheDocument();
  expect(screen.getByRole("button", {name: /06 · Katana-SpliceNet/})).toBeInTheDocument();
  expect(screen.getByRole("button", {name: /07 · Katana-SpliceTrace/})).toBeInTheDocument();
  expect(screen.getByRole("button", {name: /08 · Target Scorer & 4-Track GeneCard/})).toBeInTheDocument();
});

test("navigation adapts between bulk and single-cell assay modes", () => {
  const {rerender} = render(<ApplicationTree active="das" assayMode="bulk" onSelect={vi.fn()} />);
  expect(screen.queryByRole("button", {name: /02\.2 Single-Cell Dual Manifold/})).not.toBeInTheDocument();
  expect(screen.getByRole("button", {name: /03\.2 Isoform · DTU \/ DEXSeq \/ SUPPA2/})).toBeInTheDocument();
  screen.getAllByRole("button", {name: /Subanalyses/}).forEach(b => fireEvent.click(b));
  expect(screen.getByRole("button", {name: /04\.2 Cross-Paradigm/})).toBeInTheDocument();
  rerender(<ApplicationTree active="sc-manifold" assayMode="single-cell" onSelect={vi.fn()} />);
  expect(screen.getByRole("button", {name: /02\.2 Single-Cell Dual Manifold/})).toBeInTheDocument();
  expect(screen.queryByRole("button", {name: /03\.2 Isoform/})).not.toBeInTheDocument();
  expect(screen.queryByRole("button", {name: /04\.2 Cross-Paradigm/})).not.toBeInTheDocument();
});

test("shared figures and Live PSI Lab mount on 02 patterns without comparison leakage", async () => {
  show("patterns");
  expect(screen.getByText("Live PSI Landscape Lab")).toBeInTheDocument();
  expect(await screen.findByText("Sample PSI PCA")).toBeInTheDocument();
  expect(screen.queryByText("Native MAPP activity")).not.toBeInTheDocument();
});

test("motif figures are visible directly alongside SpliceTrace Lab", async () => {
  show("motif");
  expect(await screen.findByText("Native MAPP activity")).toBeInTheDocument();
});

test("DAS route selects only its method and comparison", () => {
  show("das", [
    {artifact_id: "1", name: "C1 DAS", metadata: {source_analysis_id: "differential.as"}, contrast_ids: ["C1"]},
    {artifact_id: "2", name: "C2 DAS", metadata: {source_analysis_id: "differential.as"}, contrast_ids: ["C2"]},
    {artifact_id: "3", name: "C1 DEG", metadata: {source_analysis_id: "differential.deg"}, contrast_ids: ["C1"]},
  ] as unknown as Artifact[]);
  expect(screen.getByText("C1 DAS")).toBeInTheDocument();
  expect(screen.queryByText("C2 DAS")).not.toBeInTheDocument();
  expect(screen.queryByText("C1 DEG")).not.toBeInTheDocument();
  fireEvent.change(screen.getByLabelText("Analysis comparison"), {target: {value: "C2"}});
  expect(screen.getByText("C2 DAS")).toBeInTheDocument();
});

test("SpliceNet mounts directly without an iframe", () => {
  const {container} = show("network");
  expect(screen.getByText("Inline interactive network")).toBeInTheDocument();
  expect(container.querySelector("iframe")).toBeNull();
});

test("CrossParadigmExplorer renders 4 classes, filters Hidden from DEG, and triggers GeneCard and Target Basket", async () => {
  const spy = vi.spyOn(api, "crossParadigm").mockResolvedValue({
    available: true,
    comparison_id: "KO_vs_WT",
    available_contrasts: ["KO_vs_WT"],
    dexseq_sign_inverted: true,
    explorer_html_url: "/api/projects/P/report-source/outputs/cross_paradigm/CROSS_PARADIGM_EXPLORER.html",
    counts: {
      class1_events: 5,
      class1_concordant_pct: 100.0,
      suppa_af_al_fdr: 2,
      suppa_af_al_nom: 3,
      suppa_af_al_novel_genes: 2,
      dtu_switch_resolved: 2,
      dtu_hidden_from_deg: 1,
      class4_exons: 1,
      class4_genes: 1,
      class4_novel_genes: 1,
    },
    artifact_ids: {class3_switches: "art_c3"},
    tables: {
      class1: [{row_id: "r1", source_artifact_id: "art_c1", gene_id: "ENSMUSG1", gene_symbol: "Gene1", event_id: "SE:chr1", event_type: "SE", rmats_dpsi: 0.35, dexseq_exon: "E002", dexseq_coord: "chr1:100-200(+)", dexseq_lfc: 1.8, overlap_bp: 90, concordant_fixed: true}],
      class2_af_al: [{row_id: "r2", source_artifact_id: "art_c2", gene_id: "ENSMUSG21", gene_symbol: "Gene21", etype: "AF", event_id: "ENSMUSG21;AF:1", dpsi: 0.48, bh_fdr: 0.001, pvalue: 0.0001, sig_fdr: true, in_ps_das: false, in_deg: false}],
      class3_switches: [
        {row_id: "r3a", source_artifact_id: "art_c3", gene_id: "ENSMUSG10", gene_symbol: "Gene10", tx_dn: "TX1", tx_up: "TX2", fate: "Coding -> NMD (Loss of ORF)", delta_aa: -120, switch_mag: 0.65, gene_q: 0.0001, hidden_from_deg: true},
        {row_id: "r3b", source_artifact_id: "art_c3", gene_id: "ENSMUSG11", gene_symbol: "Gene11", tx_dn: "TX3", tx_up: "TX4", fate: "Coding -> Coding (Isoform Switch)", delta_aa: 15, switch_mag: 0.42, gene_q: 0.002, hidden_from_deg: false},
      ],
      class4_dexseq: [{row_id: "r4", source_artifact_id: "art_c4", gene_id: "ENSMUSG12", gene_symbol: "Gene12", feature_id: "E001", coord: "chr1:1-90(+)", lfc: 1.6, padj: 0.0002, region_bin: "5prime", in_tested_rmats: false}],
    },
    bench_multi_contrast: [],
    suppa_7class: [],
    interactive_genes: [],
    gene_dex_bins: {},
  });
  const onInspect = vi.fn();
  const onAdd = vi.fn();
  render(
    <QueryClientProvider client={new QueryClient({defaultOptions: {queries: {retry: false}}})}>
      <CrossParadigmExplorer projectId="P" comparison="KO_vs_WT" onInspectGeneCard={onInspect} onAdd={onAdd} standalone />
    </QueryClientProvider>
  );
  expect(await screen.findByText(/Gene10/)).toBeInTheDocument();
  expect(screen.getByText(/Gene11/)).toBeInTheDocument();
  expect(screen.getByRole("link", {name: /CROSS_PARADIGM_EXPLORER\.html/})).toBeInTheDocument();

  fireEvent.click(screen.getByLabelText(/Hidden from DEG only/));
  expect(screen.getByText(/Gene10/)).toBeInTheDocument();
  expect(screen.queryByText(/Gene11/)).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", {name: /GeneCard/}));
  expect(onInspect).toHaveBeenCalledWith("Gene10", undefined);

  fireEvent.click(screen.getByRole("button", {name: /\+ Basket/}));
  expect(onAdd).toHaveBeenCalledWith(expect.objectContaining({
    source_artifact_id: "art_c3",
    row_ids: ["r3a"],
    target_kind: "gene",
  }));

  fireEvent.click(screen.getByRole("button", {name: /Class II · SUPPA2 AF\/AL/}));
  expect(screen.getByText(/Gene21/)).toBeInTheDocument();
  spy.mockRestore();
});
