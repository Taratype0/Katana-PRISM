import {useMemo, useState} from "react";
import {useQuery} from "@tanstack/react-query";
import {api, type CrossParadigmPayload, type CrossParadigmRow} from "../api";
import type {TargetBasketEntry} from "../types";

type ClassKey = "class1" | "class2_af_al" | "class3_switches" | "class4_dexseq";

function fmtP(v: unknown): string {
  if (v == null || v === "" || Number.isNaN(Number(v))) return "NA";
  const n = Number(v);
  if (n <= 0) return "< 1e-300";
  return n < 0.001 ? n.toExponential(2) : n.toFixed(4);
}

function fmtSigned(v: unknown, digits = 3): string {
  if (v == null || v === "" || Number.isNaN(Number(v))) return "NA";
  const n = Number(v);
  return `${n >= 0 ? "+" : ""}${n.toFixed(digits)}`;
}

function boolFlag(v: unknown): boolean {
  if (typeof v === "boolean") return v;
  const s = String(v ?? "").trim().toLowerCase();
  return s === "true" || s === "1" || s === "yes";
}

export function CrossParadigmExplorer({
  projectId,
  comparison,
  onInspectGeneCard,
  onAdd,
  standalone = false,
}: {
  projectId: string;
  comparison: string;
  onInspectGeneCard: (geneSymbol: string, eventId?: string) => void;
  onAdd?: (entry: TargetBasketEntry) => void;
  standalone?: boolean;
}) {
  const [activeClass, setActiveClass] = useState<ClassKey>("class3_switches");
  const [search, setSearch] = useState("");
  const [onlyFdrAfAl, setOnlyFdrAfAl] = useState(false);
  const [onlyHiddenDeg, setOnlyHiddenDeg] = useState(false);
  const [onlyUntestedRmats, setOnlyUntestedRmats] = useState(false);

  const query = useQuery<CrossParadigmPayload | null>({
    queryKey: ["cross-paradigm", projectId, comparison],
    queryFn: () =>
      typeof api.crossParadigm === "function"
        ? api.crossParadigm(projectId, comparison)
        : Promise.resolve(null),
    enabled: typeof api.crossParadigm === "function",
  });

  const data = query.data;
  const rowsByClass = data?.tables || {
    class1: [],
    class2_af_al: [],
    class3_switches: [],
    class4_dexseq: [],
  };

  const filteredRows = useMemo(() => {
    const base = rowsByClass[activeClass] || [];
    const q = search.trim().toLowerCase();
    return base.filter((row: CrossParadigmRow) => {
      if (activeClass === "class2_af_al" && onlyFdrAfAl && !boolFlag(row.sig_fdr)) return false;
      if (activeClass === "class3_switches" && onlyHiddenDeg && !boolFlag(row.hidden_from_deg)) return false;
      if (activeClass === "class4_dexseq" && onlyUntestedRmats && boolFlag(row.in_tested_rmats)) return false;
      if (!q) return true;
      return [
        row.gene_symbol,
        row.gene_id,
        row.gene_base,
        row.event_id,
        row.feature_id,
        row.dexseq_exon,
        row.tx_up,
        row.tx_dn,
        row.fate,
        row.etype,
      ]
        .filter(v => v != null && String(v).trim() !== "")
        .some(v => String(v).toLowerCase().includes(q));
    });
  }, [rowsByClass, activeClass, search, onlyFdrAfAl, onlyHiddenDeg, onlyUntestedRmats]);

  if (typeof api.crossParadigm !== "function") return null;
  if (query.isLoading) return <p>Loading Cross-Paradigm 4-Class explorer…</p>;
  if (!data?.available) {
    if (!standalone) return null;
    return (
      <section className="result-card mck-panel" aria-label="Cross-Paradigm 4-Class Explorer">
        <div className="mck-panel-header">
          <div>
            <span className="eyebrow">CROSS-PARADIGM SYNTHESIS · rMATS × DEXSEQ × SUPPA2 AF/AL × stageR DTU</span>
            <h3>Cross-Paradigm 4-Class &amp; Blind-Spot Explorer</h3>
          </div>
        </div>
        <p className="mck-compact-note">
          No precomputed Cross-Paradigm outputs (rMATS × DEXSeq × SUPPA2 AF/AL × stageR DTU) were found for this bulk report workspace.
        </p>
      </section>
    );
  }

  const cnt = data.counts || {};
  const classTabs: Array<{id: ClassKey; label: string; count: number}> = [
    {id: "class1", label: "Class I · 1-bp Verified", count: rowsByClass.class1.length},
    {id: "class2_af_al", label: "Class II · SUPPA2 AF/AL", count: rowsByClass.class2_af_al.length},
    {id: "class3_switches", label: "Class III · Isoform Switches", count: rowsByClass.class3_switches.length},
    {id: "class4_dexseq", label: "Class IV · DEXSeq-Only", count: rowsByClass.class4_dexseq.length},
  ];

  function addRowToBasket(row: CrossParadigmRow) {
    if (!onAdd || !row.source_artifact_id || !row.row_id) return;
    const sym = String(row.gene_symbol || row.gene_id || row.gene_base || row.row_id);
    onAdd({
      basket_id: crypto.randomUUID(),
      source_artifact_id: String(row.source_artifact_id),
      source_name: `Cross-Paradigm (${activeClass})`,
      row_ids: [String(row.row_id)],
      target_kind: "gene",
      labels: [sym],
      filters: {source_comparison_ids: (data?.comparison_id || comparison) ? [data?.comparison_id || comparison] : []},
    });
  }

  function inspectRow(row: CrossParadigmRow) {
    const sym = String(row.gene_symbol || row.gene_id || row.gene_base || "");
    const ev = activeClass === "class1" || activeClass === "class2_af_al" ? String(row.event_id || "") : "";
    onInspectGeneCard(sym, ev || undefined);
  }

  return (
    <section className="result-card mck-panel" aria-label="Cross-Paradigm 4-Class Explorer">
      <div className="mck-panel-header">
        <div>
          <span className="eyebrow">CROSS-PARADIGM SYNTHESIS · rMATS × DEXSEQ × SUPPA2 AF/AL × stageR DTU</span>
          <h3>4-Class Orthogonal Verification &amp; Blind-Spot Explorer ({data.comparison_id})</h3>
        </div>
        <div style={{display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap"}}>
          {data.dexseq_sign_inverted && (
            <span className="mck-tag mck-tag-blue">DEXSeq log2FC Sign Calibrated</span>
          )}
          {data.explorer_html_url && (
            <a className="secondary-action compact" href={data.explorer_html_url} target="_blank" rel="noreferrer">
              Open CROSS_PARADIGM_EXPLORER.html ↗
            </a>
          )}
        </div>
      </div>

      <div className="mck-kpi-grid" style={{gridTemplateColumns: "repeat(4, minmax(0, 1fr))", marginBottom: 14}}>
        <div className="mck-kpi-box">
          <small>Class I · 1-bp Verified</small>
          <strong>{Number(cnt.class1_events ?? rowsByClass.class1.length).toLocaleString()}</strong>
          <span>{cnt.class1_concordant_pct ?? 0}% direction concordant</span>
        </div>
        <div className="mck-kpi-box">
          <small>Class II · SUPPA2 AF/AL</small>
          <strong>{Number(cnt.suppa_af_al_fdr ?? 0).toLocaleString()} / {Number(cnt.suppa_af_al_nom ?? rowsByClass.class2_af_al.length).toLocaleString()}</strong>
          <span>BH-FDR &lt; 0.05 / nominal ({Number(cnt.suppa_af_al_novel_genes ?? 0).toLocaleString()} novel)</span>
        </div>
        <div className="mck-kpi-box">
          <small>Class III · stageR Switches</small>
          <strong>{Number(cnt.dtu_switch_resolved ?? rowsByClass.class3_switches.length).toLocaleString()}</strong>
          <span>{Number(cnt.dtu_hidden_from_deg ?? 0).toLocaleString()} hidden from DEG</span>
        </div>
        <div className="mck-kpi-box">
          <small>Class IV · DEXSeq-Only</small>
          <strong>{Number(cnt.class4_exons ?? rowsByClass.class4_dexseq.length).toLocaleString()}</strong>
          <span>{Number(cnt.class4_genes ?? 0).toLocaleString()} genes ({Number(cnt.class4_novel_genes ?? 0).toLocaleString()} novel)</span>
        </div>
      </div>

      <div className="mck-controls-bar">
        <div className="mck-filter-pills">
          {classTabs.map(tab => (
            <button
              key={tab.id}
              type="button"
              className={`mck-chip ${activeClass === tab.id ? "active" : ""}`}
              onClick={() => setActiveClass(tab.id)}
            >
              {tab.label} ({tab.count})
            </button>
          ))}
        </div>
        <label style={{minWidth: 210}}>
          Filter rows
          <input
            aria-label="Filter cross-paradigm rows"
            placeholder="Search gene, event, or fate…"
            value={search}
            onChange={e => setSearch(e.target.value)}
          />
        </label>
        {activeClass === "class2_af_al" && (
          <label style={{display: "inline-flex", flexDirection: "row", alignItems: "center", gap: 6}}>
            <input type="checkbox" checked={onlyFdrAfAl} onChange={e => setOnlyFdrAfAl(e.target.checked)} />
            BH-FDR &lt; 0.05 only
          </label>
        )}
        {activeClass === "class3_switches" && (
          <label style={{display: "inline-flex", flexDirection: "row", alignItems: "center", gap: 6}}>
            <input type="checkbox" checked={onlyHiddenDeg} onChange={e => setOnlyHiddenDeg(e.target.checked)} />
            Hidden from DEG only
          </label>
        )}
        {activeClass === "class4_dexseq" && (
          <label style={{display: "inline-flex", flexDirection: "row", alignItems: "center", gap: 6}}>
            <input type="checkbox" checked={onlyUntestedRmats} onChange={e => setOnlyUntestedRmats(e.target.checked)} />
            Untested in rMATS only
          </label>
        )}
      </div>

      <div className="report-table-wrap" style={{maxHeight: 340}}>
        <table>
          <thead>
            {activeClass === "class1" && (
              <tr>
                <th>Gene</th>
                <th>Event</th>
                <th className="num">rMATS ΔPSI</th>
                <th>DEXSeq Exon &amp; Coord</th>
                <th className="num">Calibrated log2FC</th>
                <th className="num">Overlap bp</th>
                <th>Concordant</th>
                <th>Actions</th>
              </tr>
            )}
            {activeClass === "class2_af_al" && (
              <tr>
                <th>Gene</th>
                <th>Type</th>
                <th>Event ID</th>
                <th className="num">ΔPSI</th>
                <th className="num">BH-FDR</th>
                <th className="num">P-value</th>
                <th>PS-DAS / DEG</th>
                <th>Actions</th>
              </tr>
            )}
            {activeClass === "class3_switches" && (
              <tr>
                <th>Gene</th>
                <th>T_dn → T_up Switch</th>
                <th>Fate</th>
                <th className="num">Δaa</th>
                <th className="num">Switch |ΔIF|</th>
                <th className="num">Gene q</th>
                <th>Hidden from DEG</th>
                <th>Actions</th>
              </tr>
            )}
            {activeClass === "class4_dexseq" && (
              <tr>
                <th>Gene</th>
                <th>Exon Bin</th>
                <th>Coord</th>
                <th className="num">log2FC</th>
                <th className="num">padj</th>
                <th>5′→3′ Region</th>
                <th>Tested in rMATS</th>
                <th>Actions</th>
              </tr>
            )}
          </thead>
          <tbody>
            {!filteredRows.length && (
              <tr>
                <td colSpan={8} style={{textAlign: "center", color: "#64748B"}}>
                  No rows match the active class/filter in {data.comparison_id}.
                </td>
              </tr>
            )}
            {filteredRows.slice(0, 120).map((r, idx) => {
              const sym = String(r.gene_symbol || r.gene_id || r.gene_base || "Target");
              const canAdd = Boolean(onAdd && r.source_artifact_id && r.row_id);
              const actions = (
                <td style={{whiteSpace: "nowrap"}}>
                  <button type="button" className="mck-link-btn" onClick={() => inspectRow(r)}>
                    GeneCard →
                  </button>
                  {canAdd && (
                    <button
                      type="button"
                      className="ghost-action"
                      style={{marginLeft: 6, padding: "2px 6px", fontSize: 12}}
                      onClick={() => addRowToBasket(r)}
                    >
                      + Basket
                    </button>
                  )}
                </td>
              );

              if (activeClass === "class1") {
                return (
                  <tr key={`${sym}:${String(r.event_id || idx)}`}>
                    <td><strong>{sym}</strong></td>
                    <td><code>{String(r.event_type || "")} · {String(r.event_id || "").slice(0, 30)}</code></td>
                    <td className="num">{fmtSigned(r.rmats_dpsi, 3)}</td>
                    <td><small>{String(r.dexseq_exon || "")} · {String(r.dexseq_coord || "")}</small></td>
                    <td className="num">{fmtSigned(r.dexseq_lfc, 2)}</td>
                    <td className="num">{String(r.overlap_bp ?? "")}</td>
                    <td>{boolFlag(r.concordant_fixed) ? "Yes" : "No"}</td>
                    {actions}
                  </tr>
                );
              }
              if (activeClass === "class2_af_al") {
                return (
                  <tr key={`${sym}:${String(r.event_id || idx)}`}>
                    <td><strong>{sym}</strong></td>
                    <td><span className="mck-tag mck-tag-blue">{String(r.etype || "")}</span></td>
                    <td><code title={String(r.event_id || "")}>{String(r.event_id || "").slice(0, 36)}</code></td>
                    <td className="num">{fmtSigned(r.dpsi, 3)}</td>
                    <td className="num">{fmtP(r.bh_fdr)}</td>
                    <td className="num">{fmtP(r.pvalue)}</td>
                    <td><small>{boolFlag(r.in_ps_das) ? "PS-DAS" : "rMATS-blind"} / {boolFlag(r.in_deg) ? "DEG" : "non-DEG"}</small></td>
                    {actions}
                  </tr>
                );
              }
              if (activeClass === "class3_switches") {
                return (
                  <tr key={`${sym}:${String(r.tx_dn || "")}:${String(r.tx_up || idx)}`}>
                    <td><strong>{sym}</strong></td>
                    <td><code>{String(r.tx_dn || "")} → {String(r.tx_up || "")}</code></td>
                    <td><small>{String(r.fate || "")}</small></td>
                    <td className="num">{r.delta_aa != null ? fmtSigned(r.delta_aa, 0) : "NA"}</td>
                    <td className="num">{r.switch_mag != null ? Number(r.switch_mag).toFixed(3) : "NA"}</td>
                    <td className="num">{fmtP(r.gene_q)}</td>
                    <td>{boolFlag(r.hidden_from_deg) ? <span className="mck-tag mck-tag-coral">Hidden from DEG</span> : "DEG sig"}</td>
                    {actions}
                  </tr>
                );
              }
              return (
                <tr key={`${sym}:${String(r.feature_id || idx)}`}>
                  <td><strong>{sym}</strong></td>
                  <td><code>{String(r.feature_id || "")}</code></td>
                  <td><small>{String(r.coord || "")}</small></td>
                  <td className="num">{fmtSigned(r.lfc, 2)}</td>
                  <td className="num">{fmtP(r.padj)}</td>
                  <td><small>{String(r.region_bin || "")}</small></td>
                  <td>{boolFlag(r.in_tested_rmats) ? "Tested" : "Untested"}</td>
                  {actions}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
