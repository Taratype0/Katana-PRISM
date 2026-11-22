"""Independent Unified 2×2 GeneCard & EventCard HTML/SVG Reports; never alters the study report renderer."""
from __future__ import annotations

from html import escape
import json
import math
from pathlib import Path
import re

import pandas as pd

from ._execution import PLOT_LOCK, portable, write_json


STYLE = """body{font:14px/1.6 'IBM Plex Sans','Inter','Segoe UI',system-ui,sans-serif;color:#1e293b;background:#f8fafc;margin:0;font-variant-numeric:tabular-nums}
header{background:linear-gradient(125deg,#051c2c 0%,#0b2f4a 100%);color:white;padding:30px max(5%,28px);border-bottom:3px solid #2251ff}
h1{margin:0 0 6px;font:400 30px/1.15 'Newsreader',Georgia,serif;letter-spacing:-.02em}
header p{margin:0;color:#cbd5e1;font-size:13.5px}
main{max-width:1360px;margin:26px auto;padding:0 28px}
section,article{background:white;padding:24px 28px;margin:18px 0;border:1px solid #e2e8f0;border-top:3px solid #051c2c;border-radius:3px;box-shadow:0 1px 2px rgba(5,28,44,.04)}
h2{font:400 21px/1.25 'Newsreader',Georgia,serif;color:#051c2c;margin:0 0 12px;display:flex;align-items:center;gap:10px}
h3{font:600 15.5px/1.3 'IBM Plex Sans',sans-serif;color:#051c2c;margin:0 0 8px}
a{color:#2251ff;text-decoration:none}a:hover{text-decoration:underline}p,li{line-height:1.6}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{text-align:left;padding:9px 12px 9px 0;border-bottom:1px solid #e2e8f0;vertical-align:top;max-width:420px;overflow-wrap:anywhere}
th{border-top:1.5px solid #051c2c;border-bottom:1px solid #94a3b8;color:#051c2c;font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;font-weight:600;background:#fff}
img{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:2px}.scroll{overflow-x:auto}
.muted{color:#64748b;font-size:12.5px}
.pill{display:inline-block;padding:3px 9px;background:#eff6ff;color:#2251ff;border:1px solid #bfdbfe;border-radius:2px;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.04em}
.complete{color:#0d9488;font-weight:600}.partial{color:#d04a02;font-weight:600}.failed{color:#b83a4b;font-weight:600}.unavailable{color:#64748b;font-weight:600}
input,button,select{font:inherit;padding:7px 10px;border:1px solid #cbd5e1;background:white;border-radius:2px}button{cursor:pointer}
.toolbar{display:flex;gap:12px;align-items:center;margin:12px 0;flex-wrap:wrap}.toolbar input{min-width:260px}
"""


def html_page(title, body, subtitle=""):
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{escape(title)}</title><style>{STYLE}</style></head><body>"
            f"<header><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></header>"
            f"<main>{body}</main></body></html>")


def _json_script(value):
    return json.dumps(portable(value), ensure_ascii=False, allow_nan=False).replace("<", "\\u003c").replace("&", "\\u0026")


def _fnum(v):
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _fmt_p(v):
    f = _fnum(v)
    if f is None:
        return "NA"
    if f <= 0:
        return "< 1e-16"
    return f"{f:.2e}" if f < 1e-3 else f"{f:.4f}"


def _clean_str(v) -> str:
    if v is None:
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"nan", "none", "nat", "<na>"} else s


def _effective_analysis_id(entry: dict) -> str:
    meta = entry.get("metadata") or {}
    return str(meta.get("source_analysis_id") or entry.get("analysis_id") or "")


def render_table(path, title, frame, note="", *, total_rows=None, full_data=None):
    """All rows searchable and paginated offline, no remote JS/fetch required."""
    columns = [str(c) for c in frame.columns]
    rows = portable(frame.to_dict("records"))
    preview = total_rows is not None and total_rows > len(frame)
    body = f"<p>{escape(note)}</p>"
    if preview:
        body += f"<p>Preview: {len(frame):,} of {total_rows:,} rows. Search covers this preview only. "
        body += f"<a href='{escape(full_data, quote=True)}'>Download all {total_rows:,} rows (Parquet)</a>.</p>"
    else:
        body += f"<p>{len(frame):,} complete rows. Search examines every row.</p>"
    body += """<div class='toolbar'><input id='search' type='search' placeholder='Search all rows' aria-label='Search all rows'>
    <select id='size' aria-label='Rows per page'><option>25</option><option>50</option><option>100</option><option>250</option></select>
    <button id='previous'>Previous</button><span id='page'></span><button id='next'>Next</button></div>
    <div class='scroll'><table><thead id='head'></thead><tbody id='rows'></tbody></table></div>"""
    if preview:
        body = body.replace('Search all rows', 'Search preview')
    body += f"<script id='table-data' type='application/json'>{_json_script({'columns': columns, 'rows': rows})}</script>"
    body += """<script>(()=>{'use strict';const data=JSON.parse(document.getElementById('table-data').textContent);
    let page=0;const search=document.getElementById('search'),size=document.getElementById('size');
    const cell=v=>v===null||v===undefined?'':typeof v==='object'?JSON.stringify(v):String(v);
    const tr=document.createElement('tr');for(const c of data.columns){const th=document.createElement('th');th.textContent=c;tr.append(th)}
    document.getElementById('head').append(tr);function draw(){const q=search.value.toLowerCase();
    const selected=data.rows.filter(r=>data.columns.some(c=>cell(r[c]).toLowerCase().includes(q)));
    const n=Number(size.value),pages=Math.max(1,Math.ceil(selected.length/n));page=Math.min(page,pages-1);
    const tbody=document.getElementById('rows');tbody.replaceChildren();for(const r of selected.slice(page*n,(page+1)*n)){
    const row=document.createElement('tr');for(const c of data.columns){const td=document.createElement('td');td.textContent=cell(r[c]);row.append(td)}tbody.append(row)}
    document.getElementById('page').textContent=`${selected.length} rows · Page ${page+1} / ${pages}`;
    document.getElementById('previous').disabled=page===0;document.getElementById('next').disabled=page===pages-1}
    search.addEventListener('input',()=>{page=0;draw()});size.addEventListener('change',()=>{page=0;draw()});
    document.getElementById('previous').addEventListener('click',()=>{page=Math.max(0,page-1);draw()});
    document.getElementById('next').addEventListener('click',()=>{page++;draw()});draw()})();</script>"""
    Path(path).write_text(html_page(title, body), encoding="utf-8")


def sample_measurements(request, target, tables):
    """Display native saved per-sample values; no model/statistical refit."""
    samples = {str(s["sample_id"]): s for s in request["project"].get("samples", [])}
    contrasts = {str(c["contrast_id"]): c for c in request["project"].get("contrasts", [])}
    rows = []
    seen = set()
    for item, frame in tables:
        aid = _effective_analysis_id(item)
        default_meas = "PSI" if ("psi" in aid or "as" in aid or "landscape" in aid) else ("Normalized counts" if "normalised" in aid or "expression" in aid else "saved_matrix")
        default_unit = "fraction" if default_meas == "PSI" else ("normalized_counts" if "expression" in aid or "normalised" in aid else item.get("metadata", {}).get("unit", "source units"))
        for _, row in frame.iterrows():
            cid = str(row.get("comparison_id", row.get("contrast_id", item.get("metadata", {}).get("comparison_id", ""))))
            eid = str(row.get("event_id", row.get("event_uid", row.get("event_key", row.get("ID", "")))))
            for sid, sample in samples.items():
                if sid in frame.columns:
                    value = pd.to_numeric(pd.Series([row[sid]]), errors="coerce").iloc[0]
                    if pd.notna(value):
                        key = (item["evidence_id"], eid, cid, sid, default_meas)
                        if key not in seen:
                            rows.append(dict(evidence_id=item["evidence_id"], event_id=eid, comparison_id=cid,
                                             sample_id=sid, condition=sample.get("condition"), measurement=default_meas,
                                             unit=default_unit, value=float(value)))
                            seen.add(key)
            if cid in contrasts:
                contrast = contrasts[cid]
                for arm_number, arm_name in ((1, "numerator"), (2, "denominator")):
                    column = f"IncLevel{arm_number}"
                    if column not in frame.columns:
                        continue
                    raw = row[column]
                    if isinstance(raw, str):
                        values = raw.split(",")
                    elif isinstance(raw, (tuple, list)) or hasattr(raw, "tolist"):
                        values = list(raw)
                    else:
                        continue
                    ids = contrast.get(arm_name, {}).get("sample_ids", [])
                    flipped = any(str(row.get(flag, "")).lower() in {"1", "true", "yes"}
                                  for flag in ("splicing_direction_flipped", "reverse_direction"))
                    if flipped:
                        other = row.get(f"IncLevel{3 - arm_number}")
                        if isinstance(other, str):
                            values = other.split(",")
                        elif isinstance(other, (tuple, list)) or hasattr(other, "tolist"):
                            values = list(other)
                    if len(ids) != len(values):
                        continue
                    for sid, value in zip(ids, values):
                        value = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
                        key = (item["evidence_id"], eid, cid, str(sid), "PSI")
                        if pd.notna(value) and key not in seen:
                            rows.append(dict(evidence_id=item["evidence_id"], event_id=eid, comparison_id=cid,
                                             sample_id=str(sid), condition=samples.get(str(sid), {}).get("condition"),
                                             measurement="PSI", unit="fraction", value=float(value)))
                            seen.add(key)
            if "sample_id" in frame.columns:
                sid = str(row["sample_id"])
                for column in ("psi", "PSI", "expression", "normalized_count", "TPM", "count", "value"):
                    if column not in frame.columns:
                        continue
                    value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                    key = (item["evidence_id"], eid, cid, sid, column)
                    if sid in samples and pd.notna(value) and key not in seen:
                        rows.append(dict(evidence_id=item["evidence_id"], event_id=eid, comparison_id=cid,
                                         sample_id=sid, condition=samples[sid].get("condition"),
                                         measurement=column, unit=item.get("metadata", {}).get("unit", "source units"),
                                         value=float(value)))
                        seen.add(key)
    items = {item["evidence_id"]: item for item, _ in tables}
    for row in rows:
        sample = samples.get(row["sample_id"], {})
        item = items[row["evidence_id"]]
        attributes = sample.get("attributes", {})
        row.update(biological_sample_id=sample.get("biological_sample_id"),
                   donor_id=attributes.get("donor_id"), cell_type=attributes.get("cell_type"),
                   count_mode=item.get("metadata", {}).get("count_mode") or item.get("parameters", {}).get("count_mode") or "unknown",
                   source_statistical_family=item.get("source_fingerprint"))
    return pd.DataFrame(rows, columns=["evidence_id", "event_id", "comparison_id", "sample_id",
                                       "biological_sample_id", "donor_id", "cell_type", "condition",
                                       "measurement", "unit", "value", "count_mode", "source_statistical_family"])


def render_sample_plots(root, frame):
    if frame.empty:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(root)
    plots = []
    for i, ((evidence_id, mode, eid, cid, measurement, unit), group) in enumerate(
            frame.groupby(["evidence_id", "count_mode", "event_id", "comparison_id", "measurement", "unit"], sort=False, dropna=False)):
        relative = f"plots/sample-{i:05d}"
        (root / "plots").mkdir(exist_ok=True)
        with PLOT_LOCK:
            fig, ax = plt.subplots(figsize=(max(6.8, min(14.0, len(group) * 0.62)), 3.8), dpi=180)
            fig.patch.set_facecolor("white")
            conditions = list(dict.fromkeys(group["condition"].fillna("unknown").astype(str)))
            palette = ("#2251FF", "#B83A4B", "#0D9488", "#D04A02", "#051C2C")
            colors = {c: palette[j % len(palette)] for j, c in enumerate(conditions)}
            vals = group["value"].to_numpy(dtype=float)
            cols = [colors[str(c)] for c in group["condition"].fillna("unknown")]
            x = range(len(group))
            ax.bar(x, vals, color=cols, alpha=0.82, width=0.58)
            if measurement == "PSI":
                ax.set_ylim(-0.04, 1.08)
            ax.scatter(x, vals, color="#051C2C", s=24, zorder=4)
            ax.set_xticks(list(x), group["sample_id"], rotation=38, ha="right", fontsize=8.5)
            ax.set_ylabel(f"{measurement} ({unit})", fontweight="bold", color="#051C2C")
            ax.set_title(f"{eid or 'Host gene'} · {cid or 'Cohort'} · {measurement}", loc="left", fontsize=10, fontweight="bold", color="#051C2C")
            ax.spines[["top", "right"]].set_visible(False)
            fig.tight_layout()
            for suffix in ("png", "svg"):
                fig.savefig(root / f"{relative}.{suffix}", dpi=180, facecolor="white")
            plt.close(fig)
        plots.append({"path": relative + ".png", "svg": relative + ".svg", "event_id": eid,
                      "comparison_id": cid, "measurement": measurement, "unit": unit,
                      "evidence_id": evidence_id, "count_mode": mode})
    return plots


def _load_target_evidence_frames(root: Path, evidence: list[dict]) -> dict[str, list[pd.DataFrame]]:
    by_aid: dict[str, list[pd.DataFrame]] = {}
    for item in evidence:
        pq_rel = item.get("parquet")
        if not pq_rel:
            continue
        fp = root / pq_rel
        if fp.is_file():
            try:
                by_aid.setdefault(_effective_analysis_id(item), []).append(pd.read_parquet(fp))
            except Exception:
                pass
    return by_aid


def _junction_schematic_svg(card: dict, psi_arms: list[tuple[str, float | None, float | None, float | None]]) -> str:
    geom = card.get("event_geometry") or {}
    et = str(card.get("event_type") or "SE").upper()
    chrom = str(geom.get("chr") or geom.get("chrom") or geom.get("chromosome") or "chr?")
    strand = str(geom.get("strand") or "+")

    def _pair(k1, k2):
        a, b = _fnum(geom.get(k1)), _fnum(geom.get(k2))
        return (int(a), int(b)) if (a is not None and b is not None and b > a) else None

    exons = []
    if et == "SE":
        for lbl, role, p in (("Upstream", "const", _pair("upstreamES", "upstreamEE")),
                             ("Cassette", "alt", _pair("exonStart_0base", "exonEnd")),
                             ("Downstream", "const", _pair("downstreamES", "downstreamEE"))):
            if p:
                exons.append((lbl, role, p[0], p[1]))
    elif et == "RI":
        u, d = _pair("upstreamES", "upstreamEE"), _pair("downstreamES", "downstreamEE")
        if u and d:
            exons = [("5' Exon", "const", u[0], u[1]), ("Retained Intron", "alt", u[1], d[0]), ("3' Exon", "const", d[0], d[1])]
    elif et in {"A3SS", "A5SS"}:
        f, l, s = _pair("flankingES", "flankingEE"), _pair("longExonStart_0base", "longExonEnd"), _pair("shortES", "shortEE")
        if f and l and s:
            exons = sorted([("Flanking", "const", f[0], f[1]), ("Core Exon", "const", s[0], s[1]), (f"{et} Ext", "alt", min(l[0], s[1]), max(l[1], s[0]))], key=lambda x: x[2])
    elif et == "MXE":
        for lbl, role, p in (("Upstream", "const", _pair("upstreamES", "upstreamEE")),
                             ("MXE 1", "alt", _pair("1stExonStart_0base", "1stExonEnd")),
                             ("MXE 2", "alt2", _pair("2ndExonStart_0base", "2ndExonEnd")),
                             ("Downstream", "const", _pair("downstreamES", "downstreamEE"))):
            if p:
                exons.append((lbl, role, p[0], p[1]))

    if not exons:
        return "<p class='muted'>Exon geometry coordinates unavailable for vector schematic.</p>"

    W, H = 760, 170
    n_ex = len(exons)
    box_w = 112 if n_ex <= 3 else 94
    gap = (W - 120 - n_ex * box_w) / max(1, n_ex - 1)
    xs = [60 + i * (box_w + gap) for i in range(n_ex)]
    y_line = 92
    parts = [f"<svg viewBox='0 0 {W} {H}' role='img' aria-label='Exon junction schematic' style='width:100%;height:auto;background:#fcfdff;border:1px solid #e2e8f0;border-radius:3px'>"]
    parts.append(f"<text x='18' y='22' font-size='11.5' font-weight='700' fill='#051c2c'>{escape(et)} Structure · {escape(chrom)} ({escape(strand)} strand)</text>")
    if psi_arms:
        arm_txt = "  |  ".join(
            f"{lbl}: PSI={(f'{psi:.3f}' if psi is not None else 'NA')}"
            + (f" (IJC={ijc:.0f}, SJC={sjc:.0f})" if ijc is not None and sjc is not None else "")
            for lbl, psi, ijc, sjc in psi_arms[:2]
        )
        parts.append(f"<text x='18' y='40' font-size='11' fill='#475569'>{escape(arm_txt)}</text>")
    parts.append(f"<line x1='45' y1='{y_line}' x2='{W-45}' y2='{y_line}' stroke='#94a3b8' stroke-width='2' stroke-dasharray='4,3'/>")
    if n_ex >= 3:
        x0e, x1s, x1e, x2s = xs[0] + box_w, xs[1], xs[1] + box_w, xs[-1]
        parts.append(f"<path d='M {x0e:.1f} {y_line-14} Q {(x0e+x1s)/2:.1f} {y_line-44} {x1s:.1f} {y_line-14}' fill='none' stroke='#2251ff' stroke-width='2.2'/>")
        parts.append(f"<path d='M {x1e:.1f} {y_line-14} Q {(x1e+x2s)/2:.1f} {y_line-44} {x2s:.1f} {y_line-14}' fill='none' stroke='#2251ff' stroke-width='2.2'/>")
        parts.append(f"<path d='M {x0e:.1f} {y_line+14} Q {(x0e+x2s)/2:.1f} {y_line+56} {x2s:.1f} {y_line+14}' fill='none' stroke='#b83a4b' stroke-width='2.0' stroke-dasharray='5,3'/>")
    for (lbl, role, st, en), x0 in zip(exons, xs):
        fill = "#051c2c" if role == "const" else ("#2251ff" if role == "alt" else "#0d9488")
        parts.append(f"<rect x='{x0:.1f}' y='{y_line-14}' width='{box_w}' height='28' rx='3' fill='{fill}'/>")
        parts.append(f"<text x='{x0+box_w/2:.1f}' y='{y_line+4}' text-anchor='middle' font-size='11' font-weight='600' fill='#ffffff'>{escape(lbl)}</text>")
        parts.append(f"<text x='{x0+box_w/2:.1f}' y='{y_line+29}' text-anchor='middle' font-size='9.5' fill='#64748b'>{st:,}–{en:,}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _parse_float_series(raw) -> list[float]:
    if raw is None:
        return []
    if isinstance(raw, (tuple, list)) or hasattr(raw, "tolist"):
        seq = list(raw)
    else:
        seq = str(raw).replace(";", ",").split(",")
    out = []
    for tok in seq:
        f = _fnum(tok)
        if f is not None:
            out.append(f)
    return out


def _mean_or_none(vals: list[float]) -> float | None:
    return (sum(vals) / len(vals)) if vals else None


def _coord_summary(event_type: str, geom: dict) -> tuple[str, int | None]:
    chrom = _clean_str(geom.get("chr") or geom.get("chrom") or geom.get("chromosome"))
    strand = _clean_str(geom.get("strand")) or "+"
    et = str(event_type or "SE").upper()
    s0, e0 = None, None
    if et == "SE":
        s0, e0 = _fnum(geom.get("exonStart_0base")), _fnum(geom.get("exonEnd"))
    elif et == "RI":
        u_e, d_s = _fnum(geom.get("upstreamEE")), _fnum(geom.get("downstreamES"))
        if u_e is not None and d_s is not None and d_s > u_e:
            s0, e0 = u_e, d_s
        else:
            s0, e0 = _fnum(geom.get("riExonStart_0base")), _fnum(geom.get("riExonEnd"))
    elif et == "MXE":
        s0, e0 = _fnum(geom.get("1stExonStart_0base")), _fnum(geom.get("1stExonEnd"))
    else:
        s0, e0 = _fnum(geom.get("longExonStart_0base")), _fnum(geom.get("longExonEnd"))
    if s0 is not None and e0 is not None and e0 > s0 and chrom:
        length = int(round(e0 - s0))
        return f"{chrom}:{int(round(s0)):,}–{int(round(e0)):,} ({strand})", length
    return (f"{chrom} ({strand})" if chrom else ""), None


def _build_target_view_model(root: Path, request: dict, target: dict, evidence: list, cards: list,
                             samples: pd.DataFrame, sashimi: list, status: str, *, asset_prefix: str = "") -> dict:
    root = Path(root)
    by_aid = _load_target_evidence_frames(root, evidence)
    label = _clean_str(target.get("gene_symbol") or target.get("gene_id") or target.get("event_id") or target["target_id"])
    target_cids = [str(c) for c in target.get("comparison_ids", []) if str(c)]
    proj_samples = request.get("project", {}).get("samples", [])
    samples_by_id = {str(s["sample_id"]): s for s in proj_samples}

    contrasts_meta = []
    contrasts_by_id = {}
    for c in request.get("project", {}).get("contrasts", []):
        cid = str(c.get("contrast_id", ""))
        if target_cids and cid not in target_cids:
            continue
        num = c.get("numerator") or {}
        den = c.get("denominator") or {}
        meta = {
            "id": cid,
            "label": str(c.get("label") or cid),
            "num_label": str(num.get("label") or "Numerator"),
            "den_label": str(den.get("label") or "Denominator"),
            "num_samples": [str(x) for x in num.get("sample_ids", [])],
            "den_samples": [str(x) for x in den.get("sample_ids", [])],
        }
        contrasts_meta.append(meta)
        contrasts_by_id[cid] = meta
    if not contrasts_meta:
        for cid in target_cids or [""]:
            contrasts_meta.append({"id": cid, "label": cid or "Cohort", "num_label": "Group 1", "den_label": "Group 2", "num_samples": [], "den_samples": []})
            contrasts_by_id[cid] = contrasts_meta[-1]

    # Biological condition order: baseline (denominator of first contrast) -> numerator -> any other study conditions
    cond_order: list[str] = []
    for c in contrasts_meta:
        for nm in (c["den_label"], c["num_label"]):
            if nm and nm not in cond_order:
                cond_order.append(nm)
    for s in proj_samples:
        cond = _clean_str(s.get("condition"))
        if cond and cond not in cond_order:
            cond_order.append(cond)

    # Host Gene Expression (Panel A)
    deg_by_comparison: dict[str, dict] = {}
    for df in by_aid.get("differential.deg", []):
        fc_c = next((c for c in ("log2fc", "log2FoldChange") if c in df.columns), None)
        q_c = next((c for c in ("deg_fdr", "padj", "fdr", "FDR") if c in df.columns), None)
        p_c = next((c for c in ("pvalue", "PValue") if c in df.columns), None)
        for _, r in df.iterrows():
            cid = _clean_str(r.get("comparison_id") or r.get("contrast_id"))
            if not cid and len(contrasts_meta) == 1:
                cid = contrasts_meta[0]["id"]
            if cid and cid not in deg_by_comparison:
                q_val = _fnum(r.get(q_c)) if q_c else None
                deg_by_comparison[cid] = {
                    "log2fc": _fnum(r.get(fc_c)) if fc_c else None,
                    "fdr": q_val,
                    "pvalue": _fnum(r.get(p_c)) if p_c else None,
                    "is_deg": bool(q_val is not None and q_val < 0.05),
                }

    expr_unit = "Normalized counts"
    expr_by_sid: dict[str, dict] = {}
    if samples is not None and not samples.empty and "measurement" in samples.columns:
        sub_ex = samples[samples["measurement"].astype(str).isin({"Normalized counts", "normalized_count", "TPM", "expression"})]
        if not sub_ex.empty:
            expr_unit = _clean_str(sub_ex.iloc[0].get("unit")) or "Normalized counts"
        for _, r in sub_ex.iterrows():
            sid = str(r.get("sample_id", ""))
            v = _fnum(r.get("value"))
            if sid and v is not None and sid not in expr_by_sid:
                expr_by_sid[sid] = {"sample_id": sid, "condition": _clean_str(r.get("condition")) or "unknown", "value": v}
    if not expr_by_sid:
        for df in by_aid.get("expression.normalised_counts", []):
            if not df.empty:
                r = df.iloc[0]
                for sid, sinfo in samples_by_id.items():
                    if sid in df.columns:
                        v = _fnum(r.get(sid))
                        if v is not None and sid not in expr_by_sid:
                            expr_by_sid[sid] = {"sample_id": sid, "condition": _clean_str(sinfo.get("condition")) or "unknown", "value": v}

    # Proteome & Motif annotation frames indexed cleanly
    prot_records: list[dict] = []
    for df in by_aid.get("prioritize.proteome_consequences", []):
        for _, r in df.iterrows():
            prot_records.append(portable(r.to_dict()))
    motif_records: list[dict] = []
    for aid_m in ("prioritize.splicetrace_targets", "prioritize.motif_consequences"):
        for df in by_aid.get(aid_m, []):
            for _, r in df.iterrows():
                motif_records.append(portable(r.to_dict()))

    # Group cards by canonical event_id (one entry per event, multi-contrast stats/sashimi keyed by cid)
    grouped_cards: dict[str, list[dict]] = {}
    for c in cards:
        geom = c.get("event_geometry") or {}
        has_chr = bool(_clean_str(geom.get("chr") or geom.get("chrom") or geom.get("chromosome")))
        has_das = any(_effective_analysis_id(ev) in {"differential.as", "differential.sc_as", "prioritize.integrative_targets"}
                      for ev in c.get("evidence", []))
        if has_chr or has_das:
            grouped_cards.setdefault(str(c["event_id"]), []).append(c)

    # Fallback if target had cards without DAS/chr tags
    if not grouped_cards and cards:
        for c in cards:
            grouped_cards.setdefault(str(c["event_id"]), []).append(c)

    sashimi_index = {(str(s.get("event_id", "")), str(s.get("comparison_id", ""))): s for s in (sashimi or [])}
    events_out = []
    for eid, ev_cards in grouped_cards.items():
        rep = max(ev_cards, key=lambda c: len(c.get("event_geometry") or {}))
        geom = rep.get("event_geometry") or {}
        etype = str(rep.get("event_type") or "SE")
        coord_label, exon_len = _coord_summary(etype, geom)
        stats_by_cid: dict[str, dict] = {}
        psi_by_sid: dict[str, dict] = {}
        fallback_prot: dict | None = None

        # Pre-seed cohort-wide PSI vector from sample_measurements if available
        if samples is not None and not samples.empty and "measurement" in samples.columns:
            sub_psi = samples[(samples["measurement"].astype(str) == "PSI") & (samples["event_id"].astype(str) == eid)]
            # Prefer rows from cohort matrices (comparison_id == "") first, then per-contrast rows
            sub_psi = sub_psi.assign(_prio=sub_psi["comparison_id"].fillna("").astype(str).ne("").astype(int)).sort_values("_prio")
            for _, r in sub_psi.iterrows():
                sid = str(r.get("sample_id", ""))
                v = _fnum(r.get("value"))
                if sid and v is not None and sid not in psi_by_sid:
                    psi_by_sid[sid] = {"sample_id": sid, "condition": _clean_str(r.get("condition")) or "unknown", "value": v}

        for c in ev_cards:
            cid = str(c.get("comparison_id", ""))
            ordered_evs = sorted(
                c.get("evidence", []),
                key=lambda e: 0 if _effective_analysis_id(e) in {"differential.as", "differential.sc_as"} else 1,
            )
            st_entry: dict | None = None
            for ev in ordered_evs:
                eff = _effective_analysis_id(ev)
                if eff not in {"differential.as", "differential.sc_as", "prioritize.integrative_targets"}:
                    continue
                row = ev.get("row", {})
                dp = _fnum(row.get("dpsi", row.get("dPSI", row.get("IncLevelDifference", row.get("raw_dpsi")))))
                qv = _fnum(row.get("fdr", row.get("FDR", row.get("padj"))))
                pv = _fnum(row.get("pvalue", row.get("PValue")))
                flipped = any(str(row.get(flag, "")).lower() in {"1", "true", "yes"}
                              for flag in ("splicing_direction_flipped", "reverse_direction"))
                ijc1 = _parse_float_series(row.get("IJC_SAMPLE_1"))
                sjc1 = _parse_float_series(row.get("SJC_SAMPLE_1"))
                ijc2 = _parse_float_series(row.get("IJC_SAMPLE_2"))
                sjc2 = _parse_float_series(row.get("SJC_SAMPLE_2"))
                p1_vals = _parse_float_series(row.get("IncLevel1"))
                p2_vals = _parse_float_series(row.get("IncLevel2"))
                if flipped:
                    ijc1, ijc2 = ijc2, ijc1
                    sjc1, sjc2 = sjc2, sjc1
                    p1_vals, p2_vals = p2_vals, p1_vals
                if st_entry is None:
                    st_entry = {
                        "dpsi": dp, "fdr": qv, "pvalue": pv,
                        "ijc_num": _mean_or_none(ijc1), "sjc_num": _mean_or_none(sjc1),
                        "ijc_den": _mean_or_none(ijc2), "sjc_den": _mean_or_none(sjc2),
                        "psi_num": _mean_or_none(p1_vals), "psi_den": _mean_or_none(p2_vals),
                    }
                else:
                    if st_entry["dpsi"] is None and dp is not None:
                        st_entry["dpsi"] = dp
                    if st_entry["fdr"] is None and qv is not None:
                        st_entry["fdr"] = qv
                    if st_entry["pvalue"] is None and pv is not None:
                        st_entry["pvalue"] = pv
                    if st_entry["ijc_num"] is None and ijc1:
                        st_entry["ijc_num"], st_entry["sjc_num"] = _mean_or_none(ijc1), _mean_or_none(sjc1)
                        st_entry["ijc_den"], st_entry["sjc_den"] = _mean_or_none(ijc2), _mean_or_none(sjc2)
                c_meta = contrasts_by_id.get(cid)
                if c_meta:
                    for vals, sids in ((p1_vals, c_meta["num_samples"]), (p2_vals, c_meta["den_samples"])):
                        if len(vals) == len(sids):
                            for sid, pv_s in zip(sids, vals):
                                if sid not in psi_by_sid:
                                    psi_by_sid[sid] = {
                                        "sample_id": sid,
                                        "condition": _clean_str(samples_by_id.get(sid, {}).get("condition")) or "unknown",
                                        "value": float(pv_s),
                                    }
                if fallback_prot is None and (row.get("nmd_switch_class") or row.get("disrupted_uniprot_domains") or row.get("consequence_summary")):
                    fallback_prot = {
                        "uniprot_acc": _clean_str(row.get("uniprot_acc")),
                        "frame_status": _clean_str(row.get("frame_status")),
                        "nmd_switch_class": _clean_str(row.get("nmd_switch_class")),
                        "aa_len_inclusion": _fnum(row.get("aa_len_inclusion")),
                        "aa_len_exclusion": _fnum(row.get("aa_len_exclusion")),
                        "delta_aa": _fnum(row.get("delta_aa")),
                        "ptc_to_last_ejc_nt": _fnum(row.get("ptc_to_last_ejc_nt")),
                        "disrupted_protein_features": _clean_str(row.get("disrupted_uniprot_domains")),
                        "functional_consequence_summary": _clean_str(row.get("consequence_summary")),
                    }
            if st_entry is not None and cid:
                stats_by_cid[cid] = st_entry

        # Sashimi per comparison
        sashimi_by_cid: dict[str, dict] = {}
        for cm in contrasts_meta:
            cid = cm["id"]
            s_rec = sashimi_index.get((eid, cid))
            c_match = next((c for c in ev_cards if str(c.get("comparison_id", "")) == cid), rep)
            st_c = stats_by_cid.get(cid, {})
            arm_stats = [
                (cm["num_label"], st_c.get("psi_num"), st_c.get("ijc_num"), st_c.get("sjc_num")),
                (cm["den_label"], st_c.get("psi_den"), st_c.get("ijc_den"), st_c.get("sjc_den")),
            ]
            views: dict[str, str] = {}
            downloads: list[dict] = []
            if s_rec and s_rec.get("files"):
                for frel in s_rec["files"]:
                    name = Path(frel).name
                    href = f"{asset_prefix}sashimi/{frel}"
                    if name == "native_summary.png":
                        views["summary"] = href
                    elif name == "native_replicates.png":
                        views["replicates"] = href
                    elif name == "native_summary_readable.png":
                        views["summary_readable"] = href
                    elif name == "native_replicates_readable.png":
                        views["replicates_readable"] = href
                    elif name == "native_rmats_evidence.png":
                        views["rmats_evidence"] = href
                    if name in {
                        "native_summary.svg", "native_summary.png", "native_summary.pdf",
                        "native_replicates.svg", "native_replicates.png", "native_replicates.pdf",
                        "native_sashimi_junction_counts.tsv", "native_sashimi_psi_summary.tsv",
                    }:
                        downloads.append({"label": name.replace("native_sashimi_", "").replace("native_", ""), "href": href})
            sashimi_by_cid[cid] = {
                "status": str(s_rec.get("status", "unavailable")) if s_rec else "unavailable",
                "reader": str(s_rec.get("reader") or "") if s_rec else "",
                "reason": str(s_rec.get("reason") or "") if s_rec else "",
                "views": views,
                "downloads": downloads,
                "vector_svg": "" if views else _junction_schematic_svg(c_match, arm_stats),
            }

        # Match Proteome & NMD consequence for this event
        matched_prot = next((r for r in prot_records if _clean_str(r.get("event_id")) == eid), None)
        if matched_prot is None and len(grouped_cards) == 1:
            sym_l = label.lower()
            matched_prot = next((r for r in prot_records if _clean_str(r.get("gene_symbol")).lower() == sym_l), None)
        if matched_prot is not None:
            prot_obj = {
                "uniprot_acc": _clean_str(matched_prot.get("uniprot_acc")),
                "frame_status": _clean_str(matched_prot.get("frame_status")),
                "nmd_switch_class": _clean_str(matched_prot.get("nmd_switch_class")),
                "aa_len_inclusion": _fnum(matched_prot.get("aa_len_inclusion")),
                "aa_len_exclusion": _fnum(matched_prot.get("aa_len_exclusion")),
                "delta_aa": _fnum(matched_prot.get("delta_aa")),
                "ptc_to_last_ejc_nt": _fnum(matched_prot.get("ptc_to_last_ejc_nt")),
                "disrupted_protein_features": _clean_str(matched_prot.get("disrupted_protein_features")),
                "functional_consequence_summary": _clean_str(matched_prot.get("functional_consequence_summary")),
            }
        else:
            prot_obj = fallback_prot

        # Match Cis-Motif hits (SpliceTrace) for this event (deduplicated)
        raw_motifs = [r for r in motif_records if _clean_str(r.get("event_id")) == eid]
        if not raw_motifs and len(grouped_cards) == 1:
            sym_l = label.lower()
            raw_motifs = [r for r in motif_records if _clean_str(r.get("gene_symbol")).lower() == sym_l]
        motifs_out = []
        seen_motifs = set()
        for mr in raw_motifs:
            rbp = _clean_str(mr.get("rbp_list") or mr.get("top_rbps") or mr.get("RBP") or mr.get("rbp"))
            reg = _clean_str(mr.get("region") or mr.get("instance_state") or "±200 nt flanking")
            rna = _clean_str(mr.get("hit_RNA") or mr.get("motif_seq"))
            dscore = _fnum(mr.get("delta_motif_score") or mr.get("delta_sequence_opportunity"))
            mk = (rbp, reg, rna)
            if rbp and mk not in seen_motifs:
                seen_motifs.add(mk)
                motifs_out.append({"rbp": rbp, "region": reg, "hit_rna": rna, "delta_score": dscore})

        max_dp = max((abs(s["dpsi"]) for s in stats_by_cid.values() if s.get("dpsi") is not None), default=-1.0)
        min_q = min((s["fdr"] for s in stats_by_cid.values() if s.get("fdr") is not None), default=2.0)
        events_out.append({
            "event_id": eid,
            "event_type": etype,
            "chrom": _clean_str(geom.get("chr") or geom.get("chrom") or geom.get("chromosome")),
            "strand": _clean_str(geom.get("strand")) or "+",
            "coord_label": coord_label or eid,
            "exon_len_nt": exon_len,
            "stats_by_comparison": stats_by_cid,
            "psi_samples": list(psi_by_sid.values()),
            "sashimi_by_comparison": sashimi_by_cid,
            "proteome": prot_obj,
            "motifs": motifs_out,
            "_sort_dp": max_dp,
            "_sort_q": min_q,
        })

    events_out.sort(key=lambda e: (-e["_sort_dp"], e["_sort_q"], e["event_id"]))
    for e in events_out:
        e.pop("_sort_dp", None)
        e.pop("_sort_q", None)

    # Downloadable evidence ledger links (omit network tables)
    ev_links = []
    for tbl in evidence:
        eff_aid = _effective_analysis_id(tbl)
        if eff_aid.startswith("programs.splicenet") or eff_aid == "prioritize.module_rbp_bridge":
            continue
        ev_links.append({
            "name": tbl.get("name", tbl.get("evidence_id", "")),
            "analysis_id": eff_aid,
            "rows": int(tbl.get("rows", 0)),
            "html": (asset_prefix + tbl["html"]) if tbl.get("html") else "",
            "parquet": (asset_prefix + tbl["parquet"]) if tbl.get("parquet") else "",
            "tsv": (asset_prefix + tbl["tsv"]) if tbl.get("tsv") else "",
        })

    return {
        "target_id": target["target_id"],
        "target_kind": target["target_kind"],
        "gene_id": _clean_str(target.get("gene_id")),
        "gene_symbol": label,
        "requested_event_id": _clean_str(target.get("event_id")),
        "status": status,
        "condition_order": cond_order,
        "comparisons": contrasts_meta,
        "deg_by_comparison": deg_by_comparison,
        "expression_unit": expr_unit,
        "expression_samples": list(expr_by_sid.values()),
        "events": events_out,
        "evidence_links": ev_links,
        "sample_measurements_href": f"{asset_prefix}sample-measurements.html",
        "provenance_href": f"{asset_prefix}target-provenance.json",
    }


DASHBOARD_CSS = """
*{box-sizing:border-box}
body{font:13.5px/1.55 'IBM Plex Sans','Inter','Segoe UI',system-ui,sans-serif;color:#1e293b;background:#f4f7fb;margin:0;font-variant-numeric:tabular-nums}
a{color:#2251ff;text-decoration:none}a:hover{text-decoration:underline}
.k-hdr{background:linear-gradient(125deg,#051c2c 0%,#0b2f4a 100%);color:#fff;padding:18px 24px;border-bottom:3px solid #2251ff}
.k-hdr-inner{max-width:1420px;margin:0 auto;display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap}
.k-eyebrow{font-size:10px;font-weight:700;letter-spacing:.12em;text-transform:uppercase;color:#93c5fd;margin-bottom:4px}
.k-hdr h1{margin:0;font:400 24px/1.15 'Newsreader',Georgia,serif;letter-spacing:-.015em}
.k-hdr-sub{margin:4px 0 0;color:#cbd5e1;font-size:12.5px}
.k-hdr-actions{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.k-btn{display:inline-flex;align-items:center;gap:5px;padding:6px 11px;font-size:12px;font-weight:600;border-radius:3px;border:1px solid rgba(255,255,255,.28);background:rgba(255,255,255,.1);color:#fff;cursor:pointer}
.k-btn:hover{background:rgba(255,255,255,.2);text-decoration:none}
.k-sticky{position:sticky;top:0;z-index:40;background:rgba(255,255,255,.97);backdrop-filter:blur(8px);border-bottom:1px solid #cbd5e1;box-shadow:0 2px 8px rgba(5,28,44,.06)}
.k-sticky-inner{max-width:1420px;margin:0 auto;padding:10px 24px;display:flex;flex-direction:column;gap:8px}
.k-tabs-row{display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.k-menu-label{font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;color:#64748b;margin-right:4px}
.k-tab{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;font-size:12.5px;font-weight:600;border:1px solid #cbd5e1;background:#f8fafc;color:#051c2c;border-radius:3px;cursor:pointer}
.k-tab:hover{border-color:#2251ff;color:#2251ff}
.k-tab.active{background:#051c2c;color:#fff;border-color:#051c2c}
.k-tab .k-kind{font-size:9.5px;font-weight:700;text-transform:uppercase;padding:1px 5px;border-radius:2px;background:#e2e8f0;color:#334155}
.k-tab.active .k-kind{background:#2251ff;color:#fff}
.k-ctrl-row{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.k-ctrl{display:inline-flex;align-items:center;gap:6px;font-size:11.5px;font-weight:600;color:#475569}
.k-ctrl select{padding:5px 9px;font-size:12.5px;font-weight:600;color:#051c2c;border:1px solid #cbd5e1;border-radius:3px;background:#fff;max-width:340px}
.k-main{max-width:1420px;margin:18px auto 36px;padding:0 24px}
.k-ribbon{background:#fff;border:1px solid #e2e8f0;border-left:4px solid #2251ff;border-radius:3px;padding:14px 18px;margin-bottom:16px;display:flex;justify-content:space-between;align-items:center;gap:14px;flex-wrap:wrap;box-shadow:0 1px 2px rgba(5,28,44,.04)}
.k-ribbon-title{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap}
.k-ribbon-title h2{margin:0;font:600 22px/1.2 'Newsreader',Georgia,serif;color:#051c2c}
.k-ribbon-sub{font-size:12px;color:#475569;margin-top:3px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.k-kpis{display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.k-kpi{background:#f8fafc;border:1px solid #e2e8f0;padding:7px 12px;border-radius:3px;min-width:128px}
.k-kpi span{display:block;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#64748b}
.k-kpi strong{font-size:15px;color:#051c2c}
.k-badge{display:inline-block;padding:2px 8px;font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;border-radius:2px;border:1px solid}
.k-badge-navy{background:#051c2c;color:#fff;border-color:#051c2c}
.k-badge-blue{background:#eff6ff;color:#1d4ed8;border-color:#bfdbfe}
.k-badge-teal{background:#f0fdfa;color:#0f766e;border-color:#99f6e4}
.k-badge-coral{background:#fff1f2;color:#b83a4b;border-color:#fecdd3}
.k-badge-slate{background:#f1f5f9;color:#475569;border-color:#cbd5e1}
.katana-grid-2x2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;align-items:stretch}
.katana-panel{background:#fff;border:1px solid #cbd5e1;border-top:3px solid #051c2c;border-radius:3px;padding:16px 18px;display:flex;flex-direction:column;gap:10px;box-shadow:0 1px 3px rgba(5,28,44,.04);min-width:0}
.k-panel-hdr{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;border-bottom:1px solid #e2e8f0;padding-bottom:8px;flex-wrap:wrap}
.k-panel-tag{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:#2251ff;display:block}
.k-panel-hdr h3{margin:2px 0 0;font-size:15px;font-weight:700;color:#051c2c}
.k-stat-row{display:flex;gap:10px;flex-wrap:wrap;align-items:center;font-size:12px;color:#334155}
.k-stat-chip{background:#f8fafc;border:1px solid #e2e8f0;padding:3px 8px;border-radius:2px}
.k-svg-wrap{background:#fcfdff;border:1px solid #e2e8f0;border-radius:3px;padding:6px}
.k-svg-wrap svg{width:100%;height:auto;display:block}
.k-seg{display:inline-flex;border:1px solid #cbd5e1;border-radius:3px;overflow:hidden;background:#f8fafc}
.k-seg button{border:0;border-right:1px solid #cbd5e1;background:transparent;padding:4px 9px;font-size:11px;font-weight:600;color:#475569;cursor:pointer}
.k-seg button:last-child{border-right:0}
.k-seg button.active{background:#051c2c;color:#fff}
.k-sashimi-frame{border:1px solid #e2e8f0;border-radius:3px;background:#fff;padding:6px;text-align:center}
.k-sashimi-img{width:100%;max-height:360px;object-fit:contain;cursor:zoom-in;display:block;margin:0 auto}
.k-dl-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;font-size:11.5px;color:#64748b;padding-top:4px;border-top:1px dashed #e2e8f0}
.k-dl-row a{padding:2px 7px;background:#f8fafc;border:1px solid #cbd5e1;border-radius:2px;font-size:11px;font-weight:600;color:#051c2c}
.k-conseq-box{background:#f8fafc;border-left:3px solid #051c2c;padding:9px 12px;font-size:12px;color:#1e293b;border-radius:2px}
.k-conseq-box.nmd{background:#fff8f8;border-left-color:#b83a4b}
.k-motif-list{display:flex;flex-wrap:wrap;gap:8px}
.k-motif-chip{background:#f8fafc;border:1px solid #cbd5e1;border-radius:3px;padding:6px 10px;font-size:11.5px;line-height:1.4}
.k-motif-chip strong{color:#051c2c}
.k-motif-chip code{background:#eff6ff;color:#1d4ed8;padding:1px 4px;border-radius:2px;font-weight:700}
.k-drawer{background:#fff;border:1px solid #e2e8f0;border-radius:3px;padding:12px 18px;margin-top:16px}
.k-drawer summary{cursor:pointer;font-weight:700;font-size:13.5px;color:#051c2c;list-style:none}
.k-drawer summary::-webkit-details-marker{display:none}
.k-drawer[open] summary{margin-bottom:12px;border-bottom:1px solid #e2e8f0;padding-bottom:8px}
.k-table{width:100%;border-collapse:collapse;font-size:12px}
.k-table th,.k-table td{padding:7px 10px 7px 0;border-bottom:1px solid #e2e8f0;text-align:left;vertical-align:top}
.k-table th{font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:#64748b;border-top:1px solid #051c2c}
.k-row-btn{background:none;border:0;padding:0;color:#2251ff;font-weight:700;cursor:pointer;text-align:left}
.k-row-btn:hover{text-decoration:underline}
.k-modal{position:fixed;inset:0;background:rgba(5,28,44,.86);z-index:100;display:none;align-items:center;justify-content:center;padding:24px}
.k-modal.open{display:flex}
.k-modal-inner{background:#fff;border-radius:4px;max-width:96vw;max-height:94vh;padding:12px;display:flex;flex-direction:column;gap:8px}
.k-modal-inner img{max-width:92vw;max-height:82vh;object-fit:contain}
.k-empty{font-size:12px;color:#64748b;padding:14px;background:#f8fafc;border:1px dashed #cbd5e1;border-radius:3px}
@media (max-width:860px){.katana-grid-2x2{grid-template-columns:1fr}}
@media print{body{background:#fff}.no-print{display:none!important}.k-sticky{position:static;box-shadow:none}.katana-grid-2x2{grid-template-columns:repeat(2,minmax(0,1fr))!important;gap:12px}.katana-panel{break-inside:avoid}}
"""


DASHBOARD_JS = r"""(() => {
  'use strict';
  const DATA = JSON.parse(document.getElementById('katana-batch-data').textContent);
  const targets = DATA.targets || [];
  if (!targets.length) return;

  const esc = s => String(s ?? '').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
  const fmtNum = (v, d = 3) => (v == null || Number.isNaN(Number(v))) ? 'NA' : Number(v).toFixed(d);
  const fmtSigned = (v, d = 3) => {
    if (v == null || Number.isNaN(Number(v))) return 'NA';
    const n = Number(v);
    return (n >= 0 ? '+' : '') + n.toFixed(d);
  };
  const fmtP = v => {
    if (v == null || Number.isNaN(Number(v))) return 'NA';
    const n = Number(v);
    if (n <= 0) return '< 1e-16';
    return n < 1e-3 ? n.toExponential(2) : n.toFixed(4);
  };

  const state = {
    tid: targets[0].target_id,
    cid: (targets[0].comparisons?.[0]?.id) || '',
    eid: (targets[0].events?.[0]?.event_id) || '',
    sashView: 'summary',
  };

  function getTarget() {
    return targets.find(t => t.target_id === state.tid) || targets[0];
  }
  function getComparisonMeta(t) {
    const comps = t.comparisons || [];
    return comps.find(c => c.id === state.cid) || comps[0] || {id: '', label: 'Cohort', num_label: 'Group 1', den_label: 'Group 2'};
  }
  function getEvent(t) {
    const evs = t.events || [];
    if (!evs.length) return null;
    if (t.target_kind === 'event' && t.requested_event_id) {
      const m = evs.find(e => e.event_id === t.requested_event_id);
      if (m) return m;
    }
    return evs.find(e => e.event_id === state.eid) || evs[0];
  }

  function selectTarget(newTid) {
    const t = targets.find(x => x.target_id === newTid) || targets[0];
    state.tid = t.target_id;
    if (!t.comparisons?.some(c => c.id === state.cid)) {
      state.cid = t.comparisons?.[0]?.id || '';
    }
    state.eid = (t.target_kind === 'event' && t.requested_event_id)
      ? t.requested_event_id
      : (t.events?.[0]?.event_id || '');
    render();
  }

  function renderStripBoxSvg(samples, condOrder, cMeta, unit, zeroToOne) {
    if (!samples || !samples.length) {
      return '<div class="k-empty">Per-sample values were not exported for this locus.</div>';
    }
    const groups = [];
    for (const c of (condOrder || [])) {
      if (samples.some(s => s.condition === c) && !groups.includes(c)) groups.push(c);
    }
    for (const s of samples) {
      if (s.condition && !groups.includes(s.condition)) groups.push(s.condition);
    }
    if (!groups.length) groups.push('All');

    const vals = samples.map(s => Number(s.value)).filter(Number.isFinite);
    const yMin = zeroToOne ? 0 : Math.min(0, ...vals);
    const yMax = zeroToOne ? 1 : (Math.max(...vals) * 1.15 || 1);
    const W = 540, H = 228, pL = 58, pR = 18, pT = 18, pB = 40;
    const pw = W - pL - pR, ph = H - pT - pB;
    const yMap = v => pT + (1 - (v - yMin) / Math.max(1e-9, yMax - yMin)) * ph;

    const colorFor = grp => {
      if (grp === cMeta.num_label) return '#051C2C';
      if (grp === cMeta.den_label) return '#2251FF';
      return '#64748B';
    };

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(unit)} by condition">`;
    svg += `<rect x="${pL}" y="${pT}" width="${pw}" height="${ph}" fill="#F8FAFC" stroke="#E2E8F0"/>`;
    [0, 0.25, 0.5, 0.75, 1].forEach(f => {
      const v = yMin + f * (yMax - yMin);
      const y = yMap(v);
      const lbl = zeroToOne ? v.toFixed(2) : (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0));
      svg += `<line x1="${pL}" y1="${y}" x2="${pL + pw}" y2="${y}" stroke="#E2E8F0" stroke-width="0.9"/>`;
      svg += `<text x="${pL - 7}" y="${y + 3.5}" text-anchor="end" font-size="9.5" fill="#64748B">${lbl}</text>`;
    });

    groups.forEach((grp, gi) => {
      const cx = pL + ((gi + 0.5) / groups.length) * pw;
      const gpts = samples.filter(s => (s.condition || 'All') === grp);
      const gvals = gpts.map(s => Number(s.value)).filter(Number.isFinite).sort((a, b) => a - b);
      if (!gvals.length) return;
      const mean = gvals.reduce((a, b) => a + b, 0) / gvals.length;
      const q1 = gvals[Math.floor(gvals.length * 0.25)] ?? mean;
      const q3 = gvals[Math.floor(gvals.length * 0.75)] ?? mean;
      const col = colorFor(grp);
      const isArm = grp === cMeta.num_label || grp === cMeta.den_label;

      if (isArm) {
        svg += `<rect x="${cx - 40}" y="${pT + 2}" width="80" height="${ph - 4}" fill="${col}" fill-opacity="0.03" rx="2"/>`;
      }
      if (gvals.length >= 2) {
        const yQ1 = yMap(q1), yQ3 = yMap(q3);
        svg += `<rect x="${cx - 22}" y="${Math.min(yQ1, yQ3)}" width="44" height="${Math.max(3, Math.abs(yQ3 - yQ1))}" fill="${col}" fill-opacity="0.16" stroke="${col}" stroke-width="1.2" rx="1"/>`;
      }
      svg += `<line x1="${cx - 26}" y1="${yMap(mean)}" x2="${cx + 26}" y2="${yMap(mean)}" stroke="${col}" stroke-width="2.6"/>`;
      gpts.slice(0, 60).forEach((pt, pi) => {
        const jit = ((pi % 5) - 2) * 5.2;
        const valTxt = zeroToOne ? Number(pt.value).toFixed(3) : Number(pt.value).toFixed(1);
        svg += `<circle cx="${cx + jit}" cy="${yMap(Number(pt.value))}" r="4.5" fill="${col}" stroke="#FFFFFF" stroke-width="1"><title>${esc(pt.sample_id)} (${esc(grp)}): ${valTxt}</title></circle>`;
      });
      const roleTag = grp === cMeta.num_label ? ' [Num]' : (grp === cMeta.den_label ? ' [Den]' : '');
      const lbl = `${grp}${roleTag} (n=${gpts.length})`;
      svg += `<text x="${cx}" y="${H - 13}" text-anchor="middle" font-size="10.2" font-weight="${isArm ? 700 : 600}" fill="${col}">${esc(lbl)}</text>`;
    });

    svg += `<text x="14" y="${pT + ph / 2}" text-anchor="middle" font-size="10" font-weight="600" fill="#051C2C" transform="rotate(-90 14 ${pT + ph / 2})">${esc(unit)}</text>`;
    svg += '</svg>';
    return svg;
  }

  function renderProteomeSvg(prot) {
    if (!prot || (prot.aa_len_inclusion == null && prot.aa_len_exclusion == null)) {
      return '<div class="k-empty">Coding isoform ORF lengths were not declared for this event.</div>';
    }
    const incAa = Math.max(20, Number(prot.aa_len_inclusion ?? prot.aa_len_exclusion ?? 100));
    const excAa = Math.max(20, Number(prot.aa_len_exclusion ?? prot.aa_len_inclusion ?? 100));
    const maxAa = Math.max(100, incAa, excAa);
    const nmdCls = String(prot.nmd_switch_class || '');
    const nmdLow = nmdCls.toLowerCase();
    const incNmd = nmdLow.includes('inclusion') && nmdLow.includes('nmd') || nmdLow.includes('poison');
    const excNmd = (nmdLow.includes('skip') || nmdLow.includes('exclusion')) && nmdLow.includes('nmd');
    const ptcDist = prot.ptc_to_last_ejc_nt != null ? Math.round(Number(prot.ptc_to_last_ejc_nt)) : null;
    const domText = String(prot.disrupted_protein_features || '').split(/[;|]/).map(s => s.trim()).filter(Boolean)[0] || '';
    const domShort = domText.replace(/^DOMAIN:/i, '').slice(0, 24);

    const W = 560, H = 136, x0 = 132, maxW = 240;
    const wInc = Math.max(38, (incAa / maxAa) * maxW);
    const wExc = Math.max(38, (excAa / maxAa) * maxW);
    const deltaAa = prot.delta_aa != null ? Math.round(Number(prot.delta_aa)) : Math.round(incAa - excAa);

    let svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Isoform proteome and 50-nt EJC NMD architecture">`;
    svg += `<text x="10" y="20" font-size="10.5" font-weight="700" fill="#051C2C">Mature mRNA ORF Architecture${prot.uniprot_acc ? ` · UniProt ${esc(prot.uniprot_acc)}` : ''}</text>`;

    // Inclusion Isoform
    svg += `<text x="10" y="52" font-size="10.5" font-weight="700" fill="#051C2C">Inclusion Isoform</text>`;
    svg += `<text x="10" y="65" font-size="9" fill="#64748B">${Math.round(incAa)} aa · ${esc(prot.frame_status || 'coding')}</text>`;
    svg += `<rect x="${x0}" y="40" width="${wInc}" height="22" rx="2" fill="${incNmd ? '#FEE2E2' : '#DBEAFE'}" stroke="${incNmd ? '#B83A4B' : '#051C2C'}" stroke-width="1.2"/>`;
    if (domShort && !incNmd) {
      const dw = Math.min(118, wInc * 0.48);
      const dx = x0 + Math.max(4, (wInc - dw) / 2);
      svg += `<rect x="${dx}" y="43" width="${dw}" height="16" rx="2" fill="#2251FF"/>`;
      svg += `<text x="${dx + dw / 2}" y="54" text-anchor="middle" font-size="8.2" font-weight="700" fill="#FFFFFF">${esc(domShort.slice(0, 18))}</text>`;
    }
    if (incNmd) {
      svg += `<polygon points="${x0 + wInc},34 ${x0 + wInc - 5},25 ${x0 + wInc + 5},25" fill="#B83A4B"/>`;
      const ptcTxt = ptcDist ? `PTC ${ptcDist} nt upstream of last EJC → NMD` : 'PTC ≥ 50 nt → NMD Target';
      svg += `<text x="${x0 + wInc + 9}" y="55" font-size="9.5" font-weight="700" fill="#B83A4B">${esc(ptcTxt)}</text>`;
    } else {
      svg += `<text x="${x0 + wInc + 9}" y="55" font-size="9.5" font-weight="600" fill="#051C2C">${Math.round(incAa)} aa intact</text>`;
    }

    // Skipping Isoform
    svg += `<text x="10" y="98" font-size="10.5" font-weight="700" fill="#051C2C">Skipping Isoform</text>`;
    svg += `<text x="10" y="111" font-size="9" fill="#64748B">${Math.round(excAa)} aa (Δ = ${deltaAa >= 0 ? '+' : ''}${deltaAa} aa)</text>`;
    svg += `<rect x="${x0}" y="86" width="${wExc}" height="22" rx="2" fill="${excNmd ? '#FEE2E2' : '#E2E8F0'}" stroke="${excNmd ? '#B83A4B' : '#334155'}" stroke-width="1.2"/>`;
    if (domShort && incNmd) {
      const dw = Math.min(126, wExc * 0.52);
      const dx = x0 + Math.max(4, (wExc - dw) / 2);
      svg += `<rect x="${dx}" y="89" width="${dw}" height="16" rx="2" fill="#2251FF"/>`;
      svg += `<text x="${dx + dw / 2}" y="100" text-anchor="middle" font-size="8.2" font-weight="700" fill="#FFFFFF">${esc(domShort.slice(0, 18))}</text>`;
    }
    if (excNmd) {
      svg += `<polygon points="${x0 + wExc},80 ${x0 + wExc - 5},71 ${x0 + wExc + 5},71" fill="#B83A4B"/>`;
      svg += `<text x="${x0 + wExc + 9}" y="101" font-size="9.5" font-weight="700" fill="#B83A4B">PTC ≥ 50 nt → NMD Target</text>`;
    } else {
      const domSuffix = domShort ? ` · ${domShort}` : '';
      svg += `<text x="${x0 + wExc + 9}" y="101" font-size="9.5" font-weight="600" fill="#334155">${Math.round(excAa)} aa${esc(domSuffix.slice(0, 26))}</text>`;
    }
    svg += '</svg>';
    return svg;
  }

  function renderMenu(t, ev) {
    const tabsEl = document.getElementById('target-tabs');
    if (tabsEl) {
      tabsEl.innerHTML = targets.map(x => {
        const act = x.target_id === t.target_id ? 'active' : '';
        const kLbl = x.target_kind === 'gene' ? 'Gene' : 'Event';
        return `<button type="button" class="k-tab ${act}" data-tid="${esc(x.target_id)}">${esc(x.gene_symbol)} <span class="k-kind">${kLbl}</span></button>`;
      }).join('');
      tabsEl.querySelectorAll('.k-tab').forEach(b => b.addEventListener('click', () => selectTarget(b.dataset.tid)));
    }

    const tSel = document.getElementById('target-select');
    if (tSel) {
      tSel.innerHTML = targets.map(x => {
        const tag = x.target_kind === 'gene' ? 'GeneCard' : 'EventCard';
        return `<option value="${esc(x.target_id)}" ${x.target_id === t.target_id ? 'selected' : ''}>${esc(x.gene_symbol)} (${tag}${x.gene_id ? ' · ' + esc(x.gene_id) : ''})</option>`;
      }).join('');
    }

    const cSel = document.getElementById('comparison-select');
    if (cSel) {
      cSel.innerHTML = (t.comparisons || []).map(c =>
        `<option value="${esc(c.id)}" ${c.id === state.cid ? 'selected' : ''}>${esc(c.label)}</option>`
      ).join('');
    }

    const eWrap = document.getElementById('event-select-wrap');
    const eSel = document.getElementById('event-select');
    const evs = t.events || [];
    if (eWrap && eSel) {
      if (t.target_kind === 'gene' && evs.length > 1) {
        eWrap.style.display = 'inline-flex';
        eSel.innerHTML = evs.map((item, idx) => {
          const st = item.stats_by_comparison?.[state.cid] || {};
          const dpTxt = st.dpsi != null ? `ΔPSI ${fmtSigned(st.dpsi)}` : '';
          return `<option value="${esc(item.event_id)}" ${ev && item.event_id === ev.event_id ? 'selected' : ''}>Event ${idx + 1}/${evs.length}: ${esc(item.event_type)} ${esc(item.coord_label)} ${dpTxt ? '(' + dpTxt + ')' : ''}</option>`;
        }).join('');
      } else {
        eWrap.style.display = 'none';
      }
    }
  }

  function render() {
    const t = getTarget();
    const cMeta = getComparisonMeta(t);
    state.cid = cMeta.id;
    const ev = getEvent(t);
    if (ev) state.eid = ev.event_id;
    renderMenu(t, ev);

    const deg = t.deg_by_comparison?.[cMeta.id] || null;
    const st = ev ? (ev.stats_by_comparison?.[cMeta.id] || {}) : {};
    const prot = ev ? ev.proteome : null;
    const motifs = ev ? (ev.motifs || []) : [];
    const sash = ev ? (ev.sashimi_by_comparison?.[cMeta.id] || {status: 'unavailable', views: {}, downloads: []}) : {status: 'unavailable', views: {}, downloads: []};

    // Ribbon KPIs
    const degBadge = deg
      ? (deg.is_deg ? '<span class="k-badge k-badge-coral">Dual DEG + DAS</span>' : '<span class="k-badge k-badge-teal">Splicing-Only (Non-DEG)</span>')
      : '<span class="k-badge k-badge-slate">Expression Summary</span>';
    const nmdCls = prot?.nmd_switch_class || 'Analyzed';
    const nmdIsAlert = nmdCls.toLowerCase().includes('nmd');

    document.getElementById('target-ribbon').innerHTML = `
      <div>
        <div class="k-ribbon-title">
          <h2>${esc(t.gene_symbol)}</h2>
          <span class="k-badge k-badge-navy">${t.target_kind === 'gene' ? 'GeneCard' : 'EventCard'}</span>
          ${degBadge}
          <span class="k-badge ${t.status === 'complete' ? 'k-badge-teal' : 'k-badge-slate'}">${esc(String(t.status || '').toUpperCase())}</span>
        </div>
        <div class="k-ribbon-sub">
          <span>${esc(t.gene_id || '')}</span>
          ${ev ? `<span>· Locus: <code>${esc(ev.coord_label)}</code>${ev.exon_len_nt ? ` · ${ev.exon_len_nt} nt` : ''}</span>` : ''}
          <span>· Contrast: <strong>${esc(cMeta.label)}</strong></span>
        </div>
      </div>
      <div class="k-kpis">
        <div class="k-kpi"><span>Host DEG log2FC</span><strong>${deg ? fmtSigned(deg.log2fc, 2) : 'NA'}</strong></div>
        <div class="k-kpi"><span>Host DEG FDR</span><strong>${deg ? fmtP(deg.fdr) : 'NA'}</strong></div>
        <div class="k-kpi"><span>Splicing ΔPSI</span><strong>${fmtSigned(st.dpsi, 3)}</strong></div>
        <div class="k-kpi"><span>Splicing FDR</span><strong>${fmtP(st.fdr)}</strong></div>
        <div class="k-kpi"><span>Isoform Fate</span><strong>${esc(nmdCls)}</strong></div>
      </div>
    `;

    // 2x2 Panel A: Host Gene Expression (DEG)
    const panelAHtml = `
      <article class="katana-panel" data-panel="A">
        <div class="k-panel-hdr">
          <div>
            <span class="k-panel-tag">Panel A · Differential Expression</span>
            <h3>Host Gene RNA Expression (${esc(t.gene_symbol)})</h3>
          </div>
          ${degBadge}
        </div>
        <div class="k-stat-row">
          <span class="k-stat-chip"><strong>Comparison:</strong> ${esc(cMeta.label)}</span>
          <span class="k-stat-chip"><strong>log2FC:</strong> ${deg ? fmtSigned(deg.log2fc, 3) : 'NA'}</span>
          <span class="k-stat-chip"><strong>FDR:</strong> ${deg ? fmtP(deg.fdr) : 'NA'}</span>
          <span class="k-stat-chip"><strong>P:</strong> ${deg ? fmtP(deg.pvalue) : 'NA'}</span>
        </div>
        <div class="k-svg-wrap">${renderStripBoxSvg(t.expression_samples, t.condition_order, cMeta, t.expression_unit || 'Normalized counts', false)}</div>
      </article>
    `;

    // 2x2 Panel B: Alternative Splicing PSI
    const ijcSjcTxt = (st.ijc_num != null && st.sjc_num != null && st.ijc_den != null && st.sjc_den != null)
      ? `${esc(cMeta.num_label)}: IJC ${Math.round(st.ijc_num)} / SJC ${Math.round(st.sjc_num)} · ${esc(cMeta.den_label)}: IJC ${Math.round(st.ijc_den)} / SJC ${Math.round(st.sjc_den)}`
      : 'Saved replicate PSI distributions';
    const panelBHtml = `
      <article class="katana-panel" data-panel="B">
        <div class="k-panel-hdr">
          <div>
            <span class="k-panel-tag">Panel B · Differential Splicing</span>
            <h3>Splicing Inclusion PSI (${ev ? esc(ev.event_type) : 'Event'})</h3>
          </div>
          <span class="k-badge k-badge-blue">${ev ? esc(ev.coord_label) : 'No event'}</span>
        </div>
        <div class="k-stat-row">
          <span class="k-stat-chip"><strong>ΔPSI:</strong> ${fmtSigned(st.dpsi, 3)}</span>
          <span class="k-stat-chip"><strong>FDR:</strong> ${fmtP(st.fdr)}</span>
          <span class="k-stat-chip"><strong>P:</strong> ${fmtP(st.pvalue)}</span>
          <span class="k-stat-chip">${ijcSjcTxt}</span>
        </div>
        <div class="k-svg-wrap">${renderStripBoxSvg(ev ? ev.psi_samples : [], t.condition_order, cMeta, 'Inclusion Level (PSI)', true)}</div>
      </article>
    `;

    // 2x2 Panel C: Native BAM RPKM Sashimi (One selected view + toggle)
    const viewOrder = [
      ['summary', 'Summary'],
      ['replicates', 'Replicates'],
      ['summary_readable', 'Readable'],
      ['rmats_evidence', 'rMATS'],
    ];
    const availViews = viewOrder.filter(([k]) => sash.views && sash.views[k]);
    if (availViews.length && !sash.views[state.sashView]) {
      state.sashView = availViews[0][0];
    }
    const activeSrc = (sash.views && sash.views[state.sashView]) || '';
    const toggleHtml = availViews.length
      ? `<div class="k-seg no-print">${availViews.map(([k, lbl]) => `<button type="button" class="k-sash-btn ${state.sashView === k ? 'active' : ''}" data-view="${k}">${lbl}</button>`).join('')}</div>`
      : '';
    const sashMediaHtml = activeSrc
      ? `<div class="k-sashimi-frame"><img id="k-sashimi-img" class="k-sashimi-img" src="${esc(activeSrc)}" alt="Native BAM RPKM Sashimi for ${esc(t.gene_symbol)} ${esc(cMeta.label)}"></div>`
      : (sash.vector_svg
          ? `<div class="k-svg-wrap">${sash.vector_svg}<p class="muted" style="margin:6px 0 0">${esc(sash.reason || 'Server BAM Sashimi not attached; showing exon-junction schematic.')}</p></div>`
          : `<div class="k-empty">${esc(sash.reason || 'Native BAM Sashimi not generated for this selection.')}</div>`);
    const dlHtml = (sash.downloads && sash.downloads.length)
      ? `<div class="k-dl-row no-print"><span>Files:</span>${sash.downloads.map(d => `<a href="${esc(d.href)}" target="_blank" rel="noopener">${esc(d.label)}</a>`).join('')}</div>`
      : '';
    const panelCHtml = `
      <article class="katana-panel" data-panel="C">
        <div class="k-panel-hdr">
          <div>
            <span class="k-panel-tag">Panel C · BAM Alignment &amp; Junction Coverage</span>
            <h3>Native BAM RPKM Sashimi (${esc(cMeta.label)})</h3>
          </div>
          ${toggleHtml}
        </div>
        <div class="k-stat-row">
          <span class="k-stat-chip"><strong>Status:</strong> ${esc(String(sash.status || 'unavailable').toUpperCase())}</span>
          ${sash.reader ? `<span class="k-stat-chip"><strong>BAM Reader:</strong> ${esc(sash.reader)}</span>` : ''}
          ${activeSrc ? '<span class="k-stat-chip no-print">Click plot to zoom</span>' : ''}
        </div>
        ${sashMediaHtml}
        ${dlHtml}
      </article>
    `;

    // 2x2 Panel D: Isoform Proteome / 50-nt EJC NMD & RBP Cis-Motifs
    const motifHtml = motifs.length
      ? `<div class="k-motif-list">${motifs.map(m => `
          <div class="k-motif-chip">
            <strong>${esc(m.rbp)}</strong>${m.hit_rna ? ` · <code>${esc(m.hit_rna)}</code>` : ''}<br>
            <span class="muted">${esc(m.region)}${m.delta_score != null ? ` · ΔScore ${fmtSigned(m.delta_score, 2)}` : ''}</span>
          </div>`).join('')}</div>`
      : '<div class="k-empty">No cis-motif hits recorded for this event.</div>';
    const panelDHtml = `
      <article class="katana-panel" data-panel="D">
        <div class="k-panel-hdr">
          <div>
            <span class="k-panel-tag">Panel D · Isoform Fate &amp; RBP Cis-Regulation</span>
            <h3>Proteome Architecture, 50-nt EJC NMD &amp; Cis-Motifs</h3>
          </div>
          <span class="k-badge ${nmdIsAlert ? 'k-badge-coral' : 'k-badge-blue'}">${esc(nmdCls)}</span>
        </div>
        <div class="k-svg-wrap">${renderProteomeSvg(prot)}</div>
        ${prot?.functional_consequence_summary ? `<div class="k-conseq-box ${nmdIsAlert ? 'nmd' : ''}"><strong>Functional Consequence:</strong> ${esc(prot.functional_consequence_summary)}</div>` : ''}
        <div>
          <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:#475569;margin-bottom:6px">Flanking RBP Cis-Motif Regulatory Signals</div>
          ${motifHtml}
        </div>
      </article>
    `;

    document.getElementById('grid-2x2').innerHTML = panelAHtml + panelBHtml + panelCHtml + panelDHtml;

    // Bind Panel C view toggles and click-to-zoom modal
    document.querySelectorAll('.k-sash-btn').forEach(btn => btn.addEventListener('click', () => {
      state.sashView = btn.dataset.view;
      render();
    }));
    const sImg = document.getElementById('k-sashimi-img');
    if (sImg) {
      sImg.addEventListener('click', () => {
        const modal = document.getElementById('k-modal');
        const mImg = document.getElementById('k-modal-img');
        const mCap = document.getElementById('k-modal-cap');
        if (modal && mImg) {
          mImg.src = sImg.src;
          if (mCap) mCap.textContent = `${t.gene_symbol} · ${ev ? ev.coord_label : ''} · ${cMeta.label} (${state.sashView})`;
          modal.classList.add('open');
        }
      });
    }

    // Overview Drawer (switch any target directly)
    const ovBody = document.getElementById('overview-tbody');
    if (ovBody) {
      ovBody.innerHTML = targets.map(item => {
        const ev0 = item.events?.[0];
        const scope = item.target_kind === 'event' && ev0
          ? ev0.coord_label
          : (item.events?.length ? `${item.events.length} measured event(s)${ev0 ? ' · Top: ' + ev0.coord_label : ''}` : 'No event rows');
        const st0 = ev0?.stats_by_comparison?.[state.cid] || ev0?.stats_by_comparison?.[item.comparisons?.[0]?.id] || {};
        const deg0 = item.deg_by_comparison?.[state.cid] || item.deg_by_comparison?.[item.comparisons?.[0]?.id] || {};
        return `<tr>
          <td><button type="button" class="k-row-btn" data-jump="${esc(item.target_id)}">${esc(item.gene_symbol)}</button><br><span class="muted">${esc(item.gene_id || '')}</span></td>
          <td><span class="k-badge k-badge-slate">${esc(item.target_kind)}</span></td>
          <td><code>${esc(scope)}</code></td>
          <td>${deg0.log2fc != null ? fmtSigned(deg0.log2fc, 2) : 'NA'} <span class="muted">(${fmtP(deg0.fdr)})</span></td>
          <td>${st0.dpsi != null ? fmtSigned(st0.dpsi, 3) : 'NA'} <span class="muted">(${fmtP(st0.fdr)})</span></td>
          <td><span class="k-badge ${item.status === 'complete' ? 'k-badge-teal' : 'k-badge-slate'}">${esc(String(item.status || '').toUpperCase())}</span></td>
        </tr>`;
      }).join('');
      ovBody.querySelectorAll('.k-row-btn').forEach(b => b.addEventListener('click', () => selectTarget(b.dataset.jump)));
    }

    // Active target evidence downloads drawer
    const evDrawer = document.getElementById('evidence-links-box');
    if (evDrawer) {
      const links = t.evidence_links || [];
      evDrawer.innerHTML = `
        <p class="muted" style="margin-top:0">
          Raw BAM alignments remain external on the execution server.
          <a href="${esc(t.sample_measurements_href)}" target="_blank" rel="noopener">Sample Measurements Table ↗</a> ·
          <a href="${esc(t.provenance_href)}" target="_blank" rel="noopener">Target Provenance JSON ↗</a>
        </p>
        <div class="scroll"><table class="k-table"><thead><tr><th>Evidence Table</th><th>Source Analysis</th><th>Rows</th><th>Files</th></tr></thead><tbody>
          ${links.map(l => `<tr>
            <td>${esc(l.name)}</td>
            <td><code>${esc(l.analysis_id)}</code></td>
            <td>${Number(l.rows).toLocaleString()}</td>
            <td>${[l.html ? `<a href="${esc(l.html)}" target="_blank" rel="noopener">HTML</a>` : '', l.tsv ? `<a href="${esc(l.tsv)}" target="_blank" rel="noopener">TSV</a>` : '', l.parquet ? `<a href="${esc(l.parquet)}" target="_blank" rel="noopener">Parquet</a>` : ''].filter(Boolean).join(' · ')}</td>
          </tr>`).join('')}
        </tbody></table></div>
      `;
    }
  }

  document.getElementById('target-select')?.addEventListener('change', e => selectTarget(e.target.value));
  document.getElementById('comparison-select')?.addEventListener('change', e => {
    state.cid = e.target.value;
    render();
  });
  document.getElementById('event-select')?.addEventListener('change', e => {
    state.eid = e.target.value;
    render();
  });
  document.getElementById('k-modal-close')?.addEventListener('click', () => document.getElementById('k-modal')?.classList.remove('open'));
  document.getElementById('k-modal')?.addEventListener('click', e => {
    if (e.target.id === 'k-modal') e.target.classList.remove('open');
  });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') document.getElementById('k-modal')?.classList.remove('open');
  });

  render();
})();"""


def _render_unified_2x2_html(batch_payload: dict, *, page_title: str, show_batch_links: bool = False,
                             has_tsv: bool = False, has_zip: bool = False) -> str:
    study_name = str(batch_payload.get("project_name") or batch_payload.get("project_id") or "Katana Study")
    assay = str(batch_payload.get("assay") or "bulk")
    n_targets = len(batch_payload.get("targets", []))

    hdr_links = []
    if show_batch_links:
        if has_tsv:
            hdr_links.append("<a class='k-btn' href='targets.tsv' target='_blank' rel='noopener'>targets.tsv</a>")
        if has_zip:
            hdr_links.append("<a class='k-btn' href='REPORT.zip'>REPORT.zip</a>")
        hdr_links.append("<a class='k-btn' href='result-manifest.json' target='_blank' rel='noopener'>Manifest JSON</a>")
    else:
        hdr_links.append("<a class='k-btn' href='../../../REPORT.html'>← All Selected Targets</a>")
    hdr_links.append("<button type='button' class='k-btn' onclick='window.print()'>Print / PDF</button>")

    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{escape(page_title)}</title><style>{DASHBOARD_CSS}</style></head><body>",
        "<header class='k-hdr'><div class='k-hdr-inner'><div>",
        f"<div class='k-eyebrow'>KATANA-PRISM SELECTED-TARGET 2×2 GENECARD &amp; EVENTCARD REPORT · {escape(assay.upper())}</div>",
        f"<h1>{escape(study_name)}</h1>",
        f"<p class='k-hdr-sub'>{n_targets} verified target(s) · 4-panel DEG, PSI, Native BAM Sashimi &amp; Proteome/Cis-Motif synthesis</p>",
        "</div>",
        f"<div class='k-hdr-actions no-print'>{''.join(hdr_links)}</div>",
        "</div></header>",
        "<nav class='k-sticky' aria-label='Target and comparison selector'>",
        "<div class='k-sticky-inner'>",
        "<div class='k-tabs-row'><span class='k-menu-label'>Selected Targets:</span><div id='target-tabs' class='k-tabs-row'></div></div>",
        "<div class='k-ctrl-row'>",
        "<label class='k-ctrl'>Target <select id='target-select' aria-label='Select Gene or Event target'></select></label>",
        "<label class='k-ctrl'>Comparison <select id='comparison-select' aria-label='Select Comparison'></select></label>",
        "<label class='k-ctrl' id='event-select-wrap' style='display:none'>Splicing Event <select id='event-select' aria-label='Select Splicing Event'></select></label>",
        "</div></div></nav>",
        "<main class='k-main'>",
        "<div id='target-ribbon' class='k-ribbon'></div>",
        "<section id='grid-2x2' class='katana-grid-2x2' aria-label='2x2 GeneCard Panels'></section>",
    ]
    if n_targets > 1:
        parts.append(
            f"<details class='k-drawer'><summary>Selected Targets Overview ({n_targets}) — click any gene or event to load 2×2 view</summary>"
            "<div class='scroll'><table class='k-table'><thead><tr>"
            "<th>Target</th><th>Kind</th><th>Genomic Event Scope</th><th>Host DEG</th><th>Splicing ΔPSI</th><th>Status</th>"
            "</tr></thead><tbody id='overview-tbody'></tbody></table></div></details>"
        )
    parts.extend([
        "<details class='k-drawer'><summary>Complete Sliced Evidence Tables &amp; Server Resource Provenance</summary>",
        "<div id='evidence-links-box'></div></details>",
        "</main>",
        "<div id='k-modal' class='k-modal' role='dialog' aria-modal='true' aria-label='Zoomed Native BAM Sashimi'>",
        "<div class='k-modal-inner'><div style='display:flex;justify-content:space-between;align-items:center;gap:12px'>",
        "<strong id='k-modal-cap' style='color:#051c2c;font-size:13px'></strong>",
        "<button type='button' id='k-modal-close' style='padding:4px 10px;font-weight:700'>Close (Esc)</button>",
        "</div><img id='k-modal-img' src='' alt='Zoomed Sashimi'></div></div>",
        f"<script id='katana-batch-data' type='application/json'>{_json_script(batch_payload)}</script>",
        f"<script>{DASHBOARD_JS}</script>",
        "</body></html>",
    ])
    return "".join(parts)


def render_target(root, request, target, evidence, cards, samples, plots, sashimi, status):
    root = Path(root)
    vm = _build_target_view_model(root, request, target, evidence, cards, samples, sashimi, status, asset_prefix="")
    payload = {
        "schema": "katana.unified_target_report/v1",
        "project_id": request["project"]["project_id"],
        "project_name": request["project"].get("name", request["project"]["project_id"]),
        "assay": request["project"].get("assay", "bulk_rna"),
        "targets": [vm],
    }
    label = vm["gene_symbol"] or vm["target_id"]
    html = _render_unified_2x2_html(payload, page_title=f"Katana 2×2 GeneCard · {label}", show_batch_links=False)
    (root / "REPORT.html").write_text(html, encoding="utf-8")


def render_index(root, manifest):
    root = Path(root)
    req_path = root / "request.json"
    request = json.loads(req_path.read_text(encoding="utf-8")) if req_path.is_file() else {
        "project": {"project_id": manifest.get("project_id", ""), "name": manifest.get("project_name", ""), "samples": [], "contrasts": []},
        "targets": [],
    }
    target_by_id = {str(t["target_id"]): t for t in request.get("targets", [])}
    view_models = []
    for t_summary in manifest.get("targets", []):
        tid = str(t_summary.get("target_id", ""))
        t_manifest_rel = t_summary.get("manifest")
        if t_manifest_rel and (root / t_manifest_rel).is_file():
            t_dir = (root / t_manifest_rel).parent
            asset_prefix = t_dir.relative_to(root).as_posix() + "/"
            t_rec = json.loads((t_dir / "target-manifest.json").read_text(encoding="utf-8"))
            cards = json.loads((t_dir / "event-cards.json").read_text(encoding="utf-8")).get("cards", []) if (t_dir / "event-cards.json").is_file() else []
            pq_samples = t_dir / "sample-measurements.parquet"
            samples = pd.read_parquet(pq_samples) if pq_samples.is_file() else pd.DataFrame()
            req_target = target_by_id.get(tid, {
                "target_id": tid,
                "target_kind": t_rec.get("target_kind", "gene"),
                "gene_id": t_rec.get("gene_id"),
                "gene_symbol": t_rec.get("gene_symbol"),
                "event_id": t_rec.get("event_id"),
                "comparison_ids": t_rec.get("comparison_ids", []),
            })
            vm = _build_target_view_model(
                t_dir,
                request,
                req_target,
                t_rec.get("evidence", []),
                cards,
                samples,
                t_rec.get("sashimi", []),
                t_rec.get("status", t_summary.get("status", "partial")),
                asset_prefix=asset_prefix,
            )
            view_models.append(vm)
        else:
            req_target = target_by_id.get(tid, t_summary)
            view_models.append({
                "target_id": tid,
                "target_kind": req_target.get("target_kind", "gene"),
                "gene_id": _clean_str(req_target.get("gene_id")),
                "gene_symbol": _clean_str(req_target.get("gene_symbol") or tid),
                "requested_event_id": _clean_str(req_target.get("event_id")),
                "status": t_summary.get("status", "failed"),
                "condition_order": [],
                "comparisons": [{"id": str(c), "label": str(c), "num_label": "Group 1", "den_label": "Group 2", "num_samples": [], "den_samples": []}
                                for c in req_target.get("comparison_ids", [""])],
                "deg_by_comparison": {},
                "expression_unit": "Normalized counts",
                "expression_samples": [],
                "events": [],
                "evidence_links": [],
                "sample_measurements_href": "",
                "provenance_href": "",
            })

    batch_payload = {
        "schema": "katana.unified_target_report/v1",
        "project_id": manifest.get("project_id", ""),
        "project_name": manifest.get("project_name", manifest.get("project_id", "")),
        "assay": request.get("project", {}).get("assay", "bulk_rna"),
        "targets": view_models,
    }
    html = _render_unified_2x2_html(
        batch_payload,
        page_title=f"Katana 2×2 Selected-Target Report · {batch_payload['project_name']}",
        show_batch_links=True,
        has_tsv=(root / "targets.tsv").is_file(),
        has_zip=True,
    )
    (root / "INDEX.html").write_text(html, encoding="utf-8")
    (root / "REPORT.html").write_text(html, encoding="utf-8")
