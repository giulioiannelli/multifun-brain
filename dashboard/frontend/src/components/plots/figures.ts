// Plotly figure builders: each takes a backend plot spec and returns
// { data, layout } for <PlotlyFigure>. Kept data-only (no React) so they're
// easy to test and reuse across explore/compare views.
import type { PlotSpec } from "../../types";

const SIGNED = { pos: "#27AE60", neg: "#E74C3C" };
const PRIMARY = "#1976D2";

const QUALITATIVE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
  "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#393b79", "#637939",
];

// Shared with the backend _LRG_PALETTE so a cluster keeps one colour across the
// dendrogram, partition-flow raster and Sankey.
const LRG_PALETTE = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
  "#e377c2", "#17becf", "#bcbd22", "#393b79", "#637939", "#8c6d31",
  "#843c39", "#7b4173", "#5254a3", "#e7969c", "#a55194", "#6b6ecf",
];
const CUT_GREY = "#c2c2c2";

export interface Figure {
  data: any[];
  layout: Record<string, any>;
}

function binCenters(edges: number[]): number[] {
  const c: number[] = [];
  for (let i = 0; i < edges.length - 1; i++) c.push((edges[i] + edges[i + 1]) / 2);
  return c;
}

// Smooth KDE overlay (density pre-scaled to count units by the backend). On a
// log count axis the near-zero tails are floored so the curve sits at a baseline
// instead of plunging to -inf and rendering as spurious spikes.
function kdeTrace(x: number[], y: number[], yLog?: boolean): any {
  const yy = yLog ? y.map((v) => Math.max(v, 0.5)) : y;
  return {
    type: "scatter",
    x,
    y: yy,
    mode: "lines",
    line: { color: "#1a1a1a", width: 2 },
    name: "KDE",
    hovertemplate: "λ ≈ %{x:.3g}<br>density %{y:.1f}<extra>KDE</extra>",
    showlegend: false,
  };
}

// Histogram y-axis spec honouring a linear/log toggle.
function countAxis(yLog?: boolean): Record<string, any> {
  return { title: { text: yLog ? "count (log)" : "count" }, type: yLog ? "log" : "linear" };
}

// Symmetric-log transform of a signed matrix for the heatmap colour mapping.
// zt = sign(z) * log10(1 + |z| / linthresh); the colourbar is re-ticked with the
// original values, and the untouched z is returned for hover (`customdata`).
function symlogMatrix(z: (number | null)[][]): {
  z: (number | null)[][];
  zmin: number;
  zmax: number;
  tickvals: number[];
  ticktext: string[];
} {
  let maxAbs = 0;
  for (const row of z)
    for (const v of row) {
      if (v == null || !Number.isFinite(v)) continue;
      const a = Math.abs(v);
      if (a > maxAbs) maxAbs = a;
    }
  const linthresh = Math.max(maxAbs * 1e-2, 1e-6);
  const t = (v: number | null) =>
    v == null || !Number.isFinite(v)
      ? null
      : Math.sign(v) * Math.log10(1 + Math.abs(v) / linthresh);
  const zt = z.map((row) => row.map(t));
  const zmax = Math.log10(1 + maxAbs / linthresh) || 1;

  const mags = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10].filter(
    (m) => m <= maxAbs,
  );
  if (!mags.length) mags.push(maxAbs);
  const tickvals: number[] = [];
  const ticktext: string[] = [];
  for (let i = mags.length - 1; i >= 0; i--) {
    tickvals.push(t(-mags[i])!);
    ticktext.push(String(-mags[i]));
  }
  tickvals.push(0);
  ticktext.push("0");
  for (const m of mags) {
    tickvals.push(t(m)!);
    ticktext.push(String(m));
  }
  return { z: zt, zmin: -zmax, zmax, tickvals, ticktext };
}

// Square correlation / partial-correlation matrix with atlas-name hover.
// opts.log switches the colour mapping to symmetric-log (handles signed data;
// hover still reports the true value).
export function buildMatrix(
  spec: PlotSpec,
  title: string,
  opts: { log?: boolean } = {},
): Figure {
  const showTicks = spec.n <= 120;
  const z = spec.z as (number | null)[][];

  let trace: Record<string, any> = {
    type: "heatmap",
    z,
    x: spec.names,
    y: spec.names,
    zmin: spec.zmin,
    zmax: spec.zmax,
    colorscale: "RdBu",
    reversescale: true,
    colorbar: { thickness: 12 },
    hovertemplate:
      "row: <b>%{y}</b><br>col: <b>%{x}</b><br>value = %{z:.3f}<extra></extra>",
  };

  if (opts.log && Array.isArray(z) && z.length) {
    const s = symlogMatrix(z);
    trace = {
      ...trace,
      z: s.z,
      zmin: s.zmin,
      zmax: s.zmax,
      customdata: z,
      colorbar: {
        thickness: 12,
        tickvals: s.tickvals,
        ticktext: s.ticktext,
        title: { text: "log", side: "right" },
      },
      hovertemplate:
        "row: <b>%{y}</b><br>col: <b>%{x}</b><br>value = %{customdata:.3f}<extra></extra>",
    };
  }

  return {
    data: [trace],
    layout: {
      title: { text: opts.log ? `${title} · log` : title },
      height: 620,
      xaxis: { constrain: "domain", tickfont: { size: 6 }, showticklabels: showTicks, tickangle: 90 },
      yaxis: {
        scaleanchor: "x",
        scaleratio: 1,
        constrain: "domain",
        autorange: "reversed",
        tickfont: { size: 6 },
        showticklabels: showTicks,
      },
      margin: { l: 80, r: 20, t: 44, b: 80 },
    },
  };
}

// Correlation eigenvalue spectrum with Marchenko-Pastur bulk bounds. Uses the
// backend's density-adaptive bins (counts/edges) so the bulk is resolved; a few
// extreme eigenvalues (n_above) are noted in an annotation rather than binned in.
export function buildSpectrum(spec: PlotSpec, opts: { yLog?: boolean; kde?: boolean } = {}): Figure {
  const edges: number[] | null = spec.edges ?? null;
  const counts: number[] | null = spec.counts ?? null;
  const lo = edges ? edges[0] : null;
  const hi = edges ? edges[edges.length - 1] : null;
  const cleaned = spec.bulk_lo != null && spec.bulk_hi != null;

  const shapes: any[] = [];
  if (cleaned) {
    // Shade the MP noise bulk [λ-, λ+] — the eigenvalues the cleaning flattens.
    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0: spec.bulk_lo, x1: spec.bulk_hi, y0: 0, y1: 1,
      fillcolor: "rgba(120,120,120,0.13)", line: { width: 0 }, layer: "below",
    });
  } else {
    // Raw mode: MP bulk bounds drawn only when inside the displayed range.
    for (const [val, c] of [
      [spec.mp_lambda_minus, "#888"],
      [spec.mp_lambda_plus, "#D32F2F"],
    ] as [number, string][]) {
      if (val != null && lo != null && hi != null && val >= lo && val <= hi) {
        shapes.push({
          type: "line",
          x0: val, x1: val, yref: "paper", y0: 0, y1: 1,
          line: { color: c, width: 1.5, dash: "dash" },
        });
      }
    }
  }

  const annotations: any[] = [];
  if (spec.n_above > 0) {
    const mx = spec.above?.[0];
    const mxTxt = typeof mx === "number" ? mx.toPrecision(4) : mx;
    annotations.push({
      xref: "paper", yref: "paper", x: 1, y: 1, xanchor: "right", yanchor: "top",
      showarrow: false, align: "right", font: { size: 11, color: "#D32F2F" },
      text: `${spec.n_above} eigenvalue${spec.n_above > 1 ? "s" : ""} above range · max λ=${mxTxt}`,
    });
  }

  const data: any[] =
    edges && counts
      ? [
          {
            type: "bar",
            x: binCenters(edges),
            y: counts,
            width: edges.slice(1).map((e, i) => e - edges[i]),
            marker: { color: PRIMARY, line: { width: 0 } },
            hovertemplate: "λ ≈ %{x:.3g}<br>count %{y}<extra></extra>",
          },
        ]
      : [
          {
            type: "histogram",
            x: spec.eigenvalues,
            nbinsx: 40,
            marker: { color: PRIMARY },
            hovertemplate: "λ ≈ %{x:.3f}<br>count %{y}<extra></extra>",
          },
        ];

  if (opts.kde && spec.kde_x && spec.kde_y) {
    data.push(kdeTrace(spec.kde_x, spec.kde_y, opts.yLog));
  }

  // MP-cleaned mode: overlay the theoretical Marchenko–Pastur noise density.
  if (spec.mp_curve_x && spec.mp_curve_y) {
    const my = opts.yLog
      ? spec.mp_curve_y.map((v: number) => Math.max(v, 0.5))
      : spec.mp_curve_y;
    data.push({
      type: "scatter",
      x: spec.mp_curve_x,
      y: my,
      mode: "lines",
      line: { color: "#D32F2F", width: 2 },
      name: "MP density",
      hovertemplate: "λ ≈ %{x:.3g}<br>MP %{y:.1f}<extra>MP noise</extra>",
      showlegend: false,
    });
  }

  return {
    data,
    layout: {
      title: { text: `eigenvalue spectrum · signal ${spec.n_signal ?? "?"} / noise ${spec.n_noise ?? "?"}` },
      xaxis: { title: { text: "eigenvalue λ" } },
      yaxis: countAxis(opts.yLog),
      shapes,
      annotations,
      bargap: 0,
    },
  };
}

// Signed weight distribution (positive vs negative bars) with optional KDE
// overlay and a linear/log count axis.
export function buildWeights(spec: PlotSpec, opts: { yLog?: boolean; kde?: boolean } = {}): Figure {
  const centers = spec.edges ? binCenters(spec.edges) : [];
  const colors = centers.map((c) => (c >= 0 ? SIGNED.pos : SIGNED.neg));
  const data: any[] = [
    {
      type: "bar",
      x: centers,
      y: spec.counts ?? [],
      marker: { color: colors },
      hovertemplate: "w ≈ %{x:.3f}<br>count %{y}<extra></extra>",
    },
  ];
  if (opts.kde && spec.kde_x && spec.kde_y) {
    data.push({
      ...kdeTrace(spec.kde_x, spec.kde_y, opts.yLog),
      hovertemplate: "w ≈ %{x:.3f}<br>density %{y:.1f}<extra>KDE</extra>",
    });
  }
  return {
    data,
    layout: {
      title: { text: `weight distribution · +${(spec.frac_positive ?? 0).toFixed?.(2)} / −${(spec.frac_negative ?? 0).toFixed?.(2)}` },
      xaxis: { title: { text: "edge weight" } },
      yaxis: countAxis(opts.yLog),
      bargap: 0.02,
    },
  };
}

// Signed-Laplacian eigenvalues, coloured by sign.
export function buildSignedLaplacian(spec: PlotSpec): Figure {
  const eig: number[] = spec.eigenvalues ?? [];
  return {
    data: [
      {
        type: "bar",
        x: eig.map((_, i) => i),
        y: eig,
        marker: { color: eig.map((v) => (v < 0 ? SIGNED.neg : SIGNED.pos)) },
        hovertemplate: "mode %{x}<br>λ = %{y:.4f}<extra></extra>",
      },
    ],
    layout: {
      title: { text: `signed Laplacian · frustration ${(spec.frustration_index ?? 0).toFixed?.(3)} · neg λ ${spec.n_negative_eigenvalues ?? 0}` },
      xaxis: { title: { text: "mode index" } },
      yaxis: { title: { text: "eigenvalue" } },
    },
  };
}

// Per-node positive vs negative strength (signed balance).
export function buildSignedBalance(spec: PlotSpec): Figure {
  return {
    data: [
      {
        type: "scatter",
        mode: "markers",
        x: spec.strength_positive ?? [],
        y: spec.strength_negative ?? [],
        text: spec.names ?? [],
        marker: { color: spec.colors ?? PRIMARY, size: 9, line: { color: "#fff", width: 0.5 } },
        hovertemplate: "%{text}<br>S+ %{x:.2f}<br>S− %{y:.2f}<extra></extra>",
      },
    ],
    layout: {
      title: { text: `signed balance · ratio ${(spec.balance_ratio ?? 0).toFixed?.(3)}` },
      xaxis: { title: { text: "positive strength" } },
      yaxis: { title: { text: "negative strength" } },
    },
  };
}

// Degree distribution histogram.
export function buildDegree(spec: PlotSpec): Figure {
  const centers = spec.edges ? binCenters(spec.edges) : [];
  return {
    data: [
      {
        type: "bar",
        x: centers,
        y: spec.counts ?? [],
        marker: { color: PRIMARY },
        hovertemplate: "deg ≈ %{x:.1f}<br>count %{y}<extra></extra>",
      },
    ],
    layout: {
      title: { text: `degree distribution · mean ${(spec.mean ?? 0).toFixed?.(2)}` },
      xaxis: { title: { text: "degree" } },
      yaxis: { title: { text: "count" } },
      bargap: 0.02,
    },
  };
}

// Colour every drawn dendrogram link for a cut at height h — entirely client-side
// (union-find over the linkage), so the height slider recolours instantly with no
// server round-trip. A link merging below h is coloured by the cluster it belongs
// to (LRG_PALETTE, ordered left-to-right to mirror SciPy); links above h are grey.
// Returns the per-link colours (aligned to icoord/dcoord) and the cluster count.
export function colorDendrogram(spec: PlotSpec, h: number): { colors: string[]; nClusters: number } {
  const n: number = spec.n_leaves ?? 0;
  const Z: number[][] = spec.linkage ?? [];
  const linkIds: number[] = spec.link_ids ?? [];
  const leaves: number[] = spec.leaves ?? [];
  if (!n || !Z.length) return { colors: linkIds.map(() => CUT_GREY), nClusters: 0 };

  const parent = new Int32Array(2 * n - 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const union = (a: number, b: number) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (let i = 0; i < Z.length; i++) {
    if (Z[i][2] <= h) { union(Z[i][0], Z[i][1]); union(n + i, Z[i][0]); }
  }

  // Stable colour order: index clusters by their leftmost leaf position.
  const leafPos = new Map<number, number>();
  leaves.forEach((lid, p) => leafPos.set(lid, p));
  const rootMinPos = new Map<number, number>();
  for (let leaf = 0; leaf < n; leaf++) {
    const r = find(leaf);
    const p = leafPos.get(leaf) ?? leaf;
    if (!rootMinPos.has(r) || p < (rootMinPos.get(r) as number)) rootMinPos.set(r, p);
  }
  const colorOfRoot = new Map<number, number>();
  [...rootMinPos.entries()].sort((a, b) => a[1] - b[1]).forEach(([r], idx) => colorOfRoot.set(r, idx));

  const colors = linkIds.map((nodeId) => {
    const mergeIdx = nodeId - n;
    if (mergeIdx < 0 || mergeIdx >= Z.length || Z[mergeIdx][2] > h) return CUT_GREY;
    const ci = colorOfRoot.get(find(nodeId));
    return ci === undefined ? CUT_GREY : LRG_PALETTE[ci % LRG_PALETTE.length];
  });
  return { colors, nClusters: colorOfRoot.size };
}

// Per-leaf community colour at a cut height h — the node-level companion to
// colorDendrogram (same union-find + leftmost-leaf ordering), so the clustered
// network and glass brain colour nodes identically to the dendrogram branches.
// Returns one hex colour per leaf index 0..n-1 (aligned to lrg_network node ids).
export function clusterLeaves(spec: PlotSpec, h: number): { colors: string[]; nClusters: number } {
  const n: number = spec.n_leaves ?? 0;
  const Z: number[][] = spec.linkage ?? [];
  const leaves: number[] = spec.leaves ?? [];
  if (!n || !Z.length) return { colors: Array.from({ length: n }, () => CUT_GREY), nClusters: 0 };

  const parent = new Int32Array(2 * n - 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = (x: number): number => {
    while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; }
    return x;
  };
  const union = (a: number, b: number) => {
    const ra = find(a), rb = find(b);
    if (ra !== rb) parent[ra] = rb;
  };
  for (let i = 0; i < Z.length; i++) {
    if (Z[i][2] <= h) { union(Z[i][0], Z[i][1]); union(n + i, Z[i][0]); }
  }
  const leafPos = new Map<number, number>();
  leaves.forEach((lid, p) => leafPos.set(lid, p));
  const rootMinPos = new Map<number, number>();
  for (let leaf = 0; leaf < n; leaf++) {
    const r = find(leaf);
    const p = leafPos.get(leaf) ?? leaf;
    if (!rootMinPos.has(r) || p < (rootMinPos.get(r) as number)) rootMinPos.set(r, p);
  }
  const colorOfRoot = new Map<number, number>();
  [...rootMinPos.entries()].sort((a, b) => a[1] - b[1]).forEach(([r], idx) => colorOfRoot.set(r, idx));

  const colors = Array.from({ length: n }, (_, leaf) => {
    const ci = colorOfRoot.get(find(leaf));
    return ci === undefined ? CUT_GREY : LRG_PALETTE[ci % LRG_PALETTE.length];
  });
  return { colors, nClusters: colorOfRoot.size };
}

// LRG dendrogram from a SciPy layout (icoord/dcoord) — log distance axis, zeros
// floored. Branches are coloured client-side by the cut at height `cutHeight`
// (defaults to the LRG's natural flat_threshold); a dotted line marks the cut and
// the title reports the clusters it yields. Dragging the cut only re-runs this
// builder — no refetch.
export function buildDendrogram(spec: PlotSpec, cutHeight?: number): Figure {
  const floor = (spec.dcoord_min_positive ?? 1e-3) * 0.5;
  const h = cutHeight != null && cutHeight > 0 ? cutHeight : (spec.flat_threshold ?? null);
  const { colors, nClusters } = h != null ? colorDendrogram(spec, h) : { colors: [], nClusters: 0 };
  const traces = (spec.icoord ?? []).map((xs: number[], i: number) => ({
    type: "scatter",
    mode: "lines",
    x: xs,
    y: spec.dcoord[i].map((v: number) => (v <= 0 ? floor : v)),
    line: { color: colors[i] ?? "#4a4a4a", width: 1.3 },
    hoverinfo: "skip",
    showlegend: false,
  }));
  const ivl: string[] = spec.ivl ?? [];
  const shapes: any[] = [];
  const annotations: any[] = [];
  if (h != null && h > floor) {
    shapes.push({
      type: "line", xref: "paper", x0: 0, x1: 1,
      y0: h, y1: h,
      line: { color: "#D32F2F", width: 1.2, dash: "dot" },
    });
    annotations.push({
      xref: "paper", yref: "y", x: 1, y: h, xanchor: "right", yanchor: "bottom",
      showarrow: false, text: `cut h=${Number(h).toPrecision(3)} · ${nClusters} clusters`,
      font: { size: 10, color: "#D32F2F" },
    });
  }
  const cutTxt = h != null ? ` · cut → ${nClusters} clusters` : "";
  const tau = typeof spec.tau === "number" ? spec.tau.toPrecision(3) : spec.tau;
  return {
    data: traces,
    layout: {
      title: { text: `dendrogram · τ=${tau}${cutTxt}` },
      height: 520,
      xaxis: {
        tickmode: "array",
        tickvals: ivl.map((_, i) => 5 + 10 * i),
        ticktext: ivl,
        tickangle: 90,
        tickfont: { size: 6 },
        showticklabels: ivl.length <= 120,
      },
      yaxis: { type: "log", title: { text: "diffusion distance" } },
      shapes,
      annotations,
      margin: { l: 70, r: 20, t: 44, b: 90 },
    },
  };
}

// LRG specific heat C(τ) = −dS/d·log τ vs τ (log axis). Peaks mark characteristic
// diffusion scales; τ′ = 1/λmax and τ* (peak) are dashed/dotted, and the currently
// selected τ (the dendrogram slider) is a solid marker so the two views stay tied.
export function buildSpecificHeat(spec: PlotSpec, opts: { currentTau?: number } = {}): Figure {
  const tau: number[] = spec.tau ?? [];
  const c: number[] = spec.C ?? [];
  const shapes: any[] = [];
  const annotations: any[] = [];
  const vline = (x: number | null | undefined, color: string, dash: string, label: string) => {
    if (x == null || !(x > 0)) return;
    shapes.push({ type: "line", xref: "x", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color, width: 1.4, dash } });
    annotations.push({ xref: "x", x, yref: "paper", y: 1, yanchor: "bottom", showarrow: false, text: label, font: { size: 10, color } });
  };
  vline(spec.tau_prime, "#888", "dash", "τ′");
  vline(spec.tau_star, "#1976D2", "dot", "τ*");
  vline(opts.currentTau, "#D32F2F", "solid", "τ");
  const tstar = spec.tau_star ? Number(spec.tau_star).toPrecision(3) : "?";
  return {
    data: [{
      type: "scatter", mode: "lines", x: tau, y: c,
      line: { color: "#1976D2", width: 2 },
      hovertemplate: "τ=%{x:.3g}<br>C=%{y:.3f}<extra></extra>", showlegend: false,
    }],
    layout: {
      title: { text: `specific heat C(τ) · τ*≈${tstar}` },
      height: 340,
      xaxis: { type: "log", title: { text: "τ (diffusion time)" } },
      yaxis: { title: { text: "C(τ) = −dS/d·log τ" } },
      shapes, annotations,
      margin: { l: 56, r: 16, t: 44, b: 44 },
    },
  };
}

// LRG partition-stability Ψ(n) vs number of clusters at the current τ. The peak
// (highlighted) is the most stable cut — the "where to cut" the dendrogram.
export function buildPsi(spec: PlotSpec): Figure {
  const psi: number[] = spec.psi ?? [];
  const nc: number[] = spec.n_clusters ?? [];
  const optK: number | undefined = spec.optimal_n_clusters;
  const optPsi: number | undefined = spec.optimal_psi;
  const annotations: any[] = [];
  if (optK != null && optPsi != null) {
    annotations.push({
      x: optK, y: optPsi, xref: "x", yref: "y", text: `optimal → ${optK} clusters`,
      showarrow: true, arrowhead: 2, ax: 0, ay: -28, font: { size: 10, color: "#D32F2F" },
    });
  }
  const data: any[] = [{
    type: "scatter", mode: "lines+markers", x: nc, y: psi,
    line: { color: "#8c564b", width: 1.5 }, marker: { size: 4, color: "#8c564b" },
    hovertemplate: "%{x} clusters<br>Ψ=%{y:.3f}<extra></extra>", showlegend: false,
  }];
  if (optK != null && optPsi != null) {
    data.push({
      type: "scatter", mode: "markers", x: [optK], y: [optPsi],
      marker: { size: 12, color: "#D32F2F", symbol: "circle-open", line: { width: 2 } },
      hoverinfo: "skip", showlegend: false,
    });
  }
  return {
    data,
    layout: {
      title: { text: `partition stability Ψ(n) · peak at ${optK ?? "?"} clusters` },
      height: 340,
      xaxis: { title: { text: "number of clusters" } },
      yaxis: { title: { text: "Ψ (stability)" } },
      annotations,
      margin: { l: 56, r: 16, t: 44, b: 44 },
    },
  };
}

function discreteColorscale(maxId: number): [number, string][] {
  const k = Math.max(maxId + 1, 1);
  const stops: [number, string][] = [];
  for (let i = 0; i < k; i++) {
    const c = QUALITATIVE[i % QUALITATIVE.length];
    stops.push([i / k, c]);
    stops.push([(i + 1) / k, c]);
  }
  return stops;
}

// Community-membership raster: rows = nodes (sorted into merging bands), cols = τ,
// colour = cross-τ-consistent community id (so a band keeps its colour while it
// stays one community). See _canonical_labels on the backend.
export function buildPartitionFlow(spec: PlotSpec): Figure {
  const maxId = Math.max(
    0,
    ...(spec.z ?? []).flat().filter((v: number) => Number.isFinite(v)),
  );
  const showTicks = (spec.names?.length ?? 0) <= 120;
  return {
    data: [
      {
        type: "heatmap",
        z: spec.z,
        x: (spec.taus ?? []).map((t: number) => t.toPrecision(2)),
        y: spec.names,
        colorscale: discreteColorscale(maxId),
        zmin: 0,
        zmax: maxId + 1,
        showscale: false,
        hovertemplate: "node %{y}<br>τ %{x}<br>community %{z}<extra></extra>",
      },
    ],
    layout: {
      title: { text: `community membership across τ · ${(spec.n_communities ?? maxId + 1)} communities` },
      height: 640,
      xaxis: { title: { text: "τ" }, type: "category" },
      yaxis: { tickfont: { size: 6 }, showticklabels: showTicks, autorange: "reversed" },
      margin: { l: 90, r: 20, t: 44, b: 44 },
    },
  };
}

// Per-region z-score (demean + unit variance) of a carpet matrix, so the
// grayscale shows temporal *dynamics* rather than the raw per-region offset
// (BOLD means differ by region). Degenerate (flat) rows map to 0.
function zscoreRows(z: number[][]): number[][] {
  return z.map((row) => {
    const finite = row.filter((v) => Number.isFinite(v));
    const n = finite.length || 1;
    const mean = finite.reduce((a, b) => a + b, 0) / n;
    const variance = finite.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
    const sd = Math.sqrt(variance);
    if (sd <= 0) return row.map(() => 0);
    return row.map((v) => (Number.isFinite(v) ? (v - mean) / sd : 0));
  });
}

// Carpet / "grayplot": regions (rows) × timepoints (cols), grayscale. With
// normalise on, each region is z-scored (raw value kept for hover); off shows
// the raw signal with an auto colour range.
export function buildSignalHeatmap(
  spec: PlotSpec,
  opts: { normalize?: boolean } = {},
): Figure {
  const z = (spec.z ?? []) as number[][];
  const names: string[] = spec.names ?? [];
  const showTicks = (spec.n_regions ?? names.length) <= 120;
  const normalize = opts.normalize ?? true;
  const band = (spec.band as string) ?? "full";
  const BAND_TITLE: Record<string, string> = {
    full: "broadband",
    s5: "Slow-5 · 0.010–0.027 Hz",
    s4: "Slow-4 · 0.027–0.073 Hz",
    sstar: "S* · 0.073–0.180 Hz",
  };
  const bandTag = band === "full" ? "" : ` · ${BAND_TITLE[band] ?? band}`;

  const trace: Record<string, any> = {
    type: "heatmap",
    y: names,
    colorscale: "Greys",
    hovertemplate: "region <b>%{y}</b><br>t = %{x}<br>value = %{z:.3f}<extra></extra>",
  };

  if (normalize) {
    trace.z = zscoreRows(z);
    trace.customdata = z;
    trace.zmin = -3;
    trace.zmax = 3;
    trace.colorbar = { thickness: 12, title: { text: "z", side: "right" } };
    trace.hovertemplate =
      "region <b>%{y}</b><br>t = %{x}<br>value = %{customdata:.3f}<br>z = %{z:.2f}<extra></extra>";
  } else {
    trace.z = z;
    trace.colorbar = { thickness: 12 };
  }

  return {
    data: [trace],
    layout: {
      title: { text: `timecourses${bandTag} · ${spec.n_regions ?? names.length} regions × ${spec.n_timepoints ?? "?"} timepoints` },
      // Taller when there are more regions so each row has room for a legible
      // tick label; automargin lets Plotly widen the left gutter to fit names.
      height: Math.min(1100, Math.max(520, (spec.n_regions ?? names.length) * 12)),
      xaxis: { title: { text: "timepoint" } },
      yaxis: {
        autorange: "reversed",
        tickfont: { size: 10 },
        showticklabels: showTicks,
        automargin: true,
      },
      margin: { l: 120, r: 20, t: 44, b: 44 },
    },
  };
}

// Single-region timecourse: value per timepoint. A spline curves the connecting
// line through every sample (it interpolates, it does not average or drop
// points), and small markers keep the real samples visible — smoother look,
// nothing masked.
export function buildSignalChannel(spec: PlotSpec): Figure {
  return {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: spec.t ?? [],
        y: spec.y ?? [],
        line: { color: PRIMARY, width: 1.5, shape: "spline", smoothing: 1.0 },
        marker: { color: PRIMARY, size: 3, opacity: 0.5 },
        hovertemplate: "t = %{x}<br>value = %{y:.3f}<extra></extra>",
      },
    ],
    layout: {
      title: { text: `timecourse · ${spec.name ?? `region ${spec.channel}`}` },
      height: 320,
      xaxis: { title: { text: "timepoint" } },
      yaxis: { title: { text: "signal" } },
      margin: { l: 60, r: 20, t: 44, b: 44 },
    },
  };
}

// EMD decomposition of one channel: original signal + each IMF + residual,
// stacked as vertically-tiled panels sharing one time axis (the classic EMD
// view). Built with manual y-axis domains so it needs no subplot helper.
export function buildSignalEMD(spec: PlotSpec): Figure {
  const x: number[] = spec.t ?? [];
  const imfs = (spec.imfs ?? []) as number[][];
  const rows: { name: string; y: number[]; color: string }[] = [
    { name: "signal", y: spec.signal ?? [], color: "#1a1a1a" },
    ...imfs.map((y, i) => ({
      name: `IMF ${i + 1}`,
      y,
      color: QUALITATIVE[i % QUALITATIVE.length],
    })),
  ];
  const resid: number[] = spec.residual ?? [];
  const residMax = resid.reduce((m, v) => (Number.isFinite(v) ? Math.max(m, Math.abs(v)) : m), 0);
  if (resid.length && residMax > 1e-9) {
    rows.push({ name: "resid", y: resid, color: "#9e9e9e" });
  }

  const R = rows.length;
  const gap = 0.02;
  const h = (1 - gap * (R - 1)) / R;
  const bottomAxis = R === 1 ? "y" : `y${R}`;

  const data: any[] = [];
  const layout: Record<string, any> = {
    title: { text: `EMD · ${spec.name ?? `region ${spec.channel}`} · ${spec.n_imf ?? imfs.length} IMFs` },
    height: Math.max(420, R * 130),
    showlegend: false,
    margin: { l: 70, r: 16, t: 44, b: 44 },
  };

  rows.forEach((row, i) => {
    const axisIdx = i + 1;
    const ykey = i === 0 ? "yaxis" : `yaxis${axisIdx}`;
    const yref = i === 0 ? "y" : `y${axisIdx}`;
    const top = 1 - i * (h + gap);
    const bottom = Math.max(0, top - h);
    layout[ykey] = {
      domain: [bottom, top],
      title: { text: row.name, font: { size: 11 } },
      zeroline: false,
      tickfont: { size: 8 },
    };
    data.push({
      type: "scatter",
      mode: "lines",
      x,
      y: row.y,
      xaxis: "x",
      yaxis: yref,
      line: { color: row.color, width: 1 },
      name: row.name,
      hovertemplate: `${row.name}: %{y:.3g}<extra></extra>`,
    });
  });

  layout.xaxis = { title: { text: "timepoint" }, anchor: bottomAxis };
  return { data, layout };
}

// Frequency bands (low→high): Slow-5, Slow-4, S*. Colours echo Daniele's figure.
const BAND_NAMES = ["s5", "s4", "sstar"] as const;
const BAND_TITLE: Record<string, string> = { s5: "Slow-5", s4: "Slow-4", sstar: "S*" };
const BAND_BAR: Record<string, string> = {
  s5: "#E91E63", s4: "#43A047", sstar: "#F9A825", drift: "#90A4AE",
};
const BAND_FILL: Record<string, string> = {
  s5: "rgba(233,30,99,0.12)", s4: "rgba(67,160,71,0.12)", sstar: "rgba(249,168,37,0.16)",
};

// Cohort IMF spectrum: histogram of every IMF's characteristic frequency
// (median instantaneous frequency, Hz, fs = 1/TR) pooled across subjects/ROIs,
// with the s5/s4/s* band ranges shaded and the per-IMF-index cluster centres
// marked. Reconstructs the collaborator's figure. opts.variant picks the
// frequency panel ("freq", default) or its 1/f companion period panel
// ("period", seconds); bands are stored in Hz and converted as needed.
export function buildCohortBands(
  spec: PlotSpec,
  opts: { yLog?: boolean; variant?: "freq" | "period" } = {},
): Figure {
  const isPeriod = opts.variant === "period";
  const edges: number[] = (isPeriod ? spec.period_edges : spec.edges) ?? [];
  const counts: number[] = (isPeriod ? spec.period_counts : spec.counts) ?? [];
  const bandsHz: Record<string, [number, number]> = spec.bands ?? {};
  const centers = binCenters(edges);
  const widths = edges.slice(1).map((e, i) => e - edges[i]);
  const scheme = spec.scheme === "data_driven" ? "data-driven" : "canonical";

  // A Hz band [lo,hi] becomes the period range [1/hi, 1/lo].
  const bandRange = (r: [number, number]): [number, number] =>
    isPeriod ? [1 / r[1], 1 / r[0]] : [r[0], r[1]];
  // Which band a plotted x falls in (period values map back to Hz first).
  const bandOf = (x: number): string => {
    const f = isPeriod ? 1 / x : x;
    for (const name of BAND_NAMES) {
      const r = bandsHz[name];
      if (r && f >= r[0] && f <= r[1]) return name;
    }
    return "drift";
  };
  const barColors = centers.map((c) => BAND_BAR[bandOf(c)]);

  const shapes: any[] = [];
  const annotations: any[] = [];
  for (const name of BAND_NAMES) {
    const r = bandsHz[name];
    if (!r) continue;
    const [x0, x1] = bandRange(r);
    shapes.push({
      type: "rect", xref: "x", yref: "paper",
      x0, x1, y0: 0, y1: 1,
      fillcolor: BAND_FILL[name], line: { width: 0 }, layer: "below",
    });
    annotations.push({
      x: Math.sqrt(x0 * x1), xref: "x", yref: "paper", y: 0.98, yanchor: "top",
      showarrow: false, text: BAND_TITLE[name], font: { size: 12, color: BAND_BAR[name] },
    });
  }
  // Per-IMF-index cluster centres (dotted verticals + IMF label).
  const centersObj: Record<string, number> = spec.cluster_centers ?? {};
  for (const [k, v] of Object.entries(centersObj)) {
    if (!v || v <= 0) continue;
    const xv = isPeriod ? 1 / v : v;
    shapes.push({
      type: "line", xref: "x", yref: "paper", x0: xv, x1: xv, y0: 0, y1: 0.86,
      line: { color: "#555", width: 1, dash: "dot" },
    });
    annotations.push({
      x: xv, xref: "x", yref: "paper", y: 0.88, yanchor: "bottom",
      showarrow: false, text: `IMF${Number(k) + 1}`, font: { size: 9, color: "#555" },
    });
  }

  const what = isPeriod ? "period" : "frequency";
  const who = spec.subject ? String(spec.subject) : `${spec.n_subjects ?? "?"} subjects`;
  const xTitle = isPeriod ? "period (s)" : "characteristic frequency (Hz)";
  const hover = isPeriod
    ? "T ≈ %{x:.2f} s<br>count %{y}<extra></extra>"
    : "f ≈ %{x:.4f} Hz<br>count %{y}<extra></extra>";

  return {
    data: [
      {
        type: "bar",
        x: centers,
        y: counts,
        width: widths,
        marker: { color: barColors, line: { width: 0 } },
        hovertemplate: hover,
      },
    ],
    layout: {
      title: {
        text: `IMF ${what} spectrum · ${spec.contrast ?? ""} ${spec.processing ?? ""} · ${who} · ${scheme} bands`,
      },
      height: 420,
      xaxis: { title: { text: xTitle } },
      yaxis: countAxis(opts.yLog),
      shapes,
      annotations,
      bargap: 0,
    },
  };
}

// Per-band reconstructed timecourses of one channel: the original signal plus the
// sum of the IMFs falling in each band (S* / Slow-4 / Slow-5), stacked. These are
// the per-band signals behind the per-band correlation matrices.
export function buildBandReconstruction(spec: PlotSpec): Figure {
  const x: number[] = spec.t ?? [];
  const order = ["sstar", "s4", "s5"];
  const colors: Record<string, string> = { sstar: "#F9A825", s4: "#43A047", s5: "#E91E63" };
  const bands: Record<string, [number, number]> = spec.bands ?? {};
  const rows: { name: string; y: number[]; color: string }[] = [
    { name: "signal", y: spec.signal ?? [], color: "#1a1a1a" },
  ];
  for (const b of order) {
    const idx = (spec.assignment?.[b] ?? []).map((a: any) => a.imf);
    const r = bands[b];
    const range = r ? ` [${r[0].toFixed(3)}–${r[1].toFixed(3)} Hz]` : "";
    rows.push({
      name: `${BAND_TITLE[b]}${range} · ${idx.length} IMF${idx.length === 1 ? "" : "s"}`,
      y: spec.signals?.[b] ?? [],
      color: colors[b],
    });
  }

  const R = rows.length;
  const gap = 0.02;
  const h = (1 - gap * (R - 1)) / R;
  const bottomAxis = R === 1 ? "y" : `y${R}`;
  const data: any[] = [];
  const layout: Record<string, any> = {
    title: { text: `band reconstruction · ${spec.name ?? `region ${spec.channel}`}` },
    height: Math.max(420, R * 130),
    showlegend: false,
    margin: { l: 80, r: 16, t: 44, b: 44 },
  };
  rows.forEach((row, i) => {
    const axisIdx = i + 1;
    const ykey = i === 0 ? "yaxis" : `yaxis${axisIdx}`;
    const yref = i === 0 ? "y" : `y${axisIdx}`;
    const top = 1 - i * (h + gap);
    const bottom = Math.max(0, top - h);
    layout[ykey] = {
      domain: [bottom, top],
      title: { text: row.name, font: { size: 10 } },
      zeroline: false,
      tickfont: { size: 8 },
    };
    data.push({
      type: "scatter", mode: "lines", x, y: row.y, xaxis: "x", yaxis: yref,
      line: { color: row.color, width: 1 },
      name: row.name,
      hovertemplate: `${row.name}: %{y:.3g}<extra></extra>`,
    });
  });
  layout.xaxis = { title: { text: "timepoint" }, anchor: bottomAxis };
  return { data, layout };
}

// Compare tab: observed vs null per-ROI cophenetic-shift distribution. Overlays
// the pooled strength-preserving null (grey) and the observed shifts (red) as
// density histograms with median lines — the "is the reorganisation beyond
// chance?" panel from the collaborator's fig6.
export function buildCompareHist(spec: PlotSpec): Figure {
  const observed: number[] = spec.observed ?? [];
  const pooledNull: number[] = spec.pooled_null ?? [];
  const all = pooledNull.length ? observed.concat(pooledNull) : observed;
  const lo = all.length ? Math.min(...all) : 0;
  const hi = all.length ? Math.max(...all) : 1;
  const median = (a: number[]) => {
    if (!a.length) return null;
    const s = [...a].sort((x, y) => x - y);
    const m = Math.floor(s.length / 2);
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  };
  const data: any[] = [];
  const shapes: any[] = [];
  if (pooledNull.length) {
    data.push({
      type: "histogram", x: pooledNull, histnorm: "probability density",
      marker: { color: "#90A4AE" }, opacity: 0.6, name: `null (n=${pooledNull.length})`,
      xbins: { start: lo, end: hi, size: (hi - lo) / 26 || 0.01 },
    });
    const mn = median(pooledNull);
    if (mn != null) shapes.push({ type: "line", x0: mn, x1: mn, yref: "paper", y0: 0, y1: 1, line: { color: "#37474F", width: 1.2, dash: "dash" } });
  }
  data.push({
    type: "histogram", x: observed, histnorm: "probability density",
    marker: { color: "#B71C1C" }, opacity: 0.75, name: `observed (n=${observed.length})`,
    xbins: { start: lo, end: hi, size: (hi - lo) / 26 || 0.01 },
  });
  const mo = median(observed);
  if (mo != null) shapes.push({ type: "line", x0: mo, x1: mo, yref: "paper", y0: 0, y1: 1, line: { color: "#B71C1C", width: 1.5, dash: "dash" } });
  return {
    data,
    layout: {
      title: { text: "observed vs null shift distribution" },
      height: 340, barmode: "overlay",
      xaxis: { title: { text: "per-ROI cophenetic shift" } },
      yaxis: { title: { text: "density" } },
      legend: { font: { size: 10 }, x: 0.98, xanchor: "right", y: 0.98 },
      shapes, margin: { l: 56, r: 16, t: 44, b: 44 },
    },
  };
}

// Compare tab: top-N reorganised (red, >0) + top-N stable (blue, <0) ROIs, as a
// horizontal bar chart of the per-ROI metric (calibrated shift when a null is on,
// else raw shift).
export function buildCompareBars(spec: PlotSpec, n = 10): Figure {
  const rois: any[] = spec.rois ?? [];
  const top = rois.slice(0, n);
  const bottom = rois.slice(-n).filter((r) => !top.includes(r));
  const calibrated = spec.metric === "calibrated";
  // Draw bottom (most stable) first so the most-reorganised sits at the top.
  const ordered = [...bottom].reverse().concat([...top].reverse());
  const names = ordered.map((r) => r.name);
  const vals = ordered.map((r) => r.value);
  const colors = ordered.map((r) => (r.value >= 0 ? "#B71C1C" : "#1565C0"));
  return {
    data: [{
      type: "bar", orientation: "h", x: vals, y: names,
      marker: { color: colors, line: { color: "white", width: 0.6 } },
      hovertemplate: "%{y}<br>%{x:.3f}<extra></extra>",
    }],
    layout: {
      title: { text: `top ${top.length} reorganised (red) + top ${bottom.length} stable (blue)` },
      height: Math.max(360, ordered.length * 22 + 80),
      xaxis: {
        title: { text: calibrated ? "calibrated shift (obs − null median)" : "cophenetic-rank shift" },
        zeroline: true, zerolinecolor: "#90A4AE",
      },
      yaxis: { automargin: true, tickfont: { size: 9 } },
      margin: { l: 10, r: 16, t: 44, b: 44 },
    },
  };
}

// Community-flow Sankey across consecutive τ. Nodes sit in τ columns (node_x) and
// are coloured by their cross-τ community; links inherit the source community's
// colour so a community's thread is traceable as it merges/splits. τ values are
// annotated above their columns.
export function buildSankey(spec: PlotSpec): Figure {
  const taus: number[] = spec.taus ?? [];
  const T = spec.n_steps ?? taus.length;
  const nodeX: number[] | undefined = spec.node_x;
  const annotations = taus.map((t, i) => ({
    xref: "paper", yref: "paper",
    x: T > 1 ? i / (T - 1) : 0.5, y: 1.04,
    xanchor: "center", yanchor: "bottom", showarrow: false,
    text: `τ=${t.toPrecision(2)}`, font: { size: 10, color: "#555" },
  }));
  return {
    data: [
      {
        type: "sankey",
        orientation: "h",
        arrangement: nodeX ? "snap" : "freeform",
        node: {
          label: spec.node_labels ?? [],
          ...(nodeX ? { x: nodeX, y: nodeX.map(() => 0.5) } : {}),
          color: spec.node_color ?? undefined,
          pad: 12,
          thickness: 14,
          line: { color: "#bbb", width: 0.5 },
          hovertemplate: "%{label}<br>%{value} nodes<extra></extra>",
        },
        link: {
          source: spec.source ?? [],
          target: spec.target ?? [],
          value: spec.value ?? [],
          color: spec.link_color ?? undefined,
          hovertemplate: "%{source.label} → %{target.label}<br>%{value} nodes<extra></extra>",
        },
      },
    ],
    layout: {
      title: { text: "community flow (Sankey) across τ" },
      height: 560,
      margin: { l: 10, r: 10, t: 68, b: 20 },
      annotations,
    },
  };
}
