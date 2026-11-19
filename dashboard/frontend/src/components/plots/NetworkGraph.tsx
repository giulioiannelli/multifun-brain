// Interactive 2-D network (Cytoscape). Layout is either a server-computed preset
// (spring / Kamada-Kawai / spectral, shipped in spec.layouts) or a client-side
// Cytoscape layout (force / concentric / circle / grid / breadth-first). Nodes can
// be coloured and sized by atlas network, degree, or strength. Edges can be
// coloured by weight (diverging when the backbone keeps signs, sequential on |w|
// otherwise) and their width scaled by a selectable law (linear / sqrt / log /
// rank) so a heavy-tailed weight distribution doesn't collapse to one width.
// Colour/size/scale changes restyle in place (no re-layout); layout /
// sparsification changes remount for a fresh arrangement. The card is kept ~1:1
// (networks are roughly square), not a wide rectangle.
import { useEffect, useMemo, useRef } from "react";
import CytoscapeComponent from "react-cytoscapejs";
import type { Core } from "cytoscape";
import type { PlotSpec } from "../../types";

const SERVER_LAYOUTS = ["spring", "kamada", "spectral"];

const CLIENT_LAYOUTS: Record<string, Record<string, any>> = {
  cose: {
    name: "cose", animate: false, nodeRepulsion: () => 9000,
    idealEdgeLength: () => 45, nodeOverlap: 8, gravity: 0.3, numIter: 1200, padding: 24,
  },
  concentric: {
    name: "concentric", animate: false, concentric: (n: any) => (n.data("degree") || 1),
    levelWidth: () => 3, minNodeSpacing: 12, padding: 24,
  },
  circle: { name: "circle", animate: false, padding: 24 },
  grid: { name: "grid", animate: false, padding: 24 },
  breadthfirst: { name: "breadthfirst", animate: false, circle: false, spacingFactor: 1.1, padding: 24 },
};

// Sequential blue ramp (light -> dark) for degree / strength node colouring.
const RAMP_LO = [0xd6, 0xe6, 0xf4];
const RAMP_HI = [0x08, 0x30, 0x6b];
const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
function rampColor(t: number): string {
  const c = [0, 1, 2].map((i) => Math.round(RAMP_LO[i] + (RAMP_HI[i] - RAMP_LO[i]) * clamp01(t)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function hexToRgb(hex: string): [number, number, number] {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}
function mix(c1: [number, number, number], c2: [number, number, number], t: number): string {
  const u = clamp01(t);
  const c = [0, 1, 2].map((i) => Math.round(c1[i] + (c2[i] - c1[i]) * u));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
// Diverging RdBu-like: negative -> blue, 0 -> light grey, positive -> red.
const DIV_NEG = hexToRgb("#2166ac");
const DIV_MID = hexToRgb("#f7f7f7");
const DIV_POS = hexToRgb("#b2182b");
// Sequential warm (magma-ish) for |weight|, distinct from the node blue ramp.
const SEQ_LO = hexToRgb("#fcfdbf");
const SEQ_MID = hexToRgb("#fc8961");
const SEQ_HI = hexToRgb("#3b0f70");

function divergingColor(w: number, absMax: number): string {
  if (absMax <= 0) return "#9e9e9e";
  const t = clamp01(Math.abs(w) / absMax);
  return w >= 0 ? mix(DIV_MID, DIV_POS, t) : mix(DIV_MID, DIV_NEG, t);
}
function sequentialColor(t: number): string {
  const u = clamp01(t);
  return u < 0.5 ? mix(SEQ_LO, SEQ_MID, u / 0.5) : mix(SEQ_MID, SEQ_HI, (u - 0.5) / 0.5);
}

function nodeColor(
  id: string, netColor: string, degree: number, strength: number,
  colorBy: string, degMax: number, strMax: number,
  communityColors?: string[],
): string {
  if (colorBy === "community" && communityColors) return communityColors[Number(id)] ?? netColor;
  if (colorBy === "degree") return rampColor((degree || 0) / degMax);
  if (colorBy === "strength") return rampColor((strength || 0) / strMax);
  return netColor;
}

function nodeSize(
  degree: number, strength: number, sizeBy: string,
  nodeScale: number, degMax: number, strMax: number,
): number {
  const base =
    sizeBy === "uniform" ? 0.45
    : sizeBy === "degree" ? (degree || 0) / degMax
    : (strength || 0) / strMax;
  return (8 + 30 * base) * nodeScale;
}

// Normalise |w| to [0,1] under the chosen law. `rankOf` maps |w| -> its percentile
// among the drawn edges (built once), so `rank` spreads a heavy-tailed weight
// distribution evenly across widths instead of piling every edge near the floor.
function weightNorm(
  w: number, mode: string, absMin: number, absMax: number,
  rankOf: (a: number) => number,
): number {
  const a = Math.abs(w);
  if (mode === "rank") return rankOf(a);
  if (absMax <= absMin) return 0.5;
  if (mode === "log") {
    const lo = Math.max(absMin, absMax * 1e-3);
    if (a <= lo) return 0;
    return clamp01(Math.log(a / lo) / Math.log(absMax / lo));
  }
  const lin = clamp01((a - absMin) / (absMax - absMin));
  return mode === "sqrt" ? Math.sqrt(lin) : lin;
}

const edgeWidth = (t: number, edgeScale: number) => (0.5 + 4.5 * clamp01(t)) * edgeScale;

function edgeColor(
  w: number, colorBy: string, hasNeg: boolean, absMin: number, absMax: number,
  rankOf: (a: number) => number,
): string {
  if (colorBy !== "weight") return "#cfd8dc";
  if (hasNeg) return divergingColor(w, absMax);
  return sequentialColor(weightNorm(w, "linear", absMin, absMax, rankOf));
}

export function NetworkGraph({
  spec,
  options = {},
}: {
  spec: PlotSpec;
  options?: Record<string, any>;
}) {
  const cyRef = useRef<Core | null>(null);
  const layout: string = options.layout ?? "spring";
  const colorBy: string = options.colorBy ?? "network";
  const sizeBy: string = options.sizeBy ?? "strength";
  const nodeScale: number = options.nodeScale ?? 1;
  const edgeScale: number = options.edgeScale ?? 1;
  const edgeColorBy: string = options.edgeColorBy ?? "weight";
  const edgeScaleMode: string = options.edgeScaleMode ?? "sqrt";
  const communityColors: string[] | undefined = options.communityColors;

  const nodes = (spec.nodes ?? []) as any[];
  const edges = (spec.edges ?? []) as any[];
  const degMax = Math.max(1, ...nodes.map((n) => n.degree || 0));
  const strMax = Math.max(1e-9, ...nodes.map((n) => Math.abs(n.strength) || 0));
  const hasNeg: boolean = spec.has_negative ?? edges.some((e) => (e.weight ?? 0) < 0);
  const absMax = Math.max(1e-9, spec.w_absmax ?? Math.max(1e-9, ...edges.map((e) => Math.abs(e.weight) || 0)));
  const absMin = spec.w_absmin ?? Math.min(...edges.map((e) => Math.abs(e.weight) || 0), absMax);

  // Percentile-rank lookup over the drawn |weights| (for the "rank" width law).
  const rankOf = useMemo(() => {
    const sorted = edges.map((e) => Math.abs(e.weight) || 0).sort((a, b) => a - b);
    const n = sorted.length;
    return (a: number): number => {
      if (n <= 1) return 0.5;
      let lo = 0, hi = n;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (sorted[mid] < a) lo = mid + 1; else hi = mid; }
      return lo / (n - 1);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [spec.label, spec.filter, spec.sparsify, spec.n_edges_shown]);

  // "lrg" positions come straight from the spec nodes' x/y (server MDS layout).
  const isPreset = SERVER_LAYOUTS.includes(layout) || layout === "lrg";
  const positions: number[][] | undefined =
    spec.layouts?.[layout] ??
    (layout === "lrg" && nodes.length && nodes[0].x != null
      ? nodes.map((n) => [Number(n.x) * 300, Number(n.y) * 300])
      : undefined);

  // Restyle in place when a colour/size/scale control changes (no re-layout).
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.nodes().forEach((node) => {
        const d = node.data();
        node.data("color", nodeColor(d.id, d.netColor, d.degree, d.strength, colorBy, degMax, strMax, communityColors));
        node.data("size", nodeSize(d.degree, d.strength, sizeBy, nodeScale, degMax, strMax));
      });
      cy.edges().forEach((e) => {
        const w = e.data("weight");
        e.data("width", edgeWidth(weightNorm(w, edgeScaleMode, absMin, absMax, rankOf), edgeScale));
        e.data("ecolor", edgeColor(w, edgeColorBy, hasNeg, absMin, absMax, rankOf));
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorBy, sizeBy, nodeScale, edgeScale, edgeColorBy, edgeScaleMode, degMax, strMax, absMin, absMax, hasNeg, rankOf, communityColors]);

  if (spec.error) return <div className="plot-error">network: {spec.error}</div>;

  const elements = [
    ...nodes.map((n, i) => ({
      data: {
        id: n.id,
        label: n.name,
        network: n.network,
        netColor: n.color,
        degree: n.degree || 0,
        strength: Math.abs(n.strength) || 0,
        color: nodeColor(n.id, n.color, n.degree, Math.abs(n.strength), colorBy, degMax, strMax, communityColors),
        size: nodeSize(n.degree, Math.abs(n.strength), sizeBy, nodeScale, degMax, strMax),
      },
      ...(isPreset && positions
        ? { position: { x: positions[i][0], y: positions[i][1] } }
        : {}),
    })),
    ...edges.map((e) => ({
      data: {
        source: e.source,
        target: e.target,
        weight: e.weight ?? 0,
        width: edgeWidth(weightNorm(e.weight, edgeScaleMode, absMin, absMax, rankOf), edgeScale),
        ecolor: edgeColor(e.weight, edgeColorBy, hasNeg, absMin, absMax, rankOf),
      },
    })),
  ];

  const stylesheet = [
    {
      selector: "node",
      style: {
        "background-color": "data(color)",
        width: "data(size)",
        height: "data(size)",
        label: "data(label)",
        "font-size": 5,
        color: "#333",
        "text-valign": "center",
        "text-halign": "center",
        "text-opacity": 0,
        "border-width": 0.5,
        "border-color": "#ffffff",
      },
    },
    { selector: "node:selected", style: { "text-opacity": 1, "border-width": 2, "border-color": "#1976D2" } },
    {
      selector: "edge",
      style: {
        "line-color": "data(ecolor)",
        width: "data(width)",
        "curve-style": "haystack",
        opacity: edgeColorBy === "weight" ? 0.7 : 0.45,
      },
    },
  ];

  const cyLayout = isPreset
    ? { name: "preset", padding: 24, fit: true }
    : CLIENT_LAYOUTS[layout] ?? CLIENT_LAYOUTS.cose;

  // Remount (fresh layout) only on structural changes — not on restyle. `tau`
  // captures the LRG-layout embedding changing as τ moves (positions differ).
  const structKey = [
    spec.label, spec.filter, spec.sparsify, layout, spec.n_nodes, spec.n_edges_shown, spec.tau,
  ].join("|");

  const edgeLegend =
    edgeColorBy === "weight"
      ? hasNeg
        ? " · edges: blue −w → red +w"
        : " · edges: light → dark |w|"
      : "";

  return (
    <div>
      <div className="plot-caption">
        {spec.n_edges_shown} of {spec.n_edges_total} edges · {spec.n_nodes} nodes ·
        {" "}sparsify: <b>{spec.sparsify}</b> · layout: <b>{layout}</b> · width law: <b>{edgeScaleMode}</b>
        {edgeLegend} · click a node for its name
      </div>
      <div className="network-square">
        <CytoscapeComponent
          key={structKey}
          cy={(cy: Core) => {
            cyRef.current = cy;
          }}
          elements={elements}
          stylesheet={stylesheet as any}
          layout={cyLayout as any}
          style={{ width: "100%", height: "100%", background: "#fff", border: "1px solid #eee", borderRadius: "6px" }}
          minZoom={0.2}
          maxZoom={4}
        />
      </div>
    </div>
  );
}
