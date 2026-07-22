// Interactive 2-D network (Cytoscape). Layout is either a server-computed preset
// (spring / Kamada-Kawai / spectral, shipped in spec.layouts) or a client-side
// Cytoscape layout (force / concentric / circle / grid / breadth-first). Nodes can
// be coloured and sized by atlas network, degree, or strength; edge width scales
// with |weight|. Colour/size changes restyle in place (no re-layout); layout /
// sparsification changes remount for a fresh arrangement.
import { useEffect, useRef } from "react";
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

// Sequential blue ramp (light -> dark) for degree / strength colouring.
const RAMP_LO = [0xd6, 0xe6, 0xf4];
const RAMP_HI = [0x08, 0x30, 0x6b];
const clamp01 = (t: number) => (t < 0 ? 0 : t > 1 ? 1 : t);
function rampColor(t: number): string {
  const c = [0, 1, 2].map((i) => Math.round(RAMP_LO[i] + (RAMP_HI[i] - RAMP_LO[i]) * clamp01(t)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

function nodeColor(
  netColor: string, degree: number, strength: number,
  colorBy: string, degMax: number, strMax: number,
): string {
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

const edgeWidth = (w: number, edgeScale: number, wMax: number) =>
  (0.4 + 3.0 * (Math.abs(w) / wMax)) * edgeScale;

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

  const nodes = (spec.nodes ?? []) as any[];
  const edges = (spec.edges ?? []) as any[];
  const degMax = Math.max(1, ...nodes.map((n) => n.degree || 0));
  const strMax = Math.max(1e-9, ...nodes.map((n) => Math.abs(n.strength) || 0));
  const wMax = Math.max(1e-9, ...edges.map((e) => Math.abs(e.weight) || 0));

  const isPreset = SERVER_LAYOUTS.includes(layout);
  const positions: number[][] | undefined = spec.layouts?.[layout];

  // Restyle in place when a colour/size/scale control changes (no re-layout).
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.batch(() => {
      cy.nodes().forEach((node) => {
        const d = node.data();
        node.data("color", nodeColor(d.netColor, d.degree, d.strength, colorBy, degMax, strMax));
        node.data("size", nodeSize(d.degree, d.strength, sizeBy, nodeScale, degMax, strMax));
      });
      cy.edges().forEach((e) => {
        e.data("width", edgeWidth(e.data("weight"), edgeScale, wMax));
      });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [colorBy, sizeBy, nodeScale, edgeScale, degMax, strMax, wMax]);

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
        color: nodeColor(n.color, n.degree, Math.abs(n.strength), colorBy, degMax, strMax),
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
        weight: Math.abs(e.weight) || 0,
        width: edgeWidth(e.weight, edgeScale, wMax),
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
        "line-color": "#cfd8dc",
        width: "data(width)",
        "curve-style": "haystack",
        opacity: 0.45,
      },
    },
  ];

  const cyLayout = isPreset
    ? { name: "preset", padding: 24, fit: true }
    : CLIENT_LAYOUTS[layout] ?? CLIENT_LAYOUTS.cose;

  // Remount (fresh layout) only on structural changes — not on restyle.
  const structKey = [
    spec.label, spec.filter, spec.sparsify, layout, spec.n_nodes, spec.n_edges_shown,
  ].join("|");

  return (
    <div>
      <div className="plot-caption">
        {spec.n_edges_shown} of {spec.n_edges_total} edges · {spec.n_nodes} nodes ·
        {" "}sparsify: <b>{spec.sparsify}</b> · layout: <b>{layout}</b> · click a node for its name
      </div>
      <CytoscapeComponent
        key={structKey}
        cy={(cy: Core) => {
          cyRef.current = cy;
        }}
        elements={elements}
        stylesheet={stylesheet as any}
        layout={cyLayout as any}
        style={{ width: "100%", height: "560px", background: "#fff", border: "1px solid #eee", borderRadius: "6px" }}
        minZoom={0.2}
        maxZoom={4}
      />
    </div>
  );
}
