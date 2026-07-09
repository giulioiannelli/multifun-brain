// Interactive 2-D network (Cytoscape): preset server-computed layout, nodes
// coloured by atlas network and sized by strength, hover/drag/zoom enabled.
import CytoscapeComponent from "react-cytoscapejs";
import type { PlotSpec } from "../../types";

export function NetworkGraph({ spec }: { spec: PlotSpec }) {
  if (spec.error) return <div className="plot-error">network: {spec.error}</div>;

  const strengths: number[] = (spec.nodes ?? []).map((n: any) => Math.abs(n.strength) || 1);
  const sMax = Math.max(1, ...strengths);

  const elements = [
    ...(spec.nodes ?? []).map((n: any) => ({
      data: {
        id: n.id,
        label: n.name,
        color: n.color,
        network: n.network,
        degree: n.degree,
        size: 8 + 26 * (Math.abs(n.strength) / sMax),
      },
      position: { x: n.x, y: n.y },
    })),
    ...(spec.edges ?? []).map((e: any) => ({
      data: { source: e.source, target: e.target, weight: e.weight },
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
      },
    },
    { selector: "node:selected", style: { "text-opacity": 1, "border-width": 2, "border-color": "#1976D2" } },
    {
      selector: "edge",
      style: {
        "line-color": "#cfd8dc",
        width: "mapData(weight, 0, 1, 0.3, 2.5)",
        "curve-style": "haystack",
        opacity: 0.5,
      },
    },
  ];

  return (
    <div>
      <div className="plot-caption">
        {spec.n_edges_shown} of {spec.n_edges_total} edges (top {Math.round((1 - spec.edge_quantile) * 100)}% by |weight|) · click a node for its name
      </div>
      <CytoscapeComponent
        elements={elements}
        stylesheet={stylesheet as any}
        layout={{ name: "preset" }}
        style={{ width: "100%", height: "560px", background: "#fff", border: "1px solid #eee", borderRadius: "6px" }}
        minZoom={0.2}
        maxZoom={4}
      />
    </div>
  );
}
