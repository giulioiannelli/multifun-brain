// Interactive correlation heatmap. Hovering a cell shows the Schaefer atlas
// region names (rows/cols) and the correlation value — the headline interaction.
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";
import type { HeatmapSpec } from "../../types";

const Plot = createPlotlyComponent(Plotly);

export function Heatmap({ spec }: { spec: HeatmapSpec }) {
  if (spec.error) {
    return <div className="plot-error">heatmap unavailable: {spec.error}</div>;
  }

  const data = [
    {
      type: "heatmap",
      z: spec.z,
      x: spec.names,
      y: spec.names,
      zmin: spec.zmin,
      zmax: spec.zmax,
      colorscale: "RdBu",
      reversescale: true,
      colorbar: { title: { text: "r", side: "right" }, thickness: 14 },
      hovertemplate:
        "row: <b>%{y}</b><br>col: <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    },
  ];

  const layout = {
    title: { text: spec.label },
    autosize: true,
    height: 680,
    margin: { l: 90, r: 30, t: 50, b: 90 },
    xaxis: { tickfont: { size: 7 }, showticklabels: spec.n <= 120, tickangle: 90 },
    yaxis: {
      tickfont: { size: 7 },
      showticklabels: spec.n <= 120,
      autorange: "reversed",
    },
  };

  return (
    <Plot
      data={data}
      layout={layout}
      useResizeHandler
      style={{ width: "100%" }}
      config={{ displaylogo: false, responsive: true, toImageButtonOptions: { format: "svg" } }}
    />
  );
}
