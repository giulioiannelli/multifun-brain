// Generic Plotly wrapper (factory + custom slim Plotly bundle). Plot builders
// produce { data, layout }; this renders them responsively.
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "../../lib/plotly";

const Plot = createPlotlyComponent(Plotly);

export function PlotlyFigure({
  data,
  layout,
  height = 420,
}: {
  data: any[];
  layout?: Record<string, any>;
  height?: number;
}) {
  return (
    <Plot
      data={data}
      layout={{
        autosize: true,
        height,
        margin: { l: 55, r: 20, t: 44, b: 44 },
        font: { size: 12 },
        ...layout,
      }}
      useResizeHandler
      style={{ width: "100%" }}
      config={{
        displaylogo: false,
        responsive: true,
        toImageButtonOptions: { format: "svg" },
      }}
    />
  );
}
