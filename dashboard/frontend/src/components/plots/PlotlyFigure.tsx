// Generic Plotly wrapper (factory + custom slim Plotly bundle). Plot builders
// produce { data, layout }; this renders them responsively.
import createPlotlyComponent from "react-plotly.js/factory";
import Plotly from "../../lib/plotly";

const Plot = createPlotlyComponent(Plotly);

export function PlotlyFigure({
  data,
  layout = {},
  height = 420,
  fill = false,
}: {
  data: any[];
  layout?: Record<string, any>;
  height?: number;
  // fill: size to the parent container (which sets the height, e.g. a square
  // aspect-ratio wrapper) instead of using a fixed pixel height.
  fill?: boolean;
}) {
  const { height: layoutHeight, margin: layoutMargin, ...restLayout } = layout;
  return (
    <Plot
      data={data}
      layout={{
        autosize: true,
        margin: layoutMargin ?? { l: 55, r: 20, t: 44, b: 44 },
        font: { size: 12 },
        ...restLayout,
        ...(fill ? {} : { height: layoutHeight ?? height }),
      }}
      useResizeHandler
      style={fill ? { width: "100%", height: "100%" } : { width: "100%" }}
      config={{
        displaylogo: false,
        responsive: true,
        toImageButtonOptions: { format: "svg" },
      }}
    />
  );
}
