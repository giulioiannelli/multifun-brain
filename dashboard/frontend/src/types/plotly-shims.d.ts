// Minimal shims: plotly.js-dist-min ships no types, and we build the React
// component via the factory. Loose typing is fine for our plot specs.
declare module "plotly.js-dist-min";
declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  const createPlotlyComponent: (plotly: unknown) => ComponentType<any>;
  export default createPlotlyComponent;
}
