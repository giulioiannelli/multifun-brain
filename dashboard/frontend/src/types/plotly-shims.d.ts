// Minimal shims: the plotly.js partial-bundle entry points ship no types, and
// we build the React component via the factory. Loose typing is fine here.
declare module "plotly.js/lib/core";
declare module "plotly.js/lib/bar";
declare module "plotly.js/lib/heatmap";
declare module "plotly.js/lib/histogram";
declare module "plotly.js/lib/sankey";
declare module "plotly.js/lib/scatter";
declare module "react-plotly.js/factory" {
  import type { ComponentType } from "react";
  const createPlotlyComponent: (plotly: unknown) => ComponentType<any>;
  export default createPlotlyComponent;
}
