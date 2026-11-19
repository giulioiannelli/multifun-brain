// Custom Plotly bundle: register ONLY the trace types the dashboard uses.
// This is the difference between shipping the full ~3.5 MB plotly.js-dist and a
// ~1 MB core build — far faster to download, parse, and rebuild.
import Plotly from "plotly.js/lib/core";
import bar from "plotly.js/lib/bar";
import heatmap from "plotly.js/lib/heatmap";
import histogram from "plotly.js/lib/histogram";
import sankey from "plotly.js/lib/sankey";
import scatter from "plotly.js/lib/scatter";

Plotly.register([heatmap, bar, histogram, scatter, sankey]);

export default Plotly;
