// One self-contained plot card: fetches its spec for (kind, params) and
// dispatches to the matching renderer (Plotly figure / Cytoscape / table).
import { usePlot } from "../hooks/usePlot";
import type { PlotSpec, QueryParams } from "../types";
import { NetworkGraph } from "./plots/NetworkGraph";
import { PlotlyFigure } from "./plots/PlotlyFigure";
import { GlobalMetrics, NodeMetrics } from "./plots/Tables";
import * as F from "./plots/figures";

type Builder = (spec: PlotSpec) => F.Figure;

const FIGURE_BUILDERS: Record<string, Builder> = {
  heatmap: (s) => F.buildMatrix(s, "correlation matrix"),
  partial_correlation: (s) => F.buildMatrix(s, "partial correlation"),
  weights: F.buildWeights,
  spectrum: F.buildSpectrum,
  signed_laplacian: F.buildSignedLaplacian,
  signed_balance: F.buildSignedBalance,
  degree_distribution: F.buildDegree,
  dendrogram: F.buildDendrogram,
  partition_flow: F.buildPartitionFlow,
  sankey: F.buildSankey,
};

export function PlotPanel({
  kind,
  title,
  params,
  wide = false,
}: {
  kind: string;
  title: string;
  params: QueryParams;
  wide?: boolean;
}) {
  const { spec, loading, error } = usePlot(kind, params);

  function body() {
    if (error) return <div className="plot-error">{error}</div>;
    if (loading && !spec) return <div className="hint">Loading…</div>;
    if (!spec) return null;
    if (spec.error) return <div className="plot-error">{spec.error}</div>;

    if (kind === "network") return <NetworkGraph spec={spec} />;
    if (kind === "global_metrics") return <GlobalMetrics spec={spec} />;
    if (kind === "node_metrics") return <NodeMetrics spec={spec} />;

    const builder = FIGURE_BUILDERS[kind];
    if (builder) {
      const fig = builder(spec);
      return <PlotlyFigure data={fig.data} layout={fig.layout} height={fig.layout.height ?? 420} />;
    }
    return <pre className="raw-spec">{JSON.stringify(spec, null, 2)}</pre>;
  }

  return (
    <section className={`plot-card${wide ? " wide" : ""}`}>
      <h3>{title}</h3>
      {body()}
    </section>
  );
}
