// One self-contained plot card. Fetch + render are deferred until the card
// scrolls into view (keeps initial paint fast). Dispatches the fetched spec to
// the matching renderer (Plotly figure / Cytoscape / table). Cytoscape is
// lazy-loaded so non-network tabs never download it.
import { lazy, Suspense } from "react";
import { useInView } from "../hooks/useInView";
import { usePlot } from "../hooks/usePlot";
import type { PlotSpec, QueryParams } from "../types";
import { PlotlyFigure } from "./plots/PlotlyFigure";
import { GlobalMetrics, NodeMetrics } from "./plots/Tables";
import * as F from "./plots/figures";

const NetworkGraph = lazy(() =>
  import("./plots/NetworkGraph").then((m) => ({ default: m.NetworkGraph })),
);

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

function PanelBody({ kind, params }: { kind: string; params: QueryParams }) {
  const { spec, loading, error } = usePlot(kind, params);
  if (error) return <div className="plot-error">{error}</div>;
  if (loading && !spec) return <div className="hint">Loading…</div>;
  if (!spec) return null;
  if (spec.error) return <div className="plot-error">{spec.error}</div>;

  if (kind === "network")
    return (
      <Suspense fallback={<div className="hint">Loading graph…</div>}>
        <NetworkGraph spec={spec} />
      </Suspense>
    );
  if (kind === "global_metrics") return <GlobalMetrics spec={spec} />;
  if (kind === "node_metrics") return <NodeMetrics spec={spec} />;

  const builder = FIGURE_BUILDERS[kind];
  if (builder) {
    const fig = builder(spec);
    return <PlotlyFigure data={fig.data} layout={fig.layout} height={fig.layout.height ?? 420} />;
  }
  return <pre className="raw-spec">{JSON.stringify(spec, null, 2)}</pre>;
}

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
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <section ref={ref} className={`plot-card${wide ? " wide" : ""}`}>
      <h3>{title}</h3>
      {inView ? <PanelBody kind={kind} params={params} /> : <div className="hint">…</div>}
    </section>
  );
}
