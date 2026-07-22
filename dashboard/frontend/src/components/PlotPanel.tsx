// One self-contained plot card. Fetch + render are deferred until the card
// scrolls into view (keeps initial paint fast). Dispatches the fetched spec to
// the matching renderer (Plotly figure / Cytoscape / table). Cytoscape is
// lazy-loaded so non-network tabs never download it.
import { lazy, Suspense, type ReactNode } from "react";
import type { PlotEndpoint } from "../api/client";
import { useInView } from "../hooks/useInView";
import { usePlot } from "../hooks/usePlot";
import type { PlotSpec, QueryParams } from "../types";
import { PlotlyFigure } from "./plots/PlotlyFigure";
import { GlobalMetrics, NodeMetrics } from "./plots/Tables";
import * as F from "./plots/figures";

const NetworkGraph = lazy(() =>
  import("./plots/NetworkGraph").then((m) => ({ default: m.NetworkGraph })),
);

type Builder = (spec: PlotSpec, opts?: Record<string, any>) => F.Figure;

const FIGURE_BUILDERS: Record<string, Builder> = {
  heatmap: (s, o) => F.buildMatrix(s, "correlation matrix", o),
  partial_correlation: (s, o) => F.buildMatrix(s, "partial correlation", o),
  precision: (s, o) => F.buildMatrix(s, "precision matrix", o),
  weights: F.buildWeights,
  spectrum: F.buildSpectrum,
  signed_laplacian: F.buildSignedLaplacian,
  signed_balance: F.buildSignedBalance,
  degree_distribution: F.buildDegree,
  dendrogram: (s) => F.buildDendrogram(s), // interactive cut lives in LrgView; here: natural cut
  specific_heat: F.buildSpecificHeat,
  psi: F.buildPsi,
  partition_flow: F.buildPartitionFlow,
  sankey: F.buildSankey,
  // Signal tab (raw timecourses)
  signal_heatmap: F.buildSignalHeatmap,
  signal_channel: F.buildSignalChannel,
  signal_emd: F.buildSignalEMD,
  cohort_bands: F.buildCohortBands,
  signal_bands: F.buildBandReconstruction,
};

function PanelBody({
  kind,
  params,
  figureOptions,
  square,
  endpoint,
}: {
  kind: string;
  params: QueryParams;
  figureOptions?: Record<string, any>;
  square?: boolean;
  endpoint?: PlotEndpoint;
}) {
  const { spec, loading, error } = usePlot(kind, params, { endpoint });
  if (error) return <div className="plot-error">{error}</div>;
  if (loading && !spec) return <div className="hint">Loading…</div>;
  if (!spec) return null;
  if (spec.error) return <div className="plot-error">{spec.error}</div>;

  if (kind === "network")
    return (
      <Suspense fallback={<div className="hint">Loading graph…</div>}>
        <NetworkGraph spec={spec} options={figureOptions} />
      </Suspense>
    );
  if (kind === "global_metrics") return <GlobalMetrics spec={spec} />;
  if (kind === "node_metrics") return <NodeMetrics spec={spec} />;

  const builder = FIGURE_BUILDERS[kind];
  if (builder) {
    const fig = builder(spec, figureOptions);
    if (square) {
      return (
        <div className="plot-square">
          <PlotlyFigure data={fig.data} layout={fig.layout} fill />
        </div>
      );
    }
    return <PlotlyFigure data={fig.data} layout={fig.layout} height={fig.layout.height ?? 420} />;
  }
  return <pre className="raw-spec">{JSON.stringify(spec, null, 2)}</pre>;
}

export function PlotPanel({
  kind,
  title,
  params,
  wide = false,
  square = false,
  figureOptions,
  caption,
  headerControls,
  endpoint,
}: {
  kind: string;
  title: string;
  params: QueryParams;
  wide?: boolean;
  square?: boolean;
  figureOptions?: Record<string, any>;
  caption?: string;
  headerControls?: ReactNode;
  endpoint?: PlotEndpoint;
}) {
  const [ref, inView] = useInView<HTMLDivElement>();
  return (
    <section ref={ref} className={`plot-card${wide ? " wide" : ""}`}>
      <div className="plot-head">
        <h3>{title}</h3>
        {headerControls && <div className="plot-controls">{headerControls}</div>}
      </div>
      {caption && <p className="plot-caption">{caption}</p>}
      {inView ? (
        <PanelBody
          kind={kind}
          params={params}
          figureOptions={figureOptions}
          square={square}
          endpoint={endpoint}
        />
      ) : (
        <div className="hint">…</div>
      )}
    </section>
  );
}
