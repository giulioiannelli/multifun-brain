// Single-result explorer with Descriptive / Network / LRG tabs. Network and LRG
// tabs expose a filter selector; LRG adds a tau selector for the dendrogram.
import { useEffect, useState } from "react";
import { PlotPanel } from "../components/PlotPanel";
import { usePlot } from "../hooks/usePlot";
import type { QueryParams, ResultItem } from "../types";

const TABS = ["Descriptive", "Network", "LRG"] as const;
type Tab = (typeof TABS)[number];

function TauSelector({
  params,
  value,
  onChange,
}: {
  params: QueryParams;
  value: number;
  onChange: (i: number) => void;
}) {
  const { spec } = usePlot("tau_grid", params);
  const taus: number[] = spec?.taus ?? [];
  if (!taus.length) return null;
  const idx = value < 0 ? taus.length + value : value;
  return (
    <label>
      τ step
      <select value={idx} onChange={(e) => onChange(Number(e.target.value))}>
        {taus.map((t, i) => (
          <option key={i} value={i}>
            {t.toPrecision(3)} · {spec?.n_clusters?.[i]} cl
          </option>
        ))}
      </select>
    </label>
  );
}

export function ExploreView({
  datasetId,
  label,
  item,
}: {
  datasetId: string | null;
  label: string | null;
  item: ResultItem | null;
}) {
  const [tab, setTab] = useState<Tab>("Descriptive");
  const [filter, setFilter] = useState<string | null>(null);
  const [tauIndex, setTauIndex] = useState<number>(-1);
  const [logScale, setLogScale] = useState<boolean>(false);

  const filters = item?.filters ?? [];
  useEffect(() => {
    if (filters.length && (filter === null || !filters.includes(filter))) {
      setFilter(filters[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, filters.join(",")]);

  if (!datasetId || !label) return <div className="hint">Select a result.</div>;
  if (item?.error) return <div className="plot-error">This result failed: {item.error}</div>;

  const base: QueryParams = { dataset: datasetId, label };
  const fparams: QueryParams = { ...base, filter: filter ?? undefined };

  return (
    <div className="explore">
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {(tab === "Network" || tab === "LRG") && filters.length > 0 && (
        <div className="subbar">
          <label>
            Filter
            <select value={filter ?? ""} onChange={(e) => setFilter(e.target.value)}>
              {filters.map((f) => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </label>
          {tab === "LRG" && (
            <TauSelector params={fparams} value={tauIndex} onChange={setTauIndex} />
          )}
        </div>
      )}

      {tab === "Descriptive" && (
        <div className="subbar">
          <div className="seg">
            <span>Matrix colour scale</span>
            <button className={!logScale ? "active" : ""} onClick={() => setLogScale(false)}>
              Linear
            </button>
            <button className={logScale ? "active" : ""} onClick={() => setLogScale(true)}>
              Log
            </button>
          </div>
        </div>
      )}

      {tab === "Descriptive" && (
        <div className="plot-grid">
          <PlotPanel
            kind="heatmap"
            title="Correlation matrix"
            params={base}
            wide
            figureOptions={{ log: logScale }}
          />
          <PlotPanel
            kind="partial_correlation"
            title="Partial correlation"
            params={base}
            wide
            figureOptions={{ log: logScale }}
          />
          <PlotPanel kind="spectrum" title="Eigenvalue spectrum" params={base} />
          <PlotPanel kind="weights" title="Weight distribution" params={base} />
          <PlotPanel kind="signed_laplacian" title="Signed Laplacian" params={base} />
          <PlotPanel kind="signed_balance" title="Signed balance" params={base} />
        </div>
      )}

      {tab === "Network" && (
        <div className="plot-grid">
          <PlotPanel kind="global_metrics" title="Global metrics" params={fparams} />
          <PlotPanel kind="degree_distribution" title="Degree distribution" params={fparams} />
          <PlotPanel kind="network" title="Network graph" params={fparams} wide />
          <PlotPanel kind="node_metrics" title="Node metrics" params={fparams} wide />
        </div>
      )}

      {tab === "LRG" && (
        <div className="plot-grid">
          <PlotPanel
            kind="dendrogram"
            title="Dendrogram"
            params={{ ...fparams, tau_index: tauIndex }}
            wide
          />
          <PlotPanel kind="partition_flow" title="Partition flow" params={fparams} wide />
          <PlotPanel kind="sankey" title="Community flow" params={fparams} wide />
        </div>
      )}
    </div>
  );
}
