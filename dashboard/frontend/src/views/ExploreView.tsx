// Single-result explorer for the Correlation / Network / LRG tabs (the active
// tab is owned by App). Network and LRG expose a filter selector; LRG adds a tau
// selector for the dendrogram.
import { useEffect, useState } from "react";
import { Brain3D } from "../components/plots/Brain3D";
import { PlotPanel } from "../components/PlotPanel";
import { usePlot } from "../hooks/usePlot";
import type { QueryParams, ResultItem } from "../types";

export type ExploreTab = "Correlation" | "Network" | "Brain 3-D" | "LRG";

// In-panel toggles for a histogram: linear/log count axis + KDE overlay.
function HistToggles({
  yLog,
  setYLog,
  kde,
  setKde,
}: {
  yLog: boolean;
  setYLog: (v: boolean) => void;
  kde: boolean;
  setKde: (v: boolean) => void;
}) {
  return (
    <>
      <div className="seg">
        <span>y</span>
        <button className={!yLog ? "active" : ""} onClick={() => setYLog(false)}>linear</button>
        <button className={yLog ? "active" : ""} onClick={() => setYLog(true)}>log</button>
      </div>
      <div className="seg">
        <button className={kde ? "active" : ""} onClick={() => setKde(!kde)}>KDE</button>
      </div>
    </>
  );
}

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
  tab,
  datasetId,
  label,
  item,
}: {
  tab: ExploreTab;
  datasetId: string | null;
  label: string | null;
  item: ResultItem | null;
}) {
  const [filter, setFilter] = useState<string | null>(null);
  const [tauIndex, setTauIndex] = useState<number>(-1);
  const [brainMode, setBrainMode] = useState<"connectome" | "markers">("connectome");
  const [edgeQuantile, setEdgeQuantile] = useState<number>(0.98);
  const [logScale, setLogScale] = useState<boolean>(false);
  const [specYLog, setSpecYLog] = useState<boolean>(true);
  const [specKde, setSpecKde] = useState<boolean>(true);
  const [wtYLog, setWtYLog] = useState<boolean>(false);
  const [wtKde, setWtKde] = useState<boolean>(true);
  const [cleaned, setCleaned] = useState<boolean>(false);

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
  // Descriptive plots can be served raw or MP-cleaned (refetched, hence a param).
  const dparams: QueryParams = { ...base, cleaned: cleaned ? 1 : undefined };
  const tag = cleaned ? " · MP-cleaned" : "";

  return (
    <div className="explore">
      {(tab === "Network" || tab === "LRG" || tab === "Brain 3-D") && filters.length > 0 && (
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
          {tab === "Brain 3-D" && (
            <>
              <div className="seg">
                <span>View</span>
                <button
                  className={brainMode === "connectome" ? "active" : ""}
                  onClick={() => setBrainMode("connectome")}
                >
                  Connectome
                </button>
                <button
                  className={brainMode === "markers" ? "active" : ""}
                  onClick={() => setBrainMode("markers")}
                >
                  Markers
                </button>
              </div>
              {brainMode === "connectome" && (
                <label>
                  Edges
                  <select
                    value={edgeQuantile}
                    onChange={(e) => setEdgeQuantile(Number(e.target.value))}
                  >
                    <option value={0.9}>top 10%</option>
                    <option value={0.95}>top 5%</option>
                    <option value={0.98}>top 2%</option>
                    <option value={0.99}>top 1%</option>
                  </select>
                </label>
              )}
            </>
          )}
        </div>
      )}

      {tab === "Correlation" && (
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
          <div className="seg">
            <span>Data</span>
            <button className={!cleaned ? "active" : ""} onClick={() => setCleaned(false)}>
              Raw
            </button>
            <button className={cleaned ? "active" : ""} onClick={() => setCleaned(true)}>
              MP-cleaned
            </button>
          </div>
        </div>
      )}

      {tab === "Correlation" && (
        <div className="plot-grid">
          <PlotPanel
            kind="heatmap"
            title={`Correlation matrix${tag}`}
            params={dparams}
            square
            figureOptions={{ log: logScale }}
          />
          <PlotPanel
            kind="precision"
            title={`Precision matrix${tag}`}
            caption={
              cleaned
                ? "Inverse of the MP-cleaned correlation (diagonal hidden). Off-diagonal |Θ| = direct coupling between two regions once all others are accounted for."
                : "Inverse of the correlation matrix (Θ). Large |Θ| between two regions ⇒ strong direct coupling once every other region is accounted for; ≈ 0 ⇒ conditionally independent."
            }
            params={dparams}
            square
            figureOptions={{ log: logScale }}
          />
          <PlotPanel
            kind="weights"
            title={`Weight distribution${tag}`}
            params={dparams}
            figureOptions={{ yLog: wtYLog, kde: wtKde }}
            headerControls={
              <HistToggles yLog={wtYLog} setYLog={setWtYLog} kde={wtKde} setKde={setWtKde} />
            }
          />
          <PlotPanel
            kind="spectrum"
            title={`Eigenvalue spectrum${tag}`}
            caption={
              cleaned
                ? "Eigenvalues of the unit-diagonal correlation, with the Marchenko–Pastur noise density (red) overlaid and the bulk [λ−, λ+] (shaded) the cleaning flattens."
                : undefined
            }
            params={dparams}
            figureOptions={{ yLog: specYLog, kde: specKde }}
            headerControls={
              <HistToggles yLog={specYLog} setYLog={setSpecYLog} kde={specKde} setKde={setSpecKde} />
            }
          />
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

      {tab === "Brain 3-D" && (
        <div className="plot-grid">
          <section className="plot-card wide">
            <div className="plot-head">
              <h3>
                3-D brain · {brainMode}
                {filter ? ` · ${filter}` : ""}
              </h3>
            </div>
            <p className="plot-caption">
              {brainMode === "connectome"
                ? "Strongest edges of the filtered network drawn between Schaefer parcel centroids (MNI). Drag to rotate, scroll to zoom."
                : "Schaefer parcels coloured by their 7-network, on survivor centroids. Drag to rotate, scroll to zoom."}
            </p>
            <Brain3D
              dataset={datasetId}
              label={label}
              filter={filter}
              mode={brainMode}
              edgeQuantile={edgeQuantile}
            />
          </section>
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
