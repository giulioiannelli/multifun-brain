// Single-result explorer for the Correlation / Network / LRG tabs (the active
// tab is owned by App). Network and LRG expose a filter selector; LRG adds a tau
// selector for the dendrogram.
import { useEffect, useState } from "react";
import { Brain3D } from "../components/plots/Brain3D";
import { ChannelExcluder } from "../components/ChannelExcluder";
import { PlotPanel } from "../components/PlotPanel";
import { LrgView } from "./LrgView";
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
  const [brainMode, setBrainMode] = useState<"connectome" | "markers">("connectome");
  const [edgeQuantile, setEdgeQuantile] = useState<number>(0.98);
  const [brainNodeSize, setBrainNodeSize] = useState<number>(9);
  const [logScale, setLogScale] = useState<boolean>(false);
  const [specYLog, setSpecYLog] = useState<boolean>(true);
  const [specKde, setSpecKde] = useState<boolean>(true);
  const [wtYLog, setWtYLog] = useState<boolean>(false);
  const [wtKde, setWtKde] = useState<boolean>(true);
  const [cleaned, setCleaned] = useState<boolean>(false);
  const [excluded, setExcluded] = useState<number[]>([]);
  // Network graph controls (client-side except sparsify/edges, which refetch).
  const [sparsify, setSparsify] = useState<string>("filter");
  const [sparsifyAlpha, setSparsifyAlpha] = useState<number>(0.05);
  const [sparsifyThreshold, setSparsifyThreshold] = useState<number>(0.3);
  const [netEdgeQ, setNetEdgeQ] = useState<number>(0.9);
  const [netLayout, setNetLayout] = useState<string>("spring");
  const [netColorBy, setNetColorBy] = useState<string>("network");
  const [netSizeBy, setNetSizeBy] = useState<string>("strength");
  const [netNodeScale, setNetNodeScale] = useState<number>(1);
  const [netEdgeScale, setNetEdgeScale] = useState<number>(1);
  const [netEdgeColorBy, setNetEdgeColorBy] = useState<string>("weight");
  const [netEdgeScaleMode, setNetEdgeScaleMode] = useState<string>("sqrt");

  const filters = item?.filters ?? [];
  useEffect(() => {
    if (filters.length && (filter === null || !filters.includes(filter))) {
      setFilter(filters[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [label, filters.join(",")]);

  // Full region-name list for the channel-exclusion selector (result-specific).
  const namesParams: QueryParams = { dataset: datasetId ?? "", label: label ?? "" };
  const { spec: namesSpec } = usePlot("region_names", namesParams, {
    enabled: tab === "Correlation" && !!datasetId && !!label,
  });
  const allNames: string[] = namesSpec?.names ?? [];
  const allColors: string[] = namesSpec?.colors ?? [];
  // Exclusion indices are result-specific; reset when the selection changes.
  useEffect(() => {
    setExcluded([]);
  }, [datasetId, label]);

  if (!datasetId || !label) return <div className="hint">Select a result.</div>;
  if (item?.error) return <div className="plot-error">This result failed: {item.error}</div>;

  const base: QueryParams = { dataset: datasetId, label };
  const fparams: QueryParams = { ...base, filter: filter ?? undefined };
  // Descriptive plots can be served raw or MP-cleaned (refetched, hence a param).
  const dparams: QueryParams = {
    ...base,
    cleaned: cleaned ? 1 : undefined,
    exclude: excluded.length ? excluded.join(",") : undefined,
  };
  const tag =
    (cleaned ? " · MP-cleaned" : "") +
    (excluded.length ? ` · −${excluded.length} ch` : "");
  // Network graph: sparsify/edge params refetch; layout/colour/size are render-only.
  const netParams: QueryParams = {
    ...fparams,
    sparsify,
    sparsify_alpha: sparsifyAlpha,
    sparsify_threshold: sparsifyThreshold,
    edge_quantile: netEdgeQ,
  };
  const netOptions = {
    layout: netLayout,
    colorBy: netColorBy,
    sizeBy: netSizeBy,
    nodeScale: netNodeScale,
    edgeScale: netEdgeScale,
    edgeColorBy: netEdgeColorBy,
    edgeScaleMode: netEdgeScaleMode,
  };

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
          {/* Shared sparsification — drives Network, LRG and Brain-3D alike. */}
          <label>
            Sparsify
            <select value={sparsify} onChange={(e) => setSparsify(e.target.value)}>
              <option value="filter">As computed (filter)</option>
              <option value="percolation">Percolation backbone</option>
              <option value="disparity">Disparity filter</option>
              <option value="lans">LANS backbone</option>
              <option value="mp_validated">MP-validated</option>
              <option value="threshold">|r| ≥ threshold</option>
            </select>
          </label>
          {(sparsify === "disparity" || sparsify === "lans") && (
            <label>
              α
              <select value={sparsifyAlpha} onChange={(e) => setSparsifyAlpha(Number(e.target.value))}>
                <option value={0.01}>0.01</option>
                <option value={0.05}>0.05</option>
                <option value={0.1}>0.10</option>
                <option value={0.2}>0.20</option>
              </select>
            </label>
          )}
          {sparsify === "threshold" && (
            <label>
              |r| ≥
              <select value={sparsifyThreshold} onChange={(e) => setSparsifyThreshold(Number(e.target.value))}>
                {[0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map((v) => (
                  <option key={v} value={v}>{v.toFixed(1)}</option>
                ))}
              </select>
            </label>
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
              <label>
                Node size
                <select
                  value={brainNodeSize}
                  onChange={(e) => setBrainNodeSize(Number(e.target.value))}
                >
                  <option value={6}>S</option>
                  <option value={9}>M</option>
                  <option value={14}>L</option>
                  <option value={20}>XL</option>
                </select>
              </label>
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
          <ChannelExcluder
            names={allNames}
            colors={allColors}
            excluded={excluded}
            onChange={setExcluded}
          />
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
        <>
          <div className="subbar">
            <label>
              Edges
              <select value={netEdgeQ} onChange={(e) => setNetEdgeQ(Number(e.target.value))}>
                <option value={0}>all</option>
                <option value={0.5}>top 50%</option>
                <option value={0.75}>top 25%</option>
                <option value={0.9}>top 10%</option>
                <option value={0.95}>top 5%</option>
                <option value={0.98}>top 2%</option>
              </select>
            </label>
            <label>
              Layout
              <select value={netLayout} onChange={(e) => setNetLayout(e.target.value)}>
                <option value="spring">Spring (force)</option>
                <option value="kamada">Kamada–Kawai</option>
                <option value="spectral">Spectral</option>
                <option value="cose">Force (CoSE)</option>
                <option value="concentric">Concentric (by degree)</option>
                <option value="circle">Circle</option>
                <option value="grid">Grid</option>
                <option value="breadthfirst">Breadth-first</option>
              </select>
            </label>
            <label>
              Colour
              <select value={netColorBy} onChange={(e) => setNetColorBy(e.target.value)}>
                <option value="network">Atlas network</option>
                <option value="degree">Degree</option>
                <option value="strength">Strength</option>
              </select>
            </label>
            <label>
              Size
              <select value={netSizeBy} onChange={(e) => setNetSizeBy(e.target.value)}>
                <option value="strength">Strength</option>
                <option value="degree">Degree</option>
                <option value="uniform">Uniform</option>
              </select>
            </label>
            <label>
              Node ×
              <select value={netNodeScale} onChange={(e) => setNetNodeScale(Number(e.target.value))}>
                <option value={0.6}>0.6</option>
                <option value={1}>1.0</option>
                <option value={1.6}>1.6</option>
                <option value={2.4}>2.4</option>
              </select>
            </label>
            <label>
              Edge ×
              <select value={netEdgeScale} onChange={(e) => setNetEdgeScale(Number(e.target.value))}>
                <option value={0.5}>0.5</option>
                <option value={1}>1.0</option>
                <option value={2}>2.0</option>
                <option value={3}>3.0</option>
              </select>
            </label>
            <label>
              Edge colour
              <select value={netEdgeColorBy} onChange={(e) => setNetEdgeColorBy(e.target.value)}>
                <option value="weight">By weight</option>
                <option value="uniform">Uniform</option>
              </select>
            </label>
            <label>
              Edge width law
              <select value={netEdgeScaleMode} onChange={(e) => setNetEdgeScaleMode(e.target.value)}>
                <option value="linear">Linear</option>
                <option value="sqrt">Sqrt</option>
                <option value="log">Log</option>
                <option value="rank">Rank (percentile)</option>
              </select>
            </label>
          </div>
          <div className="plot-grid">
            <PlotPanel kind="global_metrics" title="Global metrics" params={fparams} />
            <PlotPanel kind="degree_distribution" title="Degree distribution" params={fparams} />
            <PlotPanel
              kind="network"
              title="Network graph"
              params={netParams}
              figureOptions={netOptions}
              caption="Interactive backbone. Sparsify re-derives the graph from the signed correlation (percolation / disparity / |r|-threshold); Edges then thins the drawn links by |weight|. Layout, colour, and size are view controls (no recompute). Global/degree/node metrics below reflect the pipeline's stored filter, not the ad-hoc sparsification."
              wide
            />
            <PlotPanel kind="node_metrics" title="Node metrics" params={fparams} wide />
          </div>
        </>
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
              nodeSize={brainNodeSize}
              sparsify={sparsify}
              sparsifyAlpha={sparsifyAlpha}
              sparsifyThreshold={sparsifyThreshold}
            />
          </section>
        </div>
      )}

      {tab === "LRG" && (
        <LrgView
          dataset={datasetId}
          label={label}
          filter={filter}
          sparsify={sparsify}
          sparsifyAlpha={sparsifyAlpha}
          sparsifyThreshold={sparsifyThreshold}
        />
      )}
    </div>
  );
}
