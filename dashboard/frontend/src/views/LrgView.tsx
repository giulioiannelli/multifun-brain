// LRG multiscale explorer. The dendrogram is driven by two sliders: τ steps over
// the precomputed grid (which already spans [1/λmax, τmax] — coarser beyond is
// uninformative), and a height-h cut that colours the branches into clusters.
// Only τ is expensive (a new linkage → refetch); the h cut recolours the
// dendrogram entirely client-side, so it drags continuously with no round-trip.
// Two companion curves say where to look: C(τ) (peaks = characteristic scales,
// which τ) and Ψ(n) (peak = the most stable cut at this τ, where to put h — one
// click snaps the cut onto it). The partition-flow raster and Sankey show how
// communities evolve across τ.
import { useEffect, useMemo, useState } from "react";
import { PlotPanel } from "../components/PlotPanel";
import { buildDendrogram, buildPsi, buildSpecificHeat, clusterLeaves, colorDendrogram } from "../components/plots/figures";
import { LrgBrain } from "../components/plots/LrgBrain";
import { NetworkGraph } from "../components/plots/NetworkGraph";
import { PlotlyFigure } from "../components/plots/PlotlyFigure";
import { usePlot } from "../hooks/usePlot";
import type { QueryParams } from "../types";

export function LrgView({
  dataset,
  label,
  filter,
  sparsify = "filter",
  sparsifyAlpha = 0.05,
  sparsifyThreshold = 0.3,
}: {
  dataset: string;
  label: string;
  filter: string | null;
  sparsify?: string;
  sparsifyAlpha?: number;
  sparsifyThreshold?: number;
}) {
  // Sparsify params flow into every LRG fetch so the whole tab honours the
  // Network-tab backbone selection (feature: sparsification propagation).
  const sparsifyParams: QueryParams =
    sparsify && sparsify !== "filter"
      ? { sparsify, sparsify_alpha: sparsifyAlpha, sparsify_threshold: sparsifyThreshold }
      : {};
  const base: QueryParams = { dataset, label, filter: filter ?? undefined, ...sparsifyParams };

  const { spec: grid } = usePlot("tau_grid", base);
  const taus: number[] = grid?.taus ?? [];
  const nClusters: number[] = grid?.n_clusters ?? [];
  const nTau = taus.length;

  const [tauIndex, setTauIndex] = useState<number>(-1); // -1 = coarsest (last)
  const [cutHeight, setCutHeight] = useState<number | null>(null); // null = natural

  const idx = nTau ? (tauIndex < 0 ? nTau - 1 : Math.min(tauIndex, nTau - 1)) : 0;

  // Reset both sliders when the selected result (or sparsification) changes.
  useEffect(() => {
    setTauIndex(-1);
    setCutHeight(null);
  }, [dataset, label, filter, sparsify, sparsifyAlpha, sparsifyThreshold]);
  // A new τ has its own dendrogram; fall back to its natural cut.
  useEffect(() => {
    setCutHeight(null);
  }, [idx]);

  // τ-dependent fetches (dendrogram linkage + Ψ). NOTE: no cut_height — the cut is
  // client-side, so dragging h never refetches. C(τ) is τ-independent (one curve).
  const dparams: QueryParams = { ...base, tau_index: idx };
  const { spec: dendro, loading: dLoading } = usePlot("dendrogram", dparams, { enabled: nTau > 0 });
  const { spec: psi } = usePlot("psi", dparams, { enabled: nTau > 0 });
  const { spec: cheat } = usePlot("specific_heat", base, { enabled: nTau > 0 });
  // Clustered-network layout is τ-dependent (MDS embedding), not cut-dependent —
  // dragging the cut only recolours it client-side (communityColors below).
  // edge_quantile=0 draws the full backbone (the MDS layout, not edge thinning,
  // conveys the communities); otherwise the /plot route's 0.9 default would drop
  // ~90% of edges. Sparsify (in `base`) already controls backbone density.
  const netParams: QueryParams = { ...dparams, edge_quantile: 0 };
  const { spec: lrgNet, loading: netLoading } = usePlot("lrg_network", netParams, { enabled: nTau > 0 });

  const dmin: number | null = dendro?.dcoord_min_positive ?? null;
  const dmax: number | null = dendro?.dcoord_max ?? null;
  const flat: number | null = dendro?.flat_threshold ?? null;
  const cutUsed: number | null = cutHeight ?? flat; // actual cut applied client-side
  const lambdaMax = nTau ? 1 / taus[0] : null;

  // Cut slider: t∈[0,1] maps geometrically onto [dmin, dmax] (the y-axis is log).
  const logspan = dmin && dmax && dmax > dmin ? Math.log(dmax / dmin) : null;
  const geo = (t: number) => (dmin && logspan ? dmin * Math.exp(logspan * t) : null);
  const cutT =
    cutUsed && dmin && logspan
      ? Math.max(0, Math.min(1, Math.log(cutUsed / dmin) / logspan))
      : 0.5;

  const dendroFig = useMemo(
    () => (dendro && !dendro.error ? buildDendrogram(dendro, cutHeight ?? undefined) : null),
    [dendro, cutHeight],
  );
  // Cluster count at the live cut (for the slider readout) — cheap, client-side.
  const nAtCut = useMemo(
    () => (dendro && !dendro.error && cutUsed != null ? colorDendrogram(dendro, cutUsed).nClusters : null),
    [dendro, cutUsed],
  );
  const cheatFig = useMemo(
    () => (cheat && !cheat.error ? buildSpecificHeat(cheat, { currentTau: taus[idx] }) : null),
    [cheat, taus, idx],
  );
  const psiFig = useMemo(() => (psi && !psi.error ? buildPsi(psi) : null), [psi]);
  const psiOptimal: number | null = psi?.optimal_threshold ?? null;

  // Per-node community colours at the live cut — same union-find the dendrogram
  // uses, so the clustered network + dendrogram share one colouring, and dragging
  // the cut recolours the network instantly (no refetch).
  const communityColors = useMemo(
    () =>
      dendro && !dendro.error && cutUsed != null
        ? clusterLeaves(dendro, cutUsed).colors
        : undefined,
    [dendro, cutUsed],
  );
  const netOptions = useMemo(
    () => ({ layout: "lrg", colorBy: "community", communityColors, sizeBy: "strength", edgeScaleMode: "sqrt", edgeColorBy: "uniform" }),
    [communityColors],
  );

  return (
    <div className="explore">
      <div className="subbar lrg-controls">
        <div className="slider">
          <span className="slider-label">τ step</span>
          <input
            type="range"
            min={0}
            max={Math.max(nTau - 1, 0)}
            step={1}
            value={idx}
            disabled={!nTau}
            onChange={(e) => setTauIndex(Number(e.target.value))}
          />
          <span className="slider-val">
            {nTau ? `${taus[idx].toPrecision(3)} · ${idx + 1}/${nTau} · ${nClusters[idx]} cl` : "…"}
          </span>
        </div>

        <div className="slider">
          <span className="slider-label">cut h</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.005}
            value={cutT}
            disabled={dmin == null || dmax == null}
            onChange={(e) => setCutHeight(geo(Number(e.target.value)))}
          />
          <span className="slider-val">
            {cutUsed != null
              ? `${cutUsed.toPrecision(3)}${nAtCut != null ? ` · ${nAtCut} cl` : ""}`
              : "natural"}
          </span>
          <button className="mini" disabled={cutHeight == null} onClick={() => setCutHeight(null)}>
            natural
          </button>
          <button
            className="mini"
            disabled={psiOptimal == null}
            title="Set the cut to the Ψ-optimal height"
            onClick={() => psiOptimal != null && setCutHeight(psiOptimal)}
          >
            Ψ-optimal
          </button>
        </div>

        {lambdaMax != null && (
          <span className="lrg-note">
            τ ∈ [1/λ<sub>max</sub>, τ<sub>max</sub>] = [{taus[0].toPrecision(2)}, {taus[nTau - 1].toPrecision(2)}] ·
            λ<sub>max</sub> ≈ {lambdaMax.toPrecision(3)}
          </span>
        )}
      </div>

      <div className="plot-grid">
        <section className="plot-card wide">
          <div className="plot-head">
            <h3>Dendrogram</h3>
          </div>
          <p className="plot-caption">
            Hierarchical merges at the selected τ (diffusion scale). Drag <b>cut h</b> to move the
            horizontal cut — branches below it are coloured per cluster, grey above. The cut recolours
            live in the browser (no reload); only τ recomputes. Use <b>Ψ-optimal</b> to snap the cut to
            the most stable partition.
          </p>
          {!nTau ? (
            <div className="hint">No LRG partitions for this result.</div>
          ) : dendro?.error ? (
            <div className="plot-error">{dendro.error}</div>
          ) : dendroFig ? (
            <PlotlyFigure data={dendroFig.data} layout={dendroFig.layout} height={dendroFig.layout.height ?? 520} />
          ) : (
            <div className="hint">{dLoading ? "Loading…" : "…"}</div>
          )}
        </section>

        <section className="plot-card">
          <div className="plot-head">
            <h3>Clustered network · LRG layout</h3>
          </div>
          <p className="plot-caption">
            Nodes placed by an MDS embedding of the LRG diffusion distance at this τ (communities
            sit together — not a generic spring layout), coloured by the <b>same cut</b> as the
            dendrogram. Drag <b>cut h</b> to recolour both in lock-step; only τ recomputes the layout.
          </p>
          {!nTau ? (
            <div className="hint">No LRG partitions for this result.</div>
          ) : lrgNet?.error ? (
            <div className="plot-error">{lrgNet.error}</div>
          ) : lrgNet && lrgNet.nodes ? (
            <NetworkGraph spec={lrgNet} options={netOptions} />
          ) : (
            <div className="hint">{netLoading ? "Loading…" : "…"}</div>
          )}
        </section>

        <section className="plot-card">
          <div className="plot-head">
            <h3>Brain partition · glass brain</h3>
          </div>
          <p className="plot-caption">
            The partition at this τ / cut reprojected into brain space — each parcel's survivor
            centroid coloured by its community (colours match the dendrogram and clustered network).
            Drag to rotate, scroll to zoom. Unavailable for atlases without Schaefer centroids.
          </p>
          {!nTau ? (
            <div className="hint">No LRG partitions for this result.</div>
          ) : dendro && !dendro.error && cutUsed != null ? (
            // Mount only once τ + cut have settled, so the slow nilearn iframe
            // never gets stuck showing an intermediate (pre-grid) partition.
            <LrgBrain
              dataset={dataset}
              label={label}
              filter={filter}
              tauIndex={idx}
              cutHeight={cutUsed}
              sparsify={sparsify}
              sparsifyAlpha={sparsifyAlpha}
              sparsifyThreshold={sparsifyThreshold}
            />
          ) : (
            <div className="hint">{dLoading ? "Loading…" : "…"}</div>
          )}
        </section>

        <section className="plot-card">
          <div className="plot-head">
            <h3>Specific heat C(τ)</h3>
          </div>
          <p className="plot-caption">
            C(τ) = −dS/d·log τ over diffusion time. Peaks mark characteristic scales; τ<sub>★</sub> (dotted)
            is the dominant one and τ′=1/λ<sub>max</sub> (dashed) the finest. The red line is the τ the
            dendrogram is on — slide τ to a peak.
          </p>
          {cheat?.error ? (
            <div className="plot-error">{cheat.error}</div>
          ) : cheatFig ? (
            <PlotlyFigure data={cheatFig.data} layout={cheatFig.layout} height={cheatFig.layout.height ?? 340} />
          ) : (
            <div className="hint">…</div>
          )}
        </section>

        <section className="plot-card">
          <div className="plot-head">
            <h3>Partition stability Ψ(n)</h3>
          </div>
          <p className="plot-caption">
            Ψ across candidate cluster counts at this τ (Villegas dendrogram gap). The highlighted peak is
            the most stable partition — its height is where <b>Ψ-optimal</b> puts the cut.
          </p>
          {psi?.error ? (
            <div className="plot-error">{psi.error}</div>
          ) : psiFig ? (
            <PlotlyFigure data={psiFig.data} layout={psiFig.layout} height={psiFig.layout.height ?? 340} />
          ) : (
            <div className="hint">…</div>
          )}
        </section>

        <PlotPanel
          kind="partition_flow"
          title="Community membership across τ"
          params={base}
          caption="Each row is a node, each column a τ. Colour = community, made consistent across τ so a band keeps its colour while it stays one community; rows are sorted so merging communities read as contiguous bands."
          wide
        />
        <PlotPanel
          kind="sankey"
          title="Community flow (Sankey)"
          params={base}
          caption="Communities as τ columns (left→right); a link's width is how many nodes flow between two communities at adjacent τ. Colour follows the source community, so you can trace a community as it merges or splits across scales."
          wide
        />
      </div>
    </div>
  );
}
