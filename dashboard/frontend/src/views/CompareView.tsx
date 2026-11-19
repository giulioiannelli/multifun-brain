// Brain-comparison tab. Contrasts two selections (A vs B — any task / modality /
// processing, at group-average or single-patient level) by their LRG hierarchies:
// Baker's gamma + cophenetic correlation, a per-ROI cophenetic "shift" reprojected
// onto the brain, an observed-vs-null distribution, and the top reorganised/stable
// regions — the interactive counterpart of the collaborator's fig6.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { SelectorBar } from "../components/SelectorBar";
import { CompareBrain } from "../components/plots/CompareBrain";
import { buildCompareBars, buildCompareHist } from "../components/plots/figures";
import { PlotlyFigure } from "../components/plots/PlotlyFigure";
import type { Dataset, PlotSpec } from "../types";

// Seed A and B from a Schaefer dataset (so the glass brain works out of the box)
// with two DIFFERENT results, so the initial view is a real comparison, not a
// result vs itself.
function seed(datasets: Dataset[]): { a: { id: string; label: string }; b: { id: string; label: string } } | null {
  const d =
    datasets.find((x) => x.id === "schaefer100_april2026/global" && x.items.length > 1) ??
    datasets.find((x) => x.items.length > 1) ??
    datasets.find((x) => x.items.length > 0) ??
    datasets[0];
  if (!d || !d.items[0]) return null;
  const a = { id: d.id, label: d.items[0].label };
  const b = { id: d.id, label: (d.items[1] ?? d.items[0]).label };
  return { a, b };
}

export function CompareView({ datasets }: { datasets: Dataset[] }) {
  const init = useMemo(() => seed(datasets), [datasets]);
  const [aId, setAId] = useState<string | null>(null);
  const [aLabel, setALabel] = useState<string | null>(null);
  const [bId, setBId] = useState<string | null>(null);
  const [bLabel, setBLabel] = useState<string | null>(null);
  const [nullKind, setNullKind] = useState<"none" | "strength">("none");
  const [nSurr, setNSurr] = useState<number>(60);
  const backbone = "lans";

  const [spec, setSpec] = useState<PlotSpec | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Seed both selections from the catalog once.
  useEffect(() => {
    if (init && aId === null) {
      setAId(init.a.id); setALabel(init.a.label);
      setBId(init.b.id); setBLabel(init.b.label);
    }
  }, [init, aId]);

  const ready = aId && aLabel && bId && bLabel;

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    setLoading(true); setErr(null);
    api
      .compare({
        dataset_a: aId!, label_a: aLabel!,
        dataset_b: bId!, label_b: bLabel!,
        null_kind: nullKind, n_surrogates: nSurr, backbone,
      })
      .then((s) => { if (alive) { setSpec(s); setLoading(false); } })
      .catch((e) => { if (alive) { setErr(String(e)); setLoading(false); } });
    return () => { alive = false; };
  }, [ready, aId, aLabel, bId, bLabel, nullKind, nSurr]);

  const histFig = useMemo(() => (spec && !spec.error ? buildCompareHist(spec) : null), [spec]);
  const barsFig = useMemo(() => (spec && !spec.error ? buildCompareBars(spec) : null), [spec]);
  const calibrated = spec?.metric === "calibrated";

  const fmt = (v: unknown) => (typeof v === "number" ? v.toFixed(3) : "…");

  return (
    <div className="explore">
      <div className="compare-selectors">
        <div className="compare-side">
          <div className="compare-tag compare-tag-a">A</div>
          <SelectorBar
            datasets={datasets}
            datasetId={aId}
            label={aLabel}
            onSelect={(d, l) => { setAId(d); setALabel(l); }}
          />
        </div>
        <div className="compare-side">
          <div className="compare-tag compare-tag-b">B</div>
          <SelectorBar
            datasets={datasets}
            datasetId={bId}
            label={bLabel}
            onSelect={(d, l) => { setBId(d); setBLabel(l); }}
          />
        </div>
      </div>

      <div className="subbar">
        <div className="seg">
          <span>Null model</span>
          <button className={nullKind === "none" ? "active" : ""} onClick={() => setNullKind("none")}>
            None (raw shift)
          </button>
          <button className={nullKind === "strength" ? "active" : ""} onClick={() => setNullKind("strength")}>
            Strength-preserving
          </button>
        </div>
        {nullKind === "strength" && (
          <label>
            Surrogates
            <select value={nSurr} onChange={(e) => setNSurr(Number(e.target.value))}>
              {[40, 60, 100, 200].map((v) => (
                <option key={v} value={v}>{v}</option>
              ))}
            </select>
          </label>
        )}
        <span className="lrg-note">
          LANS backbone · τ<sub>min</sub> hierarchy · {calibrated ? "calibrated" : "raw"} cophenetic shift
        </span>
      </div>

      {err ? (
        <div className="plot-error">Compare failed: {err}</div>
      ) : spec?.error ? (
        <div className="plot-error">{spec.error}</div>
      ) : !spec && loading ? (
        <div className="hint">Computing comparison{nullKind === "strength" ? " + nulls (a few seconds)…" : "…"}</div>
      ) : spec ? (
        <>
          <div className="compare-metrics">
            <div className="metric-card"><span className="metric-label">Baker's γ</span><span className="metric-val">{fmt(spec.bakers_gamma)}</span></div>
            <div className="metric-card"><span className="metric-label">cophenetic r</span><span className="metric-val">{fmt(spec.cophenetic_r)}</span></div>
            <div className="metric-card"><span className="metric-label">ROIs in common</span><span className="metric-val">{spec.n_common}</span></div>
            <div className="metric-card"><span className="metric-label">A</span><span className="metric-val small">{spec.label_a}</span></div>
            <div className="metric-card"><span className="metric-label">B</span><span className="metric-val small">{spec.label_b}</span></div>
          </div>

          {loading && <div className="hint">Recomputing…</div>}

          <div className="plot-grid">
            <section className="plot-card wide">
              <div className="plot-head"><h3>Who reorganised most · glass brain</h3></div>
              <p className="plot-caption">
                Per-ROI {calibrated ? "calibrated" : ""} cophenetic shift reprojected onto the brain —
                {calibrated ? " red = reorganised beyond the null, blue = more stable than the null." : " brighter = more reorganised between A and B."}
                {" "}Drag to rotate, scroll to zoom. Available for Schaefer-100 results.
              </p>
              {spec.brain_available ? (
                <CompareBrain
                  datasetA={aId!} labelA={aLabel!}
                  datasetB={bId!} labelB={bLabel!}
                  nullKind={nullKind} nSurrogates={nSurr} backbone={backbone}
                />
              ) : (
                <div className="hint">Brain reprojection needs both results on the Schaefer-100 atlas.</div>
              )}
            </section>

            <section className="plot-card">
              <div className="plot-head"><h3>Observed vs null</h3></div>
              <p className="plot-caption">
                Distribution of per-ROI shifts. {nullKind === "strength"
                  ? "The observed shifts (red) vs the pooled strength-preserving null (grey) — mass to the right of the null means reorganisation beyond chance."
                  : "Turn on the strength-preserving null to calibrate against chance."}
              </p>
              {histFig ? (
                <PlotlyFigure data={histFig.data} layout={histFig.layout} height={histFig.layout.height ?? 340} />
              ) : <div className="hint">…</div>}
            </section>

            <section className="plot-card">
              <div className="plot-head"><h3>Top reorganised / stable</h3></div>
              <p className="plot-caption">
                The regions that changed hierarchical position most (red) and least (blue) between A and B.
              </p>
              {barsFig ? (
                <PlotlyFigure data={barsFig.data} layout={barsFig.layout} height={barsFig.layout.height ?? 420} />
              ) : <div className="hint">…</div>}
            </section>
          </div>
        </>
      ) : (
        <div className="hint">Pick two results to compare.</div>
      )}
    </div>
  );
}
