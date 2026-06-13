// Signal tab: raw ROI timecourses straight from the .ts.1D hand-off (no
// pipeline). Two modes:
//   • Per subject — carpet + single channel + EMD + per-band reconstruction.
//   • Cohort (EMD bands) — histogram of every IMF's characteristic frequency
//     pooled across the cohort, which *defines* the s5/s4/s* bands.
// Subject / Channel selectors are per-subject only; Contrast / Processing are
// shared. The selectors are independent of the result-bundle SelectorBar.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { PlotPanel } from "../components/PlotPanel";
import type { QueryParams, SignalCatalog } from "../types";

const CONTRAST_LABEL: Record<string, string> = { co2: "CO₂", rest: "rest" };
type Mode = "subject" | "cohort";

export function SignalView() {
  const [cat, setCat] = useState<SignalCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [mode, setMode] = useState<Mode>("subject");
  const [subject, setSubject] = useState<string>("");
  const [contrast, setContrast] = useState<string>("");
  const [processing, setProcessing] = useState<string>("");
  const [channel, setChannel] = useState<number>(0);
  const [normalize, setNormalize] = useState<boolean>(true);
  const [bandsYLog, setBandsYLog] = useState<boolean>(false);

  useEffect(() => {
    api
      .signalCatalog()
      .then((c) => {
        setCat(c);
        const first = c.entries[0];
        if (first) {
          setSubject(first.subject);
          setContrast(first.contrast);
          setProcessing(first.processing);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  // Keep the (subject, contrast, processing) triple on a real file.
  useEffect(() => {
    if (!cat || !cat.entries.length || !subject) return;
    const exact = cat.entries.some(
      (e) => e.subject === subject && e.contrast === contrast && e.processing === processing,
    );
    if (exact) return;
    const fallback =
      cat.entries.find((e) => e.subject === subject && e.contrast === contrast) ??
      cat.entries.find((e) => e.subject === subject) ??
      cat.entries[0];
    if (fallback) {
      setContrast(fallback.contrast);
      setProcessing(fallback.processing);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cat, subject, contrast, processing]);

  const names = cat?.region_names ?? [];
  const channelName = names[channel] ?? `region ${channel}`;

  const fileParams: QueryParams = useMemo(
    () => ({ subject, contrast, processing }),
    [subject, contrast, processing],
  );
  const channelParams: QueryParams = useMemo(
    () => ({ subject, contrast, processing, channel }),
    [subject, contrast, processing, channel],
  );
  const cohortParams: QueryParams = useMemo(
    () => ({ contrast, processing }),
    [contrast, processing],
  );

  if (error) return <div className="plot-error">Failed to load raw timecourses: {error}</div>;
  if (!cat) return <div className="hint">Loading raw timecourses…</div>;
  if (!cat.entries.length)
    return (
      <div className="hint">
        No raw timecourses found under <code>{cat.root}</code>. Drop the per-subject
        <code> .ts.1D</code> folders there (see dashboard/README.md).
      </div>
    );

  const ready = Boolean(contrast && processing && (mode === "cohort" || subject));

  return (
    <div className="explore">
      <div className="subbar">
        <div className="seg">
          <span>Mode</span>
          <button className={mode === "subject" ? "active" : ""} onClick={() => setMode("subject")}>
            Per subject
          </button>
          <button className={mode === "cohort" ? "active" : ""} onClick={() => setMode("cohort")}>
            Cohort (EMD bands)
          </button>
        </div>
        <label>
          Contrast
          <select value={contrast} onChange={(e) => setContrast(e.target.value)}>
            {cat.contrasts.map((c) => (
              <option key={c} value={c}>{CONTRAST_LABEL[c] ?? c}</option>
            ))}
          </select>
        </label>
        <label>
          Processing
          <select value={processing} onChange={(e) => setProcessing(e.target.value)}>
            {cat.processings.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
      </div>

      {!ready ? null : mode === "cohort" ? (
        <>
          <div className="subbar">
            <div className="seg">
              <span>y</span>
              <button className={!bandsYLog ? "active" : ""} onClick={() => setBandsYLog(false)}>linear</button>
              <button className={bandsYLog ? "active" : ""} onClick={() => setBandsYLog(true)}>log</button>
            </div>
          </div>
          <div className="plot-grid">
            <PlotPanel
              kind="cohort_bands"
              title="IMF frequency spectrum (cohort)"
              endpoint="signal"
              params={cohortParams}
              figureOptions={{ yLog: bandsYLog }}
              caption="Every IMF's characteristic frequency (amplitude-weighted mean instantaneous frequency, cycles/sample) pooled across all subjects and ROIs. The shaded s5 / s4 / s* bands are the data-driven edges — geometric midpoints between the per-IMF-index clusters (dotted). First load runs EMD over the whole cohort (a few seconds)."
              wide
            />
          </div>
        </>
      ) : (
        <>
          {/* Whole-atlas view */}
          <div className="subbar">
            <label>
              Subject
              <select value={subject} onChange={(e) => setSubject(e.target.value)}>
                {cat.subjects.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            <div className="seg">
              <span>Carpet</span>
              <button className={normalize ? "active" : ""} onClick={() => setNormalize(true)}>z-score</button>
              <button className={!normalize ? "active" : ""} onClick={() => setNormalize(false)}>Raw</button>
            </div>
          </div>
          <div className="plot-grid">
            <PlotPanel
              kind="signal_heatmap"
              title="Timecourses (carpet)"
              endpoint="signal"
              params={fileParams}
              figureOptions={{ normalize }}
              caption="Each row is one Schaefer region; columns are timepoints. z-score normalises per region so temporal dynamics drive the grayscale. In Raw mode the scale spans every region's absolute level, so between-region baseline differences dominate and each row looks like a near-uniform stripe — switch to z-score to see within-region dynamics."
              wide
            />
          </div>

          {/* Single-ROI selection + per-channel plots */}
          <div className="subbar">
            <label>
              Channel (region)
              <select value={channel} onChange={(e) => setChannel(Number(e.target.value))}>
                {names.map((n, i) => (
                  <option key={i} value={i}>{`${i}· ${n}`}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="plot-grid">
            <PlotPanel
              kind="signal_channel"
              title={`Single channel · ${channelName}`}
              endpoint="signal"
              params={channelParams}
              wide
            />
            <PlotPanel
              kind="signal_emd"
              title={`EMD decomposition · ${channelName}`}
              endpoint="signal"
              params={channelParams}
              caption="Empirical Mode Decomposition (emd.sift.sift) of the selected channel into Intrinsic Mode Functions — fast oscillations at the top down to the slow trend / residual at the bottom."
              wide
            />
            <PlotPanel
              kind="signal_bands"
              title={`Band reconstruction · ${channelName}`}
              endpoint="signal"
              params={channelParams}
              caption="Each band's signal is the sum of this channel's IMFs whose characteristic frequency falls in that band (edges from the cohort spectrum) — the per-band signals behind the per-band correlation matrices."
              wide
            />
          </div>
        </>
      )}
    </div>
  );
}
