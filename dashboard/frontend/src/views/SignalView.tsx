// Signal tab: raw ROI timecourses straight from the .ts.1D hand-off (no
// pipeline). A Dataset dropdown switches between the raw_data_<id> sets. Two
// modes (when the dataset has a co2/rest contrast):
//   • Per subject — carpet + single channel + EMD + per-band reconstruction.
//   • Cohort (EMD bands) — histogram of every IMF's characteristic frequency
//     pooled across the cohort, defining the s5/s4/s* bands.
// The older kw datasets have no contrast / no TR, so only the per-subject
// carpet + channel + EMD views show for them (no cohort / band analysis).
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { PlotPanel } from "../components/PlotPanel";
import type { QueryParams, SignalCatalog } from "../types";

const CONTRAST_LABEL: Record<string, string> = { co2: "CO₂", rest: "rest" };
type Mode = "subject" | "cohort";

export function SignalView() {
  const [cat, setCat] = useState<SignalCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [dataset, setDataset] = useState<string>(""); // "" = server default
  const [mode, setMode] = useState<Mode>("subject");
  const [subject, setSubject] = useState<string>("");
  const [contrast, setContrast] = useState<string>("");
  const [processing, setProcessing] = useState<string>("");
  const [channel, setChannel] = useState<number>(0);
  const [normalize, setNormalize] = useState<boolean>(true);
  const [bandsYLog, setBandsYLog] = useState<boolean>(false);
  const [scheme, setScheme] = useState<"canonical" | "data_driven">("canonical");

  // (Re)load the catalog when the dataset changes; reset facets to its first entry.
  useEffect(() => {
    let cancelled = false;
    setError(null);
    api
      .signalCatalog(dataset || undefined)
      .then((c) => {
        if (cancelled) return;
        setCat(c);
        const first = c.entries[0];
        if (first) {
          setSubject(first.subject);
          setContrast(first.contrast ?? "");
          setProcessing(first.processing);
        }
        setChannel(0);
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  // Keep (subject, contrast, processing) on a real file within the dataset.
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
  const hasContrast = cat?.has_contrast ?? true;
  const effMode: Mode = hasContrast ? mode : "subject"; // cohort needs a contrast

  const fileParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast, processing }),
    [dataset, subject, contrast, processing],
  );
  const channelParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast, processing, channel }),
    [dataset, subject, contrast, processing, channel],
  );
  const cohortParams: QueryParams = useMemo(
    () => ({ dataset, contrast, processing, scheme }),
    [dataset, contrast, processing, scheme],
  );
  // Band reconstruction also depends on the scheme (channel/EMD panels don't).
  const bandParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast, processing, channel, scheme }),
    [dataset, subject, contrast, processing, channel, scheme],
  );

  if (error) return <div className="plot-error">Failed to load raw timecourses: {error}</div>;
  if (!cat) return <div className="hint">Loading raw timecourses…</div>;

  const ready = Boolean(processing && (effMode === "cohort" || subject));

  return (
    <div className="explore">
      <div className="subbar">
        <label>
          Dataset
          <select value={dataset || cat.dataset} onChange={(e) => setDataset(e.target.value)}>
            {cat.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {`${d.label} · ${d.n_subjects} subj${d.has_contrast ? "" : " · no contrast"}`}
              </option>
            ))}
          </select>
        </label>
        {hasContrast && (
          <div className="seg">
            <span>Mode</span>
            <button className={mode === "subject" ? "active" : ""} onClick={() => setMode("subject")}>
              Per subject
            </button>
            <button className={mode === "cohort" ? "active" : ""} onClick={() => setMode("cohort")}>
              Cohort (EMD bands)
            </button>
          </div>
        )}
        {hasContrast && (
          <label>
            Contrast
            <select value={contrast} onChange={(e) => setContrast(e.target.value)}>
              {cat.contrasts.map((c) => (
                <option key={c} value={c}>{CONTRAST_LABEL[c] ?? c}</option>
              ))}
            </select>
          </label>
        )}
        <label>
          Processing
          <select value={processing} onChange={(e) => setProcessing(e.target.value)}>
            {cat.processings.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        {hasContrast && (
          <div className="seg">
            <span>Bands</span>
            <button className={scheme === "canonical" ? "active" : ""} onClick={() => setScheme("canonical")}>
              Canonical
            </button>
            <button className={scheme === "data_driven" ? "active" : ""} onClick={() => setScheme("data_driven")}>
              Data-driven
            </button>
          </div>
        )}
      </div>

      {!cat.entries.length ? (
        <div className="hint">
          No raw timecourses found in <code>{cat.dataset}</code> (<code>{cat.root}</code>).
          Either the folder is empty, or its files use a filename scheme the reader
          doesn't parse yet (the kw HarvardOxford/Nov-2025 sets are supported; other
          schemes need a reader update).
        </div>
      ) : !ready ? null : effMode === "cohort" ? (
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
              figureOptions={{ yLog: bandsYLog, variant: "freq" }}
              caption="Every IMF's characteristic frequency — median instantaneous frequency (HHT), in Hz via the per-variant TR (bpf 1.353 s, optcom/MIR 0.98 s) — pooled across all subjects and ROIs. Bands: Canonical = the collaborator's fixed slow-oscillation edges (Slow-5 0.010–0.027, Slow-4 0.027–0.073, S* 0.073–0.180 Hz); Data-driven = geometric midpoints between the per-IMF-index clusters (dotted). First load runs EMD over the whole cohort (a few seconds)."
              wide
            />
            <PlotPanel
              kind="cohort_bands"
              title="IMF period spectrum (cohort)"
              endpoint="signal"
              params={cohortParams}
              figureOptions={{ yLog: bandsYLog, variant: "period" }}
              caption="The same characteristic frequencies expressed as periods (1/f, seconds) — the collaborator's companion panel. The shaded bands are the period ranges of Slow-5 / Slow-4 / S*."
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
              caption="Each row is one region; columns are timepoints. z-score normalises per region so temporal dynamics drive the grayscale. In Raw mode the scale spans every region's absolute level, so between-region baseline differences dominate and each row looks like a near-uniform stripe — switch to z-score to see within-region dynamics."
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
            {hasContrast && (
              <PlotPanel
                kind="signal_bands"
                title={`Band reconstruction · ${channelName}`}
                endpoint="signal"
                params={bandParams}
                caption="Each band's signal is the sum of this channel's IMFs whose characteristic frequency (median IF, Hz) falls in that band — the per-band signals behind the per-band correlation matrices. Bands follow the selected scheme (Canonical = the collaborator's fixed edges; Data-driven = cohort clusters)."
                wide
              />
            )}
          </div>
        </>
      )}
    </div>
  );
}
