// Signal tab: raw ROI timecourses straight from the .ts.1D hand-off (no
// pipeline). A Dataset dropdown switches between the raw_data_<id> sets, then
// three facet dropdowns — Task (co2/rest), Contrast (modality: bold/vaso/cbf/
// noise) and Processing (clean/optcom/…). The facets are parsed from the
// filename token; the selection is resolved back to that original token, which
// the backend still keys files by (so file loading is unchanged).
//   • Per subject — carpet + single channel + EMD + per-band reconstruction.
//   • Cohort (EMD bands) — histogram of every IMF's characteristic frequency
//     pooled across the cohort, defining the s5/s4/s* bands.
// Cohort / band analysis needs the co2/rest task design (the April set); the
// task-less kw sets show only the per-subject carpet + channel + EMD views.
import { useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import { PlotPanel } from "../components/PlotPanel";
import type { QueryParams, SignalCatalog } from "../types";

const TASK_LABEL: Record<string, string> = { co2: "CO₂", rest: "rest" };
const CONTRAST_LABEL: Record<string, string> = {
  bold: "BOLD", vaso: "VASO", cbf: "CBF",
};
// Processing display order (matches the result-tab SelectorBar's PROC_ORDER).
const PROC_ORDER = [
  "raw", "bpf", "optcom", "optcomMIRdenoised", "clean",
  "furN", "fcurN", "MNI152", "MIRnoise",
];
const orderProc = (vals: string[]): string[] =>
  [...vals].sort((a, b) => {
    const ia = PROC_ORDER.indexOf(a), ib = PROC_ORDER.indexOf(b);
    if (ia === -1 && ib === -1) return a.localeCompare(b);
    if (ia === -1) return 1;
    if (ib === -1) return -1;
    return ia - ib;
  });
// Sentinel subject: the cross-subject mean timecourse (kept in sync with the
// backend timeseries.AVG_SUBJECT). Carpet / channel / EMD / bands all honour it.
const AVG_SUBJECT = "__avg__";
// Carpet band selector: broadband, or a canonical band whose per-region signal is
// the sum of that region's IMFs in the band (backend timeseries.band_carpet).
const CARPET_BANDS: { value: string; label: string }[] = [
  { value: "full", label: "Broadband" },
  { value: "s5", label: "Slow-5" },
  { value: "s4", label: "Slow-4" },
  { value: "sstar", label: "S*" },
];
type Mode = "subject" | "cohort";

export function SignalView() {
  const [cat, setCat] = useState<SignalCatalog | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [dataset, setDataset] = useState<string>(""); // "" = server default
  const [mode, setMode] = useState<Mode>("subject");
  const [subject, setSubject] = useState<string>("");
  const [task, setTask] = useState<string>(""); // co2 / rest, "" if task-less
  const [contrast, setContrast] = useState<string>(""); // modality
  const [processing, setProcessing] = useState<string>(""); // pipeline label
  const [channel, setChannel] = useState<number>(0);
  const [normalize, setNormalize] = useState<boolean>(true);
  const [carpetBand, setCarpetBand] = useState<string>("full"); // carpet band selector
  const [cohortSubject, setCohortSubject] = useState<string>(""); // "" = whole cohort
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
          setTask(first.task ?? "");
          setContrast(first.contrast ?? "");
          setProcessing(first.processing);
        }
        setChannel(0);
        setCarpetBand("full");
        setCohortSubject(""); // subject ids are dataset-specific; back to whole cohort
      })
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
  }, [dataset]);

  // Keep (subject, task, contrast, processing) on a real file; snap otherwise. The
  // Average subject matches any real subject's file (the variant just needs to exist).
  useEffect(() => {
    if (!cat || !cat.entries.length) return;
    const subjOk = (e: (typeof cat.entries)[number]) =>
      subject === AVG_SUBJECT || e.subject === subject;
    const eq = (e: (typeof cat.entries)[number]) =>
      subjOk(e) && (e.task ?? "") === task && (e.contrast ?? "") === contrast;
    if (cat.entries.some((e) => eq(e) && e.processing === processing)) return;
    const fallback =
      cat.entries.find(eq) ??
      cat.entries.find((e) => subjOk(e) && (e.task ?? "") === task) ??
      cat.entries.find(subjOk) ??
      cat.entries[0];
    if (fallback) {
      setTask(fallback.task ?? "");
      setContrast(fallback.contrast ?? "");
      setProcessing(fallback.processing);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cat, subject, task, contrast, processing]);

  // Resolve the (task, contrast, processing) facets back to the original token
  // the backend keys files by (subject-independent).
  const token = useMemo(() => {
    const e = cat?.entries.find(
      (e) => (e.task ?? "") === task && (e.contrast ?? "") === contrast && e.processing === processing,
    );
    return e?.token ?? "";
  }, [cat, task, contrast, processing]);

  // Cascading options: Contrast lists only the modalities present for the chosen
  // Task; Processing only the pipelines present for the chosen (Task, Contrast).
  // So VASO stops offering BOLD-only processings — impossible combinations are
  // never selectable. The fallback effect above re-snaps the current selection
  // whenever an upstream facet narrows the list.
  const contrastOpts = useMemo(() => {
    const s = new Set<string>();
    for (const e of cat?.entries ?? []) if ((e.task ?? "") === task && e.contrast) s.add(e.contrast);
    return [...s].sort((a, b) => a.localeCompare(b));
  }, [cat, task]);
  const procOpts = useMemo(() => {
    const s = new Set<string>();
    for (const e of cat?.entries ?? [])
      if ((e.task ?? "") === task && (e.contrast ?? "") === contrast) s.add(e.processing);
    return orderProc([...s]);
  }, [cat, task, contrast]);

  const names = cat?.region_names ?? [];
  const AVG_NAME = "⟨mean over regions⟩";
  const channelName = channel < 0 ? AVG_NAME : (names[channel] ?? `region ${channel}`);
  const hasTask = cat?.has_task ?? false; // cohort/band analysis needs the co2/rest design
  const hasContrast = cat?.has_contrast ?? false;
  const effMode: Mode = hasTask ? mode : "subject";

  // API params still use (contrast = task, processing = token) — the backend
  // resolution is unchanged; only the UI splits the token into facets.
  const fileParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast: task, processing: token }),
    [dataset, subject, task, token],
  );
  // Carpet honours the band selector: "full" → raw broadband array; s5/s4/s* →
  // the band-reconstructed carpet (backend timeseries.band_carpet).
  const carpetParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast: task, processing: token, band: carpetBand }),
    [dataset, subject, task, token, carpetBand],
  );
  const channelParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast: task, processing: token, channel }),
    [dataset, subject, task, token, channel],
  );
  // Cohort spectrum: a non-empty subject pools only that subject's ROIs; ""
  // (the URLSearchParams builder drops it) pools the whole cohort.
  const cohortParams: QueryParams = useMemo(
    () => ({ dataset, contrast: task, processing: token, scheme, subject: cohortSubject || undefined }),
    [dataset, task, token, scheme, cohortSubject],
  );
  // Band reconstruction also depends on the scheme (channel/EMD panels don't).
  const bandParams: QueryParams = useMemo(
    () => ({ dataset, subject, contrast: task, processing: token, channel, scheme }),
    [dataset, subject, task, token, channel, scheme],
  );

  if (error) return <div className="plot-error">Failed to load raw timecourses: {error}</div>;
  if (!cat) return <div className="hint">Loading raw timecourses…</div>;

  const ready = Boolean(token && (effMode === "cohort" || subject));

  return (
    <div className="explore">
      <div className="subbar">
        <label>
          Dataset
          <select value={dataset || cat.dataset} onChange={(e) => setDataset(e.target.value)}>
            {cat.datasets.map((d) => (
              <option key={d.id} value={d.id}>
                {`${d.label} · ${d.n_subjects} subj`}
              </option>
            ))}
          </select>
        </label>
        {hasTask && (
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
        {hasTask && (
          <label>
            Task
            <select value={task} onChange={(e) => setTask(e.target.value)}>
              {cat.tasks.map((t) => (
                <option key={t} value={t}>{TASK_LABEL[t] ?? t}</option>
              ))}
            </select>
          </label>
        )}
        {hasContrast && (
          <label>
            Contrast
            <select value={contrast} onChange={(e) => setContrast(e.target.value)}>
              {contrastOpts.map((c) => (
                <option key={c} value={c}>{CONTRAST_LABEL[c] ?? c}</option>
              ))}
            </select>
          </label>
        )}
        <label>
          Processing
          <select value={processing} onChange={(e) => setProcessing(e.target.value)}>
            {procOpts.map((p) => (
              <option key={p} value={p}>{p}</option>
            ))}
          </select>
        </label>
        {hasTask && (
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
            <label>
              Subject
              <select value={cohortSubject} onChange={(e) => setCohortSubject(e.target.value)}>
                <option value="">All subjects (cohort)</option>
                {cat.subjects.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </label>
            <div className="seg">
              <span>y</span>
              <button className={!bandsYLog ? "active" : ""} onClick={() => setBandsYLog(false)}>linear</button>
              <button className={bandsYLog ? "active" : ""} onClick={() => setBandsYLog(true)}>log</button>
            </div>
          </div>
          <div className="plot-grid">
            <PlotPanel
              kind="cohort_bands"
              title={`IMF frequency spectrum (${cohortSubject || "cohort"})`}
              endpoint="signal"
              params={cohortParams}
              figureOptions={{ yLog: bandsYLog, variant: "freq" }}
              caption="Every IMF's characteristic frequency — median instantaneous frequency (HHT), in Hz via the per-variant TR (bpf 1.353 s, optcom/MIR 0.98 s) — pooled across ROIs (all subjects, or a single subject via the Subject selector). Bands: Canonical = the collaborator's fixed slow-oscillation edges (Slow-5 0.010–0.027, Slow-4 0.027–0.073, S* 0.073–0.180 Hz); Data-driven = geometric midpoints between the per-IMF-index clusters (dotted). First load runs EMD over the selection (a few seconds for the whole cohort)."
              wide
            />
            <PlotPanel
              kind="cohort_bands"
              title={`IMF period spectrum (${cohortSubject || "cohort"})`}
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
                <option value={AVG_SUBJECT}>Average (all subjects)</option>
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
            <label>
              Band
              <select value={carpetBand} onChange={(e) => setCarpetBand(e.target.value)}>
                {CARPET_BANDS.map((b) => (
                  <option key={b.value} value={b.value}>{b.label}</option>
                ))}
              </select>
            </label>
          </div>
          <div className="plot-grid">
            <PlotPanel
              kind="signal_heatmap"
              title={
                carpetBand === "full"
                  ? "Timecourses (carpet)"
                  : `Timecourses (carpet) · ${CARPET_BANDS.find((b) => b.value === carpetBand)?.label}`
              }
              endpoint="signal"
              params={carpetParams}
              figureOptions={{ normalize }}
              caption="Each row is one region; columns are timepoints. z-score normalises per region so temporal dynamics drive the grayscale. In Raw mode the scale spans every region's absolute level, so between-region baseline differences dominate and each row looks like a near-uniform stripe — switch to z-score to see within-region dynamics. The Band selector replaces each region's signal with the sum of its IMFs in that canonical band (Slow-5 0.010–0.027, Slow-4 0.027–0.073, S* 0.073–0.180 Hz) — the band-filtered carpet behind the per-band correlation matrices."
              wide
            />
          </div>

          {/* Single-ROI selection + per-channel plots */}
          <div className="subbar">
            <label>
              Channel (region)
              <select value={channel} onChange={(e) => setChannel(Number(e.target.value))}>
                <option value={-1}>{AVG_NAME}</option>
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
            {hasTask && (
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
