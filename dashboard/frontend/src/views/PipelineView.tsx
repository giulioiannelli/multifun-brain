// Pipeline tab: a static, data-free *methodology scheme* — boxes-and-arrows
// knowledge tree of the fMRI-comparison pipeline. Two views, toggled in place:
//
//   • Current  — what the toolkit does today: Daniele's signal→bands front end
//     (EMD decomposition of ROI timecourses into slow-oscillation bands) handed
//     off to Giulio's network back end, run in multiple fashions (signed
//     descriptive → unsigned filtering variants → LRG → standard metrics).
//   • Proposed — the full comparison vision: the same chain run independently on
//     the CO₂ and rest streams, converging on the *differential-node* lens —
//     ranking ROIs by how much they change between contrasts / tasks.
//
// Pure presentation: no API calls, no result selector. Stage content is kept in
// the arrays below so the scheme is edited as data, not markup.
import { useState } from "react";

type Tone = "signal" | "clean" | "network" | "compare" | "goal";

type Stage = {
  title: string;
  sub?: string;
  tag?: string; // small corner chip (e.g. section number, owner)
  tone?: Tone;
};

// ── arrowhead marker (one inline SVG def, reused by every connector) ──────────
function ArrowDefs() {
  return (
    <svg width="0" height="0" aria-hidden="true" style={{ position: "absolute" }}>
      <defs>
        <marker
          id="pipe-arrowhead"
          viewBox="0 0 10 10"
          refX="8"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M0,0 L10,5 L0,10 z" fill="var(--muted)" />
        </marker>
      </defs>
    </svg>
  );
}

function Box({ title, sub, tag, tone = "network" }: Stage) {
  return (
    <div className={`pipe-box tone-${tone}`}>
      {tag && <span className="pipe-chip">{tag}</span>}
      <div className="pipe-box-title">{title}</div>
      {sub && <div className="pipe-box-sub">{sub}</div>}
    </div>
  );
}

// Vertical "↓" connector between two stacked boxes (SVG line + shared arrowhead).
function Down({ label }: { label?: string }) {
  return (
    <div className="pipe-down">
      <svg viewBox="0 0 20 36" preserveAspectRatio="none" aria-hidden="true">
        <line x1="10" y1="0" x2="10" y2="30" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#pipe-arrowhead)" />
      </svg>
      {label && <span className="pipe-down-label">{label}</span>}
    </div>
  );
}

function Lane({ owner, role, stages }: { owner: string; role: string; stages: Stage[] }) {
  return (
    <div className="pipe-lane">
      <div className="pipe-lane-head">
        <span className="pipe-lane-owner">{owner}</span>
        <span className="pipe-lane-role">{role}</span>
      </div>
      {stages.map((s, i) => (
        <div key={s.title}>
          <Box {...s} />
          {i < stages.length - 1 && <Down />}
        </div>
      ))}
    </div>
  );
}

// ── content: CURRENT pipeline ────────────────────────────────────────────────
const DANIELE: Stage[] = [
  {
    title: "Raw fMRI ROI timecourses",
    sub: "Schaefer-100 parcels · 3dROIstats .ts.1D · CO₂ & rest tasks",
    tone: "signal",
  },
  {
    title: "Preprocessing / noise cleaning",
    sub: "per acquisition variant — bpfBOLD · bpfVASO · optcom · MIR-denoise",
    tone: "clean",
  },
  {
    title: "EMD decomposition",
    sub: "sift each ROI signal → IMFs (fast → slow)",
    tone: "signal",
  },
  {
    title: "Characteristic frequency per IMF",
    sub: "median instantaneous frequency (Hilbert / HHT), fs = 1/TR",
    tone: "signal",
  },
  {
    title: "Assign IMFs to slow-oscillation bands",
    sub: "Slow-5 0.010–0.027 · Slow-4 0.027–0.073 · S* 0.073–0.180 Hz",
    tone: "signal",
  },
  {
    title: "Per-band reconstruction → correlation",
    sub: "sum band IMFs per ROI → per-band signed correlation matrix",
    tone: "signal",
  },
];

const GIULIO: Stage[] = [
  {
    title: "Signed dense network",
    sub: "per contrast × variant × band · NaN dead-regions dropped (never zero-filled)",
    tag: "in",
    tone: "network",
  },
  {
    title: "Descriptive analysis — signed",
    sub: "weights · spectrum · signed Laplacian |D|−A · frustration · balance · MP overlay (γ = p/n)",
    tag: "§1",
    tone: "network",
  },
  {
    title: "Filter signed → unsigned · multiple fashions",
    sub: "absolute threshold · split-sign · partial correlation · backbone (disparity / LANS / MP-validated) · percolation threshold",
    tag: "§2",
    tone: "network",
  },
  {
    title: "LRG multiscale",
    sub: "diffusion kernel → τ-scales → hierarchical partitions",
    tag: "§3",
    tone: "network",
  },
  {
    title: "Standard network metrics",
    sub: "global · node · community · degree distribution",
    tag: "§3",
    tone: "network",
  },
];

// ── content: PROPOSED pipeline (shared chain, then the differential lens) ─────
const BANDS: Stage[] = [
  { title: "Slow-5", sub: "0.010–0.027 Hz", tone: "signal" },
  { title: "Slow-4", sub: "0.027–0.073 Hz", tone: "signal" },
  { title: "S*", sub: "0.073–0.180 Hz", tone: "signal" },
];

function CurrentView() {
  return (
    <div className="pipe-current">
      <div className="pipe-lanes">
        <Lane owner="Daniele" role="signal → frequency bands" stages={DANIELE} />
        <div className="pipe-handoff">
          <svg viewBox="0 0 64 24" preserveAspectRatio="none" aria-hidden="true">
            <line x1="2" y1="12" x2="56" y2="12" stroke="var(--muted)" strokeWidth="1.5" markerEnd="url(#pipe-arrowhead)" />
          </svg>
          <span>per-band signed correlation matrices</span>
        </div>
        <Lane owner="Giulio" role="network analysis (signed / unsigned / …)" stages={GIULIO} />
      </div>
    </div>
  );
}

function ProposedView() {
  return (
    <div className="pipe-proposed">
      {/* two contrast streams enter */}
      <div className="pipe-row pipe-streams">
        <Box title="fMRI — CO₂ task" sub="hypercapnia contrast" tone="signal" tag="stream" />
        <Box title="fMRI — rest" sub="resting-state contrast" tone="signal" tag="stream" />
      </div>
      <div className="pipe-merge-note">
        the same chain below runs <strong>independently</strong> on each stream
      </div>
      <Down />

      <Box title="Noise cleaning" sub="dead-region drop · denoise variant · (MP / Marchenko–Pastur on the matrix side)" tone="clean" />
      <Down />
      <Box title="EMD decomposition → IMFs" sub="empirical mode decomposition, characteristic freq per IMF (HHT)" tone="signal" />
      <Down label="gather bands" />

      {/* fan-out into the three bands */}
      <div className="pipe-row pipe-fan">
        {BANDS.map((b) => (
          <Box key={b.title} {...b} />
        ))}
      </div>
      <Down label="per band" />

      <Box
        title="Network representation"
        sub="per-band correlation → signed descriptive → unsigned variants (absolute · split-sign · partial corr · backbone · percolation)"
        tone="network"
      />
      <Down />
      <Box
        title="LRG framework"
        sub="multiscale diffusion partitions per band × contrast — the renormalisation backbone for comparison"
        tone="network"
      />
      <Down label="align CO₂ ↔ rest" />

      <Box
        title="Compare CO₂ vs rest"
        sub="match partitions / metrics across contrasts & tasks — ARI, multiscale hierarchy comparison (lrg.compare)"
        tone="compare"
      />
      <Down />
      <Box
        title="Nodes that change most between contrasts / tasks"
        sub="rank ROIs by cross-contrast change in community role, metastability & centrality (lrg.metastable) → the differential-node lens this whole pipeline is built to reveal"
        tone="goal"
        tag="goal"
      />
    </div>
  );
}

export function PipelineView() {
  const [view, setView] = useState<"current" | "proposed">("current");

  return (
    <div className="pipe">
      <ArrowDefs />

      <div className="pipe-top">
        <p className="pipe-intro">
          A conceptual map of the fMRI-comparison pipeline — from raw ROI
          timecourses through EMD frequency bands and network representations to
          the LRG framework, ending on the nodes that change most between
          contrasts. <strong>Current</strong> is what the toolkit does today;{" "}
          <strong>Proposed</strong> is the full CO₂-vs-rest comparison vision.
        </p>
        <div className="seg pipe-seg">
          <button className={view === "current" ? "active" : ""} onClick={() => setView("current")}>
            Current
          </button>
          <button className={view === "proposed" ? "active" : ""} onClick={() => setView("proposed")}>
            Proposed
          </button>
        </div>
      </div>

      <div className="pipe-legend">
        <span className="pipe-legend-item"><i className="dot tone-signal" /> signal / frequency</span>
        <span className="pipe-legend-item"><i className="dot tone-clean" /> cleaning</span>
        <span className="pipe-legend-item"><i className="dot tone-network" /> network / LRG</span>
        <span className="pipe-legend-item"><i className="dot tone-compare" /> comparison</span>
        <span className="pipe-legend-item"><i className="dot tone-goal" /> headline goal</span>
      </div>

      {view === "current" ? <CurrentView /> : <ProposedView />}
    </div>
  );
}
