import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { SelectorBar } from "./components/SelectorBar";
import { ExploreView, type ExploreTab } from "./views/ExploreView";
import { SignalView } from "./views/SignalView";
import { PipelineView } from "./views/PipelineView";
import { CompareView } from "./views/CompareView";
import type { Dataset } from "./types";

// Pipeline (methodology overview) is the landing tab; Signal shows the raw
// timecourses; the middle tabs explore one computed result; Compare contrasts
// two. Pipeline / Signal / Compare own their own selectors (data-free of the
// shared result selector).
const TABS = ["Pipeline", "Signal", "Correlation", "Network", "Brain 3-D", "LRG", "Compare"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Pipeline");

  useEffect(() => {
    api
      .catalog()
      .then((res) => {
        setDatasets(res.datasets);
        const preferred =
          res.datasets.find((d) => d.id === "schaefer100_april2026/global") ??
          res.datasets.find((d) => d.items.length > 0) ??
          res.datasets[0];
        if (preferred) {
          setDatasetId(preferred.id);
          setLabel(preferred.items[0]?.label ?? null);
        }
      })
      .catch((e) => setError(String(e)));
  }, []);

  function handleSelect(id: string, lbl: string) {
    setDatasetId(id);
    setLabel(lbl);
  }

  const item = useMemo(() => {
    const ds = datasets.find((d) => d.id === datasetId);
    return ds?.items.find((it) => it.label === label) ?? null;
  }, [datasets, datasetId, label]);

  // Data-free tabs need no shared result selector: Pipeline is a static
  // methodology scheme; Signal owns its own dataset dropdown; Compare owns two.
  const dataFree = tab === "Pipeline" || tab === "Signal" || tab === "Compare";

  return (
    <div className="app">
      <header className="topbar">
        <h1>multifun-brain</h1>
        <span className="subtitle">results dashboard</span>
      </header>

      <div className="tabs">
        {TABS.map((t) => (
          <button key={t} className={t === tab ? "active" : ""} onClick={() => setTab(t)}>
            {t}
          </button>
        ))}
      </div>

      {error && <div className="plot-error">Failed to load catalog: {error}</div>}

      {!dataFree && (
        <SelectorBar
          datasets={datasets}
          datasetId={datasetId}
          label={label}
          onSelect={handleSelect}
        />
      )}

      <main className="content">
        {tab === "Pipeline" ? (
          <PipelineView />
        ) : tab === "Signal" ? (
          <SignalView />
        ) : tab === "Compare" ? (
          <CompareView datasets={datasets} />
        ) : (
          <ExploreView
            tab={tab as ExploreTab}
            datasetId={datasetId}
            label={label}
            item={item}
          />
        )}
      </main>
    </div>
  );
}
