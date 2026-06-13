import { useEffect, useMemo, useState } from "react";
import { api } from "./api/client";
import { SelectorBar } from "./components/SelectorBar";
import { ExploreView, type ExploreTab } from "./views/ExploreView";
import { SignalView } from "./views/SignalView";
import type { Dataset } from "./types";

// Signal (raw timecourses) comes first; the rest explore a computed result.
const TABS = ["Signal", "Correlation", "Network", "LRG"] as const;
type Tab = (typeof TABS)[number];

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("Signal");

  useEffect(() => {
    api
      .catalog()
      .then((res) => {
        setDatasets(res.datasets);
        const preferred =
          res.datasets.find((d) => d.id === "april/global") ??
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

  const isSignal = tab === "Signal";

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

      {!isSignal && (
        <SelectorBar
          datasets={datasets}
          datasetId={datasetId}
          label={label}
          onSelect={handleSelect}
        />
      )}

      <main className="content">
        {isSignal ? (
          <SignalView />
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
