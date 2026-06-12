import { useEffect, useState } from "react";
import { api } from "./api/client";
import { SelectorBar } from "./components/SelectorBar";
import { ExploreView } from "./views/ExploreView";
import type { Dataset } from "./types";

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [datasetId, setDatasetId] = useState<string | null>(null);
  const [label, setLabel] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .catalog()
      .then((res) => {
        setDatasets(res.datasets);
        // Default to a dataset that actually has items (prefer april/global).
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

  function handleDataset(id: string) {
    setDatasetId(id);
    const ds = datasets.find((d) => d.id === id);
    setLabel(ds?.items[0]?.label ?? null);
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>multifun-brain</h1>
        <span className="subtitle">results dashboard</span>
      </header>

      {error && <div className="plot-error">Failed to load catalog: {error}</div>}

      <SelectorBar
        datasets={datasets}
        datasetId={datasetId}
        label={label}
        onDataset={handleDataset}
        onLabel={setLabel}
      />

      <main className="content">
        <ExploreView datasetId={datasetId} label={label} />
      </main>
    </div>
  );
}
