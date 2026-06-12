import { useEffect, useMemo, useState } from "react";
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

  const item = useMemo(() => {
    const ds = datasets.find((d) => d.id === datasetId);
    return ds?.items.find((it) => it.label === label) ?? null;
  }, [datasets, datasetId, label]);

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
        <ExploreView datasetId={datasetId} label={label} item={item} />
      </main>
    </div>
  );
}
