// Single-result explorer. Phase A shows the correlation heatmap; Phase B adds
// tabs (descriptive / network / lrg) and more plot kinds.
import { useEffect, useState } from "react";
import { api } from "../api/client";
import { Heatmap } from "../components/plots/Heatmap";
import type { HeatmapSpec } from "../types";

export function ExploreView({
  datasetId,
  label,
}: {
  datasetId: string | null;
  label: string | null;
}) {
  const [spec, setSpec] = useState<HeatmapSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!datasetId || !label) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    api
      .heatmap(datasetId, label)
      .then((s) => !cancelled && setSpec(s))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [datasetId, label]);

  if (!datasetId || !label) return <div className="hint">Select a result.</div>;
  if (error) return <div className="plot-error">{error}</div>;
  if (loading && !spec) return <div className="hint">Loading…</div>;
  if (!spec) return null;

  return (
    <div className="explore">
      <h2>Correlation matrix</h2>
      <Heatmap spec={spec} />
    </div>
  );
}
