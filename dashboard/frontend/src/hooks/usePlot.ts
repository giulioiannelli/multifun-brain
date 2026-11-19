// Fetch a plot spec for (kind, params). Re-fetches when the request changes;
// cancels stale responses so rapid selector changes don't race. `endpoint`
// selects the backend route — "plot" (result bundles) or "signal" (raw
// timecourses).
import { useEffect, useState } from "react";
import { PLOT_FETCHERS, type PlotEndpoint } from "../api/client";
import type { PlotSpec, QueryParams } from "../types";

export function usePlot(
  kind: string,
  params: QueryParams,
  opts: { enabled?: boolean; endpoint?: PlotEndpoint } = {},
) {
  const { enabled = true, endpoint = "plot" } = opts;
  const [spec, setSpec] = useState<PlotSpec | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const key = JSON.stringify({ kind, params, endpoint });

  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    PLOT_FETCHERS[endpoint](kind, params)
      .then((s) => !cancelled && setSpec(s))
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, enabled]);

  return { spec, error, loading };
}
