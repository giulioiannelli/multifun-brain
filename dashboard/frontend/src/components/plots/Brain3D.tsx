// Interactive 3-D brain: embeds the backend's standalone nilearn page in an
// iframe (the page is ~2.5 MB, so it loads as a normal document rather than
// round-tripping as JSON). `mode` picks connectome vs markers; `edgeQuantile`
// thresholds the connectome edges. Remounts (via `key`) when params change.
import type { QueryParams } from "../../types";

export function Brain3D({
  dataset,
  label,
  filter,
  mode,
  edgeQuantile,
}: {
  dataset: string;
  label: string;
  filter: string | null;
  mode: "connectome" | "markers";
  edgeQuantile: number;
}) {
  const params: QueryParams = { dataset, label, mode };
  if (filter) params.filter = filter;
  if (mode === "connectome") params.edge_quantile = edgeQuantile;

  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  const src = `/api/brain3d?${sp.toString()}`;

  return (
    <iframe
      key={src}
      className="brain3d-frame"
      src={src}
      title="3-D brain"
      loading="lazy"
    />
  );
}
