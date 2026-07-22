// Glass brain of the LRG partition: embeds the backend's standalone nilearn page
// (survivor centroids coloured by community at the current τ / dendrogram cut) in
// an iframe. Remounts (via `key`) when τ, the cut, or the sparsification changes.
import type { QueryParams } from "../../types";

export function LrgBrain({
  dataset,
  label,
  filter,
  tauIndex,
  cutHeight,
  sparsify,
  sparsifyAlpha = 0.05,
  sparsifyThreshold = 0.3,
  nodeSize = 9,
}: {
  dataset: string;
  label: string;
  filter: string | null;
  tauIndex: number;
  cutHeight: number | null;
  sparsify?: string;
  sparsifyAlpha?: number;
  sparsifyThreshold?: number;
  nodeSize?: number;
}) {
  const params: QueryParams = { dataset, label, tau_index: tauIndex, node_size: nodeSize };
  if (filter) params.filter = filter;
  if (cutHeight != null) params.cut_height = cutHeight;
  if (sparsify && sparsify !== "filter") {
    params.sparsify = sparsify;
    params.sparsify_alpha = sparsifyAlpha;
    params.sparsify_threshold = sparsifyThreshold;
  }

  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  const src = `/api/lrg_brain?${sp.toString()}`;

  return (
    <iframe
      key={src}
      className="brain3d-frame"
      src={src}
      title="LRG partition brain"
      loading="lazy"
    />
  );
}
