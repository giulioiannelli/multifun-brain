// Glass brain of the per-ROI comparison scalar (calibrated cophenetic shift):
// embeds the backend's standalone nilearn page in an iframe. Remounts when the
// two selections or the null settings change.
import type { QueryParams } from "../../types";

export function CompareBrain({
  datasetA, labelA, filterA,
  datasetB, labelB, filterB,
  nullKind, nSurrogates, backbone,
}: {
  datasetA: string; labelA: string; filterA?: string | null;
  datasetB: string; labelB: string; filterB?: string | null;
  nullKind: string; nSurrogates: number; backbone: string;
}) {
  const params: QueryParams = {
    dataset_a: datasetA, label_a: labelA,
    dataset_b: datasetB, label_b: labelB,
    null_kind: nullKind, n_surrogates: nSurrogates, backbone,
  };
  if (filterA) params.filter_a = filterA;
  if (filterB) params.filter_b = filterB;

  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
  }
  const src = `/api/compare/brain?${sp.toString()}`;

  return (
    <iframe
      key={src}
      className="brain3d-frame"
      src={src}
      title="comparison glass brain"
      loading="lazy"
    />
  );
}
