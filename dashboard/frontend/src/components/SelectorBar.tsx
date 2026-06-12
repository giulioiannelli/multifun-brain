// Result selector: pick a dataset, then a result within it. Items are grouped
// by contrast and shown with a human-readable processing/band/subject label.
// (Faceted multi-dropdown selection arrives with the comparison views.)
import type { Dataset, ResultItem } from "../types";

function prettyItem(it: ResultItem): string {
  const bits = [it.contrast, it.processing, it.band, it.subject].filter(Boolean);
  return bits.length ? bits.join(" · ") : it.label;
}

export function SelectorBar({
  datasets,
  datasetId,
  label,
  onDataset,
  onLabel,
}: {
  datasets: Dataset[];
  datasetId: string | null;
  label: string | null;
  onDataset: (id: string) => void;
  onLabel: (label: string) => void;
}) {
  const dataset = datasets.find((d) => d.id === datasetId) ?? null;
  const items = dataset?.items ?? [];

  return (
    <div className="selector-bar">
      <label>
        Dataset
        <select
          value={datasetId ?? ""}
          onChange={(e) => onDataset(e.target.value)}
        >
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.id} ({d.n_results ?? d.items.length})
            </option>
          ))}
        </select>
      </label>

      <label>
        Result
        <select value={label ?? ""} onChange={(e) => onLabel(e.target.value)}>
          {items.map((it) => (
            <option key={it.label} value={it.label}>
              {prettyItem(it)}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
