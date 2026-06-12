// Result selector. The dataset id (e.g. "april/global") is split into two
// dropdowns — Dataset (top-level folder, "april") and Category (subfolder,
// "global") — plus the Result dropdown within the chosen bundle.
import { useMemo } from "react";
import type { Dataset, ResultItem } from "../types";

function splitId(id: string): { group: string; category: string } {
  if (id === "." || id === "") return { group: "(root)", category: "results" };
  const i = id.indexOf("/");
  if (i === -1) return { group: id, category: "(all)" };
  return { group: id.slice(0, i), category: id.slice(i + 1) };
}

function prettyItem(it: ResultItem): string {
  const bits = [it.subject, it.contrast, it.processing, it.band].filter(Boolean);
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
  // group -> ordered list of { category, id }
  const groups = useMemo(() => {
    const m = new Map<string, { category: string; id: string }[]>();
    for (const d of datasets) {
      const { group, category } = splitId(d.id);
      if (!m.has(group)) m.set(group, []);
      m.get(group)!.push({ category, id: d.id });
    }
    return m;
  }, [datasets]);

  const current = datasetId ? splitId(datasetId) : null;
  const groupNames = [...groups.keys()];
  const categories = current ? (groups.get(current.group) ?? []) : [];
  const dataset = datasets.find((d) => d.id === datasetId) ?? null;
  const items = dataset?.items ?? [];

  function onGroup(group: string) {
    const first = groups.get(group)?.[0];
    if (first) onDataset(first.id);
  }

  return (
    <div className="selector-bar">
      <label>
        Dataset
        <select value={current?.group ?? ""} onChange={(e) => onGroup(e.target.value)}>
          {groupNames.map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>
      </label>

      <label>
        Category
        <select value={datasetId ?? ""} onChange={(e) => onDataset(e.target.value)}>
          {categories.map((c) => (
            <option key={c.id} value={c.id}>{c.category}</option>
          ))}
        </select>
      </label>

      <label>
        Result
        <select value={label ?? ""} onChange={(e) => onLabel(e.target.value)}>
          {items.map((it) => (
            <option key={it.label} value={it.label}>{prettyItem(it)}</option>
          ))}
        </select>
      </label>
    </div>
  );
}
