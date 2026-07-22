// Checkbox dropdown to exclude noisy channels from the correlation views. Shows a
// searchable, scrollable list of region labels (with atlas-network colour swatch);
// checked = excluded. Closes on outside-click. Indices are positions in the
// current result's surviving-region order (what the backend `exclude` expects).
import { useEffect, useRef, useState } from "react";

export function ChannelExcluder({
  names,
  colors,
  excluded,
  onChange,
}: {
  names: string[];
  colors?: string[];
  excluded: number[];
  onChange: (next: number[]) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const set = new Set(excluded);
  const toggle = (i: number) => {
    const next = new Set(set);
    if (next.has(i)) next.delete(i);
    else next.add(i);
    onChange([...next].sort((a, b) => a - b));
  };

  const q = query.trim().toLowerCase();
  const shown = names
    .map((n, i) => ({ n, i }))
    .filter(({ n, i }) => !q || n.toLowerCase().includes(q) || String(i) === q);

  return (
    <div className="excluder" ref={ref}>
      <button className={excluded.length ? "active" : ""} onClick={() => setOpen((o) => !o)}>
        Exclude channels{excluded.length ? ` (${excluded.length})` : ""} ▾
      </button>
      {open && (
        <div className="excluder-panel">
          <div className="excluder-head">
            <input
              type="text"
              placeholder="filter regions…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              autoFocus
            />
            <button
              className="excluder-clear"
              disabled={!excluded.length}
              onClick={() => onChange([])}
            >
              Clear
            </button>
          </div>
          <ul className="excluder-list">
            {shown.map(({ n, i }) => (
              <li key={i}>
                <label>
                  <input type="checkbox" checked={set.has(i)} onChange={() => toggle(i)} />
                  {colors && (
                    <span className="excluder-swatch" style={{ background: colors[i] ?? "#888" }} />
                  )}
                  <span className="excluder-idx">{i}</span>
                  <span className="excluder-name">{n}</span>
                </label>
              </li>
            ))}
            {!shown.length && <li className="excluder-empty">no match</li>}
          </ul>
        </div>
      )}
    </div>
  );
}
