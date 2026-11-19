// Tabular renderers: scalar global metrics + sortable per-node metrics.
import { useMemo, useState } from "react";
import type { PlotSpec } from "../../types";

function fmt(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : v.toFixed(4);
  return String(v);
}

export function GlobalMetrics({ spec }: { spec: PlotSpec }) {
  if (spec.error) return <div className="plot-error">{spec.error}</div>;
  const metrics: Record<string, unknown> = spec.metrics ?? {};
  return (
    <table className="metrics-table">
      <tbody>
        {Object.entries(metrics).map(([k, v]) => (
          <tr key={k}>
            <th>{k}</th>
            <td>{fmt(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function NodeMetrics({ spec }: { spec: PlotSpec }) {
  const [sortCol, setSortCol] = useState<number | null>(null);
  const [desc, setDesc] = useState(true);

  const columns: string[] = spec.columns ?? [];
  const names: string[] = spec.names ?? [];
  const rows: number[][] = spec.rows ?? [];

  const order = useMemo(() => {
    const idx = rows.map((_, i) => i);
    if (sortCol === null) return idx;
    idx.sort((a, b) => {
      const va = rows[a][sortCol];
      const vb = rows[b][sortCol];
      return desc ? vb - va : va - vb;
    });
    return idx;
  }, [rows, sortCol, desc]);

  if (spec.error) return <div className="plot-error">{spec.error}</div>;

  function clickCol(c: number) {
    if (sortCol === c) setDesc(!desc);
    else {
      setSortCol(c);
      setDesc(true);
    }
  }

  return (
    <div className="node-metrics-wrap">
      <table className="metrics-table node-metrics">
        <thead>
          <tr>
            <th>node</th>
            {columns.map((c, ci) => (
              <th key={c} className="sortable" onClick={() => clickCol(ci)}>
                {c} {sortCol === ci ? (desc ? "▼" : "▲") : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {order.map((ri) => (
            <tr key={ri}>
              <td className="node-name">{names[ri]}</td>
              {rows[ri].map((v, ci) => (
                <td key={ci}>{v === null ? "—" : v.toFixed(4)}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
