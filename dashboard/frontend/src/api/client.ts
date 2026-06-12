// Thin typed fetch wrappers around the FastAPI backend. All calls go through
// the same-origin /api prefix (Vite proxies it in dev; FastAPI serves it in prod).
import type { CatalogResponse, PlotSpec, QueryParams } from "../types";

const BASE = "/api";

async function getJSON<T>(path: string, params?: QueryParams): Promise<T> {
  let qs = "";
  if (params) {
    const sp = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== "") sp.set(k, String(v));
    }
    qs = sp.toString() ? `?${sp.toString()}` : "";
  }
  const res = await fetch(`${BASE}${path}${qs}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  catalog: () => getJSON<CatalogResponse>("/catalog"),
  plot: (kind: string, params: QueryParams) =>
    getJSON<PlotSpec>(`/plot/${kind}`, params),
};
