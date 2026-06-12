// Thin typed fetch wrappers around the FastAPI backend. All calls go through
// the same-origin /api prefix (Vite proxies it in dev; FastAPI serves it in prod).
import type { CatalogResponse, HeatmapSpec } from "../types";

const BASE = "/api";

async function getJSON<T>(
  path: string,
  params?: Record<string, string>,
): Promise<T> {
  const qs = params ? "?" + new URLSearchParams(params).toString() : "";
  const res = await fetch(`${BASE}${path}${qs}`);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  catalog: () => getJSON<CatalogResponse>("/catalog"),
  heatmap: (dataset: string, label: string) =>
    getJSON<HeatmapSpec>("/plot/heatmap", { dataset, label }),
};
