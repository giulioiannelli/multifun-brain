// Shared types mirroring the backend JSON contracts.

export interface ResultItem {
  label: string;
  level: string | null;
  contrast: string | null;
  processing: string | null;
  band: string | null;
  subject: string | null;
  filters: string[];
  error: string | null;
}

export interface Dataset {
  id: string;
  path?: string;
  n_results?: number;
  items: ResultItem[];
  facets?: Record<string, string[]>;
  error?: string;
}

export interface CatalogResponse {
  datasets: Dataset[];
}

export interface HeatmapSpec {
  kind: string;
  label: string;
  n: number;
  z: number[][];
  names: string[];
  networks: string[];
  colors: string[];
  zmin: number;
  zmax: number;
  error?: string;
}
