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

// Plot specs are heterogeneous; we type them loosely and narrow per builder.
export type PlotSpec = Record<string, any>;

export type QueryParams = Record<string, string | number | undefined | null>;
