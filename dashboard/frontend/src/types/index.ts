// Shared types mirroring the backend JSON contracts.

export interface ResultItem {
  label: string;
  level: string | null;
  task: string | null; // experimental task: co2 / rest (kw sets: null)
  contrast: string | null; // imaging modality: bold / vaso / cbf / noise
  processing: string | null; // pipeline: clean / optcom / optcomMIRdenoised / bpf / raw / …
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

// Signal tab: raw ROI timecourses, keyed by (subject, contrast, processing).
// `contrast` is "" for the older kw datasets (no co2/rest).
export interface SignalEntry {
  subject: string;
  contrast: string;
  processing: string;
}

// One selectable raw dataset (a raw_data_<id> directory).
export interface RawDataset {
  id: string;
  label: string;
  n_subjects: number;
  n_files: number;
  has_contrast: boolean;
}

export interface SignalCatalog {
  dataset: string;
  datasets: RawDataset[];
  has_contrast: boolean;
  entries: SignalEntry[];
  subjects: string[];
  contrasts: string[];
  processings: string[];
  region_names: string[];
  n_regions: number;
  root: string;
}

// Plot specs are heterogeneous; we type them loosely and narrow per builder.
export type PlotSpec = Record<string, any>;

export type QueryParams = Record<string, string | number | undefined | null>;
