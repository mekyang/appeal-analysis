// FastAPI backend base URL - change this to your actual backend address
const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface PreprocessResponse {
  message: string;
  total_rows: number;
  valid_rows: number;
  removed_rows: number;
}

export interface ClusterResponse {
  message: string;
  total_texts: number;
  n_clusters: number;
  n_noise: number;
  output_file: string;
}

export interface EvaluateResponse {
  message: string;
  metrics: Record<string, string | number>;
}

export interface KeywordsResponse {
  message: string;
  result: Array<{ Cluster: number; LLM_Keywords: string }>;
  output_file: string;
}

export interface LoadStateResponse {
  message: string;
  total_rows: number;
  n_clusters: number;
  n_noise: number;
}

export async function preprocessFile(
  file: File,
  extractorType: string,
  useNer: boolean,
  columnName: string
): Promise<PreprocessResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("extractor_type", extractorType);
  formData.append("use_ner", String(useNer));
  formData.append("column_name", columnName);

  const res = await fetch(`${API_BASE}/api/preprocess`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function clusterData(
  file: File,
  textColumn: string,
  originalColumn: string,
  nNeighbors: number,
  nComponents: number,
  minClusterSize: number,
  keywordTopN: number,
  autoSave: boolean
): Promise<ClusterResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_column", textColumn);
  formData.append("original_column", originalColumn);
  formData.append("n_neighbors", String(nNeighbors));
  formData.append("n_components", String(nComponents));
  formData.append("min_cluster_size", String(minClusterSize));
  formData.append("keyword_top_n", String(keywordTopN));
  formData.append("auto_save", String(autoSave));

  const res = await fetch(`${API_BASE}/api/cluster`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function evaluateCluster(
  file: File,
  textColumn: string,
  clusterColumn: string
): Promise<EvaluateResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("text_column", textColumn);
  formData.append("cluster_column", clusterColumn);

  const res = await fetch(`${API_BASE}/api/evaluate`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function extractKeywords(
  file: File,
  apiKey: string,
  baseUrl: string,
  textCol: string
): Promise<KeywordsResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("api_key", apiKey);
  formData.append("base_url", baseUrl);
  formData.append("text_col", textCol);

  const res = await fetch(`${API_BASE}/api/extract-keywords`, {
    method: "POST",
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

export async function downloadFile(filename: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/api/download/${filename}`);

  if (!res.ok) {
    throw new Error(`Download failed: HTTP ${res.status}`);
  }

  return res.blob();
}

export async function loadState(): Promise<LoadStateResponse> {
  const res = await fetch(`${API_BASE}/api/load-state`);

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}
