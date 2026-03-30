export interface DOSRequest {
  t_a: number;
  t_b: number;
  t_nnn?: number;
  v_a?: number;
  v_b?: number;
  v_c?: number;
  d?: number;
  bins?: number;
  energy_range?: [number, number];
  sigma?: number;
}

export interface DOSResponse {
  dos_counts: number[];
  bin_edges: number[];
  integral: number;
}

export interface TargetDOS {
  dos_counts: number[];
  bin_edges: number[];
}

export interface OptimizationBounds {
  t_a: [number, number];
  t_b: [number, number];
}

export interface OptimizationRequest {
  n_initial?: number;
  n_batch?: number;
  batch_size?: number;
  target_dos: TargetDOS;
  bounds?: OptimizationBounds;
  use_peak_loss?: boolean;
}

export interface StartOptimizationResponse {
  job_id: string;
  status: string;
}

export interface JobStatus {
  job_id: string;
  status: "running" | "complete" | "failed";
  iteration?: number;
  total_iterations?: number;
  best_loss?: number;
  train_x?: number[][];
  train_obj?: number[];
  top_candidates?: number[][];
  error?: string;
}

export interface LocalRefinementRequest {
  candidates: number[][];
  target_dos: TargetDOS;
}

export interface RefinedResult {
  x: number[];
  loss: number;
  dos_counts: number[];
  dos_bin_edges: number[];
}

export interface LocalRefinementResponse {
  results: RefinedResult[];
}

export interface RefinementJobStatus {
  job_id: string;
  status: "running" | "complete" | "failed";
  results?: RefinedResult[];
  error?: string;
}
