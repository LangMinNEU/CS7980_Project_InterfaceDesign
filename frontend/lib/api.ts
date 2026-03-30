import type {
  DOSRequest,
  DOSResponse,
  OptimizationRequest,
  StartOptimizationResponse,
  JobStatus,
  LocalRefinementRequest,
  RefinementJobStatus,
} from "@/types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`POST ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`GET ${path} failed (${res.status}): ${text}`);
  }
  return res.json() as Promise<T>;
}

export async function computeDOS(params: DOSRequest): Promise<DOSResponse> {
  return post<DOSResponse>("/api/compute-dos", params);
}

export async function startOptimization(
  req: OptimizationRequest
): Promise<StartOptimizationResponse> {
  return post<StartOptimizationResponse>("/api/run-optimization", req);
}

export async function pollJob(jobId: string): Promise<JobStatus> {
  return get<JobStatus>(`/api/jobs/${jobId}`);
}

export async function startLocalRefinement(
  req: LocalRefinementRequest
): Promise<StartOptimizationResponse> {
  return post<StartOptimizationResponse>("/api/run-local-refinement", req);
}

export async function pollRefinementJob(jobId: string): Promise<RefinementJobStatus> {
  return get<RefinementJobStatus>(`/api/refinement-jobs/${jobId}`);
}
