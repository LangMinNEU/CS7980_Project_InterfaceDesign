# Implementation Plan

## Goal

Convert the Kagome lattice Bayesian optimization notebook (`reference/source/Copy_of_Theoreticial_problem_1_prod.ipynb`) into a full-stack web app with interactive visualization. Backend deploys to Railway; frontend deploys to Vercel.

---

## Architecture Overview

```
frontend/ (Next.js + React)          backend/ (FastAPI + Python)
┌─────────────────────────┐          ┌─────────────────────────────┐
│  Parameter Input Form   │  ──API── │  /api/run-optimization      │
│  DOS Plot (Plotly)      │          │  /api/run-local-refinement  │
│  BO Progress Chart      │          │  /api/compute-dos           │
│  Results Table          │          │  Scientific compute layer   │
└─────────────────────────┘          └─────────────────────────────┘
```

---

## Phase 1: Backend

### 1.1 Project Setup

```
backend/
├── app/
│   ├── main.py            # FastAPI app entry point
│   ├── routers/
│   │   ├── dos.py         # DOS computation endpoints
│   │   └── optimization.py # BO and local refinement endpoints
│   ├── core/
│   │   ├── lattice.py     # Kagome_lattice, lattice_model, triangle_rot
│   │   ├── dos.py         # DOS_Ganesh and all 8 distance metrics
│   │   └── optimization.py # BO pipeline, local refinement
│   └── schemas/
│       ├── requests.py    # Pydantic input models
│       └── responses.py   # Pydantic output models
├── requirements.txt
├── Dockerfile
└── .env.example
```

### 1.2 Core Science Module (`app/core/`)

Extract the notebook's functions verbatim into modules:

| Source (Notebook) | Target Module | Functions |
|---|---|---|
| Cells 6–8 | `core/lattice.py` | `numpy_to_tensor`, `triangle_rot`, `Kagome_lattice`, `lattice_model` |
| Cells 9–12 | `core/dos.py` | `DOS_Ganesh`, all 8 distance metrics |
| Cells 19–23 | `core/optimization.py` | `compute_train_obj`, `generate_initial_data`, `initialize_model`, `optimize_acqf_and_get_observation`, `local_objective`, `make_logger`, `callback` |

**Global config** (from Cell 4) lives in `core/config.py`:
```python
DIMENSION = 2
LOWER_BOUND = [-0.5, -0.5]
UPPER_BOUND = [0.5, 0.5]
BATCH_SIZE = 5
N_BATCH = 30
N_INITIAL = 20
MC_SAMPLES = 1024
NUM_RESTARTS = 10
RAW_SAMPLES = 512
TOP_CANDIDATES = 5
SEED = 42
```

### 1.3 API Endpoints

#### `POST /api/compute-dos`
Run DOS for a single set of parameters. Used for interactive preview.

**Request:**
```json
{
  "t_a": 0.0285,
  "t_b": 0.075,
  "t_nnn": 0.0,
  "v_a": 0.0, "v_b": 0.0, "v_c": 0.0,
  "d": 0.133,
  "bins": 800,
  "energy_range": [-0.15, 0.25],
  "sigma": 5.0
}
```

**Response:**
```json
{
  "dos_counts": [...],
  "bin_edges": [...],
  "integral": 1.002
}
```

#### `POST /api/run-optimization`
Run the full BO loop. Long-running; returns a job ID for polling.

**Request:**
```json
{
  "n_initial": 20,
  "n_batch": 30,
  "batch_size": 5,
  "target_dos": { "dos_counts": [...], "bin_edges": [...] },
  "bounds": { "t_a": [-0.5, 0.5], "t_b": [-0.5, 0.5] }
}
```

**Response (immediate):**
```json
{ "job_id": "abc123", "status": "running" }
```

#### `GET /api/jobs/{job_id}`
Poll job status and get partial/final results.

**Response:**
```json
{
  "job_id": "abc123",
  "status": "running | complete | failed",
  "iteration": 12,
  "best_loss": 0.0043,
  "train_x": [[...], ...],
  "train_obj": [...],
  "top_candidates": [[0.028, 0.076], ...]
}
```

#### `POST /api/run-local-refinement`
Run COBYLA refinement on provided candidate points.

**Request:**
```json
{
  "candidates": [[0.028, 0.076], [0.031, 0.072]],
  "target_dos": { "dos_counts": [...], "bin_edges": [...] }
}
```

**Response:**
```json
{
  "results": [
    { "x": [0.0285, 0.075], "loss": 0.0021, "dos_counts": [...] }
  ]
}
```

### 1.4 Background Job Handling

Use Python's `asyncio` + a simple in-memory dict for job state (acceptable for single-user research tool). For multi-user scale, swap in Celery + Redis.

```python
# Simple approach
jobs: dict[str, JobState] = {}

@app.post("/api/run-optimization")
async def run_optimization(req: OptimizationRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid4())
    jobs[job_id] = JobState(status="running")
    background_tasks.add_task(run_bo_loop, job_id, req)
    return {"job_id": job_id, "status": "running"}
```

### 1.5 Dependencies (`requirements.txt`)

```
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
pybinding>=1.0.6        # requires build-essential, gfortran (see Dockerfile)
botorch>=0.17.2
gpytorch>=1.15.2
torch>=2.10.0
scipy>=1.13.0
numpy>=1.26.0
shapely>=2.0.0
pointpats>=2.4.0
pandas>=2.0.0
matplotlib>=3.8.0
python-dotenv>=1.0.0
```

### 1.6 Dockerfile

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    build-essential python3-dev gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 1.7 Environment Variables (Railway)

| Variable | Description | Example |
|---|---|---|
| `PORT` | Server port (auto-set by Railway) | `8000` |
| `CORS_ORIGINS` | Allowed frontend origins | `https://your-app.vercel.app` |
| `SMOKE_TEST` | Set to `1` to use reduced counts | `0` |

---

## Phase 2: Frontend

### 2.1 Project Setup

```
frontend/
├── app/
│   ├── page.tsx           # Main layout
│   ├── components/
│   │   ├── DOSPlot.tsx    # Plotly DOS chart
│   │   ├── BOProgress.tsx # Iteration loss chart
│   │   ├── ResultsTable.tsx
│   │   └── ParameterForm.tsx
│   ├── lib/
│   │   └── api.ts         # API client (fetch wrappers)
│   └── types/
│       └── index.ts       # TypeScript types
├── next.config.ts
├── package.json
└── .env.local.example
```

**Stack:** Next.js 14 (App Router) + TypeScript + Tailwind CSS + Plotly.js

### 2.2 UI Sections

**Section 1 — Target DOS Setup**
- Upload a CSV/JSON of target DOS **or** set target parameters manually (compute on backend)
- Display computed target DOS curve

**Section 2 — Optimization Configuration**
- Sliders/inputs for `N_INITIAL`, `N_BATCH`, `BATCH_SIZE`, search bounds
- "Run Optimization" button

**Section 3 — Live BO Progress**
- Polling `/api/jobs/{job_id}` every 2s
- Line chart: best loss per iteration
- 2D scatter: `(t_a, t_b)` colored by loss (mirrors `plot_2D_train_obj`)

**Section 4 — Results**
- Table: top-5 candidates with their losses and DOS overlap
- "Run Local Refinement" button
- Final DOS comparison plot: target (dashed) vs refined (solid)

### 2.3 DOS Plot Component

Use Plotly.js traces:
- Target DOS: blue dashed line
- Current best DOS: red solid line
- X axis: energy (eV), Y axis: DOS (a.u.)

```tsx
import Plot from 'react-plotly.js';

const data = [
  { x: binCenters, y: targetDOS, name: 'Target', line: { dash: 'dash', color: 'blue' } },
  { x: binCenters, y: currentDOS, name: 'Current', line: { color: 'red' } },
];
```

### 2.4 API Client (`lib/api.ts`)

```typescript
const BASE_URL = process.env.NEXT_PUBLIC_API_URL;

export async function computeDOS(params: DOSRequest): Promise<DOSResponse> {
  const res = await fetch(`${BASE_URL}/api/compute-dos`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  });
  return res.json();
}

export async function startOptimization(req: OptimizationRequest): Promise<{ job_id: string }> { ... }
export async function pollJob(jobId: string): Promise<JobStatus> { ... }
export async function runLocalRefinement(req: LocalRefinementRequest): Promise<LocalRefinementResponse> { ... }
```

### 2.5 Environment Variables (Vercel)

| Variable | Description | Example |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | Backend Railway URL | `https://your-backend.railway.app` |

---

## Phase 3: Integration & Polish

### 3.1 CORS

Backend must allow the Vercel frontend origin:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 3.2 Error Handling

- Backend: return `422` for invalid parameters, `500` with message for compute errors
- Frontend: show inline error banners; disable submit during in-flight requests

### 3.3 Performance Notes

- `DOS_Ganesh` takes ~2–5s per call (10,000 k-point eigensolves). The full BO loop (170+ evaluations) may take 10–30 minutes on CPU. Consider:
  - Streaming progress updates via SSE (`/api/jobs/{id}/stream`) instead of polling
  - Reducing `N_INITIAL`/`N_BATCH` as user-configurable options with defaults lowered for the UI

---

## Implementation Order

1. `backend/core/` — extract notebook functions, no API yet, run locally to verify
2. `backend/app/` — add FastAPI, add `POST /api/compute-dos` first (synchronous, fast to test)
3. `backend/app/` — add job-based endpoints for BO loop
4. `frontend/` — scaffold Next.js, implement `ParameterForm` + `DOSPlot`
5. `frontend/` — connect to backend, implement BO progress polling
6. Integration testing: run full BO from browser UI
7. Prepare deployment docs (env var lists above)

---

## Testing

| Layer | What to test |
|---|---|
| Core science | `compute_train_obj` returns a scalar; `DOS_Ganesh` output shape matches `bins` |
| API | `/api/compute-dos` returns valid DOS for known parameters (`t_a=0.0285, t_b=0.075`) |
| Integration | Full BO loop with `N_INITIAL=4, N_BATCH=5` (smoke test) completes and returns top candidates |
| Frontend | DOS plot renders without crash; polling stops when job is `complete` |

Run backend locally:
```bash
cd backend && uvicorn app.main:app --reload
```

Run frontend locally:
```bash
cd frontend && npm run dev
```
