# CS7980 Project — Kagome Lattice MOBO Interface

A full-stack web application that converts a Python Jupyter Notebook into an interactive visualization tool for **multi-objective Bayesian optimization (MOBO)** of Kagome lattice parameters in materials science / quantum physics.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Browser (User)                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │  HTTP (polling every 2s)
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend  (Next.js 15)                        │
│  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐ │
│  │  ParameterForm   │  │   BOProgress     │  │ ResultsTable  │ │
│  │  (DOS inputs)    │  │ (convergence +   │  │ (candidates + │ │
│  └────────┬─────────┘  │  scatter chart)  │  │  refinement)  │ │
│           │            └──────────────────┘  └───────────────┘ │
│  ┌────────▼─────────┐  ┌──────────────────────────────────────┐ │
│  │    DOSPlot       │  │            lib/api.ts                │ │
│  │ (Plotly charts)  │  │    (REST client for all endpoints)   │ │
│  └──────────────────┘  └──────────────────┬───────────────────┘ │
│   Deployed: Vercel                         │                     │
└────────────────────────────────────────────┼────────────────────┘
                                             │  HTTP/JSON
                                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Backend  (FastAPI + Python)                   │
│                                                                  │
│  Routers                    Core Modules                        │
│  ┌─────────────────┐        ┌────────────────────────────────┐  │
│  │   dos.py        │──────▶ │  core/lattice.py               │  │
│  │ POST /compute-  │        │  Kagome_lattice(), lattice_     │  │
│  │       dos       │        │  model(), triangle_rot()       │  │
│  └─────────────────┘        ├────────────────────────────────┤  │
│  ┌─────────────────┐        │  core/dos.py                   │  │
│  │ optimization.py │──────▶ │  DOS_Ganesh(), Wasserstein,    │  │
│  │ POST /run-      │        │  KDE, MMD, KL, Bhattacharyya   │  │
│  │   optimization  │        ├────────────────────────────────┤  │
│  │ GET  /jobs/{id} │        │  core/optimization.py          │  │
│  │ POST /run-local-│        │  run_bo_loop() [background],   │  │
│  │   refinement    │        │  run_local_refinement() COBYLA │  │
│  │ GET  /refinement│        └────────────────────────────────┘  │
│  │      -jobs/{id} │                                            │
│  └─────────────────┘   Job state: in-memory dict (per process) │
│   Deployed: Railway                                              │
└─────────────────────────────────────────────────────────────────┘
```

### Main Components

| Layer | Technology | Responsibility |
|---|---|---|
| **Frontend** | Next.js 15, React 18, TypeScript, Tailwind CSS, Plotly.js | Interactive UI, DOS plots, BO progress, refinement results |
| **Backend** | Python 3.11, FastAPI, Uvicorn | REST API, async job execution |
| **Lattice** | `pybinding` | Kagome tight-binding lattice construction |
| **DOS** | `scipy`, `numpy` | Density of States, spectral distance metrics |
| **Optimization** | `botorch`, `gpytorch`, `torch` | Gaussian Process surrogate, qLogEI acquisition, COBYLA refinement |

### Workflow

```
User sets DOS params
        │
        ▼
POST /compute-dos ──▶ DOS preview plot
        │
User configures BO (n_initial, n_batch, etc.)
        │
        ▼
POST /run-optimization ──▶ job_id
        │
GET /jobs/{job_id} (poll) ──▶ convergence chart + scatter
        │
User clicks "Run Local Refinement"
        │
        ▼
POST /run-local-refinement ──▶ job_id
        │
GET /refinement-jobs/{job_id} (poll) ──▶ refined DOS plots
```

---

## Running the Project

### Prerequisites

- **Python 3.11** (`python3 --version`)
- **Node.js 18+** (`node --version`)
- **npm** (`npm --version`)
- Linux/macOS recommended for `pybinding` compatibility

---

### Backend Setup

```bash
cd backend
```

**1. Create and activate a virtual environment**

```bash
python3.11 -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows
```

**2. Install dependencies**

> `pybinding` requires build tools (`gcc`, `gfortran`, `python3-dev`). On macOS, install Xcode Command Line Tools first (`xcode-select --install`).

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**3. Configure environment**

```bash
cp .env.example .env
# Edit .env as needed (defaults work for local development)
```

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated list of allowed frontend origins |
| `SMOKE_TEST` | `0` | Set to `1` to run with reduced iterations for quick testing |

**4. Start the server**

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

**Health check:**

```bash
curl http://localhost:8000/health
# {"status":"ok"}
```

**Quick smoke test** (runs BO with minimal iterations):

```bash
SMOKE_TEST=1 uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

### Backend with Docker

```bash
cd backend
docker build -t kagome-backend .
docker run -p 8000:8000 -e CORS_ORIGINS=http://localhost:3000 kagome-backend
```

---

### Frontend Setup

```bash
cd frontend
```

**1. Install dependencies**

```bash
npm install
```

**2. Configure environment**

```bash
cp .env.local.example .env.local
# Edit .env.local if your backend runs on a different URL
```

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Backend API base URL |

**3. Start the development server**

```bash
npm run dev
```

The app is available at `http://localhost:3000`.

**4. Production build**

```bash
npm run build
npm run start
```

**5. Lint**

```bash
npm run lint
```

---

### Running Both Services Together

Open two terminal tabs:

```bash
# Terminal 1 — backend
cd backend && source .venv/bin/activate && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
cd frontend && npm run dev
```

Then open `http://localhost:3000`.

---

## Deployment

| Service | Platform | Environment Variables |
|---|---|---|
| Backend | Railway | `CORS_ORIGINS` (set to your Vercel URL), `SMOKE_TEST` (optional) |
| Frontend | Vercel | `NEXT_PUBLIC_API_URL` (set to your Railway URL) |

> Do not run deployment commands directly. Configure environment variables in each platform's dashboard and trigger deployments from there.

---

## Project Structure

```
CS7980_Project_InterfaceDesign/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI app, CORS, router registration
│   │   ├── routers/
│   │   │   ├── dos.py           # POST /api/compute-dos
│   │   │   └── optimization.py  # BO and refinement endpoints
│   │   ├── core/
│   │   │   ├── config.py        # Bounds, batch sizes, BO hyperparams
│   │   │   ├── lattice.py       # Kagome lattice builders
│   │   │   ├── dos.py           # DOS computation and distance metrics
│   │   │   └── optimization.py  # BO pipeline and COBYLA refinement
│   │   └── schemas/
│   │       ├── requests.py      # Pydantic input models
│   │       └── responses.py     # Pydantic output models
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx             # Main page (4 sections)
│   │   ├── globals.css
│   │   └── components/
│   │       ├── ParameterForm.tsx
│   │       ├── DOSPlot.tsx
│   │       ├── BOProgress.tsx
│   │       └── ResultsTable.tsx
│   ├── lib/api.ts               # REST API client
│   ├── types/index.ts           # TypeScript interfaces
│   ├── package.json
│   └── .env.local.example
├── reference/                   # Source notebook (do not modify)
└── implementation_plan.md
```
