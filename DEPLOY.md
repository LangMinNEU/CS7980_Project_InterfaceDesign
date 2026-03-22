# Deployment Guide

This guide covers deploying the Kagome BO application:
- **Backend** → [Railway](https://railway.app) (Docker-based)
- **Frontend** → [Vercel](https://vercel.com) (Next.js)

---

## Prerequisites

- Railway account and [Railway CLI](https://docs.railway.app/develop/cli) (or use the Railway dashboard)
- Vercel account and [Vercel CLI](https://vercel.com/docs/cli) (or use the Vercel dashboard)
- Git repository pushed to GitHub/GitLab

---

## 1. Deploy the Backend to Railway

### Step 1 — Create a new Railway project

```bash
cd backend
railway login
railway init        # creates a new project
railway up          # builds and deploys from Dockerfile
```

Or use the Railway dashboard: **New Project → Deploy from GitHub repo → select the repo → set root directory to `backend/`**.

### Step 2 — Set environment variables

In the Railway project dashboard go to **Variables** and add:

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `CORS_ORIGINS` | Yes | Comma-separated list of allowed frontend origins. Set to your Vercel URL after frontend is deployed. Use `*` temporarily if testing. | `https://your-app.vercel.app` |
| `SMOKE_TEST` | No | Set to `1` to run with reduced iteration counts (fast sanity check). Default `0`. | `0` |

> Railway automatically injects a `PORT` environment variable. The Dockerfile is already configured to read it via `${PORT:-8000}`.

### Step 3 — Verify

Once deployed, Railway provides a public URL (e.g. `https://your-backend.up.railway.app`). Confirm it is live:

```bash
curl https://your-backend.up.railway.app/health
# Expected: {"status":"ok"}
```

Also check the interactive API docs:
```
https://your-backend.up.railway.app/docs
```

---

## 2. Deploy the Frontend to Vercel

### Step 1 — Create a new Vercel project

```bash
cd frontend
vercel
# Follow prompts: link to project, framework = Next.js, root = frontend/
```

Or use the Vercel dashboard: **New Project → Import from GitHub → select the repo → set root directory to `frontend/`**.

### Step 2 — Set environment variables

In the Vercel project dashboard go to **Settings → Environment Variables** and add:

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Yes | Public URL of the Railway backend (no trailing slash). | `https://your-backend.up.railway.app` |

> This variable must start with `NEXT_PUBLIC_` to be available in the browser bundle. It is read in `frontend/lib/api.ts`.

### Step 3 — Redeploy (if env var was set after initial deploy)

```bash
vercel --prod
```

Or trigger a redeploy from the Vercel dashboard.

### Step 4 — Verify

Open the Vercel URL and confirm the app loads. Try computing a DOS with the default parameters to verify end-to-end connectivity.

---

## 3. Update Backend CORS After Frontend is Live

Once you have the Vercel URL, go back to Railway and update `CORS_ORIGINS`:

```
CORS_ORIGINS=https://your-app.vercel.app
```

Then redeploy the backend (or Railway will pick up the variable change automatically).

---

## 4. Local Development

### Backend

```bash
cd backend
cp .env.example .env          # edit CORS_ORIGINS=http://localhost:3000
pip install -r requirements.txt
uvicorn app.main:app --reload
# API running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend

```bash
cd frontend
cp .env.local.example .env.local   # NEXT_PUBLIC_API_URL=http://localhost:8000
npm install
npm run dev
# App running at http://localhost:3000
```

---

## 5. Environment Variable Reference

### Backend (`backend/.env.example`)

| Variable | Default | Description |
|----------|---------|-------------|
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins. Use `*` locally; set to Vercel URL in production. |
| `SMOKE_TEST` | `0` | Set to `1` for reduced N_BATCH/N_INITIAL counts (for fast testing). |
| `PORT` | `8000` | Injected by Railway automatically. Do not set manually. |

### Frontend (`frontend/.env.local.example`)

| Variable | Default | Description |
|----------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Full URL of the FastAPI backend, no trailing slash. |

---

## 6. Smoke Test the Full Pipeline

1. Open the app in a browser.
2. In **Target DOS Setup**, click **Compute DOS** with default parameters. You should see a DOS curve appear.
3. Set `N_INITIAL = 4`, `N_BATCH = 5` in the optimization config (or set `SMOKE_TEST=1` on the backend).
4. Click **Run Optimization**. The progress chart should update every ~2 seconds.
5. Once complete, click **Run Local Refinement** on the top candidates.
6. Confirm the final DOS comparison plot renders.
