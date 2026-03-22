from uuid import uuid4
import numpy as np
from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.core.optimization import run_bo_loop, run_local_refinement
from app.schemas.requests import OptimizationRequest, LocalRefinementRequest
from app.schemas.responses import (
    StartJobResponse,
    JobStatusResponse,
    LocalRefinementResponse,
    RefinedResult,
)

router = APIRouter()

# In-memory job store (acceptable for single-user research tool).
# Keys: job_id (str), Values: dict with job state.
jobs: dict[str, dict] = {}


@router.post("/run-optimization", response_model=StartJobResponse)
def start_optimization(
    req: OptimizationRequest,
    background_tasks: BackgroundTasks,
) -> StartJobResponse:
    """Start a Bayesian optimization job asynchronously.

    Returns a job_id that can be polled via GET /api/jobs/{job_id}.
    """
    job_id = str(uuid4())
    jobs[job_id] = {
        "status": "running",
        "iteration": 0,
        "total_iterations": req.n_batch,
        "best_loss": None,
        "train_x": None,
        "train_obj": None,
        "top_candidates": None,
        "error": None,
    }
    # FastAPI runs sync background tasks in a thread-pool executor.
    background_tasks.add_task(run_bo_loop, job_id, req, jobs)
    return StartJobResponse(job_id=job_id, status="running")


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Poll the status and partial/final results of a running or completed BO job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    state = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=state["status"],
        iteration=state.get("iteration"),
        total_iterations=state.get("total_iterations"),
        best_loss=state.get("best_loss"),
        train_x=state.get("train_x"),
        train_obj=state.get("train_obj"),
        top_candidates=state.get("top_candidates"),
        error=state.get("error"),
    )


@router.post("/run-local-refinement", response_model=LocalRefinementResponse)
def local_refinement(req: LocalRefinementRequest) -> LocalRefinementResponse:
    """Run COBYLA local refinement on a list of candidate parameter points."""
    try:
        DOS_target = np.array(req.target_dos.dos_counts, dtype=float)
        bins_target = np.array(req.target_dos.bin_edges, dtype=float)
        integral_target = float(np.trapezoid(DOS_target, bins_target[:-1]))

        raw_results = run_local_refinement(
            candidates=req.candidates,
            DOS_target=DOS_target,
            integral_target=integral_target,
            lower_bound=[-0.5, -0.5],
            upper_bound=[0.5, 0.5],
        )
        return LocalRefinementResponse(
            results=[RefinedResult(**r) for r in raw_results]
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
