from pydantic import BaseModel
from typing import Optional


class DOSResponse(BaseModel):
    dos_counts: list[float]
    bin_edges: list[float]
    integral: float


class StartJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    iteration: Optional[int] = None
    total_iterations: Optional[int] = None
    best_loss: Optional[float] = None
    train_x: Optional[list[list[float]]] = None
    train_obj: Optional[list[float]] = None
    top_candidates: Optional[list[list[float]]] = None
    error: Optional[str] = None


class RefinedResult(BaseModel):
    x: list[float]
    loss: float
    dos_counts: list[float]
    dos_bin_edges: list[float]


class LocalRefinementResponse(BaseModel):
    results: list[RefinedResult]


class RefinementJobStatusResponse(BaseModel):
    job_id: str
    status: str
    results: Optional[list[RefinedResult]] = None
    error: Optional[str] = None
