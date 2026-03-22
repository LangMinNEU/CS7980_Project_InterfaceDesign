from pydantic import BaseModel, Field
from typing import Optional


class DOSRequest(BaseModel):
    t_a: float
    t_b: float
    t_nnn: float = 0.0
    v_a: float = 0.0
    v_b: float = 0.0
    v_c: float = 0.0
    d: float = 0.133
    bins: int = Field(default=800, ge=100, le=5000)
    energy_range: list[float] = [-0.15, 0.25]
    sigma: float = 5.0


class TargetDOS(BaseModel):
    dos_counts: list[float]
    bin_edges: list[float]


class OptimizationBounds(BaseModel):
    t_a: list[float] = [-0.5, 0.5]
    t_b: list[float] = [-0.5, 0.5]


class OptimizationRequest(BaseModel):
    n_initial: int = Field(default=20, ge=1, le=200)
    n_batch: int = Field(default=30, ge=1, le=200)
    batch_size: int = Field(default=5, ge=1, le=20)
    target_dos: TargetDOS
    bounds: OptimizationBounds = OptimizationBounds()
    use_peak_loss: bool = False


class LocalRefinementRequest(BaseModel):
    candidates: list[list[float]]
    target_dos: TargetDOS
