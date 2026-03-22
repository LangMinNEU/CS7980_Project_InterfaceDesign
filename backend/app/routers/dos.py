from fastapi import APIRouter, HTTPException
import numpy as np
import pybinding as pb

from app.core.lattice import lattice_model
from app.core.dos import DOS_Ganesh
from app.schemas.requests import DOSRequest
from app.schemas.responses import DOSResponse

router = APIRouter()


@router.post("/compute-dos", response_model=DOSResponse)
def compute_dos(req: DOSRequest) -> DOSResponse:
    """Compute the DOS for a single set of Kagome lattice parameters.

    This endpoint is fast enough for interactive previews (~2–5 s per call).
    """
    try:
        lattice, model = lattice_model(
            d=req.d,
            t_a=req.t_a,
            t_b=req.t_b,
            t_nnn=req.t_nnn,
            v_a=req.v_a,
            v_b=req.v_b,
            v_c=req.v_c,
        )
        solver = pb.solver.lapack(model)
        dos_counts, bin_edges = DOS_Ganesh(
            lattice,
            model,
            solver,
            bins=req.bins,
            range=req.energy_range,
            density=True,
            sigma=req.sigma,
        )
        integral = float(np.trapezoid(dos_counts, bin_edges[:-1]))
        return DOSResponse(
            dos_counts=dos_counts.tolist(),
            bin_edges=bin_edges.tolist(),
            integral=integral,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
