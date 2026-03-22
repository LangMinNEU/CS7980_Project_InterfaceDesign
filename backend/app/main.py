import os
import warnings
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from botorch.exceptions import BadInitialCandidatesWarning

from app.routers import dos, optimization

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

app = FastAPI(
    title="Kagome BO API",
    description="Backend for Kagome lattice Bayesian optimization.",
    version="1.0.0",
)

# CORS — allow the Vercel frontend origin (set CORS_ORIGINS env var in production).
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dos.router, prefix="/api")
app.include_router(optimization.router, prefix="/api")


@app.get("/health")
def health():
    return {"status": "ok"}
