# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project converts a Python Jupyter Notebook (`reference/source/Copy_of_Theoreticial_problem_1_prod.ipynb`) into a full-stack web application with better visualization capabilities. The notebook implements multi-objective Bayesian optimization (MOBO) for finding optimal Kagome lattice parameters in materials science/quantum physics.

## Architecture

The project follows a standard frontend/backend split:

- **`backend/`** — Python API server wrapping the notebook's scientific computation logic. Target deployment: Railway.
- **`frontend/`** — Web UI providing interactive visualization. Target deployment: Vercel.
- **`reference/`** — Source material only (not part of the app). Do not modify.

Both directories are currently empty; the implementation has not started yet. See `implementation_plan.md` at the repo root for the full build plan (API design, file structure, endpoint contracts, env vars, and implementation order).

## Deployment

- **Do not deploy** backend or frontend yourself. Instead, provide environment variable documentation needed for Railway (backend) and Vercel (frontend) deployments.

## Source Notebook Domain

The notebook (`reference/source/Copy_of_Theoreticial_problem_1_prod.ipynb`) contains:

- **Lattice modeling:** `Kagome_lattice()`, `lattice_model()`, `triangle_rot()` using `pybinding`
- **DOS calculations:** `DOS_Ganesh()`, `DOS_squared_error()`, `DOS_Wasserstein_distance()`, `DOS_KDE_Wasserstein_distance()` using `scipy`, `numpy`
- **Bayesian optimization:** `initialize_model()`, `optimize_acqf_and_get_observation()`, `compute_train_obj()` using `botorch`, `gpytorch`, `torch`
- **Utilities:** `generate_initial_data()`, `plot_2D_train_obj()`, `numpy_to_tensor()`

Key dependencies: `pybinding`, `botorch`, `gpytorch`, `scipy`, `numpy`, `torch`, `pandas`, `matplotlib`, `shapely`, `pointpats`
