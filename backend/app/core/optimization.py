"""Bayesian optimization pipeline and local refinement extracted from the source notebook."""

import numpy as np
import pybinding as pb
import torch
from botorch.models import SingleTaskGP
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition import qLogExpectedImprovement
from botorch import fit_gpytorch_mll
from botorch.sampling.normal import SobolQMCNormalSampler
from scipy.optimize import minimize

from app.core.config import (
    TKWARGS,
    DIMENSION,
    SEED,
    NUM_RESTARTS,
    RAW_SAMPLES,
    MC_SAMPLES,
    TOP_CANDIDATES,
)
from app.core.lattice import lattice_model
from app.core.dos import DOS_Ganesh, DOS_Wasserstein_distance, DOS_log_peak_intensity_loss


def compute_train_obj(
    train_x: torch.Tensor,
    DOS_target: np.ndarray,
    integral_target: float,
    use_peak_loss: bool = False,
) -> torch.Tensor:
    """Evaluate the optimization objective for a batch of (t_a, t_b) parameter pairs.

    Args:
        train_x: Tensor[N, 2] of (t_a, t_b) values.
        DOS_target: Target DOS histogram (numpy array).
        integral_target: Area under the target DOS (for normalization).
        use_peak_loss: If True, use combined loss (Wasserstein + 10×peak);
                       otherwise use Wasserstein only (matches the notebook run).

    Returns:
        Tensor[N, 1] — negative loss (BO maximizes, so we negate).
    """
    results = []
    for input_x in train_x:
        lattice_current, model_current = lattice_model(
            d=0.133,
            t_a=float(input_x[0].item()),
            t_b=float(input_x[1].item()),
        )
        solver_current = pb.solver.lapack(model_current)
        DOS_current, bins_current = DOS_Ganesh(
            lattice_current,
            model_current,
            solver_current,
            sigma=5.0,
            range=[-0.15, 0.25],
            bins=800,
            density=True,
        )
        DOS_current = DOS_current * integral_target
        bins_centers = 0.5 * (bins_current[:-1] + bins_current[1:])
        error_wass = DOS_Wasserstein_distance(bins_centers, DOS_target, DOS_current)
        if use_peak_loss:
            error_peak = DOS_log_peak_intensity_loss(bins_centers, DOS_target, DOS_current)
            error = error_wass + 10.0 * error_peak
        else:
            error = error_wass
        results.append(-error)  # negate for maximization
    return torch.tensor(results, **TKWARGS).reshape(-1, 1)


def generate_initial_data(
    n: int,
    DOS_target: np.ndarray,
    integral_target: float,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    seed: int = SEED,
    use_peak_loss: bool = False,
):
    """Generate the initial Sobol-sampled training set.

    Returns:
        (train_x, train_obj) tensors.
    """
    bounds = torch.stack([lower_bound, upper_bound])
    train_x = draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1)
    train_obj = compute_train_obj(train_x, DOS_target, integral_target, use_peak_loss)
    return train_x, train_obj


def initialize_model(
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    dimension: int = DIMENSION,
    seed: int = SEED,
):
    """Create and configure the Gaussian Process surrogate model.

    Returns:
        (mll, model) — ExactMarginalLogLikelihood and SingleTaskGP.
    """
    torch.manual_seed(seed)
    train_x_norm = normalize(train_x, bounds=torch.stack([lower_bound, upper_bound]))
    model = SingleTaskGP(
        train_X=train_x_norm,
        train_Y=train_obj,
        covar_module=RBFKernel(ard_num_dims=dimension),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model


def optimize_acqf_and_get_observation(
    model,
    train_x: torch.Tensor,
    train_obj: torch.Tensor,
    sampler,
    iteration: int,
    DOS_target: np.ndarray,
    integral_target: float,
    lower_bound: torch.Tensor,
    upper_bound: torch.Tensor,
    standard_bounds: torch.Tensor,
    batch_size: int,
    num_restarts: int = NUM_RESTARTS,
    raw_samples: int = RAW_SAMPLES,
    seed: int = SEED,
    exploration_iteration: int = 10,
    use_peak_loss: bool = False,
):
    """Optimize the acquisition function and return the next batch to evaluate.

    Returns:
        (new_x, new_obj) tensors.
    """
    torch.manual_seed(seed + iteration)
    best_f = (
        torch.quantile(train_obj, 0.9)
        if iteration <= exploration_iteration
        else train_obj.max()
    )
    qEI = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler)
    candidates, _ = optimize_acqf(
        acq_function=qEI,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
        options={},
    )
    new_x = unnormalize(candidates.detach(), bounds=torch.stack([lower_bound, upper_bound]))
    new_obj = compute_train_obj(new_x, DOS_target, integral_target, use_peak_loss)
    return new_x, new_obj


def run_bo_loop(job_id: str, req, jobs: dict) -> None:
    """Synchronous BO loop that runs in a background thread.

    Updates jobs[job_id] progressively so the polling endpoint can track progress.
    """
    try:
        DOS_target = np.array(req.target_dos.dos_counts, dtype=float)
        bins_target = np.array(req.target_dos.bin_edges, dtype=float)
        integral_target = float(np.trapz(DOS_target, bins_target[:-1]))

        lower_bound = torch.tensor(
            [req.bounds.t_a[0], req.bounds.t_b[0]], **TKWARGS
        )
        upper_bound = torch.tensor(
            [req.bounds.t_a[1], req.bounds.t_b[1]], **TKWARGS
        )
        standard_bounds = torch.zeros(2, DIMENSION, **TKWARGS)
        standard_bounds[1] = 1.0

        n_initial = req.n_initial
        n_batch = req.n_batch
        batch_size = req.batch_size
        use_peak_loss = getattr(req, "use_peak_loss", False)

        # Initial data
        train_x, train_obj = generate_initial_data(
            n=n_initial,
            DOS_target=DOS_target,
            integral_target=integral_target,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            seed=SEED,
            use_peak_loss=use_peak_loss,
        )

        mll, model = initialize_model(
            train_x, train_obj, lower_bound, upper_bound, DIMENSION, SEED
        )

        sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([MC_SAMPLES]), seed=SEED
        )

        for iteration in range(1, n_batch + 1):
            torch.manual_seed(SEED + iteration)
            fit_gpytorch_mll(mll)

            new_x, new_obj = optimize_acqf_and_get_observation(
                model=model,
                train_x=train_x,
                train_obj=train_obj,
                sampler=sampler,
                iteration=iteration,
                DOS_target=DOS_target,
                integral_target=integral_target,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                standard_bounds=standard_bounds,
                batch_size=batch_size,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                seed=SEED,
                use_peak_loss=use_peak_loss,
            )

            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # Remove duplicate rows
            x_np = train_x.cpu().numpy()
            _, unique_indices = np.unique(x_np, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)
            uid_tensor = torch.tensor(unique_indices, dtype=torch.long, device=train_x.device)
            train_x = train_x[uid_tensor]
            train_obj = train_obj[uid_tensor]

            mll, model = initialize_model(
                train_x, train_obj, lower_bound, upper_bound, DIMENSION, SEED
            )

            # Top candidates
            topk = min(TOP_CANDIDATES, train_obj.shape[0])
            topk_indices = torch.topk(train_obj.squeeze(), topk).indices
            top_x = train_x[topk_indices].cpu().numpy().tolist()

            best_loss = float(-train_obj.max().item())

            # Thread-safe update (dict assignment is atomic in CPython)
            jobs[job_id].update(
                {
                    "iteration": iteration,
                    "best_loss": best_loss,
                    "train_x": train_x.cpu().numpy().tolist(),
                    "train_obj": train_obj.cpu().numpy().flatten().tolist(),
                    "top_candidates": top_x,
                }
            )

        jobs[job_id]["status"] = "complete"

    except Exception as exc:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(exc)


def run_local_refinement(
    candidates: list[list[float]],
    DOS_target: np.ndarray,
    integral_target: float,
    lower_bound: list[float],
    upper_bound: list[float],
) -> list[dict]:
    """Run COBYLA local refinement on each candidate point.

    Returns:
        List of dicts with keys: x, loss, dos_counts, dos_bin_edges.
    """
    bounds = [(lb, ub) for lb, ub in zip(lower_bound, upper_bound)]

    def local_objective(x):
        xt = torch.tensor(x, **TKWARGS).unsqueeze(0)
        val = compute_train_obj(xt, DOS_target, integral_target)
        return -float(val.item())  # minimize positive loss

    results = []
    for point in candidates:
        initial = np.array(point, dtype=float)
        result = minimize(
            local_objective,
            initial,
            method="COBYLA",
            bounds=bounds,
            options={"maxiter": 50, "tol": 1e-3},
        )
        opt_x = result.x.tolist()
        loss = float(result.fun)

        # Compute DOS at the refined point
        lattice_opt, model_opt = lattice_model(
            d=0.133, t_a=opt_x[0], t_b=opt_x[1]
        )
        solver_opt = pb.solver.lapack(model_opt)
        dos_counts, bin_edges = DOS_Ganesh(
            lattice_opt,
            model_opt,
            solver_opt,
            sigma=5.0,
            range=[-0.15, 0.25],
            bins=800,
            density=True,
        )
        results.append(
            {
                "x": opt_x,
                "loss": loss,
                "dos_counts": (dos_counts * integral_target).tolist(),
                "dos_bin_edges": bin_edges.tolist(),
            }
        )
    return results
