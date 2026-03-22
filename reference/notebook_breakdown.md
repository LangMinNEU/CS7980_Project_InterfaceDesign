# Notebook Breakdown: `Copy_of_Theoreticial_problem_1_prod.ipynb`

## 1. Scientific Overview

This notebook solves an **inverse problem in condensed matter physics**: given an experimentally measured density-of-states (DOS) spectrum for a Kagome lattice material, find the tight-binding parameters (hopping terms `t_a`, `t_b`) that reproduce it computationally.

**Optimization pipeline:**
1. **Target system** — Build a Kagome lattice with known parameters and compute its DOS as the reference.
2. **Bayesian Optimization (BO)** — Use a Gaussian Process surrogate to efficiently explore the 2D parameter space (`t_a`, `t_b` ∈ [-0.5, 0.5]).
3. **Local refinement** — Apply gradient-free COBYLA optimization starting from the top-5 BO candidates to fine-tune each solution.
4. **Visualization** — Compare target vs. BO-suggested vs. locally-optimized DOS spectra.

---

## 2. Execution Flow

### Phase 1: Setup (Cells 0–5)
| Cell | Action |
|------|--------|
| 0 | `pip install` — `pybinding-dev`, `botorch`, `gpytorch`, `shapely`, `pointpats` |
| 1 | `apt-get install` — `build-essential`, `python3-dev`, `gfortran` |
| 2 | Version check: prints `sys`, `scipy`, `torch`, `pybinding`, `botorch`, `gpytorch` versions |
| 3 | All import statements (35 imports) |
| 4 | Global configuration constants and output directory creation |
| 5 | Random seed initialization for `random`, `numpy`, `torch` |

### Phase 2: Function Definitions (Cells 6–12)
Defines all utility, lattice, DOS, and loss functions (no computation yet).

### Phase 3: Target System (Cells 13–18)
| Cell | Action |
|------|--------|
| 13 | Create target Kagome lattice: `t_a=0.0285`, `t_b=0.075`, plot structure |
| 14 | Compute target DOS via `DOS_Ganesh` (σ=5, bins=800, range=[-0.15, 0.25]) |
| 15 | Validate: print integral of `DOS_target` |
| 16 | Plot and save target DOS (`tp1_DOS_Target.png/pdf`) |
| 17 | (Commented-out test code — inactive) |
| 18 | Compute `integral_target = np.trapezoid(DOS_target, bins_target[:-1])` for normalization |

### Phase 4: Bayesian Optimization (Cells 19–24)
| Cell | Action |
|------|--------|
| 19 | Define `compute_train_obj` |
| 20 | Define `generate_initial_data` |
| 21 | Define `initialize_model` |
| 22 | Define `optimize_acqf_and_get_observation` |
| 23 | Define `plot_2D_train_obj` |
| 24 | **Run main BO loop** — 20 initial + 30×5 = 150 evaluations |

### Phase 5: Local Refinement (Cells 25–28)
| Cell | Action |
|------|--------|
| 25 | Define `local_objective` |
| 26 | Print top-5 BO results |
| 27 | Run COBYLA on each of 5 candidates; define `make_logger` and `callback` |
| 28 | Print final refined results |

### Phase 6: Visualization (Cells 29–30)
| Cell | Action |
|------|--------|
| 29 | 5-panel subplot: target vs. BO vs. local for each top candidate |
| 30 | 5 individual high-resolution figures saved as PNG/PDF |

---

## 3. Global Configuration & Constants

| Variable | Type | Value | Purpose |
|----------|------|-------|---------|
| `dimension` | int | 2 | Search space dimensionality (t_a, t_b) |
| `lower_bound` | Tensor | `[-0.5, -0.5]` | Lower bounds for parameters |
| `upper_bound` | Tensor | `[0.5, 0.5]` | Upper bounds for parameters |
| `least_count` | Tensor | `[0.1, 0.1]` | Minimum discretization step |
| `standard_bounds` | Tensor | `[[0,0],[1,1]]` | Normalized bounds used internally by BO |
| `tkwargs` | dict | `{dtype: float64, device: cuda/cpu}` | Torch tensor configuration |
| `SMOKE_TEST` | bool | env flag | Reduces all counts for quick testing |
| `NUM_RESTARTS` | int | 10 (2 if smoke) | Restarts for acquisition function optimization |
| `RAW_SAMPLES` | int | 512 (4 if smoke) | Raw candidate samples for acquisition |
| `MC_SAMPLES` | int | 1024 (16 if smoke) | Monte Carlo samples for qLogEI |
| `BATCH_SIZE` | int | 5 | New points evaluated per BO iteration |
| `N_BATCH` | int | 30 (5 if smoke) | Total BO iterations |
| `N_INITIAL` | int | 20 (4 if smoke) | Initial Sobol-sampled training points |
| `top_candidates` | int | 5 | Number of BO solutions passed to local refinement |
| `seed` | int | 42 | Global random seed |
| `subfolder` | str | `'tp1_run3_wasserstein_only'` | Output subdirectory name |
| `figure_folder` | str | ORNL path | Base directory for all output artifacts |
| `integral_target` | float | computed | Area under target DOS (for normalization) |

---

## 4. Dependencies

**Lattice Physics**
- `pybinding` (v1.0.6) — Tight-binding model construction, BZ sampling, LAPACK eigensolver

**Scientific Computing**
- `numpy` — Array operations
- `scipy.stats` — `wasserstein_distance`, `gaussian_kde`
- `scipy.signal` — `find_peaks`
- `scipy.integrate` — `simpson`, `trapezoid`
- `scipy.ndimage` — `gaussian_filter1d`
- `scipy.optimize` — `minimize` (COBYLA method)

**Machine Learning / Bayesian Optimization**
- `torch` (v2.10.0) — Tensor operations, GPU support
- `botorch` (v0.17.2) — `SingleTaskGP`, `qLogExpectedImprovement`, `optimize_acqf`, `fit_gpytorch_mll`, `draw_sobol_samples`, `SobolQMCNormalSampler`
- `gpytorch` (v1.15.2) — `RBFKernel`, `ScaleKernel`, `ExactMarginalLogLikelihood`

**Visualization**
- `matplotlib.pyplot`, `matplotlib.gridspec`

**Geometry**
- `shapely.geometry.Polygon` — Triangular boundary for finite lattice
- `pointpats` — Spatial point pattern analysis

**Utilities**
- `pandas`, `time`, `warnings`, `random`, `os`, `math`, `contextlib`

---

## 5. Function Reference

### 5.1 Utilities

---

#### `numpy_to_tensor(np_array, requires_grad=False)`

**Purpose:** Converts a NumPy array to a PyTorch tensor using the global `tkwargs` dtype and device.

**Parameters:**
- `np_array` — `np.ndarray`: Input array
- `requires_grad` — `bool` (default `False`): Whether to track gradients

**Returns:** `torch.Tensor` on the configured device

**Notes:** Thin wrapper that ensures all tensors share the same dtype/device config defined in `tkwargs`.

---

#### `triangle_rot(a, b, c, shx, shy)`

**Purpose:** Creates a PyBinding `Polygon` defining the triangular boundary used to shape the finite-size Kagome lattice.

**Parameters:**
- `a`, `b`, `c` — `float`: Edge lengths controlling triangle geometry
- `shx`, `shy` — `float`: x and y shift (translation) of the triangle

**Returns:** `pb.Polygon` with vertices at `[[shx, -a/2+shy], [shx, b/2+shy], [c+shx, shy]]`

**Notes:** The shape boundary is applied in `lattice_model` to cut the infinite lattice into a finite triangular patch.

---

### 5.2 Lattice Builders

---

#### `Kagome_lattice(d, t_a=-1, t_b=-1, t_nnn=0, v_a=0, v_b=0, v_c=0)`

**Purpose:** Defines the Kagome tight-binding lattice with three sublattices and configurable hopping/on-site parameters.

**Parameters:**
- `d` — `float`: Nearest-neighbor distance; Bravais lattice period `a = 2d`
- `t_a` — `float` (default `-1`): NN hopping amplitude on sublattice A bonds
- `t_b` — `float` (default `-1`): NN hopping amplitude on sublattice B bonds
- `t_nnn` — `float` (default `0`): Next-nearest-neighbor hopping
- `v_a`, `v_b`, `v_c` — `float` (default `0`): On-site potentials for sublattices A, B, C

**Returns:** `pb.Lattice` — PyBinding lattice object

**Algorithm:**
1. Create 2D triangular Bravais lattice with vectors `a1 = [a, 0]`, `a2 = [a/2, a√3/2]`
2. Add 3 sublattice sites (A, B, C) at fractional coordinates
3. Add 6 nearest-neighbor hoppings (mix of `t_a` and `t_b` bonds)
4. Add 6 next-nearest-neighbor hoppings with `t_nnn`

**Notes:** The two distinct NN hopping values (`t_a`, `t_b`) break the uniform Kagome symmetry, making this a **distorted** Kagome lattice. This is the key degree of freedom being optimized.

---

#### `lattice_model(d=0.133, t_a=-1.0, t_b=-1.0, t_nnn=0, v_a=0, v_b=0, v_c=0, plot_figure=False, save_figure=False)`

**Purpose:** Builds a complete finite-size Kagome lattice by combining `Kagome_lattice` with a triangular shape boundary.

**Parameters:**
- `d` — `float` (default `0.133`): Nearest-neighbor distance
- `t_a`, `t_b`, `t_nnn`, `v_a`, `v_b`, `v_c` — same as `Kagome_lattice`
- `plot_figure` — `bool`: Show structure plot
- `save_figure` — `bool`: Save PNG and PDF of the structure

**Returns:** `(lattice, model)` — tuple of PyBinding `Lattice` and `Model` objects

**Algorithm:**
1. Call `Kagome_lattice(d, t_a, t_b, t_nnn, v_a, v_b, v_c)`
2. Create `pb.Model` with the lattice + `triangle_rot(...)` shape boundary
3. Optionally plot/save

**Notes:** The `model` object contains the Hamiltonian matrix for the finite system; the solver is created separately (by the caller) to allow different solver types.

---

### 5.3 DOS Computation

---

#### `DOS_Ganesh(lattice, model, solver, bins=1000, range=[-6,6], density=True, size=10000, sigma=12.0, plot_figure=False)`

**Purpose:** Computes the density-of-states (DOS) by uniformly sampling the Brillouin zone and histogramming the eigenvalues.

**Parameters:**
- `lattice` — `pb.Lattice`: Lattice defining the BZ geometry
- `model` — `pb.Model`: Finite-size model (provides the Hamiltonian)
- `solver` — PyBinding solver (e.g., LAPACK)
- `bins` — `int` (default `1000`): Number of energy histogram bins
- `range` — `list[float, float]` (default `[-6, 6]`): Energy range in eV
- `density` — `bool` (default `True`): Normalize histogram to probability density
- `size` — `int` (default `10000`): Number of k-points to sample
- `sigma` — `float` (default `12.0`): Gaussian filter standard deviation (bin units)
- `plot_figure` — `bool`: Show DOS plot

**Returns:** `(DOS_counts, bin_edges)` — `(np.ndarray, np.ndarray)`

**Algorithm:**
1. Get Brillouin zone coordinates from `lattice`
2. Sample `size` k-points uniformly within the BZ
3. For each k-point, compute eigenvalues via `solver`
4. Collect all eigenvalues and histogram into `bins` bins over `range`
5. Apply `gaussian_filter1d(counts, sigma=sigma)` to smooth

**Notes:** This is the **most expensive function** in the pipeline — each call computes 10,000 eigenvalue problems. The Gaussian filter with `sigma=5` (used in production) provides smooth, physically realistic DOS curves. Default `sigma=12` is larger; the actual call in Cell 14 overrides to `sigma=5`.

---

### 5.4 Distance & Loss Metrics

All distance functions take DOS arrays as input and return a scalar. They are used to quantify how different the current DOS is from the target.

---

#### `DOS_squared_error(DOS_target, DOS_current)`

**Purpose:** Simple sum of squared differences (L2 loss).

**Parameters:**
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms of equal length

**Returns:** `float` — `sum((target - current)²)`

**Notes:** Not used in the final loss; provided as a baseline metric.

---

#### `DOS_Wasserstein_distance(bins, DOS_target, DOS_current)`

**Purpose:** Main loss metric — computes the Wasserstein-1 (Earth Mover's) distance between the two DOS distributions.

**Parameters:**
- `bins` — `np.ndarray`: Bin edges array (length = `len(DOS) + 1`)
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms

**Returns:** `float` — Wasserstein distance

**Algorithm:**
1. Normalize both DOS arrays: divide each by its sum
2. Compute bin centers from edges
3. Call `scipy.stats.wasserstein_distance(centers, centers, target_norm, current_norm)`

**Notes:** This is one of the two terms in the combined loss used in `compute_train_obj`. It measures global spectral shape mismatch — insensitive to small local misalignments.

---

#### `DOS_KDE_Wasserstein_distance(bins, DOS_target, DOS_current, bw_method=None, grid_points=None)`

**Purpose:** Wasserstein distance computed via Kernel Density Estimation (KDE) rather than raw histogram values.

**Parameters:**
- `bins` — `np.ndarray`: Bin edges
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms
- `bw_method` — bandwidth selector for KDE (passed to `scipy.stats.gaussian_kde`)
- `grid_points` — `int` or `None`: Number of evaluation grid points (defaults to `len(bins)`)

**Returns:** `float` — Wasserstein distance on KDE-smoothed distributions

**Algorithm:**
1. Normalize both DOS arrays
2. Create `gaussian_kde` objects for target and current
3. Evaluate KDEs on a dense grid
4. Call `wasserstein_distance` on the KDE-evaluated distributions

**Notes:** More robust to binning artifacts than `DOS_Wasserstein_distance` but slower due to KDE evaluation. Used experimentally; not in the final production loss.

---

#### `DOS_MMD_Gaussian(bins, DOS_target, DOS_current, lengthscale=0.05)`

**Purpose:** Maximum Mean Discrepancy (MMD) with a Gaussian (RBF) kernel.

**Parameters:**
- `bins` — `np.ndarray`: Bin edges (used to derive bin centers)
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms
- `lengthscale` — `float` (default `0.05`): RBF kernel bandwidth

**Returns:** `float` — MMD² value

**Algorithm:**
Computes: `MMD² = p·K·p − 2·p·K·q + q·K·q`
where `K[i,j] = exp(−(x_i − x_j)² / (2 · lengthscale²))` and `p`, `q` are the normalized DOS vectors.

**Notes:** Experimental metric; not used in production loss. Quadratic in the number of bins, so can be slow for large `bins`.

---

#### `DOS_Bhattacharya_distance(DOS_target, DOS_current)`

**Purpose:** Bhattacharyya distance between two distributions.

**Parameters:**
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms (assumed non-negative)

**Returns:** `float` — `−ln(Σ √(p·q))`

**Notes:** Experimental metric. Undefined when distributions have non-overlapping support (returns −ln(0) = inf).

---

#### `DOS_KLDivergence_distance(DOS_target, DOS_current)`

**Purpose:** Kullback-Leibler divergence from `DOS_current` to `DOS_target`.

**Parameters:**
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms

**Returns:** `float` — `Σ p·ln(q/p)`

**Notes:** Experimental metric. Asymmetric; numerically unstable where `DOS_current ≈ 0`. Not used in production.

---

#### `DOS_Helinger_distance(DOS_target, DOS_current, bins)`

**Purpose:** Hellinger distance between two continuous distributions (computed via integration).

**Parameters:**
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms
- `bins` — `np.ndarray`: Bin edges for integration

**Returns:** `float` — `√(1 − ∫√(p·q) dx)`

**Algorithm:**
1. Normalize both arrays by their integrals using `scipy.integrate.simpson`
2. Compute bin centers
3. Integrate `√(p·q)` over bin centers using `simpson`
4. Return `sqrt(1 − integral)`

**Notes:** Experimental metric; bounded in [0, 1].

---

#### `DOS_log_peak_intensity_loss(bins, DOS_target, DOS_current)`

**Purpose:** Computes loss based only on peak and valley intensities in log scale — enforces matching of spectral features.

**Parameters:**
- `bins` — `np.ndarray`: Bin edges
- `DOS_target`, `DOS_current` — `np.ndarray`: DOS histograms

**Returns:** `float` — RMS of log-scale intensity differences at peaks and valleys

**Algorithm:**
1. Compute bin centers
2. Find peaks in `DOS_target` using `scipy.signal.find_peaks`
3. Find valleys (peaks of `−DOS_target`)
4. Combine peak and valley indices
5. At those indices: compute `log(DOS_target[i]) − log(DOS_current[i])`
6. Return RMS of those differences

**Notes:** This is the second term in the combined loss (weighted 10×). By focusing only on extrema, it penalizes mismatches in feature positions and heights while ignoring flat regions. Log scale makes it invariant to overall amplitude scaling.

---

### 5.5 Bayesian Optimization Pipeline

---

#### `generate_initial_data(n=3)`

**Purpose:** Generates the initial training set for BO using Sobol quasi-random sampling.

**Parameters:**
- `n` — `int` (default `3`): Number of initial points (overridden by `N_INITIAL=20` in the main loop)

**Returns:** `(train_x, train_obj)` — `(Tensor[n,2], Tensor[n,1])`

**Algorithm:**
1. Draw `n` points from a 2D Sobol sequence in `[0,1]²`
2. Unnormalize to `[lower_bound, upper_bound]` = `[-0.5, 0.5]²`
3. Evaluate via `compute_train_obj(train_x, DOS_target, method="Zeroth")`

**Notes:** Sobol sampling provides better space coverage than uniform random for small `n`. The `method="Zeroth"` is identical to `"BO"` — the distinction may have been used in earlier experimental variants.

---

#### `compute_train_obj(train_x, DOS_target, method="BO")`

**Purpose:** Evaluates the optimization objective for a batch of parameter points — the core "black-box" function.

**Parameters:**
- `train_x` — `Tensor[N, 2]`: Batch of `(t_a, t_b)` parameter pairs
- `DOS_target` — `np.ndarray`: Target DOS histogram
- `method` — `str` (default `"BO"`): Evaluation mode (currently `"BO"` and `"Zeroth"` are equivalent)

**Returns:** `Tensor[N, 1]` — Negative combined loss (negated because BO maximizes)

**Algorithm:**
For each point `(t_a, t_b)` in `train_x`:
1. Build lattice: `lattice_model(d=0.133, t_a=t_a, t_b=t_b)`
2. Create LAPACK solver
3. Compute DOS: `DOS_Ganesh(lattice, model, solver, bins=800, range=[-0.15,0.25], sigma=5)`
4. Normalize current DOS by `integral_target`
5. Compute bin centers from `bins_target`
6. Compute `error_wass = DOS_Wasserstein_distance(bin_centers, DOS_target, DOS_current)`
7. Compute `error_peak = DOS_log_peak_intensity_loss(bin_centers, DOS_target, DOS_current)`
8. Combined loss: `error = error_wass + 10 * error_peak`
9. Append `-error` to results

**Notes:** This function is the computational bottleneck — each evaluation runs `DOS_Ganesh` which solves 10,000 eigenvalue problems. The 10× weight on `error_peak` was chosen to balance the two loss terms empirically (run named `wasserstein_only` suggests earlier variants explored single-metric loss).

---

#### `initialize_model(train_x, train_obj)`

**Purpose:** Creates and configures the Gaussian Process surrogate model.

**Parameters:**
- `train_x` — `Tensor[N, 2]`: Training inputs
- `train_obj` — `Tensor[N, 1]`: Training objectives

**Returns:** `(mll, model)` — `(ExactMarginalLogLikelihood, SingleTaskGP)`

**Algorithm:**
1. Normalize `train_x` to `[0,1]²` using `standard_bounds`
2. Initialize `SingleTaskGP` with:
   - ARD `RBFKernel` (separate lengthscale per dimension)
   - `ScaleKernel` wrapper
3. Create `ExactMarginalLogLikelihood(model.likelihood, model)` for training

**Notes:** The GP is re-initialized from scratch each BO iteration (Cell 24 calls this inside the loop). The ARD kernel allows the GP to learn different correlation lengths for `t_a` vs `t_b`.

---

#### `optimize_acqf_and_get_observation(model, train_x, train_obj, sampler, iteration, exploration_iteration=10, method="BO")`

**Purpose:** Optimizes the acquisition function to select the next batch of points to evaluate.

**Parameters:**
- `model` — `SingleTaskGP`: Fitted GP model
- `train_x`, `train_obj` — Current training data
- `sampler` — `SobolQMCNormalSampler`: QMC sampler for Monte Carlo integration
- `iteration` — `int`: Current BO iteration (1-indexed)
- `exploration_iteration` — `int` (default `10`): Transition from exploration to exploitation after this iteration
- `method` — `str`: Passed to `compute_train_obj`

**Returns:** `(new_x, new_obj)` — `(Tensor[BATCH_SIZE, 2], Tensor[BATCH_SIZE, 1])`

**Algorithm:**
1. **Best point selection:**
   - If `iteration <= exploration_iteration`: use 90th percentile of `train_obj` as reference point (encourages exploration)
   - Else: use `max(train_obj)` (switches to exploitation)
2. Build `qLogExpectedImprovement` acquisition function
3. Optimize over `standard_bounds` with `NUM_RESTARTS=10`, `RAW_SAMPLES=512`
4. Unnormalize candidates from `[0,1]²` back to `[-0.5,0.5]²`
5. Evaluate via `compute_train_obj`

**Notes:** The exploration/exploitation switch at iteration 10 is a manual schedule rather than an automatic balance. The `qLog` variant of EI is numerically more stable than standard qEI for near-zero improvement scenarios.

---

### 5.6 Local Refinement

---

#### `local_objective(x)`

**Purpose:** Wrapper that adapts `compute_train_obj` for `scipy.optimize.minimize` (which minimizes).

**Parameters:**
- `x` — `list[float, float]` or `np.ndarray`: Parameter vector `[t_a, t_b]`

**Returns:** `float` — Positive loss value (negated output of `compute_train_obj`)

**Notes:** `compute_train_obj` returns negative loss (for BO maximization); this function negates it back so `scipy.optimize.minimize` minimizes the actual loss.

---

#### `make_logger()`

**Purpose:** Factory that creates a fresh iteration log and its associated callback for one COBYLA run.

**Parameters:** None

**Returns:** `(log_list, callback_fn)` — `(list, function)`

**Notes:** Called once per top candidate to create independent logs for each of the 5 optimization runs.

---

#### `callback(xk)`

**Purpose:** Records the objective value at the current parameter vector during COBYLA iteration.

**Parameters:**
- `xk` — `np.ndarray`: Current parameter vector `[t_a, t_b]`

**Returns:** None (appends to enclosing `log_list`)

**Notes:** Created by `make_logger()` via closure. Appended to the list captured at creation time, so each run has its own independent loss history.

---

### 5.7 Visualization

---

#### `plot_2D_train_obj(train_x, new_x, train_obj, iteration, model_qEI)`

**Purpose:** Visualizes the GP posterior mean and standard deviation over the 2D parameter space, overlaid with training points.

**Parameters:**
- `train_x` — `Tensor[N, 2]`: All training points accumulated so far
- `new_x` — `Tensor[BATCH_SIZE, 2]`: Newly sampled points in this iteration
- `train_obj` — `Tensor[N, 1]`: Objective values
- `iteration` — `int`: Current iteration number (used in filename)
- `model_qEI` — `SingleTaskGP`: Fitted GP model

**Returns:** None (saves PNG and PDF)

**Algorithm:**
1. Create meshgrid over `[-0.5, 0.5]²` at 50×50 resolution
2. Query GP posterior mean and variance at all grid points
3. Plot `contourf` of posterior mean (colormap: `jet`)
4. Overlay training points as blue circles
5. Overlay new batch points as black triangles
6. Overlay top-5 solutions as red diamonds
7. Add colorbar; save as `{save_folder}/iteration_{iteration}.png/pdf`

---

## 6. Loss Function Design

The combined loss in `compute_train_obj` is:

```
Loss = DOS_Wasserstein_distance + 10 × DOS_log_peak_intensity_loss
```

| Term | Weight | Purpose |
|------|--------|---------|
| `DOS_Wasserstein_distance` | 1× | Penalizes overall spectral shape mismatch (global) |
| `DOS_log_peak_intensity_loss` | 10× | Penalizes mismatch at spectral features — peaks and valleys (local) |

**Rationale for 10× weight:** The Wasserstein distance is order-of-magnitude larger than the peak loss in typical evaluations. The weight balances both terms so neither dominates. The subfolder name `wasserstein_only` reflects that earlier runs used only the Wasserstein term; the peak loss was added to enforce feature matching.

**BO maximizes** `-Loss` (negated); all reporting of `train_obj` values is therefore negative — closer to zero is better.

---

## 7. Output Artifacts

| File | Phase | Purpose |
|------|-------|---------|
| `tp1_DOS_Target.png` / `.pdf` | Target | Target DOS spectrum plot |
| `{subfolder}/iteration_{i}.png` / `.pdf` | BO loop (×30) | GP posterior + sampling strategy per iteration |
| `{subfolder}_BO_log.txt` | BO loop | Full stdout log of all BO evaluations |
| `{subfolder}_local_optimization_log.txt` | Local refinement | COBYLA iteration logs for all 5 candidates |
| `{subfolder}_BO_local_final_result_rank_{i}.png` / `.pdf` | Final (×5) | Target vs. BO vs. local DOS comparison per rank |

All files are written to: `{figure_folder}/{subfolder}/`
