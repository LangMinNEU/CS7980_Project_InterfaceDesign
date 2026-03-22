"""DOS computation and spectral distance metrics extracted from the source notebook."""

import numpy as np
import pybinding as pb
import pointpats
from shapely.geometry import Polygon
from scipy.stats import wasserstein_distance, gaussian_kde
from scipy.signal import find_peaks
from scipy.integrate import simpson, trapezoid
from scipy.ndimage import gaussian_filter1d


def DOS_Ganesh(
    lattice,
    model,
    solver,
    bins=1000,
    range=[-6, 6],
    density=True,
    size=10000,
    sigma=12.0,
):
    """Compute the DOS by uniformly sampling the Brillouin zone and histogramming eigenvalues.

    Args:
        lattice: PyBinding Lattice object.
        model: PyBinding Model object.
        solver: PyBinding solver (e.g. pb.solver.lapack).
        bins: Number of energy histogram bins.
        range: [min, max] energy range in eV.
        density: Normalize histogram to probability density.
        size: Number of k-points to sample.
        sigma: Gaussian filter standard deviation (in bin units).

    Returns:
        (DOS_counts, bin_edges) as numpy arrays.
    """
    np.set_printoptions(threshold=np.inf)
    coords = lattice.brillouin_zone()
    pgon = Polygon(coords)
    kbzs = pointpats.random.normal(pgon, size=size, center=[0, 0])
    eigens = []
    for k in kbzs:
        solver.set_wave_vector(k)
        eigens.append(solver.eigenvalues.flatten())
    eigens = np.array(eigens).flatten("C")
    count, bin_edges = np.histogram(eigens, bins=bins, range=range, density=density)
    count = np.nan_to_num(count, nan=0)
    count = gaussian_filter1d(count, sigma=sigma)
    return count, bin_edges


# ---------------------------------------------------------------------------
# Distance / loss metrics
# ---------------------------------------------------------------------------

def DOS_squared_error(DOS_target, DOS_current):
    """Sum of squared differences (L2 loss)."""
    return np.sum((DOS_target - DOS_current) ** 2)


def DOS_Wasserstein_distance(bins, DOS_target, DOS_current):
    """Wasserstein-1 (Earth Mover's) distance between two DOS distributions.

    Args:
        bins: Bin centers array.
        DOS_target, DOS_current: DOS histograms of equal length.

    Returns:
        float — Wasserstein distance.
    """
    bins = np.asarray(bins, dtype=float)
    p = np.asarray(DOS_target, dtype=float)
    q = np.asarray(DOS_current, dtype=float)
    if q.sum() == 0:
        return float(bins.max() - bins.min())
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    return wasserstein_distance(
        u_values=bins,
        v_values=bins,
        u_weights=p,
        v_weights=q,
    )


def DOS_log_peak_intensity_loss(bins, DOS_target, DOS_current):
    """RMS of log-scale intensity differences at spectral peaks and valleys.

    Args:
        bins: Bin centers array.
        DOS_target, DOS_current: DOS histograms of equal length.

    Returns:
        float — Peak/valley log-intensity RMS distance.
    """
    bins = np.asarray(bins, dtype=float)
    p = np.asarray(DOS_target, dtype=float)
    q = np.asarray(DOS_current, dtype=float)
    if q.sum() == 0:
        return float(bins.max() - bins.min())
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    peaks_target, _ = find_peaks(p)
    valley_target, _ = find_peaks(-p)
    log_p = np.log(np.concatenate([p[peaks_target], p[valley_target]]) + 1)
    log_q = np.log(np.concatenate([q[peaks_target], q[valley_target]]) + 1)
    peak_loss = np.sqrt(np.sum((log_p - log_q) ** 2))
    return peak_loss


def DOS_KDE_Wasserstein_distance(bins, DOS_target, DOS_current, bw_method=None, grid_points=None):
    """Wasserstein distance computed via KDE smoothing."""
    bins = np.asarray(bins, dtype=float)
    p = np.asarray(DOS_target, dtype=float)
    q = np.asarray(DOS_current, dtype=float)
    if grid_points is None:
        grid_points = bins.shape[0]
    if q.sum() == 0:
        return float(bins.max() - bins.min())
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    method = bw_method if bw_method is not None else "scott"
    kde_target = gaussian_kde(dataset=bins, weights=p, bw_method=method)
    kde_current = gaussian_kde(dataset=bins, weights=q, bw_method=method)
    xmin, xmax = bins.min(), bins.max()
    grid = np.linspace(xmin, xmax, grid_points)
    density_t = kde_target(grid)
    density_c = kde_current(grid)
    area_t = np.trapezoid(density_t, x=grid)
    area_c = np.trapezoid(density_c, x=grid)
    if area_t > 0:
        density_t = density_t / area_t
    if area_c > 0:
        density_c = density_c / area_c
    return wasserstein_distance(
        u_values=grid, v_values=grid, u_weights=density_t, v_weights=density_c
    )


def DOS_MMD_Gaussian(bins, DOS_target, DOS_current, lengthscale=0.05):
    """Maximum Mean Discrepancy with Gaussian (RBF) kernel."""
    bins = np.asarray(bins, dtype=float)
    p = np.asarray(DOS_target, dtype=float)
    q = np.asarray(DOS_current, dtype=float)
    if q.sum() == 0:
        return float(bins.max() - bins.min())
    if p.sum() > 0:
        p = p / p.sum()
    if q.sum() > 0:
        q = q / q.sum()
    x = bins.reshape(-1, 1)
    diffs = x - x.T
    K = np.exp(-(diffs ** 2) / (2 * lengthscale ** 2))
    return float(p.dot(K.dot(p)) - 2.0 * p.dot(K.dot(q)) + q.dot(K.dot(q)))


def DOS_Bhattacharya_distance(DOS_target, DOS_current):
    """Bhattacharyya distance between two distributions."""
    return -np.log(np.sum(np.sqrt(DOS_target * DOS_current)))


def DOS_KLDivergence_distance(DOS_target, DOS_current):
    """Kullback-Leibler divergence from DOS_current to DOS_target."""
    return np.sum(DOS_target * np.log(DOS_current / DOS_target))


def DOS_Helinger_distance(DOS_target, DOS_current, bins):
    """Hellinger distance between two DOS distributions (bounded [0, 1])."""
    DOS_target = DOS_target / np.trapezoid(DOS_target, bins[:-1])
    if np.sum(DOS_current) == 0:
        return 1.0
    DOS_current = DOS_current / np.trapezoid(DOS_current, bins[:-1])
    return float(np.sqrt(1 - np.trapezoid(np.sqrt(DOS_target * DOS_current), bins[:-1])))
