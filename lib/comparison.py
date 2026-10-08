"""
================================================================================
comparison.py — Shared experiments for L-BFGS and structure-based QR
================================================================================

Problem:
    min_w f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2

Matrices:
    X in R^(m x n), y in R^n, w in R^m, lambda > 0
    A = [X^T; lambda*I_m], H = A^T A = XX^T + lambda^2 I_m

The module provides a numerical SVD baseline, common accuracy metrics,
complete-call timing, synthetic problems with a prescribed spectrum, and
runtime plots. Configuration constants are imported by 04_comparison.ipynb.
The SVD baseline is a floating-point reference, not an exact arithmetic oracle.
================================================================================
"""

from contextlib import nullcontext
import numpy as np
import pandas as pd
import time
try:
    from threadpoolctl import threadpool_info, threadpool_limits
except ImportError:
    threadpool_info = None
    threadpool_limits = None

from lib.utils import compute_loss, compute_gradient
from lib.lbfgs import lbfgs_optimize
from lib.qr_householder import qr_solver_structure_based


# =============================================================================
# 1. SHARED EXPERIMENT SETTINGS
# =============================================================================

SEED = 42
TINY = np.finfo(float).tiny
REPEATS = 10
LAMBDA = 0.5
TARGET_W_ERROR = 1e-6
TARGET_GRADIENT = 1e-8
LBFGS_OPTIONS = dict(
    m_history=10,
    max_iter=2000,
    tol=1e-10,
    tol_type='relative',
    line_search='exact',
    h0_scaling='bb1',
    use_restart=True,
    verbose=False,
)

TOLERANCES = np.array([1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14])
LAMBDAS = np.unique(np.r_[np.logspace(-8, 2, 11), LAMBDA])
M_SIZES = np.array([100, 250, 500, 1000, 2000, 4000])
N_SIZES = np.array([4, 8, 12, 24, 48, 96])

METHODS = ('L-BFGS', 'Structure-based QR')
COLORS = {'L-BFGS': 'steelblue', 'Structure-based QR': 'darkorange'}
MARKERS = {'L-BFGS': 'o', 'Structure-based QR': 's'}
PLOT_FLOOR = 1e-18


# =============================================================================
# 2. BLAS THREAD CONTROL
# =============================================================================

def blas_context(n_threads=1):
    """Create a context that temporarily limits the number of BLAS threads.

    Parameters
    ----------
    n_threads : int, default=1
        Requested thread limit for supported BLAS libraries.

    Returns
    -------
    context : context manager
        A ``threadpool_limits`` context when threadpoolctl is available,
        otherwise a ``nullcontext`` that leaves thread settings unchanged.

    Notes
    -----
    Use ``with blas_context():`` around numerical work. Leaving the context
    restores the previous thread limits. Controlling BLAS threads reduces
    one source of timing variability; it does not make wall times identical
    across runs or control every source of parallelism.
    """
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=n_threads, user_api='blas')


# =============================================================================
# 3. SVD BASELINE AND CONDITIONING
# =============================================================================


def svd_solver_baseline(X_data, y_data, lam, factors=None):
    """Compute the regularized SVD solution and scales used by the metrics.

    For the thin SVD X = U diag(sigma) V^T, the solution is
        w_ref = U diag(sigma / (sigma^2 + lam^2)) V^T y.
    Neither the augmented matrix A nor the m-by-m Hessian H is formed.

    Parameters
    ----------
    X_data : ndarray, shape (m, n)
        Finite real data matrix with nonzero dimensions.
    y_data : ndarray, shape (n,)
        Right-hand side of the data equations.
    lam : float
        Positive regularization parameter; the penalty coefficient is lam^2.
    factors : tuple of ndarray or None, default=None
        Thin SVD factors ``(U, singular_values, Vt)`` for the same X_data,
        with shapes (m, r), (r,), and (r, n), where r = min(m, n).
        If None, compute them with ``np.linalg.svd(full_matrices=False)``
        inside ``blas_context``. Reuse them when only lam changes.

    Returns
    -------
    baseline : dict
        ``w``: reference solution, shape (m,).
        ``f``: regularized objective value f(w).
        ``norm_w``, ``norm_y``, ``norm_Xy``: Euclidean norms of w, y, and Xy.
        ``norm_H``: spectral norm of H = XX^T + lam^2 I_m.
        ``kappa_A``: 2-norm condition number of A = [X^T; lam I_m].
        ``kappa_H``: 2-norm condition number of H, equal to kappa_A^2.

    Raises
    ------
    ValueError
        If lam is not positive.

    Notes
    -----
    When m > n, the thin SVD omits zero eigenvalues of XX^T, so the
    smallest singular value of A is lam. For m <= n, the smallest
    returned singular value of X also enters the denominator.
    The baseline is computed in floating-point arithmetic. Input shapes,
    finiteness, and consistency of supplied factors are assumed, not checked.
    """
    if lam <= 0:
        raise ValueError('Regularization must be positive.')
    if factors is None:
        with blas_context():
            factors = np.linalg.svd(X_data, full_matrices=False)
    U, singular_values, Vt = factors
    weights = singular_values / (singular_values**2 + lam**2)
    w_ref = U @ (weights * (Vt @ y_data))

    # The thin SVD omits m-n zero eigenvalues of XX.T when m > n
    sigma_min = 0.0 if X_data.shape[0] > X_data.shape[1] else singular_values[-1]
    norm_A = np.hypot(singular_values[0], lam)
    kappa_A = norm_A / np.hypot(sigma_min, lam)
    return {
        'w': w_ref,
        'f': compute_loss(w_ref, X_data, y_data, lam),
        'norm_w': np.linalg.norm(w_ref),
        'norm_y': np.linalg.norm(y_data),
        'norm_Xy': np.linalg.norm(X_data @ y_data),
        'norm_H': norm_A**2,
        'kappa_A': kappa_A,
        'kappa_H': kappa_A**2,
    }


# =============================================================================
# 4. COMMON ACCURACY METRICS
# =============================================================================


def solution_metrics(w, X_data, y_data, lam, baseline):
    """Evaluate a solution against the SVD baseline for the same problem.

    Parameters
    ----------
    w : ndarray, shape (m,)
        Computed solution to evaluate.
    X_data : ndarray, shape (m, n)
        Real data matrix used by the solver.
    y_data : ndarray, shape (n,)
        Right-hand side used by the solver.
    lam : float
        Positive regularization parameter used by the solver.
    baseline : dict
        Output of ``svd_solver_baseline`` for the same X_data, y_data, and lam.

    Returns
    -------
    metrics : dict
        With e = w - w_ref, r = X^T w - y, and g = Xr + lam^2 w:
        ``objective``: f(w) = (||r||^2 + lam^2 ||w||^2) / 2.
        ``objective_gap_signed``: (f(w) - f_ref) / f_ref.
        ``energy_discrepancy``: (||X^T e||^2 + lam^2 ||e||^2) / (2 f_ref).
        ``w_error``: ||e|| / ||w_ref||.
        ``prediction_error``: ||X^T e|| / ||y||.
        ``data_residual``: ||r|| / ||y||.
        ``augmented_residual``: sqrt(||r||^2 + lam^2 ||w||^2) / ||y||.
        ``grad_relative``: ||g|| / ||Xy||.
        ``stationarity``: ||g|| / (||H||_2 ||w|| + ||Xy||).
        Vector norms are Euclidean; denominators are bounded below by TINY.

    Notes
    -----
    Metrics are computed after the timed solve. The signed objective gap
    can be slightly negative through rounding. The energy discrepancy
    avoids subtracting nearly equal objectives and equals the relative
    objective gap only for an exact reference minimizer.
    ``grad_relative`` uses the gradient norm at zero; it differs from the
    solver's stopping ratio when its initial point is nonzero.
    """
    error = w - baseline['w']
    prediction_difference = X_data.T @ error
    residual = X_data.T @ w - y_data
    norm_w = np.linalg.norm(w)
    gradient_norm = np.linalg.norm(compute_gradient(w, X_data, y_data, lam))
    objective = compute_loss(w, X_data, y_data, lam)
    energy = 0.5 * (np.dot(prediction_difference, prediction_difference) + lam**2 * np.dot(error, error))
    return {
        'objective': objective,
        'objective_gap_signed': (objective - baseline['f']) / max(baseline['f'], TINY),
        'energy_discrepancy': energy / max(baseline['f'], TINY),
        'w_error': np.linalg.norm(error) / max(baseline['norm_w'], TINY),
        'prediction_error': np.linalg.norm(prediction_difference) / max(baseline['norm_y'], TINY),
        'data_residual': np.linalg.norm(residual) / max(baseline['norm_y'], TINY),
        'augmented_residual': np.hypot(np.linalg.norm(residual), lam * norm_w) / max(baseline['norm_y'], TINY),
        'grad_relative': gradient_norm / max(baseline['norm_Xy'], TINY),
        'stationarity': gradient_norm / max(baseline['norm_H'] * norm_w + baseline['norm_Xy'], TINY),
    }


# =============================================================================
# 5. COMPLETE-CALL BENCHMARK
# =============================================================================


def benchmark_pair(X_data, y_data, lam, baseline=None, lbfgs_options=None, repeats=REPEATS, methods=METHODS):
    """Compare selected solvers using repeated complete-call wall times.

    Parameters
    ----------
    X_data : ndarray, shape (m, n)
        Finite real floating-point data matrix shared by the solvers.
    y_data : ndarray, shape (n,)
        Shared right-hand side of the data equations.
    lam : float
        Shared positive regularization parameter.
    baseline : dict or None, default=None
        SVD baseline for this problem. If None, construct it before timing.
    lbfgs_options : dict or None, default=None
        Keyword overrides for LBFGS_OPTIONS, passed to ``lbfgs_optimize``.
        The stopping rule must remain ``tol_type='relative'``.
    repeats : int, default=REPEATS
        Positive number of measured calls per method, excluding warm-up.
        The shared default is 10; at least three repetitions are recommended.
    methods : sequence of str, default=METHODS
        Nonempty selection of ``'L-BFGS'`` and ``'Structure-based QR'``.

    Returns
    -------
    table : pandas.DataFrame
        One row per method, including dimensions, lam, kappa_A, and all
        ``solution_metrics`` fields from the last measured solve.
        ``time_ms`` is the median complete-call time; ``q25_ms`` and
        ``q75_ms`` are its interquartile limits, all in milliseconds.
        L-BFGS rows also contain iterations, restarts, requested_tol,
        and stop_ratio = ||g_final|| / max(||g_initial||, TINY).
        These iterative fields are NaN for QR. Status is ``'converged'``,
        ``'max_iter'``, or ``'stopped_before_tol'`` for finite L-BFGS results,
        ``'direct'`` for finite QR results, and ``'nonfinite'`` if the
        returned vector or any accuracy metric is nonfinite.
    solutions : dict
        Method names mapped to dictionaries containing the last solution
        ``w``, the last L-BFGS ``history`` (None for QR), and the measured
        ``times_seconds`` array of length repeats. QR factors are discarded.

    Raises
    ------
    ValueError
        If the configured L-BFGS stopping rule is not relative.

    Notes
    -----
    Each method receives one untimed warm-up. The measured execution order
    is shuffled reproducibly with SEED, inside ``blas_context``. An external
    timer covers initialization and the complete solve, including any
    histories produced by the solver. Baseline construction, post-solve
    metrics, and result disposal are excluded. The IQR describes timing
    dispersion, not a confidence interval. Status is inferred from the
    returned gradient and iteration count, not supplied by the optimizer.
    Input compatibility, valid method names, and repeats > 0 are assumed.
    """
    options = LBFGS_OPTIONS.copy()
    if lbfgs_options is not None:
        options.update(lbfgs_options)
    if options['tol_type'] != 'relative':
        raise ValueError('This comparison uses the relative gradient stopping rule.')
    if baseline is None:
        baseline = svd_solver_baseline(X_data, y_data, lam)

    solvers = {
        'L-BFGS': lambda: lbfgs_optimize(X_data, y_data, lam, **options),
        'Structure-based QR': lambda: qr_solver_structure_based(X_data, y_data, lam),
    }
    timings = {method: [] for method in methods}
    solutions = {}
    rng = np.random.default_rng(SEED)

    with blas_context():
        for method in methods:
            solvers[method]()  # Warm-up output is discarded before measurement

        for _ in range(repeats):
            for method in rng.permutation(methods):
                start = time.perf_counter()
                result = solvers[method]()
                elapsed = time.perf_counter() - start
                timings[method].append(elapsed)

                # Retain only the final vector/history; release QR factors untimed
                if method == 'L-BFGS':
                    solutions[method] = {'w': result[0], 'history': result[1]}
                else:
                    solutions[method] = {'w': result[1], 'history': None}
                del result

        rows = []
        for method in methods:
            solution = solutions[method]
            w_result = solution['w']
            metrics = solution_metrics(w_result, X_data, y_data, lam, baseline)
            history = solution['history']
            iterations = len(history['alpha']) if history is not None else np.nan
            status = 'direct'
            stop_ratio = np.nan

            if not np.isfinite(w_result).all() or not np.isfinite(list(metrics.values())).all():
                status = 'nonfinite'
            elif history is not None:
                # This scale also handles experiments with a nonzero initial point
                final_gradient = np.linalg.norm(compute_gradient(w_result, X_data, y_data, lam))
                stop_ratio = final_gradient / max(history['grad_norm'][0], TINY)
                if stop_ratio < options['tol']:
                    status = 'converged'
                elif iterations >= options['max_iter']:
                    status = 'max_iter'
                else:
                    status = 'stopped_before_tol'

            q25, median, q75 = np.quantile(timings[method], [0.25, 0.5, 0.75])
            rows.append({
                'method': method,
                'm': X_data.shape[0],
                'n': X_data.shape[1],
                'lambda': lam,
                'kappa_A': baseline['kappa_A'],
                'time_ms': median * 1e3,
                'q25_ms': q25 * 1e3,
                'q75_ms': q75 * 1e3,
                'iterations': iterations,
                'status': status,
                'stop_ratio': stop_ratio,
                'requested_tol': options['tol'] if history is not None else np.nan,
                'restarts': len(history['restarts']) if history is not None else np.nan,
                **metrics,
            })
            solution['times_seconds'] = np.asarray(timings[method])

    return pd.DataFrame(rows), solutions


# =============================================================================
# 6. SYNTHETIC PROBLEMS WITH A PRESCRIBED SPECTRUM
# =============================================================================


def make_spectral_problem(m_size, n_size, sigma_min=1.0, sigma_max=10.0, seed=SEED):
    """Generate a tall matrix with prescribed singular values and a random target.

    Parameters
    ----------
    m_size : int
        Number of rows of X, equal to the number of unknown coefficients.
    n_size : int
        Number of columns of X; must satisfy 0 < n_size < m_size.
    sigma_min : float, default=1.0
        Smallest prescribed singular value when n_size > 1.
    sigma_max : float, default=10.0
        Largest prescribed singular value. For n_size = 1, this is the
        only singular value.
    seed : int or None, default=SEED
        Seed for the local NumPy generator. The shared default is 42.

    Returns
    -------
    X_synthetic : ndarray, shape (m_size, n_size)
        Matrix U diag(sigma) V^T, with orthonormal columns in U, orthogonal
        V, and singular values geometrically spaced between the endpoints.
    y_synthetic : ndarray, shape (n_size,)
        Standard normal right-hand side drawn from the same generator.

    Raises
    ------
    ValueError
        If the dimensions do not satisfy 0 < n_size < m_size.

    Notes
    -----
    QR factorizations of Gaussian matrices produce the orthonormal factors.
    Positive endpoints with sigma_min <= sigma_max are assumed. Construction
    is reproducible for a fixed seed and is performed outside solver timings.
    """
    if not 0 < n_size < m_size:
        raise ValueError('The synthetic experiment requires 0 < n < m.')
    rng = np.random.default_rng(seed)
    with blas_context():
        U, _ = np.linalg.qr(rng.standard_normal((m_size, n_size)), mode='reduced')
        V, _ = np.linalg.qr(rng.standard_normal((n_size, n_size)))
        spectrum = np.geomspace(sigma_max, sigma_min, n_size)
        X_synthetic = (U * spectrum) @ V.T
    y_synthetic = rng.standard_normal(n_size)
    return X_synthetic, y_synthetic


# =============================================================================
# 7. RUNTIME PLOTS
# =============================================================================


def plot_time_iqr(ax, table, x_column):
    """Plot median solve times and interquartile ranges for both methods.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the runtime curves.
    table : pandas.DataFrame
        Benchmark results containing method, time_ms, q25_ms, q75_ms, and
        x_column. Each method's rows are sorted by x_column before plotting.
    x_column : str
        Column containing the horizontal coordinate, such as m, n, or lambda.

    Returns
    -------
    None
        Modify ax in place, adding curves, a logarithmic time axis, a label,
        a grid, and a legend. The caller sets the x-axis scale and label.

    Notes
    -----
    Error bars extend from the 25th to the 75th percentile and can be
    asymmetric about the median. Times are in milliseconds. The table is
    expected to contain positive times with q25_ms <= time_ms <= q75_ms.
    The figure is neither created, displayed, nor saved by this function.
    """
    for method in METHODS:
        subset = table[table['method'] == method].sort_values(x_column)
        errors = np.vstack((subset['time_ms'] - subset['q25_ms'], subset['q75_ms'] - subset['time_ms']))
        ax.errorbar(subset[x_column], subset['time_ms'], yerr=errors, color=COLORS[method], marker=MARKERS[method], capsize=3, label=method)
    ax.set_yscale('log')
    ax.set_ylabel('Complete solve time [ms], median and IQR')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()
