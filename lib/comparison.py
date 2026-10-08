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



def blas_context(n_threads=1):
    """Limit BLAS threads locally to 1 thread by default."""
    if threadpool_limits is None:
        return nullcontext()
    return threadpool_limits(limits=n_threads, user_api='blas')


def svd_solver_baseline(X_data, y_data, lam, factors=None):
    """Return a spectral svd baseline and scales, without building an m-by-m Hessian."""
    if lam <= 0:
        raise ValueError('Regularization must be positive.')
    if factors is None:
        with blas_context():
            factors = np.linalg.svd(X_data, full_matrices=False)
    U, singular_values, Vt = factors
    weights = singular_values / (singular_values**2 + lam**2)
    w_ref = U @ (weights * (Vt @ y_data))

    # The thin SVD omits m-n zero eigenvalues of XX.T when m > n.
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


def solution_metrics(w, X_data, y_data, lam, baseline):
    """Evaluate every solver with the same formulas, outside the timed region."""
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


def benchmark_pair(X_data, y_data, lam, baseline=None, lbfgs_options=None, repeats=REPEATS, methods=METHODS):
    """Benchmark complete calls and return a summary plus the last solutions."""
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

                # Retain only the final vector/history; release QR factors untimed.
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


def make_spectral_problem(m_size, n_size, sigma_min=1.0, sigma_max=10.0, seed=SEED):
    """Generate a tall matrix with a prescribed nonzero singular spectrum."""
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


def plot_time_iqr(ax, table, x_column):
    """Draw measured medians with asymmetric interquartile error bars."""
    for method in METHODS:
        subset = table[table['method'] == method].sort_values(x_column)
        errors = np.vstack((subset['time_ms'] - subset['q25_ms'], subset['q75_ms'] - subset['time_ms']))
        ax.errorbar(subset[x_column], subset['time_ms'], yerr=errors, color=COLORS[method], marker=MARKERS[method], capsize=3, label=method)
    ax.set_yscale('log')
    ax.set_ylabel('Complete solve time [ms], median and IQR')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend()