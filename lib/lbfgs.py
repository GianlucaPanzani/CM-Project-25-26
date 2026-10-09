""" Limited-Memory BFGS (L-BFGS)

L-BFGS algorithm for the problem:
    min_w  f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2

References:
    - Nocedal & Wright, "Numerical Optimization"
      - Algorithm 9.1  (Two-Loop Recursion)
      - Algorithm 9.2  (L-BFGS)
      - Eq. 9.6        (Initial scaling gamma_k)
      - Eq. 3.25       (Exact line search on quadratics)
      - Eq. 3.7        (Strong Wolfe conditions)
    - Liu & Nocedal, "On the limited memory BFGS method for large scale
      optimization", Mathematical Programming 45, 1989
      - Section 4      (Scaling of H_k^0, Eq. 4.1 = BB2)

Additional references:
    - Barzilai & Borwein, "Two-point step size gradient methods",
      IMA Journal of Numerical Analysis 8(1), 1988
      - Eq. (6) BB1 scaling:  gamma = s^T s / s^T y
      - Eq. (5) BB2 scaling:  gamma = s^T y / y^T y  (= Eq. 9.6 Nocedal)

Implemented components:
    1. Two-loop recursion          (Nocedal & Wright, Algorithm 9.1)
    2. Exact line search           (Nocedal & Wright, Eq. 3.25)
    3. Strong Wolfe line search    (Nocedal & Wright, Eq. 3.7)
    4. L-BFGS main loop            (Nocedal & Wright, Algorithm 9.2)
    5. Curvature-based restart     (heuristic safeguard)
    6. Adaptive H0 scaling         (Barzilai & Borwein 1988)
    7. Flop counter                (derived from Algorithm 9.1 structure;
                                    supports both 'exact' and 'wolfe' LS)
    8. Robust benchmarking wrapper (median over N runs with warm-up;
                                    used to obtain reliable wall-clock
                                    measurements on sub-millisecond runs)
"""

import numpy as np
import time
from lib.utils import compute_loss, compute_gradient


# =============================================================================
# 1. TWO-LOOP RECURSION  (Nocedal & Wright, Algorithm 9.1)
# =============================================================================

def lbfgs_two_loop(grad_k, s_list, y_list, rho_list, gamma_k):
    """
    Calculates the product H_k * grad_k via the two-loop recursion.

    The matrix H_k (approximation of the inverse Hessian) is NEVER
    formed explicitly. Only the m most recent {s_i, y_i} pairs and
    the scaling H_0 = gamma_k * I are used.

    Cost: (8 * m_history + 1) * dim flops (one flop = one add or multiply).

    Parameters
    ---------
    grad_k   : ndarray (dim,)       current gradient
    s_list   : list of ndarray       s_i = w_{i+1} - w_i
    y_list   : list of ndarray       y_i = grad_{i+1} - grad_i
    rho_list : list of float         rho_i = 1 / (y_i^T s_i)
    gamma_k  : float                 scaling for H_0 = gamma_k * I

    Returns
    -------
    r : ndarray (dim,)   product H_k * grad_k
    """
    mem = len(s_list)
    q = grad_k.copy()
    alpha = np.zeros(mem)

    # Backward loop: i = k-1, k-2, ..., k-m
    for i in range(mem - 1, -1, -1):
        alpha[i] = rho_list[i] * np.dot(s_list[i], q)
        q -= alpha[i] * y_list[i]

    # Multiplication by H_0 = gamma_k * I
    r = gamma_k * q

    # Forward loop: i = k-m, k-m+1, ..., k-1
    for i in range(mem):
        beta = rho_list[i] * np.dot(y_list[i], r)
        r += s_list[i] * (alpha[i] - beta)

    return r


# =============================================================================
# 2. EXACT LINE SEARCH FOR QUADRATIC PROBLEMS
# =============================================================================
#
# For f(w) = (1/2) w^T H w - b^T w with H = XX^T + lambda^2 I,
# the 1D slice phi(alpha) = f(w + alpha*p) is a parabola
# whose minimum is (Nocedal & Wright, generalized Eq. 3.25):
#
#     alpha_min = - grad^T p / (p^T H p)
#
# where p^T H p = ||X^T p||^2 + lambda^2 ||p||^2.
# Cost: a single X^T p product, which is O(mn).
# =============================================================================

def exact_line_search(grad, p, X, lam):
    """
    Optimal step length for the quadratic problem.

    Parameters
    ---------
    grad : ndarray (m,)   gradient nabla f(w)
    p    : ndarray (m,)   descent direction
    X    : ndarray (m, n) data matrix
    lam  : float          regularization parameter

    Returns
    -------
    alpha : float   optimal step length
    """
    Xtp = X.T @ p                                          # (n,)
    pHp = np.dot(Xtp, Xtp) + (lam ** 2) * np.dot(p, p)   # p^T H p
    dg = np.dot(grad, p)                                   # grad^T p

    if pHp < 1e-30:
        return 0.0
    return -dg / pHp


# =============================================================================
# 3. STRONG WOLFE LINE SEARCH  (Nocedal & Wright, Eq. 3.7)
# =============================================================================
#
# Strong Wolfe conditions:
#   (i)   f(w + alpha*p) <= f(w) + c1*alpha*grad^T p      (Armijo)
#   (ii)  |grad_new^T p| <= c2 * |grad^T p|               (curvature)
#
# For quasi-Newton: c1 = 1e-4, c2 = 0.9 (Nocedal, Section 3.1).
# =============================================================================

def _cubic_interpolation(a_lo, a_hi, f_lo, f_hi, g_lo, g_hi):
    """Cubic interpolation between two points to find the minimum."""
    d1 = g_lo + g_hi - 3.0 * (f_hi - f_lo) / (a_hi - a_lo)
    disc = d1 * d1 - g_lo * g_hi
    if disc < 0:
        return 0.5 * (a_lo + a_hi)
    d2 = np.sqrt(disc)
    if a_hi - a_lo > 0:
        a_min = a_hi - (a_hi - a_lo) * (g_hi + d2 - d1) / (g_hi - g_lo + 2.0 * d2)
    else:
        a_min = a_lo - (a_lo - a_hi) * (g_lo + d2 - d1) / (g_lo - g_hi + 2.0 * d2)
    lo, hi = min(a_lo, a_hi), max(a_lo, a_hi)
    return np.clip(a_min, lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))


def _zoom(f_func, grad_func, w, p, f_0, dg_0,
          a_lo, a_hi, f_lo, f_hi, dg_lo,
          X, y, lam, c1, c2, max_iter=10):
    """Zoom procedure for the line search (Nocedal, Algorithm 3.3)."""
    evals = 0
    for _ in range(max_iter):
        alpha_j = _cubic_interpolation(
            a_lo, a_hi, f_lo, f_hi, dg_lo,
            (f_hi - f_lo) / (a_hi - a_lo + 1e-30)
            if abs(a_hi - a_lo) > 1e-15 else 0.0
        )
        w_trial = w + alpha_j * p
        f_j = f_func(w_trial, X, y, lam)
        evals += 1

        if f_j > f_0 + c1 * alpha_j * dg_0 or f_j >= f_lo:
            a_hi, f_hi = alpha_j, f_j
        else:
            g_j = grad_func(w_trial, X, y, lam)
            dg_j = np.dot(g_j, p)
            evals += 1
            if abs(dg_j) <= c2 * abs(dg_0):
                return alpha_j, f_j, g_j, evals
            if dg_j * (a_hi - a_lo) >= 0:
                a_hi, f_hi = a_lo, f_lo
            a_lo, f_lo, dg_lo = alpha_j, f_j, dg_j

    w_trial = w + a_lo * p
    return a_lo, f_func(w_trial, X, y, lam), grad_func(w_trial, X, y, lam), evals + 2


def strong_wolfe_line_search(w, p, f_0, grad_0, dg_0,
                             X, y, lam,
                             c1=1e-4, c2=0.9, alpha_init=1.0, max_ls=25):
    """
    Line search with strong Wolfe conditions.

    Parameters
    ---------
    w       : ndarray (m,)   current point
    p       : ndarray (m,)   descent direction
    f_0     : float          f(w)
    grad_0  : ndarray (m,)   nabla f(w)
    dg_0    : float          grad_0^T p  (< 0 if p is a descent direction)
    X, y, lam                problem parameters
    c1, c2  : float          Wolfe constants
    alpha_init : float       initial step length (1.0 for quasi-Newton)
    max_ls  : int            maximum number of attempts

    Returns
    -------
    alpha, f_new, grad_new, ls_evals
    """
    alpha = alpha_init
    alpha_prev = 0.0
    f_prev = f_0
    dg_prev = dg_0
    ls_evals = 0

    for i in range(max_ls):
        w_trial = w + alpha * p
        f_trial = compute_loss(w_trial, X, y, lam)
        ls_evals += 1

        if f_trial > f_0 + c1 * alpha * dg_0 or (i > 0 and f_trial >= f_prev):
            alpha, f_new, g_new, ze = _zoom(
                compute_loss, compute_gradient, w, p, f_0, dg_0,
                alpha_prev, alpha, f_prev, f_trial, dg_prev,
                X, y, lam, c1, c2)
            return alpha, f_new, g_new, ls_evals + ze

        g_trial = compute_gradient(w_trial, X, y, lam)
        dg_trial = np.dot(g_trial, p)
        ls_evals += 1

        if abs(dg_trial) <= c2 * abs(dg_0):
            return alpha, f_trial, g_trial, ls_evals

        if dg_trial >= 0:
            alpha, f_new, g_new, ze = _zoom(
                compute_loss, compute_gradient, w, p, f_0, dg_0,
                alpha, alpha_prev, f_trial, f_prev, dg_trial,
                X, y, lam, c1, c2)
            return alpha, f_new, g_new, ls_evals + ze

        alpha_prev, f_prev, dg_prev = alpha, f_trial, dg_trial
        alpha = min(2.0 * alpha, 1e10)

    g_trial = compute_gradient(w + alpha * p, X, y, lam)
    return alpha, f_trial, g_trial, ls_evals

# =============================================================================
# 5. CURVATURE-BASED RESTART
# =============================================================================
#
# Heuristic safeguard (not taken from a specific reference).
#
# The standard L-BFGS only skips a pair when y_k^T s_k <= 0 (negative
# curvature). We add a stronger, heuristic criterion: reset the
# entire memory when the curvature ratio of the new pair drops sharply
# relative to the same ratio of the previously stored pair, i.e. when:
#
#     y_k^T s_k / ||y_k||^2  <  xi * gamma_{k-1}
#
# where gamma_{k-1} = s_{k-1}^T y_{k-1} / ||y_{k-1}||^2 is the same ratio for
# the previously stored pair (always the BB2 ratio, independently of the H0
# strategy) and xi in (0,1) is a user-chosen threshold (default 0.2).
#
# Intuition: gamma_k = s^T y / y^T y is an estimate of the inverse curvature
# along s_k. If it drops dramatically relative to the previous estimate, the
# memory contains stale / contradictory curvature information and should be
# discarded.
# =============================================================================

def _should_restart(s_k, y_k, ys, gamma_prev, xi=0.2):
    """
    Curvature-based restart criterion.

    Returns True if the memory should be reset before storing (s_k, y_k).

    Parameters
    ----------
    s_k        : ndarray   step vector
    y_k        : ndarray   gradient difference
    ys         : float     y_k^T s_k  (pre-computed)
    gamma_prev : float     previous H0 scaling (gamma_{k-1})
    xi         : float     restart threshold in (0, 1); default 0.2

    Returns
    -------
    bool
    """
    yy = np.dot(y_k, y_k)
    if yy < 1e-30:
        return False
    gamma_new = ys / yy          # = s^T y / y^T y  (BB2 / Nocedal Eq. 9.6)
    return gamma_new < xi * gamma_prev


# =============================================================================
# 6. ADAPTIVE H0 SCALING
# =============================================================================
#
# Three strategies for choosing gamma_k (the scalar that defines H_0^k = gamma_k * I):
#
#  'nocedal':
#      gamma_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}           [Eq. 9.6]
#      This is also known as the BB2 step (Barzilai & Borwein 1988, Eq. (5)).
#      It estimates the inverse curvature of the Hessian along s_{k-1}.
#
#  'bb1' (default, Barzilai & Borwein 1988, Eq. (6)):
#      gamma_k = s_{k-1}^T s_{k-1} / s_{k-1}^T y_{k-1}
#      Interpretation: gamma^{-1} minimizes || gamma^{-1} s - y ||, i.e. it is
#      the least-squares fit of the secant equation H s = y with H = gamma^{-1} I.
#      (BB2 instead minimizes || gamma y - s ||.)
#      By Cauchy-Schwarz, BB1 >= BB2: it produces larger initial steps.
#      Often produces larger steps and can accelerate convergence on
#      ill-conditioned problems.
#
#  'safeguarded':
#      gamma_k = clip( BB2_k, gamma_min, gamma_max )
#      Prevents gamma from becoming extremely large or small due to
#      numerical noise, which can destabilize the two-loop recursion.
#      Default bounds: gamma_min=1e-10, gamma_max=1e10.
# =============================================================================

def _compute_gamma(s_list, y_list, scaling='nocedal',
                   gamma_min=1e-10, gamma_max=1e10):
    """
    Compute the H0 scaling factor gamma_k.

    Parameters
    ----------
    s_list  : list of ndarray   stored step vectors
    y_list  : list of ndarray   stored gradient differences
    scaling : str               'nocedal' | 'bb1' | 'safeguarded'
    gamma_min, gamma_max : float  bounds for 'safeguarded' mode

    Returns
    -------
    gamma_k : float
    """
    if len(s_list) == 0:
        return 1.0   # fallback at first iteration

    s = s_list[-1]
    y = y_list[-1]
    sy = np.dot(s, y)   # s^T y
    yy = np.dot(y, y)   # y^T y
    ss = np.dot(s, s)   # s^T s

    if sy <= 1e-30 or yy <= 1e-30:
        return 1.0 # not reached from lbfgs_optimize (empty memory uses 1/||g||)

    if scaling == 'nocedal':
        # Nocedal & Wright Eq. 9.6  (= BB2)
        return sy / yy

    elif scaling == 'bb1':
        # Barzilai & Borwein (1988), Eq. 6
        return ss / sy

    elif scaling == 'safeguarded':
        # BB2 clipped to [gamma_min, gamma_max]
        gamma = sy / yy
        return float(np.clip(gamma, gamma_min, gamma_max))

    else:
        raise ValueError(f"Unknown scaling='{scaling}'. "
                         "Choose 'nocedal', 'bb1', or 'safeguarded'.")


# =============================================================================
# 7. FLOP COUNTER  (derived from Algorithm 9.1 structure)
# =============================================================================
#
# Convention (same as lib/qr_householder.py, Golub & Van Loan, Trefethen &
# Bau): one flop = one floating-point addition OR multiplication, so a
# length-m dot product or axpy costs 2m flops, and a product X^T v with
# X in R^{m x n} costs 2mn flops.
#
# Per-iteration flop count of L-BFGS (dominant terms only):
#
#   Two-loop recursion:  (8 * mem + 1) * m   [per stored pair: one dot (2m)
#                                             and one axpy (2m) in each of the
#                                             two loops; plus H_0 q (m)]
#   Gradient eval:        4 * m * n          [X^T w (2mn) + X r (2mn)]
#   Exact LS:             2 * m * n + 4 * m  [X^T p (2mn) + two dots (4m)]
#   Update s_k, y_k:      2 * m              [alpha*p (m) + g_new - g (m)]
#
#   Objective value:      2 * m * n          [X^T w, stagnation check]
#
# Total per iteration ≈ (8*mem + 1)*m + 8*m*n + 6*m
#                      = m * (8*mem + 8*n + 7)     flops
#
# Storage: 2 * mem * m  floats for the (s_i, y_i) pairs
#          + 4 * m  for current w, grad, s_k, y_k
#          Total: (2*mem + 4) * m  floats
# =============================================================================

def theoretical_cost(m, n, mem, n_iter,
                     line_search='exact', avg_wolfe_evals=3):
    """
    Compute theoretical flop count and storage for L-BFGS.
    Upper-bound estimate: assumes a full memory of mem pairs at every iteration;
    see actual_cost for the work really performed.

    Parameters
    ----------
    m, n, mem, n_iter : problem dimensions, memory size, iteration count
    line_search       : 'exact' or 'wolfe'
    avg_wolfe_evals   : int, average number of (f, grad) re-evaluations
                        inside the Wolfe bracketing+zoom (typical: 2-4
                        for a well-implemented quasi-Newton LS).

    Returns
    -------
    dict with keys 'flops_per_iter', 'total_flops',
                   'storage_floats', 'storage_MB', 'line_search'.
    """
    two_loop   = (8 * mem + 1) * m          # Algorithm 9.1
    grad_eval  = 4 * m * n                  # X(X^T w - y)
    update     = 2 * m                      # s_k = alpha*p, y_k = g_new - g

    if line_search == 'exact':
        # one X^T p (2mn) + p^T p and grad^T p (2m each) = 2mn + 4m,
        # plus the objective value used by the stagnation check (2mn)
        ls_cost = 2 * m * n + 4 * m + 2 * m * n
    elif line_search == 'wolfe':
        # Each trial point: f eval (2mn) + grad eval (4mn) = 6mn,
        # plus the dot product with p (2m) and the trial update (2m).
        ls_cost = avg_wolfe_evals * (6 * m * n + 4 * m)
    else:
        raise ValueError(f"Unknown line_search='{line_search}'")

    flops_per_iter = two_loop + grad_eval + ls_cost + update
    total_flops    = n_iter * flops_per_iter

    storage_floats = (2 * mem + 4) * m
    storage_MB     = storage_floats * 8 / 1e6

    return {
        'flops_per_iter' : flops_per_iter,
        'total_flops'    : total_flops,
        'storage_floats' : storage_floats,
        'storage_MB'     : storage_MB,
        'line_search'    : line_search,
    }


def print_cost_table(m, n, mem_values=(3, 5, 10, 20, 40), n_iter=50):
    """
    Print a summary table of theoretical costs for different memory sizes.
    """
    print(f"\n{'='*70}")
    print(f"  Theoretical cost analysis  (m={m}, n={n}, n_iter={n_iter})")
    print(f"{'='*70}")
    print(f"  {'mem':>4}  {'flops/iter':>12}  {'total flops':>14}  "
          f"{'storage (MB)':>13}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*14}  {'-'*13}")
    for mem in mem_values:
        c = theoretical_cost(m, n, mem, n_iter)
        print(f"  {mem:>4}  {c['flops_per_iter']:>12,}  "
              f"{c['total_flops']:>14,}  "
              f"{c['storage_MB']:>12.3f}")
    print(f"{'='*70}\n")

def actual_cost(history, m, n, line_search='exact'):
    """Flops actually performed by a completed L-BFGS run (dominant terms).

    Unlike theoretical_cost, which assumes a full memory at every iteration,
    the two-loop term uses the number of pairs really stored at each
    iteration (history['mem']): fewer than m_history while the memory fills
    up and after a restart.  It also includes the objective value computed
    at each iteration (2mn), used by the stagnation check.  Same flop
    convention as theoretical_cost.
    """
    mem = np.asarray(history['mem'], dtype=float)
    k = mem.size
    two_loop = float(np.sum((8 * mem + 1) * m))
    if line_search == 'exact':
        # gradient (4mn) + exact LS (2mn + 4m) + objective (2mn) + update (2m)
        rest = k * (4 * m * n + (2 * m * n + 4 * m) + 2 * m * n + 2 * m)
    else:
        evals = np.asarray(history['ls_evals'], dtype=float)
        rest = float(np.sum(evals * (6 * m * n + 4 * m))) + k * 2 * m
    total = int(round(two_loop + rest))
    return {'n_iter': k, 'mean_pairs': float(mem.mean()) if k else 0.0,
            'total_flops': total, 'flops_per_iter': total / max(k, 1)}

# =============================================================================
# 8. ROBUST BENCHMARKING WRAPPER
# =============================================================================
#
# Wall-time measurements with a single run are unreliable for problems
# that solve in O(ms): OS scheduling, cache warm-up, and JIT effects
# easily change the timing by 10-50%. This wrapper repeats the call N
# times and returns robust statistics (median is preferred over mean
# because it is insensitive to occasional outliers from the OS).
# =============================================================================

def benchmark_lbfgs(X, y, lam, n_runs=10, **kwargs):
    """
    Run lbfgs_optimize n_runs times and return robust timing statistics.

    Parameters
    ----------
    X, y, lam : problem data
    n_runs    : int, number of repetitions (>=3 recommended; default 10)
    **kwargs  : passed to lbfgs_optimize (verbose is forced to False)

    Returns
    -------
    w_last   : ndarray  solution from the last run
    h_last   : dict     history from the last run
    stats    : dict     keys: 'n_iter', 'n_runs', 'time_median',
                              'time_min', 'time_mean', 'time_std',
                              'time_all'
    """
    kwargs['verbose'] = False
    times  = np.empty(n_runs)
    w_last = None
    h_last = None

    # Warm-up run (not counted): primes caches, triggers any first-time
    # imports/allocations. Crucial when reporting sub-millisecond timings.
    w_last, h_last, _ = lbfgs_optimize(X, y, lam, **kwargs)

    for i in range(n_runs):
        w_last, h_last, t = lbfgs_optimize(X, y, lam, **kwargs)
        times[i] = t

    return w_last, h_last, {
        'n_iter'      : len(h_last['f']) - 1,
        'n_runs'      : n_runs,
        'time_median' : float(np.median(times)),
        'time_min'    : float(times.min()),
        'time_mean'   : float(times.mean()),
        'time_std'    : float(times.std(ddof=1)) if n_runs > 1 else 0.0,
        'time_all'    : times.tolist(),
    }

# =============================================================================
# 4. L-BFGS MAIN LOOP  (Nocedal & Wright, Algorithm 9.2)  — Enhanced
# =============================================================================

def lbfgs_optimize(X, y, lam,
                   m_history=12,
                   max_iter=1000,
                   tol=1e-14,
                   tol_type='relative',
                   line_search='exact',
                   h0_scaling='bb1',
                   use_restart=False,
                   restart_xi=0.2,
                   w_star=None,
                   f_star=None,
                   verbose=True,
                   w_init=None,
                   wolfe_c1=1e-4,
                   wolfe_c2=0.9):
    """
    Minimizes f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2
    with the L-BFGS method (Nocedal & Wright, Algorithm 9.2).

    Parameters
    ----------
    X           : ndarray (m, n)
    y           : ndarray (n,)
    lam         : float
    m_history   : int    number of (s,y) pairs stored; default 12, the
                  effective rank n of the ML-CUP problem
    max_iter    : int
    tol         : float
    tol_type    : str    'relative' or 'absolute'
    line_search : str    'exact' or 'wolfe'
    h0_scaling  : str    'bb1'     (default, Barzilai & Borwein 1988, Eq. (6))
                         'nocedal' (Eq. 9.6 = BB2)
                         'safeguarded' (BB2 clipped to [gamma_min, gamma_max])
    use_restart : bool   enable curvature-based restart (heuristic safeguard);
                  default False
    restart_xi  : float  restart threshold in (0,1); default 0.2
    w_star      : ndarray (m,) or None
                  Optional reference solution. When provided, the iterate
                  distance ||w_k - w*|| is recorded at every iteration in
                  history['dist_to_opt']. Used only for diagnostic /
                  empirical-convergence-rate experiments; the optimizer
                  itself does not use it.
    f_star      : float or None
                  Optional reference optimal value. When provided, the
                  sub-optimality gap f(w_k) - f(w*) is recorded at every
                  iteration in history['gap_to_opt'].
    verbose     : bool
    w_init      : ndarray (m,) or None
                  Initial iterate. If None (default), starts from zeros.
                  Used by experiments studying sensitivity to w_0.
    wolfe_c1    : float, sufficient-decrease (Armijo) constant for the
                  Wolfe LS. Default 1e-4 (Nocedal & Wright Sec. 3.1
                  recommendation for quasi-Newton). Ignored if
                  line_search='exact'.
    wolfe_c2    : float, curvature constant for the Wolfe LS. Default 0.9
                  (quasi-Newton standard). Smaller values produce stricter
                  line searches: more f/grad evaluations per iter, but
                  better-conditioned (s_k, y_k) pairs and potentially
                  fewer outer iterations. Must satisfy wolfe_c1 < wolfe_c2.
                  Ignored if line_search='exact'.

    Returns
    -------
    w         : ndarray (m,)   final iterate
    history   : dict
                  Always present keys:
                    'f'         : list of f(w_k)
                    'grad_norm' : list of ||grad f(w_k)||
                    'alpha'     : list of step lengths (one per iter)
                    'ls_evals'  : list of line-search evaluations per iter
                    'restarts'  : iteration indices where memory was cleared
                  Optional keys (only if w_star / f_star were passed):
                    'dist_to_opt' : list of ||w_k - w*||
                    'gap_to_opt'  : list of f(w_k) - f(w*)
    elapsed   : float          wall-clock seconds (single run; use
                               benchmark_lbfgs for robust timing)
    """
    m_dim = X.shape[0]

    # --- Initialization ---
    w    = np.zeros(m_dim) if w_init is None else np.asarray(w_init, dtype=float).copy()
    grad = compute_gradient(w, X, y, lam)
    f_val = compute_loss(w, X, y, lam)

    grad_0_norm = np.sqrt(np.dot(grad, grad))
    stop_tol = tol * grad_0_norm if tol_type == 'relative' else tol

    s_list, y_list, rho_list = [], [], []
    gamma_prev = 1.0   # needed for restart criterion

    history = {
        'f'         : [f_val],
        'grad_norm' : [grad_0_norm],
        'alpha'     : [],
        'ls_evals'  : [],
        'restarts'  : [],   # iteration indices where restart occurred
        'mem'       : [],   # stored pairs used by the two-loop at each iteration
    }
    if w_star is not None:
        history['dist_to_opt'] = [float(np.linalg.norm(w - w_star))]
    if f_star is not None:
        history['gap_to_opt'] = [float(f_val - f_star)]

    if verbose:
        print(f"[L-BFGS] m={m_dim}, n={X.shape[1]}, ls='{line_search}', "
              f"h0='{h0_scaling}', restart={use_restart}(xi={restart_xi}), "
              f"tol={tol} ({tol_type}), m_history={m_history}")

    start_time = time.perf_counter()

    for k in range(max_iter):
        grad_norm = np.sqrt(np.dot(grad, grad))

        # --- Primary stopping criterion ---
        if grad_norm < stop_tol:
            if verbose:
                print(f"[L-BFGS] Converged at iter {k}: "
                      f"||grad||={grad_norm:.2e} < {stop_tol:.2e}")
            break

        # --- Stagnation check ---
        if k > 0 and history['alpha'] and history['alpha'][-1] < 1e-12:
            f_prev     = history['f'][-2]
            rel_change = abs(f_val - f_prev) / max(abs(f_val), 1e-30)
            if rel_change < 1e-16:
                if verbose:
                    print(f"[L-BFGS] Stagnation at iter {k}: "
                          f"||grad||={grad_norm:.2e}")
                break

        # --- Compute gamma_k with chosen scaling strategy ---
        gamma_k = _compute_gamma(s_list, y_list,
                                 scaling=h0_scaling) if s_list else (
            1.0 / grad_norm if grad_norm > 0 else 1.0)

        # --- Descent direction ---
        Hg = lbfgs_two_loop(grad, s_list, y_list, rho_list, gamma_k)
        p  = -Hg
        history['mem'].append(len(s_list))

        dg = np.dot(grad, p)
        if dg >= 0:          # safeguard: revert to steepest descent
            p  = -grad
            dg = np.dot(grad, p)

        # --- Line search ---
        if line_search == 'exact':
            alpha    = exact_line_search(grad, p, X, lam)
            w_new    = w + alpha * p
            f_new    = compute_loss(w_new, X, y, lam)
            grad_new = compute_gradient(w_new, X, y, lam)
            ls_evals = 1
        else:
            alpha, f_new, grad_new, ls_evals = strong_wolfe_line_search(
                w, p, f_val, grad, dg, X, y, lam,
                c1=wolfe_c1, c2=wolfe_c2, alpha_init=1.0)

        # --- Compute new pair ---
        s_k = alpha * p
        y_k = grad_new - grad
        ys  = np.dot(y_k, s_k)

        # --- Curvature-based restart ---
        did_restart = False
        ok = ys > 1e-10 * np.linalg.norm(s_k) * np.linalg.norm(y_k)
        if use_restart and ok and len(s_list) > 0:
            if _should_restart(s_k, y_k, ys, gamma_prev, xi=restart_xi):
                s_list.clear()
                y_list.clear()
                rho_list.clear()
                history['restarts'].append(k)
                did_restart = True
                if verbose:
                    print(f"  [restart] iter {k}: curvature dropped, memory cleared")

        # --- Standard skip / store logic ---
        if ok:
            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(1.0 / ys)
            if len(s_list) > m_history:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            gamma_prev = ys / np.dot(y_k, y_k)   # update for next restart check
        elif ys <= 0 and not did_restart and len(s_list) > 0:
            # Negative curvature (y^T s <= 0): full reset
            s_list.clear()
            y_list.clear()
            rho_list.clear()
            history['restarts'].append(k)
            if verbose:
                print(f"  [reset] iter {k}: negative curvature (ys={ys:.2e})")

        # --- State update ---
        w     = w + s_k
        grad  = grad_new
        f_val = f_new

        history['f'].append(f_val)
        history['grad_norm'].append(np.sqrt(np.dot(grad, grad)))
        history['alpha'].append(alpha)
        history['ls_evals'].append(ls_evals)

        if w_star is not None:
            history['dist_to_opt'].append(float(np.linalg.norm(w - w_star)))
        if f_star is not None:
            history['gap_to_opt'].append(float(f_val - f_star))

        if verbose and (k % 100 == 0 or k < 5):
            print(f"  k={k:4d}  f={f_val:.8e}  "
                  f"||g||={history['grad_norm'][-1]}  "
                  f"α={alpha:.4e}  mem={len(s_list)}")
    else:
        if verbose:
            print(f"[L-BFGS] Max iter reached ({max_iter}).")

    elapsed = time.perf_counter() - start_time
    return w, history, elapsed