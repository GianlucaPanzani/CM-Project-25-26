"""
================================================================================
lbfgs.py — A1: Limited-Memory BFGS (L-BFGS)
================================================================================

L-BFGS algorithm for the problem:
    min_w  f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2

References:
    - Nocedal & Wright, "Numerical Optimization", 2006
      - Algorithm 9.1  (Two-Loop Recursion)
      - Algorithm 9.2  (L-BFGS)
      - Eq. 9.6        (Initial scaling gamma_k)
      - Eq. 3.25       (Exact line search on quadratics)
      - Eq. 3.7        (Strong Wolfe conditions)
    - Liu & Nocedal, "On the limited memory BFGS method for large scale
      optimization", Mathematical Programming 45, 1989
      - Section 4      (Curvature-based restart criterion)

References (fonti esterne):
    - Barzilai & Borwein, "Two-point step size gradient methods",
      IMA Journal of Numerical Analysis 8(1), 1988
      - Eq. (3.1) BB1 scaling:  gamma = s^T s / s^T y
      - Eq. (3.2) BB2 scaling:  gamma = s^T y / y^T y  (= Eq. 9.6 Nocedal)
    - Dai & Liao, "R-linear convergence of the Barzilai and Borwein
      gradient method", IMA J. Numerical Analysis 22, 2002
      - safeguarded BB: clip gamma_k to [gamma_min, gamma_max]

Implemented components:
    1. Two-loop recursion          (Nocedal & Wright, Algorithm 9.1)
    2. Exact line search           (Nocedal & Wright, Eq. 3.25)
    3. Strong Wolfe line search    (Nocedal & Wright, Eq. 3.7)
    4. L-BFGS main loop            (Nocedal & Wright, Algorithm 9.2)
    5. Curvature-based restart     (Liu & Nocedal 1989, Section 4)
    6. Adaptive H0 scaling         (Barzilai & Borwein 1988)
    7. Flop counter                (derived from Algorithm 9.1 structure)
================================================================================
"""

import numpy as np
import time
from utils import compute_loss, compute_gradient


# =============================================================================
# 1. TWO-LOOP RECURSION  (Nocedal & Wright, Algorithm 9.1)
# =============================================================================

def lbfgs_two_loop(grad_k, s_list, y_list, rho_list, gamma_k):
    """
    Calculates the product H_k * grad_k via the two-loop recursion.

    The matrix H_k (approximation of the inverse Hessian) is NEVER
    formed explicitly. Only the m most recent {s_i, y_i} pairs and
    the scaling H_0 = gamma_k * I are used.

    Cost: O(4 * m_history * dim) multiplications + O(dim) for H_0.

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
# Source: Liu & Nocedal (1989), Section 4.
#
# The standard L-BFGS only skips a pair when y_k^T s_k <= 0 (negative
# curvature). Liu & Nocedal (1989) suggest a stronger criterion: reset the
# entire memory when the curvature captured by the new pair is negligible
# relative to the current H0 scaling, i.e. when:
#
#     y_k^T s_k / ||y_k||^2  <  xi * gamma_{k-1}
#
# where gamma_{k-1} = s_{k-1}^T y_{k-1} / ||y_{k-1}||^2 is the previous
# scaling and xi in (0,1) is a user-chosen threshold (default 0.2).
#
# Intuition: gamma_k = s^T y / y^T y  is the "effective curvature radius"
# estimated along s_k. If it drops dramatically relative to the previous
# estimate, the memory contains stale / contradictory curvature information
# and should be discarded.
# =============================================================================

def _should_restart(s_k, y_k, ys, gamma_prev, xi=0.2):
    """
    Curvature-based restart criterion (Liu & Nocedal 1989, Section 4).

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
#  'nocedal' (default, already in original code):
#      gamma_k = s_{k-1}^T y_{k-1} / y_{k-1}^T y_{k-1}           [Eq. 9.6]
#      This is also known as the BB2 step (Barzilai & Borwein 1988, Eq. 3.2).
#      It estimates the inverse curvature of the Hessian along s_{k-1}.
#
#  'bb1' (Barzilai & Borwein 1988, Eq. 3.1):
#      gamma_k = s_{k-1}^T s_{k-1} / s_{k-1}^T y_{k-1}
#      Interpretation: minimizes ||gamma * y - s||, i.e. finds the scalar
#      alpha such that alpha * H_true * s ≈ s  =>  alpha ≈ 1/lambda_avg.
#      Often produces larger steps and can accelerate convergence on
#      ill-conditioned problems.
#
#  'safeguarded' (Dai & Liao 2002):
#      gamma_k = clip( BB2_k, gamma_min, gamma_max )
#      Prevents gamma from becoming extremely large or small due to
#      numerical noise, which can destabilize the two-loop recursion.
#      Default bounds: gamma_min=1e-10, gamma_max=1e10.
#
# Source: Barzilai & Borwein (1988) is external to the prof's references.
#         Nocedal & Wright Eq. 9.6 covers 'nocedal' / BB2.
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
        return 1.0

    if scaling == 'nocedal':
        # Nocedal & Wright Eq. 9.6  (= BB2)
        return sy / yy

    elif scaling == 'bb1':
        # Barzilai & Borwein (1988), Eq. 3.1
        return ss / sy

    elif scaling == 'safeguarded':
        # BB2 clipped to [gamma_min, gamma_max]  (Dai & Liao 2002)
        gamma = sy / yy
        return float(np.clip(gamma, gamma_min, gamma_max))

    else:
        raise ValueError(f"Unknown scaling='{scaling}'. "
                         "Choose 'nocedal', 'bb1', or 'safeguarded'.")


# =============================================================================
# [NEW] 7. FLOP COUNTER  (derived from Algorithm 9.1 structure)
# =============================================================================
#
# Per-iteration flop count of L-BFGS (dominant terms only):
#
#   Two-loop recursion:  (4 * mem + 1) * m      [Nocedal & Wright, p. 225]
#   Gradient eval:        2 * m * n              [X^T w (mn) + X r (mn)]
#   Exact LS:             m * n + 2 * m          [X^T p + two dots]
#   Update s_k, y_k:      2 * m                  [two axpy]
#
# Total per iteration ≈ (4*mem + 1)*m + 3*m*n + 4*m
#                      = m * (4*mem + 3*n + 5)     flops
#
# Storage: 2 * mem * m  floats for the (s_i, y_i) pairs
#          + 4 * m  for current w, grad, s_k, y_k
#          Total: (2*mem + 4) * m  floats
# =============================================================================

def theoretical_cost(m, n, mem, n_iter):
    """
    Compute theoretical flop count and storage for L-BFGS.

    Parameters
    ----------
    m      : int   problem dimension (number of rows of X)
    n      : int   number of features (columns of X)
    mem    : int   L-BFGS memory size (m_history)
    n_iter : int   number of iterations performed

    Returns
    -------
    dict with keys:
        'flops_per_iter'  : dominant flops per iteration
        'total_flops'     : total flops over n_iter iterations
        'storage_floats'  : number of floats stored (excluding X)
        'storage_MB'      : storage in megabytes (float64)
    """
    two_loop   = (4 * mem + 1) * m          # two-loop recursion
    grad_eval  = 2 * m * n                  # gradient: X(X^T w - y)
    exact_ls   = m * n + 2 * m              # X^T p + two dot products
    update     = 2 * m                      # s_k = alpha*p, y_k = g_new - g

    flops_per_iter = two_loop + grad_eval + exact_ls + update
    total_flops    = n_iter * flops_per_iter

    storage_floats = (2 * mem + 4) * m      # pairs + w, grad, s_k, y_k
    storage_MB     = storage_floats * 8 / 1e6   # float64 = 8 bytes

    return {
        'flops_per_iter' : flops_per_iter,
        'total_flops'    : total_flops,
        'storage_floats' : storage_floats,
        'storage_MB'     : storage_MB,
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


# =============================================================================
# 4. L-BFGS MAIN LOOP  (Nocedal & Wright, Algorithm 9.2)  — Enhanced
# =============================================================================

def lbfgs_optimize(X, y, lam,
                   m_history=10,
                   max_iter=1000,
                   tol=1e-7,
                   tol_type='relative',
                   line_search='exact',
                   h0_scaling='nocedal',
                   use_restart=True,
                   restart_xi=0.2,
                   verbose=True):
    """
    Minimizes f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2
    with the L-BFGS method (Nocedal & Wright, Algorithm 9.2).

    Parameters
    ----------
    X           : ndarray (m, n)
    y           : ndarray (n,)
    lam         : float
    m_history   : int    number of (s,y) pairs stored [3..20]
    max_iter    : int
    tol         : float
    tol_type    : str    'relative' or 'absolute'
    line_search : str    'exact' or 'wolfe'
    h0_scaling  : str    'nocedal' (default, Eq. 9.6 = BB2)
                         'bb1'     (Barzilai & Borwein 1988, Eq. 3.1)
                         'safeguarded' (BB2 clipped, Dai & Liao 2002)
    use_restart : bool   enable curvature-based restart (Liu & Nocedal 1989)
    restart_xi  : float  restart threshold in (0,1); default 0.2
    verbose     : bool

    Returns
    -------
    w         : ndarray (m,)
    history   : dict  keys: f, grad_norm, alpha, ls_evals, restarts
    elapsed   : float  seconds
    """
    m_dim = X.shape[0]

    # --- Initialization ---
    w    = np.zeros(m_dim)
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
    }

    if verbose:
        print(f"[L-BFGS] m={m_dim}, n={X.shape[1]}, ls='{line_search}', "
              f"h0='{h0_scaling}', restart={use_restart}(xi={restart_xi}), "
              f"tol={tol} ({tol_type}), m_history={m_history}")

    start_time = time.time()

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
                w, p, f_val, grad, dg, X, y, lam, alpha_init=1.0)

        # --- Compute new pair ---
        s_k = alpha * p
        y_k = grad_new - grad
        ys  = np.dot(y_k, s_k)

        # --- Curvature-based restart (Liu & Nocedal 1989, Sec. 4) ---
        did_restart = False
        if use_restart and ys > 1e-10 and len(s_list) > 0:
            if _should_restart(s_k, y_k, ys, gamma_prev, xi=restart_xi):
                s_list.clear()
                y_list.clear()
                rho_list.clear()
                history['restarts'].append(k)
                did_restart = True
                if verbose:
                    print(f"  [restart] iter {k}: curvature dropped, memory cleared")

        # --- Standard skip / store logic ---
        if ys > 1e-10:
            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(1.0 / ys)
            if len(s_list) > m_history:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
            gamma_prev = ys / np.dot(y_k, y_k)   # update for next restart check
        elif ys <= 1e-16 and not did_restart and len(s_list) > 0:
            # Negative curvature: full reset
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

        if verbose and (k % 100 == 0 or k < 5):
            print(f"  k={k:4d}  f={f_val:.8e}  "
                  f"||g||={history['grad_norm'][-1]}  "
                  f"α={alpha:.4e}  mem={len(s_list)}")
    else:
        if verbose:
            print(f"[L-BFGS] Max iter reached ({max_iter}).")

    elapsed = time.time() - start_time
    return w, history, elapsed