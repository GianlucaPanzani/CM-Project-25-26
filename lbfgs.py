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

Implemented components:
    1. Two-loop recursion          (Algorithm 9.1)
    2. Exact line search           (Generalized Eq. 3.25)
    3. Strong Wolfe line search    (Eq. 3.7 with zoom + cubic interpolation)
    4. L-BFGS main loop            (Algorithm 9.2)
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
# 4. L-BFGS MAIN LOOP  (Nocedal & Wright, Algorithm 9.2)
# =============================================================================

def lbfgs_optimize(X, y, lam,
                   m_history=10,
                   max_iter=1000,
                   tol=1e-7,
                   tol_type='relative',
                   line_search='exact',
                   verbose=True):
    """
    Minimizes f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2
    with the L-BFGS method (Algorithm 9.2, Nocedal & Wright).

    Parameters
    ---------
    X           : ndarray (m, n)  data matrix
    y           : ndarray (n,)    target vector
    lam         : float           regularization parameter (lambda > 0)
    m_history   : int             number of (s, y) pairs in memory [3, 20]
    max_iter    : int             maximum number of iterations
    tol         : float           tolerance for the stopping criterion
    tol_type    : str             'relative' -> ||grad_k|| <= tol * ||grad_0||
                                  'absolute' -> ||grad_k|| <= tol
    line_search : str             'exact'  -> exact quadratic LS (Eq. 3.25)
                                  'wolfe'  -> strong Wolfe LS (Eq. 3.7)
    verbose     : bool            print information during execution

    Returns
    -------
    w       : ndarray (m,)   optimal solution
    history : dict           histories of f, ||grad||, alpha, ls_evals, rel_error
    elapsed : float          execution time (seconds)
    """
    m_dim = X.shape[0]

    # --- Initialization ---
    w = np.zeros(m_dim)
    grad = compute_gradient(w, X, y, lam)
    f_val = compute_loss(w, X, y, lam)

    # Stopping threshold
    grad_0_norm = np.sqrt(np.dot(grad, grad))
    if tol_type == 'relative':
        stop_tol = tol * grad_0_norm
    else:
        stop_tol = tol

    # Limited memory
    s_list, y_list, rho_list = [], [], []

    # History
    history = {
        'f':         [f_val],
        'grad_norm': [grad_0_norm],
        'alpha':     [],
        'ls_evals':  [],
    }

    if verbose:
        print(f"[L-BFGS] m={m_dim}, ls='{line_search}', "
              f"tol={tol} ({tol_type}, threshold={stop_tol:.2e}), "
              f"m_history={m_history}")

    start_time = time.time()

    for k in range(max_iter):
        grad_norm = np.sqrt(np.dot(grad, grad))

        # --- Primary stopping criterion ---
        if grad_norm < stop_tol:
            if verbose:
                print(f"[L-BFGS] Convergence at iter {k}: "
                      f"||grad||={grad_norm:.2e} < {stop_tol:.2e}")
            break

        # Additional criterion: numerical stagnation
        if k > 0 and len(history['alpha']) > 0:
            f_prev = history['f'][-2]
            rel_change = abs(f_val - f_prev) / max(abs(f_val), 1e-30)
            if rel_change < 1e-16 and history['alpha'][-1] < 1e-12:
                if verbose:
                    print(f"[L-BFGS] Stagnation at iter {k}: "
                          f"||grad||={grad_norm:.2e}")
                break

        # --- Scaling gamma_k  (Eq. 9.6) ---
        if len(s_list) > 0:
            gamma_k = (np.dot(s_list[-1], y_list[-1])
                       / np.dot(y_list[-1], y_list[-1]))
        else:
            gamma_k = 1.0 / grad_norm if grad_norm > 0 else 1.0

        # --- Descent direction: p = -H_k grad  (Algorithm 9.1) ---
        Hg = lbfgs_two_loop(grad, s_list, y_list, rho_list, gamma_k)
        p = -Hg

        # Check descent
        dg = np.dot(grad, p)
        if dg >= 0:
            p = -grad
            dg = np.dot(grad, p)

        # --- Line search ---
        if line_search == 'exact':
            alpha = exact_line_search(grad, p, X, lam)
            w_new = w + alpha * p
            f_new = compute_loss(w_new, X, y, lam)
            grad_new = compute_gradient(w_new, X, y, lam)
            ls_evals = 1
        else:
            alpha, f_new, grad_new, ls_evals = strong_wolfe_line_search(
                w, p, f_val, grad, dg, X, y, lam, alpha_init=1.0)

        # --- Pair update (s_k, y_k) ---
        s_k = alpha * p
        y_k = grad_new - grad
        ys = np.dot(y_k, s_k)

        if ys > 1e-10:
            s_list.append(s_k)
            y_list.append(y_k)
            rho_list.append(1.0 / ys)
            if len(s_list) > m_history:
                s_list.pop(0)
                y_list.pop(0)
                rho_list.pop(0)
        elif ys <= 1e-16 and len(s_list) > 0:
            # Curvature too small: reset memory (cf. Liu & Nocedal)
            s_list.clear()
            y_list.clear()
            rho_list.clear()

        # --- State update ---
        w = w + s_k
        grad = grad_new
        f_val = f_new

        # --- History ---
        history['f'].append(f_val)
        history['grad_norm'].append(np.sqrt(np.dot(grad, grad)))
        history['alpha'].append(alpha)
        history['ls_evals'].append(ls_evals)

        # --- Periodic log ---
        if verbose and (k % 100 == 0 or k < 5):
            print(f"  k={k:4d}  f={f_val:.8e}  "
                  f"||g||={history['grad_norm'][-1]:.2e}  "
                  f"α={alpha:.4e}")
    else:
        if verbose:
            print(f"[L-BFGS] Max iter reached ({max_iter}).")

    elapsed = time.time() - start_time
    return w, history, elapsed