"""
================================================================================
utils.py — Base functions for problem (P)
================================================================================

Problem:
    min_w || hat_X w - hat_y ||

with:
    hat_X = [X^T; lambda*I_m]   ∈ R^{(n+m) x m}
    hat_y = [y; 0]              ∈ R^{n+m}

Equivalently:
    min_w  f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2

Matrices:
    X ∈ R^{m x n}   (tall-thin, from the ML-CUP dataset)
    y ∈ R^n          (random vector)
    lambda > 0       (regularization parameter)
    w ∈ R^m          (vector to optimize)

Hessian (constant):
    H = nabla^2 f(w) = X X^T + lambda^2 I_m

References:
    - Nocedal & Wright, "Numerical Optimization", 2006
    - Trefethen & Bau, "Numerical Linear Algebra", 1997
================================================================================
"""

import numpy as np
import pandas as pd
import time


# =============================================================================
# 1. OBJECTIVE FUNCTION AND GRADIENT
# =============================================================================

def compute_loss(w, X, y, lam):
    """
    Calculates the value of the objective function
    f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2.

    Parameters
    ---------
    w   : ndarray (m,)   weights vector
    X   : ndarray (m, n) data matrix (tall-thin)
    y   : ndarray (n,)   target vector
    lam : float          regularization parameter (lambda > 0)

    Returns
    -------
    float  value of the objective function
    """

    # Calculate the residual vector: (X^T * w) - y
    residual = X.T @ w - y

    # Measures how well the model fits the data (minimizes the error)
    # Data Loss term:
    # 1/2 * ||X^T w - y||^2
    loss_data = 0.5 * np.dot(residual, residual)

    # Penalizes overly large weights (w) to prevent overfitting and stabilize the problem
    # Regularization Loss term:
    # 1/2 * lambda^2 * ||w||^2
    loss_reg = 0.5 * (lam ** 2) * np.dot(w, w)

    return loss_data + loss_reg


def compute_gradient(w, X, y, lam):
    """
    Calculates the gradient vector of f(w) with respect to w.
    nabla f(w) = X(X^T w - y) + lambda^2 w.

    Parameters
    ---------
    w   : ndarray (m,)   weights vector
    X   : ndarray (m, n) data matrix
    y   : ndarray (n,)   target vector
    lam : float          regularization parameter

    Returns
    -------
    ndarray (m,)  gradient
    """

    # Calculate the residual vector: (X^T * w) - y     (dimension n)
    residual = X.T @ w - y

    # Derivative of the data loss term
    grad_data = X @ residual

    # Derivative of the regularization term
    grad_reg = (lam ** 2) * w

    return grad_data + grad_reg


# =============================================================================
# 2. GRADIENT CHECK (central finite differences)
# =============================================================================

def check_gradient(w, X, y, lam, eps=1e-7):
    """
    Verifies the analytical gradient using central finite differences.
    f'(x) approx (f(x + eps) - f(x - eps)) / (2 * eps)

    Returns
    -------
    abs_error : float  absolute error ||grad_approx - grad_an||
    rel_error : float  relative error ||grad_approx - grad_an|| / ||grad_an||
    """
    m = len(w)
    grad_approx = np.zeros(m)

    for i in range(m):
        # Forward perturbation
        w_plus = w.copy()
        w_plus[i] += eps
        loss_plus = compute_loss(w_plus, X, y, lam)

        # Backward perturbation
        w_minus = w.copy()
        w_minus[i] -= eps
        loss_minus = compute_loss(w_minus, X, y, lam)

        # Approximated i-th partial derivative
        grad_approx[i] = (loss_plus - loss_minus) / (2 * eps)

    # Calculate the analytical gradient
    grad_analytic = compute_gradient(w, X, y, lam)

    # Calculate the difference vector between approximated and exact gradient
    diff = grad_approx - grad_analytic

    # ABSOLUTE ERROR: calculate the L2 norm (Euclidean distance) of the difference.
    error = np.sqrt(np.dot(diff, diff))

    # Calculate the L2 norm of the analytical gradient to use as a scaling term
    norm_an = np.sqrt(np.dot(grad_analytic, grad_analytic))

    # RELATIVE ERROR: divide the absolute error by the magnitude of the analytical gradient.
    # Use max(norm_an, 1e-30) to ensure numerical stability: if norm_an were 0,
    # the max intervenes and prevents a program crash due to division by zero.
    return error, error / max(norm_an, 1e-30)


# =============================================================================
# 3. EXACT SOLUTION (baseline via normal equations)
# =============================================================================

def solve_exact(X, y, lam):
    """
    Calculates the exact solution w* by solving the normal equations:
        (X X^T + lambda^2 I) w = X y

    NOTE: uses np.linalg.solve only as a ground truth for verification.

    Returns
    -------
    w_baseline   : ndarray (m,)  exact solution
    f_baseline   : float         optimal value f(w*)
    baseline_time: float         execution time
    """

    m = X.shape[0]
    A = X @ X.T + (lam ** 2) * np.eye(m)
    b = X @ y

    start_time = time.time()

    w_baseline = np.linalg.solve(A, b)
    baseline_time = time.time() - start_time

    f_baseline = compute_loss(w_baseline, X, y, lam)

    return w_baseline, f_baseline, baseline_time


# =============================================================================
# 4. ML-CUP DATASET LOADING AND PREPROCESSING
# =============================================================================

def load_ml_cup(filepath, seed=42):
    """
    Loads the ML-CUP dataset and preprocesses it.

    - Reads the CSV (skipping comment lines starting with '#')
    - Extracts features (columns 1:-4) as matrix X
    - Standardizes X (mean 0, standard deviation 1)
    - Generates a random target vector y of dimension n

    Parameters
    ---------
    filepath : str   path to the ML-CUP25-TR.csv file
    seed     : int   seed for y generation

    Returns
    -------
    X : ndarray (m, n)  standardized data matrix
    y : ndarray (n,)    random target vector
    m : int             number of rows
    n : int             number of columns
    """

    df = pd.read_csv(filepath, comment='#', header=None)
    X_raw = df.iloc[:, 1:-4].values
    m, n = X_raw.shape

    # Standardization
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X_std[X_std < 1e-12] = 1.0  # avoids division by zero
    X = (X_raw - X_mean) / X_std

    # Generate random y
    np.random.seed(seed)
    y = np.random.randn(n)

    return X, y, m, n


# =============================================================================
# 5. AUGMENTED SYSTEM CONSTRUCTION (for QR)
# =============================================================================

def build_augmented_system(X, y, lam):
    """
    Constructs the augmented matrix hat_X and augmented vector hat_y.

        hat_X = [X^T; lambda*I_m]   ∈ R^{(n+m) x m}
        hat_y = [y; 0]              ∈ R^{n+m}

    Parameters
    ---------
    X   : ndarray (m, n)
    y   : ndarray (n,)
    lam : float

    Returns
    -------
    X_hat : ndarray (n+m, m)
    y_hat : ndarray (n+m,)
    """
    m, n = X.shape
    X_hat = np.vstack([X.T, lam * np.eye(m)])
    y_hat = np.concatenate([y, np.zeros(m)])
    return X_hat, y_hat


# =============================================================================
# 6. EVALUATION METRICS
# =============================================================================

def compute_condition_number(X, lam):
    """
    Calculates the condition number kappa(hat_X).

    kappa = sqrt((lambda_1 + lambda^2) / (lambda_m + lambda^2))

    where lambda_1, lambda_m are the maximum and minimum eigenvalues of XX^T.
    """
    eigvals = np.linalg.eigvalsh(X @ X.T)
    eigvals = np.maximum(eigvals, 0.0)
    lam_max = eigvals[-1] + lam ** 2
    lam_min = eigvals[0] + lam ** 2
    return np.sqrt(lam_max / lam_min)


def relative_error(w, w_star):
    """Relative error ||w - w*|| / ||w*||."""
    norm_star = np.sqrt(np.dot(w_star, w_star))
    if norm_star < 1e-30:
        return np.sqrt(np.dot(w - w_star, w - w_star))
    return np.sqrt(np.dot(w - w_star, w - w_star)) / norm_star


def residual_norm(w, X, y, lam):
    """Residual norm ||hat_X w - hat_y|| / ||hat_y||."""
    X_hat, y_hat = build_augmented_system(X, y, lam)
    r = X_hat @ w - y_hat
    return np.sqrt(np.dot(r, r)) / np.sqrt(np.dot(y_hat, y_hat))