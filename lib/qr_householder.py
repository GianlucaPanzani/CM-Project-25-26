"""
================================================================================
qr_householder.py — A2: Thin QR Factorization with Householder Reflectors
================================================================================

Thin QR factorization of the augmented matrix:

    hat_X = [X^T; lambda*I_m]  ∈ R^{(n+m) x m}

Compact variant: matrix Q is NOT formed explicitly.
Householder vectors u_k are stored and used to apply
the products Q*v and Q^T*v implicitly.

The generic factorization costs O(rows * cols^2).  The regularized
least-squares solver exploits the diagonal lambda*I_m block and costs
O(n*m^2 + m^2), which is quadratic in m when n is fixed.

References:
    - Trefethen & Bau, "Numerical Linear Algebra", Lecture 10, 1997
    - Golub & Van Loan, "Matrix Computations", 4th ed., 2013
================================================================================
"""

import time

import numpy as np


def _as_real_array(value, name):
    """Return a finite, real-valued NumPy array with floating-point entries."""
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise TypeError(f"{name} must contain real values")

    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def householder_vector(x):
    """
    Computes the Householder vector u such that H*x = s*e_1,
    where H = I - 2*u*u^T/||u||^2.

    Parameters
    ---------
    x : ndarray (d,)  input vector

    Returns
    -------
    u : ndarray (d,)  normalized Householder vector; zero if x is zero
    s : float         -sign(x_0) * ||x||
    """
    x = _as_real_array(x, "x")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("x must be a non-empty one-dimensional array")

    # Scale first so the norm calculation is robust to very large or small values.
    scale = np.max(np.abs(x))
    if scale == 0.0:
        return np.zeros_like(x), 0.0

    scaled_x = x / scale
    scaled_norm = np.linalg.norm(scaled_x)
    norm_x = scale * scaled_norm
    if not np.isfinite(norm_x):
        raise ValueError("the norm of x exceeds the floating-point range")

    # This sign choice avoids cancellation in the first component of u.
    sign = 1.0 if x[0] >= 0.0 else -1.0
    s = -sign * norm_x
    u = scaled_x.copy()
    u[0] += sign * scaled_norm
    u /= np.linalg.norm(u)

    return u, float(s)


def qr_factorize(A):
    """
    Thin QR factorization via Householder reflectors.
    Compact form: stores vectors u_k and does not form Q.

    Parameters
    ---------
    A : ndarray (rows, cols)  input matrix, rows >= cols

    Returns
    -------
    R        : ndarray (cols, cols)     upper triangular matrix R
    u_list   : list of ndarray          normalized vectors, with
                                          u_list[k].shape == (rows-k,)
    """
    A = _as_real_array(A, "A")
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional array")

    rows, cols = A.shape
    if rows < cols:
        raise ValueError("A must have at least as many rows as columns")

    # Work on a copy so that the caller's matrix is not modified.
    transformed = A.copy()
    u_list = []

    # Eliminate the entries below the diagonal one column at a time.
    for k in range(cols):
        u, s = householder_vector(transformed[k:, k])
        trailing_block = transformed[k:, k:]
        trailing_block -= 2.0 * np.outer(u, u @ trailing_block)

        # Store exact structural zeros instead of round-off-sized residuals.
        transformed[k, k] = s
        transformed[k + 1:, k] = 0.0
        u_list.append(u)

    R = np.triu(transformed[:cols, :cols])
    return R, u_list


def apply_QT(u_list, b, rows):
    """
    Computes Q^T * b without forming Q, using Householder vectors.

    Parameters
    ---------
    u_list : list of ndarray   Householder vectors from factorization
    b      : ndarray (rows,)   input vector
    rows   : int               number of rows of the original matrix

    Returns
    -------
    ndarray (rows,)   product Q^T * b
    """
    if not isinstance(rows, (int, np.integer)) or rows < 0:
        raise ValueError("rows must be a non-negative integer")

    b = _as_real_array(b, "b")
    if b.shape != (rows,):
        raise ValueError(f"b must have shape ({rows},)")

    try:
        reflectors = list(u_list)
    except TypeError as exc:
        raise TypeError("u_list must be an iterable of Householder vectors") from exc
    if len(reflectors) > rows:
        raise ValueError("u_list contains more reflectors than rows")

    # Apply H_0, H_1, ... H_k so the final result is H_k ... H_1 H_0 b = Q^T b.
    result = b.copy()
    for k, stored_u in enumerate(reflectors):
        u = _as_real_array(stored_u, f"u_list[{k}]")
        expected_shape = (rows - k,)
        if u.shape != expected_shape:
            raise ValueError(
                f"u_list[{k}] must have shape {expected_shape}, got {u.shape}"
            )

        squared_norm = np.dot(u, u)
        if squared_norm == 0.0:
            continue
        coefficient = 2.0 * np.dot(u, result[k:]) / squared_norm
        result[k:] -= coefficient * u

    return result


def back_substitution(R, b):
    """
    Solves the upper triangular system R*w = b.

    Parameters
    ---------
    R : ndarray (m, m)  upper triangular matrix
    b : ndarray (m,)    right-hand side vector

    Returns
    -------
    w : ndarray (m,)  solution
    """
    R = _as_real_array(R, "R")
    b = _as_real_array(b, "b")
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError("R must be a square two-dimensional array")

    size = R.shape[0]
    if b.shape != (size,):
        raise ValueError(f"b must have shape ({size},)")

    # Solve from the last equation to the first one.
    w = np.empty(size, dtype=float)
    for i in range(size - 1, -1, -1):
        pivot = R[i, i]
        if pivot == 0.0:
            raise np.linalg.LinAlgError(
                f"R is singular: zero diagonal entry at index {i}"
            )
        known_terms = np.dot(R[i, i + 1:], w[i + 1:])
        w[i] = (b[i] - known_terms) / pivot

    return w


def _factorize_augmented_system(X, lam):
    """
    Factor [lam*I_m; X^T] while exploiting the zeros in the identity block.

    Each stored vector contains only its nonzero entries: one entry for the
    active row of R and n entries for the dense data rows.
    """
    m, n = X.shape
    R = lam * np.eye(m)
    data_rows = X.T.copy()
    u_list = []

    # Each reflector touches one row of R and the n dense data rows.
    for k in range(m):
        active_column = np.concatenate(([R[k, k]], data_rows[:, k]))
        compact_u, s = householder_vector(active_column)

        r_tail = R[k, k:]
        data_block = data_rows[:, k:]
        projection = 2.0 * (
            compact_u[0] * r_tail + compact_u[1:] @ data_block
        )
        r_tail -= compact_u[0] * projection
        data_block -= np.outer(compact_u[1:], projection)

        # Record the triangular structure exactly.
        R[k, k] = s
        data_rows[:, k] = 0.0

        u_list.append(compact_u)

    return R, u_list


def _apply_augmented_QT(u_list, b, m):
    """Apply the compact reflectors for [lam*I_m; X^T] to b."""
    result = b.copy()
    data_part = result[m:]

    # Reflector k acts only on result[k] and the final n components.
    for k, u in enumerate(u_list):
        squared_norm = np.dot(u, u)
        if squared_norm == 0.0:
            continue
        coefficient = 2.0 * (
            u[0] * result[k] + np.dot(u[1:], data_part)
        ) / squared_norm
        result[k] -= coefficient * u[0]
        data_part -= coefficient * u[1:]

    return result


def qr_solve(X, y, lam):
    """
    Solves min_w ||hat_X w - hat_y|| via QR factorization.

    Procedure:
    1. Permute the augmented system to [lambda*I_m; X^T]
    2. Factor it while exploiting the diagonal first block
    3. Compute Q^T * [0; y] implicitly via stored reflectors
    4. Solve R*w = (Q^T*[0; y])[:m] by back-substitution

    Total cost: O(n*m^2 + m^2), which is quadratic in m for fixed n.

    Parameters
    ---------
    X   : ndarray (m, n)  data matrix
    y   : ndarray (n,)    target vector
    lam : float           regularization parameter

    Returns
    -------
    w       : ndarray (m,)  solution
    elapsed : float         execution time (seconds)
    """
    X = _as_real_array(X, "X")
    y = _as_real_array(y, "y")
    lam_array = _as_real_array(lam, "lam")

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if y.shape != (X.shape[1],):
        raise ValueError(f"y must have shape ({X.shape[1]},)")
    if lam_array.ndim != 0:
        raise ValueError("lam must be a scalar")

    lam = float(lam_array)
    if lam <= 0.0:
        raise ValueError("lam must be strictly positive")

    m, n = X.shape

    # Time the factorization, the implicit Q^T product, and the triangular solve.
    start = time.perf_counter()
    R, u_list = _factorize_augmented_system(X, lam)
    augmented_y = np.concatenate((np.zeros(m), y))
    transformed_y = _apply_augmented_QT(u_list, augmented_y, m)
    w = back_substitution(R, transformed_y[:m])
    elapsed = time.perf_counter() - start

    return w, elapsed
