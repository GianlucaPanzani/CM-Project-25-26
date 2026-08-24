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

import math
import time
import numpy as np




# =============================================================================
# 1. NAIVE QR SOLVER
# =============================================================================

def naive_qr_solver(A, b):
    """Follow the compact dense QR algorithm of the report."""
    m = A.shape[1]

    start = time.perf_counter()

    R, reflectors = qr_factorize_naive(A)
    c = apply_QT(reflectors, b)
    w = back_substitution(R, c[:m]) # Rw = c

    tot_time = time.perf_counter() - start
    return R, w, c, reflectors, tot_time


def qr_factorize_naive(A):
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

    # Work on a copy so that the caller's matrix is not modified.
    m = A.shape[1]
    A_transformed = A.copy()
    u_list = []

    # Eliminate the entries below the diagonal one column at a time
    for k in range(m):
        u, s = compute_householder_vector(A_transformed[k:, k])

        # Apply reflectors to A_transformed
        A_transformed[k:, k:] -= 2.0 * np.outer(u, u @ A_transformed[k:, k:])
        A_transformed[k, k] = s
        A_transformed[k+1:, k] = 0.0

        u_list.append(u)

    R = np.triu(A_transformed[:m, :m])
    return R, u_list


def apply_QT(u_list, b):
    """
    Apply Q^T without forming Q.

    If the factorization stores H_0, ..., H_{r-1}, then
    Q^T = H_{r-1} ... H_0.  Consequently, Q^T uses the reflectors in
    factorization order, whereas Q uses them in reverse order.

    Parameters
    ---------
    u_list : list of ndarray   Householder vectors from factorization
    b      : ndarray           input vector

    Returns
    -------
    ndarray        requested orthogonal product
    """

    # Applying the stored order gives Q^T; reversing that order gives Q.
    result = b.copy()
    for k, u in enumerate(u_list):
        u_squared_norm = np.dot(u, u)
        if u_squared_norm == 0.0:
            continue
        coefficient = 2.0 * np.dot(u, result[k:]) / u_squared_norm
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
    size = R.shape[0]
    w = np.empty(size, dtype=float)

    # Solve from the last equation to the first one
    for i in range(size-1, -1, -1):

        # Dot product between non-zero R values and w_i computed in the prev iters
        v = np.dot(R[i, i+1:], w[i+1:])

        w[i] = (b[i] - v) / R[i,i]

    return w


def compute_householder_vector(x):
    """
    Computes the Householder vector u such that H*x = s*e_1,
    where H = I - 2*u*u^T/||u||^2.

    Parameters
    ---------
    x : ndarray (d,)  input vector

    Returns
    -------
    u : ndarray (d,)  normalized Householder vector; zero if x is zero
    alpha : float         -sign(x_0) * ||x||
    """

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
    alpha = -sign * norm_x
    u = scaled_x.copy()
    u[0] += sign * scaled_norm
    u /= np.linalg.norm(u)

    return u, alpha



# =============================================================================
# 2. STRUCTURE-BASED QR SOLVER
# =============================================================================

def qr_solver_structure_based(X, y, lam):
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
    tot_time : float         execution time (seconds)
    """
    m = X.shape[0]
    b_perm = np.concatenate((np.zeros(m), y))

    start = time.perf_counter()

    R, reflectors = qr_factorize_structure_based(X, lam)
    c = apply_structure_based_QT(reflectors, b_perm, m)
    w = back_substitution(R, c[:m]) # Rw = c

    tot_time = time.perf_counter() - start
    return R, w, c, reflectors, tot_time


def qr_factorize_structure_based(X, lam):
    """
    Factor [lam*I_m; X^T] while exploiting the zeros in the identity block.

    Each stored vector contains only its nonzero entries: one entry for the
    active row of R and n entries for the dense data rows.
    """
    m, n = X.shape
    R = lam * np.eye(m)
    X_T = X.T.copy()
    u_list = []

    # Each reflector touches one row of R and the n dense data rows.
    for k in range(m):
        # Create the current active block: first row from R, remaining rows from X.T
        active_block = np.vstack((
            R[k, k:],
            X_T[:, k:]
        ))

        # Householder vector of the first column
        u, s = compute_householder_vector(active_block[:, 0])

        # Apply H = I - 2uu^T to the entire active block
        active_block -= 2.0 * np.outer(u, u @ active_block)

        # Record the exact triangular structure (for the first column)
        active_block[0, 0] = s
        active_block[1:, 0] = 0.0

        # Update R and X_T with the transformed block
        R[k, k:] = active_block[0, :]
        X_T[:, k:] = active_block[1:, :]

        u_list.append(u)

    return R, u_list


def apply_structure_based_QT(u_list, b_perm, m):
    """Apply the compact reflectors for [lam*I_m; X^T] to b_perm."""
    c = b_perm.copy()
    y_transformed = c[m:]

    active_vector = np.empty(y_transformed.size + 1, dtype=c.dtype)

    # Apply the reflectors to b_perm getting c
    for k, u in enumerate(u_list):
        u_squared_norm = np.dot(u, u)
        if u_squared_norm == 0.0:
            continue
        
        # Build the n+1 size vector to which apply the reflectors
        active_vector[0] = c[k]
        active_vector[1:] = y_transformed

        # Apply the reflectors to the active_vector
        active_vector -= (2.0 * u * (u @ active_vector) / u_squared_norm)

        # Update the modified k-th element of b_perm
        c[k] = active_vector[0]
        y_transformed = active_vector[1:]

    return c



# =============================================================================
# 3. ROW INSERTION 2D QR SOLVER
# =============================================================================

def qr_solver_row_insertion(X, y, lam):
    """Solve the project problem with the report's structured QR algorithm."""
    start = time.perf_counter()

    R, c, reflectors = qr_factorize_row_insertion_2d(X, lam, y, store_reflectors=True)
    w = back_substitution(R, c)

    tot_time = time.perf_counter() - start
    return R, w, c, reflectors, tot_time


def qr_factorize_row_insertion_2d(X, lam, y, store_reflectors=True):
    """Run row insertion, optionally transforming its right-hand side."""
    m, n = X.shape
    R = lam * np.eye(m)
    c = np.zeros(m)
    reflectors = [] if store_reflectors else None

    # Insert the n rows of X.T one at a time below the triangular factor.
    for i in range(n):
        row = X.T[i, :].copy()
        yi_transformed = y[i]

        # Eliminate the inserted row from left to right.
        for k in range(m):
            u1, u2, alpha = compute_householder_vector_2d(R[k, k], row[k])
            
            # Apply the reflectors to compute the coefficients (used to update the data)
            coefficients = 2.0 * (u1 * R[k, k:] + u2 * row[k:])
            coefficient = 2.0 * (u1 * c[k] + u2 * yi_transformed)

            # Update the triangular matrix, the row of X.T and the b_perm
            R[k, k] = alpha
            R[k, k:] -= u1 * coefficients
            row[k] = 0.0
            row[k:] -= u2 * coefficients

            c[k] -= coefficient * u1
            yi_transformed -= coefficient * u2

            # Case of storing reflectors
            if store_reflectors:
                reflectors.append((k, m+i, [u1,u2])) # iter, row index, reflector

    return R, c, reflectors


def compute_householder_vector_2d(v1: float, v2: float):
    """Return the normalized Householder vector for a two-entry column."""
    scale = max(abs(v1), abs(v2))
    if scale == 0.0:
        return 0.0, 0.0, 0.0

    # Scale to avoid overflow when computing the norm
    scaled_v1 = v1 / scale
    scaled_v2 = v2 / scale
    scaled_norm = math.sqrt(scaled_v1**2 + scaled_v2**2)

    vector_norm = scale * scaled_norm
    sign_v1 = 1.0 if v1 >= 0.0 else -1.0
    alpha = -sign_v1 * vector_norm

    # Compute v = x - alpha * e1 (but as the scaled version)
    reflector_v1 = scaled_v1 + sign_v1 * scaled_norm
    reflector_v2 = scaled_v2
    reflector_norm = math.sqrt(reflector_v1 * reflector_v1 + reflector_v2 * reflector_v2)

    u2 = reflector_v2 / reflector_norm
    u1 = reflector_v1 / reflector_norm

    return u1, u2, alpha
