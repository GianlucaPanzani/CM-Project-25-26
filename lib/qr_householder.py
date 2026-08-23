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

import lib.utils as utils
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


def _validate_matrix_and_lambda(X, lam):
    """Validate the data matrix and the positive regularization parameter."""
    X = _as_real_array(X, "X")
    lam_array = _as_real_array(lam, "lam")

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if lam_array.ndim != 0:
        raise ValueError("lam must be a scalar")

    lam = float(lam_array)
    if lam <= 0.0:
        raise ValueError("lam must be strictly positive")
    return X, lam


def _validate_regularized_problem(X, y, lam):
    """Validate a complete regularized least-squares problem."""
    X, lam = _validate_matrix_and_lambda(X, lam)
    y = _as_real_array(y, "y")
    if y.shape != (X.shape[1],):
        raise ValueError(f"y must have shape ({X.shape[1]},)")
    return X, y, lam


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
    Apply Q, or Q^T when ``transpose=True``, without forming Q.

    If the factorization stores H_0, ..., H_{r-1}, then
    Q^T = H_{r-1} ... H_0.  Consequently, Q^T uses the reflectors in
    factorization order, whereas Q uses them in reverse order.

    Parameters
    ---------
    u_list : list of ndarray   Householder vectors from factorization
    b      : ndarray (rows,)   input vector
    rows   : int               number of rows of the original matrix

    Returns
    -------
    ndarray (rows,)   requested orthogonal product
    """

    # Validate every reflector before modifying the result.
    checked_reflectors = []
    for k, stored_u in enumerate(u_list):
        u = _as_real_array(stored_u, f"u_list[{k}]")
        expected_shape = (rows - k,)
        if u.shape != expected_shape:
            raise ValueError(
                f"u_list[{k}] must have shape {expected_shape}, got {u.shape}"
            )
        checked_reflectors.append((k, u))

    # Applying the stored order gives Q^T; reversing that order gives Q.
    result = b.copy()
    for k, u in checked_reflectors:
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


def _householder_vector_2d(first, second):
    """Return the normalized Householder vector for a two-entry column."""
    first = float(first)
    second = float(second)
    if not math.isfinite(first) or not math.isfinite(second):
        raise ValueError("x must contain only finite values")

    scale = max(abs(first), abs(second))
    if scale == 0.0:
        return 0.0, 0.0, 0.0

    scaled_first = first / scale
    scaled_second = second / scale
    scaled_norm = math.sqrt(scaled_first**2 + scaled_second**2)
    norm = scale * scaled_norm
    if not math.isfinite(norm):
        raise ValueError("the norm of x exceeds the floating-point range")

    sign = 1.0 if first >= 0.0 else -1.0
    u_first = scaled_first + sign * scaled_norm
    u_second = scaled_second
    u_norm = math.sqrt(u_first * u_first + u_second * u_second)
    return u_first / u_norm, u_second / u_norm, -sign * norm


def _factorize_row_insertions(X, lam, y=None, store_reflectors=True):
    """Run row insertion, optionally transforming its right-hand side."""
    m, n = X.shape
    R = lam * np.eye(m)
    c = None if y is None else np.zeros(m)
    reflectors = [] if store_reflectors else None

    # Insert the n rows of X.T one at a time below the triangular factor.
    for i in range(n):
        inserted_row = X[:, i].copy()
        data_row_index = m + i if store_reflectors else None
        data_rhs = None if y is None else float(y[i])

        # Eliminate the inserted row from left to right.
        for k in range(m):
            u_1, u_2, diagonal_value = _householder_vector_2d(R[k, k], inserted_row[k])

            if u_1 != 0.0 or u_2 != 0.0:
                r_tail = R[k, k:]
                inserted_tail = inserted_row[k:]
                projection = 2.0 * (u_1 * r_tail + u_2 * inserted_tail)
                r_tail -= u_1 * projection
                inserted_tail -= u_2 * projection

                # Applying the reflector now is equivalent to a later Q^T scan.
                if c is not None:
                    top_value = c[k]
                    u_squared_norm = u_1 * u_1 + u_2 * u_2
                    coefficient = 2.0 * (u_1 * top_value + u_2 * data_rhs) / u_squared_norm
                    c[k] -= coefficient * u_1
                    data_rhs -= coefficient * u_2

            # Record the triangular structure exactly.
            R[k, k] = diagonal_value
            inserted_row[k] = 0.0
            if store_reflectors:
                u = np.array([u_1, u_2])
                reflectors.append((k, data_row_index, u))

    return R, reflectors, c


def qr_factorize_row_insertion(X, lam):
    """
    Factor ``[lam*I_m; X.T]`` with the report's row-insertion algorithm.

    Each data row is inserted below the current triangular factor.  A sequence
    of two-dimensional Householder reflectors removes that row from left to
    right.  A stored reflector is represented by ``(top_row, data_row, u)``.

    Parameters
    ----------
    X   : ndarray (m, n)  data matrix
    lam : float           strictly positive regularization parameter

    Returns
    -------
    R          : ndarray (m, m)  upper triangular factor
    reflectors : list             embedded two-dimensional reflectors
    """
    X, lam = _validate_matrix_and_lambda(X, lam)
    R, reflectors, _ = _factorize_row_insertions(X, lam)
    return R, reflectors


def apply_row_insertion_Q(reflectors, b, transpose=False):
    """Apply Q, or Q^T, from ``qr_factorize_row_insertion`` reflectors."""
    if not isinstance(transpose, (bool, np.bool_)):
        raise TypeError("transpose must be a Boolean")

    b = _as_real_array(b, "b")
    if b.ndim != 1:
        raise ValueError("b must be a one-dimensional array")

    try:
        stored_reflectors = list(reflectors)
    except TypeError as exc:
        raise TypeError("reflectors must be an iterable") from exc

    # Validate the embedded row indices and vectors before applying them.
    checked_reflectors = []
    for j, stored_reflector in enumerate(stored_reflectors):
        try:
            top_row, data_row, stored_u = stored_reflector
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"reflectors[{j}] must be a (top_row, data_row, u) triple"
            ) from exc

        valid_indices = all(
            isinstance(index, (int, np.integer))
            and not isinstance(index, (bool, np.bool_))
            for index in (top_row, data_row)
        )
        if not valid_indices:
            raise TypeError("reflector row indices must be integers")
        if not (0 <= top_row < b.size and 0 <= data_row < b.size):
            raise ValueError(f"reflectors[{j}] contains an invalid row index")
        if top_row == data_row:
            raise ValueError("a reflector must act on two different rows")

        u = _as_real_array(stored_u, f"reflectors[{j}].u")
        if u.shape != (2,):
            raise ValueError(f"reflectors[{j}].u must have shape (2,)")
        checked_reflectors.append((top_row, data_row, u))

    application_order = (
        checked_reflectors if transpose else reversed(checked_reflectors)
    )
    result = b.copy()

    # Each reflector changes only its two embedded coordinates.
    for top_row, data_row, u in application_order:
        active_values = np.array([result[top_row], result[data_row]])
        squared_norm = np.dot(u, u)
        if squared_norm == 0.0:
            continue
        coefficient = 2.0 * np.dot(u, active_values) / squared_norm
        active_values -= coefficient * u
        result[top_row], result[data_row] = active_values

    return result


def _factorize_augmented_system(X, lam):
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
        u, s = householder_vector(active_block[:, 0])

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


def apply_augmented_QT(u_list, b_perm, m):
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
    X, y, lam = _validate_regularized_problem(X, y, lam)
    m, n = X.shape

    start = time.perf_counter()

    # From A_perm (implicitly made by X and lam) we get R and the householder reflectors
    R, u_list = _factorize_augmented_system(X, lam)
    b_perm = np.concatenate((np.zeros(m), y))
    c = apply_augmented_QT(u_list, b_perm, m)
    w = back_substitution(R, c[:m]) # Rw = c

    end = time.perf_counter()
    elapsed = end - start

    return w, elapsed

def dense_qr_solve(X_value, y_value, lam_value):
    """Follow the compact dense QR algorithm from the report."""
    A, b = utils.build_augmented_system(X_value, y_value, lam_value)

    # Compute R and b transformed by the application of the householder vectors (reflectors)
    R, reflectors = qr_factorize(A)
    c = apply_QT(reflectors, b, A.shape[0])
    w = back_substitution(R, c[:X_value.shape[0]]) # Rw = c
    return w, R, reflectors

def qr_solve_row_insertion(X, y, lam):
    """Solve the project problem with the report's structured QR algorithm."""
    X, y, lam = _validate_regularized_problem(X, y, lam)

    # Transform the right-hand side during factorization.  This preserves the
    # row-insertion order without storing and scanning all n*m reflectors.
    start = time.perf_counter()
    R, _, c = _factorize_row_insertions(X, lam, y=y, store_reflectors=False)
    w = back_substitution(R, c)
    elapsed = time.perf_counter() - start

    return w, elapsed
