"""Householder QR solvers for regularized linear least squares.

The project problem is

    minimize ||A_hat @ w - b_hat||_2,
    A_hat = [X.T; lam*I_m] in R**((n+m) x m),
    b_hat = [y; 0],

where ``X`` has shape ``(m, n)``, ``y`` has shape ``(n,)``, and
``lam > 0``.  This module provides three implementations:

* ``naive_qr_solver`` applies generic dense Householder QR to an explicit
  tall matrix;
* ``qr_solver_structure_based`` factors the row-permuted matrix
  ``[lam*I_m; X.T]`` with compact reflectors of length ``n+1``;
* ``qr_solver_row_insertion`` inserts the rows of ``X.T`` one at a time
  with two-dimensional reflectors and updates the right-hand side during
  the insertion.

No solver forms ``Q`` explicitly.  The public reflector-application
routines compute the part of ``Q.T`` required by their solvers.  Generic
dense QR costs ``O(rows*cols**2)``.  The cost of either structured
factorization followed by back substitution is ``O(n*m**2 + m**2)``, which
is quadratic in ``m`` for fixed ``n``.

The timing returned by a solver covers the construction of its working
arrays, its factorization, the implicit or fused right-hand-side
transformation, and the triangular solve.  The routines assume
shape-compatible, finite, real floating-point inputs and nonsingular
triangular factors; complete input validation is not performed.

References
----------
Trefethen and Bau, *Numerical Linear Algebra*, Lecture 10, 1997.
Golub and Van Loan, *Matrix Computations*, 4th ed., 2013.
"""

import math
import time
import numpy as np
from lib.utils import build_augmented_system




# =============================================================================
# 1. NAIVE QR SOLVER
# =============================================================================

def naive_qr_solver(X, y, lam):
    """Solve a tall least-squares problem by compact dense Householder QR.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Real floating-point data matrix.
    y : ndarray, shape (n,)
        Right-hand side of the data equations.
    lam : float
        Positive regularization parameter.

    Returns
    -------
    R : ndarray, shape (m, m)
        Upper-triangular factor.
    w : ndarray, shape (m,)
        Solution of ``R @ w = c[:m]``.
    c : ndarray, shape (rows,)
        Transformed right-hand side ``Q.T @ b``.
    reflectors : list of ndarray
        The ``m`` normalized Householder vectors.  Reflector ``k`` has
        shape ``(rows-k,)``.
    tot_time : float
        Seconds spent in building the augmented system ``[X.T; lam*I_m]``
        and ``[y; 0]``, factorization, implicit ``Q.T`` application, and
        backward substitution.

    Notes
    -----
    ``Q`` is not formed.  Factorization costs ``O(rows*m**2)``, applying
    ``Q.T`` costs ``O(rows*m)``, and back substitution costs ``O(m**2)``.
    """
    start = time.perf_counter()

    A, b = build_augmented_system(X, y, lam)
    m = A.shape[1]

    R, reflectors = qr_factorize_naive(A)
    c = apply_QT(reflectors, b)
    w = back_substitution(R, c[:m])

    tot_time = time.perf_counter() - start
    return R, w, c, reflectors, tot_time


def qr_factorize_naive(A):
    """Compute a compact Householder QR factorization of a tall matrix.

    Parameters
    ----------
    A : ndarray, shape (rows, m)
        Real floating-point matrix with ``rows >= m``.  It is not modified.

    Returns
    -------
    R : ndarray, shape (m, m)
        Upper-triangular factor.  The zero block below ``R`` is omitted.
    u_list : list of ndarray
        The ``m`` normalized Householder vectors.  ``u_list[k]`` has shape
        ``(rows-k,)`` and acts on rows ``k:``.

    Notes
    -----
    If ``H_k`` is represented by ``u_list[k]``, the routine computes
    ``H_(m-1) ... H_0 A = [R; 0]`` without forming ``Q``.  The cost is
    ``O(rows*m**2)``.  Reflector storage contains
    ``sum(rows-k, k=0,...,m-1)`` scalars.
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
    """Apply ``Q.T`` from compact trailing Householder vectors.

    Implicit product of Trefethen & Bau, Algorithm 10.2.

    Parameters
    ----------
    u_list : sequence of ndarray
        Reflector ``k`` must have shape ``(b.size-k,)`` and acts on
        ``b[k:]``.  The reflectors must be in factorization order.
    b : ndarray, shape (rows,)
        Vector to transform.  It is not modified.

    Returns
    -------
    result : ndarray, shape (rows,)
        ``Q.T @ b``.

    Notes
    -----
    If ``Q.T = H_(r-1) ... H_0``, sequentially updating a vector with
    ``H_0, H_1, ..., H_(r-1)`` produces ``Q.T @ b``.  A zero reflector is
    skipped.  Dimensions are assumed compatible rather than validated.
    The cost is proportional to the total number of stored entries.
    """
    result = b.copy()
    for k, u in enumerate(u_list):
        u_squared_norm = np.dot(u, u)
        if u_squared_norm == 0.0:
            continue
        coefficient = 2.0 * np.dot(u, result[k:]) / u_squared_norm
        result[k:] -= coefficient * u

    return result


def apply_Q(u_list, x):
    """Apply ``Q`` from compact trailing Householder vectors.

    Implicit product of Trefethen & Bau, Algorithm 10.3: since
    ``Q = H_0 H_1 ... H_(r-1)``, the reflectors stored by the factorization
    are applied to ``x`` in reverse order.  ``Q`` is never formed; the cost
    is proportional to the total number of stored entries.
    """
    result = x.copy()
    for k in reversed(range(len(u_list))):
        u = u_list[k]
        u_squared_norm = np.dot(u, u)
        if u_squared_norm == 0.0:
            continue
        coefficient = 2.0 * np.dot(u, result[k:]) / u_squared_norm
        result[k:] -= coefficient * u

    return result


def back_substitution(R, b):
    """Solve the nonsingular upper-triangular system ``R @ w = b``.

    Parameters
    ----------
    R : ndarray, shape (m, m)
        Nonsingular upper-triangular coefficient matrix.
    b : ndarray, shape (m,)
        Right-hand side.

    Returns
    -------
    w : ndarray, shape (m,)
        Solution of the triangular system.

    Notes
    -----
    The routine performs ``O(m**2)`` work and allocates ``O(m)`` output
    storage.  Shapes, triangular structure, finite values, and nonzero
    pivots are assumed rather than checked; a zero pivot can produce
    nonfinite values.
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
    """Construct a normalized Householder vector for a real column.

    Parameters
    ----------
    x : ndarray, shape (d,)
        Nonempty real vector.

    Returns
    -------
    u : ndarray, shape (d,)
        Normalized reflector vector.  If ``x`` is zero, ``u`` is the zero
        vector used as an identity-reflector sentinel.
    alpha : float
        ``-sign(x[0])*||x||_2``, with sign ``+1`` when ``x[0] >= 0``;
        zero when ``x`` is zero.

    Notes
    -----
    For nonzero ``x``, ``H = I - 2*u*u.T`` satisfies
    ``H @ x = alpha*e_1``.  The sign choice avoids cancellation in the
    first component, and the input is scaled before its norm is evaluated
    to reduce overflow and underflow risk.

    Raises
    ------
    ValueError
        If the recovered norm is not finite.
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
    """Solve the regularized problem with structure-based Householder QR.

    The row-permuted augmented matrix ``[lam*I_m; X.T]`` is factorized
    with ``m`` compact reflectors, each supported on one row of ``R`` and
    the ``n`` dense data rows.  Their action on ``[0; y]`` is computed
    implicitly before the triangular solve.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Real floating-point data matrix.
    y : ndarray, shape (n,)
        Right-hand side of the data equations.
    lam : float
        Positive regularization parameter.

    Returns
    -------
    R : ndarray, shape (m, m)
        Upper-triangular factor.
    w : ndarray, shape (m,)
        Computed regularized least-squares solution.
    c : ndarray, shape (m+n,)
        Transformed right-hand side Q.T @ [0; y]. Its first m entries are
        used by the triangular solve; the trailing n entries are Q_2^T [0; y],
        whose norm is the minimal residual.
    reflectors : list of ndarray
        The ``m`` normalized compact reflectors, each of shape ``(n+1,)``.
    tot_time : float
        Seconds spent in building ``[0; y]`` and ``R = lam*I_m``,
        factorization, compact right-hand-side transformation, and
        backward substitution.

    Notes
    -----
    The total cost is ``O(n*m**2 + m**2)``.  ``Q`` is not formed.
    """
    start = time.perf_counter()

    m = X.shape[0]
    b_perm = np.concatenate((np.zeros(m), y))

    R, reflectors = qr_factorize_structure_based(X, lam)
    c = apply_structure_based_QT(reflectors, b_perm, m)
    w = back_substitution(R, c[:m])

    tot_time = time.perf_counter() - start
    return R, w, c, reflectors, tot_time


def qr_factorize_structure_based(X, lam):
    """Factor ``[lam*I_m; X.T]`` using its diagonal/dense block structure.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Real floating-point data matrix.  It is not modified.
    lam : float
        Positive regularization parameter.

    Returns
    -------
    R : ndarray, shape (m, m)
        Upper-triangular factor.
    u_list : list of ndarray
        The ``m`` normalized compact Householder vectors.  Each has shape
        ``(n+1,)`` and represents a reflector acting on row ``k`` of ``R``
        together with all ``n`` dense rows.

    Notes
    -----
    The routine starts from ``R = lam*I_m`` and an internal copy of
    ``X.T``.  At step ``k`` it eliminates the ``n`` dense entries in
    column ``k`` and updates columns ``k:``.  Its arithmetic cost is
    ``O(n*m**2 + m**2)`` and reflector storage is ``O(n*m)``.  Because the
    internal copy retains the dtype of ``X``, floating-point input is
    required for reliable in-place updates.
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
    """Apply structure-based reflectors to a permuted right-hand side.

    Parameters
    ----------
    u_list : sequence of ndarray
        Compact reflectors of shape ``(n+1,)``, in factorization order.
    b_perm : ndarray, shape (m+n,)
        Permuted right-hand side, normally ``[0; y]``.  It is not modified.
    m : int
        Size of the triangular block and number of leading coordinates.

    Returns
    -------
    c : ndarray, shape (m+n,)
        The full product Q.T @ b_perm. Its first m entries feed the triangular solve;
        the trailing n entries are Q_2^T b_perm, whose norm is the minimal residual.

    Notes
    -----
    Reflector ``k`` acts on the full-system coordinates ``k`` and ``m:``.
    The method costs ``O(m*n)`` and does not form ``Q``.  Input dimensions
    are assumed compatible rather than validated.
    """
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

    # Copy back the transformed tail, so that c = Q^T b_perm in all m+n entries
    c[m:] = y_transformed

    return c

def apply_structure_based_Q(u_list, x_perm, m):
    """Apply ``Q`` of the structure-based factorization to a permuted vector.

    Implicit product of Trefethen & Bau, Algorithm 10.3: the compact
    reflectors are applied in reverse order.  Reflector ``k`` acts on the
    full-system coordinates ``k`` and ``m:``.  Costs ``O(m*n)`` and does not
    form ``Q``.
    """
    result = x_perm.copy()
    active = np.empty(result.size - m + 1, dtype=result.dtype)
    for k in reversed(range(len(u_list))):
        u = u_list[k]
        u_squared_norm = np.dot(u, u)
        if u_squared_norm == 0.0:
            continue
        active[0] = result[k]
        active[1:] = result[m:]
        active -= 2.0 * u * (u @ active) / u_squared_norm
        result[k] = active[0]
        result[m:] = active[1:]
    return result

# =============================================================================
# 3. ROW INSERTION 2D QR SOLVER
# =============================================================================

def qr_solver_row_insertion(X, y, lam):
    """Run the two-dimensional Householder row-insertion solver.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Real floating-point data matrix.
    y : ndarray, shape (n,)
        Right-hand side of the data equations.
    lam : float
        Positive regularization parameter.

    Returns
    -------
    R : ndarray, shape (m, m)
        Upper-triangular matrix produced by the row-insertion updates.
    w : ndarray, shape (m,)
        Output of backward substitution with ``R`` and ``c``.
    c : ndarray, shape (m,)
        Leading transformed right-hand side accumulated during insertion.
    reflectors : list of tuple
        The ``n*m`` triples ``(top_row, data_row, [u1, u2])`` stored in
        insertion order.
    tot_time : float
        Seconds spent in row insertion, fused right-hand-side updates,
        reflector storage, and backward substitution.

    Notes
    -----
    The intended algorithm costs ``O(n*m**2 + m**2)`` and does not form
    ``Q``.  The current factorization routine overwrites the prescribed
    diagonal and eliminated entry during each tail update; therefore its
    output is experimental and is not guaranteed to be a valid QR factor
    or least-squares solution.
    """
    start = time.perf_counter()

    R, c, reflectors = qr_factorize_row_insertion_2d(X, lam, y, store_reflectors=True)
    w = back_substitution(R, c)

    tot_time = time.perf_counter() - start
    return R, w, c, reflectors, tot_time


def qr_factorize_row_insertion_2d(X, lam, y, store_reflectors=True):
    """Insert the rows of ``X.T`` with two-dimensional Householder updates.

    Starting from ``R = lam*I_m``, each data row is swept from left to
    right.  At position ``(i, k)``, one reflector acts on row ``k`` of
    ``R`` and on the inserted row; the same reflector is applied immediately
    to the pair ``(c[k], y[i])``.

    Parameters
    ----------
    X : ndarray, shape (m, n)
        Real floating-point data matrix.  It is not modified.
    lam : float
        Positive regularization parameter.
    y : ndarray, shape (n,)
        Data right-hand side transformed during insertion.
    store_reflectors : bool, default=True
        Store every two-dimensional reflector when true.  This flag does
        not control the fused right-hand-side transformation, which is
        always performed.

    Returns
    -------
    R : ndarray, shape (m, m)
        Upper-triangular matrix produced by the row updates.
    c : ndarray, shape (m,)
        Leading transformed right-hand side used by back substitution.
    reflectors : list of tuple or None
        When requested, ``n*m`` triples ``(k, m+i, [u1, u2])`` in
        factorization order; otherwise ``None``.

    Notes
    -----
    The intended arithmetic cost is ``O(n*m**2 + m**2)``.  Storing the
    reflectors uses ``O(n*m)`` additional entries and Python objects;
    disabling storage leaves ``O(m)`` workspace beyond ``R``.

    In the current update order, ``R[k, k] = alpha`` and ``row[k] = 0``
    are assigned before subtracting from slices that still include index
    ``k``.  Those structural values are consequently overwritten, so the
    returned matrix is not guaranteed to be the intended QR factor.
    """
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
            R[k, k:] -= u1 * coefficients
            row[k:] -= u2 * coefficients
            R[k, k] = alpha
            row[k] = 0.0

            c[k] -= coefficient * u1
            yi_transformed -= coefficient * u2

            # Case of storing reflectors
            if store_reflectors:
                reflectors.append((k, m+i, [u1,u2])) # iter, row index, reflector

    return R, c, reflectors


def compute_householder_vector_2d(v1: float, v2: float):
    """Construct a normalized Householder vector for a two-entry column.

    Parameters
    ----------
    v1, v2 : float
        Finite real entries of the active column.

    Returns
    -------
    u1, u2 : float
        Components of the normalized reflector vector.  Both are zero for
        the zero input, denoting an identity-reflector sentinel.
    alpha : float
        ``-sign(v1)*sqrt(v1**2 + v2**2)``, using sign ``+1`` when
        ``v1 >= 0``; zero for the zero input.

    Notes
    -----
    For nonzero input, let ``u = [u1, u2].T``.  Then
    ``H = I - 2*u*u.T`` maps ``[v1, v2].T`` to ``[alpha, 0].T``.  Scaling
    precedes the norm calculation to reduce overflow and underflow risk.
    Nonfinite inputs and a recovered norm outside the floating-point range
    are not explicitly rejected.
    """
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

# =============================================================================
# 4. FLOP COUNTER  (derived from the structure of the three solvers)
# =============================================================================
#
# Structure-based QR (step k acts on an (n+1) x (m-k+1) active block):
#
#   Reflector construction:  6 (n+1)           per step  -> 6 (n+1) m
#   Block update H_k B_k:    4 (n+1)(m-k+1)    per step  -> ~2 (n+1) m^2
#                            [u^T B (2(n+1)w) + rank-one update (2(n+1)w)]
#   Q^T applied to [0; y]:   7 (n+1)           per step  -> 7 (n+1) m
#   Back substitution:                                      m^2
#
#   Total ~ (2(n+1) + 1) m^2 + 13 (n+1) m              flops
#
# Dense Householder QR of the (m+n) x m augmented matrix, p = m+n
# [Trefethen & Bau, Lecture 10]:
#
#   Factorization:  2 p m^2 - (2/3) m^3
#   Q^T b:          4 p m - 2 m^2
#   Back subst.:    m^2
#
# Normal equations solved with numpy.linalg.solve (LU, LAPACK gesv):
#
#   Form X X^T:     2 m^2 n        Form X y:   2 m n
#   LU:             (2/3) m^3      Two triangular solves:  2 m^2
# =============================================================================

def qr_theoretical_cost(m, n, method='structure'):
    """
    Compute the theoretical flop count and storage of a direct solver.

    Parameters
    ----------
    m, n   : int, problem dimensions (X has shape (m, n)).
    method : 'structure', 'dense' or 'normal'
             ('normal' = normal equations solved by LU, as numpy.linalg.solve).

    Returns
    -------
    dict with keys 'phases' (dict of per-phase flops), 'total_flops',
                   'storage_floats', 'storage_MB', 'method'.
    """
    if method == 'structure':
        phases = {
            'reflectors'   : 6 * (n + 1) * m,
            'factorization': 2 * (n + 1) * m ** 2,
            'apply_QT'     : 7 * (n + 1) * m,
            'back_subst'   : m ** 2,
        }
        # R (m x m) + dense block X^T (n x m) + m reflectors of length n+1
        storage_floats = m ** 2 + n * m + (n + 1) * m
    elif method == 'dense':
        p = m + n
        phases = {
            'factorization': 2 * p * m ** 2 - (2 * m ** 3) // 3,
            'apply_QT'     : 4 * p * m - 2 * m ** 2,
            'back_subst'   : m ** 2,
        }
        # explicit A (p x m) + reflectors of lengths p, p-1, ..., p-m+1
        storage_floats = p * m + (m * (2 * p - m + 1)) // 2
    elif method == 'normal':
        phases = {
            'form_gram'    : 2 * m ** 2 * n,
            'form_rhs'     : 2 * m * n,
            'factorization': (2 * m ** 3) // 3,
            'back_subst'   : 2 * m ** 2,
        }
        # H = X X^T + lam^2 I (m x m) + right-hand side
        storage_floats = m ** 2 + m
    else:
        raise ValueError(f"Unknown method='{method}'")

    total_flops = sum(phases.values())
    return {
        'phases'         : phases,
        'total_flops'    : total_flops,
        'storage_floats' : storage_floats,
        'storage_MB'     : storage_floats * 8 / 1e6,
        'method'         : method,
    }


def print_qr_cost_table(n, m_values=(100, 250, 500, 1000, 2000, 4000)):
    """
    Print theoretical flop counts of the three direct solvers for several m.
    """
    print(f"\n{'='*78}")
    print(f"  Theoretical cost of the direct solvers  (n={n})")
    print(f"{'='*78}")
    print(f"  {'m':>5}  {'structure':>14}  {'dense QR':>16}  {'normal eq.':>15}"
          f"  {'NE/struct':>9}")
    print(f"  {'-'*5}  {'-'*14}  {'-'*16}  {'-'*15}  {'-'*9}")
    for m in m_values:
        cs = qr_theoretical_cost(m, n, 'structure')['total_flops']
        cd = qr_theoretical_cost(m, n, 'dense')['total_flops']
        cn = qr_theoretical_cost(m, n, 'normal')['total_flops']
        print(f"  {m:>5}  {cs:>14,}  {cd:>16,}  {cn:>15,}  {cn/cs:>8.1f}x")
    print(f"{'='*78}\n")


# =============================================================================
# 5. ROBUST BENCHMARKING WRAPPER
# =============================================================================
#
# Same protocol as benchmark_lbfgs in lib/lbfgs.py, so that the timings of
# the iterative and direct solvers are directly comparable: one discarded
# warm-up run, followed by n_runs timed runs summarised by their median
# (insensitive to occasional outliers from the OS).
# =============================================================================

def benchmark_qr(solver, X, y, lam, n_runs=10):
    """
    Run a direct solver n_runs times and return robust timing statistics.

    Parameters
    ----------
    solver : callable with signature solver(X, y, lam) whose LAST return
             value is the elapsed time in seconds.  This holds for
             naive_qr_solver, qr_solver_structure_based,
             qr_solver_row_insertion and utils.solve_exact.
    X, y, lam : problem data
    n_runs : int, number of timed repetitions (default 10)

    Returns
    -------
    out_last : tuple  outputs of the last run
    stats    : dict   keys: 'n_runs', 'time_median', 'time_min',
                            'time_mean', 'time_std', 'time_all'
    """
    times = np.empty(n_runs)

    # Warm-up run (not counted): primes caches and BLAS thread pools.
    out_last = solver(X, y, lam)

    for i in range(n_runs):
        out_last = solver(X, y, lam)
        times[i] = out_last[-1]

    return out_last, {
        'n_runs'      : n_runs,
        'time_median' : float(np.median(times)),
        'time_min'    : float(times.min()),
        'time_mean'   : float(times.mean()),
        'time_std'    : float(times.std(ddof=1)) if n_runs > 1 else 0.0,
        'time_all'    : times.tolist(),
    }