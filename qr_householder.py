"""
================================================================================
qr_householder.py — A2: Thin QR Factorization with Householder Reflectors
================================================================================

Thin QR factorization of the augmented matrix:

    hat_X = [X^T; lambda*I_m]  ∈ R^{(n+m) x m}

Compact variant: matrix Q is NOT formed explicitly.
Householder vectors u_k are stored and used to apply
the products Q*v and Q^T*v implicitly.

Computational cost: O((n+m) * m^2), which is at most quadratic in m
when n = O(m) (as required by the assignment).

References:
    - Trefethen & Bau, "Numerical Linear Algebra", Lecture 10, 1997
    - Golub & Van Loan, "Matrix Computations", 4th ed., 2013

TODO: implement
================================================================================
"""

import numpy as np


def householder_vector(x):
    """
    Computes the Householder vector u such that H*x = ||x||*e_1,
    where H = I - 2*u*u^T/||u||^2.

    Parameters
    ---------
    x : ndarray (d,)  input vector

    Returns
    -------
    u : ndarray (d,)  Householder vector (normalized)
    s : float         -sign(x_0) * ||x||
    """
    # TODO: implement
    raise NotImplementedError("A2 not implemented yet")


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
    u_list   : list of ndarray          Householder vectors
    """
    # TODO: implement
    raise NotImplementedError("A2 not implemented yet")


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
    # TODO: implement
    raise NotImplementedError("A2 not implemented yet")


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
    # TODO: implement
    raise NotImplementedError("A2 not implemented yet")


def qr_solve(X, y, lam):
    """
    Solves min_w ||hat_X w - hat_y|| via QR factorization.

    Procedure:
    1. Build hat_X and hat_y
    2. QR factorization of hat_X  -> hat_X = Q * R
    3. Compute Q^T * hat_y  (implicit, via Householder)
    4. Solve R * w = (Q^T hat_y)[:m]  via back-substitution

    Total cost: O((n+m)*m^2) + O((n+m)*m) + O(m^2) = O((n+m)*m^2)

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
    # TODO: implement
    raise NotImplementedError("A2 not implemented yet")
