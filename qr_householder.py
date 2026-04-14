"""
================================================================================
qr_householder.py — A2: Thin QR Factorization con Householder Reflectors
================================================================================

Fattorizzazione QR thin della matrice aumentata:

    hat_X = [X^T; lambda*I_m]  ∈ R^{(n+m) x m}

Variante compatta: NON si forma la matrice Q esplicitamente.
Si memorizzano i vettori di Householder u_k e si usano per applicare
implicitamente i prodotti Q*v e Q^T*v.

Costo computazionale: O((n+m) * m^2), che è al più quadratico in m
quando n = O(m) (come richiesto dalla consegna).

Riferimenti:
    - Trefethen & Bau, "Numerical Linear Algebra", Lecture 10, 1997
    - Golub & Van Loan, "Matrix Computations", 4th ed., 2013

TODO: implementare
================================================================================
"""

import numpy as np


def householder_vector(x):
    """
    Calcola il vettore di Householder u tale che H*x = ||x||*e_1,
    dove H = I - 2*u*u^T/||u||^2.

    Parametri
    ---------
    x : ndarray (d,)  vettore di input

    Ritorna
    -------
    u : ndarray (d,)  vettore di Householder (normalizzato)
    s : float         -sign(x_0) * ||x||
    """
    # TODO: implementare
    raise NotImplementedError("A2 non ancora implementato")


def qr_factorize(A):
    """
    Fattorizzazione QR thin tramite riflettori di Householder.
    Forma compatta: memorizza i vettori u_k, non forma Q.

    Parametri
    ---------
    A : ndarray (rows, cols)  matrice di input, rows >= cols

    Ritorna
    -------
    R        : ndarray (cols, cols)     matrice R triangolare superiore
    u_list   : list of ndarray          vettori di Householder
    """
    # TODO: implementare
    raise NotImplementedError("A2 non ancora implementato")


def apply_QT(u_list, b, rows):
    """
    Calcola Q^T * b senza formare Q, usando i vettori di Householder.

    Parametri
    ---------
    u_list : list of ndarray   vettori di Householder dalla fattorizzazione
    b      : ndarray (rows,)   vettore di input
    rows   : int               numero di righe della matrice originale

    Ritorna
    -------
    ndarray (rows,)   prodotto Q^T * b
    """
    # TODO: implementare
    raise NotImplementedError("A2 non ancora implementato")


def back_substitution(R, b):
    """
    Risolve il sistema triangolare superiore R*w = b.

    Parametri
    ---------
    R : ndarray (m, m)  matrice triangolare superiore
    b : ndarray (m,)    vettore dei termini noti

    Ritorna
    -------
    w : ndarray (m,)  soluzione
    """
    # TODO: implementare
    raise NotImplementedError("A2 non ancora implementato")


def qr_solve(X, y, lam):
    """
    Risolve min_w ||hat_X w - hat_y|| tramite fattorizzazione QR.

    Procedura:
    1. Costruisci hat_X e hat_y
    2. QR factorization di hat_X  -> hat_X = Q * R
    3. Calcola Q^T * hat_y  (implicito, tramite Householder)
    4. Risolvi R * w = (Q^T hat_y)[:m]  tramite back-substitution

    Costo totale: O((n+m)*m^2) + O((n+m)*m) + O(m^2) = O((n+m)*m^2)

    Parametri
    ---------
    X   : ndarray (m, n)  matrice dei dati
    y   : ndarray (n,)    vettore target
    lam : float           parametro di regolarizzazione

    Ritorna
    -------
    w       : ndarray (m,)  soluzione
    elapsed : float         tempo di esecuzione (secondi)
    """
    # TODO: implementare
    raise NotImplementedError("A2 non ancora implementato")
