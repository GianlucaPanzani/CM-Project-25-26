"""
================================================================================
utils.py — Funzioni base per il problema (P)
================================================================================

Problema:
    min_w || hat_X w - hat_y ||

con:
    hat_X = [X^T; lambda*I_m]   ∈ R^{(n+m) x m}
    hat_y = [y; 0]              ∈ R^{n+m}

Equivalentemente:
    min_w  f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2

Matrici:
    X ∈ R^{m x n}   (tall-thin, dal dataset ML-CUP)
    y ∈ R^n          (vettore random)
    lambda > 0       (parametro di regolarizzazione)
    w ∈ R^m          (vettore da ottimizzare)

Hessiana (costante):
    H = nabla^2 f(w) = X X^T + lambda^2 I_m

Riferimenti:
    - Nocedal & Wright, "Numerical Optimization", 2006
    - Trefethen & Bau, "Numerical Linear Algebra", 1997
================================================================================
"""

import numpy as np
import pandas as pd
import time


# =============================================================================
# 1. FUNZIONE OBIETTIVO E GRADIENTE
# =============================================================================

def compute_loss(w, X, y, lam):
    """
    Calcola il valore della funzione obiettivo
    f(w) = (1/2)||X^T w - y||^2 + (1/2) lambda^2 ||w||^2.

    Parametri
    ---------
    w   : ndarray (m,)   vettore dei pesi
    X   : ndarray (m, n) matrice dei dati (tall-thin)
    y   : ndarray (n,)   vettore target
    lam : float          parametro di regolarizzazione (lambda > 0)

    Ritorna
    -------
    float  valore della funzione obiettivo
    """

    # Calcola il vettore residuo: (X^T * w) - y
    residual = X.T @ w - y

    # Misura quanto il modello si adatta bene ai dati (minimizza l'errore)
    # Termine di perdita dei dati (Data Loss):
    # 1/2 * ||X^T w - y||^2
    loss_data = 0.5 * np.dot(residual, residual)

    # Penalizza i pesi (w) troppo grandi per evitare l'overfitting e stabilizzare il problema
    # Termine di regolarizzazione (Regularization Loss):
    # 1/2 * lambda^2 * ||w||^2
    loss_reg = 0.5 * (lam ** 2) * np.dot(w, w)

    return loss_data + loss_reg


def compute_gradient(w, X, y, lam):
    """
    Calcola il vettore gradiente di f(w) rispetto a w.
    nabla f(w) = X(X^T w - y) + lambda^2 w.

    Parametri
    ---------
    w   : ndarray (m,)   vettore dei pesi
    X   : ndarray (m, n) matrice dei dati
    y   : ndarray (n,)   vettore target
    lam : float          parametro di regolarizzazione

    Ritorna
    -------
    ndarray (m,)  gradiente
    """

    # Calcola il vettore residuo: (X^T * w) - y     (dimensione n)
    residual = X.T @ w - y

    # Derivata del termine di perdita dei dati
    grad_data = X @ residual

    # Derivata del termine di regolarizzazione
    grad_reg = (lam ** 2) * w

    return grad_data + grad_reg


# =============================================================================
# 2. VERIFICA DEL GRADIENTE (differenze finite centrali)
# =============================================================================

def check_gradient(w, X, y, lam, eps=1e-7):
    """
    Verifica il gradiente analitico tramite differenze finite centrali.
    f'(x) approx (f(x + eps) - f(x - eps)) / (2 * eps)

    Ritorna
    -------
    abs_error : float  errore assoluto ||grad_approx - grad_an||
    rel_error : float  errore relativo ||grad_approx - grad_an|| / ||grad_an||
    """
    m = len(w)
    grad_approx = np.zeros(m)

    for i in range(m):
        # Perturbazione in avanti
        w_plus = w.copy()
        w_plus[i] += eps
        loss_plus = compute_loss(w_plus, X, y, lam)

        # Perturbazione all'indietro
        w_minus = w.copy()
        w_minus[i] -= eps
        loss_minus = compute_loss(w_minus, X, y, lam)

        # Derivata parziale i-esima approssimata
        grad_approx[i] = (loss_plus - loss_minus) / (2 * eps)

    # Calcolo il gradiente analitico
    grad_analytic = compute_gradient(w, X, y, lam)

    # Calcolo il vettore differenza tra il gradiente approssimato e quello esatto
    diff = grad_approx - grad_analytic

    # ERRORE ASSOLUTO: calcolo la norma L2 (distanza euclidea) della differenza.
    error = np.sqrt(np.dot(diff, diff))

    # Calcolo la norma L2 del gradiente analitico per usarla come termine di scala
    norm_an = np.sqrt(np.dot(grad_analytic, grad_analytic))

    # ERRORE RELATIVO: divido l'errore assoluto per la grandezza del gradiente analitico.
    # Uso max(norm_an, 1e-30) per garantire la stabilità numerica: se norm_an fosse 0,
    # il max interviene ed evita un crash del programma dovuto a una divisione per zero.
    return error, error / max(norm_an, 1e-30)


# =============================================================================
# 3. SOLUZIONE ESATTA (baseline via equazioni normali)
# =============================================================================

def solve_exact(X, y, lam):
    """
    Calcola la soluzione esatta w* risolvendo le equazioni normali:
        (X X^T + lambda^2 I) w = X y

    NOTA: usa np.linalg.solve solo come ground truth per la verifica.

    Ritorna
    -------
    w_baseline : ndarray (m,)  soluzione esatta
    f_baseline : float         valore ottimo f(w*)
    """

    m = X.shape[0]
    A = X @ X.T + (lam ** 2) * np.eye(m)
    b = X @ y

    start_time = time.time()

    w_baseline = np.linalg.solve(A, b)
    tempo_baseline = time.time() - start_time

    f_baseline = compute_loss(w_baseline, X, y, lam)

    return w_baseline, f_baseline, tempo_baseline


# =============================================================================
# 4. CARICAMENTO E PREPROCESSING DEL DATASET ML-CUP
# =============================================================================

def load_ml_cup(filepath, seed=42):
    """
    Carica il dataset ML-CUP e lo preprocessa.

    - Legge il CSV (saltando le righe di commento '#')
    - Estrae le feature (colonne 1:-4) come matrice X
    - Standardizza X (media 0, deviazione standard 1)
    - Genera un vettore y random di dimensione n

    Parametri
    ---------
    filepath : str   percorso al file ML-CUP25-TR.csv
    seed : int       seme per la generazione di y

    Ritorna
    -------
    X : ndarray (m, n)  matrice dei dati standardizzata
    y : ndarray (n,)    vettore target random
    m : int             numero di righe
    n : int             numero di colonne
    """

    df = pd.read_csv(filepath, comment='#', header=None)
    X_raw = df.iloc[:, 1:-4].values
    m, n = X_raw.shape

    # Standardizzazione
    X_mean = np.mean(X_raw, axis=0)
    X_std = np.std(X_raw, axis=0)
    X_std[X_std < 1e-12] = 1.0  # evita divisione per zero
    X = (X_raw - X_mean) / X_std

    # Genera y random
    np.random.seed(seed)
    y = np.random.randn(n)

    return X, y, m, n


# =============================================================================
# 5. COSTRUZIONE DEL SISTEMA AUMENTATO (per QR)
# =============================================================================

def build_augmented_system(X, y, lam):
    """
    Costruisce la matrice aumentata hat_X e il vettore aumentato hat_y.

        hat_X = [X^T; lambda*I_m]   ∈ R^{(n+m) x m}
        hat_y = [y; 0]              ∈ R^{n+m}

    Parametri
    ---------
    X   : ndarray (m, n)
    y   : ndarray (n,)
    lam : float

    Ritorna
    -------
    X_hat : ndarray (n+m, m)
    y_hat : ndarray (n+m,)
    """
    m, n = X.shape
    X_hat = np.vstack([X.T, lam * np.eye(m)])
    y_hat = np.concatenate([y, np.zeros(m)])
    return X_hat, y_hat


# =============================================================================
# 6. METRICHE DI VALUTAZIONE
# =============================================================================

def compute_condition_number(X, lam):
    """
    Calcola il condition number kappa(hat_X).

    kappa = sqrt((lambda_1 + lambda^2) / (lambda_m + lambda^2))

    dove lambda_1, lambda_m sono il massimo e minimo autovalore di XX^T.
    """
    eigvals = np.linalg.eigvalsh(X @ X.T)
    lam_max = eigvals[-1] + lam ** 2
    lam_min = eigvals[0] + lam ** 2
    return np.sqrt(lam_max / lam_min)


def relative_error(w, w_star):
    """Errore relativo ||w - w*|| / ||w*||."""
    norm_star = np.sqrt(np.dot(w_star, w_star))
    if norm_star < 1e-30:
        return np.sqrt(np.dot(w - w_star, w - w_star))
    return np.sqrt(np.dot(w - w_star, w - w_star)) / norm_star


def residual_norm(w, X, y, lam):
    """Norma del residuo ||hat_X w - hat_y|| / ||hat_y||."""
    X_hat, y_hat = build_augmented_system(X, y, lam)
    r = X_hat @ w - y_hat
    return np.sqrt(np.dot(r, r)) / np.sqrt(np.dot(y_hat, y_hat))