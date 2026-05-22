import numpy as np


def paper_score(found_acc, true_acc, acc_t0, Th, T):
    """
    Implements Equation (1) exactly.
    """

    denom = true_acc - acc_t0

    if abs(denom) < 1e-12:
        term1 = 1.0 if abs(found_acc - true_acc) < 1e-9 else 0.0
    else:
        term1 = 1.0 - (true_acc - found_acc) / denom
        term1 = float(np.clip(term1, 0.0, 1.0))

    term2 = 1.0 - (Th / T)

    return term1 + term2
