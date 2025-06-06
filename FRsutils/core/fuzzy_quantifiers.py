import numpy as np

def fuzzy_quantifier_linear(p, alpha, beta):
    """
    Compute the degree of membership to a fuzzy quantifier.

    Parameters:
    - p : float or np.array
        Proportion(s) in the [0, 1] range.
    - alpha : float
        Lower threshold (start of transition).
    - beta : float
        Upper threshold (full membership).

    Returns:
    - float or np.array
        Degree(s) of truth for the fuzzy quantifier.
    """
    # raise ValueError("This function is not implemented yet.")
    p = np.asarray(p)


    # For quantifiers like "most"
    return np.where(p <= alpha, 0,
            np.where(p >= beta, 1,
                    (p - alpha) / (beta - alpha)))



def fuzzy_quantifier_quadratic(x, alpha, beta):
    """
    Smooth parameterized fuzzy quantifier using quadratic transition.

    Parameters:
    - x: float or np.array
        Input value(s), typically in [0, 1].
    - alpha: float
        Start of the transition (0 <= alpha < beta <= 1).
    - beta: float
        End of the transition.

    Returns:
    - float or np.array: Degree of membership.
    """
    x = np.asarray(x)
    mid = (alpha + beta) / 2
    denom = (beta - alpha) ** 2

    result = np.zeros_like(x)

    # Region 1: x <= alpha → Q = 0
    mask1 = x <= alpha

    # Region 2: alpha < x <= (alpha + beta)/2
    mask2 = (x > alpha) & (x <= mid)
    result[mask2] = 2 * ((x[mask2] - alpha) ** 2) / denom

    # Region 3: (alpha + beta)/2 < x <= beta
    mask3 = (x > mid) & (x <= beta)
    result[mask3] = 1 - 2 * ((x[mask3] - beta) ** 2) / denom

    # Region 4: x > beta → Q = 1
    mask4 = x > beta
    result[mask4] = 1

    return result
