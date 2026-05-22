import math
import numpy as np


def acquisition(means, variances, best):
    """
    Standard Expected Improvement baseline.
    Uses variance (NOT std) to match FunBO evaluator.
    Returns index.
    """
    scores = []
    for m, v in zip(means, variances):
        sigma = math.sqrt(max(v, 1e-12))
        z = (best - m) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
        scores.append((best - m) * cdf + sigma * pdf)

    return int(np.argmax(scores))


def acquisition_ucb(means, variances, best, beta=2.0):
    """
    GP-UCB: Upper Confidence Bound (Srinivas et al., 2010).
    UCB(x) = mean(x) + beta * std(x)
    Higher beta -> more exploration.
    """
    stds = np.sqrt(np.maximum(variances, 1e-12))
    scores = means + beta * stds
    return int(np.argmax(scores))


def acquisition_pi(means, variances, best):
    """
    PI: Probability of Improvement (Kushner, 1964).
    PI(x) = CDF((mean(x) - best) / std(x))
    More exploitative than EI — prefers points likely to improve.
    """
    scores = []
    for m, v in zip(means, variances):
        sigma = math.sqrt(max(v, 1e-12))
        z = (m - best) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        scores.append(cdf)

    return int(np.argmax(scores))


BASE_EI_CODE = '''import math, numpy as np
def acquisition(means, variances, best):
    scores = []
    for m, v in zip(means, variances):
        sigma = math.sqrt(max(v, 1e-12))
        z = (best - m) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
        scores.append((best - m) * cdf + sigma * pdf)
    return int(np.argmax(scores))
'''

