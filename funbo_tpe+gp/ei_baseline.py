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
