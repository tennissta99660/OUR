import warnings
warnings.filterwarnings("ignore")
import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

DATA_DIR = r"D:\hckthon\our\funbo_fast"


def run_bo_paper(AF_fn, T=30, seed=0):
    """
    - Sobol grid
    - worst initial design
    - fixed GP hyperparams
    - acquisition(mean, variance, incumbent)
    """

    rng = np.random.RandomState(seed)

    acc   = np.load(os.path.join(DATA_DIR, "acc_res_sgd_caltech101.npy"))
    sobol = np.load(os.path.join(DATA_DIR, "sobol_cifar100_d3.npy"))

    N, d = sobol.shape

    true_acc = float(acc.max())
    true_idx = int(acc.argmax())

    # ---- worst initial design (paper) ----
    init_idx = int(acc.argmin())

    observed = [init_idx]
    values   = [float(acc[init_idx])]

    acc_t0 = values[0]
    found_acc = acc_t0
    Th = T

    kernel = RBF(length_scale=np.ones(d))

    for t in range(1, T):

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer=None,     # critical: fixed hyperparams
            normalize_y=True
        )

        gp.fit(sobol[observed], np.array(values))

        mu, std = gp.predict(sobol, return_std=True)
        var = std ** 2

        try:
            idx = int(AF_fn(mu, var, found_acc))
        except:
            idx = -1

        if idx in observed or idx < 0 or idx >= N:
            remaining = np.setdiff1d(np.arange(N), observed)
            idx = int(rng.choice(remaining))

        observed.append(idx)

        val = float(acc[idx])
        values.append(val)

        if val > found_acc:
            found_acc = val

        if idx == true_idx and Th == T:
            Th = t

    return found_acc, true_acc, acc_t0, Th
