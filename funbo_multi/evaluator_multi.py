"""
Multi-dataset BO evaluator.

Runs the same BO loop (GP + LLM-evolved AF) on multiple datasets,
returning per-dataset results so the caller can compute paper_score
for each and average them — exactly as FunBO's original multi-task
scoring does.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# ──────────────────────────────────────────────────────────────
# Dataset registry
# Each entry: (sobol_path, acc_path)
# ──────────────────────────────────────────────────────────────
BASE = r"D:\hckthon\our\funbo_fast"

SOBOL_PATH = os.path.join(BASE, "sgd", "sobol_res_sgd.npy")   # shared 1024×3 grid

DATASETS = {
    "cifar10": {
        "sobol": SOBOL_PATH,
        "acc":   os.path.join(BASE, "sgd", "acc_res_sgd.npy"),
    },
    "cifar100": {
        "sobol": SOBOL_PATH,
        "acc":   os.path.join(BASE, "acc_res_sgd_cifar100.npy"),
    },
    "caltech101": {
        "sobol": SOBOL_PATH,
        "acc":   os.path.join(BASE, "acc_res_sgd_caltech101.npy"),
    },
}


def run_bo_single(AF_fn, sobol, acc, T=120, seed=0):
    """
    Run one BO loop on a SINGLE dataset.

    Returns (found_acc, true_acc, acc_t0, Th) — the 4 values
    needed by paper_score().
    """
    rng = np.random.RandomState(seed)
    N, d = sobol.shape

    true_acc = float(acc.max())
    true_idx = int(acc.argmax())

    # worst initial design (paper protocol)
    init_idx = int(acc.argmin())
    observed = [init_idx]
    values   = [float(acc[init_idx])]

    acc_t0    = values[0]
    found_acc = acc_t0
    Th        = T

    kernel = RBF(length_scale=np.ones(d))

    for t in range(1, T):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer=None,
            normalize_y=True,
        )
        gp.fit(sobol[observed], np.array(values))
        mu, std = gp.predict(sobol, return_std=True)
        var = std ** 2

        try:
            idx = int(AF_fn(mu, var, found_acc))
        except Exception:
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


# ──────────────────────────────────────────────────────────────
# Pre-load datasets once (module-level cache)
# ──────────────────────────────────────────────────────────────
_LOADED = {}

def _load_datasets():
    """Load all registered datasets into memory (called once)."""
    if _LOADED:
        return
    for name, paths in DATASETS.items():
        sobol = np.load(paths["sobol"])
        acc   = np.load(paths["acc"])
        _LOADED[name] = (sobol, acc)
        print(f"  [{name}] sobol {sobol.shape}, acc {acc.shape}, "
              f"best={acc.max():.6f}, worst={acc.min():.6f}")


def run_bo_multi(AF_fn, T=120, seed=0):
    """
    Run BO on ALL datasets.

    Returns a dict:  { dataset_name: (found_acc, true_acc, acc_t0, Th) }
    """
    _load_datasets()
    results = {}
    for name, (sobol, acc) in _LOADED.items():
        results[name] = run_bo_single(AF_fn, sobol, acc, T=T, seed=seed)
    return results
