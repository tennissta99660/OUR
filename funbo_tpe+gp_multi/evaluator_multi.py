"""
Multi-dataset BO evaluator — Hybrid GP -> TPE handoff.

Runs the same GP->TPE hybrid BO loop on multiple datasets
(CIFAR-10, CIFAR-100, Caltech-101), returning per-dataset results
so the caller can compute paper_score for each and average them.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import os
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import optuna
from optuna.distributions import FloatDistribution

# Suppress Optuna logging to prevent terminal spam
optuna.logging.set_verbosity(optuna.logging.WARNING)

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


def get_nearest_unobserved(point, grid, observed_set):
    """
    L2 nearest-neighbor search to map continuous TPE suggestions
    to the discrete Sobol grid, masking already observed points.
    """
    dists = np.linalg.norm(grid - point, axis=1)
    dists[list(observed_set)] = np.inf
    return int(np.argmin(dists))


def run_bo_single(AF_fn, sobol, acc, T=120, T_GP=35, seed=0):
    """
    Hybrid BO on a SINGLE dataset: GP phase (T_GP iters) -> TPE phase (remaining).

    Returns (found_acc, true_acc, acc_t0, Th) — the 4 values
    needed by paper_score().
    """
    rng = np.random.RandomState(seed)
    N, d = sobol.shape

    mins = sobol.min(axis=0)
    maxs = sobol.max(axis=0)

    true_acc = float(acc.max())
    true_idx = int(acc.argmax())

    # Worst initial design (paper protocol)
    init_idx = int(acc.argmin())
    observed = [init_idx]
    values   = [float(acc[init_idx])]

    acc_t0    = values[0]
    found_acc = acc_t0
    Th        = T

    kernel = RBF(length_scale=np.ones(d))

    # =========================================================
    # PHASE 1: BO-GP using LLM-Evolved Acquisition Function
    # =========================================================
    for t in range(1, min(T_GP, T)):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer=None,     # Fixed hyperparams for speed
            normalize_y=True,
        )
        gp.fit(sobol[observed], np.array(values))
        mu, std = gp.predict(sobol, return_std=True)
        var = std ** 2

        try:
            idx = int(AF_fn(mu, var, found_acc))
        except Exception:
            idx = -1

        # Fallback if LLM suggests invalid/duplicate index
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

    if T <= T_GP:
        return found_acc, true_acc, acc_t0, Th

    # =========================================================
    # PHASE 2: BO-TPE Handoff (Optuna)
    # =========================================================
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)
    study = optuna.create_study(sampler=sampler, direction="maximize")

    distributions = {f"x_{i}": FloatDistribution(mins[i], maxs[i]) for i in range(d)}

    # Inject Phase 1 (GP) history to aggressively bias TPE priors
    for idx, val in zip(observed, values):
        trial = optuna.trial.create_trial(
            params={f"x_{i}": sobol[idx, i] for i in range(d)},
            distributions=distributions,
            value=val
        )
        study.add_trial(trial)

    observed_set = set(observed)

    # Execute TPE density sampling for remaining iterations
    for t in range(T_GP, T):
        trial = study.ask()
        point = np.array([trial.suggest_float(f"x_{i}", mins[i], maxs[i]) for i in range(d)])

        idx = get_nearest_unobserved(point, sobol, observed_set)

        observed.append(idx)
        observed_set.add(idx)
        val = float(acc[idx])
        values.append(val)

        study.tell(trial, val)

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


def run_bo_multi(AF_fn, T=120, T_GP=35, seed=0):
    """
    Run hybrid GP->TPE BO on ALL datasets.

    Returns a dict:  { dataset_name: (found_acc, true_acc, acc_t0, Th) }
    """
    _load_datasets()
    results = {}
    for name, (sobol, acc) in _LOADED.items():
        results[name] = run_bo_single(AF_fn, sobol, acc, T=T, T_GP=T_GP, seed=seed)
    return results
