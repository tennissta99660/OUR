import numpy as np
import os
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import optuna
from optuna.distributions import FloatDistribution

# Suppress Optuna logging to prevent terminal spam during parallel island evolution
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

DATA_DIR = r"D:\hckthon\our\funbo_fast"

def get_nearest_unobserved(point, grid, observed_set):
    """
    SOTA L2 nearest-neighbor search to map continuous TPE suggestions 
    to the discrete Sobol grid, masking already observed points.
    """
    dists = np.linalg.norm(grid - point, axis=1)
    dists[list(observed_set)] = np.inf
    return int(np.argmin(dists))

def run_bo_paper(AF_fn, T=120, T_GP=35, seed=0):
    """
    Hybrid BO Evaluator: LLM-driven BO-GP -> BO-TPE handoff.
    """
    rng = np.random.RandomState(seed)

    acc   = np.load(os.path.join(DATA_DIR, "acc_res_sgd_caltech101.npy"))
    sobol = np.load(os.path.join(DATA_DIR, "sobol_cifar100_d3.npy"))

    N, d = sobol.shape
    mins = sobol.min(axis=0)
    maxs = sobol.max(axis=0)

    true_acc = float(acc.max())
    true_idx = int(acc.argmax())

    # Worst initial design (matches paper baseline)
    init_idx = int(acc.argmin())
    observed = [init_idx]
    values   = [float(acc[init_idx])]

    acc_t0 = values[0]
    found_acc = acc_t0
    Th = T

    kernel = RBF(length_scale=np.ones(d))

    # =========================================================
    # PHASE 1: BO-GP using LLM-Evolved Acquisition Function
    # =========================================================
    for t in range(1, min(T_GP, T)):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            optimizer=None, # Fixed hyperparams for speed
            normalize_y=True
        )
        gp.fit(sobol[observed], np.array(values))
        mu, std = gp.predict(sobol, return_std=True)
        var = std ** 2

        try:
            # Execute LLM AF
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