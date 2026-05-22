"""
FunBO core — multi-dataset scoring (CIFAR-10, CIFAR-100, Caltech-101).

The paper_score is computed separately for each dataset, then
AVERAGED across datasets. This is the original FunBO approach:
the score reflects how well an AF generalises across multiple
ground-truth tasks (each with its own true_acc, acc_t0, etc.).
"""

import os, random, numpy as np
from llm_client import generate_af
from score_paper import paper_score
from evaluator_multi import run_bo_multi, DATASETS
from ei_baseline import BASE_EI_CODE, acquisition as baseline_af, acquisition_ucb, acquisition_pi

ISLANDS = 3
GENS    = 20
SEEDS   = [0, 42, 123]
T       = 120

PROG_DIR = r"D:\hckthon\our\funbo_fast\funbo_logs_3_cif_cal\v1_gp"
os.makedirs(PROG_DIR, exist_ok=True)

DATASET_NAMES = list(DATASETS.keys())    # ["cifar10", "cifar100"]
N_DATASETS    = len(DATASET_NAMES)


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def multi_score(AF_fn, T=T, seeds=SEEDS):
    """
    Run AF on every dataset × every seed, return the MEAN paper_score.

    score = (1 / |D|) * Σ_d  mean_over_seeds[ paper_score(d) ]

    This is the original FunBO multi-task scoring.
    """
    per_dataset_scores = []

    for seed in seeds:
        ds_results = run_bo_multi(AF_fn, T=T, seed=seed)   # {name: (found, true, t0, Th)}
        for name in DATASET_NAMES:
            r = ds_results[name]
            per_dataset_scores.append(paper_score(*r, T=T))

    # Average over all (seed × dataset) pairs
    return float(np.mean(per_dataset_scores))


def multi_score_detailed(AF_fn, T=T, seeds=SEEDS):
    """
    Same as multi_score but also returns per-dataset breakdowns
    for logging.
    """
    all_scores  = []                            # flat list for overall mean
    per_ds      = {n: [] for n in DATASET_NAMES}
    per_ds_acc  = {n: [] for n in DATASET_NAMES}
    per_ds_th   = {n: [] for n in DATASET_NAMES}

    for seed in seeds:
        ds_results = run_bo_multi(AF_fn, T=T, seed=seed)
        for name in DATASET_NAMES:
            found_acc, true_acc, acc_t0, Th = ds_results[name]
            s = paper_score(found_acc, true_acc, acc_t0, Th, T)
            all_scores.append(s)
            per_ds[name].append(s)
            per_ds_acc[name].append(found_acc)
            per_ds_th[name].append(Th)

    return {
        "score":    float(np.mean(all_scores)),
        "per_ds":   {n: float(np.mean(v)) for n, v in per_ds.items()},
        "per_acc":  {n: float(np.mean(v)) for n, v in per_ds_acc.items()},
        "per_th":   {n: float(np.mean(v)) for n, v in per_ds_th.items()},
    }


# ──────────────────────────────────────────────────────────────
# Island database
# ──────────────────────────────────────────────────────────────

class FunBODatabase:
    def __init__(self, num_islands, baseline_code, baseline_score):
        self.islands = [[{
            'code': baseline_code,
            'score': baseline_score,
            'name': 'baseline_ei'
        }] for _ in range(num_islands)]

    def sample(self, isl_idx):
        isl = self.islands[isl_idx]
        v1 = sorted(isl, key=lambda x: x['score'])[-1]
        v0 = random.choice(isl)
        return v0, v1

    def add(self, isl_idx, code, score, name):
        self.islands[isl_idx].append({'code': code, 'score': score})
        with open(os.path.join(PROG_DIR, f"{name}.py"), "w") as f:
            f.write(f"# Score: {score:.4f}\n{code}")


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FunBO — Multi-Dataset Scoring (CIFAR-10, CIFAR-100, Caltech-101)")
    print("=" * 65)
    print(f"Datasets : {DATASET_NAMES}")
    print(f"Islands  : {ISLANDS}")
    print(f"Gens     : {GENS}")
    print(f"Seeds    : {SEEDS}")
    print(f"T        : {T}")
    print()

    # ---- Baselines (multi-dataset) ----
    print("Computing baseline scores (multi-dataset) ...")

    ei_info  = multi_score_detailed(baseline_af)
    ucb_info = multi_score_detailed(acquisition_ucb)
    pi_info  = multi_score_detailed(acquisition_pi)

    print(f"\nBaseline EI  score : {ei_info['score']:.6f}  {ei_info['per_ds']}")
    print(f"Baseline UCB score : {ucb_info['score']:.6f}  {ucb_info['per_ds']}")
    print(f"Baseline PI  score : {pi_info['score']:.6f}  {pi_info['per_ds']}")
    print()

    db = FunBODatabase(ISLANDS, BASE_EI_CODE, ei_info['score'])

    for gen in range(GENS):
        for isl in range(ISLANDS):

            v0, v1 = db.sample(isl)
            new_code = generate_af(v0['code'], v0['score'], v1['code'], v1['score'])
            if not new_code:
                print(f"Gen {gen} Isl {isl} | SKIP (LLM returned no code)")
                continue

            try:
                scope = {}
                exec(new_code, scope)
                af = scope['acquisition']

                info = multi_score_detailed(af)
                score = info['score']

                db.add(isl, new_code, score, f"gen{gen}_isl{isl}")

                # Compact log line
                ds_parts = " | ".join(
                    f"{n}: s={info['per_ds'][n]:.4f} acc={info['per_acc'][n]:.4f} Th={info['per_th'][n]:.1f}"
                    for n in DATASET_NAMES
                )
                print(
                    f"Gen {gen} Isl {isl} | "
                    f"Score {score:.4f} | "
                    f"{ds_parts}"
                )
            except Exception as e:
                print(f"Gen {gen} Isl {isl} | SKIP (exec error: {e})")


if __name__ == "__main__":
    main()
