"""
Transfer evaluation: top-K AFs evolved on CIFAR-10/100 (GP-only)
evaluated on Caltech-101 using the precomputed accuracy grid.

Demonstrates that FunBO-discovered AFs generalise across datasets.
"""

import warnings
warnings.filterwarnings("ignore")

import os, sys, glob, numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from score_paper import paper_score

# ── CONFIG ──────────────────────────────────────────────────────
# Source AF directory (evolved on CIFAR-10 + CIFAR-100, GP-only)
SOURCE_AF_DIR = r"D:\hckthon\our\funbo_fast\funbo_logs_multi\programs_multi"

# Precomputed Caltech-101 data (shared Sobol grid)
BASE = r"D:\hckthon\our\funbo_fast"
SOBOL_PATH = os.path.join(BASE, "sgd", "sobol_res_sgd.npy")
ACC_PATH   = os.path.join(BASE, "acc_res_sgd_caltech101.npy")

T      = 120
SEEDS  = [0, 42, 123]
TOP_K  = 10
# ────────────────────────────────────────────────────────────────


# =====================================================================
# BO loop (GP-only, same as funbo_multi evaluator)
# =====================================================================
def run_bo_single(AF_fn, sobol, acc, T=120, seed=0):
    rng = np.random.RandomState(seed)
    N, d = sobol.shape

    true_acc = float(acc.max())
    true_idx = int(acc.argmax())

    init_idx = int(acc.argmin())
    observed = [init_idx]
    values   = [float(acc[init_idx])]

    acc_t0    = values[0]
    found_acc = acc_t0
    Th        = T

    kernel = RBF(length_scale=np.ones(d))

    for t in range(1, T):
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6,
            optimizer=None, normalize_y=True,
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


def score_af(AF_fn, sobol, acc, T=T, seeds=SEEDS):
    """Run AF on Caltech-101 across seeds, return mean paper_score + details."""
    scores, accs, ths = [], [], []
    for seed in seeds:
        found_acc, true_acc, acc_t0, Th = run_bo_single(AF_fn, sobol, acc, T=T, seed=seed)
        s = paper_score(found_acc, true_acc, acc_t0, Th, T)
        scores.append(s)
        accs.append(found_acc)
        ths.append(Th)
    return {
        "score": float(np.mean(scores)),
        "acc":   float(np.mean(accs)),
        "Th":    float(np.mean(ths)),
    }


# =====================================================================
# Load AF from saved .py file
# =====================================================================
def load_af_from_file(path):
    with open(path, "r") as f:
        code = f.read()
    lines = code.split("\n")
    clean_lines = [l for l in lines if not l.strip().startswith("# Score:")]
    clean_code = "\n".join(clean_lines)
    scope = {}
    exec(clean_code, scope)
    return scope["acquisition"]


# =====================================================================
# Baselines
# =====================================================================
import math

def acquisition_ei(means, variances, best):
    scores = []
    for m, v in zip(means, variances):
        sigma = math.sqrt(max(v, 1e-12))
        z = (best - m) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
        scores.append((best - m) * cdf + sigma * pdf)
    return int(np.argmax(scores))


def acquisition_ucb(means, variances, best, beta=2.0):
    stds = np.sqrt(np.maximum(variances, 1e-12))
    return int(np.argmax(means + beta * stds))


def acquisition_pi(means, variances, best):
    scores = []
    for m, v in zip(means, variances):
        sigma = math.sqrt(max(v, 1e-12))
        z = (m - best) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        scores.append(cdf)
    return int(np.argmax(scores))


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("TRANSFER EVAL: Top-K AFs (CIFAR-10/100, GP-only) → Caltech-101")
    print("Using precomputed accuracy grid (no training)")
    print("=" * 70)

    # Load Caltech-101 data
    sobol = np.load(SOBOL_PATH)
    acc   = np.load(ACC_PATH)
    print(f"\nCaltech-101: sobol {sobol.shape}, acc {acc.shape}")
    print(f"  best={acc.max():.6f}, worst={acc.min():.6f}")
    print(f"  T={T}, seeds={SEEDS}")

    # ── Baselines ──
    print(f"\n{'─' * 70}")
    print("Computing baselines on Caltech-101 ...")
    ei_info  = score_af(acquisition_ei, sobol, acc)
    ucb_info = score_af(acquisition_ucb, sobol, acc)
    pi_info  = score_af(acquisition_pi, sobol, acc)

    print(f"  EI  : score={ei_info['score']:.6f}  acc={ei_info['acc']:.6f}  Th={ei_info['Th']:.1f}")
    print(f"  UCB : score={ucb_info['score']:.6f}  acc={ucb_info['acc']:.6f}  Th={ucb_info['Th']:.1f}")
    print(f"  PI  : score={pi_info['score']:.6f}  acc={pi_info['acc']:.6f}  Th={pi_info['Th']:.1f}")

    # ── Load and rank source AFs by their CIFAR-10/100 score ──
    af_files = sorted(glob.glob(os.path.join(SOURCE_AF_DIR, "gen*.py")))
    print(f"\nFound {len(af_files)} AFs in {SOURCE_AF_DIR}")

    af_entries = []
    for path in af_files:
        with open(path) as f:
            first_line = f.readline().strip()
        try:
            src_score = float(first_line.replace("# Score:", "").strip())
        except ValueError:
            src_score = -999.0
        af_entries.append((path, src_score))

    # Sort by source (CIFAR-10/100) score, descending
    af_entries.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"Evaluating top-{TOP_K} AFs on Caltech-101 (precomputed)")
    print(f"{'=' * 70}")

    results = []
    for rank, (path, src_score) in enumerate(af_entries[:TOP_K], 1):
        fname = os.path.basename(path)
        try:
            af_fn = load_af_from_file(path)
            info = score_af(af_fn, sobol, acc)
            results.append({
                "rank": rank,
                "file": fname,
                "src_score": src_score,
                "cal_score": info["score"],
                "cal_acc":   info["acc"],
                "cal_Th":    info["Th"],
                "vs_ei":     info["score"] - ei_info["score"],
            })
            print(f"  [{rank:2d}/{TOP_K}] {fname:>16s}  "
                  f"src={src_score:.4f}  "
                  f"cal_score={info['score']:.4f}  "
                  f"cal_acc={info['acc']:.4f}  "
                  f"Th={info['Th']:.1f}  "
                  f"vs_EI={info['score'] - ei_info['score']:+.4f}")
        except Exception as e:
            print(f"  [{rank:2d}/{TOP_K}] {fname:>16s}  FAILED: {e}")

    # ── Summary ──
    print(f"\n\n{'=' * 70}")
    print(f"SUMMARY: Top-{TOP_K} AFs (CIFAR-10/100) → Caltech-101")
    print(f"EI Baseline on Caltech-101: score={ei_info['score']:.6f}  acc={ei_info['acc']:.6f}")
    print(f"{'=' * 70}")
    print(f"{'Rank':>4s}  {'File':>16s}  {'SrcScore':>9s}  {'CalScore':>9s}  {'CalAcc':>9s}  {'Th':>5s}  {'vs EI':>8s}")
    print("─" * 70)

    for r in results:
        marker = " ✓" if r["vs_ei"] > 0 else ""
        print(f"  {r['rank']:>2d}    {r['file']:>16s}  {r['src_score']:>9.4f}  "
              f"{r['cal_score']:>9.4f}  {r['cal_acc']:>9.4f}  "
              f"{r['cal_Th']:>5.1f}  {r['vs_ei']:>+8.4f}{marker}")

    wins = sum(1 for r in results if r["vs_ei"] > 0)
    if results:
        best_r = max(results, key=lambda x: x["cal_score"])
        print(f"\n  Best:  {best_r['file']}  →  score={best_r['cal_score']:.4f}  "
              f"acc={best_r['cal_acc']:.4f}  ({best_r['vs_ei']:+.4f} vs EI)")
        print(f"  Wins:  {wins}/{len(results)} AFs beat EI baseline on Caltech-101")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
