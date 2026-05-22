import os, random, numpy as np
from llm_client_3_gp import generate_af
from score_paper_gp import paper_score
from evaluator_paper_gp import run_bo_paper
from ei_baseline_gp import BASE_EI_CODE, acquisition as baseline_af, acquisition_ucb, acquisition_pi

ISLANDS = 3
GENS = 20
SEEDS = [0, 42, 123]
T = 120

PROG_DIR = r"D:\hckthon\our\funbo_fast\funbo_logs_res_caltech101_gp\v1"
os.makedirs(PROG_DIR, exist_ok=True)


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


def main():

    ei_results = [run_bo_paper(baseline_af, T=T, seed=s) for s in SEEDS]
    ei_score = np.mean([paper_score(*r, T=T) for r in ei_results])

    ucb_results = [run_bo_paper(acquisition_ucb, T=T, seed=s) for s in SEEDS]
    ucb_score = np.mean([paper_score(*r, T=T) for r in ucb_results])

    pi_results = [run_bo_paper(acquisition_pi, T=T, seed=s) for s in SEEDS]
    pi_score = np.mean([paper_score(*r, T=T) for r in pi_results])

    print(f"Baseline EI score:  {ei_score:.6f}")
    print(f"Baseline UCB score: {ucb_score:.6f}")
    print(f"Baseline PI score:  {pi_score:.6f}")

    db = FunBODatabase(ISLANDS, BASE_EI_CODE, ei_score)

    for gen in range(GENS):
        for isl in range(ISLANDS):

            v0, v1 = db.sample(isl)
            new_code = generate_af(v0['code'], v0['score'], v1['code'], v1['score'])
            if not new_code:
                continue

            try:
                scope = {}
                exec(new_code, scope)
                af = scope['acquisition']

                results = [run_bo_paper(af, T=T, seed=s) for s in SEEDS]

                scores = [paper_score(*r, T=T) for r in results]
                val_accs = [r[0] for r in results]   # <-- found_acc

                score = np.mean(scores)
                avg_acc = np.mean(val_accs)

                db.add(isl, new_code, score, f"gen{gen}_isl{isl}")

                true_accs = [r[1] for r in results]
                ths = [r[3] for r in results]

                print(
                   f"Gen {gen} Isl {isl} | "
                   f"Score {score:.4f} | "
                   f"ValAcc {avg_acc:.4f} | "
                   f"Th {np.mean(ths):.1f}"
                     )
            except:
                pass


if __name__ == "__main__":
    main()
