import os, random, numpy as np
from llm_client_3 import generate_af
from score_paper import paper_score
from evaluator_paper import run_bo_paper
from ei_baseline import BASE_EI_CODE, acquisition as baseline_af

ISLANDS = 3
GENS = 20
SEEDS = [0, 42, 123]

# Hybrid configuration
T = 120
T_GP = 35  # Matches the ESTMS paper's 35 initial trial split before TPE handoff

PROG_DIR = r"D:\hckthon\our\funbo_fast\funbo_caltech_results\v1_tpe_gp"
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
    ei_results = [run_bo_paper(baseline_af, T=T, T_GP=T_GP, seed=s) for s in SEEDS]
    ei_score = np.mean([paper_score(*r, T=T) for r in ei_results])

    print(f"Baseline EI score (Hybrid GP->TPE): {ei_score:.4f}")

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

                results = [run_bo_paper(af, T=T, T_GP=T_GP, seed=s) for s in SEEDS]

                scores = [paper_score(*r, T=T) for r in results]
                val_accs = [r[0] for r in results]

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
            except Exception as e:
                # Silently fail on bad LLM code generation to keep evolution running
                pass

if __name__ == "__main__":
    main()