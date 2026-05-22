"""
Transfer evaluation: take the top-K AFs evolved on CIFAR-10/100 and run
REAL Bayesian Optimization on Caltech-101 (actual KD training, no precomputed
acc file needed).

This demonstrates that FunBO-discovered AFs generalise across datasets.
"""

import os, sys, glob, math, random, numpy as np, torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from torch.quasirandom import SobolEngine
import warnings
warnings.filterwarnings("ignore")

# ── CONFIG ──────────────────────────────────────────────────────────────
# Directory containing the AFs evolved on the SOURCE dataset (CIFAR-10/100)
SOURCE_AF_DIR = r"D:\hckthon\our\funbo_fast\funbo_logs_multi\programs_multi_v1"

# Caltech-101 teacher weights
TEACHER_PATH = r"D:\hckthon\our\funbo_fast\best_res50_cal101.pth"

# Search space (same as CIFAR-10/100: d=3)
LR_LOW, LR_HIGH = 0.01, 0.5
TEMP_LOW, TEMP_HIGH = 2, 20
ALPHA_LOW, ALPHA_HIGH = 0.5, 0.9

# BO settings
T_BUDGET = 30          # number of BO iterations (each = real training run)
PROXY_EPOCHS = 5       # epochs per KD training (proxy, keep small for speed)
DATA_PCT = 0.30        # fraction of Caltech-101 training data
N_SOBOL = 512          # Sobol grid size
SEED = 42
TOP_K = 3             # evaluate top-K AFs

DATA_ROOT = r"D:\hckthon\our\funbo_fast\data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 101
# ────────────────────────────────────────────────────────────────────────


# =====================================================================
# DATA — Caltech-101
# =====================================================================
class IndexWrapper(torch.utils.data.Dataset):
    """Wraps a dataset to return (data, target, index)."""
    def __init__(self, dataset):
        self.dataset = dataset
    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index
    def __len__(self):
        return len(self.dataset)


def get_caltech101_loaders(subset_pct=0.30, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    tf_train = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandomCrop(224, padding=28, padding_mode='reflect'),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    tf_val = T.Compose([
        T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    full_ds = torchvision.datasets.Caltech101(
        root=DATA_ROOT, download=True, transform=tf_train
    )
    val_ds = torchvision.datasets.Caltech101(
        root=DATA_ROOT, download=False, transform=tf_val
    )

    # Train/val split (80/20)
    n = len(full_ds)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = perm[:split]
    val_idx = perm[split:]

    # Subset training data for speed
    subset_size = int(len(train_idx) * subset_pct)
    train_idx = train_idx[:subset_size]

    train_set = IndexWrapper(Subset(full_ds, train_idx))
    val_set = Subset(val_ds, val_idx)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False,
                            num_workers=12, pin_memory=True)

    print(f"[Caltech-101] Train: {len(train_set)} | Val: {len(val_set)}")
    return train_loader, val_loader


# =====================================================================
# MODELS
# =====================================================================
class StudentWithMLP(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()
        base = torchvision.models.resnet18(weights=None)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


TEACHER_FINETUNE_EPOCHS = 5
TEACHER_SAVE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "teacher_resnet50_caltech101.pth"
)


def finetune_teacher(train_loader, val_loader, epochs=TEACHER_FINETUNE_EPOCHS):
    """Fine-tune ImageNet-pretrained ResNet-50 on Caltech-101."""
    from torch.cuda.amp import GradScaler, autocast

    print(f"  Fine-tuning ImageNet ResNet-50 on Caltech-101 ({epochs} epochs)...")
    model = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
    )
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    nn.init.kaiming_normal_(model.fc.weight, nonlinearity='relu')
    nn.init.zeros_(model.fc.bias)
    model = model.to(DEVICE)

    # Freeze backbone, only train FC head (fast)
    for name, p in model.named_parameters():
        if "fc" not in name:
            p.requires_grad = False

    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(DEVICE == "cuda"))

    for epoch in range(epochs):
        model.train()
        correct = total = 0
        for batch in train_loader:
            x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            optimizer.zero_grad()
            with autocast(enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            correct += logits.argmax(1).eq(y).sum().item()
            total += y.size(0)

        val_acc = evaluate_model(model, val_loader)
        print(f"    Epoch {epoch+1}/{epochs} | train_acc={correct/total:.4f} | val_acc={val_acc:.4f}")

    torch.save(model.state_dict(), TEACHER_SAVE_PATH)
    print(f"  Teacher saved to {TEACHER_SAVE_PATH}")
    return model


def get_teacher(train_loader=None, val_loader=None):
    """Load cached fine-tuned teacher, or fine-tune from ImageNet weights."""
    model = torchvision.models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

    if os.path.exists(TEACHER_SAVE_PATH):
        model.load_state_dict(torch.load(TEACHER_SAVE_PATH, map_location=DEVICE))
        print(f"[Teacher] Loaded cached Caltech-101 teacher from {TEACHER_SAVE_PATH}")
        model.to(DEVICE).eval()
    elif train_loader is not None:
        model = finetune_teacher(train_loader, val_loader)
        model.eval()
    else:
        print("[WARNING] No cached teacher and no data loaders — using ImageNet pretrained (FC will be random)")
        model = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        )
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model.to(DEVICE).eval()

    for p in model.parameters():
        p.requires_grad = False
    return model


# =====================================================================
# KD TRAINING — single config → accuracy
# =====================================================================
def kd_loss_fn(s_logits, t_logits, temperature):
    return nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(s_logits / temperature, dim=1),
        F.softmax(t_logits / temperature, dim=1)
    ) * (temperature ** 2)


@torch.no_grad()
def precompute_teacher_logits(teacher, train_loader, n_samples):
    logits = torch.zeros(n_samples, NUM_CLASSES, pin_memory=True)
    for x, _, idx in train_loader:
        x = x.to(DEVICE, non_blocking=True)
        logits[idx] = teacher(x).cpu()
    return logits


@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    correct = total = 0
    for batch in loader:
        x, y = batch[0], batch[1]
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(1)
        total += y.size(0)
        correct += (preds == y).sum().item()
    return correct / total


def train_kd_single(lr, temp, alpha, train_loader, val_loader, teacher_logits, epochs=5):
    """Train one KD config and return val accuracy."""
    student = StudentWithMLP(num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = optim.SGD(student.parameters(), lr=float(lr),
                          momentum=0.9, weight_decay=1e-4, nesterov=True)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[epochs // 2], gamma=0.1
    )
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        student.train()
        for x, y, idx in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            t_logits = teacher_logits[idx].to(DEVICE)
            s_logits = student(x)

            loss = (alpha * kd_loss_fn(s_logits, t_logits, temp)
                    + (1.0 - alpha) * criterion_ce(s_logits, y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return evaluate_model(student, val_loader)


# =====================================================================
# SOBOL GRID
# =====================================================================
def generate_sobol_grid(n=512, seed=42):
    engine = SobolEngine(dimension=3, scramble=True, seed=seed)
    sobol = engine.draw(n).numpy()

    grid = []
    for u_lr, u_temp, u_alpha in sobol:
        lr = 10 ** (np.log10(LR_LOW) + (np.log10(LR_HIGH) - np.log10(LR_LOW)) * u_lr)
        temp_raw = TEMP_LOW + (TEMP_HIGH - TEMP_LOW) * u_temp
        temp = int(round(temp_raw / 2.0) * 2)
        temp = max(TEMP_LOW, min(TEMP_HIGH, temp))

        alpha_raw = ALPHA_LOW + (ALPHA_HIGH - ALPHA_LOW) * u_alpha
        alpha = round(round(alpha_raw / 0.1) * 0.1, 1)
        alpha = max(ALPHA_LOW, min(ALPHA_HIGH, alpha))

        grid.append((lr, temp, alpha))
    return np.array(grid, dtype=np.float32)


# =====================================================================
# BASELINE EI
# =====================================================================
def acquisition_ei(means, variances, best):
    scores = []
    for m, v in zip(means, variances):
        sigma = math.sqrt(max(v, 1e-12))
        z = (best - m) / sigma
        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        pdf = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z)
        scores.append((best - m) * cdf + sigma * pdf)
    return int(np.argmax(scores))


# =====================================================================
# REAL BO LOOP — actually trains models
# =====================================================================
def run_real_bo(af_fn, sobol_grid, train_loader, val_loader,
                teacher_logits, T_budget=15, seed=42):
    """
    Runs BO with actual KD training at each step.
    No precomputed acc file needed.
    """
    rng = np.random.RandomState(seed)
    N, d = sobol_grid.shape

    # Start with the worst-looking point (lowest LR)
    init_idx = int(sobol_grid[:, 0].argmin())

    observed = [init_idx]
    lr, temp, alpha = sobol_grid[init_idx]
    print(f"    Init config: lr={lr:.5f}, T={int(temp)}, α={alpha:.1f}", end="")

    init_acc = train_kd_single(lr, temp, alpha, train_loader, val_loader,
                                teacher_logits, epochs=PROXY_EPOCHS)
    values = [init_acc]
    best_acc = init_acc
    print(f" → acc={init_acc:.4f}")

    kernel = RBF(length_scale=np.ones(d))

    for t in range(1, T_budget):
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=1e-6,
            optimizer=None, normalize_y=True
        )
        gp.fit(sobol_grid[observed], np.array(values))
        mu, std = gp.predict(sobol_grid, return_std=True)
        var = std ** 2

        try:
            idx = int(af_fn(mu, var, best_acc))
        except:
            idx = -1

        if idx in observed or idx < 0 or idx >= N:
            remaining = np.setdiff1d(np.arange(N), observed)
            idx = int(rng.choice(remaining))

        observed.append(idx)

        lr, temp, alpha = sobol_grid[idx]
        acc = train_kd_single(lr, temp, alpha, train_loader, val_loader,
                              teacher_logits, epochs=PROXY_EPOCHS)
        values.append(acc)

        if acc > best_acc:
            best_acc = acc

        print(f"    t={t:2d} | lr={lr:.5f}, T={int(temp)}, α={alpha:.1f} "
              f"| acc={acc:.4f} | best={best_acc:.4f}")

    return best_acc, sobol_grid[observed[-1]], observed


# =====================================================================
# LOAD AF FROM FILE
# =====================================================================
def load_af_from_file(path):
    with open(path, "r") as f:
        code = f.read()
    lines = code.split("\n")
    clean_lines = [l for l in lines if not l.strip().startswith("# Score:")]
    clean_code = "\n".join(clean_lines)
    scope = {}
    exec(clean_code, scope)
    return scope["acquisition"], lines[0]


# =====================================================================
# MAIN
# =====================================================================
def main():
    print("=" * 70)
    print("REAL BO TRANSFER: Evolved AFs (CIFAR-10/100) → Caltech-101")
    print("No precomputed acc file — actual KD training at each step")
    print("=" * 70)

    # ── Setup ──
    print("\n[1/4] Loading Caltech-101 data...")
    train_loader, val_loader = get_caltech101_loaders(
        subset_pct=DATA_PCT, seed=SEED
    )

    print("\n[2/4] Loading teacher model...")
    teacher = get_teacher()
    n_samples = len(train_loader.dataset)
    print(f"  Precomputing teacher logits ({n_samples} samples)...")
    teacher_logits = precompute_teacher_logits(teacher, train_loader, n_samples)

    print(f"\n[3/4] Generating Sobol grid ({N_SOBOL} points, d=3)...")
    sobol_grid = generate_sobol_grid(N_SOBOL, seed=SEED)

    # ── Load and rank source AFs ──
    af_files = sorted(glob.glob(os.path.join(SOURCE_AF_DIR, "gen*.py")))
    print(f"\n[4/4] Found {len(af_files)} AFs in {SOURCE_AF_DIR}")

    af_entries = []
    for path in af_files:
        with open(path) as f:
            first_line = f.readline().strip()
        try:
            src_score = float(first_line.replace("# Score:", "").strip())
        except ValueError:
            src_score = -999.0
        af_entries.append((path, src_score))

    af_entries.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'=' * 70}")
    print(f"Running EI baseline ({T_BUDGET} BO steps, {PROXY_EPOCHS} epochs each)")
    print(f"{'=' * 70}")
    ei_best, ei_cfg, _ = run_real_bo(
        acquisition_ei, sobol_grid, train_loader, val_loader,
        teacher_logits, T_budget=T_BUDGET, seed=SEED
    )

    print(f"\n{'=' * 70}")
    print(f"Evaluating top-{TOP_K} evolved AFs on Caltech-101")
    print(f"{'=' * 70}")

    results = []
    for rank, (path, src_score) in enumerate(af_entries[:TOP_K], 1):
        fname = os.path.basename(path)
        print(f"\n── [{rank}/{TOP_K}] {fname} (source score: {src_score:.4f}) ──")

        try:
            af_fn, _ = load_af_from_file(path)
            best_acc, best_cfg, _ = run_real_bo(
                af_fn, sobol_grid, train_loader, val_loader,
                teacher_logits, T_budget=T_BUDGET, seed=SEED
            )
            results.append({
                "rank": rank,
                "file": fname,
                "src_score": src_score,
                "caltech_acc": best_acc,
                "vs_ei": best_acc - ei_best,
            })
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            results.append({
                "rank": rank, "file": fname,
                "src_score": src_score, "caltech_acc": 0.0,
                "vs_ei": -ei_best,
            })

    # ── Final Summary ──
    print(f"\n\n{'=' * 70}")
    print(f"SUMMARY: Top-{TOP_K} Evolved AFs vs EI on Caltech-101")
    print(f"EI Baseline Best Accuracy: {ei_best:.4f}")
    print(f"{'=' * 70}")
    print(f"{'Rank':>4s}  {'File':>16s}  {'SrcScore':>9s}  {'Caltech':>9s}  {'vs EI':>8s}")
    print("-" * 55)

    for r in results:
        marker = " ✓" if r["vs_ei"] > 0 else ""
        print(f"  {r['rank']:>2d}    {r['file']:>16s}  {r['src_score']:>9.4f}  "
              f"{r['caltech_acc']:>9.4f}  {r['vs_ei']:>+8.4f}{marker}")

    wins = sum(1 for r in results if r["vs_ei"] > 0)
    if results:
        best_r = max(results, key=lambda x: x["caltech_acc"])
        print(f"\n  Best:  {best_r['file']}  →  {best_r['caltech_acc']:.4f}  ({best_r['vs_ei']:+.4f} vs EI)")
        print(f"  Wins:  {wins}/{len(results)} AFs beat EI baseline")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
