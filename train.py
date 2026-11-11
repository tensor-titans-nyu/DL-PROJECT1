#!/usr/bin/env python
"""
CIFAR-10 ResNet Challenge (NYU Tandon DL F26) -- teacher / student / distillation trainer.

  * Pre-activation Wide ResNet (residual CNN, fits the rules)
  * Strong aug: reflect-pad crop, flip, TrivialAugmentWide, RandomErasing, CutMix
  * SGD+Nesterov, warmup + cosine, label smoothing, EMA weights, bf16/fp16 AMP
  * Optional knowledge distillation from teachers trained FROM SCRATCH on the same
    CIFAR-10 training data (allowed by Rule 5). Teachers are not part of the final model.
  * Flip TTA at prediction time; writes submission.csv + test probabilities.
  * Auto-resume from last checkpoint (safe for SLURM preemption / time limits).

Examples
  Teacher (no param limit):
    python train.py --data DATA --out runs/t0 --depth 28 --width 10 --epochs 200 --max_params 0 --seed 0
  Student baseline (<=5M):
    python train.py --data DATA --out runs/s_base --depth 22 --width 4 --epochs 250
  Student with distillation:
    python train.py --data DATA --out runs/s_kd --depth 22 --width 4 --epochs 300 \
        --teachers runs/t0/final.pt runs/t1/final.pt
"""
import argparse, glob, math, os, re, time, random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


# ----------------------------------------------------------------------------- args
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="competition folder (has train_images/, test_images/, csvs)")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", default=None, help="where to cache decoded arrays (default: <out>/../cache)")
    p.add_argument("--depth", type=int, default=22)
    p.add_argument("--width", type=float, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.1, help="LR per 128 images (scaled linearly with --bs)")
    p.add_argument("--wd", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--cutmix_prob", type=float, default=0.5)
    p.add_argument("--mix_off_epochs", type=int, default=15, help="turn CutMix off for the last N epochs")
    p.add_argument("--erase_prob", type=float, default=0.25)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--teachers", nargs="*", default=[], help="teacher final.pt files (enables KD)")
    p.add_argument("--kd_alpha", type=float, default=0.9, help="weight of KD loss vs hard-label CE")
    p.add_argument("--kd_T", type=float, default=2.0)
    p.add_argument("--max_params", type=int, default=5_000_000, help="0 disables the check (teachers)")
    p.add_argument("--holdout", type=int, default=0, help="hold out N train images to measure val acc")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--predict_only", default=None, help="path to final.pt: skip training, just predict")
    return p.parse_args()


# ----------------------------------------------------------------------------- data
def _num(x):
    d = re.findall(r"\d+", str(x))
    return int(d[-1]) if d else None


def _find(data_dir, name):
    hits = glob.glob(os.path.join(data_dir, "**", name), recursive=True)
    return hits[0] if hits else None


def load_split(data_dir, split, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)
    cache = os.path.join(cache_dir, f"{split}.npz")
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["X"], (z["y"] if z["y"].size else None), z["ids"]

    csv = _find(data_dir, "train_labels.csv" if split == "train" else "test.csv")
    folder = _find(data_dir, f"{split}_images")
    df = pd.read_csv(csv)
    idcol = "id" if "id" in df.columns else df.columns[0]
    pngs = glob.glob(os.path.join(folder, "**", "*.png"), recursive=True)
    by_name = {os.path.basename(f): f for f in pngs}
    by_num = {_num(os.path.basename(f)): f for f in pngs}
    fncol = next((c for c in df.columns if df[c].astype(str).str.lower().str.endswith(".png").any()), None)
    if fncol:
        paths = [by_name[os.path.basename(str(v))] for v in df[fncol]]
    else:
        paths = [by_num[_num(v)] for v in df[idcol]]
    X = np.stack([np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8) for p in paths])
    y = np.array([], dtype=np.int64)
    if split == "train":
        labcol = "label" if "label" in df.columns else df.columns[-1]
        y = df[labcol].values.astype(np.int64)
    ids = df[idcol].values
    np.savez(cache, X=X, y=y, ids=ids)
    print(f"[data] {split}: {X.shape} cached -> {cache}")
    return X, (y if y.size else None), ids


class ArrDS(Dataset):
    def __init__(self, X, y, tf):
        self.X, self.y, self.tf = X, y, tf

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        img = torch.from_numpy(self.X[i]).permute(2, 0, 1).contiguous()  # uint8 CHW
        return self.tf(img), int(self.y[i])


def train_transform(erase_prob):
    return T.Compose([
        T.RandomCrop(32, padding=4, padding_mode="reflect"),
        T.RandomHorizontalFlip(),
        T.TrivialAugmentWide(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=erase_prob, value=0),
    ])


def to_input(Xu8, device):
    x = torch.from_numpy(Xu8).to(device).permute(0, 3, 1, 2).float().div_(255)
    m = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    s = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return ((x - m) / s).contiguous(memory_format=torch.channels_last)


# ----------------------------------------------------------------------------- model
class PreActBlock(nn.Module):
    def __init__(self, cin, cout, stride, drop):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Conv2d(cin, cout, 1, stride, 0, bias=False)

    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = self.short(o) if self.short is not None else x
        o = self.conv1(o)
        o = self.conv2(self.drop(F.relu(self.bn2(o))))
        return o + sc  # residual (skip) connection


class WideResNet(nn.Module):
    def __init__(self, depth=22, width=4, drop=0.0, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        w = [16, int(round(16 * width)), int(round(32 * width)), int(round(64 * width))]
        self.conv1 = nn.Conv2d(3, w[0], 3, 1, 1, bias=False)
        layers, cin = [], w[0]
        for stage, stride in zip(range(1, 4), (1, 2, 2)):
            for b in range(n):
                layers.append(PreActBlock(cin, w[stage], stride if b == 0 else 1, drop))
                cin = w[stage]
        self.blocks = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin)
        self.fc = nn.Linear(cin, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.blocks(self.conv1(x))
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def load_model(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = WideResNet(ck["depth"], ck["width"], 0.0)
    m.load_state_dict(ck["state"])
    return m.to(device).to(memory_format=torch.channels_last).eval()


# ----------------------------------------------------------------------------- helpers
def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    H, W = x.shape[2:]
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = np.clip(cy - rh // 2, 0, H), np.clip(cy + rh // 2, 0, H)
    x1, x2 = np.clip(cx - rw // 2, 0, W), np.clip(cx + rw // 2, 0, W)
    x = x.clone()
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((y2 - y1) * (x2 - x1) / (H * W))
    return x, y, y[idx], lam


@torch.no_grad()
def predict_probs(model, Xu8, device, bs=500, tta=True):
    model.eval()
    out = []
    for i in range(0, len(Xu8), bs):
        x = to_input(Xu8[i:i + bs], device)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
            p = F.softmax(model(x).float(), 1)
            if tta:
                p = p + F.softmax(model(torch.flip(x, dims=[3])).float(), 1)
        out.append((p / (2 if tta else 1)).cpu())
    return torch.cat(out).numpy()


def write_submission(args, probs, test_ids, tag):
    preds = probs.argmax(1)
    sub = pd.DataFrame({"id": test_ids, "label": preds.astype(int)})
    sample = _find(args.data, "sample_submission.csv")
    if sample:
        s = pd.read_csv(sample)[["id"]]
        sub = s.merge(sub, on="id", how="left")
        assert sub["label"].notna().all(), "id mismatch between test.csv and sample_submission.csv"
        sub["label"] = sub["label"].astype(int)
    path = os.path.join(args.out, f"submission_{tag}.csv")
    sub.to_csv(path, index=False)
    np.save(os.path.join(args.out, f"test_probs_{tag}.npy"), probs)
    print(f"[out] {path}  class counts: {np.bincount(preds, minlength=10).tolist()}")


# ----------------------------------------------------------------------------- main
def main():
    global AMP_DTYPE
    args = get_args()
    os.makedirs(args.out, exist_ok=True)
    cache = args.cache or os.path.join(os.path.dirname(os.path.abspath(args.out)), "cache")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16 = device.type == "cuda" and torch.cuda.get_device_capability()[0] >= 8  # A100/H100 yes, T4/P100 no
    AMP_DTYPE = torch.bfloat16 if bf16 else torch.float16
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[env] device={device} {torch.cuda.get_device_name(0) if device.type == 'cuda' else ''} amp={AMP_DTYPE}")

    Xtr, ytr, _ = load_split(args.data, "train", cache)
    Xte, _, test_ids = load_split(args.data, "test", cache)

    if args.predict_only:
        m = load_model(args.predict_only, device)
        write_submission(args, predict_probs(m, Xte, device), test_ids, "pred")
        return

    Xval = yval = None
    if args.holdout > 0:
        perm = np.random.RandomState(1234).permutation(len(Xtr))  # fixed split across runs
        v, t = perm[:args.holdout], perm[args.holdout:]
        Xval, yval, Xtr, ytr = Xtr[v], ytr[v], Xtr[t], ytr[t]
    print(f"[data] train={len(Xtr)} val={0 if Xval is None else len(Xval)} test={len(Xte)}")

    dl = DataLoader(ArrDS(Xtr, ytr, train_transform(args.erase_prob)), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True,
                    persistent_workers=args.workers > 0)

    model = WideResNet(args.depth, args.width, args.dropout).to(device).to(memory_format=torch.channels_last)
    n_params = count_params(model)
    print(f"[model] PreAct WRN-{args.depth}-{args.width}  trainable params = {n_params:,}")
    if args.max_params and n_params > args.max_params:
        raise SystemExit(f"Model has {n_params:,} params > limit {args.max_params:,}")

    teachers = [load_model(t, device) for t in args.teachers]
    if teachers:
        print(f"[kd] {len(teachers)} teacher(s), alpha={args.kd_alpha}, T={args.kd_T}")

    decay, no_decay = [], []
    for p_ in model.parameters():
        (no_decay if p_.ndim <= 1 else decay).append(p_)
    peak_lr = args.lr * args.bs / 128
    opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                           {"params": no_decay, "weight_decay": 0.0}],
                          lr=peak_lr, momentum=0.9, nesterov=True)
    steps_per_epoch = len(dl)
    total, warm = args.epochs * steps_per_epoch, args.warmup * steps_per_epoch

    def lr_at(step):
        if step < warm:
            return peak_lr * (step + 1) / warm
        return 0.5 * peak_lr * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema), use_buffers=True)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not bf16))
    fwd = torch.compile(model) if args.compile else model

    start_epoch, ckpt_path = 0, os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_epoch = ck["epoch"]
        print(f"[resume] from epoch {start_epoch}")

    ce = nn.CrossEntropyLoss(label_smoothing=0.0 if teachers else args.label_smoothing)
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        use_mix = epoch < args.epochs - args.mix_off_epochs
        tot_loss, nb, te = 0.0, 0, time.time()
        for i, (x, y) in enumerate(dl):
            step = epoch * steps_per_epoch + i
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            mixed = use_mix and np.random.rand() < args.cutmix_prob
            if mixed:
                x, ya, yb, lam = cutmix(x, y)
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=device.type == "cuda"):
                logits = fwd(x)
                if mixed:
                    hard = lam * ce(logits.float(), ya) + (1 - lam) * ce(logits.float(), yb)
                else:
                    hard = ce(logits.float(), y)
                if teachers:
                    with torch.no_grad():
                        pt = sum(F.softmax(t(x).float() / args.kd_T, 1) for t in teachers) / len(teachers)
                    kd = F.kl_div(F.log_softmax(logits.float() / args.kd_T, 1), pt,
                                  reduction="batchmean") * args.kd_T ** 2
                    loss = args.kd_alpha * kd + (1 - args.kd_alpha) * hard
                else:
                    loss = hard
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update_parameters(model)
            tot_loss += loss.item(); nb += 1

        msg = f"epoch {epoch + 1:03d}/{args.epochs} loss {tot_loss / nb:.4f} lr {lr_at(step):.4f} " \
              f"{time.time() - te:.1f}s"
        if Xval is not None and ((epoch + 1) % 10 == 0 or epoch + 1 == args.epochs):
            acc = (predict_probs(ema.module, Xval, device).argmax(1) == yval).mean()
            msg += f" | val(EMA+TTA) {acc * 100:.2f}%"
        print(msg, flush=True)

        if (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                        "scaler": scaler.state_dict(), "epoch": epoch + 1}, ckpt_path)

    print(f"[done] training {(time.time() - t0) / 60:.1f} min")
    final = ema.module
    torch.save({"state": final.state_dict(), "depth": args.depth, "width": args.width,
                "params": n_params, "args": vars(args)}, os.path.join(args.out, "final.pt"))
    write_submission(args, predict_probs(final, Xte, device), test_ids, "ema_tta")
    print(f"[final] trainable params = {n_params:,}")


AMP_DTYPE = torch.bfloat16
if __name__ == "__main__":
    main()
