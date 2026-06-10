"""
LayerProx — Linux DGX server version.

Changes from the Windows/Modal version:
  1. Modal fully removed (just run: python layerprox_server.py)
  2. num_workers=4 + persistent_workers=True for fast data loading on Linux
  3. Per-run result saved to JSON (safe for multi-hour sweeps)
  4. debug=True adds diagnostic prints to find the sanity-check anomaly:
       - Round 0 (warmup): confirms µ = µ_base/2 uniform
       - Round 1 (first real δ): prints Δw norm, δ, µ per group
       - Local train round 0: prints CE vs proximal magnitude

GPU: use CUDA_VISIBLE_DEVICES=0 python layerprox_server.py
"""

import copy
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

torch.set_float32_matmul_precision('high')

RESULT_DIR = "./results"
os.makedirs(RESULT_DIR, exist_ok=True)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def default_config():
    return dict(
        method="layerprox",
        dataset="cifar10",
        n_clients=10,
        participation=1.0,
        alpha=0.1,           # Dirichlet concentration
        rounds=25,
        local_epochs=5,
        batch_size=64,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        mu_base=0.1,
        sigmoid_alpha=5.0,
        per_layer=True,
        gate_direction="divergent",
        seed=0,
        eval_every=5,
        num_workers=4,       # Linux: 4 is fine; set 0 for Windows
        debug=False,         # True → diagnostic prints (first 2 rounds)
    )


# --------------------------------------------------------------------------- #
# Result saving
# --------------------------------------------------------------------------- #
def save_result(result, result_dir=RESULT_DIR):
    cfg = result["config"]
    fname = (
        f"{cfg['method']}_mu{cfg['mu_base']}_a{cfg.get('sigmoid_alpha',0)}"
        f"_alpha{cfg['alpha']}_seed{cfg['seed']}.json"
    )
    path = os.path.join(result_dir, fname)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [saved → {path}]")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_datasets(dataset, root):
    import torchvision
    import torchvision.transforms as T

    if dataset == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)
        DS, n_classes = torchvision.datasets.CIFAR10, 10
    elif dataset == "cifar100":
        mean, std = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        DS, n_classes = torchvision.datasets.CIFAR100, 100
    else:
        raise ValueError(dataset)

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = DS(root, train=True,  download=True, transform=train_tf)
    test  = DS(root, train=False, download=True, transform=test_tf)
    return train, test, n_classes


def dirichlet_partition(labels, n_clients, alpha, seed):
    rng    = np.random.default_rng(seed)
    labels = np.array(labels)
    n_cls  = labels.max() + 1
    idx    = [[] for _ in range(n_clients)]
    for c in range(n_cls):
        ic   = np.where(labels == c)[0]
        rng.shuffle(ic)
        props = rng.dirichlet(alpha * np.ones(n_clients))
        cuts  = (np.cumsum(props) * len(ic)).astype(int)[:-1]
        for cid, part in enumerate(np.split(ic, cuts)):
            idx[cid].extend(part.tolist())
    return [np.array(i) for i in idx]


# --------------------------------------------------------------------------- #
# Model: CIFAR ResNet-20
# --------------------------------------------------------------------------- #
class _BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


def build_model(n_classes):
    class ResNet20(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.conv1  = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
            self.bn1    = nn.BatchNorm2d(16)
            self.layer1 = nn.Sequential(_BasicBlock(16,16), _BasicBlock(16,16), _BasicBlock(16,16))
            self.layer2 = nn.Sequential(_BasicBlock(16,32,2), _BasicBlock(32,32), _BasicBlock(32,32))
            self.layer3 = nn.Sequential(_BasicBlock(32,64,2), _BasicBlock(64,64), _BasicBlock(64,64))
            self.fc     = nn.Linear(64, nc)
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer3(self.layer2(self.layer1(out)))
            return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))
    return ResNet20(n_classes)


def group_key(name):
    # 컴파일된 모델의 접두사 제거
    if name.startswith("_orig_mod."):
        name = name[10:]
        
    parts = name.split(".")
    if parts[0].startswith("layer"):
        return ".".join(parts[:2])
    return parts[0]


# --------------------------------------------------------------------------- #
# LayerProx: per-layer µ from drift signal
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _flatten_group(named_tensors, names):
    return torch.cat([named_tensors[n].flatten() for n in names])


def compute_mu_per_group(model, global_state, prev_state, loader, cfg, device,
                         _round=0, _cid=0):
    """Return {group: mu} for one client this round.

    Also prints diagnostic info when cfg['debug']=True and _round <= 1.
    """
    method = cfg["method"]
    debug  = cfg.get("debug", False) and _cid == 0 and _round <= 1

    if method == "fedavg":
        return defaultdict(lambda: 0.0)
    if method == "fedprox":
        return defaultdict(lambda: cfg["mu_base"])

    # --- LayerProx ---
    groups = defaultdict(list)
    for n, _ in model.named_parameters():
        groups[group_key(n)].append(n)

    # Round 0: no previous update → warmup with µ = µ_base * 0.5
    if prev_state is None:
        mu_map = {g: cfg["mu_base"] * 0.5 for g in groups}
        if debug:
            print(f"\n[DEBUG r={_round} warmup] µ_base={cfg['mu_base']} sigmoid_alpha={cfg['sigmoid_alpha']}")
            print(f"  prev_state is None → µ = µ_base/2 = {cfg['mu_base']*0.5:.5f} (uniform, as expected)")
        return mu_map

    # One-step gradient at global weights w^t
    model.load_state_dict(global_state)
    model.train()
    model.zero_grad(set_to_none=True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    F.cross_entropy(model(x), y).backward()
    grads = {n: p.grad.detach().clone()
             for n, p in model.named_parameters() if p.grad is not None}

    # Δw_l = w^{t-1} - w^t  (points in +∇f direction)
    dw = {n: (prev_state[n] - global_state[n]).to(device) for n in grads}

    delta = {}
    for g, names in groups.items():
        gv = _flatten_group(grads, names)
        dv = _flatten_group(dw,    names)
        dv_norm = dv.norm().item()
        gv_norm = gv.norm().item()
        # === 크기(Magnitude) 기반 vs 방향(Cosine) 기반 분기 ===
        if cfg.get("use_magnitude", False):
            delta[g] = gv_norm  # 그래디언트의 크기 자체를 이질성 지표로 사용
        else:
            if dv_norm < 1e-12 or gv_norm < 1e-12:
                delta[g] = 1.0
            else:
                cos = F.cosine_similarity(gv.unsqueeze(0), dv.unsqueeze(0)).item()
                delta[g] = 1.0 - cos

    if not cfg["per_layer"]:
        mean_d = float(np.mean(list(delta.values())))
        delta  = {g: mean_d for g in delta}

    if cfg["gate_direction"] == "similar":
        delta = {g: 2.0 - d for g, d in delta.items()}

    a      = cfg["sigmoid_alpha"]
    mu_map = {g: cfg["mu_base"] / (1.0 + math.exp(-a * delta[g])) for g in delta}

    # ------------------------------------------------------------------ DEBUG
    if debug:
        print(f"\n[DEBUG r={_round} cid={_cid}] µ_base={cfg['mu_base']} sigmoid_alpha={a}")
        print(f"  {'group':20s}  {'dw_norm':>10s}  {'gv_norm':>10s}  {'delta':>8s}  {'mu':>8s}")
        for g, names in sorted(groups.items()):
            gv_n = _flatten_group(grads, names).norm().item()
            dv_n = _flatten_group(dw,    names).norm().item()
            print(f"  {g:20s}  {dv_n:10.5f}  {gv_n:10.5f}  {delta[g]:8.4f}  {mu_map[g]:8.5f}")
        print(f"  → µ range: [{min(mu_map.values()):.5f}, {max(mu_map.values()):.5f}]"
              f"  mean={np.mean(list(mu_map.values())):.5f}")
    # -----------------------------------------------------------------------

    return mu_map


# --------------------------------------------------------------------------- #
# Local training with per-layer proximal term
# --------------------------------------------------------------------------- #
def local_train(model, global_state, mu_per_group, loader, cfg, device,
                _round=0, _cid=0):
    model.load_state_dict(global_state)
    model.train()
    debug  = cfg.get("debug", False) and _cid == 0 and _round == 0
    anchor = {n: global_state[n].to(device) for n, _ in model.named_parameters()}
    mu_per_param = {n: mu_per_group[group_key(n)] for n, _ in model.named_parameters()}

    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler('cuda')
    prox_params = [(n, p, mu_per_param[n])
                   for n, p in model.named_parameters() if mu_per_param[n] > 0]

    printed_debug = False
    for epoch in range(cfg["local_epochs"]):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)

            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                loss = F.cross_entropy(model(x), y)
                if cfg["method"] != "fedavg" and prox_params:
                    prox = sum(mu * torch.sum((p - anchor[n]) ** 2)
                            for n, p, mu in prox_params)
                    # -------------------------------------------- DEBUG
                    if debug and not printed_debug:
                        ce_val   = loss.item()
                        prox_val = (0.5 * prox).item()
                        print(f"\n[DEBUG local_train r={_round} cid={_cid} epoch=0 batch=0]")
                        print(f"  CE loss   = {ce_val:.4f}")
                        print(f"  Prox term = {prox_val:.4f}  (0.5 * Σ µ_l||p-anchor||²)")
                        print(f"  Ratio prox/CE = {prox_val / max(ce_val, 1e-8):.3f}")
                        print(f"  # prox params = {len(prox_params)}")
                        printed_debug = True
                    # ---------------------------------------------
                    loss = loss + 0.5 * prox

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def aggregate(client_states, weights):
    total = sum(weights)
    agg   = copy.deepcopy(client_states[0])
    for k in agg:
        if agg[k].dtype.is_floating_point:
            agg[k] = sum(w * cs[k] for w, cs in zip(weights, client_states)) / total
        else:
            agg[k] = client_states[0][k]
    return agg


@torch.no_grad()
def evaluate(model, state, test_loader, device):
    model.load_state_dict(state)
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred  = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    return correct / total


# --------------------------------------------------------------------------- #
# FL main loop
# --------------------------------------------------------------------------- #
def run_fl(cfg, data_root="./data"):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[device={device}] {cfg['method']:9s} µ={cfg['mu_base']} "
          f"α_sig={cfg.get('sigmoid_alpha','-')} dir_alpha={cfg['alpha']} seed={cfg['seed']}")

    train, test, n_classes = load_datasets(cfg["dataset"], data_root)
    parts  = dirichlet_partition(train.targets, cfg["n_clients"], cfg["alpha"], cfg["seed"])
    nw     = cfg.get("num_workers", 4)
    client_loaders = [
        DataLoader(Subset(train, ix), batch_size=cfg["batch_size"],
                   shuffle=True, drop_last=False,
                   num_workers=nw, pin_memory=True,
                   persistent_workers=(nw > 0))
        for ix in parts
    ]
    client_sizes = [len(ix) for ix in parts]
    test_loader  = DataLoader(test, batch_size=256, shuffle=False,
                              num_workers=nw, pin_memory=True)

    model        = build_model(n_classes).to(device)
    if hasattr(torch, 'compile'):
        model = torch.compile(model) # 모델 컴파일 (첫 라운드만 조금 걸리고 이후 훨씬 빠름)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prev_state   = None
    rng          = np.random.default_rng(cfg["seed"])

    history  = []
    n_select = max(1, int(round(cfg["participation"] * cfg["n_clients"])))

    for r in range(cfg["rounds"]):
        selected = rng.choice(cfg["n_clients"], size=n_select, replace=False)
        client_states, weights = [], []

        # === 속도 최적화: 서버 상태를 라운드 시작 시 한 번만 GPU로 이동 ===
        global_state_gpu = {k: v.to(device, non_blocking=True) for k, v in global_state.items()}
        prev_state_gpu = {k: v.to(device, non_blocking=True) for k, v in prev_state.items()} if prev_state is not None else None

        for cid in selected:
            loader = client_loaders[cid]
            if len(loader.dataset) == 0:
                continue
            mu  = compute_mu_per_group(model, global_state_gpu, prev_state_gpu,
                                       loader, cfg, device, _round=r, _cid=cid)
            ns  = local_train(model, global_state_gpu, mu, loader, cfg, device,
                              _round=r, _cid=cid)
            client_states.append(ns)
            weights.append(client_sizes[cid])

        new_global = aggregate(client_states, weights)
        prev_state, global_state = global_state, new_global

        if (r + 1) % cfg["eval_every"] == 0 or r == cfg["rounds"] - 1:
            acc = evaluate(model, global_state, test_loader, device)
            history.append((r + 1, acc))
            print(f"  round {r+1:3d}/{cfg['rounds']}  test_acc={acc:.4f}")

    result = {"config": cfg, "history": history, "final_acc": history[-1][1]}
    save_result(result)
    return result


# --------------------------------------------------------------------------- #
# Entry point — diagnostic sweep (Option A: find the bug)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Table 3용 diagnostic 런 — 2 rounds만, debug=True
    # cos 런: gv_norm / delta_cos / mu_cos 출력
    # mag 런: delta_mag(=gv_norm) / mu_mag 출력
    configs = [
        dict(default_config(),
             method="layerprox", mu_base=0.01, sigmoid_alpha=5.0,
             use_magnitude=False,
             alpha=0.1, seed=0, rounds=2, eval_every=2,
             debug=True, num_workers=4),
        dict(default_config(),
             method="layerprox", mu_base=0.01, sigmoid_alpha=1.0,
             use_magnitude=True,
             alpha=0.1, seed=0, rounds=2, eval_every=2,
             debug=True, num_workers=4),
    ]
    for c in configs:
        run_fl(c, data_root="./data")