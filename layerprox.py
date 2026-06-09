"""
LayerProx: per-layer proximal regularization for federated learning.

Single-file FL simulator that runs BOTH locally (e.g. a GTX 1650, for debugging)
and on Modal (for the full sweep). The scientifically critical pieces -- layer
grouping, the drift signal delta, the per-layer proximal term, the server-side
previous-update direction, and the t=0 warmup -- are implemented here directly.

Methods supported in this scaffold:
    - "fedavg"    : mu_base = 0 (no proximal)
    - "fedprox"   : constant mu = mu_base for every layer (no gating)
    - "layerprox" : mu_l = mu_base * sigmoid(alpha * delta_l), drift-gated per layer

Baselines that already exist elsewhere (SCAFFOLD, FedDyn, MOON, FedCKA, FedLWS)
should be pulled from NIID-Bench / FedDC or run from their official code rather
than reimplemented here -- see the project notes.

Local debug:   python layerprox.py
Modal sweep:   modal run layerprox.py
"""

import copy
import math
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

import modal

# --------------------------------------------------------------------------- #
# Modal setup (only used for the remote sweep; local runs ignore it)
# --------------------------------------------------------------------------- #
app = modal.App("layerprox")
image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchvision", "numpy")
)
# Cache CIFAR across runs so it is not re-downloaded every container.
data_volume = modal.Volume.from_name("layerprox-data", create_if_missing=True)
DATA_DIR = "/data"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def default_config():
    return dict(
        method="layerprox",        # fedavg | fedprox | layerprox
        dataset="cifar10",         # cifar10 | cifar100
        n_clients=10,
        participation=1.0,         # fraction of clients per round (1.0 = cross-silo)
        alpha=0.5,                 # Dirichlet concentration (smaller = more non-IID)
        rounds=100,
        local_epochs=5,
        batch_size=256,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        # --- LayerProx / FedProx ---
        mu_base=0.1,               # global proximal scale
        sigmoid_alpha=5.0,         # gate sensitivity ("temperature")
        # --- ablation toggles ---
        per_layer=True,            # False -> use mean delta as a single scalar (device-adaptive)
        gate_direction="divergent",  # "divergent" (delta up -> mu up) | "similar" (FedCKA principle)
        seed=0,
        eval_every=1,
    )


# --------------------------------------------------------------------------- #
# Data: Dirichlet label partition
# --------------------------------------------------------------------------- #
def load_datasets(dataset, root):
    import torchvision
    import torchvision.transforms as T

    if dataset == "cifar10":
        mean, std, DS, n_classes = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261), torchvision.datasets.CIFAR10, 10
    elif dataset == "cifar100":
        mean, std, DS, n_classes = (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), torchvision.datasets.CIFAR100, 100
    else:
        raise ValueError(dataset)

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    test_tf = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    train = DS(root, train=True, download=True, transform=train_tf)
    test = DS(root, train=False, download=True, transform=test_tf)
    return train, test, n_classes


def dirichlet_partition(labels, n_clients, alpha, seed):
    """Return a list of index arrays, one per client, via per-class Dirichlet."""
    rng = np.random.default_rng(seed)
    labels = np.array(labels)
    n_classes = labels.max() + 1
    client_idx = [[] for _ in range(n_clients)]
    for c in range(n_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet(alpha * np.ones(n_clients))
        # cut points proportional to props
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for cid, part in enumerate(np.split(idx_c, cuts)):
            client_idx[cid].extend(part.tolist())
    return [np.array(ix) for ix in client_idx]


# --------------------------------------------------------------------------- #
# Model: CIFAR-adapted ResNet-20
# --------------------------------------------------------------------------- #
class _BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False);      self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x))); out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

def build_model(n_classes):
    class ResNet20(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(16)
            self.layer1 = nn.Sequential(_BasicBlock(16,16), _BasicBlock(16,16), _BasicBlock(16,16))
            self.layer2 = nn.Sequential(_BasicBlock(16,32,2), _BasicBlock(32,32), _BasicBlock(32,32))
            self.layer3 = nn.Sequential(_BasicBlock(32,64,2), _BasicBlock(64,64), _BasicBlock(64,64))
            self.fc = nn.Linear(64, nc)
        def forward(self, x):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer3(self.layer2(self.layer1(out)))
            return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))
    return ResNet20(n_classes)

def group_key(name):
    """Map a parameter name to its layer group.

    Granularity ~ residual block. e.g. conv1, bn1, layer1.0, layer1.1, ..., fc.
    This is a design knob: too fine -> noisy delta, too coarse -> low resolution.
    """
    parts = name.split(".")
    if parts[0].startswith("layer"):
        return ".".join(parts[:2])      # e.g. "layer3.1"
    return parts[0]                      # "conv1", "bn1", "fc"


# --------------------------------------------------------------------------- #
# LayerProx: per-layer coefficient from the drift signal
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _flatten_group(named_tensors, names):
    return torch.cat([named_tensors[n].flatten() for n in names])


def compute_mu_per_group(model, global_state, prev_state, loader, cfg, device):
    """Return {group: mu} for one client this round.

    delta_l = 1 - cos(g_l, dw_l), where g_l is the one-step local gradient at w^t
    and dw_l = w^{t-1}_l - w^t_l is the previous global update direction.
    """
    method = cfg["method"]
    if method == "fedavg":
        return defaultdict(lambda: 0.0)
    if method == "fedprox":
        return defaultdict(lambda: cfg["mu_base"])

    # ---- LayerProx ----
    # group the parameter names
    groups = defaultdict(list)
    for n, _ in model.named_parameters():
        groups[group_key(n)].append(n)

    # t = 0 warmup: no previous update -> neutral default mu_base/2 = mu_base*sigmoid(0)
    if prev_state is None:
        return {g: cfg["mu_base"] * 0.5 for g in groups}

    # one-step gradient at the global weights w^t
    model.load_state_dict(global_state)
    model.train()
    model.zero_grad(set_to_none=True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    F.cross_entropy(model(x), y).backward()
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}

    # dw_l = w^{t-1} - w^t  (direction of the previous global move, in +grad direction)
    dw = {n: (prev_state[n] - global_state[n]).to(device) for n in grads}

    delta = {}
    for g, names in groups.items():
        gv = _flatten_group(grads, names)
        dv = _flatten_group(dw, names)
        if dv.norm() < 1e-12 or gv.norm() < 1e-12:
            delta[g] = 1.0  # undefined direction -> treat as orthogonal (neutral-ish)
        else:
            cos = F.cosine_similarity(gv.unsqueeze(0), dv.unsqueeze(0)).item()
            delta[g] = 1.0 - cos          # in [0, 2]

    # ablation: collapse to a single scalar (device-adaptive, not per-layer)
    if not cfg["per_layer"]:
        mean_d = float(np.mean(list(delta.values())))
        delta = {g: mean_d for g in delta}

    # ablation: flip the gate to the FedCKA principle (regularize *aligned* layers)
    if cfg["gate_direction"] == "similar":
        delta = {g: 2.0 - d for g, d in delta.items()}

    a = cfg["sigmoid_alpha"]
    return {g: cfg["mu_base"] * (1.0 / (1.0 + math.exp(-a * delta[g]))) for g in delta}


# --------------------------------------------------------------------------- #
# Local training with the per-layer proximal term
# --------------------------------------------------------------------------- #
def local_train(model, global_state, mu_per_group, loader, cfg, device):
    model.load_state_dict(global_state)
    model.train()
    anchor = {n: global_state[n].to(device) for n, _ in model.named_parameters()}
    mu_per_param = {n: mu_per_group[group_key(n)] for n, _ in model.named_parameters()}

    opt = torch.optim.SGD(model.parameters(), lr=cfg["lr"],
                          momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    # === 추가된 부분: 매 배치마다 for문을 돌지 않도록 미리 리스트로 캐싱 ===
    prox_params = [(n, p, mu_per_param[n]) for n, p in model.named_parameters() if mu_per_param[n] > 0]
    for _ in range(cfg["local_epochs"]):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            if cfg["method"] != "fedavg" and prox_params:
                # 리스트 내포로 연산 후 한 번에 더하기
                prox = sum(mu * torch.sum((p - anchor[n]) ** 2) for n, p, mu in prox_params)
                loss = loss + 0.5 * prox
            loss.backward()
            opt.step()
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Aggregation (FedAvg, weighted by sample count)
# --------------------------------------------------------------------------- #
def aggregate(client_states, weights):
    total = sum(weights)
    agg = copy.deepcopy(client_states[0])
    for k in agg:
        if agg[k].dtype.is_floating_point:
            agg[k] = sum(w * cs[k] for w, cs in zip(weights, client_states)) / total
        else:
            agg[k] = client_states[0][k]  # e.g. num_batches_tracked
    return agg


@torch.no_grad()
def evaluate(model, state, test_loader, device):
    model.load_state_dict(state)
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


# --------------------------------------------------------------------------- #
# FL main loop  (pure PyTorch -- runs anywhere)
# --------------------------------------------------------------------------- #
def run_fl(cfg, data_root="./data"):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device={device}] config={cfg}")

    train, test, n_classes = load_datasets(cfg["dataset"], data_root)
    parts = dirichlet_partition(train.targets, cfg["n_clients"], cfg["alpha"], cfg["seed"])
    client_loaders = [
        DataLoader(
            Subset(train, ix), 
            batch_size=cfg["batch_size"], 
            shuffle=True, 
            drop_last=False,
            num_workers=0,          # window에선 0이 맞음
            pin_memory=True,        # CPU -> GPU 데이터 전송 속도 극대화
            # persistent_workers=True # 에포크마다 워커를 재시작하는 오버헤드 방지
        )
        for ix in parts
    ]
    client_sizes = [len(ix) for ix in parts]
    test_loader = DataLoader(test, batch_size=256, shuffle=False)

    model = build_model(n_classes).to(device)
    global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    prev_state = None  # w^{t-1}; None at t=0
    rng = np.random.default_rng(cfg["seed"])

    history = []
    n_select = max(1, int(round(cfg["participation"] * cfg["n_clients"])))
    for r in range(cfg["rounds"]):
        selected = rng.choice(cfg["n_clients"], size=n_select, replace=False)
        client_states, weights = [], []
        for cid in selected:
            loader = client_loaders[cid]
            if len(loader.dataset) == 0:
                continue
            mu = compute_mu_per_group(model, global_state, prev_state, loader, cfg, device)
            new_state = local_train(model, global_state, mu, loader, cfg, device)
            client_states.append(new_state)
            weights.append(client_sizes[cid])

        new_global = aggregate(client_states, weights)
        prev_state, global_state = global_state, new_global  # shift: prev = W_r, global = W_{r+1}

        if (r + 1) % cfg["eval_every"] == 0 or r == cfg["rounds"] - 1:
            acc = evaluate(model, global_state, test_loader, device)
            history.append((r + 1, acc))
            print(f"round {r+1:3d}/{cfg['rounds']}  test_acc={acc:.4f}")

    return {"config": cfg, "history": history, "final_acc": history[-1][1]}


# --------------------------------------------------------------------------- #
# Modal remote entrypoint
# --------------------------------------------------------------------------- #
@app.function(gpu="A10G", image=image, timeout=60 * 60 * 4,
              volumes={DATA_DIR: data_volume})
def run_remote(cfg):
    return run_fl(cfg, data_root=DATA_DIR)


@app.local_entrypoint()
def main():
    # Example sweep: launch several configs in parallel with modal.map-style fan-out.
    base = default_config()
    configs = []
    for method in ["fedprox", "layerprox"]:
        for alpha in [0.1, 0.5]:
            c = dict(base, method=method, alpha=alpha, rounds=100)
            configs.append(c)
    results = list(run_remote.map(configs))
    for res in results:
        c = res["config"]
        print(f"{c['method']:9s} alpha={c['alpha']}  final_acc={res['final_acc']:.4f}")


# --------------------------------------------------------------------------- #
# Local debug entrypoint (no Modal): python layerprox.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    base = default_config()
    base.update(n_clients=10, rounds=25, local_epochs=5, alpha=0.1, eval_every=5)
    configs = [dict(base, method="fedavg")]
    for mu in [0.01, 0.1, 1.0]:
        configs.append(dict(base, method="fedprox", mu_base=mu))
    for a in [2.0, 5.0]:
        configs.append(dict(base, method="layerprox", mu_base=0.01, sigmoid_alpha=a))
    for a in [2.0, 5.0]:
        configs.append(dict(base, method="layerprox", mu_base=0.1, sigmoid_alpha=a))
    for s in [0, 1]:
        for c in configs:
            cfg = dict(c, seed=s)
            out = run_fl(cfg, data_root="./data")
            print(f"=> {cfg['method']:9s} mu={cfg['mu_base']} a={cfg['sigmoid_alpha']} seed={s} acc={out['final_acc']:.4f}")