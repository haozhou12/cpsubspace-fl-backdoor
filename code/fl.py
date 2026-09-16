"""
Rebuilt FL harness (2026-09-11) after the loss of the original scratchpad scripts.

PURPOSE OF THIS FILE
--------------------
Decides round-6 audit finding B1: the previous implementation clipped only
over-long client updates,

    gbar_i = g_i * (|g0|/|g_i|)   if |g_i| > |g0|   else   g_i        [CLIP]

whereas published RFLPA (NeurIPS 2024, Eq. 3) normalizes EVERY client update
to the reference norm, unconditionally,

    gbar_i = g_i * (|g0|/|g_i|)                                       [NORM]

Under [NORM] the attack's "exact norm match => the clip is the identity"
property is vacuous, because every update is rescaled anyway. Whether the
attack still works is an empirical question. This script runs both rules on
identical seeds and partitions so the comparison is paired.

Trust and aggregation are otherwise as in the paper:
    TS_i = max(0, cos(g_i, g0)),    G = sum_i TS_i gbar_i / sum_i TS_i
Note cos() is scale-invariant, so the normalization rule does not change TS.
It changes only the MASS each honest client contributes: under [NORM] a short
honest update is amplified to |g0|, under [CLIP] it is left short.

NOTE ON PROVENANCE: this is a NEW implementation. Its numbers are new and must
be entered in FACTS.md as a new experiment. They are NOT reproductions of the
lost runs and must never be conflated with them.
"""
import argparse, gzip, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# ---------------------------------------------------------------- data
def _idx(path, kind):
    with gzip.open(path, "rb") as f:
        b = f.read()
    if kind == "img":
        n, r, c = int.from_bytes(b[4:8], "big"), int.from_bytes(b[8:12], "big"), int.from_bytes(b[12:16], "big")
        return np.frombuffer(b, np.uint8, offset=16).reshape(n, r, c)
    return np.frombuffer(b, np.uint8, offset=8)

def load_fmnist():
    Xtr = _idx(f"{DATA}/train-images-idx3-ubyte.gz", "img")
    Ytr = _idx(f"{DATA}/train-labels-idx1-ubyte.gz", "lbl")
    Xte = _idx(f"{DATA}/t10k-images-idx3-ubyte.gz", "img")
    Yte = _idx(f"{DATA}/t10k-labels-idx1-ubyte.gz", "lbl")
    MU, SD = 0.2860, 0.3530
    f = lambda X: ((torch.from_numpy(X.copy()).float() / 255.0 - MU) / SD).unsqueeze(1)
    return f(Xtr), torch.from_numpy(Ytr.copy()).long(), f(Xte), torch.from_numpy(Yte.copy()).long(), MU, SD

def load_cifar10():
    """Raw binary format: 10000 records/file, each 1 label byte + 3072 pixel bytes
    (1024 R, then G, then B). torchvision is not installed, so parse directly."""
    d = f"{DATA}/cifar-10-batches-bin"
    def rd(fn):
        b = np.frombuffer(open(f"{d}/{fn}", "rb").read(), np.uint8).reshape(-1, 3073)
        return b[:, 1:].reshape(-1, 3, 32, 32), b[:, 0]
    xs, ys = zip(*[rd(f"data_batch_{i}.bin") for i in range(1, 6)])
    Xtr, Ytr = np.concatenate(xs), np.concatenate(ys)
    Xte, Yte = rd("test_batch.bin")
    MU, SD = 0.4734, 0.2516                      # channel-pooled scalars
    f = lambda X: (torch.from_numpy(X.copy()).float() / 255.0 - MU) / SD
    return f(Xtr), torch.from_numpy(Ytr.copy()).long(), f(Xte), torch.from_numpy(Yte.copy()).long(), MU, SD

# ---------------------------------------------------------------- model
class CompactCNN(nn.Module):
    """2-conv compact CNN. GroupNorm, never BatchNorm: BN running stats are
    buffers, not parameters, so under update-based FL aggregation they are
    neither aggregated nor carried across rounds (documented FL pitfall)."""
    def __init__(self, ncls=10, cin=1, side=28):
        super().__init__()
        self.c1 = nn.Conv2d(cin, 16, 3, padding=1); self.n1 = nn.GroupNorm(4, 16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.n2 = nn.GroupNorm(4, 32)
        self.fc = nn.Linear(32 * (side // 4) ** 2, ncls)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.n1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.n2(self.c2(x))), 2)
        return self.fc(x.flatten(1))

def flat(ps):   return torch.cat([p.detach().reshape(-1) for p in ps])
def set_flat(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n

# ---------------------------------------------------------------- task
TGT = 0          # backdoor target label
TRIG = 3         # 3x3 trigger, bottom-right

def make_trig(MU, SD):
    val = (1.0 - MU) / SD
    def add(x):
        x = x.clone(); x[:, :, -TRIG:, -TRIG:] = val; return x
    return add

# ---------------------------------------------------------------- one run
def run(rule, seed, tau=0.97, M=20, mf=0.3, rounds=40, root_n=100,
        steps=30, bs=64, lr=0.05, attack="cpsub", ds="fmnist", log=None):
    """rule: 'norm' = published RFLPA Eq.(3) (unconditional), 'clip' = old one-sided."""
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr, Ytr, Xte, Yte, MU, SD = load_fmnist() if ds == "fmnist" else load_cifar10()
    cin, side = Xtr.shape[1], Xtr.shape[2]
    add_trig = make_trig(MU, SD)
    Xte_d, Yte_d = Xte.to(DEV), Yte.to(DEV)
    nont = (Yte_d != TGT).nonzero().squeeze(1)          # ASR on non-target inputs only
    Xbd = add_trig(Xte_d[nont])

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(Ytr))
    cl = np.array_split(perm[root_n:], M)               # IID split, disjoint from root
    root = perm[:root_n]
    nmal = int(round(mf * M)); mal = set(range(nmal))
    mpool = np.concatenate([cl[i] for i in sorted(mal)])

    mk = lambda: CompactCNN(cin=cin, side=side).to(DEV)
    w = flat(mk().parameters())

    def local(w0, idx, steps=steps, poison=False):
        m = mk(); set_flat(m, w0); m.train()
        opt = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9)
        idx = np.asarray(idx)
        if len(idx) == 0: return w0.clone()
        for _ in range(steps):
            b = idx[np.random.randint(0, len(idx), size=min(bs, len(idx)))]
            xb, yb = Xtr[b].to(DEV), Ytr[b].to(DEV)
            if poison:                                   # half-batch poisoned
                k = len(b) // 2
                xb = torch.cat([xb[:k], add_trig(xb[k:])])
                yb = torch.cat([yb[:k], torch.full((len(b) - k,), TGT, device=DEV)])
            opt.zero_grad(); F.cross_entropy(m(xb), yb).backward(); opt.step()
        return flat(m.parameters())

    @torch.no_grad()
    def ev(w0):
        m = mk(); set_flat(m, w0); m.eval()
        cl_acc = (m(Xte_d).argmax(1) == Yte_d).float().mean().item()
        asr = (m(Xbd).argmax(1) == TGT).float().mean().item()
        return cl_acc, asr

    hist = []
    for r in range(rounds):
        g0 = local(w, root, steps=20)
        g0 = g0 - w
        n0 = g0.norm() + 1e-12

        gmal = None
        if attack == "cpsub" and nmal > 0:
            gp = local(w, mpool, steps=10) - w           # gray-box anchor estimate
            e0h = gp / (gp.norm() + 1e-12)
            b = local(w, mpool, steps=steps, poison=True) - w
            v = b - torch.dot(b, e0h) * e0h
            v = v / (v.norm() + 1e-12)
            gmal = (tau * e0h + math.sqrt(1 - tau * tau) * v) * n0
            cos_anchor = float(torch.dot(e0h, g0) / n0)
        else:
            cos_anchor = float("nan")

        ups = []
        for i in range(M):
            if i in mal and attack == "cpsub":
                ups.append(gmal.clone())
            elif i in mal and attack == "scaling":
                # UNTARGETED control: 10x a CLEAN local update. Does not attempt
                # the trigger, so low ASR here does not mean "rejected". (audit B6)
                ups.append(10.0 * (local(w, cl[i]) - w))
            elif i in mal and attack == "signflip":
                ups.append(-(local(w, cl[i]) - w))       # UNTARGETED control
            else:
                ups.append(local(w, cl[i]) - w)

        num = torch.zeros_like(w); den = 0.0; tsm, tsb = [], []
        # mass diagnostics: the clip rule and the norm rule differ ONLY in how much
        # mass each honest client contributes. Track it so the adversary's influence
        # can be decomposed into a trust factor and a mass factor.
        rat_ben, infl_mal, infl_tot = [], 0.0, 0.0
        for i, g in enumerate(ups):
            ng = g.norm() + 1e-12
            if rule == "norm":
                gb = g * (n0 / ng)                        # RFLPA Eq.(3): unconditional
            elif rule == "clip":
                gb = g * (n0 / ng) if ng > n0 else g      # old one-sided
            else:
                raise ValueError(rule)
            t = max(0.0, float(torch.dot(g, g0) / (ng * n0)))
            num += t * gb; den += t
            infl = t * float(gb.norm()); infl_tot += infl
            if i in mal:
                tsm.append(t); infl_mal += infl
            else:
                tsb.append(t); rat_ben.append(float(ng / n0))
        w = w + (num / den if den > 1e-9 else torch.zeros_like(w))

        c, a = ev(w)
        hist.append(dict(r=r, clean=c, asr=a, ts_mal=float(np.mean(tsm)) if tsm else 0.0,
                         ts_ben=float(np.mean(tsb)), cos_anchor=cos_anchor,
                         # mean |g_i|/|g0| over honest clients: <1 means the old clip
                         # rule leaves them short while the attacker sits at |g0|
                         ben_norm_ratio=float(np.mean(rat_ben)) if rat_ben else float("nan"),
                         # share of sum_i TS_i*|gbar_i| held by the adversary
                         mass_share=infl_mal / infl_tot if infl_tot > 1e-12 else float("nan")))
        if log and (r % 10 == 0 or r == rounds - 1):
            h = hist[-1]
            print(f"  [{rule} s{seed}] r{r:02d} clean={h['clean']:.3f} ASR={h['asr']:.3f} "
                  f"TSmal={h['ts_mal']:.3f} TSben={h['ts_ben']:.3f} cos={h['cos_anchor']:.3f}",
                  flush=True)
    return hist

# ---------------------------------------------------------------- main
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed_start", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.97)
    ap.add_argument("--rules", default="norm,clip")
    ap.add_argument("--attack", default="cpsub")
    ap.add_argument("--ds", default="fmnist", choices=["fmnist","cifar"])
    ap.add_argument("--mf", type=float, default=0.3)
    ap.add_argument("--out", default="b1_ab.json")
    a = ap.parse_args()

    res = {}
    t0 = time.time()
    for rule in a.rules.split(","):
        for s in range(a.seed_start, a.seed_start + a.seeds):
            k = f"{a.ds}|{a.attack}|{rule}|tau{a.tau}|mf{a.mf}|s{s}"
            t1 = time.time()
            h = run(rule, s, tau=a.tau, rounds=a.rounds, attack=a.attack, ds=a.ds, mf=a.mf, log=True)
            res[k] = h
            print(f"{k}: final clean={h[-1]['clean']:.3f} ASR={h[-1]['asr']:.3f} "
                  f"({time.time()-t1:.0f}s)", flush=True)
            json.dump(res, open(a.out, "w"))

    print("\n===== SUMMARY =====", flush=True)
    for rule in a.rules.split(","):
        ks = [k for k in res if f"|{rule}|" in k]
        fa = [res[k][-1]["asr"] for k in ks]
        fc = [res[k][-1]["clean"] for k in ks]
        print(f"{a.ds} {a.attack} {rule:5s} n={len(ks)} finalASR {np.mean(fa):.3f}+-{np.std(fa):.3f} "
              f"{[round(x,3) for x in fa]} succ(>=0.8)={sum(x>=0.8 for x in fa)}/{len(ks)} "
              f"| clean {np.mean(fc):.3f}+-{np.std(fc):.3f}", flush=True)
    print(f"total {time.time()-t0:.0f}s")
    open("B1_DONE", "w").write("done")
