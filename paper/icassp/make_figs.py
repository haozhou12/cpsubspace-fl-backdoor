"""
ICASSP figures. Every numeric input is read from ../../results/*.json (the
rebuilt-harness results) -- NEVER from the lost runs or from FACTS transcriptions.
  fig1_geometry.pdf   schematic, no data
  fig2_gating.pdf     clean / ASR trajectory, r2_long.json (120 rounds, 5 seeds)
  fig3_tau.pdf        ASR and clean vs tau, e4/e4b (5 seeds where available)
"""
import glob, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, FancyArrowPatch

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")
COL_W = 3.35   # ICASSP column is 86 mm = 3.39 in
C_ANCHOR, C_PAYLOAD, C_ATTACK, C_MUTED, C_FAINT = "#0072B2", "#D55E00", "#000000", "#8C8C8C", "#BDBDBD"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Nimbus Roman", "STIXGeneral", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                     "axes.linewidth": 0.6, "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})

def _arrow(ax, p, q, color, lw, head, z, alpha=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=f"-|>,head_length={head},head_width={head*0.55}",
                                 mutation_scale=1.0, shrinkA=0, shrinkB=0, linestyle=ls, color=color,
                                 lw=lw, zorder=z, alpha=alpha, joinstyle="miter", capstyle="butt"))

# ------------------------------------------------------------------ Fig. 1
def make_fig1(path):
    a = np.deg2rad(35.0); tau, s = np.cos(a), np.sin(a); gA = np.array([tau, s])
    fig, ax = plt.subplots(figsize=(COL_W, 2.55)); ax.set_aspect("equal")
    ax.set_xlim(-0.50, 1.50); ax.set_ylim(-0.50, 1.14); ax.axis("off")
    th = np.linspace(0, np.pi / 2 + 0.16, 300)
    ax.plot(np.cos(th), np.sin(th), ls=(0, (4, 2.2)), lw=0.9, color=C_MUTED, zorder=1)
    la = np.deg2rad(58.0)
    ax.text(1.115*np.cos(la), 1.115*np.sin(la), r"$\|g\|=\|g_0\|$", color="#6E6E6E", fontsize=8.5,
            rotation=-32, rotation_mode="anchor", ha="center", va="center")
    ax.plot([-0.20, 1.46], [0, 0], lw=0.5, color=C_FAINT, zorder=0)
    ax.plot([0, 0], [-0.16, 1.10], lw=0.5, color=C_FAINT, zorder=0)
    _arrow(ax, (0, 0), (1.0, 0), C_ANCHOR, 0.8, 4.0, 4, alpha=0.55)
    ax.text(1.04, 0.05, r"$\hat{e}_0$", color=C_ANCHOR, fontsize=9, ha="left", va="bottom")
    ax.text(1.04, -0.04, "estimated\nreference dir.", color=C_ANCHOR, fontsize=8.5, ha="left", va="top", linespacing=1.1)
    _arrow(ax, (0, 0), (0, 1.0), C_PAYLOAD, 0.8, 4.0, 4, alpha=0.55)
    ax.text(-0.085, 1.00, r"$v\perp\hat{e}_0$", color=C_PAYLOAD, fontsize=9, ha="right", va="center")
    ax.text(-0.085, 0.895, "backdoor dir.", color=C_PAYLOAD, fontsize=8.5, ha="right", va="center")
    _arrow(ax, (0, 0), (tau, 0), C_ANCHOR, 2.6, 6.0, 5)
    ax.text(tau/2, -0.075, r"$\tau\,\hat{e}_0$", color=C_ANCHOR, fontsize=8.5, ha="center", va="top")
    ax.text(tau/2, -0.185, "cover", color=C_ANCHOR, fontsize=8.5, ha="center", va="top")
    _arrow(ax, (tau, 0), (tau, s), C_PAYLOAD, 2.6, 6.0, 5)
    ax.text(tau+0.05, s/2+0.03, r"$\sqrt{1-\tau^{2}}\;v$", color=C_PAYLOAD, fontsize=8.5, ha="left", va="center")
    ax.text(tau+0.05, s/2-0.08, "payload", color=C_PAYLOAD, fontsize=8.5, ha="left", va="center")
    _arrow(ax, (0, 0), tuple(gA), C_ATTACK, 2.0, 7.0, 6)
    ax.text(0.30, 0.40, r"$g_A$", color=C_ATTACK, fontsize=9.5, ha="center", va="center")
    ax.add_patch(Arc((0, 0), 0.62, 0.62, theta1=0, theta2=35.0, lw=0.8, color=C_ATTACK, zorder=4))
    ax.text(0.385, 0.098, r"$\alpha$", fontsize=8.5, color=C_ATTACK, ha="center", va="center")
    # Box: proxy cosine (not server trust -- audit N1) and payload budget. No
    # "tau -> 1": the tau sweep shows tau -> 1 is where the attack FAILS.
    ax.text(-0.48, -0.27,
            r"$\cos(g_A,\hat e_0)=\tau$; server cosine follows Eq. (5)" "\n"
            r"payload magnitude $\|g_0\|\sqrt{1-\tau^{2}}$, direction free;  $\|g_A\|=\|g_0\|$",
            fontsize=8.5, ha="left", va="top", linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.3", fc="#F5F5F2", ec="#CCCCCC", lw=0.5))
    fig.savefig(path); plt.close(fig); print("wrote", path)

# ------------------------------------------------------------------ Fig. 2
def make_fig2(path):
    d = json.load(open(f"{RES}/r2_long.json")); n = json.load(open(f"{RES}/r2_long_none.json"))
    ks = sorted(k for k in d if "|norm|" in k)
    R = np.arange(len(d[ks[0]]))
    A = np.array([[h["asr"] for h in d[k]] for k in ks]); C = np.array([[h["clean"] for h in d[k]] for k in ks])
    Cn = np.array([[h["clean"] for h in n[k]] for k in n]); An = np.array([[h["asr"] for h in n[k]] for k in n])
    fig, ax = plt.subplots(figsize=(COL_W, 1.95))
    ax.fill_between(R, C.min(0), C.max(0), color=C_ANCHOR, alpha=0.15, lw=0)
    ax.plot(R, C.mean(0), color=C_ANCHOR, lw=1.3, label="clean acc. (attacked)")
    ax.plot(R, Cn.mean(0), color=C_ANCHOR, lw=0.9, ls="--", label="clean acc. (no attack)")
    ax.fill_between(R, A.min(0), np.minimum(A.max(0),1.0), color=C_PAYLOAD, alpha=0.15, lw=0)
    ax.plot(R, A.mean(0), color=C_PAYLOAD, lw=1.3, label="ASR (attacked)")
    ax.plot(R, An.mean(0), color=C_PAYLOAD, lw=0.9, ls="--", label="ASR (no attack)")
    ax.set_xlabel("round"); ax.set_ylabel("accuracy / ASR"); ax.set_xlim(0, R[-1]); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4); ax.legend(loc="center right", frameon=False, ncol=1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    fig.savefig(path); plt.close(fig)
    print("wrote", path, f"| n={len(ks)} attacked, n={len(n)} control; final ASR mean {A[:,-1].mean():.3f}")

# ------------------------------------------------------------------ Fig. 3
def _load_tau():
    """Merge e4 (seeds 0-2), e4c (seeds 3-4 top-up) and e4b (seeds 0-4) per tau,
    keyed by seed id so no seed is double-counted; later files override earlier."""
    per = {}
    for pat in ("e4_tau*.json", "e4c_tau*.json", "e4b_tau*.json"):
        for f in sorted(glob.glob(f"{RES}/{pat}")):
            t = float(os.path.basename(f).split("tau")[1].replace(".json", ""))
            d = json.load(open(f))
            for k, h in d.items():
                per.setdefault(t, {})[k.split("|")[-1]] = (h[-1]["asr"], h[-1]["clean"])
    rows = {}
    for t, seeds in sorted(per.items()):
        v = [seeds[s] for s in sorted(seeds)]
        rows[t] = dict(n=len(v), asr=np.array([x[0] for x in v]), clean=np.array([x[1] for x in v]))
    return rows

def make_fig3(path):
    rows = _load_tau()
    n0 = json.load(open(f"{RES}/r2_long_none.json")); c0 = np.mean([n0[k][-1]["clean"] for k in n0])
    T = np.array(list(rows)); B = np.sqrt(1 - T**2)
    asr_m = np.array([rows[t]["asr"].mean() for t in T]); asr_s = np.array([rows[t]["asr"].std() for t in T])
    asr_hi = np.minimum(asr_m+asr_s, 1.0) - asr_m
    cl_m = np.array([rows[t]["clean"].mean() for t in T]); cl_s = np.array([rows[t]["clean"].std() for t in T])
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    ax.errorbar(T, asr_m, yerr=[np.minimum(asr_s, asr_m), asr_hi], color=C_PAYLOAD, marker="o", ms=3.2, lw=1.2, capsize=2, label="final ASR (mean $\\pm$ s.d.)")
    for t in T:
        ax.scatter([t]*rows[t]["n"], rows[t]["asr"], s=6, color=C_PAYLOAD, alpha=0.45, zorder=3, lw=0)
    ax.errorbar(T, cl_m, yerr=cl_s, color=C_ANCHOR, marker="s", ms=3.0, lw=1.2, capsize=2, label="clean acc. (attacked)")
    ax.axhline(c0, color=C_ANCHOR, ls="--", lw=0.8, label="clean acc. (no attack, $n{=}3$)")
    ax.set_xlabel(r"$\tau$"); ax.set_ylabel("final ASR / accuracy"); ax.set_ylim(0, 1.05); ax.set_xlim(0.22, 1.02)
    ax2 = ax.twiny(); ax2.set_xlim(ax.get_xlim()); ax2.set_xticks(T)
    ax2.set_xticklabels([f"{b:.2f}" if t < 0.99 else "" for t, b in zip(T, B)], fontsize=8.5); ax2.set_xlabel(r"payload budget $\sqrt{1-\tau^2}$", fontsize=9)
    ax.grid(alpha=0.25, lw=0.4); ax.legend(loc="lower center", frameon=False, fontsize=8)
    for s in ("right",): ax.spines[s].set_visible(False)
    fig.savefig(path); plt.close(fig)
    print("wrote", path, "| n per tau:", {t: rows[t]["n"] for t in rows})


# ------------------------------------------------------------------ Fig. mechanism
def make_fig_mech(path):
    import numpy as np
    def seeds(f, rule=None):
        d = json.load(open(f)); ks = [k for k in sorted(d) if (rule is None or f"|{rule}|" in k)]
        return np.array([d[k][-1]["asr"] for k in ks])
    conds = [
        ("no\nattack",   seeds(f"{RES}/r2_long_none.json"),                 C_MUTED),
        ("naive\nbackdoor", seeds(f"{RES}/a_naive.json", "norm"),           C_ATTACK),
        ("random\ncover",   seeds(f"{RES}/c_randanchor_tau0.97.json"),      C_ANCHOR),
        ("ours\n(anchor cover)", seeds(f"{RES}/r2_long.json", "norm"),      C_PAYLOAD),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    rng = np.random.RandomState(0)
    for i, (lab, a, col) in enumerate(conds):
        x = i + (rng.rand(len(a)) - 0.5) * 0.28
        ax.scatter(x, a, s=22, color=col, alpha=0.75, zorder=3, edgecolor="white", lw=0.4)
        ax.plot([i-0.22, i+0.22], [a.mean(), a.mean()], color=col, lw=2.0, zorder=2)
    ax.axhline(0.8, color=C_FAINT, ls=":", lw=0.8, zorder=1)
    ax.text(3.45, 0.83, "success", fontsize=9, color="#888", ha="right", va="bottom")
    ax.set_xticks(range(4)); ax.set_xticklabels([c[0] for c in conds], fontsize=9)
    ax.set_ylabel("final ASR"); ax.set_ylim(-0.04, 1.06); ax.set_xlim(-0.5, 3.5)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    fig.savefig(path); plt.close(fig); print("wrote", path)

if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
    make_fig1("figs/fig1_geometry.pdf"); make_fig3("figs/fig3_tau.pdf"); make_fig_mech("figs/fig_mech.pdf")
