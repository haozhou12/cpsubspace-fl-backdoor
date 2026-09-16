"""
Analyze the B1 paired A/B (RFLPA Eq.(3) unconditional normalization vs the old
one-sided clip) and isolate WHICH channel the attack's advantage came through.

Under both rules trust is identical in form (cosine is scale-invariant), so the
adversary's weight share is

    w_A = rho*TS_mal / (rho*TS_mal + (1-rho)*TS_ben)

and is a pure TRUST-channel quantity. What the two rules differ in is MASS: the
old clip leaves an honest update shorter than |g0| untouched, so the attacker's
full-norm update carries more mass than an average honest one; RFLPA's rule
rescales every honest update up to |g0|, equalizing mass exactly.

So: if the attack weakens under 'norm' while w_A is unchanged, the lost
advantage was the MASS asymmetry -- an artifact of the wrong clip rule, not a
property of cosine-trust aggregation.
"""
import json, sys
import numpy as np

RHO = 0.3

def wa(ts_mal, ts_ben):
    d = RHO * ts_mal + (1 - RHO) * ts_ben
    return RHO * ts_mal / d if d > 1e-12 else float("nan")

def summarize(path, label):
    res = json.load(open(path))
    rules = sorted({k.split("|")[1] for k in res})
    print(f"\n===== {label} =====")
    out = {}
    for rule in rules:
        ks = sorted(k for k in res if f"|{rule}|" in k)
        fa = np.array([res[k][-1]["asr"] for k in ks])
        fc = np.array([res[k][-1]["clean"] for k in ks])
        # late-window trust channel (last quarter of training)
        wl, wm = [], []
        for k in ks:
            h = res[k]; n = len(h); late = h[int(0.75 * n):]
            wl.append(np.mean([wa(x["ts_mal"], x["ts_ben"]) for x in late]))
            wm.append(np.mean([x["cos_anchor"] for x in late]))
        out[rule] = dict(asr=fa, clean=fc, wA=np.array(wl))
        print(f"{rule:5s} n={len(ks)}  finalASR {fa.mean():.3f}+-{fa.std():.3f} "
              f"{[round(x,3) for x in fa]}  succ(>=0.8)={int((fa>=0.8).sum())}/{len(fa)}")
        print(f"      clean {fc.mean():.3f}+-{fc.std():.3f}   "
              f"late w_A {np.mean(wl):.3f}+-{np.std(wl):.3f} (rho={RHO})   "
              f"late cos(e0h,g0) {np.nanmean(wm):.3f}")
    if len(rules) == 2 and all(r in out for r in ("norm", "clip")):
        a, b = out["norm"], out["clip"]
        print(f"\n  paired dASR (norm - clip), per seed: "
              f"{[round(x,3) for x in (a['asr']-b['asr'])]}")
        print(f"  mean dASR = {(a['asr']-b['asr']).mean():+.3f}")
        print(f"  trust channel unchanged? w_A norm {a['wA'].mean():.3f} vs "
              f"clip {b['wA'].mean():.3f}  (d={a['wA'].mean()-b['wA'].mean():+.3f})")
        try:
            from scipy.stats import wilcoxon, ttest_rel
            if len(a["asr"]) >= 3:
                print(f"  paired t-test p={ttest_rel(a['asr'], b['asr']).pvalue:.4f}"
                      f"   (n={len(a['asr'])}; small n, report descriptively)")
        except Exception as e:
            print("  (scipy unavailable:", e, ")")
    return out

if __name__ == "__main__":
    summarize(sys.argv[1] if len(sys.argv) > 1 else "b1_ab.json", "CP-Subspace attack")
    try:
        summarize("b1_none.json", "No attack (control)")
    except FileNotFoundError:
        print("\n(no-attack control not present yet)")
