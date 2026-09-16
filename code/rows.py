"""
Print paper-ready rows (mean +- pstd, per-seed, success count) for every
results/*.json, so nothing is transcribed by hand. Run from code/ or paper/.
Usage: python3 rows.py [glob ...]     default: all results/*.json
"""
import glob, json, os, sys
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")

def rows(path):
    d = json.load(open(path)); out = {}
    for k, h in d.items():
        parts = k.split("|"); seed = parts[-1]; cfg = "|".join(parts[:-1])
        out.setdefault(cfg, []).append((seed, h[-1]["asr"], h[-1]["clean"], len(h)))
    for cfg, v in sorted(out.items()):
        v.sort(); a = np.array([x[1] for x in v]); c = np.array([x[2] for x in v]); R = {x[3] for x in v}
        print(f"{os.path.basename(path):26s} {cfg:38s} rounds={R} n={len(v)} "
              f"ASR {a.mean():.3f}+-{a.std():.3f} {np.round(a,3).tolist()} succ={int((a>=0.8).sum())}/{len(v)} "
              f"clean {c.mean():.3f}+-{c.std():.3f}")

if __name__ == "__main__":
    pats = sys.argv[1:] or ["*.json"]
    for p in pats:
        for f in sorted(glob.glob(os.path.join(RES, p))):
            if not f.endswith(".json"): continue
            rows(f)
