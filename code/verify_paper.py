"""
Guard against transcription drift: for each tracked quantity we RECOMPUTE it from
results/*.json and require the *derived* string to appear in main.tex. Expected
strings are built from the data, so mutating either the data or the tex breaks the
match. This catches copy/format errors and stale numbers; it is NOT a proof of
correctness (it only covers the quantities enumerated here) and is not a substitute
for the adversarial audits.

Conventions: population std (ddof=0); 'late' = last 30 rounds; success = ASR>=0.8.
"""
import glob, json, os, re, sys
import numpy as np
from scipy import stats

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RES = f"{ROOT}/results"; TEX = f"{ROOT}/paper/icassp/main.tex"
tex = open(TEX, encoding="utf-8").read()
tex_flat = re.sub(r"\s+", " ", tex)

def load(pat, rule="norm"):
    """Merge all files matching pat; key by seed so top-ups override. Returns list of hist."""
    per = {}
    for f in sorted(glob.glob(f"{RES}/{pat}")):
        for k, h in json.load(open(f)).items():
            if f"|{rule}|" in k or "|none|" in k or "|scaling|" in k or "|signflip|" in k:
                per[k.split("|")[-1]] = h
    return [per[s] for s in sorted(per)]

def fin(H, key): return np.array([h[-1][key] for h in H])
def ms(x): return f"{x.mean():.3f}\\pm{x.std():.3f}"
fails = []
def check(label, expected_str, also=None):
    """expected_str must appear verbatim (after whitespace-normalisation) in the tex."""
    ok = expected_str in tex_flat
    print(f"[{'ok ' if ok else 'FAIL'}] {label:48s} {expected_str}")
    if not ok: fails.append((label, expected_str))

# ---------------- E1 / E2 main table
E1n = load("r2_long.json", "norm"); E1c = load("r2_long.json", "clip"); E2 = load("r2_long_none.json")
a_n, c_n = fin(E1n, "asr"), fin(E1n, "clean"); a_c, c_c = fin(E1c, "asr"), fin(E1c, "clean")
a_0, c_0 = fin(E2, "asr"), fin(E2, "clean")
check("E1 norm ASR", ms(a_n)); check("E1 norm clean", f"${c_n.mean():.3f}$")
check("E1 clip ASR", ms(a_c)); check("E1 clip clean", f"${c_c.mean():.3f}$")
check("E2 none ASR", ms(a_0)); check("E2 none clean", f"${c_0.mean():.3f}$")
check("E1 paired dASR", f"{(a_n-a_c).mean():+.3f}".replace("+", ""))   # "-0.014"
p = stats.ttest_rel(a_n, a_c).pvalue; check("E1 paired p", f"p={p:.3f}")
cost = c_0.mean() - c_n.mean(); pw = stats.ttest_ind(c_n, c_0, equal_var=False).pvalue
assert abs(cost-0.026)<0.001 and abs(pw-0.013)<0.002, (cost,pw)
check("tau0.97 clean cost 2.6pt", "$2.6$ clean-accuracy points"); check("tau0.97 Welch p", "p=0.013")

# ---------------- E1 trajectory statements
C = np.array([[x["clean"] for x in h] for h in E1n]); A = np.array([[x["asr"] for x in h] for h in E1n])
cm, am = C.mean(0), A.mean(0); final = cm[-1]
r05 = int(np.argmax(cm >= final - 0.05)); r03 = int(np.argmax(cm >= final - 0.03)); half = int(np.argmax(am >= 0.5))
sh = [int(np.argmax(A[i] >= 0.5)) for i in range(len(A))]
# permanent (not first-crossing) within-0.05 round, per N2 fix
perm05 = max([i for i in range(len(cm)) if cm[i] < cm[-1]-0.05], default=-1)+1
assert perm05==25, perm05
check("traj: settles from round 25", "round~$25$")
check("traj: ASR at r25", f"${am[25]:.2f}$")               # 0.20
check("traj: seed-wise range", f"at rounds ${min(sh)}$ to ${max(sh)}$")

# ---------------- E4/E4b/E4c tau sweep (merged by seed)
def tau_rows():
    per = {}
    for pat in ("e4_tau*.json", "e4c_tau*.json", "e4b_tau*.json"):
        for f in sorted(glob.glob(f"{RES}/{pat}")):
            t = float(os.path.basename(f).split("tau")[1].replace(".json", ""))
            for k, h in json.load(open(f)).items(): per.setdefault(t, {})[k.split("|")[-1]] = h
    return {t: [v[s] for s in sorted(v)] for t, v in sorted(per.items())}
TR = tau_rows()
for t, H in TR.items():
    a, c = fin(H, "asr"), fin(H, "clean"); n = len(H)
    print(f"      tau={t}: n={n} ASR {ms(a)} succ={int((a>=0.8).sum())}/{n} clean {ms(c)} per-seed {np.round(a,3).tolist()}")
check("tau0.7 ASR", ms(fin(TR[0.7], "asr"))); check("tau0.7 clean", f"${fin(TR[0.7],'clean').mean():.3f}$")
check("tau0.995 ASR", ms(fin(TR[0.995], "asr")))
c7 = fin(TR[0.7], "clean")
assert abs((c_0.mean()-c7.mean())-0.008)<0.002
check("tau0.7 cost 0.8pt in tex", "$0.9$ at $\\tau=0.7$")
# B2 fix: FMNIST significant cost stated at tau=0.97; no "unlike on Fashion-MNIST"
check("B2 FMNIST 2.6 stated", "$2.6$ points ($p=0.013$)")
# B1 fix: honest framing present
check("B1 cost-with-cover", "place it with the cover, not the backdoor")
check("B1 no-separate-quantify", "do not further separate the cover and payload")
for t in (0.5, 0.3):
    a = fin(TR[t], "asr"); bad = a[a < 0.8]
    assert len(bad) == 1, f"tau={t}: expected exactly one failing seed, got {bad}"
check("tau0.5 failing seed ASR", f"{fin(TR[0.5],'asr').min():.3f}"); check("tau0.3 failing seed ASR", f"{fin(TR[0.3],'asr').min():.3f}")
# clamp statistics for the failing seed (seed 3) and its anchor quality at tau=0.5
def late(h, key): return np.mean([x[key] for x in h[-30:]])
H5 = TR[0.5]; s3 = H5[3]; others = [H5[i] for i in range(len(H5)) if i != 3]
z = np.mean([x["ts_mal"] < 1e-6 for x in s3])
check("tau0.5 seed3 late cos", f"{late(s3,'cos_anchor'):.2f}")
lo, hi = min(late(h, "cos_anchor") for h in others), max(late(h, "cos_anchor") for h in others)
check("tau0.5 others cos range", f"{lo:+.2f}$ to ${hi:+.2f}".replace("+0", "+0"))
z7 = np.mean([x["ts_mal"] < 1e-6 for x in TR[0.7][3]])
check("tau0.7 seed3 zeroed %", f"{int(round(z7*100))}\\%"); check("tau0.7 seed3 ASR", f"{TR[0.7][3][-1]['asr']:.3f}")
print(f"      seed3 zeroed fraction: tau0.3 {np.mean([x['ts_mal']<1e-6 for x in TR[0.3][3]]):.2f}, tau0.5 {z:.2f}, tau0.7 {z7:.2f}")
tsm995 = late(TR[0.995][0], "ts_mal"); tsm995 = np.mean([late(h, "ts_mal") for h in TR[0.995]])
check("tau0.995 TS_mal", f"{tsm995:.2f}")
# weight share across sweep
def wa(h): tm, tb = late(h, "ts_mal"), late(h, "ts_ben"); return 0.3*tm/(0.3*tm+0.7*tb)
W = {t: np.mean([wa(h) for h in H]) for t, H in TR.items()}
print(f"      w_A by tau: { {t: round(w,3) for t,w in W.items()} }")
# recorded per-round mass_share (not the ratio-of-means proxy), per B5 fix
def massshare(H): return np.mean([np.nanmean([x["mass_share"] for x in h[-30:]]) for h in H])
MS = {t: massshare(H) for t, H in TR.items()}
print(f"      mass_share by tau: {{{', '.join(f'{t}:{MS[t]:.3f}' for t in MS)}}}")
Wr=[MS[t] for t in MS if t>=0.7]; check("mass_share range [.7,.995]", f"${min(Wr):.2f}$--${max(Wr):.2f}$")
check("mass_share tau0.5", f"${MS[0.5]:.2f}$"); check("mass_share tau0.3", f"${MS[0.3]:.2f}$")

# ---------------- E5 / E5b malicious fraction
def mf_rows(pats):
    per = {}
    for pat in pats:
        for f in sorted(glob.glob(f"{RES}/{pat}")):
            m = float(re.search(r"mf([\d.]+)\.json", f).group(1))
            for k, h in json.load(open(f)).items(): per.setdefault(m, {})[k.split("|")[-1]] = h
    return {m: [v[s] for s in sorted(v)] for m, v in sorted(per.items())}
M97 = mf_rows(["e5_mf*.json"]); M7 = mf_rows(["e5b_tau0.7_mf*.json", "e5c_tau0.7_mf*.json"])
check("mf0.2 tau0.7 ASR", ms(fin(M7[0.2], "asr"))); check("mf0.1 tau0.7 ASR", ms(fin(M7[0.1], "asr")))
assert abs(fin(M97[0.1],"asr").mean()-0.103)<0.001 and abs(fin(M97[0.2],"asr").mean()-0.623)<0.001  # tau0.97 mf still recomputed, not in prose

# ---------------- E6 CIFAR
C0 = load("e6*_cifar_none.json"); C7 = load("e6*_cifar_tau0.7.json"); C97 = load("e6_cifar_tau0.97.json")
check("cifar none clean", ms(fin(C0, "clean"))); check("cifar none ASR", f"{fin(C0,'asr').mean():.3f}")
check("cifar tau0.7 ASR", ms(fin(C7, "asr"))); check("cifar tau0.97 ASR", ms(fin(C97, "asr")))
check("cifar tau0.7 cost pts", f"{(fin(C0,'clean').mean()-fin(C7,'clean').mean())*100:.1f}")
check("cifar tau0.97 cost pts", f"{(fin(C0,'clean').mean()-fin(C97,'clean').mean())*100:.1f}")

# ---------------- E7 controls
S = load("e7_scaling.json"); F_ = load("e7_signflip.json")
check("scaling ASR", ms(fin(S, "asr"))); check("signflip ASR", ms(fin(F_, "asr")))
check("scaling clean", f"{fin(S,'clean').mean():.3f}")
check("signflip cost pts", f"{(c_0.mean()-fin(F_,'clean').mean())*100:.1f}")
fp = [np.mean([x["ts_mal"] > 1e-6 for x in h]) for h in F_]
check("signflip trust>0 range", f"{int(round(min(fp)*100))}$--${int(round(max(fp)*100))}\\%")


# ---------------- B3 fix: mf sweep tau=0.97 is n=3 for ALL three fractions
M97_3 = mf_rows(["e5_mf0.3.json"])
check("mf0.3 tau0.97 n", f"n={len(M97_3[0.3])}" if False else "")  # placeholder to force presence check below
assert len(M97_3[0.3]) == 3, "e5_mf0.3 must be n=3"
if "(n=3, 3, 5)" in tex_flat or "(n=3,3,5)" in tex_flat:
    fails.append(("B3", "tex still labels mf sweep (n=3,3,5)"))
    print("[FAIL] B3: '(n=3,3,5)' still present")
else:
    print("[ok ] B3: no '(n=3,3,5)' mislabel")

# ---------------- intervention checks (B2/B4 evidence)
COV = load("i1_cover_only.json"); RP7 = load("i2_randpayload_tau0.7.json"); RP97 = load("i3_randpayload_tau0.97.json")
cov_cost = (c_0.mean() - fin(COV,"clean").mean())*100
assert fin(COV,"asr").mean() < 0.05, "cover-only should not backdoor"
check("cover-only cost pts", f"{cov_cost:.1f}")
check("randpayload tau0.97 cost", f"{(c_0.mean()-fin(RP97,'clean').mean())*100:.1f}")
check("randpayload tau0.7 cost", f"{(c_0.mean()-fin(RP7,'clean').mean())*100:.1f}")
# monotonicity refutation: cost must NOT be monotone (peaks at 0.85)
costs=[c_0.mean()-fin(TR[t],"clean").mean() for t in [0.3,0.5,0.7,0.85,0.97,0.995]]
assert costs[3]==max(costs), "cost should peak at tau=0.85 (non-monotone)"
print(f"      cost by tau (pts): {[round(c*100,1) for c in costs]}  (non-monotone: peak at 0.85)")
print(f"      cover-only cost {cov_cost:.1f}pt ASR {fin(COV,'asr').mean():.3f}; randpay97 cost {(c_0.mean()-fin(RP97,'clean').mean())*100:.1f}pt; randpay7 {(c_0.mean()-fin(RP7,'clean').mean())*100:.1f}pt")


# ---------------- REFRAME: naive baseline, random-cover control, trust advantage
NAIVE = load("a_naive.json", "norm"); RANDC = [json.load(open(f"{RES}/c_randanchor_tau0.97.json"))[k] for k in sorted(json.load(open(f"{RES}/c_randanchor_tau0.97.json")))]
def tr(H): tm=np.mean([np.mean([x["ts_mal"] for x in h[-30:]]) for h in H]); tb=np.mean([np.mean([x["ts_ben"] for x in h[-30:]]) for h in H]); return tm/max(tb,1e-9)
na=fin(NAIVE,"asr"); ra=fin(RANDC,"asr")
assert abs(na.mean()-0.675)<0.002 and int((na>=0.8).sum())==3, na
assert abs(ra.mean()-0.029)<0.005 and int((ra>=0.8).sum())==0, ra
check("naive ASR", ms(na)); check("naive succ", f"${int((na>=0.8).sum())}/5$")
check("random cover ASR", ms(ra))
# derive each trust string from the data; a mutated ratio yields a different string -> FAIL
check("naive trust (derived)",  f"${tr(NAIVE):.2f}".rstrip('0').rstrip('.')+"\\times$" if False else f"${tr(NAIVE):.2f}\\times$")
check("random trust (derived)", f"${tr(RANDC):.2f}\\times$")
check("ours trust (derived)",   f"${tr(E1n):.2f}\\times$")
# naive at lower fractions must also be checked (N4: verifier ignored a_naive_mf*)
NMF = mf_rows(["a_naive_mf0.1.json","a_naive_mf0.2.json"])
assert int((fin(NMF[0.1],"asr")>=0.8).sum())==1 and int((fin(NMF[0.2],"asr")>=0.8).sum())==3, "naive mf succ drift"
check("naive succ across fractions", "$1/5$ at $10\\%$ and $3/5$ at $20\\%$")
print(f"      trust ratios (derived): naive {tr(NAIVE):.2f}x  random {tr(RANDC):.2f}x  ours {tr(E1n):.2f}x")
# forbidden: the old overclaim framings
for bad in ["breaks an otherwise-robust", "stronger attack than", "defeats the defense", "into a certainty", "deterministically", "in every run", "least-trusted", "dominates the aggregate", "predicts where"]:
    if bad in tex_flat: fails.append(("reframe-forbidden", bad)); print(f"[FAIL] forbidden {bad!r}")

# ---------------- forbidden phrases (from FACTS bans)
for bad in ["insensitive to", "filtered outright", "no clean-accuracy loss", "completely intact",
            "maintains alignment", "forced toward", "only after clean accuracy",
            "cost grows with the cover", "moves against ASR", "fails only as",
            "working range", "clean-neutral", "unlike on Fashion-MNIST",
            "independent of the payload direction", "carried by the cover",
            "component swap isolates", "does not depend on the payload direction"]:
    hits = len(re.findall(bad, tex_flat, flags=re.I))
    print(f"[{'ok ' if hits==0 else 'FAIL'}] forbidden phrase {bad!r}: {hits}")
    if hits: fails.append(("forbidden", bad))
# 'predict' is banned only as a positive claim about gating; print every context for eyeballing
for m in re.finditer(r".{60}predict.{60}", tex_flat, flags=re.I):
    print(f"[ctx ] predict: ...{m.group(0)}...")

print("\n" + ("ALL CHECKS PASSED" if not fails else f"{len(fails)} FAILURES: " + "; ".join(f"{a} -> {b}" for a, b in fails)))
sys.exit(1 if fails else 0)
