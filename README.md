# Closed-form subspace backdoor vs. cosine-trust aggregation — harness and results

Everything in the ICASSP 2027 submission is produced by the files in this directory
and reproducible from them. No number in the paper comes from anywhere else.

## Layout
```
code/fl.py             the FL harness (data loading, compact CNN, trust aggregation, cpsub attack, scaling/signflip controls)
code/fl_ctrl.py        harness variant: 'naive' attack (submit poisoned update directly) and COVER=rand (random-direction cover)
code/rows.py           print mean±std / per-seed / success rows for any results/*.json
code/verify_paper.py   recompute every number in paper/icassp/main.tex from results/ and grep for it
code/analyze_b1.py     trust-vs-mass decomposition for the two normalization rules
results/*.json         per-round histories: clean acc, ASR, TS_mal, TS_ben, cos(ê0,g0), mass diagnostics
results/*.log          stdout of the corresponding runs
data/                  Fashion-MNIST (idx.gz) and CIFAR-10 (binary) as downloaded; loaders are in fl.py
paper/icassp/          LaTeX source, make_figs.py (reads results/ directly), audit reports
```

## Reproduce
```bash
# main result (Table 1): both normalization rules, 5 seeds, 120 rounds
python3 fl.py --seeds 5 --rounds 120 --rules norm,clip --tau 0.97 --out r2_long.json
python3 fl.py --seeds 3 --rounds 120 --rules norm --attack none  --out r2_long_none.json
# tau sweep (Fig. 2)
for t in 0.3 0.5 0.7 0.85 0.97 0.995; do python3 fl.py --seeds 5 --rounds 120 --rules norm --tau $t --out e4_tau$t.json; done
# malicious fraction
for m in 0.1 0.2; do python3 fl.py --seeds 5 --rounds 120 --rules norm --tau 0.7 --mf $m --out e5_tau0.7_mf$m.json; done
# CIFAR-10 cross-dataset check
python3 fl.py --ds cifar --seeds 5 --rounds 120 --rules norm --attack none --out e6_cifar_none.json
python3 fl.py --ds cifar --seeds 5 --rounds 120 --rules norm --tau 0.7    --out e6_cifar_tau0.7.json
# untargeted controls
python3 fl.py --seeds 3 --rounds 120 --rules norm --attack scaling  --out e7_scaling.json
python3 fl.py --seeds 3 --rounds 120 --rules norm --attack signflip --out e7_signflip.json
# reframe experiments (naive baseline + component controls; use fl_ctrl.py)
python3 fl_ctrl.py --seeds 5 --rounds 120 --attack naive --rules norm,clip --out a_naive.json          # naive backdoor
for m in 0.1 0.2; do python3 fl_ctrl.py --seeds 5 --rounds 120 --attack naive --rules norm --mf $m --out a_naive_mf$m.json; done
COVER=rand python3 fl_ctrl.py --seeds 5 --rounds 120 --rules norm --tau 0.97 --out c_randanchor_tau0.97.json  # random-cover control
COVER=rand python3 fl_ctrl.py --seeds 5 --rounds 120 --rules norm --tau 0.7  --out c_randanchor_tau0.7.json
# then
python3 verify_paper.py      # every paper number vs results/
cd ../paper/icassp && python3 make_figs.py && pdflatex main && bibtex main && pdflatex main && pdflatex main
```
One 120-round Fashion-MNIST run takes ~2.5 min on an RTX 3090 when alone on the GPU; CIFAR-10 ~8 min.

## Conventions (fixed; the paper states them)
- `--rules norm` = published RFLPA Eq. (3), `ḡ_i = (‖g0‖/‖g_i‖) g_i` for every client (unconditional).
  `clip` = the one-sided variant `min(1, ‖g0‖/‖g_i‖)` that leaves short updates unchanged.
- Trust `TS_i = max(0, cos(g_i, g0))`; aggregate = trust-weighted mean of normalized updates.
- Gray-box: the adversary estimates the reference direction with 10 local steps on its own data;
  it never reads `g0`, the root set, or any honest update (see `run()` — there is no such path).
- ASR is measured on triggered test inputs whose true label is not the target class.
- Std is population std (ddof=0). Success = final ASR ≥ 0.8. "Late" statistics = last 30 rounds.
- GroupNorm, never BatchNorm (BN running stats are buffers, not parameters, and are not aggregated).
- Seeds 0..4; `--seed_start` lets a top-up run add seeds to an existing cell without re-running.

## Provenance note
An earlier implementation and ~420 runs were lost on 2026-09-10 (see `../DATA-LOSS-2026-09-10.md`).
Nothing here descends from them; all results were produced by this harness between 2026-09-11 and 09-14.
