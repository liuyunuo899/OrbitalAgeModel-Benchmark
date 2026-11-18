# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 11:03:20 2026

@author: 喝水吨吨吨
"""

# -*- coding: utf-8 -*-
"""
exp_c2_deployable.py — Round-1 supplement S2: deployable diagnostic statistics
================================================================================
Reviewer Round-1 criticism: Exp C's flagship diagnostic (cycle-count deficit
vs duration bias, r = -0.999) used Ne_true = true_duration/0.405, which is
NOT available at deployment, and duration ~ (N_a-1) x 0.405 makes the
correlation near-identical by construction. This script re-evaluates which
diagnostic statistics are computable from DEPLOYABLE information only and
tests whether any of them predicts anchoring failure.

Records (42 total):
  * 30 Monte-Carlo realizations from Exp B2 (expB2_mc.npz, expB2_mc.csv);
  * the 12 Exp C census series (expA_series.npz x9 + expB_results.npz x3),
    with sr_est taken from the archived Exp A/B scan results and the EMD
    re-run deterministically (identical to exp_c_mode_splitting.py).

Deployable statistics per record (no truth used):
  ne_prior        expected 405-cycle count from the SR scan prior
  count_dev_prior (n_cycles - ne_prior)/ne_prior of the selected IMF
  mtm_per_kyr     MTM peak period (kyr) of the selected IMF at sr_est
  mtm_dev405      |P - 405|/405
  matched         the framework's dual criterion (count tolerance + MTM check)
  share_sel       405-band (+/-20%) power share of the selected IMF (sr_est axis)
  share_second    second-largest IMF share (mode-splitting indicator)
  eemd_stab       EEMD-lite: 3 noise-added EMD reruns (noise = 0.2*std(sig));
                  fraction whose 405-power-dominant IMF index equals the
                  base run's dominant index (ensemble stability)

Truth (evaluation only): rmse_const / rmse_anchor vs t_true,
  fail = rmse_anchor > rmse_const, signed anchored-duration bias.

Analysis: Pearson/Spearman of each statistic vs the anchoring error;
sign-agreement of count_dev_prior with the true duration bias; classification
performance (confusion matrix, accuracy) of candidate failure-prediction
rules, incl. the framework gate itself.

Output: expC2_diag.csv  +  console conclusion block.
Run:  python3 exp_c2_deployable.py   (after exp_b2_montecarlo.py)
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from PyEMD import EMD

from exp_common import EXP_DIR, AstroTuner
from exp_c_mode_splitting import load_series

F405 = 1000.0 / 405.0
BAND = (F405 * 0.8, F405 * 1.2)
EEMD_NOISE = 0.20          # noise std = 0.2 * std(signal)
EEMD_N = 3
EEMD_SEEDS = (91001, 92001, 93001)


def imf_405_shares(tuner, imfs, depth, sr_est):
    """405-band power share of each IMF at the sr_est time axis (deployable)."""
    t = tuner.depth_to_time(depth, sr_est)
    shares = []
    for imf in imfs:
        f, p = tuner.mtm_spectrum(t, imf)
        m = (f >= BAND[0]) & (f <= BAND[1])
        shares.append(float(np.trapezoid(p[m], f[m])))
    shares = np.asarray(shares)
    return shares / shares.sum() if shares.sum() > 0 else shares * 0.0


def eemd_lite_stability(tuner, depth, sig, sr_est, base_dom_idx):
    """EEMD-lite: consistency of the 405-power-dominant IMF index across
    EEMD_N noise-added EMD reruns (deployable)."""
    sd = float(np.std(sig))
    agree = 0
    doms = []
    for sd_seed in EEMD_SEEDS:
        rng = np.random.default_rng(sd_seed)
        xn = sig + rng.normal(0.0, EEMD_NOISE * sd, sig.size)
        imfs_n = EMD().emd(xn, max_imf=8)
        sh = imf_405_shares(tuner, imfs_n, depth, sr_est)
        dom = int(np.argmax(sh))
        doms.append(dom)
        agree += int(dom == base_dom_idx)
    return agree / EEMD_N, doms


def process_record(tuner, tag, source, depth, sig, t_true, sr_est, imfs):
    depth = np.asarray(depth, float)
    t_true = np.asarray(t_true, float)
    sel = tuner.select_405_imf(imfs, depth, sr_est=sr_est)
    idx = sel['imf_index']
    age_anchor, _ = tuner.build_age_model_405(depth, imfs[idx],
                                              anchor_phase='trough')
    age_const = tuner.build_age_model(depth, sr_est)
    rmse_c = float(np.sqrt(np.mean((age_const - t_true) ** 2)))
    rmse_a = float(np.sqrt(np.mean((age_anchor - t_true) ** 2)))
    duration = float(t_true[-1])
    dur_bias = float((age_anchor[-1] - age_anchor[0] - duration) / duration)

    span = float(depth.max() - depth.min())
    ne_prior = span / sr_est / 0.405
    shares = imf_405_shares(tuner, imfs, depth, sr_est)
    order = np.argsort(shares)[::-1]
    stab, doms = eemd_lite_stability(tuner, depth, sig, sr_est,
                                     int(order[0]))
    return dict(
        record=tag, source=source, sr_est=float(sr_est),
        ne_prior=float(ne_prior), selected_imf=int(idx),
        n_cycles=int(sel['n_cycles']),
        count_dev_prior=float((sel['n_cycles'] - ne_prior) / ne_prior),
        mtm_per_kyr=float(sel['diag']['mtm_peak_period_kyr']),
        mtm_dev405=float(abs(sel['diag']['mtm_peak_period_kyr'] - 405.0)
                         / 405.0),
        matched=bool(sel['matched']),
        share_sel=float(shares[idx]),
        share_best=float(shares[order[0]]),
        share_second=float(shares[order[1]]),
        eemd_stab=float(stab),
        eemd_doms=','.join(map(str, doms)),
        # ---- truth (evaluation only) ----
        rmse_const=rmse_c, rmse_anchor=rmse_a,
        fail=bool(rmse_a > rmse_c),
        rmse_diff=float(rmse_a - rmse_c),
        duration_bias=dur_bias,
    )


def confusion(y_true, y_pred):
    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    acc = (tp + tn) / len(y_true)
    return dict(tp=tp, tn=tn, fp=fp, fn=fn, acc=acc)


def run():
    t_start = time.time()
    tuner = AstroTuner(n_jobs=1, seed=42)
    rows = []

    # ---- 30 Exp B2 Monte-Carlo realizations -------------------------------
    zb = np.load(os.path.join(EXP_DIR, 'expB2_mc.npz'))
    dfb2 = pd.read_csv(os.path.join(EXP_DIR, 'expB2_mc.csv'))
    base = {'linear': 2001, 'step': 2002, 'sine': 2003}
    rows_b2 = []
    for _, r in dfb2.iterrows():
        k = int(round((r['seed'] - base[r['scenario']]) / 1000.0))
        tag = f"{r['scenario']}_s{k:02d}"
        rows_b2.append((tag, r))
    for tag, r in rows_b2:
        rec = process_record(tuner, f'B2_{tag}', 'B2',
                             zb[f'{tag}__depth'], zb[f'{tag}__signal'],
                             zb[f'{tag}__t_true'], float(r['scan_sr']),
                             zb[f'{tag}__imfs'])
        rows.append(rec)
        print(f"  B2_{tag}  fail={int(rec['fail'])} matched={int(rec['matched'])} "
              f"cdev={rec['count_dev_prior']:+.2f} stab={rec['eemd_stab']:.2f}")

    # ---- 12 Exp C census series -------------------------------------------
    for s in load_series():
        imfs = EMD().emd(np.asarray(s['signal'], float), max_imf=10)
        rec = process_record(tuner, s['name'], 'C',
                             s['depth'], s['signal'], s['t_true'],
                             s['sr_est'], imfs)
        rows.append(rec)
        print(f"  C {s['name']:12s} fail={int(rec['fail'])} "
              f"matched={int(rec['matched'])} cdev={rec['count_dev_prior']:+.2f} "
              f"stab={rec['eemd_stab']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_DIR, 'expC2_diag.csv'), index=False)
    print(f'\nwrote expC2_diag.csv ({len(df)} records)  '
          f'[{time.time() - t_start:.0f} s]')

    # ======================================================================
    # Analysis
    # ======================================================================
    stats_cols = ['count_dev_prior', 'mtm_dev405', 'share_sel', 'share_second',
                  'eemd_stab', 'matched']
    print('\n--- correlation with anchoring error (rmse_diff = rmse_anchor - '
          'rmse_const, Myr) and |duration_bias| ---')
    for c in stats_cols:
        x = df[c].astype(float).values
        for target in ('rmse_diff', 'duration_bias'):
            y = df[target].values if target == 'rmse_diff' \
                else np.abs(df[target].values)
            rp, pp = pearsonr(x, y)
            rs, ps = spearmanr(x, y)
            lab = 'rmse_diff' if target == 'rmse_diff' else '|dur_bias|'
            print(f'  {c:16s} vs {lab:10s} Pearson r={rp:+.3f} (p={pp:.1e})  '
                  f'Spearman r={rs:+.3f} (p={ps:.1e})')

    # sign agreement: count_dev_prior sign vs true duration bias sign
    sign_dev = np.sign(df['count_dev_prior'].values)
    sign_bias = np.sign(df['duration_bias'].values)
    agree = np.mean(sign_dev == sign_bias)
    print(f'\nsign(count_dev_prior) == sign(true duration bias): '
          f'{int((sign_dev == sign_bias).sum())}/{len(df)} = {agree:.2f}')

    # ---- classification: predict anchoring failure ------------------------
    y = df['fail'].values.astype(bool)
    print(f'\n--- failure prediction (fail = rmse_anchor > rmse_const; '
          f'base rate {y.mean():.2f}) ---')
    rules = {
        'gate: matched=False -> fail': ~df['matched'].values.astype(bool),
        '|count_dev_prior| > 0.15':
            df['count_dev_prior'].abs().values > 0.15,
        '|count_dev_prior| > 0.30':
            df['count_dev_prior'].abs().values > 0.30,
        'mtm_dev405 > 0.20': df['mtm_dev405'].values > 0.20,
        'eemd_stab < 1.0': df['eemd_stab'].values < 1.0 - 1e-9,
        'share_sel < 0.5': df['share_sel'].values < 0.5,
        'count|>0.15 OR mtm>0.20':
            (df['count_dev_prior'].abs().values > 0.15)
            | (df['mtm_dev405'].values > 0.20),
    }
    for name, pred in rules.items():
        cm = confusion(y, np.asarray(pred, bool))
        print(f"  {name:28s} acc={cm['acc']:.2f}  "
              f"TP={cm['tp']} TN={cm['tn']} FP={cm['fp']} FN={cm['fn']}")
        # ======================================================================
    # Full threshold sweep for |count_dev_prior|
    # Paper: threshold t = 0.010–0.300
    # ======================================================================
    print('\n--- full threshold sweep: |count_dev_prior| -> failure ---')

    thresholds = np.arange(0.010, 0.300 + 1e-12, 0.005)

    sweep_rows = []

    for t in thresholds:
        pred = df['count_dev_prior'].abs().values > t
        cm = confusion(y, np.asarray(pred, bool))

        sweep_rows.append({
            'threshold': float(t),
            'accuracy': float(cm['acc']),
            'TP': int(cm['tp']),
            'TN': int(cm['tn']),
            'FP': int(cm['fp']),
            'FN': int(cm['fn']),
        })

    sweep_df = pd.DataFrame(sweep_rows)

    # 保存全部阈值扫描结果
    sweep_path = os.path.join(EXP_DIR, 'expC2_threshold_sweep.csv')
    sweep_df.to_csv(sweep_path, index=False)

    # 找最高准确率
    best_acc = sweep_df['accuracy'].max()
    best_rows = sweep_df[np.isclose(sweep_df['accuracy'], best_acc)]

    base_rate = float(y.mean())
    majority_acc = max(base_rate, 1.0 - base_rate)

    print(f'base failure rate          = {base_rate:.3f}')
    print(f'always-majority accuracy   = {majority_acc:.3f}')
    print(f'best sweep accuracy        = {best_acc:.3f}')

    print('best threshold(s):')
    for _, r in best_rows.iterrows():
        print(
            f"  t={r['threshold']:.3f}: "
            f"acc={r['accuracy']:.3f}, "
            f"TP={int(r['TP'])}, TN={int(r['TN'])}, "
            f"FP={int(r['FP'])}, FN={int(r['FN'])}"
        )

    print(f'wrote {sweep_path}')
    return df


if __name__ == '__main__':
    run()
