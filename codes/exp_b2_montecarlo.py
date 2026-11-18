# -*- coding: utf-8 -*-
"""
exp_b2_montecarlo.py — Round-1 supplement S1: Exp B Monte-Carlo extension
==========================================================================
Reviewer Round-1 criticism: Exp B had a single realization per scenario
(n=1), so the "anchored beats constant-SR" conclusion and the gate
(matched True/False) behaviour could be seed luck. This script repeats the
full Exp B chain for the three SR(d) scenarios x N_SEEDS noise realizations:

    synth_series (same profiles/noise as Exp B, seed = base + 1000*k)
    -> scan_sed_rates -> best_sr (match score)
    -> EMD (max_imf=10) -> select_405_imf (matched gate)
    -> constant-SR age model  vs  405-kyr anchored age model
    -> RMSE of both against Age_true(d) = integral dd'/SR_true

Outputs
-------
expB2_mc.csv          one row per realization (30 rows): scenario, seed,
                      scan_sr, match, matched, selected_imf, n_cycles,
                      Ne_prior, rmse_const, rmse_anchor (+ derived metrics)
expB2_mc_summary.csv  per-scenario mean +/- std, anchor-win fraction
expB2_mc.npz          archive (depth/signal/t_true/sr_true/imfs/selected
                      imf per realization) reused by exp_c2_deployable.py
Console               gate confusion matrix: matched x (anchor beats const)

Run:  python3 exp_b2_montecarlo.py
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
from PyEMD import EMD

from exp_common import EXP_DIR, AstroTuner, synth_series
from exp_b_variable_sr import PROFILES, LENGTH_M, NOISE, SCAN_GRID

N_SEEDS = 10          # realizations per scenario (EMD ~0.1 s -> 30 cheap)
SEED_STRIDE = 1000    # seed = profile base seed + k*SEED_STRIDE, k = 0..N_SEEDS-1
                      # k=0 reproduces the original Exp B single realization


def one_realization(tuner, name, cfg, seed):
    d = synth_series(cfg['sr_fn'], length_m=LENGTH_M,
                     noise_ratio=NOISE[1], seed=seed)
    depth, sig, t_true = d['depth'], d['signal'], d['t_true']

    # (i) constant-SR linear age model (timeOpt scan optimum)
    scan = tuner.scan_sed_rates(depth, sig)
    sr_opt, ms = tuner.best_sr(scan, depth=depth, signal=sig)
    age_const = tuner.build_age_model(depth, sr_opt)

    # (ii) 405-kyr anchored piecewise age model
    imfs = EMD().emd(sig, max_imf=10)
    sel = tuner.select_405_imf(imfs, depth, sr_est=sr_opt)
    idx = sel['imf_index']
    age_anchor, anchor_df = tuner.build_age_model_405(
        depth, imfs[idx], anchor_phase='trough')

    ne_prior = (depth.max() - depth.min()) / sr_opt / 0.405
    err_c = age_const - t_true
    err_a = age_anchor - t_true
    rmse_c = float(np.sqrt(np.mean(err_c ** 2)))
    rmse_a = float(np.sqrt(np.mean(err_a ** 2)))
    duration = float(t_true[-1])
    row = dict(
        scenario=name, seed=int(seed),
        scan_sr=float(sr_opt), match=float(ms),
        matched=bool(sel['matched']), selected_imf=int(idx),
        n_cycles=int(sel['n_cycles']), ne_prior=float(ne_prior),
        ne_true=float(duration / 0.405),
        n_anchors=int(len(anchor_df)),
        imf_mtm_period_kyr=float(sel['diag']['mtm_peak_period_kyr']),
        count_dev_prior=float((sel['n_cycles'] - ne_prior) / ne_prior),
        rmse_const=rmse_c, rmse_anchor=rmse_a,
        rmse_ratio=float(rmse_a / rmse_c),
        anchor_wins=bool(rmse_a < rmse_c),
        bias_const=float(age_const[-1] - t_true[-1]),
        bias_anchor=float(age_anchor[-1] - t_true[-1]),
        duration_true_myr=duration,
        duration_anchor_myr=float(age_anchor[-1] - age_anchor[0]),
    )
    art = dict(depth=depth, signal=sig, t_true=t_true, sr_true=d['sr_true'],
               imfs=imfs, sel_idx=np.array([idx]))
    return row, art


def run():
    t_start = time.time()
    tuner = AstroTuner(sr_grid=SCAN_GRID, n_jobs=2, seed=42)
    rows, npz = [], {}
    for name, cfg in PROFILES.items():
        base = cfg['seed']
        for k in range(N_SEEDS):
            seed = base + k * SEED_STRIDE
            row, art = one_realization(tuner, name, cfg, seed)
            rows.append(row)
            tag = f'{name}_s{k:02d}'
            for kk, vv in art.items():
                npz[f'{tag}__{kk}'] = vv
            print(
    f"  {name:6s} "
    f"seed={seed} "
    f"sr={row['scan_sr']:5.2f} "
    f"match={row['match']:.3f} "
    f"matched={int(row['matched'])} "
    f"IMF{row['selected_imf'] + 1} "
    f"ncyc={row['n_cycles']:3d} "
    f"Ne_p={row['ne_prior']:.1f} "
    f"RMSE c={row['rmse_const']:.2f} "
    f"a={row['rmse_anchor']:.2f} "
    f"({'WIN' if row['anchor_wins'] else 'lose'})"
)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_DIR, 'expB2_mc.csv'), index=False)
    np.savez(os.path.join(EXP_DIR, 'expB2_mc.npz'), **npz)

    # ---- per-scenario summary --------------------------------------------
    summ = df.groupby('scenario').agg(
        n=('seed', 'size'),
        rmse_const_mean=('rmse_const', 'mean'),
        rmse_const_std=('rmse_const', 'std'),
        rmse_anchor_mean=('rmse_anchor', 'mean'),
        rmse_anchor_std=('rmse_anchor', 'std'),
        anchor_win_frac=('anchor_wins', 'mean'),
        matched_frac=('matched', 'mean'),
        match_score_mean=('match', 'mean'),
        scan_sr_mean=('scan_sr', 'mean'),
        scan_sr_std=('scan_sr', 'std'),
    ).reset_index()
    summ.to_csv(os.path.join(EXP_DIR, 'expB2_mc_summary.csv'), index=False)

    # ---- gate confusion matrix: matched x anchor_wins ---------------------
    cm = pd.crosstab(df['matched'], df['anchor_wins'],
                     rownames=['matched'], colnames=['anchor_wins'],
                     dropna=False)
    for b in (False, True):
        for a in (False, True):
            if (b, a) not in [(i, j) for i in cm.index for j in cm.columns]:
                cm.loc[b, a] = 0
    cm = cm.sort_index().sort_index(axis=1)
    print('\nGate confusion matrix (matched x anchor_wins):')
    print(cm.to_string())
    n = len(df)
    gate_ok = int(((df['matched'] == df['anchor_wins'])).sum())
    print(f'gate agreement (matched <=> anchor wins): {gate_ok}/{n} '
          f'= {gate_ok / n:.2f}')
    print(f'\noverall anchor win fraction: {df["anchor_wins"].mean():.2f}  '
          f'[{time.time() - t_start:.0f} s]')
    print(summ.to_string(index=False))
    return df, summ, cm


if __name__ == '__main__':
    run()
