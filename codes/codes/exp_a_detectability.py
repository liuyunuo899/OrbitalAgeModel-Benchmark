# -*- coding: utf-8 -*-
"""
exp_a_detectability.py — Experiment A: detectability window (tests H1)
======================================================================
Grid: true SR in {2,3,5,7.5,10,15,20,30,40,50} m/Myr x noise levels
{low 0.2, mid 0.5, high 0.8} (noise/signal RMS ratio). Segment 600 m,
0.1 m sampling (~6000 points).

For each combination:
  * MTM at the TRUE SR; per-band peak/neighbourhood-median significance
    (threshold 5) for 405/95/40/21 kyr -> detection 0/1;
  * AstroTuner.scan_sed_rates -> SR recovery error + match score.

Results compared against the theoretical Nyquist prediction:
  period P (kyr) detectable  <=>  lambda = P*SR/1000 >= 2*dz = 0.2 m.

Outputs
-------
expA_detection.csv    one row per combo x band (detected, significance, ...)
expA_sr_recovery.csv  one row per combo (sr_est, error, match_score)
expA_series.npz       9 representative series reused by Experiment C
figures/fig12.png     (a) detection phase diagram + Nyquist line
                      (b) noise robustness
                      (c) SR recovery vs truth

Run:  python3 exp_a_detectability.py
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from exp_common import (EXP_DIR, PERIODS_KYR, DZ, AstroTuner,
                        synth_series, band_significances, const_sr_fn,
                        nyquist_sr, ptitle, savefig,
                        C_INK, C_BLUE, C_RED, C_GREEN, C_TAN, C_GRAY, PALETTE)

SR_LIST = [2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50]
NOISES = [('low', 0.2), ('mid', 0.5), ('high', 0.8)]
LENGTH_M = 600.0
SCAN_GRID = np.arange(1.0, 55.0001, 0.5)      # m/Myr, covers SR up to 50

# representative series kept for Experiment C (EMD census):
#   SR=5 and 15 at all three noise levels + SR=2/30/50 at mid noise
KEEP = {(5.0, 'low'), (5.0, 'mid'), (5.0, 'high'),
        (15.0, 'low'), (15.0, 'mid'), (15.0, 'high'),
        (2.0, 'mid'), (30.0, 'mid'), (50.0, 'mid')}


def run():
    t_start = time.time()
    tuner = AstroTuner(sr_grid=SCAN_GRID, n_jobs=2, seed=42)
    det_rows, rec_rows = [], []
    kept = {}
    for i, sr in enumerate(SR_LIST):
        for j, (nlab, nrat) in enumerate(NOISES):
            seed = 1000 + i * 10 + j
            d = synth_series(const_sr_fn(sr), length_m=LENGTH_M,
                             noise_ratio=nrat, seed=seed)
            # --- detection at the TRUE SR ---
            sigs, det, _, _ = band_significances(tuner, d['t_true'], d['signal'])
            for p, sg, dt_ in zip(PERIODS_KYR, sigs, det):
                det_rows.append(dict(sr_true=sr, noise=nlab, noise_ratio=nrat,
                                     period_kyr=p, wavelength_m=p * sr / 1000.0,
                                     significance=float(sg), detected=int(dt_),
                                     nyquist_detectable=int(p * sr / 1000.0 >= 2 * DZ)))
            # --- SR recovery via timeOpt-style scan ---
            scan = tuner.scan_sed_rates(d['depth'], d['signal'])
            sr_opt, ms = tuner.best_sr(scan, depth=d['depth'], signal=d['signal'])
            rec_rows.append(dict(sr_true=sr, noise=nlab, noise_ratio=nrat,
                                 sr_est=float(sr_opt),
                                 rel_err=float((sr_opt - sr) / sr),
                                 match_score=float(ms), seed=seed))
            if (sr, nlab) in KEEP:
                kept[f'A_sr{sr:g}_{nlab}'] = d
            print(f'  SR={sr:4g} noise={nlab:4s} det={det} '
                  f'sr_est={sr_opt:6.2f} match={ms:.3f}')
    det_df = pd.DataFrame(det_rows)
    rec_df = pd.DataFrame(rec_rows)
    det_df.to_csv(os.path.join(EXP_DIR, 'expA_detection.csv'), index=False)
    rec_df.to_csv(os.path.join(EXP_DIR, 'expA_sr_recovery.csv'), index=False)

    # save representative series for Experiment C
    npz = {}
    for name, d in kept.items():
        npz[f'{name}__depth'] = d['depth']
        npz[f'{name}__signal'] = d['signal']
        npz[f'{name}__t_true'] = d['t_true']
        npz[f'{name}__sr_true'] = d['sr_true']
    np.savez(os.path.join(EXP_DIR, 'expA_series.npz'), **npz)
    print(f'  wrote expA_detection.csv ({len(det_df)} rows), '
          f'expA_sr_recovery.csv ({len(rec_df)} rows), '
          f'expA_series.npz ({len(kept)} series)  [{time.time()-t_start:.0f} s]')
    return det_df, rec_df


# ----------------------------------------------------------------------
def make_fig12(det_df, rec_df):
    fig = plt.figure(figsize=(11.5, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.95, 1.0], wspace=0.32)

    # ---- (a) detection phase diagram in (SR, wavelength) space ----------
    axa = fig.add_subplot(gs[0, 0])
    mid = det_df[det_df['noise'] == 'mid']
    pcolors = {405: C_BLUE, 95: C_GREEN, 40: C_TAN, 21: C_RED}
    for p in PERIODS_KYR:
        sub = mid[mid['period_kyr'] == p].sort_values('sr_true')
        c = pcolors[int(p)]
        lam = sub['period_kyr'].values * sub['sr_true'].values / 1000.0
        axa.plot(sub['sr_true'], lam, '-', color=c, lw=0.8, alpha=0.55, zorder=1)
        ok = sub['detected'].values == 1
        axa.plot(sub['sr_true'][ok], lam[ok], 'o', color=c, ms=6, zorder=3,
                 label=f'{int(p)} kyr')
        axa.plot(sub['sr_true'][~ok], lam[~ok], 'o', mfc='white', mec=c,
                 ms=6, mew=1.2, zorder=3)
    axa.axhline(2 * DZ, color=C_INK, ls='--', lw=1.0)
    axa.text(9.0, 0.075,
    r'Nyquist limit  $\lambda = 2\Delta z$ = 0.2 m',
    fontsize=7.5,
    color=C_INK,
    ha='left',
    va='center')
    axa.axhspan(0.03, 2 * DZ, color=C_GRAY, alpha=0.12, zorder=0)
    axa.set_xscale('log'); axa.set_yscale('log')
    axa.set_xlim(1.8, 60); axa.set_ylim(0.03, 25)
    axa.set_xticks([2, 5, 10, 20, 50]); axa.set_xticklabels([2, 5, 10, 20, 50])
    axa.set_xlabel('True sedimentation rate SR (m/Myr)')
    axa.set_ylabel(r'Wavelength $\lambda$ = P·SR/1000 (m)')
    axa.legend(loc='upper left', ncol=2, handletextpad=0.2,
               columnspacing=0.8, borderpad=0.2)
    # filled/open legend proxy
    axa.plot([], [], 'o', color=C_GRAY, ms=6, label='detected')
    axa.plot([], [], 'o', mfc='white', mec=C_GRAY, ms=6, mew=1.2,
             label='not detected')
    axa.legend(loc='upper left', ncol=2, handletextpad=0.2,
               columnspacing=0.8, borderpad=0.2, fontsize=7)
    ptitle(axa, '(a)', 'Detection phase diagram (mid noise)')

    # ---- (b) noise robustness: detection fraction vs period -------------
    axb = fig.add_subplot(gs[0, 1])
    w = 0.26
    for k, (nlab, _) in enumerate(NOISES):
        fr = [det_df[(det_df['noise'] == nlab) &
                     (det_df['period_kyr'] == p)]['detected'].mean()
              for p in PERIODS_KYR]
        axb.bar(np.arange(4) + (k - 1) * w, fr, width=w,
                color=PALETTE[k], alpha=0.9, label=nlab)
    axb.set_xticks(range(4))
    axb.set_xticklabels([f'{int(p)}' for p in PERIODS_KYR])
    axb.set_xlabel('Target period (kyr)')
    axb.set_ylabel('Detection fraction (10 SR levels)')
    axb.set_ylim(0, 1.12)
    axb.legend(loc='upper right', title='noise/signal', title_fontsize=7)
    ptitle(axb, '(b)', 'Noise robustness')

    # ---- (c) SR recovery -------------------------------------------------
    axc = fig.add_subplot(gs[0, 2])
    ncolors = {'low': C_BLUE, 'mid': C_GREEN, 'high': C_RED}
    for nlab, _ in NOISES:
        sub = rec_df[rec_df['noise'] == nlab]
        axc.plot(sub['sr_true'], sub['sr_est'], 'o', color=ncolors[nlab],
                 ms=5, alpha=0.85, label=nlab)
    lim = [1, 60]
    axc.plot(lim, lim, '-', color=C_INK, lw=1.0, label='1:1')
    axc.fill_between(lim, [v / 1.1 for v in lim], [v * 1.1 for v in lim],
                     color=C_GRAY, alpha=0.15, lw=0)
    axc.text(22, 12, r'$\pm$10%', fontsize=7.5, color=C_GRAY, rotation=24)
    axc.set_xscale('log'); axc.set_yscale('log')
    axc.set_xticks([2, 5, 10, 20, 50]); axc.set_xticklabels([2, 5, 10, 20, 50])
    axc.set_yticks([2, 5, 10, 20, 50]); axc.set_yticklabels([2, 5, 10, 20, 50])
    axc.set_xlabel('True SR (m/Myr)')
    axc.set_ylabel('Recovered SR (m/Myr)')
    axc.legend(loc='upper left', title='noise/signal', title_fontsize=7)
    ptitle(axc, '(c)', 'SR recovery (timeOpt-inspired scan)')

    savefig(fig, 'fig12.png')


if __name__ == '__main__':
    det_df, rec_df = run()
    make_fig12(det_df, rec_df)
