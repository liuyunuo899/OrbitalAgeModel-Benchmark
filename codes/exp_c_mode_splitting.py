# -*- coding: utf-8 -*-
"""
exp_c_mode_splitting.py — Experiment C: EMD mode-splitting census (tests H2, part 2)
====================================================================================
12 representative series from Experiments A (9: SR=5/15 at low/mid/high
noise + SR=2/30/50 at mid noise) and B (3: linear/step/sine SR profiles),
each ~6000 points, PyEMD (max_imf=10) — 12 EMD runs total.

For each series:
  * distribution of the 405-kyr signal power among IMFs
    (power in the 405 kyr +/-20% band, normalised across IMFs);
  * splitting flag: the two IMF with the largest 405-band shares both
    carry > 20% of the total 405 power;
  * peak-to-peak cycle count of the 405-bearing IMF vs the theoretical
    cycle number Ne_true = segment duration / 405 kyr;
  * total-duration bias of the 405-kyr anchored age model
    (select_405_imf + build_age_model_405; sr_est taken from the Exp A/B
    timeOpt scan results) vs the true duration;
  * correlation between the cycle-count miss rate and the duration bias
    ("is the bias diagnosable from the count deficit?").

Outputs
-------
expC_imf_405_power.csv  per series x IMF 405-band power share
expC_summary.csv        per series census (split, counts, biases, r, p)
figures/fig14.png       (a) stacked 405-power distribution across IMFs
                        (b) split rate vs noise level / vs SR
                        (c) count deficit vs duration bias scatter

Run:  python3 exp_c_mode_splitting.py   (after exp_a and exp_b)
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr, spearmanr
from PyEMD import EMD

from exp_common import (EXP_DIR, AstroTuner, ptitle, savefig,
                        C_INK, C_BLUE, C_RED, C_GREEN, C_TAN, C_GRAY, C_LGRAY)

F405 = 1000.0 / 405.0
BAND = (F405 * 0.8, F405 * 1.2)          # 405 kyr +/- 20%
SPLIT_THRESH = 0.20


def load_series():
    """Return list of dicts: name, source, sr_tag, sr_mean, noise, depth,
    signal, t_true, sr_est."""
    out = []
    za = np.load(os.path.join(EXP_DIR, 'expA_series.npz'))
    rec = pd.read_csv(os.path.join(EXP_DIR, 'expA_sr_recovery.csv'))
    names = sorted({k.split('__')[0] for k in za.files})
    for nm in names:
        # name pattern: A_sr{sr}_{noise}
        parts = nm.split('_')
        sr = float(parts[1][2:])
        noise = parts[2]
        row = rec[(rec['sr_true'] == sr) & (rec['noise'] == noise)].iloc[0]
        out.append(dict(name=nm, source='A', sr_tag=f'{sr:g}',
                        sr_mean=sr, noise=noise,
                        depth=za[f'{nm}__depth'], signal=za[f'{nm}__signal'],
                        t_true=za[f'{nm}__t_true'],
                        sr_est=float(row['sr_est'])))
    zb = np.load(os.path.join(EXP_DIR, 'expB_results.npz'))
    dfb = pd.read_csv(os.path.join(EXP_DIR, 'expB_age_error.csv'))
    for scn, lab in [('linear', 'Linear'), ('step', 'Step'), ('sine', 'Sine')]:
        r = dfb[dfb['scenario'] == scn].iloc[0]
        sr_true = zb[f'{scn}__sr_true']
        out.append(dict(name=f'B_{scn}', source='B', sr_tag=lab,
                        sr_mean=float(600.0 / zb[f'{scn}__t_true'][-1]),
                        noise='mid',
                        depth=zb[f'{scn}__depth'], signal=zb[f'{scn}__signal'],
                        t_true=zb[f'{scn}__t_true'],
                        sr_est=float(r['sr_scan_opt'])))
    return out


def run():
    t_start = time.time()
    tuner = AstroTuner(n_jobs=1, seed=42)
    series = load_series()
    pow_rows, sum_rows = [], []
    for s in series:
        depth, sig, t_true = s['depth'], s['signal'], s['t_true']
        duration = float(t_true[-1])
        ne_true = duration / 0.405
        emd = EMD()
        imfs = emd.emd(sig, max_imf=10)
        # --- 405-band power share of each IMF (true time axis) ----------
        shares, ncyc = [], []
        for i, imf in enumerate(imfs):
            f, p = tuner.mtm_spectrum(t_true, imf)
            m = (f >= BAND[0]) & (f <= BAND[1])
            bp = float(np.trapezoid(p[m], f[m]))
            shares.append(bp)
            n, _, _ = tuner.count_cycles(imf)
            ncyc.append(n)
        shares = np.asarray(shares)
        frac = shares / shares.sum() if shares.sum() > 0 else shares * 0.0
        order = np.argsort(frac)[::-1]
        i_best = int(order[0])
        split = bool(frac[order[0]] > SPLIT_THRESH
                     and frac[order[1]] > SPLIT_THRESH)
        # --- anchored model total-duration bias --------------------------
        sel = tuner.select_405_imf(imfs, depth, sr_est=s['sr_est'])
        i_sel = sel['imf_index']
        age_anchor, anchor_df = tuner.build_age_model_405(
            depth, imfs[i_sel], anchor_phase='trough')
        dur_model = float(age_anchor[-1] - age_anchor[0])
        dur_bias = (dur_model - duration) / duration
        n_best = int(ncyc[i_best])
        # miss rate of the 405-power-dominant IMF (diagnostic)
        miss_rate_best = (ne_true - n_best) / ne_true
        # miss rate of the IMF actually anchoring the age model (primary,
        # paired with the duration bias below); <0 = over-counting
        miss_rate = (ne_true - int(sel['n_cycles'])) / ne_true
        for i, fr in enumerate(frac):
            pow_rows.append(dict(series=s['name'], source=s['source'],
                                 sr_tag=s['sr_tag'], noise=s['noise'],
                                 imf_index=i, share_405=float(fr),
                                 n_cycles=int(ncyc[i])))
        sum_rows.append(dict(
            series=s['name'], source=s['source'], sr_tag=s['sr_tag'],
            sr_mean=s['sr_mean'], noise=s['noise'], n_imfs=len(imfs),
            ne_true=float(ne_true), imf_best=i_best,
            share_best=float(frac[order[0]]),
            share_second=float(frac[order[1]]), split=split,
            n_cycles_best=n_best, miss_rate=float(miss_rate),
            miss_rate_best=float(miss_rate_best),
            sel_imf=int(i_sel), sel_matched=bool(sel['matched']),
            sel_n_cycles=int(sel['n_cycles']),
            duration_true_myr=duration, duration_model_myr=dur_model,
            duration_bias=float(dur_bias)))
        print(
    f"  {s['name']:14s} "
    f"split={int(split)} "
    f"powerIMF=IMF{i_best + 1} "
    f"share={frac[order[0]]:.2f}/{frac[order[1]]:.2f} "
    f"ncyc_power={n_best} "
    f"anchorIMF=IMF{i_sel + 1} "
    f"ncyc_anchor={int(sel['n_cycles'])} "
    f"Ne={ne_true:.0f} "
    f"miss_anchor={miss_rate:+.2f} "
    f"durBias={dur_bias:+.2f}"
)

    pow_df = pd.DataFrame(pow_rows)
    sum_df = pd.DataFrame(sum_rows)
    # correlation: miss rate vs duration bias
    r_p, p_p = pearsonr(sum_df['miss_rate'], sum_df['duration_bias'])
    r_s, p_s = spearmanr(sum_df['miss_rate'], sum_df['duration_bias'])
    sum_df.attrs.update(r_pearson=r_p, p_pearson=p_p,
                        r_spearman=r_s, p_spearman=p_s)
    pow_df.to_csv(os.path.join(EXP_DIR, 'expC_imf_405_power.csv'), index=False)
    sum_df.to_csv(os.path.join(EXP_DIR, 'expC_summary.csv'), index=False)
    print(f'  correlation miss_rate vs duration_bias: '
          f'Pearson r={r_p:.3f} (p={p_p:.2e}), '
          f'Spearman r={r_s:.3f} (p={p_s:.2e})')
    print(f'  split rate overall: {sum_df["split"].mean():.2f}  '
          f'[{time.time()-t_start:.0f} s]')
    return pow_df, sum_df, dict(r_pearson=r_p, p_pearson=p_p,
                                r_spearman=r_s, p_spearman=p_s)


# ----------------------------------------------------------------------
def make_fig14(pow_df, sum_df, stats=None):
    # always recompute the correlation from the archived summary table so the
    # in-figure annotation can never drift from the data (Round-2 fix)
    r_p, p_p = pearsonr(sum_df['miss_rate'], sum_df['duration_bias'])
    r_s, p_s = spearmanr(sum_df['miss_rate'], sum_df['duration_bias'])
    stats = dict(r_pearson=r_p, p_pearson=p_p, r_spearman=r_s, p_spearman=p_s)
    fig = plt.figure(figsize=(11.5, 3.8))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.37)

    # ---- (a) stacked 405-power distribution across IMFs ------------------
    axa = fig.add_subplot(gs[0, 0])
    names = sum_df['series'].tolist()
    n_imf_max = int(pow_df['imf_index'].max()) + 1
    # discrete low-saturation palette for IMF indices
    imf_colors = ['#41627E', '#A65A4A', '#6E8B67', '#B08C57', '#8A8F94',
                  '#7E8B9B', '#C9CDD1', '#5F4B2E', '#97794A', '#D9C795',
                  '#2F3B46']
    bottoms = np.zeros(len(names))
    for im in range(n_imf_max):
        vals = [pow_df[(pow_df['series'] == nm) &
                       (pow_df['imf_index'] == im)]['share_405']
                for nm in names]
        vals = np.array([v.iloc[0] if len(v) else 0.0 for v in vals])
        if vals.max() < 1e-3:
            continue
        axa.bar(range(len(names)), vals, bottom=bottoms,
                color=imf_colors[im % len(imf_colors)], width=0.72,
                label=f'IMF{im + 1}')
        bottoms += vals
    # mark split series
    for xi, sp in enumerate(sum_df['split']):
        if sp:
            axa.text(xi, 1.02, 'S', ha='center', va='bottom', fontsize=7,
                     color=C_RED, fontweight='bold')
    axa.set_xticks(range(len(names)))
    axa.set_xticklabels([nm.replace('A_sr', '').replace('_', '\n')
                         for nm in names], fontsize=6.5)
    axa.set_ylim(0, 1.12)
    axa.set_ylabel('Share of 405-kyr band power')
    axa.set_xlabel('Series (SR m/Myr / noise, or SR profile)')
    leg = axa.legend(loc='center left', bbox_to_anchor=(1.005, 0.5),
                     fontsize=6, title='IMF', title_fontsize=6.5,
                     handlelength=1.0, handletextpad=0.4, borderaxespad=0.0,
                     labelspacing=0.3)
    ptitle(axa, '(a)', '405-kyr power among IMFs (S = split)')

    # ---- (b) split rate vs noise level / vs SR ---------------------------
    axb1 = fig.add_subplot(gs[0, 1])
    noises = ['low', 'mid', 'high']
    rates = [sum_df[sum_df['noise'] == nl]['split'].mean() for nl in noises]
    cnts = [len(sum_df[sum_df['noise'] == nl]) for nl in noises]
    axb1.bar(range(3), rates, width=0.55,
             color=[C_BLUE, C_GREEN, C_RED], alpha=0.9)
    for xi, (rt, ct) in enumerate(zip(rates, cnts)):
        axb1.text(xi, rt, f'{rt:.2f}\n(n={ct})', ha='center', va='bottom',
                  fontsize=7, color=C_INK)
    axb1.set_xticks(range(3)); axb1.set_xticklabels(noises)
    axb1.set_xlabel('Noise level')
    axb1.set_ylabel('Mode-splitting rate')
    axb1.set_ylim(0, 1.25)
    ptitle(axb1, '(b)', 'Split rate vs noise (inset: vs SR)')
    # inset: split indicator vs mean SR
    axin = axb1.inset_axes([0.06, 0.50, 0.50, 0.44])
    axin.axhline(0.5, color=C_LGRAY, lw=0.8, ls=':')
    jitter = np.linspace(-0.06, 0.06, len(sum_df))
    axin.plot(sum_df['sr_mean'], sum_df['split'].astype(float) + jitter,
              'o', ms=3.5, color=C_BLUE, alpha=0.75)
    axin.set_xscale('log')
    axin.set_xticks([2, 5, 10, 20, 50])
    axin.set_xticklabels([2, 5, 10, 20, 50], fontsize=6)
    # Draw yes/no labels INSIDE the inset rather than outside its left spine.
    axin.set_yticks([0, 1])
    axin.set_yticklabels([])
    axin.tick_params(axis='y', length=0)
    axin.text(0.10, 0.89, 'yes', transform=axin.transAxes,
              ha='left', va='center', fontsize=6, color=C_INK)
    axin.text(0.035, 0.10, 'no', transform=axin.transAxes,
              ha='left', va='center', fontsize=6, color=C_INK)
    axin.set_xlabel('mean SR (m/Myr)', fontsize=6.5, labelpad=1)
    axin.tick_params(length=2)

    # ---- (c) count deficit vs duration bias ------------------------------
    axc = fig.add_subplot(gs[0, 2])
    src_color = {'A': C_BLUE, 'B': C_RED}
    for src in ['A', 'B']:
        sub = sum_df[sum_df['source'] == src]
        axc.plot(sub['miss_rate'] * 100, sub['duration_bias'] * 100, 'o',
                 color=src_color[src], ms=6, alpha=0.85,
                 label=f'Exp {src} series')
    x = sum_df['miss_rate'].values * 100
    y = sum_df['duration_bias'].values * 100
    cf = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    axc.plot(xs, np.polyval(cf, xs), '--', color=C_GRAY, lw=1.0)
    axc.axvline(0, color=C_LGRAY, lw=0.8, ls=':')
    axc.axhline(0, color=C_LGRAY, lw=0.8, ls=':')
    axc.set_xlabel('405-cycle count deficit (Ne_true − n)/Ne_true (%)')
    axc.set_ylabel('Anchored-model duration bias (%)')
    axc.legend(loc='lower left', fontsize=7)
    # Left-align the correlation summary in the open upper region;
    # four compact lines prevent text from extending beyond the right border.
    axc.text(0.35, 0.96,
             f"Pearson r = {stats['r_pearson']:.3f}\n"
             f"(p = {stats['p_pearson']:.1e})\n"
             f"Spearman r = {stats['r_spearman']:.3f}\n"
             f"(p = {stats['p_spearman']:.1e})",
             transform=axc.transAxes, ha='left', va='top',
             fontsize=6.8, linespacing=1.12, color=C_INK)
    ptitle(axc, '(c)', 'Count deficit diagnoses duration bias')

    savefig(fig, 'fig14.png')


if __name__ == '__main__':
    pow_df, sum_df, stats = run()
    make_fig14(pow_df, sum_df, stats)
