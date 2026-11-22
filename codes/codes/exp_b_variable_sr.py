# -*- coding: utf-8 -*-
"""
exp_b_variable_sr.py — Experiment B: variable sedimentation rate (tests H2, part 1)
====================================================================================
Three true SR(d) profiles on a 600 m segment (mid noise, distinct seeds):
    1. linear drift : SR = 10 -> 20 m/Myr
    2. central step : SR = 10 -> 20 m/Myr at 300 m
    3. sinusoidal   : SR = 15 +/- 5 m/Myr

Age models compared against the truth Age_true(d) = integral dd'/SR_true:
    (i)  constant-SR linear model (timeOpt scan optimum);
    (ii) 405-kyr anchored piecewise model
         (EMD -> select_405_imf -> build_age_model_405).

Outputs
-------
expB_age_error.csv   per-scenario metrics (RMSE, max deviation, diagnostics)
expB_results.npz     depth/signal/true age/recovered ages (reused by Exp C)
figures/fig13.png    (a) SR profiles true vs recovered (3 scenarios)
                     (b) age-error curves (constant vs anchored;
                         representative realization, seed k=0)
                     (c) Monte-Carlo error statistics from expB2_mc.csv
                         (10 realizations per scenario; box + scatter,
                         medians, paired RMSE differences)

Run:  python3 exp_b_variable_sr.py
"""

from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PyEMD import EMD

from exp_common import (EXP_DIR, PERIODS_KYR, AstroTuner, synth_series,
                        ptitle, savefig,
                        C_INK, C_BLUE, C_RED, C_GREEN, C_TAN, C_GRAY)

LENGTH_M = 600.0
NOISE = ('mid', 0.5)
SCAN_GRID = np.arange(5.0, 30.0001, 0.25)

PROFILES = {
    'linear': dict(
        label='Linear drift 10→20',
        sr_fn=lambda d: 10.0 + 10.0 * d / LENGTH_M,
        seed=2001),
    'step': dict(
        label='Central step 10→20',
        sr_fn=lambda d: np.where(d < LENGTH_M / 2.0, 10.0, 20.0),
        seed=2002),
    'sine': dict(
        label='Sinusoidal 15±5',
        sr_fn=lambda d: 15.0 + 5.0 * np.sin(2.0 * np.pi * d / LENGTH_M),
        seed=2003),
}


def run():
    t_start = time.time()
    tuner = AstroTuner(sr_grid=SCAN_GRID, n_jobs=2, seed=42)
    rows, store = [], {}
    for name, cfg in PROFILES.items():
        d = synth_series(cfg['sr_fn'], length_m=LENGTH_M,
                         noise_ratio=NOISE[1], seed=cfg['seed'])
        depth, sig, t_true = d['depth'], d['signal'], d['t_true']

        # (i) constant-SR linear age model (timeOpt scan optimum)
        scan = tuner.scan_sed_rates(depth, sig)
        sr_opt, ms = tuner.best_sr(scan, depth=depth, signal=sig)
        age_const = tuner.build_age_model(depth, sr_opt)

        # (ii) 405-kyr anchored piecewise age model
        emd = EMD()
        imfs = emd.emd(sig, max_imf=10)
        sel = tuner.select_405_imf(imfs, depth, sr_est=sr_opt)
        idx = sel['imf_index']
        matched = bool(sel['matched'])
        age_anchor, anchor_df = tuner.build_age_model_405(
            depth, imfs[idx], anchor_phase='trough')
        n_anchors = len(anchor_df)

        err_c = age_const - t_true
        err_a = age_anchor - t_true
        rows.append(dict(
            scenario=name, label=cfg['label'], sr_scan_opt=float(sr_opt),
            match_score=float(ms), imf_index=int(idx), matched=matched,
            n_cycles=int(sel['n_cycles']), ne_true=float(t_true[-1] / 0.405),
            n_anchors=int(n_anchors),
            rmse_const=float(np.sqrt(np.mean(err_c ** 2))),
            maxdev_const=float(np.max(np.abs(err_c))),
            bias_const=float(age_const[-1] - t_true[-1]),
            rmse_anchor=float(np.sqrt(np.mean(err_a ** 2))),
            maxdev_anchor=float(np.max(np.abs(err_a))),
            bias_anchor=float(age_anchor[-1] - t_true[-1]),
            duration_true_myr=float(t_true[-1])))
        store[name] = dict(depth=depth, signal=sig, t_true=t_true,
                           sr_true=d['sr_true'], age_const=age_const,
                           age_anchor=age_anchor,
                           anchor_depth=anchor_df['depth'].values,
                           interval_sr=anchor_df['interval_sr'].values[:-1],
                           anchor_mid=(anchor_df['depth'].values[:-1]
                                       + anchor_df['depth'].values[1:]) / 2.0)
        r = rows[-1]
        print(
    f"  {name:6s} "
    f"sr_opt={sr_opt:5.2f} "
    f"matched={matched} "
    f"IMF{idx + 1} "
    f"ncyc={r['n_cycles']} "
    f"(Ne={r['ne_true']:.1f}) "
    f"RMSE const={r['rmse_const']:.3f} "
    f"anchor={r['rmse_anchor']:.3f} Myr"
)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_DIR, 'expB_age_error.csv'), index=False)
    npz = {}
    for name, s in store.items():
        for k, v in s.items():
            npz[f'{name}__{k}'] = v
    np.savez(os.path.join(EXP_DIR, 'expB_results.npz'), **npz)
    print(f'  wrote expB_age_error.csv, expB_results.npz  '
          f'[{time.time()-t_start:.0f} s]')
    return df, store


# ----------------------------------------------------------------------
def make_fig13(df, store):
    names = list(PROFILES.keys())
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1.0, 1.0, 0.85],
                           hspace=0.34, wspace=0.30)

    for i, name in enumerate(names):
        s = store[name]
        r = df[df['scenario'] == name].iloc[0]
        # ---- (a) SR profile: true vs recovered --------------------------
        axa = fig.add_subplot(gs[i, 0])
        axa.plot(s['sr_true'], s['depth'], '-', color=C_INK, lw=1.4,
                 label='true SR(d)')
        axa.axvline(r['sr_scan_opt'], color=C_TAN, ls='--', lw=1.2,
                    label=f"constant scan\nopt. ({r['sr_scan_opt']:.1f})")
        axa.step(s['interval_sr'], s['anchor_mid'], where='mid',
                 color=C_BLUE, lw=1.3, label='405-kyr anchored\nintervals')
        axa.set_ylim(LENGTH_M, 0)
        if i == 2:
            axa.set_xlabel('SR (m/Myr)')
        axa.set_ylabel('Depth (m)')
        if i == 0:
            # Two-line labels keep the legend short enough for the clear
            # upper-left corner, away from the SR curves and intervals.
            axa.legend(
                loc='upper left', bbox_to_anchor=(0.01, 0.99),
                fontsize=6.1, frameon=False, handlelength=1.25,
                handletextpad=0.35, labelspacing=0.20, borderpad=0.15
            )
            ptitle(axa, '(a)', 'SR profiles: true vs recovered')
        axa.text(0.97, 0.97, PROFILES[name]['label'], transform=axa.transAxes,
                 ha='right', va='top', fontsize=8, color=C_GRAY)

        # ---- (b) age-error curves ---------------------------------------
        axb = fig.add_subplot(gs[i, 1])
        axb.plot(s['age_const'] - s['t_true'], s['depth'], '-', color=C_TAN,
                 lw=1.2, label='constant-SR model')
        axb.plot(s['age_anchor'] - s['t_true'], s['depth'], '-', color=C_BLUE,
                 lw=1.2, label='405-kyr anchored model')
        axb.axvline(0, color=C_GRAY, lw=0.8, ls=':')
        axb.set_ylim(LENGTH_M, 0)
        if i == 2:
            axb.set_xlabel('Age error (model − truth, Myr)')
        axb.set_ylabel('Depth (m)')
        if i == 0:
            axb.legend(loc='upper right', fontsize=7)
            ptitle(axb, '(b)', 'Age-model error curves')
        # Place each scenario's explanatory labels in its own open area.
        # Axes-fraction coordinates keep placement independent of age-error units.
        if i == 0:        # linear: clear upper-right area, beyond the curves
            note_x, note_y, note_ha = 0.97, 0.72, 'right'
        elif i == 1:      # step: upper-right open space
            note_x, note_y, note_ha = 0.98, 0.78, 'right'
        else:             # sine: upper-left open space
            note_x, note_y, note_ha = 0.06, 0.77, 'left'

        axb.text(
            note_x, note_y,
            'representative realization\n(seed k=0)',
            transform=axb.transAxes, ha=note_ha, va='top',
            fontsize=6.6, linespacing=1.15, style='italic', color=C_GRAY
        )
        axb.text(
            note_x, note_y - 0.13,
            f"RMSE {r['rmse_const']:.2f} → {r['rmse_anchor']:.2f} Myr",
            transform=axb.transAxes, ha=note_ha, va='top',
            fontsize=7.1, color=C_GRAY
        )

    # ---- (c) Monte-Carlo error statistics (expB2_mc.csv, n=10/scenario) --
    mc_path = os.path.join(EXP_DIR, 'expB2_mc.csv')
    mc = pd.read_csv(mc_path) if os.path.exists(mc_path) else None
    rng = np.random.default_rng(7)
    if mc is not None:
        scen_names = ['linear', 'step', 'sine']
        axc1 = fig.add_subplot(gs[0:2, 2])
        x = np.arange(3)
        w = 0.30
        pos_c, pos_a, data = [], [], []
        for xi, scen in enumerate(scen_names):
            g = mc[mc['scenario'] == scen]
            pos_c.append(xi - w / 2); pos_a.append(xi + w / 2)
            data.append((g['rmse_const'].values, g['rmse_anchor'].values))
        bp = axc1.boxplot([v for pair in data for v in pair],
                          positions=[p for pair in zip(pos_c, pos_a)
                                     for p in pair],
                          widths=w * 0.8, patch_artist=True,
                          medianprops=dict(color=C_INK, lw=1.6),
                          flierprops=dict(marker='', markersize=0))
        for j, patch in enumerate(bp['boxes']):
            patch.set_facecolor(C_TAN if j % 2 == 0 else C_BLUE)
            patch.set_alpha(0.35)
        for j, (pc, pa) in enumerate(zip(pos_c, pos_a)):
            rc, ra = data[j]
            axc1.scatter(pc + rng.uniform(-0.04, 0.04, rc.size), rc,
                         s=11, color=C_TAN, zorder=3,
                         label='constant-SR' if j == 0 else None)
            axc1.scatter(pa + rng.uniform(-0.04, 0.04, ra.size), ra,
                         s=11, color=C_BLUE, zorder=3,
                         label='405-kyr anchored' if j == 0 else None)
        axc1.set_xticks(x)
        axc1.set_xticklabels(scen_names)
        axc1.set_ylabel('Age RMSE (Myr)')
        axc1.legend(
            loc='upper right', bbox_to_anchor=(1.00, 0.86),
            fontsize=7, frameon=False,
            handletextpad=0.5, labelspacing=0.35
        )
        ptitle(axc1, '(c)', 'MC error statistics (n=10 realizations each)')

        axc2 = fig.add_subplot(gs[2, 2])
        for xi, scen in enumerate(scen_names):
            g = mc[mc['scenario'] == scen]
            diff = (g['rmse_anchor'] - g['rmse_const']).values
            axc2.scatter(xi + rng.uniform(-0.08, 0.08, diff.size), diff,
                         s=11, color=C_GRAY, zorder=3)
            axc2.hlines(np.median(diff), xi - 0.18, xi + 0.18,
                        color=C_RED, lw=1.6, zorder=4)
        axc2.axhline(0, color=C_INK, lw=0.9, ls='--')
        # Reserve an empty header band above all scatter points and place
        # each 2-line group label inside its own third of the panel.
        # This avoids both the negative-error points and the panel border.
        lower, upper = axc2.get_ylim()
        axc2.set_ylim(lower, max(11.0, upper))
        for xi, scen in enumerate(scen_names):
            g = mc[mc['scenario'] == scen]
            wins = int(g['anchor_wins'].sum())

            # Only move the first (linear) label slightly to the right.
            x_text = xi + (0.15 if xi == 0 else -0.12 if xi == 2 else 0.0)

            axc2.text(
                x_text, 0.96,
                f'anchor wins\n{wins}/{len(g)}',
                transform=axc2.get_xaxis_transform(),
                ha='center',
                va='top',
                fontsize=6.6,
                linespacing=1.12,
                color=C_GRAY
            )
        axc2.set_xticks(x)
        axc2.set_xticklabels(scen_names)
        axc2.set_ylabel('RMSE anchor − const (Myr)')
    else:
        axc1 = fig.add_subplot(gs[0:2, 2])
        x = np.arange(3)
        w = 0.34
        axc1.bar(x - w / 2, df['rmse_const'], width=w, color=C_TAN,
                 label='constant-SR')
        axc1.bar(x + w / 2, df['rmse_anchor'], width=w, color=C_BLUE,
                 label='405-kyr anchored')
        axc1.set_xticks(x)
        axc1.set_xticklabels(['linear', 'step', 'sine'])
        axc1.set_ylabel('Age RMSE (Myr)')
        axc1.legend(
            loc='upper right', bbox_to_anchor=(0.98, 0.86),
            fontsize=7, frameon=False
        )
        ptitle(axc1, '(c)', 'Error statistics')

        axc2 = fig.add_subplot(gs[2, 2])
        axc2.bar(x - w / 2, df['maxdev_const'], width=w, color=C_TAN)
        axc2.bar(x + w / 2, df['maxdev_anchor'], width=w, color=C_BLUE)
        axc2.set_xticks(x)
        axc2.set_xticklabels(['linear', 'step', 'sine'])
        axc2.set_ylabel('Max |age deviation| (Myr)')

    savefig(fig, 'fig13.png')


if __name__ == '__main__':
    df, store = run()
    make_fig13(df, store)
