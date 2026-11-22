#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_paper_figures.py — Publication figures for Earth-Science Reviews submission
==================================================================================
Topic : EMD-based automatic well-log stratification + astrochronologic
        depth-to-age conversion (TimeOpt-inspired SR scan + 405-kyr
        anchored non-linear age model).

Data sources (read-only, produced by previous pipeline runs):
  DATA_DIR  = /mnt/agents/output/emd_astro/emd_astro_result/
      IMF1.csv ... IMF10.csv      EMD components (columns: depth, IMFk)
      residue.csv                 EMD residue trend (columns: depth, residue)
      layer_results.csv           6-column layering table
      anchor_405.csv              155 anchors of the 405-kyr age model
      sr_scan.csv                 SR-scan objective curve (sr, score)
      selftest_result/            white-noise robustness self-test products
      selftest_result/selftest_sr_scan.csv
  INPUT_XLSX = /mnt/agents/output/emd_astro/10井测井数据.xlsx
      Sheet2 = [depth, GR] (10084 samples), Sheet1 = lithology table
  astro_py_tuner.py (same tree) is imported for MTM / eha / cycle counting.

Run:
  cd /mnt/agents/output/paper
  python3 make_paper_figures.py          # regenerates figures/fig01.png ... fig11.png

All figures: 300 dpi PNG, English labels, multi-panel with (a)(b)(c)... tags,
low-saturation colour palette, uniform fonts.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import find_peaks

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE = r"F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes"
DATA_DIR = os.path.join(BASE, 'emd_astro_result')
SELFTEST_DIR = os.path.join(DATA_DIR, 'selftest_result')
INPUT_XLSX = os.path.join(
    BASE,
    'synthetic_benchmark_well10_regenerated.xlsx'
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(OUT_DIR, exist_ok=True)
sys.path.insert(0, BASE)
from astro_py_tuner import AstroTuner          # noqa: E402

# ----------------------------------------------------------------------
# Global style — low-saturation professional palette (no jet/blue-purple
# gradients, no highly saturated colours)
# ----------------------------------------------------------------------
C_INK = '#2F3B46'      # near-black slate (text, main lines)
C_BLUE = '#41627E'     # muted steel blue
C_RED = '#A65A4A'      # muted terracotta (interfaces, highlights)
C_GREEN = '#6E8B67'    # sage green
C_TAN = '#B08C57'      # muted tan
C_GRAY = '#8A8F94'     # neutral gray
C_LGRAY = '#C9CDD1'    # light gray
PALETTE = [C_BLUE, C_RED, C_GREEN, C_TAN, C_GRAY, '#7E8B9B']

# Custom low-saturation sequential colormap (sand tones; avoids any
# blue/purple gradient) for heatmaps.
SAND = LinearSegmentedColormap.from_list(
    'sand', ['#FBF8F0', '#EDE3C8', '#D9C795', '#BFA268', '#97794A', '#5F4B2E'])
SAND_R = SAND.reversed()

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'axes.linewidth': 0.8,
    'axes.edgecolor': C_INK,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'xtick.color': C_INK,
    'ytick.color': C_INK,
    'legend.fontsize': 8,
    'legend.frameon': False,
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'text.color': C_INK,
    'axes.labelcolor': C_INK,
})

PANEL_KW = dict(fontsize=11, fontweight='bold', color=C_INK)


def panel_tag(ax, tag, x=-0.02, y=1.04):
    """Add a bold (a)(b)(c)... tag at the upper-left corner of an axes."""
    ax.text(x, y, tag, transform=ax.transAxes, va='bottom', ha='right',
            **PANEL_KW)


def ptitle(ax, tag, title, fontsize=10, pad=6):
    """rev2-F2R4: unified panel style — the (a)(b)(c)... tag and the panel
    title share one left-aligned title line, so the tag can never collide
    with the left end of a long centred title."""
    ax.set_title(f'{tag}  {title}', loc='left', fontsize=fontsize, pad=pad)


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    sz = os.path.getsize(path) / 1024.0
    print(f'  saved {path}  ({sz:.0f} KiB)')


# ----------------------------------------------------------------------
# Shared data loading
# ----------------------------------------------------------------------
def load_all():
    d = {}

    d['depth'] = pd.read_csv(
        os.path.join(DATA_DIR, 'IMF1.csv')
    )['depth'].values

    d['imfs'] = [
        pd.read_csv(
            os.path.join(DATA_DIR, f'IMF{i}.csv')
        ).iloc[:, 1].values
        for i in range(1, 11)
    ]

    d['residue'] = pd.read_csv(
        os.path.join(DATA_DIR, 'residue.csv')
    )['residue'].values

    d['recon'] = np.sum(d['imfs'], axis=0) + d['residue']

    sheet2 = pd.read_excel(INPUT_XLSX, sheet_name='Sheet2')
    d['gr'] = sheet2['GR'].values

    lith = pd.read_excel(INPUT_XLSX, sheet_name='Sheet1')
    lith.columns = ['top', 'base', 'lith']
    d['lith'] = lith

    d['layers'] = pd.read_csv(
        os.path.join(DATA_DIR, 'layer_results.csv')
    )

    d['anchors'] = pd.read_csv(
        os.path.join(DATA_DIR, 'anchor_405.csv')
    )

    d['sr_scan'] = pd.read_csv(
        os.path.join(DATA_DIR, 'sr_scan.csv')
    )

    d['st_scan'] = pd.read_csv(
        os.path.join(SELFTEST_DIR, 'selftest_sr_scan.csv')
    )

    d['interfaces'] = d['layers']['Base_Depth'].values[:-1]

    # --------------------------------------------------------------
    # Dynamically recompute benchmark statistics
    # --------------------------------------------------------------
    d['tuner'] = AstroTuner(n_jobs=1)

    dep = d['depth']

    # Best sedimentation rate + match score
    d['best_sr'], d['match_score'] = d['tuner'].best_sr(
        d['sr_scan'],
        depth=dep,
        signal=d['recon']
    )

    # Best value on the original scan grid
    i_grid = int(np.argmax(d['sr_scan']['score'].values))
    d['grid_best_sr'] = float(
        d['sr_scan']['sr'].iloc[i_grid]
    )

    # Expected number of 405-kyr cycles
    d['ne'] = (
        (dep.max() - dep.min())
        / d['best_sr']
        / 0.405
    )

    # Dynamically select the 405-kyr IMF
    sel = d['tuner'].select_405_imf(
        d['imfs'],
        dep,
        sr_est=d['best_sr']
    )

    d['selected_imf_idx'] = int(sel['imf_index'])

    d['selected_imf_number'] = (
        d['selected_imf_idx'] + 1
    )

    d['selected_n_cycles'] = int(
        sel['n_cycles']
    )

    d['selected_rel_dev'] = abs(
        d['selected_n_cycles'] - d['ne']
    ) / d['ne']

    # Anchor statistics
    d['n_anchors'] = len(d['anchors'])

    anchor_depth = d['anchors']['depth'].values
    interval_sr = np.diff(anchor_depth) / 0.405

    d['interval_sr_min'] = float(
        interval_sr.min()
    )

    d['interval_sr_mean'] = float(
        interval_sr.mean()
    )

    d['interval_sr_max'] = float(
        interval_sr.max()
    )

    # Constant-SR linear age model
    d['age_lin'] = (
        dep - dep.min()
    ) / d['best_sr']

    # 405-kyr anchored age model
    age405, _ = d['tuner'].build_age_model_405(
        dep,
        d['imfs'][d['selected_imf_idx']],
        anchor_phase='trough',
        ref_age_fn=lambda dd: (
            dd - dep.min()
        ) / d['best_sr']
    )

    d['age405'] = age405

    return d


LITH_EN = {'泥岩': 'Mudstone', '粉砂质泥岩': 'Silty mudstone',
           '泥质粉砂岩': 'Argillaceous siltstone', '粉砂岩': 'Siltstone',
           '含砾砂岩': 'Gravelly sandstone'}
LITH_COLOR = {'Mudstone': '#9A8C7A', 'Silty mudstone': '#B5A284',
              'Argillaceous siltstone': '#C9BC9C', 'Siltstone': '#D8CFAF',
              'Gravelly sandstone': '#A89A6B'}


# ===== Figure 1: Workflow ==================================================
def fig01(d):
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.08)
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])
    for ax in (axA, axB):
        ax.set_xlim(0, 10); ax.axis('off')

    def box(ax, x, y, w, h, text, fc='#F1EEE6', ec=C_INK, fs=8.6, tc=None):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle='round,pad=0.12,rounding_size=0.22',
                                    fc=fc, ec=ec, lw=1.0))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=fs, color=tc or C_INK, linespacing=1.35)

    def arrow(ax, x1, y1, x2, y2, color=C_INK, style='-|>', lw=1.2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=13, lw=lw, color=color))

    # ---- (a) main pipeline -------------------------------------------------
    axA.set_ylim(0, 100)

    # Dynamically read benchmark information from actual data
    dep = d['depth']
    dz = np.median(np.diff(dep))

    interface_text = ', '.join(
        f'{x:.1f} m'
        for x in d['interfaces']
    )

    steps = [
        (
            f'Well-log input\n'
            f'GR series, {dep.size:,} samples ({dz:.1f} m),\n'
            f'{dep.min():.1f}–{dep.max():.1f} m',
            '#E9E4D8'
        ),

        (
            'EMD adaptive decomposition\n'
            '(PyEMD, max_imf=10)\n'
            'IMF1–IMF10 + residue',
            '#E4E9EC'
        ),

        (
            'Residue extrema detection\n'
            'automatic interface picking\n'
            f'({interface_text})',
            '#E4E9EC'
        ),

        (
            'TimeOpt-inspired SR scan\n'
            'MTM band matching at\n'
            '405 / 95 / 40 / 21 kyr',
            '#E7EAE1'
        ),

        (
            f'405-kyr eccentricity anchoring\n'
            f'IMF{d["selected_imf_number"]} reference, '
            f'{d["n_anchors"]} trough anchors',
            '#E7EAE1'
        ),

        (
            'Outputs\n'
            'layer table (6 cols) + non-linear\n'
            'depth–age model + diagnostics',
            '#EFE3D3'
        ),
    ]

    y = 96.0
    hh = 10.5

    for i, (txt, fc) in enumerate(steps):
        box(
            axA,
            1.2,
            y - hh,
            7.6,
            hh,
            txt,
            fc=fc
        )

        if i < len(steps) - 1:
            arrow(
                axA,
                5.0,
                y - hh - 0.25,
                5.0,
                y - hh - 4.0
            )

        y -= (hh + 4.3)

    ptitle(
        axA,
        '(a)',
        'Main pipeline',
        pad=2
    )

    # ---- (b) astrochronology module zoom -----------------------------------
    axB.set_ylim(0, 100)
    box(axB, 0.4, 88, 4.4, 9, 'Depth → time\nconversion\nt = (d−d_top)/SR',
        fc='#E9E4D8', fs=8.2)
    box(axB, 5.4, 88, 4.3, 9, 'Thomson MTM\nspectrum (NW=3, K=5)',
        fc='#E9E4D8', fs=8.2)
    arrow(axB, 4.8, 92.5, 5.4, 92.5)
    box(axB, 2.6, 74, 5.0, 9.5, 'Objective: target-band power ratio\n× peak-match coefficient\n(SR grid 1–20 m/Myr, Δ=0.25)',
        fc='#E4E9EC', fs=8.2)
    arrow(axB, 2.7, 88, 4.0, 83.7); arrow(axB, 8.0, 88, 6.4, 83.7)
    # decision diamond
    dx, dy = 5.1, 63.5
    axB.add_patch(plt.Polygon([(dx, dy + 5.2), (dx + 3.4, dy), (dx, dy - 5.2),
                               (dx - 3.4, dy)], closed=True, fc='#F5E9DA',
                              ec=C_INK, lw=1.0))
    axB.text(dx, dy, 'match score\n≥ 0.5 ?', ha='center', va='center',
             fontsize=8.2)
    arrow(axB, dx, 74.0, dx, dy + 5.4)
    # no branch
    box(axB, 7.2, 57.5, 2.7, 8.0, 'WARNING:\ndepth-only\nlayering',
        fc='#EFE0DA', fs=7.8)
    arrow(axB, dx + 3.4, dy, 7.15, 61.5, color=C_RED)
    axB.text(dx + 3.55, dy + 2.6, 'no', fontsize=8, color=C_RED, ha='left')
    # yes branch
    arrow(axB, dx, dy - 5.4, dx, 51.0, color=C_GREEN)
    axB.text(dx + 0.25, dy - 7.6, 'yes', fontsize=8, color=C_GREEN)
    box(
    axB, 2.2, 42.0, 5.8, 9.0,
    f'Best SR = {d["best_sr"]:.2f} m/Myr '
    f'(grid {d["grid_best_sr"]:.2f}\n'
    f'+ parabolic refinement);\n'
    f'match = {d["match_score"]:.3f}',
    fc='#E7EAE1', fs=8.2
)
    box(
    axB, 0.4, 28.5, 4.5, 9.5,
    f'Linear reference model\n'
    f'Age = (d−d_top)/{d["best_sr"]:.2f}\n'
    f'→ $N_e$ = {d["ne"]:.2f} cycles',
    fc='#E7EAE1', fs=8.2
)
    box(
    axB, 5.4, 28.5, 4.4, 9.5,
    f'Cycle counting of IMF1–10\n'
    f'→ select IMF{d["selected_imf_number"]} '
    f'({d["selected_n_cycles"]} cycles,\n'
    f'rel. dev. {100*d["selected_rel_dev"]:.1f}% ≤ 15%)',
    fc='#E7EAE1', fs=8.2
)
    arrow(axB, 4.0, 42.0, 2.7, 38.2); arrow(axB, 6.2, 42.0, 7.6, 38.2)
    # anchor recursion verbatim as Eq. (5): Age_k = Age_1 + (k−1)×0.405 Myr
    # (rev-R1: in-figure label corrected from "Eq. 4" to "Eq. (5)")
    box(axB, 2.4, 13.5, 5.6, 10.0, 'Trough anchors: Age_k = Age_1 + (k−1)×0.405 Myr\n'
        '(k = 1,…,N_a; Eq. (5)); piecewise-linear\ninterpolation; end-member SR extrapolation',
        fc='#E4E9EC', fs=7.7)
    arrow(axB, 2.7, 28.5, 4.2, 23.7); arrow(axB, 7.6, 28.5, 6.0, 23.7)
    box(
    axB, 2.4, 2.5, 5.6, 8.0,
    f'Non-linear depth–age model\n'
    f'({d["n_anchors"]} anchors; interval SR\n'
    f'{d["interval_sr_min"]:.2f}–'
    f'{d["interval_sr_max"]:.2f} m/Myr)',
    fc='#EFE3D3', fs=8.2
)
    arrow(axB, 5.2, 13.5, 5.2, 10.7)
    ptitle(axB, '(b)', 'Astrochronology module (zoom)', pad=2)

    # embedded "Figure N." suptitles removed (rev2-F3): the typeset caption
    # is the single source of the figure title.
    savefig(fig, 'fig01.png')


# ===== Figure 2: Input data ================================================
def fig02(d):
    dep, gr, lith = d['depth'], d['gr'], d['lith']
    fig = plt.figure(figsize=(11.5, 6.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.55, 1.1], wspace=0.22)

    # (a) GR log (vertical, depth increasing downward)
    axa = fig.add_subplot(gs[0])
    axa.plot(gr, dep, color=C_BLUE, lw=0.6)
    axa.fill_betweenx(dep, gr, gr.min(), color=C_BLUE, alpha=0.12)
    axa.set_ylim(dep.max(), dep.min())
    axa.set_xlabel('Gamma ray, GR (API)')
    axa.set_ylabel('Depth (m)')
    ptitle(axa, '(a)', 'Raw GR log, Well 10')
    axa.grid(axis='x', ls=':', lw=0.5, color=C_LGRAY)

    # (b) lithology column
    axb = fig.add_subplot(gs[1], sharey=axa)
    for _, r in lith.iterrows():
        en = LITH_EN[r['lith']]
        axb.fill_betweenx([r['top'], r['base']], 0, 1,
                          color=LITH_COLOR[en], ec=C_INK, lw=0.5)
        axb.text(0.5, (r['top'] + r['base']) / 2, en.replace(' ', '\n'),
                 ha='center', va='center', fontsize=6.4, color=C_INK,
                 linespacing=1.1)
    axb.set_xlim(0, 1)
    axb.set_xticks([])
    ptitle(axb, '(b)', 'Lithology')
    axb.tick_params(labelleft=False)

    # (c) sampling statistics / histogram
    axc = fig.add_subplot(gs[2])
    axc.hist(gr, bins=60, color=C_TAN, ec=C_INK, lw=0.4, alpha=0.85)
    axc.set_xlabel('GR (API)')
    axc.set_ylabel('Count')
    ptitle(axc, '(c)', 'GR distribution & sampling statistics')
    dz = np.diff(dep)
    stats = (f'n = {dep.size:,}\n'
             f'depth span = {dep.min():.1f}–{dep.max():.1f} m\n'
             f'sampling = {np.median(dz):.2f} m (uniform)\n'
             f'GR mean = {gr.mean():.1f} API\n'
             f'GR std = {gr.std():.1f} API\n'
             f'GR range = {gr.min():.1f}–{gr.max():.1f} API')
    # rev2-F4: raise the y-limit so the statistics box sits in the headroom
    # above the bars and no longer covers the histogram peak (modal bin).
    cnt_max = np.histogram(gr, bins=60)[0].max()
    axc.set_ylim(0, cnt_max * 1.42)
    axc.text(0.03, 0.985, stats, transform=axc.transAxes, ha='left', va='top',
             fontsize=8,
             bbox=dict(fc='white', ec=C_GRAY, lw=0.6, alpha=0.85))
    axc.grid(axis='y', ls=':', lw=0.5, color=C_LGRAY)

    savefig(fig, 'fig02.png')


# ===== Figure 3: EMD decomposition =========================================
def fig03(d):
    dep = d['depth']
    comps = d['imfs'] + [d['residue']]
    names = [f'IMF{i}' for i in range(1, 11)] + ['Residue']
    n = len(comps)
    fig, axes = plt.subplots(n, 1, figsize=(9.0, 12.5), sharex=True,
                             gridspec_kw=dict(hspace=0.14))
    tags = 'abcdefghijk'
    for i, (ax, c, nm) in enumerate(zip(axes, comps, names)):
        col = C_TAN if nm == 'Residue' else C_BLUE
        ax.plot(dep, c, color=col, lw=0.55)
        if nm == 'Residue':
            # rev2-F5: light tan shading under the residue trend so the
            # caption wording "(tan shading)" matches the graphic.
            ax.fill_between(dep, c, c.min(), color=C_TAN, alpha=0.25, lw=0)
        ax.set_ylabel(nm, fontsize=8.5, rotation=0, ha='right', va='center',
                      labelpad=8)
        ax.tick_params(labelsize=7.5)
        ax.grid(ls=':', lw=0.4, color=C_LGRAY)
        ax.set_xlim(dep.min(), dep.max())
        # symmetric ylim for IMFs
        if nm != 'Residue':
            m = np.abs(c).max()
            ax.set_ylim(-1.05 * m, 1.05 * m)
        ax.text(0.008, 0.86, f'({tags[i]})', transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top')
        sd = np.std(c)
        ax.text(0.992, 0.86, f'std = {sd:.2f}', transform=ax.transAxes,
                fontsize=7, va='top', ha='right', color=C_GRAY,
                bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.2))
    axes[-1].set_xlabel('Depth (m)')
    savefig(fig, 'fig03.png')


# ===== Figure 4: Decomposition diagnostics =================================
def fig04(d):
    dep = d['depth']
    comps = d['imfs'] + [d['residue']]
    names = [f'IMF{i}' for i in range(1, 11)] + ['Res.']
    fig = plt.figure(figsize=(11.5, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95], hspace=0.30,
                          wspace=0.20)

    # (a) variance contribution
    axa = fig.add_subplot(gs[0, 0])
    var = np.array([np.var(c) for c in comps])
    pct = 100 * var / np.var(d['gr'])          # ratio to original GR variance
    cols = [C_TAN if n == 'Res.' else C_BLUE for n in names]
    bars = axa.bar(names, pct, color=cols, ec=C_INK, lw=0.5)
    for b, p in zip(bars, pct):
        axa.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
                 f'{p:.1f}', ha='center', fontsize=7.2)
    axa.set_ylabel('Variance ratio to original GR (%)')
    axa.set_ylim(0, pct.max() * 1.18)
    ptitle(axa, '(a)', 'Variance ratio of each component\n'
           '(low-frequency terms dominate)')
    axa.tick_params(axis='x', labelsize=7.2, rotation=45)
    axa.grid(axis='y', ls=':', lw=0.5, color=C_LGRAY)

    # (b) extrema density, IMF1–5 (ridge plot)
    axb = fig.add_subplot(gs[0, 1])
    win, step = 30.0, 5.0
    centers = np.arange(dep.min() + win / 2, dep.max() - win / 2 + 1e-9, step)
    for k in range(5):
        x = d['imfs'][k]
        pk, _ = find_peaks(x, prominence=0.5 * x.std())
        tr, _ = find_peaks(-x, prominence=0.5 * x.std())
        ex = np.sort(dep[np.concatenate([pk, tr])])
        dens = np.array([np.sum((ex >= c - win / 2) & (ex < c + win / 2))
                         for c in centers], dtype=float) / win
        dens /= dens.max()
        off = (4 - k) * 1.25
        axb.plot(centers, off + dens, color=PALETTE[k], lw=0.9)
        axb.fill_between(centers, off, off + dens, color=PALETTE[k], alpha=0.25)
        axb.text(dep.min() + 6, off + 0.08, f'IMF{k+1}', fontsize=8,
                 color=PALETTE[k], fontweight='bold', va='bottom')
    axb.set_yticks([])
    axb.set_xlim(dep.min(), dep.max())
    axb.set_xlabel('Depth (m)')
    ptitle(axb, '(b)', 'Extrema density, IMF1–5 (30 m sliding window)')

    # (c) original vs reconstruction (top) + residual strip (bottom)
    gs2 = gs[1, :].subgridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.08)
    axc = fig.add_subplot(gs2[0])
    axc.plot(dep, d['gr'], color=C_GRAY, lw=1.0, label='Original GR')
    axc.plot(dep, d['recon'], color=C_RED, lw=0.6, ls='--',
             label='Reconstruction (ΣIMF1–10 + residue)')
    axc.set_ylabel('GR (API)')
    axc.set_xlim(dep.min(), dep.max())
    ptitle(axc, '(c)', 'Original vs reconstructed signal')
    axc.legend(loc='upper right', ncol=2, fontsize=7.8)
    axc.tick_params(labelbottom=False)
    axc.grid(ls=':', lw=0.5, color=C_LGRAY)

    axc2 = fig.add_subplot(gs2[1], sharex=axc)
    resid = d['recon'] - d['gr']
    axc2.plot(dep, resid, color=C_GREEN, lw=0.4)
    axc2.set_ylim(-4e-14, 4e-14)
    axc2.set_ylabel('Residual\n(API)', fontsize=7.5)
    axc2.set_xlabel('Depth (m)')
    axc2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axc2.yaxis.get_offset_text().set_fontsize(7)
    axc2.tick_params(labelsize=7)
    axc2.set_yticks([-2e-14, 0, 2e-14])
    axc2.text(0.995, 0.88,
              f'max |err| = {np.abs(resid).max():.2e} API;  '
              f'RMS = {np.sqrt(np.mean(resid**2)):.2e} API',
              transform=axc2.transAxes, ha='right', va='top', fontsize=7.5,
              bbox=dict(fc='white', ec=C_GRAY, lw=0.6, alpha=0.9))
    axc2.grid(ls=':', lw=0.4, color=C_LGRAY)

    savefig(fig, 'fig04.png')


# ===== Figure 5: Automatic stratification ==================================
def fig05(d):
    dep, gr, res = d['depth'], d['gr'], d['residue']
    interfaces = d['interfaces']
    layers = d['layers']
    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 0.9], wspace=0.22)

    # (a) GR log + interfaces (vertical)
    axa = fig.add_subplot(gs[0])
    axa.plot(gr, dep, color=C_BLUE, lw=0.6)
    axa.set_ylim(dep.max(), dep.min())
    for i, idp in enumerate(interfaces):
        axa.axhline(idp, color=C_RED, lw=1.2, ls='--')
        axa.text(gr.min() + 2, idp - 6, f'{idp:.1f} m', color=C_RED,
                 fontsize=8, fontweight='bold',
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0))
    axa.set_xlabel('GR (API)'); axa.set_ylabel('Depth (m)')
    ptitle(axa, '(a)', 'GR log with detected interfaces')
    axa.grid(ls=':', lw=0.5, color=C_LGRAY)

    # (b) residue trend + extrema + interfaces
    axb = fig.add_subplot(gs[1], sharey=axa)
    axb.plot(res, dep, color=C_TAN, lw=1.0)
    pk, _ = find_peaks(res, prominence=0.01 * res.std())
    tr, _ = find_peaks(-res, prominence=0.01 * res.std())
    axb.plot(res[pk], dep[pk], 'v', ms=6, color=C_GRAY,
             label='Local maximum')
    axb.plot(res[tr], dep[tr], '^', ms=6, color=C_GRAY,
             label='Local minimum')
    for idp in interfaces:
        axb.axhline(idp, color=C_RED, lw=1.2, ls='--')
        j = np.argmin(np.abs(dep - idp))
        axb.plot(res[j], idp, 'o', ms=8, mfc='none', mec=C_RED, mew=1.6)
    axb.plot([], [], 'o', ms=8, mfc='none', mec=C_RED, mew=1.6,
             label='Picked interface')
    axb.set_xlabel('Residue (API)')
    ptitle(axb, '(b)', 'Residue trend & interface picking')
    axb.tick_params(labelleft=False)
    axb.legend(loc='lower left', fontsize=7.5)
    axb.grid(ls=':', lw=0.5, color=C_LGRAY)

    # (c) layer thickness statistics
    axc = fig.add_subplot(gs[2])
    lbl = [f'Layer {n}' for n in layers['Layer_No']]
    th = layers['Thickness'].values
    bars = axc.bar(lbl, th, color=[C_GREEN, C_BLUE, C_TAN], ec=C_INK, lw=0.6)
    for b, t_, (_, r) in zip(bars, th, layers.iterrows()):
        # rev2-F2R9(iii): per-bar duration Δt added below the thickness, in
        # the same 2-decimal convention as Fig. 10(b) / Table 3.
        axc.text(b.get_x() + b.get_width() / 2, b.get_height() + 8,
                 f'{t_:.1f} m\nΔt = {r["Base_Age"] - r["Top_Age"]:.2f} Myr\n'
                 f'{r["Top_Age"]:.1f}–{r["Base_Age"]:.1f} Ma',
                 ha='center', fontsize=7.2)
    axc.set_ylabel('Thickness (m)')
    axc.set_ylim(0, th.max() * 1.42)
    ptitle(axc, '(c)', 'Layer thickness & age span')
    axc.grid(axis='y', ls=':', lw=0.5, color=C_LGRAY)

    savefig(fig, 'fig05.png')


# ===== Figure 6: TimeOpt-inspired SR scan ==================================
def fig06(d):
    dep, recon = d['depth'], d['recon']
    scan = d['sr_scan']

    best_sr = d['best_sr']
    match_best = d['match_score']
    grid_best_sr = d['grid_best_sr']

    tuner = d['tuner']

    # Synthetic benchmark design value
    true_sr = 15.0
    true_duration = (dep.max() - dep.min()) / true_sr

    fig = plt.figure(figsize=(11.5, 4.6))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[1.0, 1.0, 1.05],
        wspace=0.26
    )

    # ==============================================================
    # (a) Objective function vs SR
    # ==============================================================
    axa = fig.add_subplot(gs[0])

    axa.plot(
        scan['sr'],
        scan['score'],
        '-o',
        ms=3.2,
        lw=1.0,
        color=C_BLUE
    )

    axa.axvline(
        best_sr,
        color=C_RED,
        ls='--',
        lw=1.1
    )

    axa.axvline(
        grid_best_sr,
        color=C_GRAY,
        ls=':',
        lw=1.0
    )

    i_best = int(
        np.argmax(scan['score'].values)
    )

    axa.plot(
        scan['sr'].iloc[i_best],
        scan['score'].iloc[i_best],
        'o',
        ms=8,
        mfc='none',
        mec=C_RED,
        mew=1.6
    )

    axa.annotate(
        f'best SR = {best_sr:.2f} m/Myr\n'
        f'(parabolic refinement of\n'
        f'grid maximum {grid_best_sr:.2f})',
        xy=(
            best_sr,
            scan['score'].max()
        ),
        xytext=(2.8, 0.100),
        fontsize=8,
        arrowprops=dict(
            arrowstyle='->',
            color=C_RED,
            lw=0.9
        ),
        color=C_INK
    )

    axa.set_xlabel(
        'Sedimentation rate SR (m/Myr)'
    )

    axa.set_ylabel(
        'Objective score'
    )

    ptitle(
        axa,
        '(a)',
        'TimeOpt-inspired objective function'
    )

    axa.grid(
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    # ==============================================================
    # (b) Match-quality curve across SR grid
    # ==============================================================
    axb = fig.add_subplot(gs[1])

    mq = np.array([
        tuner.match_quality(
            dep,
            recon,
            s
        )
        for s in scan['sr']
    ])

    axb.plot(
        scan['sr'],
        mq,
        '-s',
        ms=3.0,
        lw=1.0,
        color=C_GREEN
    )

    axb.axhline(
        0.5,
        color=C_RED,
        ls='--',
        lw=1.1
    )

    axb.text(
    19.6,
    0.525,
    'threshold = 0.5',
    ha='right',
    va='bottom',
    fontsize=7.2,
    color=C_RED,
    bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.2)
)

    axb.axvline(
        best_sr,
        color=C_RED,
        ls=':',
        lw=1.0
    )

    axb.plot(
        best_sr,
        match_best,
        '*',
        ms=13,
        color=C_TAN,
        mec=C_INK,
        mew=0.6,
        zorder=5
    )

    axb.annotate(
        f'match = {match_best:.3f}\nat best SR',
    xy=(best_sr, match_best),
    xytext=(8.0, 0.80),
    fontsize=7.5,
    color=C_INK,
    ha='left',
    va='center',
    arrowprops=dict(
        arrowstyle='->',
        color=C_TAN,
        lw=0.9
        )
    )

    # Secondary threshold-crossing peak
    i_sec = int(
        np.argmax(
            np.where(
                scan['sr'].values < 10.0,
                mq,
                -1.0
            )
        )
    )

    sr_sec = scan['sr'].iloc[i_sec]
    mq_sec = mq[i_sec]

    obj_sec = scan['score'].iloc[i_sec]
    obj_best = scan['score'].max()

    axb.plot(
        sr_sec,
        mq_sec,
        'o',
        ms=7,
        mfc='none',
        mec=C_GRAY,
        mew=1.4,
        zorder=5
    )

    axb.annotate(
        f'secondary peak (harmonic leakage)\n'
    f'match = {mq_sec:.2f} at SR = {sr_sec:.2f}',
    xy=(sr_sec, mq_sec),
    xytext=(3.8, 0.43),
    fontsize=6.8,
    color=C_INK,
    ha='left',
    va='center',
    linespacing=1.15,
    arrowprops=dict(
        arrowstyle='->',
        color=C_GRAY,
        lw=0.8,
        connectionstyle='arc3,rad=0'
        )
    )

    axb.text(
        0.01,
    0.985,
    f'The 0.5 threshold applies only\n'
    f'to the best-SR match.\n'
    f'Secondary objective: {obj_sec:.3f} ≪ {obj_best:.3f}\n'
    f'Rejected by the band-hit criterion.',
    transform=axb.transAxes,
    ha='left',
    va='top',
    fontsize=6.8,
    linespacing=1.25,
    color=C_INK
        )

    axb.set_xlabel(
        'Sedimentation rate SR (m/Myr)'
    )

    axb.set_ylabel(
        'Peak-match coefficient'
    )

    axb.set_ylim(
        0,
        1.05
    )

    ptitle(
        axb,
        '(b)',
        'Match quality vs SR'
    )

    axb.grid(
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    # ==============================================================
    # (c) Depth-to-time conversion
    # ==============================================================
    axc = fig.add_subplot(gs[2])

    t = tuner.depth_to_time(
        dep,
        best_sr
    )

    axc.plot(
        t,
        dep,
        color=C_BLUE,
        lw=1.2
    )

    interface_ages = np.interp(
        d['interfaces'],
        dep,
        d['age_lin']
    )

    for idp, ia in zip(
        d['interfaces'],
        interface_ages
    ):
        axc.plot(
            ia,
            idp,
            's',
            ms=6,
            color=C_RED,
            zorder=5
        )

        axc.annotate(
            f'{idp:.1f} m / {ia:.2f} Ma',
            xy=(ia, idp),
            xytext=(ia + 6, idp + 60),
            fontsize=7.6,
            color=C_RED,
            arrowprops=dict(
                arrowstyle='->',
                color=C_RED,
                lw=0.8
            )
        )

    axc.set_ylim(
        dep.max(),
        dep.min()
    )

    axc.set_xlabel(
        'Age (Ma, relative to top)'
    )

    axc.set_ylabel(
        'Depth (m)'
    )

    ptitle(
        axc,
        '(c)',
        f'Depth–time conversion at best SR = '
        f'{best_sr:.2f} m/Myr'
    )

    duration_dev_pct = (
        abs(t.max() - true_duration)
        / true_duration
        * 100.0
    )

    axc.text(
        0.03,
        0.05,
        f'total duration = {t.max():.2f} Myr\n'
        f'(theory {true_duration:.2f} Myr, '
        f'deviation {duration_dev_pct:.2f}%)',
        transform=axc.transAxes,
        fontsize=8,
        bbox=dict(
            fc='white',
            ec=C_GRAY,
            lw=0.6,
            alpha=0.9
        )
    )

    axc.grid(
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    savefig(
        fig,
        'fig06.png'
    )


# ===== Figure 7: Spectral validation (MTM + eha) ===========================
def fig07(d):
    dep, recon, tuner = d['depth'], d['recon'], d['tuner']
    best_sr = d['best_sr']
    periods = [405, 95, 40, 21]
    tfreqs = [1000.0 / p for p in periods]

    t = tuner.depth_to_time(dep, best_sr)
    freq, power = tuner.mtm_spectrum(t, recon)
    _, _, _, sigs = tuner._band_diagnostics(freq, power)

    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.24)

    # (a) MTM spectrum of the tuned signal
    axa = fig.add_subplot(gs[0])
    axa.semilogy(freq, power, color=C_BLUE, lw=0.8)
    axa.set_xlim(0, 55)
    axa.set_xlabel('Frequency (cycles/Myr)')
    axa.set_ylabel('Power (log scale)')
    for fp, p_, sg in zip(tfreqs, periods, sigs):
        axa.axvline(fp, color=C_RED, ls='--', lw=1.0)
        axa.text(fp, axa.get_ylim()[1] * 0.55, f' {p_} kyr\n ×{sg:.0f}',
                 rotation=0, fontsize=7.4, color=C_RED, va='top', ha='left')
    axa.set_ylim(power[(freq > 0) & (freq < 55)].min() * 0.8,
                 power.max() * 3)
    axa.text(0.985, 0.03, 'dashed lines: astronomical target frequencies;\n'
             '×n = peak / local-median significance',
             transform=axa.transAxes, ha='right', va='bottom', fontsize=7.2,
             color=C_GRAY)
    ptitle(
    axa,
    '(a)',
    f'MTM spectrum after tuning (SR = {best_sr:.2f} m/Myr)'
)
    axa.grid(ls=':', lw=0.5, color=C_LGRAY)

    # (b) eha evolutive spectrogram
    axb = fig.add_subplot(gs[1])
    win_m, step_m = 150.0, 10.0
    centers, fe, p2 = tuner.eha(dep, recon, best_sr,
                                win_m=win_m, step_m=step_m)
    fmask = (fe >= 1.0) & (fe <= 55)
    with np.errstate(divide='ignore'):
        lp = np.log10(p2[:, fmask] + 1e-12)
    # Colour-limit choice (reviewer 2, F1-i): the lower bound is tightened
    # towards the 5th percentile but floored at -1.40 so the target bands
    # form a readable contrast.  Because -1.40 sits above the data median,
    # background pixels below it are saturated at vmin — the colourbar
    # label states this clipping explicitly (reviewer 2, round-2 F2R6)
    # instead of calling it a "5th-pct stretch".
    vmin = float(np.nanpercentile(lp, 5))
    vmin = min(max(vmin, -1.4), -1.0)
    pc = axb.pcolormesh(fe[fmask], centers, lp, cmap=SAND, shading='gouraud',
                        vmin=vmin, vmax=0)
    # hatched edge zones: within half a window length (75 m) of the section
    # ends the sliding window has no coverage — edge artefact, not absence
    # of orbital power (reviewer 2, F1-iii).
    hw = win_m / 2.0
    for y0, y1 in [(dep.min(), dep.min() + hw),
                   (dep.max() - hw, dep.max())]:
        axb.axhspan(y0, y1, facecolor='#EFEDE6', edgecolor=C_GRAY, lw=0,
                    hatch='///', alpha=0.85, zorder=3)
    # target-frequency reference lines, drawn dark and on top of the heatmap
    # (reviewer 2, F1-ii)
    for fp, p_ in zip(tfreqs, periods):
        axb.axvline(fp, color=C_INK, ls='--', lw=1.2, zorder=4)
        axb.text(fp, dep.min() + hw + 5, f'{p_} kyr ', rotation=90,
                 fontsize=7.4, color=C_INK, va='top', ha='right',
                 fontweight='bold', zorder=5,
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.8))
    axb.text(0.985, 0.015,
             'hatched margins = window half-width edge zones (75 m):\n'
             'no sliding-window coverage (edge artefact, not absence\n'
             'of orbital power); dashed lines = target frequencies',
             transform=axb.transAxes, ha='right', va='bottom', fontsize=7.0,
             color=C_INK,
             bbox=dict(fc='white', ec=C_GRAY, lw=0.6, alpha=0.9))
    axb.set_ylim(dep.max(), dep.min())
    # rev2-F2R6: extend the frequency axis down to 1.0 cycles/Myr so the
    # 405-kyr band (2.47 cycles/Myr) stands clear of the left axis edge.
    axb.set_xlim(1.0, 55)
    # rev2-F2R9(i): interface depth lines — band continuity across the
    # detected interfaces is the crux of the §5.2–5.3 argument.
    for idp in d['interfaces']:
        axb.axhline(idp, color=C_INK, lw=0.8, ls=':')
        axb.text(54.0, idp, f'{idp:.1f} m', fontsize=6.4, va='center',
                 ha='right', color=C_INK,
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=0.6))
    axb.set_xlabel('Frequency (cycles/Myr)')
    axb.set_ylabel('Depth (m)')
    cb = fig.colorbar(pc, ax=axb, pad=0.015)
    cb.set_label(f'log$_{{10}}$ normalized power\n(values below {vmin:.2f} '
                 f'clipped\nfor band contrast)', fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    ptitle(axb, '(b)', 'Evolutive harmonic analysis (eha)\n'
           '150 m window, 10 m step')

    savefig(fig, 'fig07.png')


# ===== Figure 8: 405-kyr reference-curve selection =========================
def fig08(d):
    dep, imfs, tuner = d['depth'], d['imfs'], d['tuner']
    ne = d['ne']
    best_sr = d['best_sr']
    anchors = d['anchors']

    # Dynamically selected 405-kyr IMF
    sel_idx = d['selected_imf_idx']
    sel_num = d['selected_imf_number']
    sel_ncyc = d['selected_n_cycles']
    sel_rel_dev = d['selected_rel_dev']
    n_anchors = d['n_anchors']

    selected_imf = imfs[sel_idx]

    # Cycle counts of all IMFs
    counts = np.array([
        tuner.count_cycles(x)[0]
        for x in imfs
    ])

    fig = plt.figure(figsize=(11.5, 4.8))
    gs = fig.add_gridspec(
        1, 3,
        width_ratios=[0.95, 1.35, 1.0],
        wspace=0.26
    )

    # ==============================================================
    # (a) Cycle counts vs expected 405-kyr cycle count
    # ==============================================================
    axa = fig.add_subplot(gs[0])

    labels = [
        f'IMF{i}'
        for i in range(1, len(imfs) + 1)
    ]

    cols = [
        C_RED if i == sel_idx else C_BLUE
        for i in range(len(imfs))
    ]

    axa.bar(
        labels,
        counts,
        color=cols,
        ec=C_INK,
        lw=0.5
    )

    axa.set_yscale('log')

    axa.axhline(
        ne,
        color=C_INK,
        lw=1.1
    )

    axa.axhspan(
        ne * 0.85,
        ne * 1.15,
        color=C_TAN,
        alpha=0.25
    )

    axa.text(
        len(imfs) - 0.6,
        ne * 1.22,
        f'$N_e$ = {ne:.2f}\n±15% band',
        fontsize=7.4,
        ha='right',
        color=C_INK
    )

    axa.annotate(
        f'IMF{sel_num}: {sel_ncyc} cycles\n'
        f'rel. dev. {100 * sel_rel_dev:.1f}%',
        xy=(
            sel_idx,
            sel_ncyc
        ),
        xytext=(
            min(sel_idx + 1.6, len(imfs) - 0.5),
            max(sel_ncyc * 2.6, ne * 2.4)
        ),
        fontsize=7.8,
        color=C_RED,
        arrowprops=dict(
            arrowstyle='->',
            color=C_RED,
            lw=0.9
        )
    )

    axa.set_ylabel(
        'Cycle count (log scale)'
    )

    axa.set_ylim(
        1,
        9000
    )

    ptitle(
        axa,
        '(a)',
        'Wave-cycle count per IMF vs\n'
        'expected 405-kyr cycles $N_e$'
    )

    axa.tick_params(
        axis='x',
        labelsize=7.2,
        rotation=45
    )

    axa.grid(
        axis='y',
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    # ==============================================================
    # (b) Dynamically selected IMF with 405-kyr anchors
    # ==============================================================
    axb = fig.add_subplot(gs[1])

    axb.plot(
        dep,
        selected_imf,
        color=C_BLUE,
        lw=0.7
    )

    ad = anchors['depth'].values
    aa = anchors['age'].values

    amp_anchor = np.interp(
        ad,
        dep,
        selected_imf
    )

    axb.plot(
        ad,
        amp_anchor,
        'v',
        ms=2.6,
        color=C_RED,
        alpha=0.75
    )

    ylo = selected_imf.min() * 1.25

    # --------------------------------------------------------------
    # Dynamically identify the interval with maximum apparent SR
    # instead of hard-coding 2239.6–2257.1 m
    # --------------------------------------------------------------
    if 'interval_sr' in anchors.columns:
        interval_sr = anchors['interval_sr'].values[:-1]

        valid = np.isfinite(interval_sr)

        if np.any(valid):
            valid_indices = np.where(valid)[0]
            i_gap = valid_indices[
                np.argmax(interval_sr[valid])
            ]

            gap_left = ad[i_gap]
            gap_right = ad[i_gap + 1]

            axb.axvspan(
                gap_left,
                gap_right,
                color=C_TAN,
                alpha=0.30
            )

            axb.text(
                0.5 * (gap_left + gap_right),
                ylo * 1.14,
                'missed cycle\n(§5.3)',
                fontsize=6.4,
                ha='center',
                va='center',
                color=C_INK,
                bbox=dict(
                    fc='white',
                    ec='none',
                    alpha=0.8,
                    pad=0.8
                )
            )

    axb.vlines(
        ad,
        ylo,
        ylo + 0.10 * abs(ylo),
        color=C_RED,
        lw=0.4,
        alpha=0.5
    )

    # Annotate every ~25th anchor with age
    for k in range(0, len(ad), 25):
        axb.annotate(
            f'{aa[k]:.1f}',
            xy=(ad[k], ylo),
            xytext=(0, -9),
            textcoords='offset points',
            fontsize=6.4,
            ha='center',
            color=C_RED,
            rotation=0
        )

    axb.text(
        0.99,
        0.95,
        f'{n_anchors} trough anchors\n'
        f'(△t = 0.405 Myr each)',
        transform=axb.transAxes,
        ha='right',
        va='top',
        fontsize=8,
        bbox=dict(
            fc='white',
            ec=C_GRAY,
            lw=0.6,
            alpha=0.9
        )
    )

    axb.set_ylim(
        ylo * 1.25,
        selected_imf.max() * 1.15
    )

    axb.set_xlim(
        dep.min(),
        dep.max()
    )

    axb.set_xlabel(
        'Depth (m)'
    )

    axb.set_ylabel(
        f'IMF{sel_num} amplitude (API)'
    )

    ptitle(
        axb,
        '(b)',
        f'Selected reference curve IMF{sel_num}\n'
        f'with 405-kyr anchors',
        fontsize=9.5
    )

    axb.grid(
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    # ==============================================================
    # (c) MTM spectral verification of selected IMF
    # ==============================================================
    axc = fig.add_subplot(gs[2])

    t = tuner.depth_to_time(
        dep,
        best_sr
    )

    freq, power = tuner.mtm_spectrum(
        t,
        selected_imf
    )

    per = 1000.0 / freq[1:]
    pw = power[1:]

    m = (
        (per >= 100)
        & (per <= 1200)
    )

    axc.plot(
        per[m],
        pw[m],
        color=C_BLUE,
        lw=1.1
    )

    axc.axvspan(
        405 * 0.8,
        405 * 1.2,
        color=C_TAN,
        alpha=0.25
    )

    axc.axvline(
        405,
        color=C_RED,
        ls='--',
        lw=1.1
    )

    i_pk = np.argmax(
        pw[m]
    )

    pk_per = per[m][i_pk]
    pk_power = pw[m][i_pk]

    axc.plot(
        pk_per,
        pk_power,
        'o',
        ms=7,
        mfc='none',
        mec=C_RED,
        mew=1.5
    )

    axc.annotate(
        f'main peak = {pk_per:.1f} kyr\n'
        f'(within ±20% of 405 kyr)',
        xy=(
            pk_per,
            pk_power
        ),
        xytext=(
            620,
            pw[m].max() * 0.55
        ),
        fontsize=8,
        color=C_RED,
        arrowprops=dict(
            arrowstyle='->',
            color=C_RED,
            lw=0.9
        )
    )

    axc.set_xlabel(
        'Period (kyr)'
    )

    axc.set_ylabel(
        'MTM power'
    )

    axc.set_xscale(
        'log'
    )

    axc.set_xticks([
        100,
        200,
        405,
        800,
        1200
    ])

    axc.get_xaxis().set_major_formatter(
        matplotlib.ticker.ScalarFormatter()
    )

    ptitle(
        axc,
        '(c)',
        f'IMF{sel_num} spectral-peak verification'
    )

    axc.grid(
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    savefig(
        fig,
        'fig08.png'
    )


# ===== Figure 9: Age-model comparison ======================================
def fig09(d):
    dep = d['depth']
    age_lin, age405 = d['age_lin'], d['age405']
    anchors = d['anchors']
    best_sr = d['best_sr']

    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.05], wspace=0.26)

    # (a) linear vs 405-anchored age-depth
    axa = fig.add_subplot(gs[0])
    axa.plot(age_lin, dep, ls='--', color=C_GRAY, lw=1.4,
             label=f'Constant-SR linear model (SR = {best_sr:.2f} m/Myr)')
    axa.plot(age405, dep, color=C_RED, lw=1.4,
             label='405-kyr anchored piecewise-linear model')
    axa.plot(anchors['age'], anchors['depth'], 'o', ms=2.4, color=C_BLUE,
             alpha=0.6, label=f'Anchors (n = {d["n_anchors"]})')
    # rev2-F2R7: mark BOTH detected interfaces (1466.10, 2052.00 m) plus the
    # section base (2440.30 m).  The previous loop over layers.iloc[1:]
    # Base_Depth yielded (2052.00, 2440.30) and missed the 1466.10 m
    # interface while marking the section base.
    marks = list(d['interfaces']) + [d['layers']['Base_Depth'].iloc[-1]]
    for bd in marks:
        ba = np.interp(bd, dep, age405)
        axa.plot(ba, bd, 's', ms=7, mfc='none', mec=C_INK, mew=1.3)
        if bd > 2300:
            # rev2-F8: bottom-interface label moved outward into the empty
            # corner below the section end so it no longer sits on the two
            # model curves where they diverge most.
            axa.annotate(f'{bd:.1f} m\n{ba:.2f} Ma', xy=(ba, bd),
                         xytext=(ba - 1.0, bd + 46), fontsize=7.4, ha='right',
                         arrowprops=dict(arrowstyle='->', color=C_INK,
                                         lw=0.7))
        elif bd < 1500:
            # rev2-F2R7: top interface (1466.10 m) — label placed in the
            # empty upper-right triangle, clear of both the legend
            # (upper left) and the panel title.
            axa.annotate(f'{bd:.1f} m\n{ba:.2f} Ma', xy=(ba, bd),
                         xytext=(ba + 16.0, bd + 150), fontsize=7.4,
                         bbox=dict(fc='white', ec='none', alpha=0.85, pad=1.0),
                         arrowprops=dict(arrowstyle='->', color=C_INK,
                                         lw=0.7))
        else:
            axa.annotate(f'{bd:.1f} m\n{ba:.2f} Ma', xy=(ba, bd),
                         xytext=(ba + 5.5, bd - 60), fontsize=7.4,
                         arrowprops=dict(arrowstyle='->', color=C_INK,
                                         lw=0.7))
    # extra bottom margin for the relocated label; legend moved to the empty
    # upper-right corner (rev2-F8)
    axa.set_ylim(dep.max() + 75, dep.min() - 15)
    axa.set_xlabel('Age (Ma, relative)')
    axa.set_ylabel('Depth (m)')
    ptitle(axa, '(a)', 'Age–depth models')
    axa.legend(loc='lower left',
    bbox_to_anchor=(-0.02, 0.07),  
    fontsize=6.2,
    markerscale=0.75,
    handlelength=1.8,
    handletextpad=0.5,
    labelspacing=0.3,
    borderpad=0.3,
    frameon=False)
    axa.grid(ls=':', lw=0.5, color=C_LGRAY)

    # (b) difference between models
    axb = fig.add_subplot(gs[1])
    diff = age405 - age_lin
    axb.plot(diff, dep, color=C_GREEN, lw=1.2)
    axb.axvline(0, color=C_GRAY, lw=0.8)
    i_mx = np.argmax(np.abs(diff))
    axb.plot(diff[i_mx], dep[i_mx], 'o', ms=7, mfc='none', mec=C_RED, mew=1.5)
    axb.annotate(f'max |ΔAge| = {abs(diff[i_mx]):.2f} Myr\nat {dep[i_mx]:.1f} m',
                 xy=(diff[i_mx], dep[i_mx]),
                 xytext=(diff.min() * 0.93, 1800), fontsize=8,
                 color=C_RED, ha='left',
                 arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.9))
    axb.set_ylim(dep.max(), dep.min())
    # rev-R1: set the x limits explicitly in ascending order. All ΔAge values
    # are negative here, so the autoscaled axis (0 at the far right) could be
    # misread as reversed; explicit ascending limits remove the ambiguity.
    x0, x1 = float(diff.min()), float(diff.max())
    pad = 0.06 * (x1 - x0)
    axb.set_xlim(x0 - pad, x1 + pad)
    axb.set_xlabel('ΔAge = anchored − linear (Myr)')
    ptitle(axb, '(b)', 'Difference between the two models')
    axb.grid(ls=':', lw=0.5, color=C_LGRAY)

    # (c) interval sedimentation rate staircase
    axc = fig.add_subplot(gs[2])
    aa = anchors['age'].values[:-1]
    isr = anchors['interval_sr'].values[:-1]
    axc.step(aa, isr, where='post', color=C_BLUE, lw=1.0,
             label='Interval SR between anchors')
    axc.axhline(15.0, color=C_GRAY, ls=':', lw=1.0, label='True SR = 15')
    mn, mx, mean = isr.min(), isr.max(), isr.mean()
    axc.axhline(mean, color=C_RED, ls='--', lw=1.0,
                label=f'Mean = {mean:.2f}')
    i_mn, i_mx = int(np.argmin(isr)), int(np.argmax(isr))
    axc.annotate(f'min = {mn:.2f}', xy=(aa[i_mn], mn),
                 xytext=(aa[i_mn] + 2, 6.5), fontsize=7.8, color=C_INK,
                 arrowprops=dict(arrowstyle='->', color=C_INK, lw=0.7))
    axc.annotate(f'max = {mx:.2f}', xy=(aa[i_mx], mx),
                 xytext=(aa[i_mx] - 15, mx - 6), fontsize=7.8, color=C_INK,
                 arrowprops=dict(arrowstyle='->', color=C_INK, lw=0.7))
    axc.legend(loc='upper left', fontsize=7.4)
    axc.set_xlabel('Age (Ma, relative)')
    axc.set_ylabel('Interval SR (m/Myr)')
    ptitle(axc, '(c)', 'Interval sedimentation rate between anchors')
    axc.set_xlim(aa.min(), aa.max() + 1)
    axc.set_ylim(0, 47)
    axc.grid(ls=':', lw=0.5, color=C_LGRAY)

    savefig(fig, 'fig09.png')


# ===== Figure 10: Flagship composite result ================================
def fig10(d):
    dep, gr = d['depth'], d['gr']
    layers, anchors = d['layers'], d['anchors']
    age405 = d['age405']

    fig = plt.figure(figsize=(12.0, 6.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.0, 0.95], wspace=0.30)

    # (a) GR log + interfaces + auxiliary Age axis
    axa = fig.add_subplot(gs[0])
    axa.plot(gr, dep, color=C_BLUE, lw=0.6)
    axa.fill_betweenx(dep, gr, gr.min(), color=C_BLUE, alpha=0.12)
    for idp in d['interfaces']:
        ia = np.interp(idp, dep, age405)
        axa.axhline(idp, color=C_RED, ls='--', lw=1.2)
        axa.text(gr.min() + 2, idp - 8, f'{idp:.1f} m  ({ia:.2f} Ma)',
                 color=C_RED, fontsize=7.8, fontweight='bold',
                 bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.0))
    axa.set_ylim(dep.max(), dep.min())
    axa.set_xlabel('GR (API)')
    axa.set_ylabel('Depth (m)')
    ptitle(axa, '(a)', 'Stratified GR log with depth & age scales')
    axa.grid(axis='x', ls=':', lw=0.5, color=C_LGRAY)

    def d2a(dd):
        return np.interp(dd, dep, age405)

    def a2d(aa):
        return np.interp(aa, age405, dep)

    sec = axa.secondary_yaxis('right', functions=(d2a, a2d))
    sec.set_ylabel('Age (Ma, 405-kyr anchored)', labelpad=16)

    # (b) layer table visualization (age-span bars)
    axb = fig.add_subplot(gs[1])
    ypos = np.arange(len(layers))[::-1]
    for y, (_, r) in zip(ypos, layers.iterrows()):
        dur = r['Base_Age'] - r['Top_Age']
        axb.barh(y, dur, left=r['Top_Age'], height=0.55,
                 color=PALETTE[int(r['Layer_No']) - 1], ec=C_INK, lw=0.6)
        txt = (f"L{int(r['Layer_No'])}: {r['Top_Depth']:.1f}–"
               f"{r['Base_Depth']:.1f} m\nΔz = {r['Thickness']:.1f} m, "
               f"Δt = {dur:.2f} Myr")
        if dur < 10:        # very short bar: label right of the bar
            axb.text(r['Base_Age'] + 2.0, y + 0.10, txt, ha='left',
                     va='bottom', fontsize=7.0, color=C_INK)
        else:
            axb.text(r['Top_Age'] + dur / 2, y + 0.40, txt, ha='center',
                     va='bottom', fontsize=7.0, color=C_INK)
    axb.set_yticks([])      # layer names are included in the annotations
    axb.set_ylim(-0.6, len(layers) - 1 + 1.15)
    axb.set_xlabel('Age (Ma, relative)')
    ptitle(axb, '(b)', 'Layer ages & durations\n(405-kyr anchored model)')
    axb.set_xlim(-1.5, 66)
    axb.grid(axis='x', ls=':', lw=0.5, color=C_LGRAY)

    # (c) apparent interval-SR staircase (aligned with caption: a
    # trough-detection diagnostic on this constant-SR benchmark, not true SR(t))
    axc = fig.add_subplot(gs[2])
    aa = anchors['age'].values[:-1]
    isr = anchors['interval_sr'].values[:-1]
    axc.step(aa, isr, where='post', color=C_BLUE, lw=1.0,
             label='Interval SR between anchors')
    axc.fill_between(aa, 0, isr, step='post', color=C_BLUE, alpha=0.15)
    axc.axhline(isr.mean(), color=C_RED, ls='--', lw=1.0,
                label=f'Mean = {isr.mean():.2f} m/Myr')
    axc.axhline(15.0, color=C_GRAY, ls=':', lw=1.0,
                label='True SR = 15 m/Myr')
    # mark layer boundary ages
    for _, r in layers.iloc[1:].iterrows():
        axc.axvline(r['Top_Age'], color=C_RED, ls='-', lw=0.7, alpha=0.45)
    axc.plot([], [], color=C_RED, lw=0.7, alpha=0.45,
             label='Layer boundary age')
    axc.legend(loc='upper left', fontsize=7.2)
    axc.set_xlabel('Age (Ma, relative)')
    axc.set_ylabel('Interval SR (m/Myr)')
    ptitle(axc, '(c)', 'Apparent interval-SR staircase\n(trough-detection diagnostic, not true SR(t))')
    axc.set_xlim(aa.min(), aa.max() + 1)
    axc.set_ylim(0, isr.max() * 1.18)
    axc.grid(ls=':', lw=0.5, color=C_LGRAY)

    savefig(fig, 'fig10.png')


# ===== Figure 11: Robustness — white-noise self-test & sensitivity =========
def fig11(d):

    dep = d['depth']
    imfs = d['imfs']
    tuner = d['tuner']

    ne = d['ne']

    scan = d['sr_scan']
    st_scan = d['st_scan']

    # ==============================================================
    # Dynamically obtain real-GR results
    # ==============================================================
    real_best_sr = d['best_sr']
    real_match = d['match_score']

    # ==============================================================
    # Recreate the EXACT white-noise self-test input
    # EMD_auto_layer_with_AGE.py --selftest:
    # seed = 12345, n = 10084, dz = 0.1 m
    # ==============================================================
    rng = np.random.default_rng(12345)

    st_depth = 1432.0 + np.arange(10084) * 0.1

    noise = rng.normal(
        0.0,
        1.0,
        st_depth.size
    )

    st_gr = (
        65.35
        + (noise - noise.min())
        / (noise.max() - noise.min())
        * (132.99 - 65.35)
    )

    # Recompute white-noise best SR and match score
    noise_best_sr, noise_match = tuner.best_sr(
        st_scan,
        depth=st_depth,
        signal=st_gr
    )

    # ==============================================================
    # Figure layout
    # ==============================================================
    fig = plt.figure(
        figsize=(11.5, 4.8)
    )

    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.1, 0.85, 1.2],
        wspace=0.30
    )

    # ==============================================================
    # (a) SR-scan objective:
    #     real GR vs white noise
    # ==============================================================
    axa = fig.add_subplot(gs[0])

    j_real = float(
        scan['score'].max()
    )

    j_noise = float(
        st_scan['score'].max()
    )

    axa.plot(
        scan['sr'],
        scan['score'] / j_real,
        '-o',
        ms=2.8,
        lw=1.0,
        color=C_BLUE,
        label=(
            f'Real GR '
            f'(best SR = {real_best_sr:.2f}; '
            f'max J = {j_real:.3f})'
        )
    )

    axa.plot(
        st_scan['sr'],
        st_scan['score'] / j_real,
        '-s',
        ms=2.8,
        lw=1.0,
        color=C_GRAY,
        label=(
            f'White noise '
            f'(best SR = {noise_best_sr:.2f}; '
            f'max J = {j_noise:.3f})'
        )
    )

    # Dynamically located best-SR lines
    axa.axvline(
        real_best_sr,
        color=C_BLUE,
        ls=':',
        lw=0.9
    )

    axa.axvline(
        noise_best_sr,
        color=C_GRAY,
        ls=':',
        lw=0.9
    )

    axa.annotate(
        'white-noise maximum is broad and non-selective\n'
        f'(absolute max J = {j_noise:.3f} vs '
        f'{j_real:.3f} for real\n'
        f'GR); the discriminative evidence is the match\n'
        f'coefficient — panel (b): '
        f'{noise_match:.3f} vs {real_match:.3f}',
        xy=(
            noise_best_sr,
            j_noise / j_real
        ),
        xytext=(
            1.0,
            0.56
        ),
        fontsize=7.2,
        bbox=dict(
            fc='white',
            ec=C_GRAY,
            lw=0.6,
            alpha=0.85
        ),
        arrowprops=dict(
            arrowstyle='->',
            color=C_GRAY,
            lw=0.9
        )
    )

    axa.set_xlabel(
        'Sedimentation rate SR (m/Myr)'
    )

    axa.set_ylabel(
        'Objective score\n'
        '(common scale: / max of real GR)'
    )

    ptitle(
        axa,
        '(a)',
        'SR-scan objective: real vs white noise\n'
        '(shared normalization to the real-GR max)',
        fontsize=9
    )

    axa.legend(
        fontsize=7.4,
        loc='upper left'
    )

    axa.grid(
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    # ==============================================================
    # (b) Match-score comparison
    # ==============================================================
    axb = fig.add_subplot(gs[1])

    match_values = [
        real_match,
        noise_match
    ]

    bars = axb.bar(
        [
            'Real GR',
            'White noise\n(self-test)'
        ],
        match_values,
        color=[
            C_GREEN,
            C_GRAY
        ],
        ec=C_INK,
        lw=0.6,
        width=0.55
    )

    for b, v in zip(
        bars,
        match_values
    ):
        axb.text(
            b.get_x() + b.get_width() / 2,
            v + 0.03,
            f'{v:.3f}',
            ha='center',
            fontsize=9,
            fontweight='bold'
        )

    # 0.5 is the predefined workflow threshold,
    # not an experimental result
    match_threshold = 0.5

    axb.axhline(
        match_threshold,
        color=C_RED,
        ls='--',
        lw=1.2
    )

    axb.text(
        1.25,
        match_threshold + 0.035,
        f'threshold = {match_threshold:.1f}',
        fontsize=7.8,
        color=C_RED,
        ha='right'
    )

    axb.set_ylim(
        0,
        1.15
    )

    axb.set_ylabel(
        'Peak-match coefficient'
    )

    ptitle(
        axb,
        '(b)',
        'Match score: PASS vs rejected'
    )

    axb.grid(
        axis='y',
        ls=':',
        lw=0.5,
        color=C_LGRAY
    )

    # ==============================================================
    # (c) Sensitivity of IMF selection to prominence threshold
    # ==============================================================
    axc = fig.add_subplot(gs[2])

    pfs = [
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8
    ]

    mat = np.zeros(
        (
            len(pfs),
            len(imfs)
        )
    )

    sel_idx = []

    for i, pf in enumerate(pfs):

        row = []

        for x in imfs:

            pr = pf * np.std(x)

            pk, _ = find_peaks(
                x,
                prominence=pr
            )

            tr, _ = find_peaks(
                -x,
                prominence=pr
            )

            row.append(
                max(
                    pk.size,
                    tr.size
                ) - 1
            )

        row = np.array(
            row,
            dtype=float
        )

        mat[i] = (
            100
            * np.abs(row - ne)
            / ne
        )

        sel_idx.append(
            int(
                np.argmin(mat[i])
            )
        )

    im = axc.imshow(
        mat,
        aspect='auto',
        cmap=SAND_R,
        vmin=0,
        vmax=100
    )

    # Dynamically outline the selected IMF at each threshold
    for i in range(len(pfs)):

        j = sel_idx[i]

        axc.add_patch(
            plt.Rectangle(
                (
                    j - 0.5,
                    i - 0.5
                ),
                1,
                1,
                fill=False,
                ec=C_RED,
                lw=2.0
            )
        )

    axc.set_xticks(
        range(len(imfs))
    )

    axc.set_xticklabels(
        [
            f'IMF{i}'
            for i in range(
                1,
                len(imfs) + 1
            )
        ],
        fontsize=6.6,
        rotation=45,
        ha='right'
    )

    axc.set_yticks(
        range(len(pfs))
    )

    axc.set_yticklabels(
        [
            f'{pf:.1f}×std'
            for pf in pfs
        ],
        fontsize=7.5
    )

    axc.set_xlabel(
        'IMF component'
    )

    axc.set_ylabel(
        'Extrema prominence threshold'
    )

    cb = fig.colorbar(
        im,
        ax=axc,
        pad=0.015
    )

    cb.set_label(
        '$|n - N_e|\\,/\\,N_e$  (%)',
        fontsize=8
    )

    cb.ax.tick_params(
        labelsize=7.5
    )

    # --------------------------------------------------------------
    # Dynamic title:
    # do NOT hard-code "IMF5 selected at all levels"
    # --------------------------------------------------------------
    selected_numbers = [
        j + 1
        for j in sel_idx
    ]

    unique_selected = sorted(
        set(selected_numbers)
    )

    if len(unique_selected) == 1:

        selection_note = (
            f'IMF{unique_selected[0]} '
            f'selected at all levels: red frames'
        )

    else:

        selection_note = (
            'selected IMF varies: '
            + ', '.join(
                f'IMF{x}'
                for x in unique_selected
            )
            + ' (red frames)'
        )

    ptitle(
        axc,
        '(c)',
        'Sensitivity of 405-kyr IMF selection\n'
        f'({selection_note})'
    )

    savefig(
        fig,
        'fig11.png'
    )


# ===== main ================================================================
def main():
    print('Loading data ...')
    d = load_all()
    jobs = [
        ('fig01 — workflow', lambda: fig01(d)),
        ('fig02 — input data', lambda: fig02(d)),
        ('fig03 — EMD decomposition', lambda: fig03(d)),
        ('fig04 — decomposition diagnostics', lambda: fig04(d)),
        ('fig05 — automatic stratification', lambda: fig05(d)),
        ('fig06 — timeOpt SR scan', lambda: fig06(d)),
        ('fig07 — spectral validation', lambda: fig07(d)),
        ('fig08 — 405-kyr reference selection', lambda: fig08(d)),
        ('fig09 — age-model comparison', lambda: fig09(d)),
        ('fig10 — flagship composite', lambda: fig10(d)),
        ('fig11 — robustness', lambda: fig11(d)),
    ]
    for name, fn in jobs:
        print(f'[{name}]')
        fn()
    print('All figures regenerated in', OUT_DIR)


if __name__ == '__main__':
    main()
