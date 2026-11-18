# -*- coding: utf-8 -*-
"""
exp_common.py — shared utilities for the ESR numerical experiment matrix
=========================================================================
Shared by exp_a_detectability.py / exp_b_variable_sr.py / exp_c_mode_splitting.py.

Contents
--------
- synth_series(...) : common synthetic stratigraphic series generator.
      1) build t(d) = integral of dd'/SR(d') (cumulative integral) on a
         0.1 m depth grid;
      2) generate the astronomical periodic sum (405/95/40/21 kyr) plus
         AR(1)+white noise on a fine uniform TIME grid;
      3) map time -> depth via t(d) and resample onto the 0.1 m grid.
- band_significances(...) : MTM peak/neighbourhood-median significance per
  target band (same recipe as AstroTuner._band_diagnostics, threshold 5).
- Figure style identical to ../make_paper_figures.py (low-saturation
  palette, (a)(b)(c) panel tags, 300 dpi).

Units: depth m; SR m/Myr; time Myr; frequency cycles/Myr.
All generators take an explicit integer seed (fixed seeds throughout).
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.signal import lfilter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = '/mnt/agents/output/emd_astro'
EXP_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(os.path.dirname(EXP_DIR), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)
sys.path.insert(0, BASE)
from astro_py_tuner import AstroTuner  # noqa: E402

# ----------------------------------------------------------------------
# Target astronomical model
# ----------------------------------------------------------------------
PERIODS_KYR = (405.0, 95.0, 40.0, 21.0)
AMPS = (6.0, 4.0, 2.5, 1.5)
DZ = 0.1                      # m, uniform depth sampling
SIG_RMS = float(np.sqrt(sum(a * a for a in AMPS) / 2.0))   # 5.5

# ----------------------------------------------------------------------
# Figure style — copied from make_paper_figures.py for visual consistency
# ----------------------------------------------------------------------
C_INK = '#2F3B46'
C_BLUE = '#41627E'
C_RED = '#A65A4A'
C_GREEN = '#6E8B67'
C_TAN = '#B08C57'
C_GRAY = '#8A8F94'
C_LGRAY = '#C9CDD1'
PALETTE = [C_BLUE, C_RED, C_GREEN, C_TAN, C_GRAY, '#7E8B9B']
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


def ptitle(ax, tag, title, fontsize=10, pad=6):
    """Unified panel title: (a)(b)(c) tag + left-aligned title (same line)."""
    ax.set_title(f'{tag}  {title}', loc='left', fontsize=fontsize, pad=pad)


def savefig(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f'  saved {path}  ({os.path.getsize(path)/1024.0:.0f} KiB)')


# ----------------------------------------------------------------------
# Common synthetic series generator
# ----------------------------------------------------------------------
def synth_series(sr_fn, length_m=600.0, dz=DZ, noise_ratio=0.5, seed=42,
                 rho=0.7, periods=PERIODS_KYR, amps=AMPS, dt_myr=5e-4):
    """
    Generate a synthetic stratigraphic proxy series.

    Parameters
    ----------
    sr_fn : callable depth(m) -> SR (m/Myr), vectorised. Constant SR:
            ``sr_fn = lambda d: np.full_like(d, 15.0)``.
    length_m : segment length (m); depth grid = 0 .. length_m step dz.
    noise_ratio : total noise std / RMS of the periodic sum (0.2/0.5/0.8).
    rho : AR(1) coefficient of the red-noise component.
    dt_myr : fine time-domain grid step (0.5 kyr) used before remapping.

    Returns
    -------
    dict(depth, signal, t_true, sr_true)
        depth  : 0.1 m grid (m)
        signal : proxy value on the depth grid
        t_true : true age (Myr) on the depth grid, t(0)=0
        sr_true: SR (m/Myr) on the depth grid
    """
    rng = np.random.default_rng(int(seed))
    depth = np.arange(0.0, length_m + 0.5 * dz, dz)
    sr = np.asarray(sr_fn(depth), dtype=float)
    assert np.all(sr > 0), 'SR must be positive'
    # depth -> time by cumulative integration: t(d) = ∫ dd'/SR(d')
    t_true = np.concatenate(([0.0], cumulative_trapezoid(1.0 / sr, depth)))

    # --- time-domain generation on a fine uniform grid ---
    t_fine = np.arange(0.0, t_true[-1] + dt_myr, dt_myr)
    x_t = np.zeros_like(t_fine)
    for p, a in zip(periods, amps):
        x_t += a * np.sin(2.0 * np.pi * t_fine / (p / 1000.0))
    # AR(1) red noise (stationary, unit-std) + white noise (unit-std)
    w = rng.normal(0.0, 1.0, t_fine.size)
    ar1 = lfilter([1.0], [1.0, -rho], w)
    ar1 /= ar1.std()
    white = rng.normal(0.0, 1.0, t_fine.size)
    noise_std = noise_ratio * SIG_RMS
    x_t += noise_std * (ar1 + white) / np.sqrt(2.0)

    # --- map time -> depth: interpolate onto t_true of the depth grid ---
    signal = np.interp(t_true, t_fine, x_t)
    return dict(depth=depth, signal=signal, t_true=t_true, sr_true=sr)


# ----------------------------------------------------------------------
# MTM band significance (peak / neighbourhood median), threshold = 5
# ----------------------------------------------------------------------
def band_significances(tuner, t, x, periods=PERIODS_KYR):
    """
    MTM spectrum at the given time axis; for each target band return
    (significance, detected). significance = band peak / neighbourhood
    median; detected = significance >= tuner.sig_thresh (5).
    """
    freq, power = tuner.mtm_spectrum(t, x)
    _, _, _, sigs = tuner._band_diagnostics(freq, power)
    sigs = np.asarray(sigs, dtype=float)
    det = (sigs >= tuner.sig_thresh).astype(int)
    return sigs, det, freq, power


def nyquist_sr(period_kyr, dz=DZ):
    """Theoretical Nyquist SR (m/Myr): lambda = P*SR/1000 >= 2*dz."""
    return 2000.0 * dz / period_kyr


def const_sr_fn(v):
    return lambda d: np.full_like(d, float(v), dtype=float)
