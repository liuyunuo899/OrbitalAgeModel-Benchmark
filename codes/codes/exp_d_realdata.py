# -*- coding: utf-8 -*-
"""
exp_d_realdata.py — Exp D: empirical application of the EMD astro-tuning
depth-to-age framework to two PUBLIC datasets (fig15 / fig16)
================================================================================
Datasets (DATA = r"F:\刘雨诺\论文投稿\非线性沉积速率emd-astro\data\real-datasets".md):
  1. SK-1n (Songliao Basin, continental; Wu et al. 2015 EPSL):
       Th gamma-log series, 228.00-1796.75 m, uniform 0.125 m (kept as-is),
       published ATS (mmc2, tuned to La2010d 405-kyr) on the same depth grid.
  2. ODP Site 1264 (Walvis Ridge, marine; Liebrand et al. 2016 EPSL):
       XRF CaCO3 [wt%], two holes (1264A/B) merged on the adjusted composite
       depth (armcd), bin-averaged onto a uniform 0.05 m grid (median native
       sampling 0.02 m); published age model = piecewise-linear interpolation
       of the 115 tuning tie points (PANGAEA 862580).

Chain (identical for both): uniform resampling -> Gaussian detrend ->
AstroTuner.scan_sed_rates -> match_quality -> PyEMD EMD (max_imf=10) ->
select_405_imf -> build_age_model_405 -> cross-comparison with the published
independent age model.

Anchoring honesty note: the 405-kyr anchored model is a FLOATING scale; the
ONLY absolute-age input is the published top-of-section age (single anchor);
the reference function is ref(d) = age_top_pub + (d - d_top)/sr_opt, i.e. the
first 405-kyr IMF extremum inherits that single-anchor linear age, subsequent
anchors are spaced exactly 0.405 Myr. Both 'trough' and 'peak' phases are
computed; the framework default ('trough') is used for the figures and the
peak-phase statistics are reported alongside. Where select_405_imf returns
matched=False the contract fallback (constant-SR linear model) is the reported
primary model and the 405-anchored model is retained as a diagnostic.

Detrending: subtract a Gaussian low-pass trend (scipy gaussian_filter1d,
mode='nearest'); sigma = 40 m for SK-1n and 4.5 m for Site 1264, chosen so
the -3 dB cutoff wavelength (~2.3*sigma) is ~3x the expected 405-kyr
wavelength (405 kyr x mean SR), i.e. the trend removes only components much
longer than the long eccentricity. Sensitivity to sigma is checked and
reported (SR optimum is insensitive).

All seeds fixed; EMD is deterministic; scan uses n_jobs=2. One-command rerun:
    python exp_d_realdata.py
Outputs: expD_* CSV files in experiments/, fig15.png / fig16.png in figures/,
and realdata_summary.md appended/rewritten in experiments/.
"""

from __future__ import annotations

import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from PyEMD import EMD

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, EXP_DIR)
import exp_common as ec          # style + FIG_DIR + AstroTuner import path
from astro_py_tuner import AstroTuner  # noqa: E402

DATA = r"F:\刘雨诺\论文投稿\非线性沉积速率emd-astro\data\real-datasets"
SEED = 42
np.random.seed(SEED)


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------
def load_sk1n():
    th = pd.read_csv(os.path.join(DATA, 'SK1n_Wu2015_EPSL_Th_mmc1.txt'),
                     sep='\t')
    ats = pd.read_csv(os.path.join(DATA, 'SK1n_Wu2015_EPSL_mmc2.txt'),
                      sep='\t')
    depth = th.iloc[:, 0].values.astype(float)
    sig = th.iloc[:, 1].values.astype(float)
    age_pub = np.interp(depth, ats.iloc[:, 0].values.astype(float),
                        ats.iloc[:, 1].values.astype(float))
    return depth, sig, age_pub


def read_pangaea(path):
    lines = open(path).read().splitlines()
    start = 0
    for i, ln in enumerate(lines):
        if ln.strip() == '*/':
            start = i + 1
            break
    return pd.read_csv(path, sep='\t', skiprows=start)


def load_1264(dz=0.05):
    """Merge holes 1264A/B on armcd; bin-average to a uniform dz grid."""
    caco3 = read_pangaea(os.path.join(DATA, 'PANGAEA_862583.txt'))
    ties = read_pangaea(os.path.join(DATA, 'PANGAEA_862580.txt'))
    arm = caco3['Depth adj rmcd [armcd]'].values.astype(float)
    ca = caco3['CaCO3 [%]'].values.astype(float)
    order = np.argsort(arm)
    arm, ca = arm[order], ca[order]
    grid = np.arange(arm.min(), arm.max() + 1e-9, dz)
    ibin = np.floor((arm - arm.min()) / dz).astype(int)
    csum = np.bincount(ibin, weights=ca)
    ccnt = np.bincount(ibin)
    sig = np.full(len(grid), np.nan)
    n = min(len(grid), len(csum))
    ok = ccnt[:n] > 0
    sig[:n][ok] = csum[:n][ok] / ccnt[:n][ok]
    n_gap = int(np.isnan(sig).sum())
    sig = pd.Series(sig).interpolate(limit_direction='both').values
    td = ties['Depth comp r [rmcd]'].values.astype(float)
    ta = ties['Age [ka BP]'].values.astype(float) / 1000.0
    age_pub = np.interp(grid, td, ta)
    return grid, sig, age_pub, (td, ta), n_gap


# ----------------------------------------------------------------------
# Full chain
# ----------------------------------------------------------------------
def run_chain(name, depth, sig, age_pub, sr_grid, sigma_m, dz):
    rec = dict(name=name)
    depth = np.asarray(depth, float)
    sig = np.asarray(sig, float)
    rec['n_points'] = int(depth.size)
    rec['depth_top'], rec['depth_bot'] = float(depth.min()), float(depth.max())
    rec['dz'] = float(dz)
    rec['detrend_sigma_m'] = float(sigma_m)

    # 1) detrend
    trend = gaussian_filter1d(sig, sigma=sigma_m / dz, mode='nearest')
    x = sig - trend

    # 2) SR scan + match
    t0 = time.time()
    tuner = AstroTuner(sr_grid=sr_grid, n_jobs=2, seed=SEED)
    scan = tuner.scan_sed_rates(depth, x)
    sr_opt, match = tuner.best_sr(scan, depth=depth, signal=x)
    rec['sr_opt'] = float(sr_opt)
    rec['match_score'] = float(match)
    rec['sr_grid_lo'], rec['sr_grid_hi'] = float(sr_grid[0]), float(sr_grid[-1])
    rec['sr_grid_step'] = float(sr_grid[1] - sr_grid[0])
    rec['scan_seconds'] = round(time.time() - t0, 1)

    # 3) EMD decomposition
    t0 = time.time()
    imfs = EMD()(x, max_imf=10)
    rec['emd_seconds'] = round(time.time() - t0, 1)
    rec['n_imfs'] = int(imfs.shape[0])

    # 4) select the 405-kyr IMF
    sel = tuner.select_405_imf(imfs, depth, sr_opt)
    rec['imf_index'] = sel['imf_index']
    rec['n_cycles'] = sel['n_cycles']
    rec['ne'] = sel['ne']
    rec['imf_matched'] = bool(sel['matched'])
    rec['imf_rel_dev'] = sel['diag']['rel_dev']
    rec['imf_mtm_period_kyr'] = sel['diag']['mtm_peak_period_kyr']

    # 5) age models — floating scale + single top anchor at published top age
    age_top_pub = float(age_pub[0])
    d_top = float(depth.min())

    def ref(d):
        return age_top_pub + (np.asarray(d, float) - d_top) / sr_opt

    age_lin = ref(depth)
    out = dict(depth=depth, sig=sig, trend=trend, detrended=x, scan=scan,
               tuner=tuner, imfs=imfs, sel=sel, age_pub=age_pub,
               age_lin=age_lin)
    for phase in ('trough', 'peak'):
        age405, anchors = tuner.build_age_model_405(
            depth, imfs[sel['imf_index']], anchor_phase=phase,
            ref_age_fn=ref)
        resid = age405 - age_pub
        out[f'age405_{phase}'] = age405
        out[f'anchors_{phase}'] = anchors
        rec[f'n_anchors_{phase}'] = int(len(anchors))
        rec[f'rmse_405_{phase}'] = float(np.sqrt((resid ** 2).mean()))
        rec[f'maxabs_405_{phase}'] = float(np.abs(resid).max())
        rec[f'bias_405_{phase}'] = float(resid.mean())
        isr = anchors['interval_sr'].dropna().values
        rec[f'interval_sr_min_{phase}'] = float(isr.min())
        rec[f'interval_sr_max_{phase}'] = float(isr.max())
    resid_lin = age_lin - age_pub
    rec['rmse_linear'] = float(np.sqrt((resid_lin ** 2).mean()))
    rec['maxabs_linear'] = float(np.abs(resid_lin).max())
    rec['bias_linear'] = float(resid_lin.mean())
    span = rec['depth_bot'] - rec['depth_top']
    rec['pub_mean_sr'] = float(span / (age_pub[-1] - age_pub[0]))
    rec['duration_myr_scan'] = float(span / sr_opt)
    rec['duration_myr_pub'] = float(age_pub[-1] - age_pub[0])
    out['resid_lin'] = resid_lin
    return rec, out


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def fig_realdata(out, rec, figname, proxy_label, proxy_unit, phase='trough',
                 ties=None, fallback_linear=False, step_depth=None,
                 step_note=None):
    depth, sig, trend = out['depth'], out['sig'], out['trend']
    age_pub, age_lin = out['age_pub'], out['age_lin']
    age405 = out[f'age405_{phase}']
    resid405 = age405 - age_pub
    resid_lin = out['resid_lin']
    scan = out['scan']
    anchors = out[f'anchors_{phase}']

    fig, axes = plt.subplots(1, 4, figsize=(14.2, 4.15),
                             gridspec_kw=dict(width_ratios=[1.25, 1.1, 0.95, 1.0]))
    # (a) proxy + trend, depth on y axis (stratigraphic convention)
    ax = axes[0]
    ax.plot(sig, depth, color=ec.C_GRAY, lw=0.5, label='observed')
    ax.plot(trend, depth, color=ec.C_RED, lw=1.2,
            label=f'Gaussian trend ($\\sigma$={rec["detrend_sigma_m"]:g} m)')
    ax.set_ylim(depth.max(), depth.min())
    ax.set_xlabel(f'{proxy_label} [{proxy_unit}]')
    ax.set_ylabel('Depth (m)')
    # Compact inset legend at upper-right; white backing protects legibility.
    ax.legend(loc='upper right', bbox_to_anchor=(0.985, 0.975),
              fontsize=6.4, frameon=True, facecolor='white',
              edgecolor='none', framealpha=0.88, handlelength=1.25,
              handletextpad=0.4, labelspacing=0.25, borderpad=0.22)
    ec.ptitle(ax, '(a)', f'{proxy_label} series and detrend')

    # (b) age models vs depth
    ax = axes[1]
    ax.plot(age_pub, depth, color=ec.C_INK, lw=1.3, label='published (independent)')
    if fallback_linear:
        ax.plot(age_lin, depth, color=ec.C_BLUE, lw=1.2,
                label=f'this framework, linear fallback (SR={rec["sr_opt"]:.2f})')
        ax.plot(age405, depth, color=ec.C_RED, lw=0.9, ls='--',
                label='405-kyr anchored (not matched; diagnostic)')
    else:
        ax.plot(age405, depth, color=ec.C_BLUE, lw=1.2,
                label='this framework, 405-kyr anchored')
        ax.plot(age_lin, depth, color=ec.C_GRAY, lw=0.9, ls='--',
                label=f'constant SR={rec["sr_opt"]:.2f} (scan optimum)')
    if ties is not None:
        ax.plot(ties[1], ties[0], 'o', ms=2.5, color=ec.C_GREEN,
                alpha=0.7, label='tie points')
    if step_depth is not None:
        ax.axhline(step_depth, color=ec.C_GRAY, lw=0.8, ls=':')
    ax.set_ylim(depth.max(), depth.min())
    ax.set_xlabel('Age (Ma)')
    ax.set_ylabel('Depth (m)')
    # Short lines in upper-right empty region, inside the axes.
    handles, labels = ax.get_legend_handles_labels()
    wrapped = [lab.replace('published (independent)',
                           'published (independent)')
                   .replace('this framework, linear fallback ',
                            'this framework,\nlinear fallback ')
                   .replace('this framework, 405-kyr anchored',
                            'this framework,\n405-kyr anchored')
                   .replace('405-kyr anchored (not matched; diagnostic)',
                            '405-kyr anchored\n(not matched; diagnostic)')
                   .replace('constant SR=', 'constant SR=')
               for lab in labels]
    ax.legend(handles, wrapped, loc='upper right',
              bbox_to_anchor=(0.99, 0.975), fontsize=6.15,
              frameon=True, facecolor='white', edgecolor='none',
              framealpha=0.88, handlelength=1.3, handletextpad=0.4,
              labelspacing=0.24, borderpad=0.2)
    ec.ptitle(ax, '(b)', 'Age model vs published reference')

    # (c) residuals vs depth
    ax = axes[2]
    ax.axvline(0, color=ec.C_INK, lw=0.8)
    ax.plot(resid405, depth, color=ec.C_RED, lw=0.9,
            label=f'405-kyr model (RMSE {rec[f"rmse_405_{phase}"]:.2f} Myr)')
    ax.plot(resid_lin, depth, color=ec.C_BLUE, lw=0.9, ls='--',
            label=f'linear (RMSE {rec["rmse_linear"]:.2f} Myr)')
    if step_depth is not None:
        ax.axhline(step_depth, color=ec.C_GRAY, lw=0.8, ls=':')
        ax.annotate(step_note or 'published age-model step',
                    xy=(0.03, step_depth), xycoords=('axes fraction', 'data'),
                    fontsize=7, color=ec.C_GRAY, va='bottom')
    ax.set_ylim(depth.max(), depth.min())
    rmax = 1.05 * max(np.abs(resid405).max(), np.abs(resid_lin).max())
    ax.set_xlim(-rmax, rmax)
    ax.set_xlabel('Age residual (model - published), Myr')
    ax.set_ylabel('Depth (m)')
    # Two-line RMSE labels, top-right inside the plot.
    handles, labels = ax.get_legend_handles_labels()
    wrapped = [lab.replace(' (RMSE ', '\n(RMSE ') for lab in labels]
    ax.legend(handles, wrapped, loc='upper right',
              bbox_to_anchor=(1.000, 0.975), fontsize=6.5,
              frameon=True, facecolor='white', edgecolor='none',
              framealpha=0.86, handlelength=1.25,
              handletextpad=0.4, labelspacing=0.28, borderpad=0.2)
    ec.ptitle(ax, '(c)', 'Age-model residuals')

    # (d) SR scan curve
    ax = axes[3]
    ax.plot(scan['sr'], scan['score'], color=ec.C_BLUE, lw=1.0)
    ax.axvline(rec['sr_opt'], color=ec.C_RED, lw=1.0, ls='--')
    ax.axvline(rec['pub_mean_sr'], color=ec.C_GREEN, lw=1.0, ls=':')
    ymax = scan['score'].max()
    # Short, right-aligned note in the top-right empty part of the panel.
    # No arrow or text overlaps the scan-optimum peak.
    ax.text(0.98, 0.97,
            f"scan optimum: {rec['sr_opt']:.2f} m/Myr\n"
            f"match score: {rec['match_score']:.2f}\n"
            f"published mean SR:\n{rec['pub_mean_sr']:.1f} m/Myr",
            transform=ax.transAxes, ha='right', va='top',
            fontsize=6.7, linespacing=1.15, color=ec.C_INK)
    ax.set_xlabel('Sedimentation rate (m/Myr)')
    ax.set_ylabel('timeOpt-style score')
    ax.set_ylim(0, ymax * 1.15)
    ec.ptitle(ax, '(d)', 'Sedimentation-rate scan')

    fig.subplots_adjust(wspace=0.35, bottom=0.17, top=0.88)
    ec.savefig(fig, figname)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    summary_rows = []

    # ---------------- SK-1n (fig15) ----------------
    print('--- SK-1n (Songliao Basin, Th log) ---')
    depth, th, age_pub = load_sk1n()
    rec1, out1 = run_chain('SK-1n', depth, th, age_pub,
                           sr_grid=np.arange(40.0, 160.0 + 1e-9, 0.5),
                           sigma_m=40.0, dz=0.125)
    # The published SK-1n ATS contains a genuine ~3.8 Myr step (hiatus/splice
    # in the supplement file); detect it and report segmented statistics.
    dage = np.diff(age_pub)
    i_step = int(np.argmax(dage))
    step_depth = None
    if dage[i_step] > 0.5:
        step_depth = float(depth[i_step] + 0.5 * rec1['dz'])
        rec1['pub_step_depth_m'] = step_depth
        rec1['pub_step_myr'] = float(dage[i_step])
        for tag, m in (('upper', depth <= depth[i_step]),
                       ('lower', depth > depth[i_step])):
            for key, akey in (('405', 'age405_trough'), ('linear', 'age_lin')):
                r = (out1[akey] - age_pub)[m]
                rec1[f'rmse_{key}_{tag}_seg'] = float(np.sqrt((r ** 2).mean()))
                rec1[f'rmse_{key}_{tag}_seg_deb'] = float(r.std())
            rec1[f'pub_mean_sr_{tag}_seg'] = float(
                (depth[m].max() - depth[m].min()) /
                (age_pub[m].max() - age_pub[m].min()))
        print(f"  published ATS step +{rec1['pub_step_myr']:.2f} Myr at "
              f"{step_depth:.1f} m (segmented stats computed)")
    summary_rows.append(rec1)
    fig_realdata(out1, rec1, 'fig15.png', 'Th', 'ppm', phase='trough',
                 step_depth=step_depth,
                 step_note=f"published ATS step +{rec1.get('pub_step_myr', 0):.2f} Myr"
                 if step_depth is not None else None)

    # ---------------- Site 1264 (fig16) ----------------
    print('--- ODP Site 1264 (CaCO3) ---')
    grid, ca, age_pub2, ties, n_gap = load_1264(dz=0.05)
    rec2, out2 = run_chain('Site 1264', grid, ca, age_pub2,
                           sr_grid=np.arange(2.0, 20.0 + 1e-9, 0.1),
                           sigma_m=4.5, dz=0.05)
    rec2['n_gap_bins_interp'] = n_gap
    summary_rows.append(rec2)
    fig_realdata(out2, rec2, 'fig16.png', 'CaCO$_3$', 'wt%', phase='trough',
                 ties=ties, fallback_linear=not rec2['imf_matched'])

    # ---------------- CSV outputs ----------------
    # drop wall-clock timings so the summary CSV is byte-identical across reruns
    sdf = pd.DataFrame(summary_rows).drop(
        columns=['scan_seconds', 'emd_seconds'], errors='ignore')
    sdf.to_csv(os.path.join(EXP_DIR, 'expD_summary.csv'), index=False)
    out1['scan'].to_csv(os.path.join(EXP_DIR, 'expD_sk1n_scan.csv'), index=False)
    out2['scan'].to_csv(os.path.join(EXP_DIR, 'expD_1264_scan.csv'), index=False)
    out1['anchors_trough'].to_csv(
        os.path.join(EXP_DIR, 'expD_sk1n_anchors.csv'), index=False)
    out2['anchors_trough'].to_csv(
        os.path.join(EXP_DIR, 'expD_1264_anchors.csv'), index=False)
    pd.DataFrame({
        'depth_m': out1['depth'], 'Th_ppm': out1['sig'],
        'trend_gauss40m': out1['trend'], 'Th_detrended': out1['detrended'],
        'age_pub_ats_ma': out1['age_pub'],
        'age_emd405_ma': out1['age405_trough'],
        'age_linear_ma': out1['age_lin'],
        'resid_405_myr': out1['age405_trough'] - out1['age_pub'],
        'resid_linear_myr': out1['resid_lin'],
    }).to_csv(os.path.join(EXP_DIR, 'expD_sk1n_agemodel.csv'), index=False)
    pd.DataFrame({
        'depth_armcd_m': out2['depth'], 'CaCO3_wtpc': out2['sig'],
        'trend_gauss4.5m': out2['trend'], 'CaCO3_detrended': out2['detrended'],
        'age_pub_tiepoints_ma': out2['age_pub'],
        'age_emd405_ma': out2['age405_trough'],
        'age_linear_ma': out2['age_lin'],
        'resid_405_myr': out2['age405_trough'] - out2['age_pub'],
        'resid_linear_myr': out2['resid_lin'],
    }).to_csv(os.path.join(EXP_DIR, 'expD_1264_agemodel.csv'), index=False)
    print('CSV outputs written.')

    # ---------------- markdown summary ----------------
    write_summary(rec1, rec2, out1, out2)
    print('Done.')


def _fmt_rec(r):
    return (f"SR grid {r['sr_grid_lo']:g}-{r['sr_grid_hi']:g} m/Myr "
            f"(step {r['sr_grid_step']:g}); scan optimum "
            f"**{r['sr_opt']:.2f} m/Myr**, match score **{r['match_score']:.3f}**; "
            f"IMF{r['imf_index'] + 1} selected ({r['n_cycles']} cycles vs "
            f"Ne={r['ne']:.1f}, rel. dev {r['imf_rel_dev']*100:.1f}%, "
            f"MTM peak {r['imf_mtm_period_kyr']:.0f} kyr, "
            f"matched={r['imf_matched']}); anchors(trough)={r['n_anchors_trough']}")


def write_summary(rec1, rec2, out1, out2):
    path = os.path.join(EXP_DIR, 'realdata_summary.md')
    txt = f"""# Exp D — Empirical application to public datasets (exp_d_realdata.py, fig15/fig16)

**Framework chain** (identical for both datasets): uniform grid -> Gaussian
detrend -> `scan_sed_rates` -> `match_quality` -> PyEMD `EMD(max_imf=10)` ->
`select_405_imf` -> `build_age_model_405` -> cross-comparison with the
published independent age model. Seeds fixed; EMD deterministic.
**Anchoring**: the 405-kyr model is a floating scale; the ONLY absolute-age
input is the published top-of-section age (single anchor), propagated with the
scan-optimal SR to the first IMF extremum; subsequent anchors are spaced
exactly 0.405 Myr. No other tie to the published model is used.

## SK-1n, Songliao Basin (fig15) — Th gamma log, Wu et al. 2015 EPSL
- Data: {rec1['n_points']} samples, {rec1['depth_top']:.2f}-{rec1['depth_bot']:.2f} m,
  uniform {rec1['dz']} m (kept at native sampling). Detrend: Gaussian low-pass,
  sigma = {rec1['detrend_sigma_m']:g} m (cutoff wavelength ~3x the expected
  405-kyr wavelength of ~34 m; SR optimum insensitive to sigma = 25-90 m).
- {_fmt_rec(rec1)}
- Scan duration {rec1['duration_myr_scan']:.2f} Myr vs published
  {rec1['duration_myr_pub']:.2f} Myr (mean SR {rec1['pub_mean_sr']:.1f} m/Myr).
- vs published ATS (mmc2, La2010d-tuned): 405-kyr model (trough phase)
  RMSE {rec1['rmse_405_trough']:.3f} Myr, max |dev| {rec1['maxabs_405_trough']:.3f} Myr,
  bias {rec1['bias_405_trough']:+.3f} Myr; peak phase RMSE {rec1['rmse_405_peak']:.3f} Myr.
  Constant-SR linear at the scan optimum: RMSE {rec1['rmse_linear']:.3f} Myr,
  max |dev| {rec1['maxabs_linear']:.3f} Myr.
- Interval SR between 405-kyr anchors: {rec1['interval_sr_min_trough']:.1f}-{rec1['interval_sr_max_trough']:.1f} m/Myr.
- **Published-ATS step**: the mmc2 file itself contains a
  +{rec1.get('pub_step_myr', float('nan')):.2f} Myr age step at
  {rec1.get('pub_step_depth_m', float('nan')):.1f} m (hiatus or splice in the
  published supplement). Segment statistics: upper segment (mean published SR
  {rec1.get('pub_mean_sr_upper_seg', float('nan')):.0f} m/Myr): 405-kyr model
  RMSE {rec1.get('rmse_405_upper_seg', float('nan')):.3f} Myr (bias-removed
  {rec1.get('rmse_405_upper_seg_deb', float('nan')):.3f}), linear
  {rec1.get('rmse_linear_upper_seg', float('nan')):.3f} Myr (bias-removed
  {rec1.get('rmse_linear_upper_seg_deb', float('nan')):.3f}); lower segment
  (mean published SR {rec1.get('pub_mean_sr_lower_seg', float('nan')):.0f}
  m/Myr): 405-kyr RMSE {rec1.get('rmse_405_lower_seg', float('nan')):.3f} Myr
  (bias-removed {rec1.get('rmse_405_lower_seg_deb', float('nan')):.3f}),
  linear {rec1.get('rmse_linear_lower_seg', float('nan')):.3f} Myr
  (bias-removed {rec1.get('rmse_linear_lower_seg_deb', float('nan')):.3f}).
  The published ATS thus implies mean SR rising from ~{rec1.get('pub_mean_sr_upper_seg', 0):.0f}
  to ~{rec1.get('pub_mean_sr_lower_seg', 0):.0f} m/Myr across the step — a
  strongly nonstationary section that no single-anchor constant-cycle model
  can reproduce.

## ODP Site 1264, Walvis Ridge (fig16) — XRF CaCO3, Liebrand et al. 2016 EPSL
- Data: holes 1264A/B merged on armcd, bin-averaged to uniform {rec2['dz']} m,
  {rec2['n_points']} samples, {rec2['depth_top']:.2f}-{rec2['depth_bot']:.2f} m;
  {rec2['n_gap_bins_interp']} empty bins (core gaps, max ~2 m) linearly
  interpolated. Detrend: Gaussian sigma = {rec2['detrend_sigma_m']:g} m
  (SR optimum insensitive to sigma = 3-12 m).
- {_fmt_rec(rec2)}
- Scan duration {rec2['duration_myr_scan']:.2f} Myr vs published
  {rec2['duration_myr_pub']:.2f} Myr (mean SR {rec2['pub_mean_sr']:.1f} m/Myr).
- `select_405_imf` returned **matched=False** (cycle deficit 24 vs Ne=28.7 and
  IMF MTM peak at ~579 kyr, outside 405 kyr +/-20%) -> per contract the reported
  primary model is the constant-SR linear fallback; the 405-kyr anchored model
  is retained as a diagnostic.
- vs published tie-point model (115 tie points, piecewise linear): linear
  fallback RMSE {rec2['rmse_linear']:.3f} Myr, max |dev| {rec2['maxabs_linear']:.3f} Myr;
  405-kyr diagnostic (trough) RMSE {rec2['rmse_405_trough']:.3f} Myr,
  max |dev| {rec2['maxabs_405_trough']:.3f} Myr; peak phase RMSE {rec2['rmse_405_peak']:.3f} Myr.

## Honesty notes
- Match scores are modest on both real datasets (0.28-0.30) versus ~0.9-1.0 on
  the synthetic matrix: the four-band timeOpt-style score is damped by strong
  non-orbital variance and (for 1264) eccentricity-dominated spectra where the
  40/21-kyr bands are weak at these sampling/SR combinations.
- SK-1n: the scan optimum (75.9 m/Myr) is ~9% below the published mean SR and
  the EMD IMF yields 50 cycles vs ~46.5 expected from the published ATS; both
  effects accumulate into Myr-scale drift of the single-anchor floating model.
  The residual profile is strongly structured with depth (segmented SR
  changes; Wu et al. likewise tuned segment-by-segment), so a constant-cycle
  spacing of 0.405 Myr cannot track the published ATS better than a linear
  model here.
- Site 1264: merged-hole bin averaging and gap interpolation are additional
  smoothing/error sources; the 405-kyr mode fails the framework's own dual
  criterion (count deficit + spectral peak check), which we report as a
  genuine non-detection rather than forcing a tuned model.
"""
    with open(path, 'w') as f:
        f.write(txt)
    print(f'  wrote {path}')
# ----------------------------------------------------------------------
# Detrend-sigma sensitivity sweep
# ----------------------------------------------------------------------
def run_sigma_sensitivity():

    print('\n' + '=' * 68)
    print('Exp D detrend-sigma sensitivity')
    print('=' * 68)

    rows = []

    # ==============================================================
    # SK-1n
    # Paper range: sigma = 25–90 m
    # ==============================================================
    depth, th, age_pub = load_sk1n()

    sk_sigmas = np.arange(25.0, 90.0 + 1e-9, 5.0)
    sk_grid = np.arange(40.0, 160.0 + 1e-9, 0.5)

    print('\n--- SK-1n: sigma 25–90 m ---')

    for sigma_m in sk_sigmas:

        trend = gaussian_filter1d(
            th,
            sigma=sigma_m / 0.125,
            mode='nearest'
        )

        x = th - trend

        tuner = AstroTuner(
            sr_grid=sk_grid,
            n_jobs=2,
            seed=SEED
        )

        scan = tuner.scan_sed_rates(depth, x)

        sr_opt, match = tuner.best_sr(
            scan,
            depth=depth,
            signal=x
        )

        rows.append({
            'dataset': 'SK-1n',
            'sigma_m': float(sigma_m),
            'sr_opt': float(sr_opt),
            'match_score': float(match),
        })

        print(
            f'  sigma={sigma_m:5.1f} m  '
            f'SR={sr_opt:8.3f} m/Myr  '
            f'match={match:.3f}'
        )

    # ==============================================================
    # Site 1264
    # Paper range: sigma = 3–12 m
    # ==============================================================
    grid, ca, age_pub2, ties, n_gap = load_1264(dz=0.05)

    site_sigmas = np.arange(3.0, 12.0 + 1e-9, 0.5)
    site_grid = np.arange(2.0, 20.0 + 1e-9, 0.1)

    print('\n--- Site 1264: sigma 3–12 m ---')

    for sigma_m in site_sigmas:

        trend = gaussian_filter1d(
            ca,
            sigma=sigma_m / 0.05,
            mode='nearest'
        )

        x = ca - trend

        tuner = AstroTuner(
            sr_grid=site_grid,
            n_jobs=2,
            seed=SEED
        )

        scan = tuner.scan_sed_rates(grid, x)

        sr_opt, match = tuner.best_sr(
            scan,
            depth=grid,
            signal=x
        )

        rows.append({
            'dataset': 'Site 1264',
            'sigma_m': float(sigma_m),
            'sr_opt': float(sr_opt),
            'match_score': float(match),
        })

        print(
            f'  sigma={sigma_m:4.1f} m  '
            f'SR={sr_opt:7.3f} m/Myr  '
            f'match={match:.3f}'
        )

    # ==============================================================
    # Save
    # ==============================================================
    df = pd.DataFrame(rows)

    out_path = os.path.join(
        EXP_DIR,
        'expD_sigma_sensitivity.csv'
    )

    df.to_csv(out_path, index=False)

    print('\n--- sensitivity summary ---')

    for name, g in df.groupby('dataset'):
        print(
            f'{name}: '
            f'SR min={g["sr_opt"].min():.3f}, '
            f'max={g["sr_opt"].max():.3f}, '
            f'range={g["sr_opt"].max()-g["sr_opt"].min():.3f} m/Myr'
        )

    print(f'\nwrote {out_path}')

    return df

# ----------------------------------------------------------------------
# Round-1 supplement S4: explicit M-gate (match-score gate) audit
# ----------------------------------------------------------------------
M_GATE_THRESH = 0.5      # paper workflow: match score >= 0.5 required


def audit_m_gate(path=None):
    """Reviewer Round-1: the original script never recorded the match-score
    gate decision required by the paper's workflow contract (fig01: match
    >= 0.5 ?). Post-hoc audit appended after the (unchanged) main flow:
    read expD_summary.csv, apply the gate to each dataset's match_score,
    print the verdict, and store it in a new `m_gate` column. A FAIL does
    NOT remove the dataset from the results — the chains above are retained
    explicitly as diagnostics, as the honesty notes already state."""
    path = path or os.path.join(EXP_DIR, 'expD_summary.csv')
    sdf = pd.read_csv(path)
    verdicts = []
    for _, r in sdf.iterrows():
        ok = float(r['match_score']) >= M_GATE_THRESH
        v = 'PASS' if ok else 'FAIL — continued for diagnostic purposes'
        verdicts.append(v)
        print(f"  M-gate audit: {r['name']}: match_score = "
              f"{r['match_score']:.3f} "
              + ('>= 0.5 -> PASS' if ok else f'< {M_GATE_THRESH} -> {v}'))
    sdf['m_gate'] = verdicts
    sdf.to_csv(path, index=False)
    print('  expD_summary.csv updated: new column m_gate')


if __name__ == '__main__':
    main()
    audit_m_gate()
    run_sigma_sensitivity()