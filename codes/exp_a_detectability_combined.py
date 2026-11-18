# -*- coding: utf-8 -*-
"""Experiment A + pre-sampling vertical-response sensitivity (audit-ready).

Place this file beside the ORIGINAL exp_common.py, astro_py_tuner.py,
exp_a_detectability.py and (preferably) expA_detection.csv.

Default: compare original MTM band-detection results at mid noise (0.5)
against Gaussian response FWHM=0, 0.3, 0.5 m. Both 40 and 21 kyr are
explicitly reported, with all four original frequency-band diagnostics saved.

The instrument model is ILLUSTRATIVE, not calibrated to deMenocal (1992).
It filters the complete synthetic proxy (including the pre-existing noise)
BEFORE sampling at 0.1 m. There is no modeled post-instrument noise.
The zero-response condition MUST reproduce exp_common.synth_series() and,
where present, archived expA_detection.csv within numerical tolerance.

Optional --full-original reruns all 30 original realizations, computes
SR scans and regenerates nine representative series IN THE NEW OUTPUT FOLDER,
never overwriting the original experiment files. This is slow.

Options:
  python exp_a_detectability_combined.py --check-only
  python exp_a_detectability_combined.py
  python exp_a_detectability_combined.py --full-original
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.signal import lfilter
from scipy.ndimage import gaussian_filter1d

from exp_common import (AstroTuner, DZ, SIG_RMS, PERIODS_KYR, AMPS,
                        synth_series, band_significances, const_sr_fn)
# Original Experiment A design (copied unchanged; original script NOT required).
SR_LIST = [2, 3, 5, 7.5, 10, 15, 20, 30, 40, 50]
NOISES = [('low', 0.2), ('mid', 0.5), ('high', 0.8)]
LENGTH_M = 600.0
SCAN_GRID = np.arange(1.0, 55.0001, 0.5)
KEEP = {(5.0, 'low'), (5.0, 'mid'), (5.0, 'high'),
        (15.0, 'low'), (15.0, 'mid'), (15.0, 'high'),
        (2.0, 'mid'), (30.0, 'mid'), (50.0, 'mid')}


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'supplement_results' / 'expA_vertical_response_combined'
FIG = ROOT / 'supplement_figures'
FWHMS = (0.0, 0.3, 0.5)  # illustrative Gaussian full widths at half maximum (m)
FINE_DZ = 0.01          # m, ten times finer than observed sampling interval
MID_NOISE = ('mid', 0.5)
FREQ_BANDS = (40.0, 21.0)


def build_original_continuous_signal(sr, noise_ratio, seed):
    """Rebuild exactly the fine-time noise draw used in exp_common.synth_series.

    Not a new random realization: same RNG calls, same seeds, same fine-time
    astronomical signal and AR(1)+white-noise construction.
    """
    depth = np.arange(0, LENGTH_M + 0.5 * DZ, DZ)
    sr_arr = np.asarray(const_sr_fn(sr)(depth), dtype=float)
    t = np.concatenate(([0.0], cumulative_trapezoid(1.0 / sr_arr, depth)))
    dt_myr = 5e-4
    t_fine = np.arange(0.0, t[-1] + dt_myr, dt_myr)
    x_t = np.zeros_like(t_fine)
    for period, amp in zip(PERIODS_KYR, AMPS):
        x_t += amp * np.sin(2 * np.pi * t_fine / (period / 1000.0))
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, 1.0, t_fine.size)
    ar1 = lfilter([1.0], [1.0, -0.7], w)
    ar1 /= ar1.std()
    white = rng.normal(0.0, 1.0, t_fine.size)
    x_t += noise_ratio * SIG_RMS * (ar1 + white) / np.sqrt(2)
    return depth, t, t_fine, x_t


def pre_sample_response(sr, ratio, seed, fwhm):
    """Perform continuous-domain-approximation convolution before digitization."""
    original = synth_series(const_sr_fn(sr), length_m=LENGTH_M,
                            noise_ratio=ratio, seed=seed)
    if fwhm == 0:
        return original['signal'].copy(), original, 0.0

    depth, t_true, t_fine, x_t = build_original_continuous_signal(sr, ratio, seed)
    check = np.interp(t_true, t_fine, x_t)
    err = float(np.max(np.abs(check - original['signal'])))
    if err > 1e-9:
        raise AssertionError('Original fine-time signal regeneration failed: max error %.3g' % err)

    # Continuous-space approximation: interpolate to 1-cm depth grid,
    # Gaussian-convolve the entire proxy there, and finally observe at 0.1 m.
    fine_depth = np.arange(0.0, LENGTH_M + FINE_DZ * 0.5, FINE_DZ)
    fine_time = fine_depth / float(sr)  # constant SR; same truth as above
    fine_signal = np.interp(fine_time, t_fine, x_t)
    sigma_m = fwhm / (2 * np.sqrt(2 * np.log(2)))
    filtered = gaussian_filter1d(fine_signal, sigma=sigma_m/FINE_DZ,
                                  mode='reflect', truncate=5.0)
    measured = np.interp(depth, fine_depth, filtered)
    return measured, original, err


def read_original_archive():
    path = ROOT / 'expA_detection.csv'
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    required = {'sr_true', 'noise', 'period_kyr', 'significance', 'detected'}
    if not required.issubset(df.columns):
        raise ValueError(f'Archived baseline {path} lacks columns: {required-set(df.columns)}')
    if len(df) != 120:
        raise ValueError(f'Archived baseline has {len(df)} rows, expected 120')
    return df


def peak_in_band(tuner, freq, power, period):
    f0 = 1000.0 / period
    band = (freq >= f0*(1-tuner.band_frac)) & (freq <= f0*(1+tuner.band_frac))
    if not np.any(band):
        return float('nan')
    return float(1000.0 / freq[band][np.argmax(power[band])])


def run(check_only=False):
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    archive = read_original_archive()
    if archive is None:
        print('ARCHIVE NOT FOUND: expA_detection.csv is missing; zero-response'
              ' baseline will be checked against synth_series, but original 120-row'
              ' CSV equality cannot be verified. Keep original file for submission.')
    tuner = AstroTuner(sr_grid=SCAN_GRID, n_jobs=1, seed=42)
    rows = []
    checks = []
    for i, sr in enumerate(SR_LIST):
        j = 1  # Original mid noise seed index: 1000 + i*10 + 1
        seed = 1000 + 10*i + j
        nlab, ratio = MID_NOISE
        base = synth_series(const_sr_fn(sr), length_m=LENGTH_M,
                            noise_ratio=ratio, seed=seed)
        # Independent reproduction of archived rows at FWHM=0.
        sigs, flags, _, _ = band_significances(tuner, base['t_true'], base['signal'])
        for p, sg, fl in zip(PERIODS_KYR, sigs, flags):
            if archive is not None:
                match = archive[(archive.sr_true == sr) &
                                (archive.noise == nlab) &
                                (archive.period_kyr == p)]
                if len(match) != 1:
                    raise AssertionError(f'Archive missing/duplicated SR={sr}, period={p}')
                old = match.iloc[0]
                if int(old.detected) != int(fl):
                    raise AssertionError(f'Baseline detection differs at SR={sr}, P={p}')
                # Floating-point math/library differences may cause small rounding.
                if not np.isclose(float(old.significance), float(sg), rtol=1e-6, atol=1e-6):
                    raise AssertionError(f'Baseline significance differs at SR={sr}, P={p}: '
                                         f'{old.significance} vs {sg}')
            checks.append(dict(sr=sr, period_kyr=p, baseline_flag=int(fl),
                               baseline_significance=float(sg)))
        print(f'BASELINE PASS: SR={sr:g}, seed={seed}; bands={list(flags)}')
        if check_only:
            continue

        for fwhm in FWHMS:
            observed, orig, regeneration_error = pre_sample_response(sr, ratio, seed, fwhm)
            # Exactly the original true-time-axis MTM scoring (no threshold tuning).
            sigs, flags, freq, power = band_significances(
                tuner, orig['t_true'], observed)
            for period, sig, flag in zip(PERIODS_KYR, sigs, flags):
                spatial_wavelength = period * sr / 1000
                sigma_m = fwhm / (2*np.sqrt(2*np.log(2))) if fwhm else 0.
                theoretical_gain = float(np.exp(-.5*(2*np.pi*sigma_m/spatial_wavelength)**2))
                rows.append(dict(sr_true=sr, noise=nlab, noise_ratio=ratio,
                                 seed=seed, fwhm_m=fwhm,
                                 period_kyr=period, wavelength_m=spatial_wavelength,
                                 nyquist_safe=int(spatial_wavelength > 2*DZ),
                                 significance=float(sig), detected=int(flag),
                                 peak_period_within_band_kyr=peak_in_band(tuner, freq, power, period),
                                 theoretical_harmonic_gain=theoretical_gain,
                                 baseline_regeneration_max_error=regeneration_error))
            vals = [f'{int(p)} kyr:{int(flags[k])}(sig={sigs[k]:.2f})'
                    for k,p in enumerate(PERIODS_KYR) if p in FREQ_BANDS]
            print(f'  SR={sr:g}, FWHM={fwhm:g} m: '+', '.join(vals))

    checkdf = pd.DataFrame(checks)
    checkdf.to_csv(OUT / 'baseline_detection_check_40.csv', index=False)
    if check_only:
        print('CHECK PASS: original 40 mid-noise spectral-detection rows verified.'
              if archive is not None else
              'CHECK COMPLETE: original CSV absent; local baseline computed only.')
        return None

    df = pd.DataFrame(rows)
    if len(df) != len(SR_LIST)*len(FWHMS)*len(PERIODS_KYR):
        raise AssertionError('Incomplete sensitivity grid')
    base = df[df.fwhm_m == 0].sort_values(['sr_true','period_kyr']).reset_index(drop=True)
    for _, r in base.iterrows():
        found = checkdf[(checkdf.sr == r.sr_true)&(checkdf.period_kyr == r.period_kyr)]
        if int(found.iloc[0].baseline_flag) != int(r.detected):
            raise AssertionError('Zero-response results differ within the combined script')
    df.to_csv(OUT / 'expA_response_all120_midnoise.csv', index=False)
    selected = df[df.period_kyr.isin(FREQ_BANDS)].copy()
    selected.to_csv(OUT / 'expA_response_40_21_kyr.csv', index=False)
    summary = (selected.groupby(['period_kyr', 'fwhm_m'], as_index=False)
               .agg(detected_count=('detected','sum'),
                    tested_sr_count=('detected','size'),
                    min_detected_grid_sr=('sr_true', lambda s: float('nan'))))
    # 'first observed detected grid SR' is descriptive only, NOT physical threshold.
    for ix, row in summary.iterrows():
        subset = selected[(selected.period_kyr == row.period_kyr)&
                          (selected.fwhm_m == row.fwhm_m)&
                          (selected.detected == 1)]
        summary.at[ix,'min_detected_grid_sr'] = (float(subset.sr_true.min()) if len(subset) else float('nan'))
    summary.to_csv(OUT / 'expA_response_summary.csv', index=False)

    fig, axes = plt.subplots(2,2,figsize=(12.8,8.5),constrained_layout=False)
    colors={0.0:'#41627E',0.3:'#B08C57',0.5:'#A65A4A'}
    for col, period in enumerate(FREQ_BANDS):
        sub=selected[selected.period_kyr==period]
        ax=axes[0,col]
        for fwhm in FWHMS:
            z=sub[sub.fwhm_m==fwhm].sort_values('sr_true')
            ax.plot(z.sr_true,z.detected,marker='o',ms=5,lw=1.4,
                    color=colors[fwhm],label=f'FWHM {fwhm:g} m')
        ax.set_ylim(-0.13,1.18)
        ax.set_yticks([0,1],['Not flagged','Flagged'])
        ax.set_xticks(SR_LIST)
        ax.tick_params(axis='x',rotation=35)
        ax.grid(alpha=.2)
        ax.set_ylabel('Original MTM detection flag' if col==0 else '')
        ax.set_title(f'({"ab"[col]})  {int(period)}-kyr band: flags',loc='left')
        if col==1:ax.legend(loc='lower right',fontsize=8,ncol=1)
        ax.set_xlabel('True sedimentation rate (m/Myr)')

        ax=axes[1,col]
        for fwhm in FWHMS:
            z=sub[sub.fwhm_m==fwhm].sort_values('sr_true')
            ax.plot(z.sr_true,z.significance,marker='o',ms=4,lw=1.4,
                    color=colors[fwhm],label=f'FWHM {fwhm:g} m')
        ax.axhline(tuner.sig_thresh,ls='--',c='#64748b',lw=1,label='Original threshold = 5')
        ax.set_yscale('log')
        ax.set_ylim(bottom=0.45)
        ax.set_xticks(SR_LIST)
        ax.tick_params(axis='x',rotation=35)
        ax.grid(alpha=.2)
        ax.set_ylabel('MTM band-peak / neighboring-median' if col==0 else '')
        ax.set_xlabel('True sedimentation rate (m/Myr)')
        ax.set_title(f'({"cd"[col]})  {int(period)}-kyr band: significance',loc='left')
        if col==1:ax.legend(loc='upper left',fontsize=8)
    fig.suptitle('Experiment A: illustrative pre-sampling vertical-response sensitivity',fontsize=13,y=.99)
    fig.subplots_adjust(left=.10,right=.98,top=.91,bottom=.11,wspace=.22,hspace=.38)
    for ext in ('png','pdf'):
        fig.savefig(FIG / f'expA_vertical_response_2x2.{ext}',dpi=250,bbox_inches='tight')
    plt.close(fig)

    design=dict(input='original exp_common.synth_series, mid noise only',
                same_seeds=True, noise_ratio=.5, original_noise_is_pre_response=True,
                post_response_noise_added=False, sr_grid=SR_LIST,
                fwhm_m=FWHMS, fine_depth_grid_m=FINE_DZ,
                original_depth_grid_m=DZ, gaussian_boundary='reflect',
                mtm_scoring='original band_significances with true time axis',
                target_periods_kyr=PERIODS_KYR,
                interpretation='illustrative sensitivity, not a calibrated instrument threshold',
                archive_verified=archive is not None)
    (OUT/'expA_response_design.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    print('\nSUMMARY (not a field-calibrated detection threshold):\n',summary.to_string(index=False))
    print('SAVED:',OUT,'and',FIG/'expA_vertical_response_2x2.png')
    return df


def rerun_full_original():
    """Reproduce the entire original 30-realization A experiment in a new folder.

    Writes original-format 120-row detection and 30-row SR recovery CSVs plus
    nine preserved representative series for Experiment C in OUT only.
    Does not overwrite the archived CSV/NPZ or original publication figures.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    tuner = AstroTuner(sr_grid=SCAN_GRID, n_jobs=1, seed=42)
    det_rows, rec_rows, kept = [], [], {}
    for i, sr in enumerate(SR_LIST):
        for j, (nlab, nrat) in enumerate(NOISES):
            seed = 1000 + i*10 + j
            d = synth_series(const_sr_fn(sr), length_m=LENGTH_M,
                             noise_ratio=nrat, seed=seed)
            sigs, det, _, _ = band_significances(tuner, d['t_true'], d['signal'])
            for p, sg, dt_ in zip(PERIODS_KYR, sigs, det):
                det_rows.append(dict(sr_true=sr, noise=nlab, noise_ratio=nrat,
                      period_kyr=p, wavelength_m=p*sr/1000.0,
                      significance=float(sg), detected=int(dt_),
                      nyquist_detectable=int(p*sr/1000.0 >= 2*DZ)))
            scan = tuner.scan_sed_rates(d['depth'], d['signal'])
            sr_opt, m = tuner.best_sr(scan, depth=d['depth'], signal=d['signal'])
            rec_rows.append(dict(sr_true=sr, noise=nlab, noise_ratio=nrat,
                      sr_est=float(sr_opt), rel_err=float((sr_opt-sr)/sr),
                      match_score=float(m), seed=seed))
            if (sr,nlab) in KEEP:
                kept[f'A_sr{sr:g}_{nlab}'] = d
            print(f'FULL ORIGINAL: SR={sr:g} noise={nlab} detected={list(det)} '
                  f'sr_est={sr_opt:.3f} M={m:.3f}')
    detdf = pd.DataFrame(det_rows)
    recdf = pd.DataFrame(rec_rows)
    original_det = read_original_archive()
    if original_det is not None:
        order = ['sr_true','noise','period_kyr']
        a=detdf.sort_values(order).reset_index(drop=True)
        b=original_det.sort_values(order).reset_index(drop=True)
        if not np.array_equal(a.detected.to_numpy(), b.detected.to_numpy()):
            raise AssertionError('Original A 120 detection flags changed')
        if not np.allclose(a.significance.to_numpy(),b.significance.to_numpy(),rtol=1e-6,atol=1e-6):
            raise AssertionError('Original A significances changed')
    detdf.to_csv(OUT/'expA_original_full30_detection120.csv',index=False)
    recdf.to_csv(OUT/'expA_original_full30_sr_recovery.csv',index=False)
    npz = {}
    for label, d in kept.items():
        for field in ('depth','signal','t_true','sr_true'):
            npz[f'{label}__{field}']=d[field]
    np.savez(OUT/'expA_original_nine_representative_series.npz',**npz)
    print('FULL ORIGINAL PASS: 120 detections, 30 SR scans, 9 representative series; files in',OUT)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only',action='store_true',help='Check original mid-noise detections, no sensitivity')
    parser.add_argument('--full-original',action='store_true',
                        help='Additionally rerun all 30 original cases including SR scans; slow')
    args=parser.parse_args()
    if args.check_only and args.full_original:
        parser.error('--check-only and --full-original cannot be combined')
    if args.full_original: rerun_full_original()
    run(check_only=args.check_only)


if __name__=='__main__':main()
