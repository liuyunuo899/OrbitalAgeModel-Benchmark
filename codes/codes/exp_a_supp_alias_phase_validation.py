#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Supplementary validation for Reviewer 1, Comment 1.

Audits the ORIGINAL Experiment A detection CSV without altering results, and
runs two independent, deterministic, noise-free sampling demonstrations:
(1) 95 kyr at 2 m/Myr: demonstrate the 105.56-kyr spectral alias.
(2) 40 kyr at 5 m/Myr: demonstrate the role of initial phase at Nyquist.

This is NOT a rerun of the 120 original noisy synthetic records, and is NOT
an independently calibrated instrument-response experiment. Outputs are
supplementary diagnostics; no original files are edited.

Usage:
    python exp_a_supp_alias_phase_validation.py
    python exp_a_supp_alias_phase_validation.py --source /path/to/AstroTuner-405kyr.zip
    python exp_a_supp_alias_phase_validation.py --source /path/to/AstroTuner-405kyr/code

Dependencies: numpy, pandas, matplotlib (no PyEMD needed).
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ORIGINAL_CSV = 'expA_detection.csv'
DZ_M = 0.1
LENGTH_M = 600.0
PALETTE = {'blue': '#41627E', 'orange': '#A65A4A', 'green': '#6E8B67'}


def locate_source(source: str | None):
    script_dir = Path(__file__).resolve().parent
    paths = ([Path(source)] if source else [
        script_dir / 'AstroTuner-405kyr.zip',
        script_dir.parent / 'AstroTuner-405kyr.zip',
        script_dir / 'expA_detection.csv',
        script_dir / 'AstroTuner-405kyr' / 'code',
        script_dir,
    ])
    for p in paths:
        if p.is_file() and p.suffix.lower() == '.zip':
            with zipfile.ZipFile(p) as z:
                matches = [n for n in z.namelist() if n.endswith('/' + ORIGINAL_CSV)]
                if len(matches) == 1:
                    return pd.read_csv(io.BytesIO(z.read(matches[0]))), str(p), matches[0]
        elif p.is_file() and p.name == ORIGINAL_CSV:
            return pd.read_csv(p), str(p), p.name
        elif p.is_dir():
            matches = [p / ORIGINAL_CSV, p / 'code' / ORIGINAL_CSV]
            for item in matches:
                if item.is_file():
                    return pd.read_csv(item), str(item), item.name
    raise FileNotFoundError(
        'Cannot find original expA_detection.csv. Put this script beside the ZIP, '
        'or pass --source with the ZIP or code directory path.'
    )


def alias_period_kyr(period_kyr: float, sr_m_myr: float, dz_m: float = DZ_M):
    # Discrete angular frequency, folded to [0, 0.5] cycles/sample.
    f_depth = 1000.0 / (period_kyr * sr_m_myr)  # cycles/m
    f_samples = f_depth * dz_m
    folded = abs((f_samples + 0.5) % 1.0 - 0.5)
    if folded < 1e-12:
        return np.inf
    wavelength_observed_m = dz_m / folded
    return 1000.0 * wavelength_observed_m / sr_m_myr


def source_audit(df: pd.DataFrame):
    required = {'sr_true','noise','period_kyr','wavelength_m','significance',
                'detected','nyquist_detectable'}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f'Original CSV is missing columns: {missing}')
    audit = df.copy()
    audit['nyquist_threshold_sr_m_per_Myr'] = 2000 * DZ_M / audit['period_kyr']
    audit['recalculated_nyquist'] = (audit['wavelength_m'] >= 2 * DZ_M).astype(int)
    audit['agrees_with_original_nyquist'] = (
        audit['recalculated_nyquist'] == audit['nyquist_detectable']
    )
    audit['detection_nyquist_agreement'] = (
        audit['detected'] == audit['nyquist_detectable']
    )
    audit['alias_period_if_sampled_kyr'] = [
        alias_period_kyr(float(p), float(s))
        for p,s in zip(audit['period_kyr'], audit['sr_true'])
    ]
    return audit


def plot_demonstrations(figpath: Path, resultsdir: Path):
    # Test 1: 95 kyr at SR = 2 m/Myr, wavelength = 0.19 m < Nyquist 0.2 m.
    p95, sr95 = 95.0, 2.0
    lam95 = p95 * sr95 / 1000.0
    p_alias = alias_period_kyr(p95, sr95)
    lam_alias = p_alias * sr95 / 1000.0
    d_fine = np.linspace(0, 3, 6001)
    d_samples = np.arange(0, 3.0 + DZ_M/2, DZ_M)
    sig_fine = np.sin(2 * np.pi * d_fine / lam95)
    sig_samples = np.sin(2 * np.pi * d_samples / lam95)
    # At regular grid points the observed samples are identical to those of
    # this alias sinusoid with reversed sign (since f_alias = f_sample-f_true).
    alias_samples = -np.sin(2 * np.pi * d_samples / lam_alias)
    max_alias_residual = float(np.max(np.abs(sig_samples - alias_samples)))
    if max_alias_residual > 1e-10:
        raise AssertionError(f'Alias identity failed: {max_alias_residual}')

    # Test 2: 40 kyr at SR = 5, wavelength = 0.2 m == 2*DZ.
    # For phase phi, sampled x_n = sin(pi*n+phi)=(-1)^n sin(phi).
    p40, sr40 = 40.0, 5.0
    lam40 = p40 * sr40 / 1000.0
    depth40 = np.arange(0, 10 + DZ_M/2, DZ_M)
    phases = np.linspace(0, 2*np.pi, 73)
    amps = []
    rows = []
    for phi in phases:
        x = np.sin(2*np.pi*depth40/lam40 + phi)
        # RMS of the sampled noiseless harmonic; this measures sampling-phase
        # dependence, not spectral significance or detection probability.
        rms = float(np.sqrt(np.mean(x*x)))
        amps.append(rms)
        rows.append({'phase_rad': phi, 'phase_deg': np.rad2deg(phi),
                     'sampled_rms': rms,
                     'predicted_abs_sin_phase': abs(np.sin(phi))})
    phase_df = pd.DataFrame(rows)
    phase_df.to_csv(resultsdir / 'expA_supp_nyquist_phase_rms.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.30, wspace=0.32)
    axes[0].plot(d_fine, sig_fine, color=PALETTE['blue'], lw=1.3,
                 label='Underlying 95-kyr sine')
    axes[0].scatter(d_samples, sig_samples, s=17, color=PALETTE['orange'],
                    zorder=3, label='0.1-m samples')
    axes[0].plot(d_fine, -np.sin(2*np.pi*d_fine/lam_alias), '--',
                 color=PALETTE['green'], lw=1.4,
                 label=f'Indistinguishable alias ({p_alias:.2f} kyr)')
    axes[0].set(xlabel='Depth (m)', ylabel='Signal amplitude',
                title='(a) 95-kyr alias at SR = 2 m/Myr', xlim=(0,3))
    axes[0].legend(loc='upper center', bbox_to_anchor=(0.5,-0.25),
                   ncol=1, frameon=False, fontsize=8)

    axes[1].plot(np.rad2deg(phases), amps, '-', color=PALETTE['blue'], lw=1.8,
                 label='Sampled RMS (noise-free)')
    axes[1].plot(np.rad2deg(phases), np.abs(np.sin(phases)), ':',
                 color=PALETTE['orange'], lw=1.8,
                 label='Analytical |sin(phase)|')
    axes[1].set(xlabel='Initial phase (degrees)', ylabel='Sampled RMS',
                title='(b) 40-kyr sine at the Nyquist boundary',
                xlim=(0,360), ylim=(-0.04,1.06))
    axes[1].legend(loc='upper center', bbox_to_anchor=(0.5,-0.25),
                   ncol=1, frameon=False, fontsize=8)

    fig.savefig(figpath, dpi=240, bbox_inches='tight')
    plt.close(fig)
    return max_alias_residual, p_alias, phase_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=str, help='Original ZIP, CSV, or code directory')
    args = parser.parse_args()
    original, source_path, internal_path = locate_source(args.source)
    script_dir = Path(__file__).resolve().parent
    result_dir = script_dir / 'supplement_results'
    figure_dir = script_dir / 'supplement_figures'
    result_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    audit = source_audit(original)
    audit.to_csv(result_dir / 'expA_supp_original_detection_audit.csv', index=False)
    disagree = audit.loc[~audit['detection_nyquist_agreement']]
    disagree.to_csv(result_dir / 'expA_supp_disagreement_rows.csv', index=False)
    unique_disagree = disagree[['sr_true','period_kyr']].drop_duplicates()

    figpath = figure_dir / 'expA_supp_alias_phase_validation.png'
    resid, p_alias, phase_df = plot_demonstrations(figpath, result_dir)
    summary = [
        'Supplementary sampling validation -- deterministic diagnostic',
        f'Original source: {source_path} (internal: {internal_path})',
        f'Original detection rows audited: {len(audit)}',
        f'Nyquist/detection disagreement rows: {len(disagree)}',
        f'Unique disagreeing period-SR cells: {len(unique_disagree)}',
        'Disagreeing cells: '+ '; '.join(
            f'{r.period_kyr:g} kyr at {r.sr_true:g} m/Myr'
            for r in unique_disagree.itertuples(index=False)),
        f'95-kyr true wavelength: {95*2/1000:.3f} m',
        f'95-kyr predicted alias period: {p_alias:.6f} kyr',
        f'95-kyr sample/alias identity max absolute residual: {resid:.3g}',
        f'40-kyr Nyquist wavelength: {40*5/1000:.3f} m',
        f'40-kyr sampled RMS at phase 0 deg: {phase_df.iloc[0].sampled_rms:.6g}',
        f'40-kyr sampled RMS at phase 90 deg: {phase_df.loc[phase_df.phase_deg==90, "sampled_rms"].iloc[0]:.6g}',
        '',
        'IMPORTANT: These demonstrations are noiseless, single-component and do not',
        'recompute Experiment A detections or establish new detection success rates.',
        'They do not simulate real instrument calibration or effective vertical resolution.',
        'Nyquist boundary equality is a theoretical limit, not robust cycle recovery.',
    ]
    (result_dir / 'expA_supp_validation_summary.txt').write_text(
        '\n'.join(summary)+'\n', encoding='utf-8')
    print('\n'.join(summary))
    print(f'\nFigure: {figpath}')
    print(f'Results: {result_dir}')


if __name__ == '__main__':
    main()
