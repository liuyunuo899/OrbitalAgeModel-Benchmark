#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment E: combined integer/non-integer hiatus benchmark (Reviewer 1, Q2)

Usage
-----
1) Place this file alongside AstroTuner-405kyr.zip OR inside the extracted
   AstroTuner-405kyr/code/ folder (beside astro_py_tuner.py).
2) In the ORIGINAL paper's Python environment (which has PyEMD / EMD-signal):
       python exp_e_hiatus_combined.py
   Run a dependency-free generator check first if desired:
       python exp_e_hiatus_combined.py --smoke-test

This file does not modify any original experiment file. All outputs go into
new experiment-specific filenames under supplement_results/ and
supplement_figures/ beside this script.

Design (fixed before evaluation; NOT tuned to results)
---------------------------------------------------
Depth: 0..600 m, dz=0.1 m, background constant SR=15 m/Myr.
Hiatus at 300.0 m. Durations H = 0, 0.200, 0.405, 0.600, 0.810, 1.000, 1.620 Myr.
0.405/0.810/1.620 are integer 405-kyr cycles; 0.200/0.600/1.000 are not.
No-hiatus control is counted ONCE (70 unique runs, not 80).
10 paired noise seeds = 2001 + 1000*k (k=0..9).
Four exact orbital sinusoids (405,95,40,21 kyr; amplitudes 6,4,2.5,1.5)
plus AR(1) rho=0.7 and independent white noise, at RMS ratio=0.5.
Same continuous time-domain noise realization is sampled for each H in a
seed pair. The same unchanging signal exists across the missing time; it
is simply not preserved in the depth record.

Caution
-------
No erosion, time-variable sedimentation, phase modulation, real gamma-ray
instrument response, or duration estimation is modeled. An imposed hiatus
is known only for scoring; it is NOT supplied to AstroTuner.
Original AstroTuner Stage-1 match (>=0.5) and Stage-2 IMF match both
recorded. A forced anchored age estimate is computed for diagnosis even
when a gate fails; it must not be described as an accepted/deployed output.
The baseline constant-SR model is a reference, not a certified age model.
Absolute age RMSE is calculated with t(0)=0 for truth AND each model;
forced anchoring retains the original build_age_model_405 default initial
anchor age 0, matching Experiment B for comparability.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import lfilter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

LENGTH_M = 600.0
DZ = 0.1
SR = 15.0
HIATUS_DEPTH_M = 300.0
INTEGER_HIATUS = (0.405, 0.810, 1.620)
NONINTEGER_HIATUS = (0.200, 0.600, 1.000)
HIATUS_MYR = (0.0, 0.200, 0.405, 0.600, 0.810, 1.000, 1.620)
SEEDS = tuple(2001 + 1000*k for k in range(10))
PERIODS_KYR = (405.0, 95.0, 40.0, 21.0)
AMPS = (6.0, 4.0, 2.5, 1.5)
NOISE_RATIO = 0.5
AR1_RHO = 0.7
DT_MYR = 0.0005
SCAN_GRID = np.arange(5.0, 30.00001, 0.25)
MTM_GATE = 0.5
BASE = Path(__file__).resolve().parent
FIG_DIR = BASE / 'supplement_figures'
RES_DIR = BASE / 'supplement_results'


def configure_import():
    """Use the user's unmodified project module from extracted folder or zip."""
    candidates = [BASE, BASE / 'code', BASE / 'AstroTuner-405kyr' / 'code',
                  BASE.parent / 'AstroTuner-405kyr' / 'code']
    for path in candidates:
        if (path / 'astro_py_tuner.py').is_file():
            sys.path.insert(0, str(path))
            from astro_py_tuner import AstroTuner
            return AstroTuner, str(path / 'astro_py_tuner.py')
    zip_candidates = [BASE / 'AstroTuner-405kyr.zip', BASE.parent / 'AstroTuner-405kyr.zip']
    for z in zip_candidates:
        if z.is_file():
            with zipfile.ZipFile(z) as archive:
                if 'AstroTuner-405kyr/code/astro_py_tuner.py' not in archive.namelist():
                    continue
            sys.path.insert(0, str(z) + '/AstroTuner-405kyr/code')
            from astro_py_tuner import AstroTuner
            return AstroTuner, str(z) + '::AstroTuner-405kyr/code/astro_py_tuner.py'
    raise FileNotFoundError('Cannot find astro_py_tuner.py. Put the script beside the original '
                            'AstroTuner-405kyr.zip or in AstroTuner-405kyr/code/.')


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_time_domain(seed):
    """Shared time-domain signal for all seven H values of a seed pair.

    Reproduces the signal construction in exp_common.synth_series, with one
    deliberate extension: one common time grid covers the MAX hiatus duration.
    Therefore it is paired exactly across H within a seed, but is NOT
    byte-identical to the original Experiment B realization.
    """
    tmax = LENGTH_M / SR + max(HIATUS_MYR)
    fine_t = np.arange(0.0, tmax + DT_MYR * 1.01, DT_MYR)
    periodic = np.zeros_like(fine_t)
    for period, amp in zip(PERIODS_KYR, AMPS):
        periodic += amp * np.sin(2.0*np.pi*fine_t/(period/1000.0))
    rng = np.random.default_rng(int(seed))
    w = rng.normal(0.0, 1.0, size=fine_t.size)
    ar1 = lfilter([1.0], [1.0, -AR1_RHO], w)
    ar1 /= ar1.std()
    white = rng.normal(0.0, 1.0, size=fine_t.size)
    signal_rms = np.sqrt(sum(a*a for a in AMPS)/2.0)
    noise = NOISE_RATIO*signal_rms*(ar1 + white)/np.sqrt(2.0)
    return fine_t, periodic + noise


def make_hiatus_series(fine_t, x_time, hiatus_myr):
    """Age jump between last pre-hiatus sample and first post-hiatus sample.

    The discontinuity is at the interval (299.9 m, 300.0 m]. In particular,
    it is NOT a finite zero-sedimentation-rate layer of depth samples.
    """
    depth = np.arange(0.0, LENGTH_M + DZ/2.0, DZ)
    age = depth/SR + float(hiatus_myr)*(depth >= HIATUS_DEPTH_M)
    signal = np.interp(age, fine_t, x_time)
    assert np.all(np.diff(age) > 0)
    assert np.isclose(age[-1], LENGTH_M/SR + hiatus_myr)
    jump_index = int(np.searchsorted(depth, HIATUS_DEPTH_M))
    assert np.isclose(age[jump_index]-age[jump_index-1], DZ/SR + hiatus_myr)
    return dict(depth=depth, signal=signal, t_true=age)


def score(model_age, truth, depth):
    e = np.asarray(model_age) - np.asarray(truth)
    upper = depth < HIATUS_DEPTH_M
    lower = ~upper
    return dict(rmse=float(np.sqrt(np.mean(e*e))),
                max_abs=float(np.max(np.abs(e))),
                end_bias=float(e[-1]),
                rmse_pre=float(np.sqrt(np.mean(e[upper]**2))),
                rmse_post=float(np.sqrt(np.mean(e[lower]**2))))


def hiatus_group(H):
    if abs(H) < 1e-9:
        return 'control'
    if any(np.isclose(H, x) for x in INTEGER_HIATUS):
        return 'integer_cycle'
    return 'noninteger_cycle'


def assemble_row(seed, H, sr_opt, match_score, selected, anchor_count, const, forced, status):
    info = dict(seed=int(seed), hiatus_myr=float(H), hiatus_group=hiatus_group(H), true_duration_myr=LENGTH_M/SR+H,
                hiatus_depth_m=HIATUS_DEPTH_M, baseline_sr_m_per_myr=SR,
                scan_sr_m_per_myr=float(sr_opt), match_score=float(match_score),
                stage1_pass=bool(match_score >= MTM_GATE),
                imf_selected=int(selected['imf_index']+1) if selected is not None else np.nan,
                imf_cycles=int(selected['n_cycles']) if selected is not None else np.nan,
                imf_matched=bool(selected['matched']) if selected is not None else False,
                gate_accepted=(bool(match_score >= MTM_GATE and selected['matched'])
                               if selected is not None else False),
                anchor_count=int(anchor_count), forced_anchor_status=status)
    for prefix, scores in (('constant', const), ('forced_anchor', forced)):
        for k in ('rmse', 'max_abs', 'end_bias', 'rmse_pre', 'rmse_post'):
            info[f'{prefix}_{k}_myr'] = scores[k] if scores is not None else np.nan
    if const is not None and forced is not None:
        info['forced_anchor_wins'] = bool(forced['rmse'] < const['rmse'])
        info['forced_minus_constant_rmse_myr'] = forced['rmse'] - const['rmse']
    else:
        info['forced_anchor_wins'] = False
        info['forced_minus_constant_rmse_myr'] = np.nan
    return info


def smoke_test():
    # Run model-free checks, including deterministic generation and exact gap.
    d1 = make_time_domain(SEEDS[0]); d2 = make_time_domain(SEEDS[0])
    assert np.array_equal(d1[1], d2[1])
    ref = make_hiatus_series(*d1, 0.0)
    one = make_hiatus_series(*d1, 0.405)
    assert np.array_equal(ref['depth'], one['depth'])
    assert np.allclose(ref['signal'][:3000], one['signal'][:3000])
    assert np.isclose(one['t_true'][-1]-ref['t_true'][-1], 0.405)
    assert np.allclose(one['t_true'][:3000], ref['t_true'][:3000])
    for h in HIATUS_MYR:
        current = make_hiatus_series(*d1, h)
        assert np.array_equal(ref['signal'][:3000], current['signal'][:3000])
        assert np.isclose(current['t_true'][-1] - ref['t_true'][-1], h)
    print('SMOKE TEST PASS: deterministic time signal; same pre-hiatus samples; '
          'correct 0.405 Myr gap and section duration; original files untouched.')


def plot_four_panels(df, summary):
    """A publication-oriented 2x2 figure. All legends sit in a dedicated figure footer.

    Each dot is an actual run; lines join medians for display only. Integer and
    noninteger durations use different markers. The zero-hiatus control is separate.
    """
    colors = dict(constant='#AA8251', anchor='#3D6686', stage1='#708F70', gate='#B06656')
    mark = dict(control='s', integer_cycle='o', noninteger_cycle='D')
    xvalues = np.asarray(HIATUS_MYR)
    fig, axes = plt.subplots(2,2,figsize=(13.3,9.2))
    ax_a,ax_b,ax_c,ax_d = axes.flat
    for ax in (ax_a,ax_b,ax_c,ax_d):
        ax.grid(axis='y',color='#D6DCE2',alpha=.65,lw=.65)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    def points_and_medians(ax, metric, color):
        for H in HIATUS_MYR:
            rows = df[df['hiatus_myr']==H]
            vals = rows[metric].to_numpy(dtype=float)
            valid = np.isfinite(vals)
            ax.scatter(np.full(valid.sum(),H),vals[valid],s=29,marker=mark[hiatus_group(H)],
                       color=color,edgecolors='white',linewidths=.45,alpha=.55,zorder=3)
        med = df.groupby('hiatus_myr')[metric].median().reindex(HIATUS_MYR)
        ax.plot(HIATUS_MYR,med.to_numpy(),'-',color=color,lw=1.7,zorder=4)
        ax.scatter(HIATUS_MYR,med.to_numpy(),s=59,marker='o',color=color,
                   edgecolors='white',linewidths=.9,zorder=5)
        ax.set_xticks(HIATUS_MYR,[f'{H:.3f}' for H in HIATUS_MYR],rotation=25,ha='right')
        ax.set_xlabel('Imposed hiatus duration (Myr)')
        ax.set_ylabel('Absolute age RMSE (Myr)')
        ax.set_xlim(-.10,1.73)
    points_and_medians(ax_a,'constant_rmse_myr',colors['constant'])
    ax_a.set_title('(a) Constant-SR age-model error',loc='left',fontsize=12)
    points_and_medians(ax_b,'forced_anchor_rmse_myr',colors['anchor'])
    ax_b.set_title('(b) Forced 405-kyr anchoring error',loc='left',fontsize=12)
    xpos=np.arange(len(HIATUS_MYR)); width=.34
    s=summary.set_index('hiatus_myr').reindex(HIATUS_MYR)
    ax_c.bar(xpos-width/2,s['n_stage1_pass'].to_numpy(),width,color=colors['stage1'])
    ax_c.bar(xpos+width/2,s['n_gate_accepted'].to_numpy(),width,color=colors['gate'])
    ax_c.set_xticks(xpos,[f'{H:.3f}' for H in HIATUS_MYR],rotation=25,ha='right')
    ax_c.set_xlabel('Imposed hiatus duration (Myr)')
    ax_c.set_ylabel('Gate passes (out of 10)')
    ax_c.set_ylim(0,len(SEEDS)+1.0)
    ax_c.set_yticks([0,2,4,6,8,10])
    ax_c.set_title('(c) Original algorithm acceptance gates',loc='left',fontsize=12)
    # Delta is computed as a per-seed paired difference, never differences of medians.
    for H in HIATUS_MYR:
        rows=df[df['hiatus_myr']==H]
        vals=rows['forced_minus_constant_rmse_myr'].to_numpy(dtype=float)
        valid=np.isfinite(vals)
        col=colors['anchor'] if hiatus_group(H)!='control' else '#68727C'
        ax_d.scatter(np.full(valid.sum(),H),vals[valid],s=29,
                     marker=mark[hiatus_group(H)],color=col,alpha=.55,
                     edgecolors='white',linewidths=.45,zorder=3)
    delta=df.groupby('hiatus_myr')['forced_minus_constant_rmse_myr'].median().reindex(HIATUS_MYR)
    ax_d.plot(HIATUS_MYR,delta.to_numpy(),color=colors['anchor'],lw=1.7,zorder=4)
    ax_d.scatter(HIATUS_MYR,delta.to_numpy(),color=colors['anchor'],s=59,
                 edgecolors='white',linewidths=.9,zorder=5)
    ax_d.axhline(0,ls='--',lw=1.1,color='#6F7D86')
    ax_d.set_xticks(HIATUS_MYR,[f'{H:.3f}' for H in HIATUS_MYR],rotation=25,ha='right')
    ax_d.set_xlim(-.10,1.73)
    ax_d.set_xlabel('Imposed hiatus duration (Myr)')
    ax_d.set_ylabel('Forced anchor − constant RMSE (Myr)')
    ax_d.set_title('(d) Paired model-error difference',loc='left',fontsize=12)
    # Two separate legends entirely BELOW the panels. No axes.legend() is used.
    handles=[
        Line2D([],[],marker='s',linestyle='None',color='#75828C',markersize=6,label='No-hiatus control'),
        Line2D([],[],marker='o',linestyle='None',color='#75828C',markersize=6,label='Integer-cycle hiatus'),
        Line2D([],[],marker='D',linestyle='None',color='#75828C',markersize=6,label='Non-integer hiatus'),
        Line2D([],[],lw=2,color=colors['constant'],label='Constant-SR median'),
        Line2D([],[],lw=2,color=colors['anchor'],label='Forced-anchor median'),
        Patch(facecolor=colors['stage1'],label='Stage 1 pass'),
        Patch(facecolor=colors['gate'],label='Both gates pass'),
    ]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(0.5,0.015),
               ncol=4,frameon=False,fontsize=9,columnspacing=2,handlelength=2.2)
    fig.subplots_adjust(left=.095,right=.98,top=.945,bottom=.18,hspace=.50,wspace=.30)
    fig.savefig(FIG_DIR/'expE_hiatus_2x2_comparison.png',dpi=300,facecolor='white')
    fig.savefig(FIG_DIR/'expE_hiatus_2x2_comparison.pdf',facecolor='white')
    plt.close(fig)


def run():
    AstroTuner, tuner_location = configure_import()
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise SystemExit('Missing PyEMD (package name: EMD-signal). Run this script '
                         'in the paper\'s original environment with PyEMD installed. '
                         'No substitute EMD algorithm will be used.') from exc
    FIG_DIR.mkdir(exist_ok=True)
    RES_DIR.mkdir(exist_ok=True)
    tuner = AstroTuner(sr_grid=SCAN_GRID, n_jobs=1, seed=42)
    rows = []
    npz_arrays = {}
    failures = []
    for seed in SEEDS:
        fine_t, x_time = make_time_domain(seed)
        for H in HIATUS_MYR:
            dd = make_hiatus_series(fine_t, x_time, H)
            depth, signal, truth = dd['depth'], dd['signal'], dd['t_true']
            tag = f'seed{seed}_H{int(round(H*1000)):04d}kyr'
            # Original identical depth-domain SR scan for all models
            try:
                scan = tuner.scan_sed_rates(depth, signal)
                sr_opt, ms = tuner.best_sr(scan, depth=depth, signal=signal)
                linear = tuner.build_age_model(depth, sr_opt)
                lin_score = score(linear, truth, depth)
            except Exception as exc:
                msg = f'{tag}: scan exception: {type(exc).__name__}: {exc}'
                failures.append(msg)
                rows.append(assemble_row(seed,H,np.nan,np.nan,None,0,None,None,'scan_error'))
                npz_arrays[f'{tag}__depth'] = depth
                npz_arrays[f'{tag}__signal'] = signal
                npz_arrays[f'{tag}__t_true'] = truth
                print(msg)
                continue
            selected = None
            anchor = None
            count = 0
            forced_score = None
            status = 'not_computed'
            try:
                # Identical EMD and anchor selection as original Exp B/B2
                imfs = EMD().emd(signal, max_imf=10)
                if len(imfs) == 0:
                    raise ValueError('EMD returned zero IMFs')
                selected = tuner.select_405_imf(imfs, depth, sr_est=sr_opt)
                anchor, anchor_table = tuner.build_age_model_405(
                    depth, imfs[selected['imf_index']], anchor_phase='trough')
                count = len(anchor_table)
                forced_score = score(anchor, truth, depth)
                status = 'computed_diagnostic_only'
            except (ValueError, AssertionError, FloatingPointError, IndexError) as exc:
                status = f'anchor_error_{type(exc).__name__}'
                failures.append(f'{tag}: {type(exc).__name__}: {exc}')
            row = assemble_row(seed, H, sr_opt, ms, selected, count,
                               lin_score, forced_score, status)
            rows.append(row)
            npz_arrays[f'{tag}__depth'] = depth
            npz_arrays[f'{tag}__signal'] = signal
            npz_arrays[f'{tag}__t_true'] = truth
            npz_arrays[f'{tag}__constant_age'] = linear
            if anchor is not None:
                npz_arrays[f'{tag}__forced_anchor_age'] = anchor
            print(f'{tag}: match={ms:.3f} scan_sr={sr_opt:.2f} '
                  f'gate={row["gate_accepted"]} const RMSE={lin_score["rmse"]:.3f} '
                  f'forced RMSE={forced_score["rmse"]:.3f}' if forced_score else
                  f'{tag}: match={ms:.3f} scan_sr={sr_opt:.2f} '
                  f'anchor unavailable ({status})')
    df = pd.DataFrame(rows).sort_values(['hiatus_myr','seed'])
    assert len(df) == len(SEEDS)*len(HIATUS_MYR), 'Expected 70 records'
    assert not df.duplicated(['seed','hiatus_myr']).any()
    raw_path = RES_DIR / 'expE_hiatus_combined_all70.csv'
    df.to_csv(raw_path, index=False)
    summary = (df.groupby('hiatus_myr', as_index=False).agg(
        n=('seed', 'size'), n_stage1_pass=('stage1_pass','sum'),
        n_gate_accepted=('gate_accepted','sum'),
        n_forced_computed=('forced_anchor_rmse_myr','count'),
        mean_constant_rmse=('constant_rmse_myr','mean'),
        mean_forced_anchor_rmse=('forced_anchor_rmse_myr','mean'),
        median_constant_rmse=('constant_rmse_myr','median'),
        median_forced_anchor_rmse=('forced_anchor_rmse_myr','median'),
        mean_pre_forced_rmse=('forced_anchor_rmse_pre_myr','mean'),
        mean_post_forced_rmse=('forced_anchor_rmse_post_myr','mean'),
        forced_anchor_wins=('forced_anchor_wins','sum')))
    summary.to_csv(RES_DIR / 'expE_hiatus_combined_summary.csv',index=False)
    by_group = (df.groupby('hiatus_group', as_index=False).agg(
        n=('seed','size'),
        gate_pass=('gate_accepted','sum'),
        median_constant_rmse=('constant_rmse_myr','median'),
        median_forced_rmse=('forced_anchor_rmse_myr','median'),
        median_forced_minus_constant=('forced_minus_constant_rmse_myr','median'),
        forced_anchor_wins=('forced_anchor_wins','sum')))
    by_group.to_csv(RES_DIR/'expE_hiatus_combined_by_group.csv',index=False)
    np.savez_compressed(RES_DIR / 'expE_hiatus_combined_series.npz',**npz_arrays)
    (RES_DIR/'expE_hiatus_combined_errors.txt').write_text('\n'.join(failures) or 'No per-realization errors.',encoding='utf-8')
    metadata = dict(experiment='E, combined paired integer/non-integer hiatus sensitivity',
        values={'length_m':LENGTH_M,'dz_m':DZ,'sr_m_per_myr':SR,
                'hiatus_depth_m':HIATUS_DEPTH_M,'hiatus_myr':HIATUS_MYR,
                'integer_cycle_hiatus':INTEGER_HIATUS,'noninteger_cycle_hiatus':NONINTEGER_HIATUS,
                'seeds':SEEDS,'periods_kyr':PERIODS_KYR,'amplitudes':AMPS,
                'noise_ratio':NOISE_RATIO,'AR1_rho':AR1_RHO,
                'fine_dt_myr':DT_MYR,'sr_scan_min':float(SCAN_GRID.min()),
                'sr_scan_max':float(SCAN_GRID.max()),'sr_scan_step':0.25,
                'imf_max':10,'stage1_gate':MTM_GATE,'tuner_seed':42},
        tuner_source=tuner_location,
        limitations=['Only one central nondeposition hiatus; no erosion',
            'Four ideal sinusoids rather than real astronomical solution',
            'Seed-paired noise is identical in time, but not identical to Experiment B data',
            'No instrument response',
            'Forced anchoring is diagnostic even when gates fail',
            'Hiatus known for scoring only; not an input to AstroTuner'],
        outputs=['expE_hiatus_combined_all70.csv','expE_hiatus_combined_summary.csv',
                 'expE_hiatus_combined_by_group.csv','expE_hiatus_combined_series.npz',
                 'expE_hiatus_combined_errors.txt','expE_hiatus_2x2_comparison.png'])
    (RES_DIR/'expE_hiatus_combined_design.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    # Check that the 40 previously run integer-cycle/control cases reproduce.
    # If the user's earlier CSV is present, verify results; do not overwrite it.
    original40 = RES_DIR / 'expE_hiatus_all40.csv'
    if original40.is_file():
        old = pd.read_csv(original40)
        old = old[old['hiatus_myr'].isin((0.0,)+INTEGER_HIATUS)]
        new = df[df['hiatus_myr'].isin((0.0,)+INTEGER_HIATUS)]
        cols = ['constant_rmse_myr','forced_anchor_rmse_myr','match_score',
                'scan_sr_m_per_myr','gate_accepted']
        matched = old.merge(new, on=['seed','hiatus_myr'], suffixes=('_old','_new'),validate='one_to_one')
        if len(matched) != 40:
            raise AssertionError(f'Old/new comparison has {len(matched)} matches, expected 40')
        for col in cols:
            x,y = matched[f'{col}_old'], matched[f'{col}_new']
            if col == 'gate_accepted':
                assert np.array_equal(x.astype(str).to_numpy(),y.astype(str).to_numpy()),col
            else:
                assert np.allclose(x.to_numpy(dtype=float),y.to_numpy(dtype=float),rtol=1e-7,atol=1e-8,equal_nan=True),col
        print('REPRODUCIBILITY PASS: all 40 prior records agree with combined rerun.')
    else:
        print('INFO: previous expE_hiatus_all40.csv not found; prior-run comparison skipped.')

    plot_four_panels(df, summary)
    print('\nSUMMARY (NOT manuscript-ready until reviewed):\n',summary.to_string(index=False))
    print('\nSaved:',raw_path)
    print('Saved:',RES_DIR/'expE_hiatus_combined_summary.csv')
    print('Saved:',FIG_DIR/'expE_hiatus_2x2_comparison.png')
    return df,summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test',action='store_true',help='Test pairing/hiatus generator without PyEMD')
    args=parser.parse_args()
    smoke_test()
    if not args.smoke_test:
        run()
