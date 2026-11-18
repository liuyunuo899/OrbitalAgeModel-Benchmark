# -*- coding: utf-8 -*-
"""Experiment G — anchor aggregation and resolution sensitivity.

Uses the ORIGINAL, archived Experiment B2 output (no new orbital signal, noise,
EMD or IMF selection). Predefined aggregation factors K=(1,2,4,8).
The first and final anchors are retained and intermediate anchors are sampled
at the original-index stride K. Retained ages preserve the ORIGINAL number of
405-kyr intervals, i.e. (index difference)*0.405 Myr, not a new 405-kyr period.

IMPORTANT: results are post-processing sensitivity diagnostics, NOT an improved
AstroTuner method, new cycle identification, or a validated hiatus correction.

Place this script alongside expB2_mc.npz and expB2_mc.csv, then run in Spyder:
%runfile '.../codes/exp_g_anchor_spacing.py' --wdir
Optional validation only: --args "--check-only"

Outputs in supplement_results and supplement_figures, never overwrites B2.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

FACTORS = (1, 2, 4, 8)
SCENARIOS = ('linear', 'step', 'sine')
N_SEEDS = 10
PERIOD_MYR = 0.405
PROMINENCE_FRACTION = 0.5
SCRIPT_DIR = Path(__file__).resolve().parent
SOURCE_NPZ = SCRIPT_DIR / 'expB2_mc.npz'
SOURCE_CSV = SCRIPT_DIR / 'expB2_mc.csv'
RES_DIR = SCRIPT_DIR / 'supplement_results'
FIG_DIR = SCRIPT_DIR / 'supplement_figures'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda: f.read(1024*1024), b''):
            h.update(c)
    return h.hexdigest()


def rmse(a,b):
    return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))


def anchor_model(depth, anchor_depth, anchor_age):
    """Match original np.interp + nearest-interval-SR extrapolation."""
    local_sr=np.diff(anchor_depth)/np.diff(anchor_age)
    if np.any(local_sr<=0) or not np.all(np.isfinite(local_sr)):
        raise ValueError('Invalid interval sedimentation rates')
    t=np.interp(depth, anchor_depth, anchor_age)
    head=depth<anchor_depth[0]
    tail=depth>anchor_depth[-1]
    t[head]=anchor_age[0]-(anchor_depth[0]-depth[head])/local_sr[0]
    t[tail]=anchor_age[-1]+(depth[tail]-anchor_depth[-1])/local_sr[-1]
    return t


def get_anchors(depth, imf):
    prom=PROMINENCE_FRACTION*float(np.std(imf))
    peaks, _=find_peaks(imf,prominence=prom)
    troughs, _=find_peaks(-imf,prominence=prom)
    if len(troughs)<2: raise ValueError('Insufficient trough anchors')
    return depth[troughs], int(max(len(peaks),len(troughs))-1)


def audited_inputs():
    if not SOURCE_NPZ.is_file() or not SOURCE_CSV.is_file():
        raise FileNotFoundError('Place expB2_mc.npz AND expB2_mc.csv from ORIGINAL AstroTuner-405kyr/code folder alongside this script. Do not use expB_results.npz instead.')
    z=np.load(SOURCE_NPZ, allow_pickle=False)
    old=pd.read_csv(SOURCE_CSV)
    assert len(old)==30 and not old[['scenario','seed']].duplicated().any(), 'Original B2 CSV must contain exactly 30 unique scenario-seed records'
    for name in SCENARIOS:
        subset=old[old.scenario==name]
        assert len(subset)==N_SEEDS, f'{name}: expected 10 runs'
    print('INPUT PASS: 30 archived B2 records, original NPZ and CSV present')
    return z,old


def compute(z,old,check_only=False):
    rows=[]; local=[]; failures=[]; checked=0
    for scenario in SCENARIOS:
        for k in range(N_SEEDS):
            tag=f'{scenario}_s{k:02d}'
            try:
                depth=z[f'{tag}__depth']; truth=z[f'{tag}__t_true']
                sr_true=z[f'{tag}__sr_true']; imfs=z[f'{tag}__imfs']
                ix=int(z[f'{tag}__sel_idx'][0]); imf=imfs[ix]
                if np.any(np.diff(depth)<=0) or np.any(np.diff(truth)<=0):
                    raise AssertionError('Nonmonotonic depth or true age')
                expected_seed=(2001 if scenario=='linear' else 2002 if scenario=='step' else 2003)+1000*k
                old_row=old[(old.scenario==scenario)&(old.seed==expected_seed)]
                if len(old_row)!=1 or ix!=int(old_row.iloc[0].selected_imf):
                    raise AssertionError('Archived selected IMF differs from original CSV')
                orig_depth, ncyc=get_anchors(depth,imf)
                if len(orig_depth)!=int(old_row.iloc[0].n_anchors) or ncyc!=int(old_row.iloc[0].n_cycles):
                    raise AssertionError('Original anchor counts do not reproduce')
                orig_age=np.arange(len(orig_depth))*PERIOD_MYR  # original B2 uses ref_age_fn=None
                orig_t=anchor_model(depth,orig_depth,orig_age)
                original_rmse=rmse(orig_t,truth)
                if not np.isclose(original_rmse,float(old_row.iloc[0].rmse_anchor),atol=1e-8,rtol=1e-9):
                    raise AssertionError(f'Baseline RMSE mismatch: recomputed {original_rmse}, original {old_row.iloc[0].rmse_anchor}')
                checked+=1
                if check_only: continue
                for factor in FACTORS:
                    indices=np.arange(0,len(orig_depth),factor,dtype=int)
                    if indices[-1]!=len(orig_depth)-1:
                        indices=np.r_[indices,len(orig_depth)-1]
                    ad=orig_depth[indices]
                    aa=orig_age[indices]  # DO NOT compress time: preserve missed cycles and original epochs
                    model=anchor_model(depth,ad,aa)
                    segment_d=np.diff(ad)
                    segment_t=np.diff(aa)
                    estimated_sr=segment_d/segment_t
                    # True EFFECTIVE SR across each identical interval (not SR at a single point)
                    true_at_anchors=np.interp(ad,depth,truth)
                    true_eff_sr=segment_d/np.diff(true_at_anchors)
                    abs_pct=np.abs(estimated_sr-true_eff_sr)/true_eff_sr*100.0
                    signed_pct=(estimated_sr-true_eff_sr)/true_eff_sr*100.0
                    absolute_sr_err=np.abs(estimated_sr-true_eff_sr)
                    widths=segment_d
                    anchor_offset=float(truth[np.searchsorted(depth,ad[0])]-aa[0])
                    # ORACLE-offset diagnostic: NOT deployable, only isolates first-anchor origin error
                    offset_corrected=rmse(model+anchor_offset,truth)
                    for j in range(len(segment_d)):
                        local.append(dict(scenario=scenario,seed=expected_seed,factor=factor,
                            interval_index=j,depth_start_m=ad[j],depth_end_m=ad[j+1],
                            interval_width_m=segment_d[j],nominal_span_myr=segment_t[j],
                            sr_est_m_per_myr=estimated_sr[j],sr_true_effective_m_per_myr=true_eff_sr[j],
                            signed_pct_sr_error=signed_pct[j],abs_pct_sr_error=abs_pct[j],
                            abs_sr_error_m_per_myr=absolute_sr_err[j]))
                    rows.append(dict(scenario=scenario,seed=expected_seed,factor=factor,
                        original_gate_accepted=bool(old_row.iloc[0].matched),
                        n_original_anchors=len(orig_depth),n_retained_anchors=len(ad),
                        n_intervals=len(segment_d),median_width_m=float(np.median(widths)),
                        mean_width_m=float(np.mean(widths)),
                        median_nominal_span_myr=float(np.median(segment_t)),
                        median_abs_pct_sr_error=float(np.median(abs_pct)),
                        mean_abs_pct_sr_error=float(np.mean(abs_pct)),
                        p90_abs_pct_sr_error=float(np.percentile(abs_pct,90)),
                        rmse_age_myr=rmse(model,truth),
                        max_abs_age_error_myr=float(np.max(np.abs(model-truth))),
                        offset_corrected_rmse_myr=offset_corrected,
                        original_const_rmse_myr=float(old_row.iloc[0].rmse_const),
                        original_anchor_rmse_myr=original_rmse,
                        age_origin_offset_myr=anchor_offset))
            except Exception as e:
                failures.append(f'{tag}: {type(e).__name__}: {e}')
    if failures:
        raise RuntimeError('\n'.join(failures))
    assert checked==30, f'Only {checked} of 30 old records reproduced'
    print('REPRODUCIBILITY PASS: all 30 K=1 anchored models reproduce original B2 RMSE, anchor count, and IMF selections')
    if check_only:
        print('CHECK ONLY PASS: no Experiment G files written')
        return None,None
    result=pd.DataFrame(rows)
    interval=pd.DataFrame(local)
    assert len(result)==120
    assert not result[['scenario','seed','factor']].duplicated().any()
    return result,interval


def make_plot(runs, intervals, path_png,path_pdf):
    """All legends outside axes, in a dedicated lower region."""
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                         'pdf.fonttype':42,'ps.fonttype':42})
    fig, ax=plt.subplots(2,2,figsize=(13.5,9.8))
    fig.subplots_adjust(left=.085,right=.977,bottom=.19,top=.94,wspace=.25,hspace=.38)
    colors={'linear':'#3c6382','step':'#b16451','sine':'#6c8262'}
    markers={'linear':'o','step':'s','sine':'D'}
    from matplotlib.lines import Line2D
    for sc in SCENARIOS:
        sub=runs[runs.scenario==sc]
        med=sub.groupby('factor').median(numeric_only=True).reindex(FACTORS)
        c=colors[sc]
        # a, interval SR error: per-realization median, and between-run median
        ax[0,0].scatter(sub.factor+({'linear':-.07,'step':0,'sine':.07}[sc]),
                        sub.median_abs_pct_sr_error,alpha=.25,s=19,color=c,marker=markers[sc])
        ax[0,0].plot(FACTORS,med.median_abs_pct_sr_error,marker=markers[sc],color=c,lw=2)
        # b, absolute RMSE against ground truth, no wins metric
        ax[0,1].scatter(sub.factor+({'linear':-.07,'step':0,'sine':.07}[sc]),
                        sub.rmse_age_myr,alpha=.25,s=19,color=c,marker=markers[sc])
        ax[0,1].plot(FACTORS,med.rmse_age_myr,marker=markers[sc],color=c,lw=2)
        # d: widths
        ax[1,1].scatter(sub.factor+({'linear':-.07,'step':0,'sine':.07}[sc]),
                        sub.median_width_m,alpha=.20,s=19,color=c,marker=markers[sc])
        ax[1,1].plot(FACTORS,med.median_width_m,marker=markers[sc],color=c,lw=2)
    ax[0,0].set(title='(a) Local sedimentation-rate error',ylabel='Median absolute SR error per run (%)',xlabel='Aggregation factor (original 405-kyr intervals)')
    ax[0,1].set(title='(b) Absolute chronological error',ylabel='Age-model RMSE (Myr)',xlabel='Aggregation factor (original 405-kyr intervals)')
    ax[1,1].set(title='(d) Loss of depth resolution',ylabel='Median retained interval thickness (m)',xlabel='Aggregation factor (original 405-kyr intervals)')
    # c: SR conditional bias across intervals, by scenario, using fixed original K=1 and coarse K=8.
    a=ax[1,0]
    for sc in SCENARIOS:
        for factor,alpha,style in [(1,.2,'o'),(8,.60,'^')]:
            data=intervals[(intervals.scenario==sc)&(intervals.factor==factor)]
            # deterministic full population, no selection of favourable cases
            a.scatter(data.sr_true_effective_m_per_myr,data.signed_pct_sr_error,
                s=8 if factor==1 else 17,alpha=alpha,marker=style,color=colors[sc],edgecolors='none')
    a.set_yscale('symlog',linthresh=40,linscale=1.1)
    a.axhline(0,color='gray',lw=1,ls='--')
    a.set(title='(c) Local SR bias vs true effective SR',ylabel='Signed SR error (%)',xlabel='True interval-effective SR (m/Myr)')
    for a in ax.flat:
        a.grid(alpha=.16)
        if a is not ax[1,0]: a.set_xticks(FACTORS)
    handles=[Line2D([0],[0],color=colors[s],marker=markers[s],label={'linear':'Linear drift','step':'Step change','sine':'Sinusoidal SR'}[s],lw=2) for s in SCENARIOS]
    handles.extend([Line2D([0],[0],color='#707070',lw=0,marker='o',markersize=5,alpha=.4,label='Run / interval'),
                    Line2D([0],[0],color='#707070',lw=2,marker='o',markersize=6,label='Across-run median'),
                    Line2D([0],[0],color='#707070',lw=0,marker='^',label='K=8 in panel (c)')])
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.06),ncol=3,
        frameon=False,fontsize=10,columnspacing=2.3,handlelength=2.4)
    fig.text(.5,.018,'Fixed archived B2 realizations (3 SR regimes x 10 seeds); coarsening preserves original anchor ages and omitted-cycle bias.',
        ha='center',fontsize=8,color='#555555')
    fig.savefig(path_png,dpi=300,facecolor='white')
    fig.savefig(path_pdf,facecolor='white')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    z,old=audited_inputs()
    results,interval=compute(z,old,check_only=args.check_only)
    z.close()
    if args.check_only: return
    RES_DIR.mkdir(exist_ok=True);FIG_DIR.mkdir(exist_ok=True)
    summary=results.groupby(['scenario','factor'],as_index=False).agg(
        n=('seed','size'),median_sr_pct_err=('median_abs_pct_sr_error','median'),
        mean_sr_pct_err=('mean_abs_pct_sr_error','mean'),
        median_rmse_myr=('rmse_age_myr','median'),
        median_max_abs_age_err_myr=('max_abs_age_error_myr','median'),
        median_offset_corrected_rmse_myr=('offset_corrected_rmse_myr','median'),
        median_retained_anchors=('n_retained_anchors','median'),
        median_interval_width_m=('median_width_m','median'))
    files={'runs':RES_DIR/'expG_anchor_spacing_all120.csv',
           'intervals':RES_DIR/'expG_anchor_spacing_intervals.csv',
           'summary':RES_DIR/'expG_anchor_spacing_summary.csv',
           'design':RES_DIR/'expG_anchor_spacing_design.json',
           'figure_png':FIG_DIR/'expG_anchor_spacing_2x2.png',
           'figure_pdf':FIG_DIR/'expG_anchor_spacing_2x2.pdf'}
    results.to_csv(files['runs'],index=False)
    interval.to_csv(files['intervals'],index=False)
    summary.to_csv(files['summary'],index=False)
    design=dict(input_npz=SOURCE_NPZ.name,input_csv=SOURCE_CSV.name,
        input_npz_sha256=sha256(SOURCE_NPZ),input_csv_sha256=sha256(SOURCE_CSV),
        aggregation_factors=FACTORS,scenario_names=SCENARIOS,
        original_realizations=30,derived_records=120,
        nominal_period_myr=PERIOD_MYR,
        reference_age_fn='None in original exp_b2_montecarlo.py; first IMF trough assigned age zero',
        retained_anchor_policy='indices 0,K,2K,... plus final original anchor; preserve index*0.405 ages',
        note='Post-processing resolution sensitivity only. Coarsening does not correct missed orbital cycles or change original EMD, SR scan, IMF selection, gate logic or input data.',
        interval_true_sr='depth span / original true age span',
        oracle_offset='Offset-corrected RMSE uses true age at first anchor ONLY as diagnostic; not deployment result')
    files['design'].write_text(json.dumps(design,indent=2,ensure_ascii=False),encoding='utf-8')
    make_plot(results,interval,files['figure_png'],files['figure_pdf'])
    print('\nSUMMARY (review before manuscript use):\n'+summary.to_string(index=False))
    for k,p in files.items():print('Saved',k,':',p)


if __name__=='__main__': main()
