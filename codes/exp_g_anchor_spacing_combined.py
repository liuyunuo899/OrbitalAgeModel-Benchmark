# -*- coding: utf-8 -*-
"""Experiment G — integrated anchor aggregation, no-truth anchor reselection and bias diagnostics.

Uses ORIGINAL archived Experiment B2 output (no new orbital signal, noise,
EMD or IMF selection). Predefined aggregation factors K=(1,2,4,8).
The first and final anchors are retained and intermediate anchors are sampled
at the original-index stride K. Retained ages preserve the ORIGINAL number of
405-kyr intervals, i.e. (index difference)*0.405 Myr, not a new 405-kyr period.

IMPORTANT: This compares fixed-stride thinning against a pre-specified no-truth
IMF trough prominence reselection rule. It is NOT fresh EMD re-extraction,
orbital-period re-identification, or a validated optimized age-model method.

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




def prominent_reselection_indices(imf, depth, orig_anchor_depth, factor):
    """No-ground-truth reselection, fixed BEFORE inspecting outcomes.

    Divide original interior troughs into disjoint blocks of K consecutive troughs;
    choose the most-prominent trough in each block (tie -> earliest). The first and
    last original troughs are protected to preserve the original anchoring domain.
    Retained ages use ORIGINAL trough indices *0.405 Myr; never fabricate time.
    This cannot fix a missing cycle: it changes WHICH original trough is kept.
    """
    from scipy.signal import peak_prominences
    if factor==1:
        return np.arange(len(orig_anchor_depth),dtype=int)
    _, alltroughs = None, find_peaks(-imf,prominence=PROMINENCE_FRACTION*float(np.std(imf)))[0]
    if len(alltroughs)!=len(orig_anchor_depth) or not np.allclose(depth[alltroughs],orig_anchor_depth):
        raise AssertionError('Trough set differs from audited baseline')
    prominence = peak_prominences(-imf,alltroughs)[0]
    n=len(alltroughs)
    ids=[0]
    # Disjoint blocks; final original endpoint protected
    for start in range(1,n-1,factor):
        choices=np.arange(start,min(start+factor,n-1))
        if not len(choices): continue
        picked=int(choices[np.argmax(prominence[choices])])
        if picked>ids[-1]: ids.append(picked)
    if ids[-1]!=n-1:ids.append(n-1)
    ids=np.asarray(ids,dtype=int)
    if np.any(np.diff(ids)<=0):raise AssertionError('Nonincreasing selected trough indices')
    return ids


def interval_summaries(ad,aa,depth,truth):
    sr_est=np.diff(ad)/np.diff(aa)
    true_t=np.interp(ad,depth,truth)
    sr_eff=np.diff(ad)/np.diff(true_t)
    signed=(sr_est-sr_eff)/sr_eff*100
    return dict(median_abs_pct_sr_error=float(np.median(np.abs(signed))),
                median_signed_pct_sr_error=float(np.median(signed)),
                p90_abs_pct_sr_error=float(np.percentile(np.abs(signed),90)),
                median_width_m=float(np.median(np.diff(ad))),
                median_true_sr_m_per_myr=float(np.median(sr_eff))),signed,sr_eff


def integrate(z,old):
    fixed,intervals=compute(z,old)
    records=[]; local=[]
    for row in fixed.itertuples():
        records.append(dict(scenario=row.scenario,seed=row.seed,factor=row.factor,
            strategy='fixed_stride',rmse_age_myr=row.rmse_age_myr,
            median_abs_pct_sr_error=row.median_abs_pct_sr_error,
            median_width_m=row.median_width_m,n_retained_anchors=row.n_retained_anchors,
            max_abs_age_error_myr=row.max_abs_age_error_myr,
            offset_corrected_rmse_myr=row.offset_corrected_rmse_myr,
            age_origin_offset_myr=row.age_origin_offset_myr))
    for sc in SCENARIOS:
      for k in range(N_SEEDS):
        tag=f'{sc}_s{k:02d}'
        seed=(2001 if sc=='linear' else 2002 if sc=='step' else 2003)+1000*k
        depth=z[f'{tag}__depth'];truth=z[f'{tag}__t_true'];imfs=z[f'{tag}__imfs']
        imf=imfs[int(z[f'{tag}__sel_idx'][0])]
        anchor_depth,_=get_anchors(depth,imf)
        for factor in FACTORS:
            ids=prominent_reselection_indices(imf,depth,anchor_depth,factor)
            ad=anchor_depth[ids];aa=ids*PERIOD_MYR
            model=anchor_model(depth,ad,aa)
            summary,signed,sr_eff=interval_summaries(ad,aa,depth,truth)
            offset=float(np.interp(ad[0],depth,truth)-aa[0])
            if factor==1:
                orig=fixed[(fixed.scenario==sc)&(fixed.seed==seed)&(fixed.factor==1)].iloc[0]
                if not np.isclose(rmse(model,truth),orig.rmse_age_myr,atol=1e-8):
                    raise AssertionError('Reselection K=1 does not reproduce baseline')
            records.append(dict(scenario=sc,seed=seed,factor=factor,strategy='prominence_reselect',
                rmse_age_myr=rmse(model,truth),
                median_abs_pct_sr_error=summary['median_abs_pct_sr_error'],
                median_width_m=summary['median_width_m'],n_retained_anchors=len(ids),
                max_abs_age_error_myr=float(np.max(np.abs(model-truth))),
                offset_corrected_rmse_myr=rmse(model+offset,truth),
                age_origin_offset_myr=offset))
            for j in range(len(ids)-1):
                local.append(dict(scenario=sc,seed=seed,factor=factor,strategy='prominence_reselect',
                    original_index_start=int(ids[j]),original_index_end=int(ids[j+1]),
                    depth_start_m=float(ad[j]),depth_end_m=float(ad[j+1]),
                    nominal_span_myr=float(aa[j+1]-aa[j]),
                    true_sr_effective_m_per_myr=float(sr_eff[j]),
                    signed_pct_sr_error=float(signed[j])))
    df=pd.DataFrame(records)
    if len(df)!=240 or df[['scenario','seed','factor','strategy']].duplicated().any():
        raise AssertionError('Expected 240 unique paired strategy×factor evaluations')
    # No selection according to true age. The following is evaluation only.
    s=df.groupby(['scenario','strategy','factor'],as_index=False).agg(
       n=('seed','size'),median_rmse_myr=('rmse_age_myr','median'),
       median_local_sr_error_pct=('median_abs_pct_sr_error','median'),
       median_interval_thickness_m=('median_width_m','median'),
       median_max_abs_age_error_myr=('max_abs_age_error_myr','median'),
       median_offset_corrected_rmse_myr=('offset_corrected_rmse_myr','median'),
       median_origin_offset_myr=('age_origin_offset_myr','median'))
    original=intervals.copy();original['strategy']='fixed_stride'
    return df,pd.concat([original,pd.DataFrame(local)],ignore_index=True,sort=False),s


def plot_integrated(df, path):
    # 2x2 fixed-vs-reselect medians; legends below axes, no occlusion
    from matplotlib.lines import Line2D
    colors={'linear':'#3c6382','step':'#b16451','sine':'#6c8262'}
    scnames={'linear':'Linear drift','step':'Step change','sine':'Sinusoidal SR'}
    metrics=[('median_abs_pct_sr_error','(a) Local SR error','Median absolute local SR error (%)'),
             ('rmse_age_myr','(b) Age-model accuracy','Age-model RMSE (Myr)'),
             ('median_width_m','(c) Depth resolution','Median interval thickness (m)'),
             ('offset_corrected_rmse_myr','(d) Age-origin diagnostic (oracle only)','Origin-corrected RMSE (Myr)')]
    fig,ax=plt.subplots(2,2,figsize=(13,9))
    fig.subplots_adjust(left=.09,right=.98,top=.94,bottom=.19,wspace=.25,hspace=.37)
    for a,(metric,title,ylabel) in zip(ax.flat,metrics):
      for sc in SCENARIOS:
        for st,style in [('fixed_stride','-'),('prominence_reselect','--')]:
            sub=df[(df.scenario==sc)&(df.strategy==st)].groupby('factor')[metric].median().reindex(FACTORS)
            a.plot(FACTORS,sub.values,color=colors[sc],ls=style,marker='o' if st=='fixed_stride' else 's',lw=1.6)
      a.set(title=title,xlabel='Aggregation factor K',ylabel=ylabel,xticks=FACTORS)
      a.grid(alpha=.2)
    legend=[Line2D([0],[0],color=colors[sc],lw=2,label=scnames[sc]) for sc in SCENARIOS]
    legend += [Line2D([0],[0],color='#555',ls='-',marker='o',label='Fixed stride'),
               Line2D([0],[0],color='#555',ls='--',marker='s',label='Prominence reselection')]
    fig.legend(handles=legend,loc='lower center',bbox_to_anchor=(.5,.045),ncol=3,frameon=False)
    fig.text(.5,.015,'Same 30 archived B2 runs; no truth used in reselection. Panel (d) uses true age ONLY as a non-deployable diagnostic.',ha='center',fontsize=8)
    fig.savefig(path,dpi=220,facecolor='white');plt.close(fig)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--check-only',action='store_true')
    args=ap.parse_args()
    z,old=audited_inputs()
    if args.check_only:
        compute(z,old,check_only=True);z.close();return
    df,interval,summary=integrate(z,old)
    z.close()
    RES_DIR.mkdir(exist_ok=True);FIG_DIR.mkdir(exist_ok=True)
    targets=[RES_DIR/'expG_integrated_all240.csv',RES_DIR/'expG_integrated_intervals.csv',
             RES_DIR/'expG_integrated_summary.csv',FIG_DIR/'expG_integrated_2x2.png',
             RES_DIR/'expG_integrated_design.json']
    # Refuse overwrites, even of previous integrated run; original A-F and G remain untouched.
    for t in targets:
        if t.exists():raise FileExistsError(f'Refusing to overwrite {t}. Archive/move old integrated output first.')
    df.to_csv(targets[0],index=False);interval.to_csv(targets[1],index=False)
    summary.to_csv(targets[2],index=False)
    plot_integrated(df,targets[3])
    design=dict(n_original_realizations=30,n_derived_evaluations=240,
         aggregation_factors=list(FACTORS),strategies=['fixed_stride','prominence_reselect'],
         reselection_rule='Select largest negative-IMF trough prominence within consecutive non-overlapping blocks of K original interior troughs; preserve endpoints; no true age used.',
         anchor_age_rule='original trough index times 0.405 Myr; preserved even if representative index moves',
         caution='Postprocessing reselection is not independent EMD rerun or detection of missing orbital cycles; corrected age-origin RMSE is oracle-only.',
         source_files={str(SOURCE_NPZ.name):sha256(SOURCE_NPZ),str(SOURCE_CSV.name):sha256(SOURCE_CSV)})
    targets[4].write_text(json.dumps(design,indent=2,ensure_ascii=False),encoding='utf-8')
    print('PASS: original 30 reproduced; 240 paired model evaluations; outputs:',*[str(t) for t in targets],sep='\n')
    print(summary.to_string(index=False))
if __name__=='__main__': main()
