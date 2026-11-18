#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment F: La2004-driven synthetic sensitivity study (Reviewer 1 Q3).

Run alongside astro_py_tuner.py and astronomical_data/INSOLN.LA2004.BTL.ASC,
or alongside AstroTuner-405kyr.zip with that data file in astronomical_data/.

    python exp_f_astronomical_solution.py --check-only
    python exp_f_astronomical_solution.py

Original A-E experiments, their scripts, and the published AstroTuner code are
never modified. Results have separate expF_ filenames.

IMPORTANT: La2004 contains orbital PARAMETERS, NOT sedimentary observations.
Eccentricity, obliquity and climatic-precession proxy are standardized and
combined with prescribed weights. These are response-model assumptions.
A 1-kyr source series is interpolated to a 0.5-kyr noise grid; the interpolation
DOES NOT add astronomical information. Three predeclared treatments:
  ideal_four_sines: 405/95/40/21 kyr at amplitudes 6/4/2.5/1.5
  la2004_ecc_dominant: standardized e, eps, e*sin(pibar) at weights 6/2.5/1.5
  la2004_balanced: standardized e, eps, e*sin(pibar at weights 1/1/1
All clean composite carriers are normalized to the same RMS as the original
four-sine benchmark, so the prescribed noise level is comparable.

The scalar e*sin(pibar) is a transparent climatic-precession proxy convention;
sign conventions differ in the literature and do not affect its period content.
The 40-Myr input window (42->2 Ma) is deliberately fixed, not optimized.
No claim that an astronomical solution alone reproduces depositional filtering,
proxy-system response, orbital modulation in sediments, or actual instrument noise.

NOTE on science: the balanced setting changes combined component amplitudes;
it does not replicate a measured stratigraphic proxy response. All cases and
failures are saved. Forced 405-kyr anchors are diagnostic, not deployed models
when acceptance gates reject them.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import traceback
import numpy as np
import pandas as pd
from scipy.signal import lfilter, periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
DATA_NAME = 'INSOLN.LA2004.BTL.ASC'
WINDOW_OLD_MA = 42.0
WINDOW_YOUNG_MA = 2.0
LENGTH_M = 600.0
DZ_M = 0.1
SR_TRUE = 15.0
DT_MYR = 0.0005
NOISE_RATIO = 0.5
RHO = 0.7
PERIODS_KYR = (405.,95.,40.,21.)
AMPS = (6.,4.,2.5,1.5)
SEEDS = tuple(2001 + 1000*k for k in range(10))
CASES = ('ideal_four_sines','la2004_ecc_dominant','la2004_balanced')
WEIGHTS = {
  'la2004_ecc_dominant': (6.0,2.5,1.5),
  'la2004_balanced': (1.0,1.0,1.0),
}
SCAN_GRID = np.arange(5.0,30.000001,0.25)
STAGE1_GATE = 0.5
BASELINE_RMS = float(np.sqrt(np.sum(np.square(AMPS))/2.))
RDIR = BASE / 'supplement_results'
FDIR = BASE / 'supplement_figures'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def find_data():
    choices=(BASE/'astronomical_data'/DATA_NAME, BASE/DATA_NAME,
             BASE.parent/'astronomical_data'/DATA_NAME, BASE.parent/DATA_NAME)
    for p in choices:
        if p.is_file(): return p
    raise FileNotFoundError(f'Missing {DATA_NAME}; put it into astronomical_data/ beside this script. ')


def import_tuner():
    locations=(BASE,BASE/'code',BASE/'AstroTuner-405kyr'/'code',
               BASE.parent/'AstroTuner-405kyr'/'code')
    for loc in locations:
        if (loc/'astro_py_tuner.py').is_file():
            sys.path.insert(0,str(loc))
            from astro_py_tuner import AstroTuner
            return AstroTuner,str(loc/'astro_py_tuner.py')
    for z in (BASE/'AstroTuner-405kyr.zip',BASE.parent/'AstroTuner-405kyr.zip'):
        if z.is_file():
            with zipfile.ZipFile(z) as arc:
                if 'AstroTuner-405kyr/code/astro_py_tuner.py' not in arc.namelist(): continue
            sys.path.insert(0,str(z)+'/AstroTuner-405kyr/code')
            from astro_py_tuner import AstroTuner
            return AstroTuner,str(z)+'::AstroTuner-405kyr/code/astro_py_tuner.py'
    raise FileNotFoundError('Missing astro_py_tuner.py / AstroTuner-405kyr.zip beside script.')


def read_la2004(path):
    # Fortran D exponents, exactly four columns; official README: t in kyr
    x=np.loadtxt(path,encoding='utf-8',converters={j:lambda s:float(s.replace('D','E').replace('d','E')) for j in (1,2,3)})
    if x.ndim!=2 or x.shape[1]!=4: raise ValueError(f'Expected 4 columns, got {x.shape}')
    if not np.isfinite(x).all(): raise ValueError('Nonfinite astronomical data')
    if not np.allclose(np.diff(x[:,0]),-1.,atol=1e-9):
        raise ValueError('Expected monotonically decreasing time at 1 kyr spacing')
    if not (np.isclose(x[0,0],0) and x[-1,0]<=-42000):
        raise ValueError('Unexpected La2004 time coverage')
    # Select fixed window from -42 to -2 Myr. Orient forward from older to younger.
    keep=(x[:,0]>=-WINDOW_OLD_MA*1000)&(x[:,0]<=-WINDOW_YOUNG_MA*1000)
    w=x[keep][::-1].copy()
    assert len(w)==40001 and np.isclose(w[0,0],-42000) and np.isclose(w[-1,0],-2000)
    t=(w[:,0]+WINDOW_OLD_MA*1000)/1000.0  # 0..40 Myr
    return t,w[:,1],w[:,2],w[:,3],x.shape[0]


def standardized(x):
    a=np.asarray(x,dtype=float);a=a-np.mean(a)
    s=np.std(a)
    if s<=0:raise ValueError('Zero-variance astronomical component')
    return a/s


def make_clean_inputs(t_src,e,eps,pibar):
    t_fine=np.arange(0.0, (WINDOW_OLD_MA-WINDOW_YOUNG_MA)+DT_MYR*.1,DT_MYR)
    # The original ideal carrier uses 0..40 Myr from the same start t=0.
    ideal=sum(a*np.sin(2*np.pi*t_fine/(p/1000.)) for p,a in zip(PERIODS_KYR,AMPS))
    outputs={'ideal_four_sines':ideal}
    e_z=standardized(e)
    ob_z=standardized(eps)
    pre_z=standardized(e*np.sin(pibar))
    components=(e_z,ob_z,pre_z)
    for name,weights in WEIGHTS.items():
        signal=sum(w*x for w,x in zip(weights,components))
        signal=standardized(signal)*BASELINE_RMS
        outputs[name]=np.interp(t_fine,t_src,signal)
    return t_fine,outputs,dict(eccentricity=e_z,obliquity=ob_z,climatic_precession=pre_z)


def make_noise(seed,n):
    rng=np.random.default_rng(int(seed))
    white1=rng.normal(0,1,n)
    ar1=lfilter([1.],[1.,-RHO],white1)
    ar1=ar1/np.std(ar1)
    white2=rng.normal(0,1,n)
    return NOISE_RATIO*BASELINE_RMS*(ar1+white2)/np.sqrt(2.)


def arrays_on_depth(fine_t,carrier,noise):
    d=np.arange(0.0,LENGTH_M+DZ_M/2.,DZ_M)
    true=d/SR_TRUE
    x=np.interp(true,fine_t,carrier+noise)
    return d,true,x


def errors(pred,truth):
    a=np.asarray(pred);b=np.asarray(truth)
    if a.shape!=b.shape:raise ValueError('Age model shape mismatch')
    dif=a-b
    return {'rmse_myr':float(np.sqrt(np.mean(dif**2))),
            'max_abs_myr':float(np.max(np.abs(dif))),
            'end_bias_myr':float(dif[-1])}


def get_spectral_periods(x,dt):
    # scipy periodogram on time-domain signals; uniform grid and mean removed.
    freq,pow=periodogram(np.asarray(x),fs=1./dt,detrend='linear',window='hann',scaling='spectrum')
    mask=(freq>0)&(1./freq>=0.015)&(1./freq<=0.65)
    return 1./freq[mask]*1000.,pow[mask]


def data_smoke(data):
    t,e,eps,pibar,n=data
    assert n==51001
    assert np.isclose(t[0],0.) and np.isclose(t[-1],40.)
    fine,clean,parts=make_clean_inputs(t,e,eps,pibar)
    assert fine.size>79000
    assert set(clean)==set(CASES)
    assert np.all(np.isfinite([c.std() for c in clean.values()]))
    for name in CASES:
        assert np.isclose(np.std(clean[name]),BASELINE_RMS,rtol=0.02),name
    z=make_noise(SEEDS[0],len(fine))
    assert np.array_equal(z,make_noise(SEEDS[0],len(fine)))
    d,truth,x=arrays_on_depth(fine,clean['la2004_balanced'],z)
    assert np.isclose(d[-1],600.) and np.isclose(truth[-1],40.) and len(x)==6001
    print('DATA CHECK PASS: 51,001 La2004 rows; 42–2 Ma window; 1-kyr source step;'
          ' valid orbital components; deterministic paired noise; 6001 depth samples.')
    return fine,clean,parts


def run_models(fine,clean,source_path,tuner_source):
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise SystemExit('The original PyEMD is missing. Use your existing paper environment '
                         '(install package EMD-signal if necessary). No alternative algorithm is substituted.') from exc
    AstroTuner,_=import_tuner()
    tuner=AstroTuner(sr_grid=SCAN_GRID,n_jobs=1,seed=42)
    rows=[]; npz={}; failures=[]
    for seed in SEEDS:
        noise=make_noise(seed,len(fine))
        for case in CASES:
            tag=f'{case}__seed{seed}'
            d,truth,x=arrays_on_depth(fine,clean[case],noise)
            npz[tag+'__signal']=x
            npz[tag+'__true_age']=truth
            row=dict(case=case,seed=seed,true_sr_m_per_myr=SR_TRUE,
                     true_duration_myr=40.,stage1_pass=False,imf_gate_pass=False,
                     gate_accepted=False,anchor_status='not_computed',
                     match_score=np.nan,scan_sr_m_per_myr=np.nan,
                     imf_index=np.nan,imf_cycles=np.nan,anchor_count=0,
                     constant_rmse_myr=np.nan,forced_anchor_rmse_myr=np.nan,
                     forced_minus_constant_rmse_myr=np.nan,forced_anchor_wins=False)
            try:
                scan=tuner.scan_sed_rates(d,x)
                sr,ms=tuner.best_sr(scan,depth=d,signal=x)
                linear=tuner.build_age_model(d,sr)
                const=errors(linear,truth)
                row.update(scan_sr_m_per_myr=float(sr),match_score=float(ms),
                           stage1_pass=bool(ms>=STAGE1_GATE),
                           constant_rmse_myr=const['rmse_myr'],
                           constant_end_bias_myr=const['end_bias_myr'])
                npz[tag+'__constant_age']=linear
            except Exception as exc:
                row['anchor_status']='scan_error'
                failures.append(f'{tag}: scan: {type(exc).__name__}: {exc}\n{traceback.format_exc()}')
                rows.append(row);print(f'{tag}: SCAN ERROR: {exc}')
                continue
            try:
                imfs=EMD().emd(x,max_imf=10)
                if not len(imfs):raise ValueError('No IMFs returned')
                selection=tuner.select_405_imf(imfs,d,sr_est=sr)
                anchor,table=tuner.build_age_model_405(d,imfs[selection['imf_index']],anchor_phase='trough')
                fa=errors(anchor,truth)
                row.update(imf_index=int(selection['imf_index'])+1,
                           imf_cycles=int(selection['n_cycles']),
                           imf_gate_pass=bool(selection['matched']),
                           gate_accepted=bool(ms>=STAGE1_GATE and selection['matched']),
                           anchor_count=len(table),anchor_status='forced_diagnostic_only',
                           forced_anchor_rmse_myr=fa['rmse_myr'],
                           forced_anchor_end_bias_myr=fa['end_bias_myr'],
                           forced_minus_constant_rmse_myr=fa['rmse_myr']-const['rmse_myr'],
                           forced_anchor_wins=bool(fa['rmse_myr']<const['rmse_myr']))
                npz[tag+'__forced_anchor_age']=anchor
            except Exception as exc:
                row['anchor_status']='anchor_error'
                failures.append(f'{tag}: anchor: {type(exc).__name__}: {exc}\n{traceback.format_exc()}')
            rows.append(row)
            print(f'{tag}: match={ms:.3f}, SR={sr:.2f}, gate={row["gate_accepted"]}, '
                  f'constant={row["constant_rmse_myr"]:.3f}, '
                  f'forced={row["forced_anchor_rmse_myr"]:.3f} Myr')
    df=pd.DataFrame(rows)
    assert len(df)==len(CASES)*len(SEEDS)
    assert not df.duplicated(['case','seed']).any()
    return df,npz,failures


def draw_figure(fine,clean,df):
    # All legends placed outside axes, in a dedicated bottom margin. 2x2 panels.
    palette={'ideal_four_sines':'#A65A4A','la2004_ecc_dominant':'#41627E',
             'la2004_balanced':'#6E8B67'}
    labels={'ideal_four_sines':'Four-sine control',
            'la2004_ecc_dominant':'La2004: eccentricity-dominant',
            'la2004_balanced':'La2004: balanced'}
    fig,axs=plt.subplots(2,2,figsize=(13.5,9.5))
    a,b,c,d=axs.flat
    for ax in axs.flat:
        ax.grid(axis='y',alpha=.30,color='#9EAEB9',lw=.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    # Panel a uses an illustrative 1-Myr excerpt, not the entire 40-Myr trace.
    show=(fine>=10.)&(fine<=11.)
    for case in CASES:
        a.plot(fine[show],clean[case][show],color=palette[case],lw=.9,alpha=.9)
    a.set_title('(a) Clean synthetic astronomical carriers (10–11 Myr excerpt)',loc='left',fontsize=11)
    a.set_xlabel('Time since 42 Ma (Myr)');a.set_ylabel('Normalized proxy (a.u.)')
    for case in CASES:
        p,ps=get_spectral_periods(clean[case],DT_MYR)
        ii=np.argsort(p)
        b.plot(p[ii],ps[ii],color=palette[case],lw=1.05,alpha=.9)
    for per in (21.,41.,95.,120.,405.):
        b.axvline(per,color='#9BA7B0',lw=.7,ls=':',alpha=.7)
    b.set_xlim(15,450);b.set_yscale('log')
    b.set_title('(b) Power spectra of noise-free synthetic inputs',loc='left',fontsize=11)
    b.set_xlabel('Period (kyr)');b.set_ylabel('Power (a.u., log)')
    order=range(len(CASES)); xx=np.arange(len(CASES))
    gate=df.groupby('case',sort=False)['gate_accepted'].sum().reindex(CASES).to_numpy()
    s1=df.groupby('case',sort=False)['stage1_pass'].sum().reindex(CASES).to_numpy()
    c.bar(xx-.18,s1,.34,color='#708F70',label='Stage 1 pass')
    c.bar(xx+.18,gate,.34,color='#B06656',label='Both gates pass')
    c.set_xticks(xx,['Four-sine','La2004\necc-dominant','La2004\nbalanced'])
    c.set_ylim(0,11);c.set_ylabel('Count (out of 10)')
    c.set_title('(c) Original algorithm acceptance gates',loc='left',fontsize=11)
    for j,case in enumerate(CASES):
        dat=df[df.case==case]
        for k,(name,col,dx) in enumerate([('constant_rmse_myr','#B08C57',-.12),('forced_anchor_rmse_myr','#41627E',.12)]):
            vals=dat[name].to_numpy(dtype=float)
            vals=vals[np.isfinite(vals)]
            if len(vals):
                d.scatter(np.full(len(vals),j+dx),vals,color=col,s=24,alpha=.50)
                d.scatter(j+dx,np.median(vals),color=col,s=67,edgecolor='white',linewidth=.8,zorder=5)
    d.set_xticks(xx,['Four-sine','La2004\necc-dominant','La2004\nbalanced'])
    d.set_ylabel('Age RMSE (Myr)')
    d.set_title('(d) Age-model error (points: runs; large: median)',loc='left',fontsize=11)
    handles=[Line2D([],[],lw=2,color=palette[case],label=labels[case]) for case in CASES]
    handles += [Line2D([],[],color='#708F70',lw=6,label='Stage 1 passed'),
                Line2D([],[],color='#B06656',lw=6,label='Both gates passed'),
                Line2D([],[],marker='o',linestyle='None',color='#B08C57',label='Constant-SR RMSE'),
                Line2D([],[],marker='o',linestyle='None',color='#41627E',label='Forced-anchor RMSE')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.015),
               ncol=4,frameon=False,fontsize=9,handlelength=2.3,columnspacing=2)
    fig.subplots_adjust(left=.085,right=.98,top=.95,bottom=.185,wspace=.30,hspace=.49)
    fig.savefig(FDIR/'expF_astronomical_2x2.png',dpi=300,facecolor='white')
    fig.savefig(FDIR/'expF_astronomical_2x2.pdf',facecolor='white')
    plt.close(fig)


def main(check_only=False):
    source=find_data()
    t,e,eps,pibar,n=read_la2004(source)
    fine,clean,parts=data_smoke((t,e,eps,pibar,n))
    if check_only:
        print('CHECK ONLY passed: no EMD runs, output files, or original code modified.')
        return
    AstroTuner,tuner_source=import_tuner()
    RDIR.mkdir(parents=True,exist_ok=True);FDIR.mkdir(parents=True,exist_ok=True)
    df,npz,failures=run_models(fine,clean,source,tuner_source)
    summary=df.groupby('case',as_index=False).agg(
        n=('seed','size'),stage1_pass=('stage1_pass','sum'),
        both_gates_pass=('gate_accepted','sum'),
        forced_computed=('forced_anchor_rmse_myr','count'),
        median_const_rmse_myr=('constant_rmse_myr','median'),
        median_forced_rmse_myr=('forced_anchor_rmse_myr','median'),
        mean_forced_rmse_myr=('forced_anchor_rmse_myr','mean'),
        forced_wins=('forced_anchor_wins','sum'))
    df.to_csv(RDIR/'expF_all30.csv',index=False)
    summary.to_csv(RDIR/'expF_summary.csv',index=False)
    np.savez_compressed(RDIR/'expF_series.npz',**npz)
    (RDIR/'expF_errors.txt').write_text('\n'.join(failures) or 'No per-realization errors.',encoding='utf-8')
    metadata={'experiment':'F, La2004 astronomical forcing sensitivity',
              'source_file':source.name,'source_sha256':sha256(source),
              'source_rows':n,'window_ma_before_present':[WINDOW_OLD_MA,WINDOW_YOUNG_MA],
              'source_spacing_kyr':1.0,'working_interpolation_step_kyr':DT_MYR*1000,
              'depth_length_m':LENGTH_M,'depth_step_m':DZ_M,'true_sr_m_per_myr':SR_TRUE,
              'cases':list(CASES),'weights':WEIGHTS,'common_clean_rms':BASELINE_RMS,
              'noise_ratio':NOISE_RATIO,'AR1_rho':RHO,'seeds':list(SEEDS),
              'scan_grid_m_per_myr':[5,30,.25],'stage1_gate':STAGE1_GATE,
              'tuner_file':tuner_source,
              'limitations':['La2004 orbital parameters, not measured sediments',
                             'Normalized component weights are assumptions',
                             'Only fixed 42–2 Ma window and constant sedimentation rate',
                             'No hiatus or instrument response',
                             'Forced anchoring is diagnostic when gates reject',
                             'Interpolation of 1-kyr source data does not increase resolution']}
    (RDIR/'expF_design.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    draw_figure(fine,clean,df)
    print('\nSUMMARY (verify before manuscript use):\n'+summary.to_string(index=False))
    print('\nSaved:',RDIR/'expF_all30.csv')
    print('Saved:',RDIR/'expF_summary.csv')
    print('Saved:',FDIR/'expF_astronomical_2x2.png')
    print('Any errors:',len(failures))


if __name__=='__main__':
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument('--check-only',action='store_true',help='Read source and verify data before heavy model runs')
    opt=arg.parse_args()
    main(check_only=opt.check_only)
