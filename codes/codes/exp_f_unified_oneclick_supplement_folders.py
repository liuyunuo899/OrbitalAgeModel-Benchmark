#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment F unified one-click reproduction, 70 original evaluations.

HOW TO USE: Put THIS FILE beside astro_py_tuner.py in codes/ and place
  astronomical_data/INSOLN.LA2004.BTL.ASC
  astronomical_data/PT-ZB18a_1-1.dat
  astronomical_data/PT-ZB20a_1-1.dat
then OPEN this file in Spyder and press the GREEN RUN button / F5.
No console commands or other Experiment F scripts are required.

Default RUN_MODE='full': 10 seeds x 7 cases (one ideal four-sine control,
2 La2004 cases, 4 PT-ZB cases), use unchanged AstroTuner and PyEMD pipeline.
Outputs go into two matching timestamped subfolders under
  codes/supplement_results/ and codes/supplement_figures/, with:
  expF_unified_all70.csv, expF_unified_summary70.csv, expF_unified_series.npz,
  expF_unified_provenance.json, expF_unified_errors.txt,
  expF_integrated_La2004_ZB18a_ZB20a.png/.svg
  expF_original_four_sine_La2004_2x2.png/.pdf

RUN_MODE='check' is available for fast data validation without EMD.
Nothing overwrites historical original scripts, data, or experiment output.
The embedded implementation sections below preserve the original, validated
La2004, PT-ZB and integrated plotting functions. No fitting or cherry-picking.
"""
from __future__ import annotations

# EDIT HERE ONLY IF YOU NEED A DATA-ONLY CHECK:
RUN_MODE = 'full'   # 'full' or 'check'; default is full 70 evaluations

from pathlib import Path
from datetime import datetime
from types import ModuleType
import hashlib
import json
import uuid
import sys
import traceback
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RESULT_ROOT = BASE / 'supplement_results'
FIGURE_ROOT = BASE / 'supplement_figures'

# Original implementation copied verbatim into this single file. These are
# visible text sections, evaluated in isolated module namespaces so identical
# globals (CASES, WEIGHTS, BASE etc.) cannot shadow each other. The guard
# if __name__ == '__main__' in each embedded source is NOT executed.
def _load_embedded(name, source):
    module = ModuleType(name)
    module.__file__ = str(BASE / 'exp_f_unified_oneclick.py')
    module.__package__ = ''
    sys.modules[name] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


# --- BEGIN UNMODIFIED ORIGINAL EXPERIMENT F SOURCE ---
ORIGINAL_SOURCE = r'''
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

'''
# --- END ORIGINAL SOURCE ---

# --- BEGIN UNMODIFIED ZB SOURCE ---
ZB_SOURCE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent Experiment F extension: FULL ZB18a/ZB20a forcing comparison.

DATA SOURCES (place both in codes/astronomical_data):
  PT-ZB18a_1-1.dat
  PT-ZB20a_1-1.dat

Published, internally consistent precession-tilt solutions from
  https://github.com/japhir/paleoinsolation
  Kocken & Zeebe, New Insolation Forcing for Paleoclimate Studies,
  Paleoceanography and Paleoclimatology, DOI:10.1029/2025PA005287.

Six ASCII columns, as defined by upstream src/data.f90:
  0: time (kyr before present; negative in the past)
  1: eccentricity e (dimensionless)
  2: obliquity epsilon (radians)
  3: precession angle (radians)
  4: longitude of perihelion relative to moving equinox (radians)
  5: climatic precession e*sin(longitude_of_perihelion) (dimensionless)

Two independent full-solution inputs x two predeclared response weightings:
  ZB18a_ecc_dominant, ZB18a_balanced,
  ZB20a_ecc_dominant, ZB20a_balanced.
Each uses ALL THREE components (e, epsilon, climatic precession) from its OWN
corresponding ZB file. NO La2004 components are substituted.

Other model settings, seeds, algorithm and evaluation logic are taken from the
UNCHANGED original exp_f_astronomical_solution.py. Both response weightings
remain synthetic proxy assumptions, not observed sedimentary responses.

The file is read-only with respect to original scripts and source data. Outputs
are written only to fresh timestamped directories under expFZB_results/ and
expFZB_figures/ (never to supplement_results or supplement_figures).

Spyder IPython console:
  %runfile 'F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes/exp_f_zb_full_comparison.py' --args="--check-only"
  %runfile 'F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes/exp_f_zb_full_comparison.py' --args="--smoke"
  %runfile 'F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes/exp_f_zb_full_comparison.py' --args="--run"

Run --check-only first, then --smoke, then --run. --check-only writes nothing.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import traceback
import uuid

import numpy as np
import pandas as pd
from scipy.signal import periodogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = Path(__file__).resolve().parent
DATA = BASE / 'astronomical_data'
OLD_SCRIPT = BASE / 'exp_f_astronomical_solution.py'
SOLUTIONS = ('ZB18a', 'ZB20a')
WEIGHTS = {'ecc_dominant': (6.0, 2.5, 1.5), 'balanced': (1.0, 1.0, 1.0)}
CASES = tuple(f'{solution}_{weight}' for solution in SOLUTIONS for weight in WEIGHTS)
RROOT = BASE / 'expFZB_results'
FROOT = BASE / 'expFZB_figures'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def import_original():
    if not OLD_SCRIPT.is_file():
        raise FileNotFoundError(f'Missing {OLD_SCRIPT}. Keep the original Experiment F script unchanged.')
    spec = importlib.util.spec_from_file_location('original_exp_f_zb_read_only', OLD_SCRIPT)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    expected = {'WINDOW_OLD_MA': 42.0, 'WINDOW_YOUNG_MA': 2.0,
                'LENGTH_M': 600.0, 'DZ_M': 0.1, 'SR_TRUE': 15.0,
                'DT_MYR': 0.0005, 'NOISE_RATIO': 0.5, 'RHO': 0.7}
    for name, target in expected.items():
        if not np.isclose(getattr(original, name), target):
            raise ValueError(f'Original parameter {name} differs from documented baseline; stop and inspect.')
    if not np.allclose(original.SCAN_GRID, np.arange(5.0, 30.000001, 0.25)):
        raise ValueError('Original SR scan grid differs from documented baseline.')
    if tuple(original.SEEDS) != tuple(2001 + 1000 * k for k in range(10)):
        raise ValueError('Original seed list differs; stop and inspect.')
    if not np.isclose(original.STAGE1_GATE, 0.5):
        raise ValueError('Original Stage 1 threshold differs; stop and inspect.')
    return original


def read_zb(solution):
    """Read and validate every raw ZB row; interpolate only for 42--2 Ma window.

    Source uses adaptive/nonuniform time steps, unlike the La2004 1-kyr table.
    For reproducibility, construct the identical 1-kyr source grid in ALL cases;
    original Experiment F then interpolates from 1 kyr to its 0.5 kyr noise grid.
    """
    path = DATA / f'PT-{solution}_1-1.dat'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}. Copy downloaded original DAT file into astronomical_data/.')
    # Six plain-text numeric columns; ~750k rows x 6 numbers per file.
    a = np.loadtxt(path, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 6 or a.shape[0] < 100000:
        raise ValueError(f'{path.name}: expected six columns and a complete source file, got {a.shape}')
    if not np.isfinite(a).all():
        raise ValueError(f'{path.name}: NaN/Infinity in source')
    t = a[:, 0]
    if not np.all(np.diff(t) < 0):
        raise ValueError(f'{path.name}: source times must decrease monotonically')
    if not (abs(t[0]) < 0.01 and t[-1] <= -42000.0):
        raise ValueError(f'{path.name}: 42 Ma is not fully covered')
    if not np.all((a[:, 1] > 0) & (a[:, 1] < 0.2)):
        raise ValueError(f'{path.name}: eccentricity range invalid')
    if not np.all((a[:, 2] > 0.25) & (a[:, 2] < 0.55)):
        raise ValueError(f'{path.name}: expected obliquity in RADIANS, around 0.4')
    # Verify that climatic-precession column is consistent with the solution's
    # own longitude of perihelion. No recombination of different solutions.
    cp_error = float(np.max(np.abs(a[:, 5] - a[:, 1] * np.sin(a[:, 4]))))
    if cp_error > 1e-6:
        raise ValueError(f'{path.name}: column 6 is not consistent with e*sin(col5): {cp_error}')
    # Include some samples on both sides of the window endpoints for interpolation.
    subset = a[(t >= -42002.) & (t <= -1998.)][::-1].copy()
    if len(subset) < 40000 or subset[0, 0] > -42000 or subset[-1, 0] < -2000:
        raise ValueError(f'{path.name}: incomplete 42--2 Ma window after selecting samples')
    source_t = subset[:, 0]       # increasing from older to younger
    common_kyr = np.arange(-42000.0, -1999.999, 1.0)
    if common_kyr.shape[0] != 40001:
        raise AssertionError('Unexpected target 1-kyr grid length')
    # e, obliquity and climatic precession are EACH taken from this SAME file.
    e = np.interp(common_kyr, source_t, subset[:, 1])
    obl = np.interp(common_kyr, source_t, subset[:, 2])
    cp = np.interp(common_kyr, source_t, subset[:, 5])
    t_from_old_myr = (common_kyr + 42000.0) / 1000.0
    meta = {
        'filename': path.name, 'sha256': sha256(path), 'source_rows': int(len(a)),
        'source_time_first_kyr': float(t[0]), 'source_time_last_kyr': float(t[-1]),
        'selected_raw_rows_with_padding': int(len(subset)),
        'median_raw_spacing_kyr_within_window': float(np.median(np.diff(source_t))),
        'source_columns': ['time_kyr', 'eccentricity', 'obliquity_radians',
                           'precession_radians', 'perihelion_longitude_radians',
                           'climatic_precession_e_sin_perihelion'],
        'climatic_precession_consistency_max_abs': cp_error,
        'working_1kyr_rows': int(len(common_kyr)),
        'working_time_old_to_young_myr': [float(t_from_old_myr[0]), float(t_from_old_myr[-1])]
    }
    return (t_from_old_myr, e, obl, cp), meta


def construct_carriers(original, solution_inputs):
    common_t = None
    fine = None
    out = {}
    component_stats = {}
    for name in SOLUTIONS:
        (t, e, obl, cp), _ = solution_inputs[name]
        if common_t is None:
            common_t = t
            fine = np.arange(0.0, 40.0 + original.DT_MYR * .1, original.DT_MYR)
        elif not np.array_equal(t, common_t):
            raise ValueError('Different astronomical solutions produced different working time axes')
        parts = tuple(original.standardized(v) for v in (e, obl, cp))
        component_stats[name] = dict(eccentricity_std=float(np.std(e)),
                                     obliquity_std_rad=float(np.std(obl)),
                                     climatic_precession_std=float(np.std(cp)))
        for mode, weights in WEIGHTS.items():
            combined = sum(w * c for w, c in zip(weights, parts))
            combined = original.standardized(combined) * original.BASELINE_RMS
            carrier = np.interp(fine, t, combined)
            case = f'{name}_{mode}'
            if not np.isclose(np.std(carrier), original.BASELINE_RMS, rtol=.02):
                raise ValueError(f'{case}: bad RMS after interpolation')
            out[case] = carrier
    if set(out) != set(CASES):
        raise AssertionError('Wrong cases generated')
    return fine, out, component_stats


def smoke_checks(original, fine, clean):
    if len(fine) != 80001 or not np.isclose(fine[-1], 40.):
        raise ValueError('Incorrect 0.5 kyr noise grid')
    n1 = original.make_noise(original.SEEDS[0], len(fine))
    n2 = original.make_noise(original.SEEDS[0], len(fine))
    if not np.array_equal(n1, n2):
        raise ValueError('Noise must be deterministic for matched seed')
    for case in CASES:
        depth, truth, record = original.arrays_on_depth(fine, clean[case], n1)
        if len(depth) != 6001 or not np.isclose(truth[-1], 40.) or not np.isfinite(record).all():
            raise ValueError(f'{case}: invalid depth sampling')
    print('ZB DATA CHECK PASS: ZB18a/ZB20a complete e, obliquity, climatic precession; '
          '42--2 Ma; all inputs standardized and independently combined; '
          'fixed matched noise; 6001 depth samples.', flush=True)


def run_models(original, fine, clean, seeds):
    try:
        from PyEMD import EMD
    except ImportError as exc:
        raise RuntimeError('Cannot import PyEMD; use the original research environment (EMD-signal).') from exc
    AstroTuner, tuner_path = original.import_tuner()
    tuner = AstroTuner(sr_grid=original.SCAN_GRID, n_jobs=1, seed=42)
    results, arrays, failures = [], {}, []
    for seed in seeds:
        noise = original.make_noise(seed, len(fine))
        for case in CASES:
            tag = f'{case}__seed{seed}'
            depth, truth, x = original.arrays_on_depth(fine, clean[case], noise)
            arrays[tag + '__signal'] = x
            arrays[tag + '__true_age'] = truth
            row = dict(case=case, seed=int(seed), true_sr_m_per_myr=float(original.SR_TRUE),
                       stage1_pass=False, imf_gate_pass=False, gate_accepted=False,
                       anchor_status='not_computed', match_score=np.nan,
                       scan_sr_m_per_myr=np.nan, imf_index=np.nan, imf_cycles=np.nan,
                       anchor_count=0, constant_rmse_myr=np.nan,
                       forced_anchor_rmse_myr=np.nan,
                       forced_minus_constant_rmse_myr=np.nan,
                       forced_anchor_wins=False)
            try:
                scan = tuner.scan_sed_rates(depth, x)
                sr, match = tuner.best_sr(scan, depth=depth, signal=x)
                const_age = tuner.build_age_model(depth, sr)
                const_err = original.errors(const_age, truth)
                row.update(scan_sr_m_per_myr=float(sr), match_score=float(match),
                           stage1_pass=bool(match >= original.STAGE1_GATE),
                           constant_rmse_myr=float(const_err['rmse_myr']),
                           constant_end_bias_myr=float(const_err['end_bias_myr']))
                arrays[tag + '__constant_age'] = const_age
            except Exception as exc:
                row['anchor_status'] = 'scan_error'
                failures.append(f'{tag}: scan: {traceback.format_exc()}')
                results.append(row)
                print(f'{tag}: SCAN ERROR: {exc}', flush=True)
                continue
            try:
                imfs = EMD().emd(x, max_imf=10)
                if not len(imfs):
                    raise ValueError('EMD returned no IMFs')
                selection = tuner.select_405_imf(imfs, depth, sr_est=sr)
                anchor, table = tuner.build_age_model_405(
                    depth, imfs[selection['imf_index']], anchor_phase='trough')
                forced_err = original.errors(anchor, truth)
                row.update(imf_index=int(selection['imf_index']) + 1,
                           imf_cycles=int(selection['n_cycles']),
                           imf_gate_pass=bool(selection['matched']),
                           gate_accepted=bool(match >= original.STAGE1_GATE and selection['matched']),
                           anchor_count=len(table), anchor_status='forced_diagnostic_only',
                           forced_anchor_rmse_myr=float(forced_err['rmse_myr']),
                           forced_anchor_end_bias_myr=float(forced_err['end_bias_myr']),
                           forced_minus_constant_rmse_myr=float(forced_err['rmse_myr'] - const_err['rmse_myr']),
                           forced_anchor_wins=bool(forced_err['rmse_myr'] < const_err['rmse_myr']))
                arrays[tag + '__forced_anchor_age'] = anchor
            except Exception as exc:
                row['anchor_status'] = 'anchor_error'
                failures.append(f'{tag}: anchor: {traceback.format_exc()}')
            results.append(row)
            print(f'{tag}: match={match:.3f}, SR={sr:.2f}, '
                  f'accepted={row["gate_accepted"]}, '
                  f'constant={row["constant_rmse_myr"]:.3f}, '
                  f'forced={row["forced_anchor_rmse_myr"]:.3f} Myr', flush=True)
    df = pd.DataFrame(results)
    if len(df) != len(CASES) * len(seeds) or df.duplicated(['case', 'seed']).any():
        raise AssertionError('Missing or duplicated case-by-seed results')
    return df, arrays, failures, tuner_path


def draw_figure(fine, clean, df, output_dir, original):
    palette = {'ZB18a_ecc_dominant': '#467B9B',
               'ZB18a_balanced': '#69A090',
               'ZB20a_ecc_dominant': '#C57051',
               'ZB20a_balanced': '#B09B55'}
    labels = {'ZB18a_ecc_dominant': 'ZB18a eccentricity-dominant',
              'ZB18a_balanced': 'ZB18a balanced',
              'ZB20a_ecc_dominant': 'ZB20a eccentricity-dominant',
              'ZB20a_balanced': 'ZB20a balanced'}
    fig, axs = plt.subplots(2, 2, figsize=(14, 9.6))
    a, b, c, d = axs.flat
    for ax in axs.flat:
        ax.grid(axis='y', alpha=.25, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    mask = (fine >= 10.) & (fine <= 11.)
    for case in CASES:
        a.plot(fine[mask], clean[case][mask], lw=.92, color=palette[case], alpha=.9)
        freq, power = periodogram(clean[case], fs=1. / original.DT_MYR,
                                  detrend='linear', window='hann', scaling='spectrum')
        good = (freq > 0) & (1. / np.maximum(freq, 1e-100) >= .015) & (1. / np.maximum(freq, 1e-100) <= .65)
        period_kyr = 1000. / freq[good]
        idx = np.argsort(period_kyr)
        b.plot(period_kyr[idx], power[good][idx], lw=.95, color=palette[case], alpha=.9)
    a.set(title='(a) Clean full-solution carriers (10–11 Myr excerpt)',
          xlabel='Time since 42 Ma (Myr)', ylabel='Normalized proxy (a.u.)')
    b.set(title='(b) Power spectra of clean full-solution carriers',
          xlabel='Period (kyr)', ylabel='Power (log scale)')
    b.set_xlim(15, 450)
    b.set_yscale('log')
    for period in (21, 41, 95, 120, 405):
        b.axvline(period, color='#9AA4AF', lw=.65, ls=':', alpha=.5)
    xx = np.arange(len(CASES))
    labels_short = ['ZB18a\necc-dominant', 'ZB18a\nbalanced',
                    'ZB20a\necc-dominant', 'ZB20a\nbalanced']
    s1 = df.groupby('case')['stage1_pass'].sum().reindex(CASES).to_numpy()
    gates = df.groupby('case')['gate_accepted'].sum().reindex(CASES).to_numpy()
    c.bar(xx - .18, s1, .34, color='#76987B')
    c.bar(xx + .18, gates, .34, color='#C77A64')
    c.set_xticks(xx, labels_short)
    c.set_ylim(0, len(df['seed'].unique()) + 1)
    c.set_ylabel('Number of runs')
    c.set_title('(c) Original algorithm acceptance counts')
    for i, case in enumerate(CASES):
        frame = df[df['case'] == case]
        for key, color, offset in [('constant_rmse_myr', '#B38E50', -.12),
                                   ('forced_anchor_rmse_myr', '#3F7191', .12)]:
            values = frame[key].dropna().to_numpy(dtype=float)
            if values.size:
                d.scatter(np.full(len(values), i + offset), values,
                          color=color, s=25, alpha=.50)
                d.scatter(i + offset, np.median(values), color=color,
                          s=70, edgecolor='white', linewidth=.9, zorder=5)
    d.set_xticks(xx, labels_short)
    d.set_ylabel('Age RMSE (Myr)')
    d.set_title('(d) Age-model errors (large dots: median)')
    handles = [Line2D([], [], color=palette[k], lw=2, label=labels[k]) for k in CASES]
    handles += [Line2D([], [], color='#76987B', lw=6, label='Stage 1 pass'),
                Line2D([], [], color='#C77A64', lw=6, label='Both gates pass'),
                Line2D([], [], marker='o', lw=0, color='#B38E50', label='Constant-SR RMSE'),
                Line2D([], [], marker='o', lw=0, color='#3F7191', label='Forced-anchor RMSE')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .01),
               ncol=4, fontsize=9, frameon=False, columnspacing=1.7)
    fig.subplots_adjust(left=.08, right=.985, top=.94, bottom=.20, wspace=.30, hspace=.48)
    output_dir.mkdir(parents=False, exist_ok=False)
    fig.savefig(output_dir / 'expFZB_full_comparison_2x2.png', dpi=300, facecolor='white')
    fig.savefig(output_dir / 'expFZB_full_comparison_2x2.pdf', facecolor='white')
    plt.close(fig)


def fresh_dirs(mode):
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:8]
    name = f'{mode}_{stamp}'
    RROOT.mkdir(exist_ok=True)
    FROOT.mkdir(exist_ok=True)
    rdir = RROOT / name
    fdir = FROOT / name
    rdir.mkdir(exist_ok=False)
    # Figure dir is created by draw_figure().
    return rdir, fdir


def main(mode):
    original = import_original()
    source_la2004 = original.find_data()
    # Validate original La2004 as an unchanged reference, but do not use its
    # orbital components for constructing any ZB signal.
    original_data = original.read_la2004(source_la2004)
    if original_data[-1] != 51001:
        raise ValueError('Original La2004 data format or coverage differs')
    inputs = {}
    for name in SOLUTIONS:
        inputs[name] = read_zb(name)
        print(f'{name}: raw rows={inputs[name][1]["source_rows"]}; '
              f'window rows={inputs[name][1]["selected_raw_rows_with_padding"]}; '
              f'median step={inputs[name][1]["median_raw_spacing_kyr_within_window"]:.5f} kyr; '
              f'climatic-precession check={inputs[name][1]["climatic_precession_consistency_max_abs"]:.2e}')
    fine, clean, stats = construct_carriers(original, inputs)
    smoke_checks(original, fine, clean)
    print('SCIENTIFIC SCOPE: Each ZB case uses eccentricity, obliquity and climatic precession '
          'from its OWN matching PT-ZB solution. Proxy weights are prescribed assumptions.')
    if mode == 'check-only':
        print('CHECK ONLY PASSED. No EMD run and no output files written.')
        return
    seeds = original.SEEDS[:1] if mode == 'smoke' else original.SEEDS
    rdir, fdir = fresh_dirs(mode)
    df, arrays, failures, tuner_path = run_models(original, fine, clean, seeds)
    summary = df.groupby('case', as_index=False).agg(
        n=('seed', 'size'), stage1_pass=('stage1_pass', 'sum'),
        both_gates_pass=('gate_accepted', 'sum'),
        forced_computed=('forced_anchor_rmse_myr', 'count'),
        median_const_rmse_myr=('constant_rmse_myr', 'median'),
        median_forced_rmse_myr=('forced_anchor_rmse_myr', 'median'),
        mean_forced_rmse_myr=('forced_anchor_rmse_myr', 'mean'),
        forced_wins=('forced_anchor_wins', 'sum'))
    df.to_csv(rdir / 'expFZB_all.csv', index=False)
    summary.to_csv(rdir / 'expFZB_summary.csv', index=False)
    np.savez_compressed(rdir / 'expFZB_series.npz', **arrays)
    (rdir / 'expFZB_errors.txt').write_text('\n'.join(failures) if failures else 'No per-realization errors.', encoding='utf-8')
    metadata = {
        'experiment': 'Independent full-astronomical-forcing sensitivity: PT-ZB18a(1,1) and PT-ZB20a(1,1)',
        'mode': mode, 'cases': list(CASES), 'seeds': list(seeds),
        'source_repository': 'https://github.com/japhir/paleoinsolation',
        'publication_doi': '10.1029/2025PA005287',
        'source_files': {k: v[1] for k, v in inputs.items()},
        'original_script': str(OLD_SCRIPT), 'original_script_sha256': sha256(OLD_SCRIPT),
        'original_la2004_file': str(source_la2004),
        'original_la2004_sha256': sha256(source_la2004),
        'tuner_source': tuner_path,
        'time_window_ma_before_present': [42.0, 2.0],
        'source_interpolation': 'Adaptive source times -> shared 1 kyr grid -> original 0.5 kyr noise grid; no additional frequency information',
        'complete_components': 'For each solution e, obliquity and climatic precession all from SAME PT-ZB file',
        'component_statistics': stats,
        'weights': WEIGHTS, 'source_obliquity_units': 'radians',
        'depth_length_m': original.LENGTH_M, 'depth_sampling_m': original.DZ_M,
        'true_sedimentation_rate_m_per_myr': original.SR_TRUE,
        'noise_ratio': original.NOISE_RATIO, 'rho': original.RHO,
        'stage1_match_gate': original.STAGE1_GATE,
        'sr_scan_m_per_myr': [5.0, 30.0, .25],
        'forced_anchor_rule': 'Computed for diagnosis even if original gates fail',
        'limitations': ['These are astronomical parameters, not observed sedimentary proxies',
                        'Both ZB solutions depend on their published orbital and Earth-spin model assumptions',
                        'Constant true SR only, 42--2 Ma window only',
                        'No modeled hiatus, instrument vertical response or depositional filtering',
                        'Accepted gates are screening rules, not accuracy guarantees'],
        'original_files_modified': False,
    }
    (rdir / 'expFZB_design.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    draw_figure(fine, clean, df, fdir, original)
    print('\nSUMMARY (all cases, no cherry-picking):\n' + summary.to_string(index=False))
    print('\nRESULTS:', rdir)
    print('FIGURES:', fdir)
    print('FAILURES:', len(failures))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--check-only', action='store_true', help='Parse and validate data; no model runs, no writes')
    group.add_argument('--smoke', action='store_true', help='Run the four cases for one matched seed')
    group.add_argument('--run', action='store_true', help='Run the four cases for ten matched seeds')
    opts = parser.parse_args()
    main('smoke' if opts.smoke else 'run' if opts.run else 'check-only')

'''
# --- END ZB SOURCE ---

# --- BEGIN INTEGRATED PLOTTING SOURCE ---
PLOT_SOURCE = r'''
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Experiment F integrated comparison -- plot reproducibly from REAL data.

Requirements (put this script alongside the existing original scripts):
  codes/exp_f_astronomical_solution.py
  codes/exp_f_zb_full_comparison.py
  codes/astronomical_data/INSOLN.LA2004.BTL.ASC
  codes/astronomical_data/PT-ZB18a_1-1.dat
  codes/astronomical_data/PT-ZB20a_1-1.dat
  codes/supplement_results/expF_all30.csv
  codes/expFZB_results/run_*/expFZB_all.csv (complete 40-row RUN)

This is a POSTPROCESSING + FIGURE script. It never reruns EMD, rescans SR,
changes parameters, or overwrites source files. It loads ALL ORIGINAL per-seed
results (La2004 n=20 + ZB n=40) and reconstructs clean astronomical carriers
from the same original functions and actual orbital data used in experiments.

Running inside Spyder:
  %runfile 'F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes/exp_f_integrated_three_solutions.py' --wdir

Output: new unique subfolder in codes/expF_integrated_outputs/ containing PNG,
SVG (editable in vector editors), combined CSV, summary CSV and audit JSON.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime
from pathlib import Path
import uuid
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
LA_SCRIPT = BASE / 'exp_f_astronomical_solution.py'
ZB_SCRIPT = BASE / 'exp_f_zb_full_comparison.py'
LA_RESULTS = BASE / 'supplement_results' / 'expF_all30.csv'
ZB_RESULT_ROOT = BASE / 'expFZB_results'
OUTPUT_ROOT = BASE / 'expF_integrated_outputs'
SEEDS = tuple(2001 + 1000*i for i in range(10))
SOLUTIONS = ('La2004', 'ZB18a', 'ZB20a')
MODES = ('ecc_dominant', 'balanced')
PALETTE = {'La2004':'#2855A5', 'ZB18a':'#008B63', 'ZB20a':'#D66A35'}
COLUMNS_REQUIRED = ('case', 'seed', 'stage1_pass', 'gate_accepted',
                    'constant_rmse_myr', 'forced_anchor_rmse_myr')


def import_module(module_name: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError('Missing script: ' + str(path))
    spec = importlib.util.spec_from_file_location(module_name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = obj
    spec.loader.exec_module(obj)
    return obj


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def bool_values(s, label):
    # Read actual CSV logical values without converting arbitrary nonempty
    # strings to True (which would give incorrect acceptance statistics).
    txt = s.astype(str).str.strip().str.lower()
    if not txt.isin(('true', 'false', '1', '0')).all():
        raise ValueError('Invalid boolean values in '+label+': '+str(sorted(set(txt))))
    return txt.isin(('true','1'))


def read_results(path: Path, case_map, expected_count):
    df = pd.read_csv(path)
    missing = set(COLUMNS_REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing fields {missing}')
    if df.duplicated(['case','seed']).any():
        raise ValueError(f'{path}: duplicate (case, seed) values')
    if expected_count == 30:
        # Ignore only the original ideal-four-sine control; keep its data intact.
        if len(df) != 30 or set(df['case']) != {'ideal_four_sines', *case_map}:
            raise ValueError('Original Experiment F CSV must contain all 30 runs (four-sine + two La2004 cases).')
        df = df[df['case'].isin(case_map)].copy()
    elif len(df) != expected_count or set(df['case']) != set(case_map):
        raise ValueError(f'ZB CSV must contain {expected_count} runs covering exactly four ZB cases.')
    df['seed'] = pd.to_numeric(df['seed'], errors='raise').astype(int)
    for oldcase in case_map:
        ids = tuple(sorted(df.loc[df['case']==oldcase, 'seed']))
        if ids != SEEDS:
            raise ValueError(f'{path}: {oldcase} seeds do not match original 10 paired seeds: {ids}')
    for k in ('stage1_pass','gate_accepted'):
        df[k] = bool_values(df[k], k)
    if (df['gate_accepted'] & ~df['stage1_pass']).any():
        raise ValueError('An accepted run failed Stage 1, contradicting original gate rule')
    for k in ('constant_rmse_myr','forced_anchor_rmse_myr'):
        df[k] = pd.to_numeric(df[k], errors='raise')
        if not np.isfinite(df[k]).all():
            raise ValueError('Incomplete RMSE data in '+str(path))
    df[['solution','weight']] = df['case'].map(case_map).apply(pd.Series)
    df['origin_file'] = str(path)
    return df


def choose_full_zb_run():
    if not ZB_RESULT_ROOT.is_dir():
        raise FileNotFoundError('Cannot find '+str(ZB_RESULT_ROOT))
    candidates = []
    for folder in sorted(ZB_RESULT_ROOT.glob('run_*')):
        file = folder / 'expFZB_all.csv'
        design = folder / 'expFZB_design.json'
        if not file.is_file():
            continue
        if design.is_file():
            meta = json.loads(design.read_text(encoding='utf-8'))
            if meta.get('mode') != 'run':
                continue
        try:
            check = pd.read_csv(file, usecols=['case','seed'])
            if len(check)==40:
                candidates.append(file)
        except (ValueError, OSError):
            pass
    if not candidates:
        raise FileNotFoundError('No complete 40-row ZB run CSV was found in '
                                f'{ZB_RESULT_ROOT}/run_*/expFZB_all.csv. '
                                'Do NOT point to smoke_*/ data.')
    # Use latest folder-name timestamp (generated by original ZB script).
    chosen = sorted(candidates, key=lambda p:p.parent.name)[-1]
    print('Selected full ZB run:', chosen)
    if len(candidates)>1:
        print('Found',len(candidates),'full runs; using newest run folder by name.')
    return chosen


def get_clean_signals(original, zb):
    source_la = original.find_data()
    t, e, obl, pibar, _ = original.read_la2004(source_la)
    fine, laclean, _ = original.make_clean_inputs(t, e, obl, pibar)
    # The ZB script itself parses, checks, and resamples each published full
    # astronomical solution, using all three of its matching components.
    zb_inputs = {name: zb.read_zb(name) for name in ('ZB18a', 'ZB20a')}
    zfine, zbclean, _ = zb.construct_carriers(original, zb_inputs)
    if not np.array_equal(fine, zfine):
        raise ValueError('La2004 and ZB carrier time grids differ; stop instead of silently resampling')
    carriers = {}
    for mode in MODES:
        carriers[('La2004',mode)] = laclean['la2004_'+mode]
        for name in ('ZB18a','ZB20a'):
            carriers[(name,mode)] = zbclean[name+'_'+mode]
    return fine, carriers, source_la, zb_inputs


def make_summary(combined):
    rows=[]
    for sol in SOLUTIONS:
        for mode in MODES:
            sub=combined[(combined.solution==sol)&(combined.weight==mode)]
            if len(sub)!=10:
                raise ValueError(f'{sol}/{mode}: not 10 records')
            rows.append(dict(solution=sol,weight=mode,n=10,
              stage1_pass=int(sub.stage1_pass.sum()),
              both_gates_pass=int(sub.gate_accepted.sum()),
              median_const_rmse_myr=float(sub.constant_rmse_myr.median()),
              median_forced_rmse_myr=float(sub.forced_anchor_rmse_myr.median()),
              forced_wins=int((sub.forced_anchor_rmse_myr<sub.constant_rmse_myr).sum())))
    return pd.DataFrame(rows)


def make_figure(fine, carriers, combined, summary, outdir, original):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
        'axes.titlesize':12, 'axes.labelsize':10,'xtick.labelsize':9,
        'ytick.labelsize':9,'axes.spines.top':False,'axes.spines.right':False,
        'savefig.facecolor':'white'})
    fig, axs=plt.subplots(2, 2, figsize=(15.5, 9.4))
    a,b,c,d = axs.ravel()
    for ax in axs.ravel():
        ax.set_axisbelow(True)
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Experiment F: comparison of complete astronomical solutions',
                 fontweight='bold', fontsize=16, x=.07, y=.985,ha='left')
    fig.text(.07,.939,
             'La2004  |  ZB18a  |  ZB20a     ·     fixed experimental settings     ·     10 paired seeds per case',
             fontsize=10,color='#576476')

    # Panels (a,b): make nearly overlapping ZB18a and ZB20a both visible.
    # Draw ZB18a last using a dashed green line without square markers.
    # Verify that the underlying source signals differ.
    keep=(fine>=10)&(fine<=11)
    zb18 = np.asarray(carriers[('ZB18a','balanced')])
    zb20 = np.asarray(carriers[('ZB20a','balanced')])
    if zb18.shape != zb20.shape:
        raise ValueError('ZB18a/ZB20a carrier lengths differ')
    max_difference = float(np.max(np.abs(zb18-zb20)))
    rms_difference = float(np.sqrt(np.mean((zb18-zb20)**2)))
    print(f'ZB18a vs ZB20a balanced signal differences: '
          f'max_abs={max_difference:.9g}, rms={rms_difference:.9g}')
    if max_difference < 1e-12:
        raise ValueError('ZB18a and ZB20a carriers are numerically identical; '
                         'check source parsing before publishing this figure')

    LINE_STYLES = {
        'La2004': dict(color=PALETTE['La2004'], ls='-', lw=1.25,
                       alpha=.9, zorder=2),
        'ZB20a': dict(color=PALETTE['ZB20a'], ls='-', lw=1.05,
                      alpha=.75, zorder=3),
        'ZB18a': dict(color=PALETTE['ZB18a'], ls='--', lw=1.5,
                      alpha=1.0, zorder=5),
    }
    # Plot green LAST, not underneath the orange curve.
    for sol in ('La2004','ZB20a','ZB18a'):
        clean=np.asarray(carriers[(sol,'balanced')])
        style=LINE_STYLES[sol]
        a.plot(fine[keep],clean[keep],label=sol, **style)
        periods,powers=original.get_spectral_periods(clean,original.DT_MYR)
        idx=np.argsort(periods)
        per=np.asarray(periods)[idx]
        pow=np.maximum(np.asarray(powers)[idx],1e-12)
        b.plot(per,pow,**style)
    a.set(title='(a) Noise-free synthetic proxy  |  balanced weights',
          xlabel='Time since 42 Ma (Myr)',ylabel='Normalized proxy (a.u.)',xlim=(10,11))
    handles, labels = a.get_legend_handles_labels()
    legend_map = dict(zip(labels, handles))
    a.legend([legend_map[x] for x in SOLUTIONS], SOLUTIONS,
             loc='upper center',bbox_to_anchor=(.5,-.18),ncol=3,frameon=False)
    b.axvline(405,color='#717c88',lw=.85,ls=':')
    b.text(410,0.06,'405 kyr',fontsize=9,color='#596777')
    b.set(title='(b) Power spectra  |  balanced weights',
          xlabel='Period (kyr)',ylabel='Power (log scale)',
          xlim=(15,450),yscale='log',ylim=(1e-10,40))
    b.grid(axis='both',ls=':',alpha=.18)

    groups=[(sol,mode) for sol in SOLUTIONS for mode in MODES]
    xx=np.arange(len(groups))
    for i,(sol,mode) in enumerate(groups):
        item=summary[(summary.solution==sol)&(summary.weight==mode)].iloc[0]
        c.bar(i-.15,item.stage1_pass,.28,color=PALETTE[sol],alpha=.28,
              edgecolor=PALETTE[sol],linewidth=1)
        c.bar(i+.15,item.both_gates_pass,.28,color=PALETTE[sol],alpha=.95)
        c.text(i+.15,item.both_gates_pass+.14,str(int(item.both_gates_pass)),
               ha='center',fontweight='bold',color=PALETTE[sol],fontsize=9)
        rows=combined[(combined.solution==sol)&(combined.weight==mode)].sort_values('seed')
        forced=rows.forced_anchor_rmse_myr.to_numpy(float)
        const=rows.constant_rmse_myr.to_numpy(float)
        rng=np.random.default_rng(731+i)  # jitter ONLY; does not change model data
        jitter=rng.uniform(-.075,.075,len(forced))
        d.scatter(i+jitter,forced,s=29,color=PALETTE[sol],alpha=.37,
                  linewidths=0,zorder=2)
        d.scatter(i,np.median(forced),s=90,color=PALETTE[sol],
                  edgecolor='white',linewidth=1.2,zorder=5)
        d.scatter(np.full(len(const),i-.13),const,s=18,color='#AC8C41',alpha=.40,linewidths=0,zorder=2)
        d.scatter(i-.13,np.median(const),s=72,color='#AC8C41',marker='D',
                  edgecolor='white',linewidth=1,zorder=5)
    ticklabels=['Ecc.\ndominant','Balanced']*3
    for ax in (c,d):
        ax.set_xticks(xx,ticklabels)
        for xpos in (1.5,3.5):
            ax.axvline(xpos,color='#D9DFE6',lw=1.25)
        ax.set_xlim(-.6,5.6)
    c.set(title='(c) Algorithm acceptance by forcing solution',
          ylabel='Accepted runs out of 10',ylim=(0,11.25))
    d.set(title='(d) Age-model RMSE  |  10 seeds per case',
          ylabel='RMSE (Myr)',ylim=(-.18,max(6.5,combined.forced_anchor_rmse_myr.max()+.55)))
    ylimd=d.get_ylim()[1]
    for mid,sol in zip((.5,2.5,4.5),SOLUTIONS):
        c.text(mid,11.02,sol,ha='center',va='top',color=PALETTE[sol],fontweight='bold')
        d.text(mid,ylimd-.18,sol,ha='center',va='top',color=PALETTE[sol],fontweight='bold')
    c.legend(handles=[Patch(fc='#8A9CAC',alpha=.30,ec='#7C8998',label='Stage 1'),
                      Patch(fc='#718599',label='Both gates')],
             loc='upper center',bbox_to_anchor=(.5,-.20),frameon=False,ncol=2)
    d.legend(handles=[Line2D([],[],marker='o',ls='none',markerfacecolor='#67849C',
                              markeredgecolor='none',label='Individual forced anchors'),
                      Line2D([],[],marker='o',ls='none',markerfacecolor='#67849C',
                              markeredgecolor='white',markersize=9,label='Median forced anchor'),
                      Line2D([],[],marker='D',ls='none',markerfacecolor='#AC8C41',
                              markeredgecolor='white',markersize=7,label='Median constant SR')],
             loc='upper center',bbox_to_anchor=(.5,-.20),frameon=False,ncol=3,fontsize=8)
    fig.subplots_adjust(left=.065,right=.985,top=.88,bottom=.205,wspace=.23,hspace=.48)
    fig.text(.065,.044,
             'All RMSE points come directly from original per-seed CSV files; no manually entered values.',
             fontsize=8.5,color='#627085')
    fig.text(.065,.024,
             'Forced-anchor RMSE includes diagnostic runs rejected by acceptance gates; published astronomical parameters are not observed sedimentary proxies.',
             fontsize=8.5,color='#627085')
    fn=outdir/'expF_integrated_La2004_ZB18a_ZB20a'
    fig.savefig(fn.with_suffix('.png'),dpi=300,facecolor='white')
    fig.savefig(fn.with_suffix('.svg'),facecolor='white')
    plt.close(fig)
    return fn.with_suffix('.png'), fn.with_suffix('.svg')


def main():
    print('=== Experiment F, reproducible three-solution integration ===',flush=True)
    original=import_module('_expF_original_for_postprocess', LA_SCRIPT)
    zb=import_module('_expF_ZB_for_postprocess', ZB_SCRIPT)
    if not LA_RESULTS.is_file():
        raise FileNotFoundError('Missing original La2004 results: '+str(LA_RESULTS))
    zb_result=choose_full_zb_run()
    map_la={f'la2004_{mode}':('La2004',mode) for mode in MODES}
    map_zb={f'{sol}_{mode}':(sol,mode) for sol in ('ZB18a','ZB20a') for mode in MODES}
    la=read_results(LA_RESULTS,map_la,30)
    z=read_results(zb_result,map_zb,40)
    combined=pd.concat([la,z],ignore_index=True)
    if len(combined)!=60:
        raise ValueError('Integrated comparison must contain exactly 60 actual evaluations.')
    summary=make_summary(combined)
    fine,carriers,source_la,zb_sources=get_clean_signals(original,zb)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:6]
    out=OUTPUT_ROOT/('integrated_'+stamp)
    out.mkdir(parents=True,exist_ok=False)
    combined.to_csv(out/'expF_integrated_all60.csv',index=False,encoding='utf-8-sig')
    summary.to_csv(out/'expF_integrated_summary.csv',index=False,encoding='utf-8-sig')
    manifest={
        'kind':'postprocess_only; no EMD or AstroTuner rerun',
        'original_la2004_results':str(LA_RESULTS),
        'original_la2004_results_sha256':file_hash(LA_RESULTS),
        'original_zb_results':str(zb_result),
        'original_zb_results_sha256':file_hash(zb_result),
        'original_la2004_source':str(source_la),
        'original_la2004_source_sha256':file_hash(source_la),
        'zb_source_files':{key:{'name':str(zb.DATA/f'PT-{key}_1-1.dat'),
                                 'sha256':info[1]['sha256']} for key,info in zb_sources.items()},
        'rows':60,'seeds':SEEDS,'time_window_before_present_ma':[42,2],
        'carrier_panels_weighting':'balanced only',
        'result_panels_weighting':'eccentricity-dominant and balanced',
        'note':'All 60 points read from original CSV; rejected forced anchors are diagnostics.',
    }
    (out/'expF_integrated_provenance.json').write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    png,svg=make_figure(fine,carriers,combined,summary,out,original)
    print('\nALL 60 REAL RUNS VERIFIED. NO EXPERIMENTS RE-RUN.',flush=True)
    print('\nSUMMARY:\n'+summary.to_string(index=False),flush=True)
    print('\nINTEGRATED PNG:',png,flush=True)
    print('INTEGRATED SVG:',svg,flush=True)
    print('INTEGRATED CSV:',out/'expF_integrated_all60.csv',flush=True)
    print('SUMMARY CSV:',out/'expF_integrated_summary.csv',flush=True)
    print('SOURCE MANIFEST:',out/'expF_integrated_provenance.json',flush=True)

if __name__ == '__main__':
    main()

'''
# --- END PLOTTING SOURCE ---

def _summary_all70(combined):
    rows = []
    for case, sub in combined.groupby('case', sort=False):
        rows.append({
            'case': case, 'n': int(len(sub)),
            'stage1_pass': int(sub['stage1_pass'].sum()),
            'both_gates_pass': int(sub['gate_accepted'].sum()),
            'forced_computed': int(sub['forced_anchor_rmse_myr'].notna().sum()),
            'median_const_rmse_myr': float(sub['constant_rmse_myr'].median()),
            'median_forced_rmse_myr': float(sub['forced_anchor_rmse_myr'].median()),
            'forced_wins': int((sub['forced_anchor_rmse_myr'] < sub['constant_rmse_myr']).sum()),
        })
    return pd.DataFrame(rows)


def main():
    if RUN_MODE not in ('full', 'check'):
        raise ValueError('RUN_MODE must be full or check')
    original = _load_embedded('_expf_embedded_original', ORIGINAL_SOURCE)
    zb = _load_embedded('_expf_embedded_zb', ZB_SOURCE)
    plotting = _load_embedded('_expf_embedded_integrated_plot', PLOT_SOURCE)
    # Adapt the two embedded source modules to the actual file location.
    # Original algorithm, weights, gates and simulation logic stay unchanged.
    original.BASE = BASE
    original.RDIR = RESULT_ROOT  # original routines kept in their original hierarchy
    original.FDIR = FIGURE_ROOT
    zb.BASE = BASE
    zb.DATA = BASE / 'astronomical_data'
    zb.OLD_SCRIPT = BASE / 'exp_f_astronomical_solution.py'  # unused: import_original() is never called
    plotting.BASE = BASE

    # All raw sources are read directly and checked before any model runs.
    source_la = original.find_data()
    t, e, obl, pibar, n = original.read_la2004(source_la)
    fine_la, clean_la, _ = original.data_smoke((t, e, obl, pibar, n))
    sources_zb = {}
    for solution in zb.SOLUTIONS:
        info = zb.read_zb(solution)
        sources_zb[solution] = info
        print(f'{solution}: source rows={info[1]["source_rows"]}; '
              f'climatic precession check={info[1]["climatic_precession_consistency_max_abs"]:.3e}',
              flush=True)
    fine_zb, clean_zb, component_stats = zb.construct_carriers(original, sources_zb)
    zb.smoke_checks(original, fine_zb, clean_zb)
    if not np.array_equal(fine_la, fine_zb):
        raise ValueError('The La2004 and ZB model time grids do not match; do not silently resample')
    if RUN_MODE == 'check':
        print('CHECK PASSED. No model evaluations or output files produced.')
        return

    # Exactly the original published 10 matched random seeds. Run original
    # three cases (30), then ZB four cases (40), with identical actual functions.
    stamp = 'expF_unified_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    out = RESULT_ROOT / stamp
    figout = FIGURE_ROOT / stamp
    out.mkdir(parents=False, exist_ok=False)
    figout.mkdir(parents=False, exist_ok=False)
    print('RESULTS:', out, flush=True)
    print('FIGURES:', figout, flush=True)
    print('RUNNING original ideal-four-sine + La2004 (30 models)...', flush=True)
    _, tuner_path = original.import_tuner()
    df_la, arrays_la, errors_la = original.run_models(fine_la, clean_la, source_la, tuner_path)
    print('RUNNING ZB18a + ZB20a complete 3-component cases (40 models)...', flush=True)
    df_zb, arrays_zb, errors_zb, _ = zb.run_models(original, fine_zb, clean_zb, original.SEEDS)

    if len(df_la) != 30 or len(df_zb) != 40:
        raise ValueError('Expected exactly 30 + 40 run records')
    if set(df_la['case']) & set(df_zb['case']):
        raise ValueError('Duplicate case names from independent implementations')
    all70 = pd.concat([df_la, df_zb], ignore_index=True)
    if len(all70) != 70 or all70.duplicated(['case','seed']).any():
        raise AssertionError('Expected exactly 70 unique case-by-seed records')
    if set(all70['seed']) != set(original.SEEDS):
        raise AssertionError('Seed mismatch')
    summary70 = _summary_all70(all70)
    all70.to_csv(out / 'expF_unified_all70.csv', index=False, encoding='utf-8-sig')
    summary70.to_csv(out / 'expF_unified_summary70.csv', index=False, encoding='utf-8-sig')
    df_la.to_csv(out / 'expF_original_all30.csv', index=False, encoding='utf-8-sig')
    df_zb.to_csv(out / 'expF_ZB_all40.csv', index=False, encoding='utf-8-sig')
    np.savez_compressed(out / 'expF_unified_series.npz', **arrays_la, **arrays_zb)
    failures = errors_la + errors_zb
    (out / 'expF_unified_errors.txt').write_text(
        '\n'.join(failures) if failures else 'No per-realization errors.', encoding='utf-8')
    provenance = {
        'experiment': 'Experiment F unified, one click, original 30 + full ZB 40',
        'run_mode': RUN_MODE, 'n_runs': len(all70),
        'source_La2004': str(source_la),
        'sha256_La2004': original.sha256(source_la),
        'source_ZB': {sol: {'path': str(zb.DATA / f'PT-{sol}_1-1.dat'),
                            'sha256': zb.sha256(zb.DATA / f'PT-{sol}_1-1.dat'),
                            'data_checks': sources_zb[sol][1]} for sol in zb.SOLUTIONS},
        'implementation_source_sha256': {
            'original_expF': hashlib.sha256(ORIGINAL_SOURCE.strip('\n').encode('utf-8')).hexdigest(),
            'ZB_full': hashlib.sha256(ZB_SOURCE.strip('\n').encode('utf-8')).hexdigest(),
            'integrated_plot': hashlib.sha256(PLOT_SOURCE.strip('\n').encode('utf-8')).hexdigest()},
        'algorithm_source': tuner_path,
        'seeds': list(original.SEEDS),
        'cases': list(original.CASES) + list(zb.CASES),
        'weights_la': original.WEIGHTS,
        'weights_zb': zb.WEIGHTS,
        'window_ma_before_present': [original.WINDOW_OLD_MA, original.WINDOW_YOUNG_MA],
        'true_sr_m_per_myr': original.SR_TRUE,
        'depth_step_m': original.DZ_M,
        'noise_ratio': original.NOISE_RATIO,
        'noise_rho': original.RHO,
        'SR_scan_grid': [5.0,30.0,0.25],
        'La2004_and_ZB_series': 'ZB solutions use their own e, obliquity, climatic precession',
        'ZB_component_stats': component_stats,
        'diagnostic_note': 'Forced anchors reported even if accepted gates reject the run; no results dropped.',
        'original_input_scripts_modified': False,
    }
    (out / 'expF_unified_provenance.json').write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    # Main integrated figure: 6 published-solution cases (60 records).
    map_la = {f'la2004_{mode}': ('La2004', mode) for mode in plotting.MODES}
    map_zb = {f'{sol}_{mode}': (sol, mode) for sol in ('ZB18a','ZB20a') for mode in plotting.MODES}
    visual = all70[all70['case'] != 'ideal_four_sines'].copy()
    visual[['solution','weight']] = visual['case'].map({**map_la,**map_zb}).apply(pd.Series)
    if len(visual) != 60 or visual[['solution','weight']].isna().any().any():
        raise AssertionError('Integrated plot must have 60 valid published-solution entries')
    table60 = plotting.make_summary(visual)
    visual.to_csv(out / 'expF_integrated_all60.csv', index=False, encoding='utf-8-sig')
    table60.to_csv(out / 'expF_integrated_summary.csv', index=False, encoding='utf-8-sig')
    carriers = {('La2004',m): clean_la[f'la2004_{m}'] for m in plotting.MODES}
    carriers.update({(sol,m): clean_zb[f'{sol}_{m}'] for sol in ('ZB18a','ZB20a') for m in plotting.MODES})
    png, svg = plotting.make_figure(fine_la, carriers, visual, table60, figout, original)

    # Secondary original baseline figure: ideal four sine and two La2004 groups.
    original.FDIR = figout
    original.draw_figure(fine_la, clean_la, df_la)
    (figout/'expF_astronomical_2x2.png').rename(figout/'expF_original_four_sine_La2004_2x2.png')
    (figout/'expF_astronomical_2x2.pdf').rename(figout/'expF_original_four_sine_La2004_2x2.pdf')

    print('\n=== COMPLETE: 70 RECOMPUTED EVALUATIONS ===', flush=True)
    print(summary70.to_string(index=False), flush=True)
    print('Failures:', len(failures), flush=True)
    print('Main integrated comparison:', png, flush=True)
    print('All 70 original per-seed results:', out / 'expF_unified_all70.csv', flush=True)
    print('Results folder:', out, flush=True)
    print('Figures folder:', figout, flush=True)


if __name__ == '__main__':
    main()
