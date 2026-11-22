# -*- coding: utf-8 -*-
"""Experiment D: published-age-reference segmentation diagnostics for Reviewer 1 Q6.

Place beside the ORIGINAL exp_d_realdata.py and astro_py_tuner.py.
Four unmodified raw data files remain in ../data/real-datasets/.

Usage (Spyder):
  %runfile 'F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes/exp_d_realdata_combined_srcheck.py' --wdir --args "--check-only"
  %runfile 'F:/刘雨诺/论文投稿/非线性沉积速率emd-astro/codes/exp_d_realdata_combined_srcheck.py' --wdir

Scientific boundaries:
* SK-1n separation uses a discontinuity in the PUBLISHED ATS, not an automatically
  inferred hiatus. Each independently fitted section uses its published starting
  age as a DIAGNOSTIC reference; this is not autonomous astronomical dating.
* Site 1264 is evaluated in its uppermost 10 m and remainder USING THE EXISTING
  full-section model, not with an under-resolved 10-m EMD refit.
* Below-gate model results are diagnostics, not accepted AstroTuner outputs.
* Existing original and previous Q6 supplementary results are NOT overwritten.
* Paired scans compare 40-160 vs 40-240 m/Myr for BOTH SK-1n segments.
  The expanded scan is a diagnostic sensitivity check, not a tuned algorithm.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from exp_d_realdata import load_sk1n, load_1264, run_chain

OUTPUT = HERE / 'supplement_results' / 'expD_sr_range_sensitivity_v2'
FIG_DIR = HERE / 'supplement_figures'
SK_STEP_APPROX_M = 1021.6
SITE_TOP_WINDOW_M = 10.0
M_GATE = 0.5

COLORS = {'published':'#475569','linear':'#3D6484','anchored':'#B56B54'}

def rmse(x):
    a = np.asarray(x, float)
    return float(np.sqrt(np.mean(a*a)))

def stats(name, label, depth, truth, lin, anch, match=np.nan, sr=np.nan,
          accepted=None, diagnostic_reference='published_age_model'):
    d, t, l, a = [np.asarray(v, float) for v in (depth,truth,lin,anch)]
    if not (len(d)==len(t)==len(l)==len(a)) or len(d)<2:
        raise ValueError(f'Invalid arrays for {name}/{label}')
    return dict(dataset=name, segment=label, n_samples=len(d),
                depth_min_m=float(d[0]), depth_max_m=float(d[-1]),
                published_start_age_ma=float(t[0]),
                published_end_age_ma=float(t[-1]),
                published_interval_effective_sr_m_myr=float((d[-1]-d[0])/(t[-1]-t[0])),
                sr_scan_opt_m_myr=float(sr), match_score=float(match),
                stage1_pass=(bool(match>=M_GATE) if np.isfinite(match) else None),
                imf_dual_gate_pass=accepted,
                rmse_linear_myr=rmse(l-t), rmse_forced_anchor_myr=rmse(a-t),
                maxabs_linear_myr=float(np.max(np.abs(l-t))),
                maxabs_forced_anchor_myr=float(np.max(np.abs(a-t))),
                reference_use=diagnostic_reference)

def load_archived_models(fname, depth, truth, expected_dcol, expected_tcol):
    path = HERE / fname
    if not path.exists():
        raise FileNotFoundError(
            f'Missing {path}. Run the ORIGINAL exp_d_realdata.py once to regenerate '
            'original expD_*_agemodel.csv outputs, then rerun THIS combined script. '
            'Do not change original input data or overwrite the scripts.')
    df = pd.read_csv(path)
    for col in (expected_dcol,expected_tcol,'age_linear_ma','age_emd405_ma'):
        if col not in df.columns:
            raise ValueError(f'{fname} lacks column {col}')
    if len(df)!=len(depth) or not np.allclose(df[expected_dcol],depth,atol=1e-5,rtol=0):
        raise ValueError(f'{fname}: depth grid differs from original data loader')
    if not np.allclose(df[expected_tcol],truth,atol=1e-5,rtol=0):
        raise ValueError(f'{fname}: published chronology differs from data loader')
    if not np.isfinite(df[['age_linear_ma','age_emd405_ma']].to_numpy()).all():
        raise ValueError(f'{fname}: non-finite age models')
    return df

def split_sk(depth, age):
    delta = np.diff(age)
    i = int(np.argmax(delta))
    step=float(delta[i]); mid=float((depth[i]+depth[i+1])/2)
    if step < 0.5 or abs(mid-SK_STEP_APPROX_M)>2.0:
        raise ValueError('Published SK-1n age discontinuity not matched to 1021.6 m; investigate raw ATS, do not force a split')
    return i,mid,step

def figure(rows, skfull, skupper, sklower, site, site_top_mask, split_at, outfile):
    fig, axs=plt.subplots(2,2,figsize=(12.8,9.1),constrained_layout=True)
    a,b,c,d=axs.ravel()
    # SK-1n published chronology and full-section model (raw, unchanged)
    z=skfull['depth_m'].to_numpy(float)
    a.plot(skfull['age_pub_ats_ma'],z,lw=1.6,c=COLORS['published'],label='Published ATS')
    a.plot(skfull['age_linear_ma'],z,lw=1,c=COLORS['linear'],ls='--',label='Original full-section linear')
    a.plot(skfull['age_emd405_ma'],z,lw=1,c=COLORS['anchored'],ls=':',label='Original full-section forced anchor')
    a.axhline(split_at,c='k',lw=1,ls='--',alpha=.7)
    a.set_ylim(z.max(),z.min());a.set_xlabel('Age (Ma)');a.set_ylabel('SK-1n depth (m)')
    a.set_title('(a) SK-1n: published hiatus and original models')
    # SK-1n independently refitted pieces show local-relative age errors
    for rec,clr,label in [(skupper,COLORS['linear'],'Upper'),(sklower,COLORS['anchored'],'Lower')]:
        zz=rec['depth']; tt=rec['age_pub'];
        b.plot(zz,rec['age_lin']-tt,c=clr,ls='-',lw=1.25,label=f'{label} linear')
        b.plot(zz,rec['age405_trough']-tt,c=clr,ls='--',lw=.95,label=f'{label} forced anchor')
    b.axhline(0,c='#666',lw=.8)
    b.axvline(split_at,c='k',lw=.8,ls=':')
    b.set_xlabel('SK-1n depth (m)');b.set_ylabel('Age residual vs published (Myr)')
    b.set_title('(b) Split-section refits (reference-constrained)')
    # Site 1264 published age change in top 10m, local effective SR
    dep=site['depth_armcd_m'].to_numpy(float)
    tt=site['age_pub_tiepoints_ma'].to_numpy(float)
    c.plot(dep,tt,c=COLORS['published'],label='Published tie-point chronology',lw=1.5)
    c.plot(dep,site['age_linear_ma'],c=COLORS['linear'],ls='--',label='Original linear',lw=1)
    c.plot(dep,site['age_emd405_ma'],c=COLORS['anchored'],ls=':',label='Original forced anchor',lw=1)
    c.axvline(dep[0]+SITE_TOP_WINDOW_M,c='k',ls='--',lw=1)
    c.set_xlim(dep[0],min(dep[-1],dep[0]+35))
    c.set_xlabel('Site 1264 depth (rmcd m)');c.set_ylabel('Age (Ma)')
    c.set_title('(c) Site 1264: upper 10 m (zoom)')
    # Both record model residual absolute RMS within windows (existing model on 1264)
    vals=[];names=[]
    for lab,mask in [('Top 10m',site_top_mask),('Below 10m',~site_top_mask)]:
        orig=site.loc[mask]
        vals.append([rmse(orig['age_linear_ma']-orig['age_pub_tiepoints_ma']),
                     rmse(orig['age_emd405_ma']-orig['age_pub_tiepoints_ma'])]);names.append(lab)
    x=np.arange(2);width=.32
    d.bar(x-width/2,[v[0] for v in vals],width,color=COLORS['linear'],label='Constant-SR linear')
    d.bar(x+width/2,[v[1] for v in vals],width,color=COLORS['anchored'],label='Forced 405-kyr anchor')
    d.set_xticks(x,names);d.set_ylabel('Age RMSE (Myr)')
    d.set_title('(d) Site 1264: original-model residual RMSE')
    for ax in axs.ravel():
        h,l=ax.get_legend_handles_labels()
        if h: ax.legend(loc='best',fontsize=8,frameon=True)
        ax.grid(alpha=.15)
    fig.savefig(outfile,dpi=220,bbox_inches='tight')
    plt.close(fig)

# Two predeclared scan settings, identical grid spacing and processing parameters.
SCAN_CONFIGS = {'original_40_160': (40.0, 160.0, 0.5),
                'expanded_40_240': (40.0, 240.0, 0.5)}


def plot_scan_comparison(refits, dest):
    """Audit the raw scan OBJECTIVE J versus SR (not the Stage-1 match score M)."""
    fig, axs = plt.subplots(1, 2, figsize=(12.2, 4.3), constrained_layout=True)
    for ax, seg in zip(axs, ('upper', 'lower')):
        for key, color, ls in [('original_40_160','#3D6484','-'),
                               ('expanded_40_240','#B56B54','--')]:
            record, out = refits[(seg, key)]
            df = out['scan']
            if not {'sr', 'score'}.issubset(df.columns):
                raise ValueError(f'Unexpected scan columns: {list(df.columns)}')
            ax.plot(df['sr'].to_numpy(float), df['score'].to_numpy(float),
                    lw=1.4, color=color, ls=ls,
                    label=f'{key.replace("_", "–")}: SRopt={record["sr_opt"]:.2f}')
        ax.set_title(f'SK-1n {seg}: SR-scan objective')
        ax.set_xlabel('Candidate SR (m/Myr)')
        ax.set_ylabel('Original scan objective J (not match M)')
        ax.grid(alpha=.15)
        ax.legend(loc='best', fontsize=8)
    fig.savefig(dest, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main(check_only=False):
    depth,th,age_sk=load_sk1n()
    site_dep,ca,age_site,ties,n_gaps=load_1264(dz=.05)
    i,step_depth,jump=split_sk(depth,age_sk)
    print(f'DATA PASS: SK-1n {len(depth)} points; Site 1264 {len(site_dep)} points; interpolated bins={n_gaps}')
    print(f'PUBLISHED ATS STEP: depth={step_depth:.4f} m, age jump={jump:.3f} Myr')
    full_sk=load_archived_models('expD_sk1n_agemodel.csv',depth,age_sk,'depth_m','age_pub_ats_ma')
    full_site=load_archived_models('expD_1264_agemodel.csv',site_dep,age_site,'depth_armcd_m','age_pub_tiepoints_ma')
    print('REPRODUCIBILITY PASS: archived original full-section model CSVs match source data')
    cut=site_dep[0]+SITE_TOP_WINDOW_M
    masktop=site_dep < cut
    if masktop.sum()<3 or (~masktop).sum()<3: raise ValueError('Site 1264 window too small')
    if check_only:
        print('CHECK ONLY PASS: no new outputs written')
        return

    rows=[]
    for lab,sl in [('original_full_upper',slice(0,i+1)),('original_full_lower',slice(i+1,None))]:
        rr=full_sk.iloc[sl]
        record=stats('SK-1n',lab,rr.depth_m,rr.age_pub_ats_ma,rr.age_linear_ma,
                     rr.age_emd405_ma,diagnostic_reference='original_full_section')
        record['scan_setting']='original_full_section'
        rows.append(record)

    refits={}
    for label,sl in [('upper',slice(0,i+1)),('lower',slice(i+1,None))]:
        dep=depth[sl]; sig=th[sl]; pub=age_sk[sl]
        for setting,(lo,hi,st) in SCAN_CONFIGS.items():
            print(f'RUN SK-1n {label}: {setting}, grid={lo:g}..{hi:g} step={st:g} m/Myr')
            rec,out=run_chain('SK-1n_'+label+'_'+setting,dep,sig,pub,
                              sr_grid=np.arange(lo,hi+st/2,st),sigma_m=40.0,dz=.125)
            refits[(label,setting)]=(rec,out)
            row=stats('SK-1n','split_refit_'+label,dep,pub,out['age_lin'],
                      out['age405_trough'],rec['match_score'],rec['sr_opt'],rec['imf_matched'],
                      diagnostic_reference='published_segment_start_age__reference_assisted')
            row['scan_setting']=setting
            rows.append(row)
            print(f'  SR={rec["sr_opt"]:.3f}; M={rec["match_score"]:.3f}; '
                  f'IMF_gate={rec["imf_matched"]}; RMSE linear={rec["rmse_linear"]:.3f}; '
                  f'forced anchor={rec["rmse_405_trough"]:.3f} Myr')
    for label,mask in [('top_10m',masktop),('below_10m',~masktop)]:
        rr=full_site.loc[mask]
        row=stats('Site 1264',label,rr.depth_armcd_m,rr.age_pub_tiepoints_ma,
                  rr.age_linear_ma,rr.age_emd405_ma,
                  diagnostic_reference='published_tiepoint_model_for_evaluation_only')
        row['scan_setting']='original_full_section_only'
        rows.append(row)
        print(f'Site 1264 {label}: published effective SR={row["published_interval_effective_sr_m_myr"]:.3f}; '
              f'linear RMSE={row["rmse_linear_myr"]:.3f}; forced RMSE={row["rmse_forced_anchor_myr"]:.3f}')

    # Keep all diagnostics separate from original and earlier supplementary outputs.
    OUTPUT.mkdir(parents=True,exist_ok=True)
    FIG_DIR.mkdir(parents=True,exist_ok=True)
    met=OUTPUT/'expD_sr_range_metrics.csv'
    pd.DataFrame(rows).to_csv(met,index=False)
    compare=[]
    for label in ('upper','lower'):
        a=next(r for r in rows if r['segment']=='split_refit_'+label and r['scan_setting']=='original_40_160')
        b=next(r for r in rows if r['segment']=='split_refit_'+label and r['scan_setting']=='expanded_40_240')
        compare.append({'segment':label,'original_sr_opt':a['sr_scan_opt_m_myr'],
                       'expanded_sr_opt':b['sr_scan_opt_m_myr'],
                       'original_match':a['match_score'],'expanded_match':b['match_score'],
                       'original_gate_pass':a['imf_dual_gate_pass'],'expanded_gate_pass':b['imf_dual_gate_pass'],
                       'original_linear_rmse_myr':a['rmse_linear_myr'],
                       'expanded_linear_rmse_myr':b['rmse_linear_myr'],
                       'original_forced_rmse_myr':a['rmse_forced_anchor_myr'],
                       'expanded_forced_rmse_myr':b['rmse_forced_anchor_myr'],
                       'delta_linear_rmse_myr':b['rmse_linear_myr']-a['rmse_linear_myr'],
                       'delta_forced_rmse_myr':b['rmse_forced_anchor_myr']-a['rmse_forced_anchor_myr']})
    pd.DataFrame(compare).to_csv(OUTPUT/'expD_sr_range_comparison.csv',index=False)
    for (label,setting),(rec,out) in refits.items():
        pd.DataFrame({'depth_m':out['depth'],'age_pub_ma':out['age_pub'],
                      'age_linear_ma':out['age_lin'],'age_forced405_ma':out['age405_trough'],
                      'residual_linear_myr':out['age_lin']-out['age_pub'],
                      'residual_forced405_myr':out['age405_trough']-out['age_pub']}).to_csv(
                       OUTPUT/f'expD_sk1n_{label}_{setting}_refit.csv',index=False)
        out['scan'].to_csv(OUTPUT/f'expD_sk1n_{label}_{setting}_scan.csv',index=False)

    # Same Q6 four-panel presentation, using the EXPANDED scans on panel b.
    wide_up=refits[('upper','expanded_40_240')][1]
    wide_lo=refits[('lower','expanded_40_240')][1]
    figfile=FIG_DIR/'expD_segmented_srcheck_2x2_v2.png'
    figure(rows,full_sk,wide_up,wide_lo,full_site,masktop,step_depth,figfile)
    # Extra scan audit is separate; it does not have to be added to the manuscript.
    scanfig=FIG_DIR/'expD_sr_range_scan_audit_corrected.png'
    plot_scan_comparison(refits,scanfig)
    design=dict(source_script='exp_d_realdata.py',
                original_supplement_script='exp_d_realdata_combined.py',
                sk_step_depth_m=step_depth,sk_pub_step_myr=jump,
                site_top_window_m=SITE_TOP_WINDOW_M,
                sk_segment_scan_settings=SCAN_CONFIGS,
                sk_segment_detrend_sigma_m=40,
                site_segment_strategy='evaluate_original_full_section_models_only',
                sk_segment_age_origin='published_age_at_first_sample_of_each_segment__reference_assisted',
                gate_threshold=M_GATE,source_inputs='four original downloaded TXT files',
                warning='No automatic hiatus detection; reference-assisted segment origins; below-gate forced anchoring diagnostic only')
    (OUTPUT/'expD_sr_range_design.json').write_text(json.dumps(design,indent=2,ensure_ascii=False),encoding='utf-8')
    print('RANGE COMPARISON:')
    print(pd.DataFrame(compare).to_string(index=False))
    print('SAVED:',met)
    print('SAVED:',OUTPUT/'expD_sr_range_comparison.csv')
    print('SAVED:',figfile)
    print('SAVED:',scanfig)
    print('PASS: original Experiment D and previous supplementary results were not modified')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--check-only',action='store_true',help='Verify raw data and original models only')
    args=p.parse_args()
    main(args.check_only)
