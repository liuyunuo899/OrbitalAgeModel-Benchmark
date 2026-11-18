# -*- coding: utf-8 -*-
"""
exp_a2_threshold.py — Round-1 supplement S3: significance-threshold sensitivity
================================================================================
Reviewer Round-1 criticism: Exp A used a single significance threshold (5)
for the MTM peak/neighbourhood-median detection test; the detectability
window and the 95% Nyquist agreement could be threshold artefacts.

This script reuses the archived per-band significances in expA_detection.csv
(no recomputation needed — the threshold is applied post hoc) and tabulates,
for thresholds T in {3, 4, 5, 6, 7,8}:
  * per-band detection rate (fraction of the 30 SR x noise combos detected);
  * overall detection rate (all 120 combo x band cells);
  * agreement with the theoretical Nyquist rule
    (lambda = P*SR/1000 >= 2*dz = 0.2 m): fraction of cells where
    (significance >= T) == nyquist_detectable.

Output: expA2_threshold_sensitivity.csv + console conclusion.
Run:  python3 exp_a2_threshold.py
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from exp_common import EXP_DIR, PERIODS_KYR

THRESHOLDS = [3, 4, 5, 6,7, 8]


def run():
    det = pd.read_csv(os.path.join(EXP_DIR, 'expA_detection.csv'))
    nyq = det['nyquist_detectable'].values.astype(bool)
    rows = []
    for T in THRESHOLDS:
        d = det['significance'].values >= T
        row = dict(threshold=T,
                   overall_rate=float(d.mean()),
                   nyquist_agree=float(np.mean(d == nyq)))
        for p in PERIODS_KYR:
            m = det['period_kyr'].values == p
            row[f'det_rate_{int(p)}kyr'] = float(d[m].mean())
        # lowest SR (per noise level) at which 21/40 kyr are still detected:
        for p in (21.0, 40.0):
            m = (det['period_kyr'].values == p) & d
            row[f'min_sr_detected_{int(p)}kyr'] = float(
                det.loc[m, 'sr_true'].min()) if m.any() else np.nan
        rows.append(row)
        print(f"  T={T}: overall {row['overall_rate']:.2f} "
              f"Nyquist agree {row['nyquist_agree']:.3f} | "
              + ' '.join(f"{int(p)}:{row[f'det_rate_{int(p)}kyr']:.2f}"
                         for p in PERIODS_KYR))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(EXP_DIR, 'expA2_threshold_sensitivity.csv'),
              index=False)

    # per-band x per-noise detection counts at each threshold (detail table)
    print('\n--- detection count by band x noise (of 10 SR levels) ---')
    for T in THRESHOLDS:
        d = det[det['significance'] >= T]
        ct = d.groupby(['period_kyr', 'noise']).size().unstack(fill_value=0)
        print(f'  T={T}:\n{ct.to_string()}\n')

    # where do the Nyquist disagreements live, per threshold?
    print('--- Nyquist disagreements (combo x band cells) ---')
    for T in THRESHOLDS:
        d = det['significance'].values >= T
        bad = det[d != nyq]
        print(f"  T={T}: {len(bad)} disagreements: "
              + '; '.join(f"SR={r.sr_true:g}/{r.noise}/{r.period_kyr:g}kyr"
                          f"(sig={r.significance:.1f})"
                          for r in bad.itertuples()))
    return df


if __name__ == '__main__':
    run()
