#!/usr/bin/env python3
"""
sensitivity_weights_check.py — Round-2/3 review response harness.
Verifies that the real-signal vs white-noise discrimination of match_quality
does not depend on the band-weight choice (default 6/4/2.5/1.5 vs uniform 1/1/1/1).
One-off sensitivity harness (not part of the production pipeline); fixed seed for
reproducibility. Note: the production self-test chain reports 0.273 for its own
white-noise realization; this harness uses an independent realization (seed=123),
which yields 0.315-0.317. Both lie far below the 0.5 acceptance threshold.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/mnt/agents/output/emd_astro")
import astro_py_tuner as apt

SR_BEST = 14.989657015589426  # from sr_scan.csv (parabolic refinement)

def main():
    df = pd.read_excel("/mnt/agents/output/emd_astro/10井测井数据.xlsx", sheet_name="Sheet2")
    depth = df["深度"].values.astype(float)
    gr = df["GR"].values.astype(float)
    rng = np.random.default_rng(123)
    wn = rng.standard_normal(len(depth)) * np.std(gr)

    tuner = apt.AstroTuner()
    for weights in ([6, 4, 2.5, 1.5], [1, 1, 1, 1]):
        tuner.band_weights = np.array(weights, dtype=float)
        m_real = tuner.match_quality(depth, gr, SR_BEST)
        m_wn = tuner.match_quality(depth, wn, SR_BEST)
        print(f"weights={weights}: real={m_real:.3f}  white-noise={m_wn:.3f}  separation={m_real - m_wn:.3f}")

if __name__ == "__main__":
    main()
