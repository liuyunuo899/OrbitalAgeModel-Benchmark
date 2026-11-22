# Exp D — Empirical application to public datasets (exp_d_realdata.py, fig15/fig16)

**Framework chain** (identical for both datasets): uniform grid -> Gaussian
detrend -> `scan_sed_rates` -> `match_quality` -> PyEMD `EMD(max_imf=10)` ->
`select_405_imf` -> `build_age_model_405` -> cross-comparison with the
published independent age model. Seeds fixed; EMD deterministic.
**Anchoring**: the 405-kyr model is a floating scale; the ONLY absolute-age
input is the published top-of-section age (single anchor), propagated with the
scan-optimal SR to the first IMF extremum; subsequent anchors are spaced
exactly 0.405 Myr. No other tie to the published model is used.

## SK-1n, Songliao Basin (fig15) — Th gamma log, Wu et al. 2015 EPSL
- Data: 12551 samples, 228.00-1796.75 m,
  uniform 0.125 m (kept at native sampling). Detrend: Gaussian low-pass,
  sigma = 40 m (cutoff wavelength ~3x the expected
  405-kyr wavelength of ~34 m; SR optimum insensitive to sigma = 25-90 m).
- SR grid 40-160 m/Myr (step 0.5); scan optimum **75.88 m/Myr**, match score **0.295**; IMF6 selected (50 cycles vs Ne=51.0, rel. dev 2.1%, MTM peak 417 kyr, matched=True); anchors(trough)=51
- Scan duration 20.67 Myr vs published
  18.85 Myr (mean SR 83.2 m/Myr).
- vs published ATS (mmc2, La2010d-tuned): 405-kyr model (trough phase)
  RMSE 1.804 Myr, max |dev| 4.904 Myr,
  bias -0.883 Myr; peak phase RMSE 1.883 Myr.
  Constant-SR linear at the scan optimum: RMSE 1.437 Myr,
  max |dev| 4.385 Myr.
- Interval SR between 405-kyr anchors: 46.0-188.9 m/Myr.
- **Published-ATS step**: the mmc2 file itself contains a
  +3.82 Myr age step at
  1021.6 m (hiatus or splice in the
  published supplement). Segment statistics: upper segment (mean published SR
  72 m/Myr): 405-kyr model
  RMSE 0.919 Myr (bias-removed
  0.446), linear
  0.287 Myr (bias-removed
  0.240); lower segment
  (mean published SR 193
  m/Myr): 405-kyr RMSE 2.393 Myr
  (bias-removed 2.189),
  linear 2.024 Myr
  (bias-removed 1.820).
  The published ATS thus implies mean SR rising from ~72
  to ~193 m/Myr across the step — a
  strongly nonstationary section that no single-anchor constant-cycle model
  can reproduce.

## ODP Site 1264, Walvis Ridge (fig16) — XRF CaCO3, Liebrand et al. 2016 EPSL
- Data: holes 1264A/B merged on armcd, bin-averaged to uniform 0.05 m,
  2213 samples, 207.21-317.81 m;
  87 empty bins (core gaps, max ~2 m) linearly
  interpolated. Detrend: Gaussian sigma = 4.5 m
  (SR optimum insensitive to sigma = 3-12 m).
- SR grid 2-20 m/Myr (step 0.1); scan optimum **9.50 m/Myr**, match score **0.276**; IMF5 selected (24 cycles vs Ne=28.7, rel. dev 16.5%, MTM peak 579 kyr, matched=False); anchors(trough)=25
- Scan duration 11.64 Myr vs published
  12.95 Myr (mean SR 8.5 m/Myr).
- `select_405_imf` returned **matched=False** (cycle deficit 24 vs Ne=28.7 and
  IMF MTM peak at ~579 kyr, outside 405 kyr +/-20%) -> per contract the reported
  primary model is the constant-SR linear fallback; the 405-kyr anchored model
  is retained as a diagnostic.
- vs published tie-point model (115 tie points, piecewise linear): linear
  fallback RMSE 1.111 Myr, max |dev| 1.465 Myr;
  405-kyr diagnostic (trough) RMSE 1.683 Myr,
  max |dev| 2.675 Myr; peak phase RMSE 1.772 Myr.

## Honesty notes
- Match scores are modest on both real datasets (0.28-0.30) versus ~0.9-1.0 on
  the synthetic matrix: the four-band timeOpt-style score is damped by strong
  non-orbital variance and (for 1264) eccentricity-dominated spectra where the
  40/21-kyr bands are weak at these sampling/SR combinations.
- SK-1n: the scan optimum (75.9 m/Myr) is ~9% below the published mean SR and
  the EMD IMF yields 50 cycles vs ~46.5 expected from the published ATS; both
  effects accumulate into Myr-scale drift of the single-anchor floating model.
  The residual profile is strongly structured with depth (segmented SR
  changes; Wu et al. likewise tuned segment-by-segment), so a constant-cycle
  spacing of 0.405 Myr cannot track the published ATS better than a linear
  model here.
- Site 1264: merged-hole bin averaging and gap interpolation are additional
  smoothing/error sources; the 405-kyr mode fails the framework's own dual
  criterion (count deficit + spectral peak check), which we report as a
  genuine non-detection rather than forcing a tuned model.
