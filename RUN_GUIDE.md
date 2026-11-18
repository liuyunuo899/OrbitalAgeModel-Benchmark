# Reproduction and GitHub upload guide

Commands are shown from the **`astrotuner/codes/`** directory unless noted. Use a clean copy of the archive if you want to regenerate outputs: most scripts use fixed output filenames and **may overwrite existing archived products**.

## 1. Core workflow and original experiments

| Target | Script | Related archived files |
|---|---|---|
| Core scanner, matching, 405-kyr anchoring | `astro_py_tuner.py` | Imported by other scripts |
| Synthetic generation, shared utilities | `exp_common.py` | Imported by original A–C scripts |
| A: sampled-band detection and SR recovery | `exp_a_detectability.py` | `expA_detection.csv`, `expA_sr_recovery.csv`, `expA_series.npz`; `figures/fig12.png` is stored separately in repository root `figures/` |
| A2: threshold sensitivity | `exp_a2_threshold.py` | `expA2_threshold_sensitivity.csv` |
| B: variable SR | `exp_b_variable_sr.py` | `expB_age_error.csv`, `expB_results.npz` |
| B2: Monte Carlo extension | `exp_b2_montecarlo.py` | `expB2_mc.csv`, `expB2_mc.npz`, `expB2_mc_summary.csv` |
| C: IMF splitting | `exp_c_mode_splitting.py` | `expC_imf_405_power.csv`, `expC_summary.csv` |
| C2: deployable diagnostics | `exp_c2_deployable.py` | `expC2_diag.csv`, `expC2_threshold_sweep.csv` |
| D: published-record comparisons | `exp_d_realdata.py` | `expD_*.csv`; requires `../data/real-datasets/` |
| Benchmark figure generation | `make_paper_figures.py` | `codes/figures/fig01.png` … `fig11.png` |
| Well-log and age demonstration | `EMD_auto_layer_with_AGE.py` | `codes/emd_astro_result/` |

The repository contains an `exp_c2_deployable_backup.py` historical backup and other legacy/diagnostic scripts. These are preserved for auditability, **not** necessarily the current manuscript's primary workflow.

A logical dependency order for the original experimental pipeline is:

```text
exp_a_detectability.py       exp_b_variable_sr.py
           |                        |
           |                  exp_b2_montecarlo.py
           +-----------+------------+
                       |
              exp_c_mode_splitting.py
                       |
               exp_c2_deployable.py

exp_d_realdata.py  (independent real-data application after correcting DATA path)
```

This is a dependency guide, **not an execution log or a claim that all scripts were rerun from this ZIP**.

## 2. Supplementary investigations and archived outputs

| Analysis | Script | Archived results / figure |
|---|---|---|
| A: vertical Gaussian response (FWHM 0, 0.3, 0.5 m) | `exp_a_detectability_combined.py` | `supplement_results/expA_vertical_response_combined/`, `supplement_figures/expA_vertical_response_2x2.png` and `.pdf` |
| A: aliasing / Nyquist phase | `exp_a_supp_alias_phase_validation.py` | `supplement_results/expA_supp_*`, `supplement_figures/expA_supp_alias_phase_validation.png` |
| A: additional response / alias diagnostic | `exp_a_resolution_aliasing_diagnostic_v3.py` | `supplement_figures/expA_supp_sampling_resolution_diagnostic.png`, associated CSV |
| D: reference-assisted SK-1n segmentation and Site 1264 SR check | `exp_d_realdata_combined_srcheck_v2.py` | `supplement_results/expD_sr_range_sensitivity_v2/`, `supplement_figures/expD_segmented_srcheck_2x2_v2.png` |
| E: single-hiatus sensitivity (70 synthetic trials) | `exp_e_hiatus_combined.py` | `supplement_results/expE_hiatus_combined_*`, `supplement_figures/expE_hiatus_2x2_comparison.png` and `.pdf` |
| F: La2004-based synthetic forcing (30 synthetic trials) | `exp_f_astronomical_solution.py` | `supplement_results/expF_*`, `supplement_figures/expF_astronomical_2x2.png` and `.pdf` |
| G: anchor spacing / reselection (240 evaluations from 30 archived B2 runs) | `exp_g_anchor_spacing_combined.py` | `supplement_results/expG_integrated_*`, `supplement_figures/expG_integrated_2x2.png` |

Some older alternative scripts exist (e.g. `exp_g_anchor_spacing.py`) and are preserved, without substituting them for the integrated scripts above.

### Suggested validation sequence (read-only where supported)

```bash
cd astrotuner/codes
python exp_a_detectability_combined.py --check-only
python exp_f_astronomical_solution.py --check-only
python exp_g_anchor_spacing_combined.py --check-only
python exp_d_realdata_combined_srcheck_v2.py --check-only
python exp_e_hiatus_combined.py --smoke-test
```

Review each script's options and dependencies before execution; some `--check-only` or smoke-test routines **may write their own diagnostic artifacts**. Never assume these commands are side-effect free without inspecting the script. For `exp_d_realdata.py`, correct the hard-coded `DATA` path first in your own working copy.

### Important scientific distinctions

- Original A: detection flags are assessed on the known synthetic time axis, not unknown field time; agreement with Nyquist flags does **not** establish recovered period identity.
- The assumed 0.3/0.5 m Gaussian FWHM responses are illustrative, **not actual tool response calibrations**.
- E: one prescribed synthetic nondeposition hiatus at fixed position; pass gates are not evidence of correct recovered missing time.
- F: La2004 orbital parameters are used to construct synthetic proxy forcing, **not observed sedimentary proxy data**.
- G: anchor thinning / prominence reselection **does not repeat EMD** or independently identify omitted cycles.
- D: published chronologies are comparison references and include astronomical tuning; SK-1n segmentation is prescribed using the published gap, **not predicted by the model**.

## 3. Uploading to GitHub

Unzip the archive, then run these commands **inside the extracted `astrotuner` directory** (replace placeholders yourself):

```bash
git init
git add .
git commit -m "Archive AstroTuner source data and reproducible analyses"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPOSITORY.git
git push -u origin main
```

Create the empty repository on GitHub first. Check the public redistribution conditions of third-party datasets and inspect locally embedded paths before publishing. The **compressed ZIP itself is not the repository tree**. A personal access token or SSH authentication may be required; never put credentials into source files.

This documentation was prepared by inspecting the supplied archive; no claims of successful end-to-end replication or verification against the final manuscript are made.
