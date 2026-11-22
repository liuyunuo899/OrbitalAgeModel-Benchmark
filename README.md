# AstroTuner — EMD-based 405-kyr astronomical age-model benchmark

This repository provides Python source code, input-data documentation, archived numerical outputs, and figures accompanying the manuscript **“Evaluating 405-kyr astronomical age models using empirical mode decomposition across variable sedimentation regimes.”**

The benchmark evaluates orbital-band detectability, sedimentation-rate (SR) recovery, EMD-based 405-kyr anchoring, mode splitting, and diagnostic comparisons with SK-1n and ODP Site 1264. Supplementary analyses examine vertical response, aliasing and sampling phase, sedimentary hiatuses, astronomical forcing, anchor spacing, and reference-guided empirical segmentation.

> **Interpretation.** A spectral match or passed algorithmic acceptance criterion is not, by itself, proof of a correct chronology. The forcing-based tests use synthetic proxy responses constructed from published astronomical solutions, not observed stratigraphic proxy records. Forced-anchor calculations from rejected runs are retained as *diagnostic* outputs.

## Repository layout

```text
astrotuner/
├── README.md                         # Repository overview
├── requirements.txt                  # Python dependencies (not a locked environment)
├── RUN_GUIDE.md                      # Experiment-to-script mapping and rerun guidance
├── DATA_AND_OUTPUTS.md               # Data provenance and output descriptions
├── ORIGINAL_FILE_SHA256.tsv          # Checksums for the original supplied archive
├── .gitignore
├── codes/
│   ├── astro_py_tuner.py             # Core AstroTuner implementation
│   ├── exp_common.py                 # Shared synthetic signal generator
│   ├── exp_[a-d]*.py                 # Core experiments A–D and diagnostics
│   ├── exp_e_hiatus_combined.py      # Experiment E: hiatus sensitivity
│   ├── exp_f_unified_oneclick_supplement_folders.py
│   │                                 # Experiment F: four-sinusoid, La2004,
│   │                                 # ZB18a and ZB20a; 7 cases × 10 seeds
│   ├── exp_g_anchor_spacing_combined.py
│   │                                 # Experiment G: anchor-spacing sensitivity
│   ├── figures/                      # Archived principal benchmark figures
│   ├── supplement_figures/           # Supplementary figures, including F comparison
│   ├── supplement_results/           # Supplementary CSV/JSON/NPZ results
│   ├── astronomical_data/            # Astronomical solution inputs and provenance
│   └── emd_astro_result/             # Archived decomposition/well-log outputs
├── data/
│   └── real-datasets/                # SK-1n, ODP Site 1264 and relevant inputs
└── figures/                          # Additional archived manuscript figures
```

The tree above describes the **intended updated layout**. Before publication, check that the unified Experiment F script, required astronomical inputs, and supplementary outputs have been committed. Historical scripts may also remain in the repository as archival files; they are not required for the unified Experiment F run. The original archive's checksums should not be rewritten to imply that newly added files belonged to the earlier archive.

## Quick start

Create a Python environment and install the supplied requirements:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The requirements file is not a fully locked, independently validated environment. `from PyEMD import EMD` requires the PyPI package **EMD-signal**, not the unrelated `pyemd` package.

### Experiment F — unified one-click reproduction

Open `codes/exp_f_unified_oneclick_supplement_folders.py` in Spyder and select **Run (F5)**. The default full-run configuration executes all seven Experiment F configurations and generates the numerical results and figures without command-line flags.

Alternatively, from a terminal in the repository root:

```bash
cd codes
python exp_f_unified_oneclick_supplement_folders.py
```

Experiment F comprises **70 realizations**: ten runs of the original four-sinusoid reference plus ten paired-seed runs for each of two proxy-response weighting schemes (eccentricity-dominant and balanced) for **La2004, ZB18a, and ZB20a**. In each published-solution case, eccentricity, obliquity, and climatic precession come from the *same corresponding astronomical solution*. The workflow and acceptance criteria are kept fixed; proxy-response weights are prescribed synthetic assumptions.

New outputs are placed beneath the original supplementary directories:

- `codes/supplement_results/` — per-run data, experiment summaries and provenance records, in run-specific subdirectories.
- `codes/supplement_figures/` — figures, including the integrated La2004/ZB18a/ZB20a comparison, in run-specific subdirectories.

The integrated four-panel figure displays the three published-solution configurations (six weighting cases). The four-sinusoid reference is reported in the accompanying Experiment F numerical summary rather than as a seventh group in that integrated figure. **Stage-1 acceptance, full acceptance and chronological RMSE are different metrics**; forced-anchor RMSE distributions include diagnostic calculations that failed acceptance gates.

Required input datasets must be present in `codes/astronomical_data/` at paths expected by the unified script. In particular, check the La2004 source data and the complete PT-ZB18a/PT-ZB20a solution input files. See `DATA_AND_OUTPUTS.md` for input provenance; consult the data providers' terms before redistribution.

## Other experiments

Use `RUN_GUIDE.md` for scripts, their expected prerequisites, and the reproduction order of Experiments A–E and G. Experiment E tests single-hiatus sensitivity; Experiment G examines anchor-spacing post-processing of existing Experiment B2 realizations (240 derived evaluations, **not** 240 independent new simulations).

Some original scripts contain machine-specific Windows file paths, notably `codes/exp_d_realdata.py` and `codes/make_paper_figures.py`. Adapt those paths **in a separate working copy** before trying to reproduce their outputs.

## Data integrity and reproducibility

- Preserve original input data, seed assignments, unmodified archival outputs, and original checksums.
- The unified Experiment F run should write to its own output subdirectories instead of overwriting earlier results.
- Do not interpret synthetic forcing experiments as field validation or instrument-calibrated measurements.
- In the constant-true-SR Experiment F simulations, a constant-SR reference is expected to have a favorable baseline. Report accepted runs and diagnostic forced-anchor errors distinctly.
- `ORIGINAL_FILE_SHA256.tsv` documents the **original supplied archive**; newly generated or added files should be documented separately rather than silently entered as original archive members.

For data descriptions and external sources, see [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md). For script-level reproduction instructions, see [RUN_GUIDE.md](RUN_GUIDE.md).

## Citation and license

Please cite the accompanying manuscript and the original publications for the astronomical solutions and empirical datasets used. This repository contains a `LICENSE` file; third-party astronomical and stratigraphic datasets may have **separate** reuse and redistribution terms. Check those terms independently of the repository code license.
