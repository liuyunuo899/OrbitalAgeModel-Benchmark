# AstroTuner — EMD-based 405-kyr astronomical age-model benchmark

Repository of the Python source code, input datasets, archived numerical outputs, and figures accompanying the manuscript **“Evaluating 405-kyr astronomical age models using empirical mode decomposition across variable sedimentation regimes.”**

**Scope:** controlled tests of orbital-band detectability, sedimentation-rate recovery, 405-kyr age anchoring, EMD mode splitting, and diagnostic comparisons with SK-1n and ODP Site 1264. Supplementary sensitivity analyses address vertical response, aliasing/sampling phase, stratigraphic hiatuses, astronomical forcing complexity, anchor spacing, and reference-guided empirical segmentation.

> **Scientific interpretation:** A significant spectral match or a passed algorithmic gate is not proof of a correct chronology. Forced-anchor results and below-gate analyses are **diagnostic**, not accepted independent age reconstructions. The synthetic tests use prescribed conditions and should not be described as instrument-calibrated or as field validation.

## Repository layout

```text
astrotuner/
├── README.md                     # This overview
├── requirements.txt              # Python dependencies (not an exact environment lock)
├── RUN_GUIDE.md                  # Script-to-experiment/figure mapping and rerun order
├── DATA_AND_OUTPUTS.md           # Input provenance, archived products, scope notes
├── ORIGINAL_FILE_SHA256.tsv      # SHA-256 for all files in the uploaded archive
├── .gitignore                    # Ignore caches/local clutter; track archived data/results
├── codes/
│   ├── astro_py_tuner.py         # Core AstroTuner class
│   ├── exp_common.py             # Shared synthetic signal generator
│   ├── exp_[a-d]*.py             # Original experiments and diagnostics
│   ├── exp_e_hiatus_combined.py  # Supplement: hiatus sensitivity
│   ├── exp_f_astronomical_solution.py  # Supplement: La2004 forcing
│   ├── exp_g_anchor_spacing_combined.py # Supplement: anchor spacing
│   ├── figures/                 # Archived main benchmark figures (fig01–fig11)
│   ├── supplement_figures/      # Archived supplementary figure outputs
│   ├── supplement_results/      # Archived supplementary CSV/JSON/NPZ outputs
│   ├── astronomical_data/       # La2004 source data, source README
│   └── emd_astro_result/        # Archived well-log/decomposition outputs
├── data/
│   └── real-datasets/           # Public SK-1n, ODP Site 1264 and orbital input files
└── figures/                     # Additional archived figures (fig12–fig16)
```

The directory tree is intentionally preserved from the supplied archive. Some inputs and outputs appear in more than one location; these copies have **not** been deduplicated. `__pycache__` entries are present in the original upload but are not needed for execution. Some original PNG filenames may display incorrectly because of ZIP filename encoding; they were preserved rather than renamed.

## Quick start

Python 3.12 was used to generate bytecode found in the provided archive; an independently verified, locked Python environment is **not** included. Install the dependencies in a fresh environment:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
cd codes
python exp_a_detectability_combined.py --check-only
python exp_f_astronomical_solution.py --check-only
python exp_g_anchor_spacing_combined.py --check-only
```

The commands above are **suggested validation commands, not claims of successful execution on a clean machine**. For the run order and expected prerequisites, see [RUN_GUIDE.md](RUN_GUIDE.md).

## Key guidance

- **Do not run every script blindly.** Several scripts write files into `codes/`, `codes/figures/` or `codes/supplement_results/`, potentially replacing archived results if rerun. Work on a copy or commit/archive existing outputs first.
- **Portable-path issue:** `codes/exp_d_realdata.py` defines a Windows-local `DATA` directory. Set it to the extracted repository's `data/real-datasets/` path **in a working copy** before rerunning Experiment D. `codes/make_paper_figures.py` also has a Windows-local `BASE` path. These source files were left unchanged.
- **Dependencies:** `from PyEMD import EMD` is provided by the PyPI package **`EMD-signal`**, not by the unrelated package named `pyemd`.
- **Reproducibility:** Original outputs and added supplementary outputs remain distinguishable. E and F are synthetic sensitivity tests; G uses previously archived Experiment B2 simulations, not 240 independent new runs. Reference-guided SK-1n segmentation is not automatic hiatus detection.
- **Citation and reuse:** Cite the underlying scientific datasets and their original publications; see [DATA_AND_OUTPUTS.md](DATA_AND_OUTPUTS.md). The archive supplied no project LICENSE file. Consequently, this repository does **not** assert a new permission to reuse or redistribute third-party input datasets. Confirm redistribution conditions before making the repository public.

## Verification of archive preservation

`ORIGINAL_FILE_SHA256.tsv` lists the SHA-256 and byte size of every file in the original supplied ZIP. All of those files are included **byte-for-byte without modification** in the GitHub-ready ZIP; only the repository-level documentation files listed above were added.

## GitHub upload

Extract this ZIP, open a terminal inside the `astrotuner/` directory, and initialize/push the repository (or upload the extracted folders through your preferred GitHub workflow). GitHub does not directly turn an uploaded ZIP into a browsable source tree. See the GitHub instructions in [RUN_GUIDE.md](RUN_GUIDE.md).
