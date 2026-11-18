# Data sources, preservation, and output interpretation

This repository preserves the input and output files as supplied in the `astrotuner.zip` archive. No raw data, figure, or source-code bytes were modified to create the GitHub-ready package.

## Inputs supplied in `data/real-datasets/`

| Files | Role | Literature / provenance |
|---|---|---|
| `SK1n_Wu2015_EPSL_Th_mmc1.txt` and `SK1n_Wu2015_EPSL_mmc2.txt` | SK-1n thorium gamma-ray series and the *published astronomically tuned* age reference | Wu et al. (2015), *Earth and Planetary Science Letters* 407:82–95. DOI: `10.1016/j.epsl.2014.09.038` |
| `PANGAEA_862583.txt` and `PANGAEA_862580.txt` (also supplied as `.tab`) | ODP Site 1264 CaCO3 series and published tuning age-control information | Liebrand et al. (2016), *Earth and Planetary Science Letters*; PANGAEA dataset DOIs `10.1594/PANGAEA.862583` and `10.1594/PANGAEA.862580` |
| `INSOLN.LA2004.BTL.ASC` | Published orbital solution input used in synthetic Experiment F | La2004 astronomical solution (Laskar et al., 2004); another copy in `codes/astronomical_data/` with `README.txt` |
| `La2010a_ecc3L.dat` | Supplied astronomical solution input | La2010a solution; the file is preserved but not automatically assumed to be a direct input to Experiment F |
| Original well-log `.xlsx` under `data/` | Original demonstration input; filename encoding in the ZIP may appear damaged | See `codes/EMD_auto_layer_with_AGE.py` and `codes/synthetic_benchmark_well10_regenerated.xlsx` |

The archived reference chronologies have their **own** dating assumptions and uncertainty; they should not be described as completely independent absolute-time ground truth.

## Original numerical outputs

- `codes/expA*`, `expB*`, `expC*`, `expD*`: original synthetic and empirical experiment CSV/NPZ outputs and sensitivity results.
- `codes/emd_astro_result/`: saved well-log decomposition outputs and related figures.
- `codes/figures/fig01.png`–`fig11.png`: archived figure renders.
- `figures/fig12.png`–`fig16.png`: additional archived figure renders; these are distinct from `codes/figures/`.
- `data/`: includes a second set of original data and some earlier computation products, kept unchanged for provenance.

## Supplementary products

- `codes/supplement_results/`: CSV, JSON, text, and NPZ derived values. Original outputs are stored separately from supplementary results, with filenames indicating source analysis.
- `codes/supplement_figures/`: corresponding PNG and some vector PDF plots.
- Supplementary La2004 forcing and hiatus examples are **synthetic reconstructions**, not measured sedimentary profiles.
- The Exp G 240 records are **derived evaluations** on 30 earlier simulation realizations, not 240 independent experimental replicates.

## Verification and archiving

`ORIGINAL_FILE_SHA256.tsv` contains a line for every non-directory item in the user's uploaded ZIP. It enables byte-by-byte verification that the original code, figures, data and outputs have been preserved.

### Publication / license caveat

The original archive contains third-party datasets and includes no explicit `LICENSE` file. Merely copying them into a public repository does not establish a redistribution license. Before publishing, verify that you may publicly rehost each third-party file and select a license for the original project code only after the rights holder(s) approve it.
