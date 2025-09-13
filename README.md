# baygaud-PI

 _                                       _         ___ _____ 
| |__   __ _ _   _  __ _  __ _ _   _  __| |       / _ \\_   \
| '_ \ / _` | | | |/ _` |/ _` | | | |/ _` |_____ / /_)/ / /\/
| |_) | (_| | |_| | (_| | (_| | |_| | (_| |_____/ ___/\/ /_  
|_.__/ \__,_|\__, |\__, |\__,_|\__,_|\__,_|     \/   \____/  
             |___/ |___/                                     
                           v.2.0.0

**BAY**esian **GAU**ssian **D**ecomposer — a tool to decompose spectral-line profiles into multiple Gaussian components using Bayesian nested sampling.

- **Version:** 2.0.0 (2025-09-12)  
- **Author:** Se-Heon Oh (Department of Physics and Astronomy, Sejong University, South Korea)  
- **Contact:** seheon.oh@gmail.com

---

## Overview

**baygaud-PI** automatically determines the optimal number of Gaussians for each line-of-sight spectrum via Bayes factors, and produces 2D maps (velocity, dispersion, integrated intensity, N-Gauss, peak S/N, etc.) from an H I data cube. Outputs can be used for downstream kinematic analysis (e.g., rotation curves).

---

## Features

- Bayesian nested sampling (via **dynesty**) for robust model selection  
- **Classification of Gaussian components** using Bayes factors  
- **Ray**-based parallel processing for large cubes  
- **Segment-wise** saving and safe resume  
- Post-processing utilities to **merge segments** and **classify components**  
- Interactive **viewer** to inspect per-spectrum decompositions

---

## Requirements

- **Python:** 3.13 recommended (3.10–3.12 also work)  
- **OS:** Ubuntu 20.04.6 LTS (expected to work on most releases) and macOS 15.6.1 (Apple M2 Pro)  
- **Core packages:** `dynesty (2.1.5)`, `ray (2.x)`, `astropy`, `spectral-cube`, `numpy`, `matplotlib`, etc.  
- **System deps (Linux):** `libbz2-dev` (needed by `fitsio`)  
- **Viewer (optional):** `python3.x-tk`

> **Always use a dedicated virtual environment.** If any dependency does not yet support 3.13 on your system, use Python 3.10–3.12 with the same steps.

---

## Installation

Create and activate a fresh venv, then install:

```bash
# Clone
[seheon@Mac ~] git clone https://github.com/seheon-oh/baygaud-PI.git
[seheon@Mac ~] cd baygaud-PI

# Create a Python 3.13 venv in the repo (recommended)
[seheon@Mac baygaud-PI] python3.13 -m venv .venv313

# Activate (bash/zsh)
[seheon@Mac baygaud-PI] source .venv313/bin/activate
(.venv313) [seheon@Mac baygaud-PI] python --version

# OR activate (csh/tcsh)
[seheon@Mac baygaud-PI] source .venv313/bin/activate.csh
(.venv313) [seheon@Mac baygaud-PI] python --version
```

Linux-only prerequisites and install:

```bash
# System dependency (Ubuntu)
(.venv313) [seheon@Mac baygaud-PI] sudo apt-get install -y libbz2-dev

# Install package + pinned deps
(.venv313) [seheon@Mac baygaud-PI] pip install .

# (Optional) developer mode
(.venv313) [seheon@Mac baygaud-PI] python3 setup.py develop

# (Optional) viewer on Ubuntu
(.venv313) [seheon@Mac baygaud-PI] sudo apt-get install -y python3.13-tk
```

To leave the environment later: `deactivate`.

---

## Quick Start (venv strongly recommended)

### 1) Prepare your data & YAML

Put your H I data cube (FITS) in a working directory. Copy and edit a YAML template:

```bash
(.venv313) [seheon@Mac baygaud-PI] cp src/baygaud_pi/_baygaud_params.ngc2403.yaml my_params.yaml
(.venv313) [seheon@Mac baygaud-PI] vim my_params.yaml
```

**What you must edit (REQUIRED):**

- **Data paths**
  - `wdir`: absolute working directory (where the FITS lives)
  - `input_datacube`: FITS filename under `wdir`
  - `num_hanning_passes`: use `0` if unsure
- **ROI (inclusive indices)**
  - `naxis1_s0`, `naxis1_e0`, `naxis2_s0`, `naxis2_e0`  
  *(Start small for a quick test.)*
- **Thresholds & model size**
  - `max_ngauss`, `mom0_nrms_limit`, `peak_sn_pass_for_ng_opt`, `peak_sn_limit`, `int_sn_limit`
- **Classification bounds**
  - `g_vlos_lower`, `g_vlos_upper`, `g_sigma_upper`
- **Parallelization**
  - `num_cpus_ray` (don’t exceed physical cores)

> Keys like `g_sigma_lower`, `naxis1/2/3`, `cdelt*`, `vel_min/max`, `_rms_med`, `_bg_med` are filled at runtime — leave defaults.

**Minimal starter snippet:**

```yaml
# --- REQUIRED minimum you should customize ---
wdir: '/ABSOLUTE/PATH/TO/YOUR/WORKDIR'
input_datacube: 'your_cube.fits'
num_hanning_passes: 0

# small ROI for a quick test (inclusive)
naxis1_s0: 0
naxis1_e0: 40
naxis2_s0: 0
naxis2_e0: 40

# thresholds and model size
max_ngauss: 3
mom0_nrms_limit: 1
peak_sn_pass_for_ng_opt: 3
peak_sn_limit: 2
int_sn_limit: 1.0

# classification bounds
g_vlos_lower: 0
g_vlos_upper: 400
g_sigma_upper: 999

# parallelization
num_cpus_ray: 8
```

**Defaults you can usually keep (RECOMMENDED):**

- `dynesty`: `sample: 'rwalk'`, `bound: 'multi'`, `nlive: 100`, `dlogz: 0.01`, `update_interval: 2.0`, `facc/fmove/max_move`  
- **Matched filter & robust RMS** parameters  
- **SG-fit bounds:** `use_phys_for_x_bounds: true`, provided scale factors  
- **Tiling/batching:** `y_chunk_size`, `gather_batch`  
- **Threading:** set all thread counts to `1` to avoid oversubscription under Ray

### 2) Run baygaud

From the source tree:

```bash
(.venv313) [seheon@Mac baygaud-PI] cd src/baygaud_pi
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud.py ../my_params.yaml
```

*(Prompts are shown as examples; adjust paths as needed.)*

---

## Output layout & resume

Segment results are saved under `_segdir` (inside `wdir`):

```
segmts/
  G03.x10.ys10ye390.npy   # example: max_ngauss=3, column x=10, rows 10..390
```

If a run stops, you can **resume** by adjusting the ROI in your YAML to skip already-processed tiles:
`naxis1_s0`, `naxis1_e0`, `naxis2_s0`, `naxis2_e0`.

---

## Post-processing (merge & classify)

After segments are done, merge and classify components:

```bash
# Prints usage when run without args
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_classify.py

# Recommended: specify YAML and output index (1, 2, …)
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_classify.py ../my_params.yaml 1
```

This creates a directory like `segmts_merged_n_classified.1/` with classified components (`bulk`, `warm`, `hot`, `non_bulk`, `sgfit`, `psgfit`, `hvc`, etc.) and writes combined results:

- `baygaud_gfit_results.fits`  
- `baygaud_gfit_results.npy`

---

## Viewer

Inspect per-spectrum decompositions interactively:

```bash
# Prints usage when run without args
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_viewer.py

# With YAML (recommended) — use the same index as classification
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_viewer.py ../my_params.yaml 1
```

Tips: hover to locate spectra; mouse wheel to zoom; choose which 2D map to display (single-Gaussian VF, dispersion, integrated intensity, N-Gauss, peak S/N, etc.).

---

## Troubleshooting (quick notes)

- **CPU oversubscription:** keep all threading knobs at `1` when using Ray (`OMP/MKL/OPENBLAS/numba` threads).  
- **Python version:** if a dependency lags on 3.13, recreate the venv with 3.10–3.12.  
- **Long runs:** start with a **small ROI** to validate configuration and performance.

---

## Cite

If you use baygaud-PI, please cite the main algorithm and relevant applications:

1. Oh, S. H., Staveley-Smith, L., For, B. Q. (2019), **MNRAS**, 485, 5021–5034.  
2. Oh, S.-H., Kim, S., For, B.-Q., Staveley-Smith, L. (2022), **ApJ**, 928, 177.  
3. Park, H. J., Oh, S. H., Wang, J., et al. (2022), **AJ**, 164, 82.  
4. Kim, S. J., Oh, S. H., et al. (2023), **MNRAS** (WALLABY Pilot).  
5. Kim, M., Oh, S. H. (2022), **JKAS**, 55, 149–172.  
6. Wang, J., Yang, D., Oh, S.-H., et al. (2023), **ApJ**.  
7. Oh, S.-H., Wang, J. (2025), **MNRAS**, 538, 1816.

---

## Support & Issues


Happy decomposing!!!
