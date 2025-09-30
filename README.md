# baygaud-PI

```
 _                                       _         ___ _____ 
| |__   __ _ _   _  __ _  __ _ _   _  __| |       / _ \\_   \
| '_ \ / _` | | | |/ _` |/ _` | | | |/ _` |_____ / /_)/ / /\/
| |_) | (_| | |_| | (_| | (_| | |_| | (_| |_____/ ___/\/ /_  
|_.__/ \__,_|\__, |\__, |\__,_|\__,_|\__,_|     \/   \____/  
             |___/ |___/                                     
                           v.2.0.0
```

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
- **OS:** Tested on Ubuntu 20.04.6 LTS (expected to work on most releases) and macOS 15.6.1 (Apple M2 Pro)  
- **Core packages:** `dynesty (2.1.5)`, `ray (2.x)`, `astropy`, `spectral-cube`, `numpy`, `matplotlib`, etc.  
- **Tkinter note (or the viewer only):** `python3.x-tk` (see Tkinter installation note below)
  > `baygaud_viewer.py` requires **Tkinter**. The core pipeline (`baygaud.py`, `baygaud_classify.py`) does **not** need it.
  >

> **Always use a dedicated virtual environment.** If any dependency does not yet support 3.13 on your system, use Python 3.10–3.12 with the same steps.

---

## 🟢 Installation

### 1) Create a Python 3.13 venv in your prefered directory, then install: (recommended)

#### 1-1) macOS (recommended): Install Python 3.13 from python.org — it includes Tkinter out of the box. Then create your venv with that interpreter:  

```bash
  [seheon@Mac project] /Library/Frameworks/Python/Frameworks/Versions/3.13/bin/python3 -m venv .venv313

  # Check the created .venv313
  [seheon@Mac project] ls -al
  ...
  ...
  drwxr-xr-x@  8 seheon  staff    256 Sep 10 15:23 .venv313
  ...
  ...

  # Activate (bash/zsh) --------
  [seheon@Mac project] source .venv313/bin/activate
  (.venv313) [seheon@Mac project] python --version
  Python 3.13.7

  # OR activate (csh/tcsh) -----
  [seheon@Mac project] source .venv313/bin/activate.csh
  (.venv313) [seheon@Mac project] python --version
  Python 3.13.7


  > Quick Tkinter check: 

  (.venv313) [seheon@Mac project] python -c "import tkinter; import tkinter as tk; print('Tk OK, version=', tk.TkVersion)"

  > If 'Tk OK' printed, the current python supports Tkinter.

  > Homebrew’s `python@3.13` may not ship Tkinter. If you prefer Homebrew, install a Tk-enabled Python (e.g., `python-tk@3.12`) and run baygaud_viewer.py under that version.
  ```

  #### 1-2) Linux (Debian/Ubuntu):

  ```bash
  [seheon@Mac project] sudo apt update
  [seheon@Mac project] sudo apt install -y python3.13-tk
  ```

  > If have just installed Python 3.13 on Ubuntu and want to set up baygaud-PI cleanly without conflicts from the system’s older Python (e.g., 3.8), please follow these steps to prepare a clean environment before installing baygaud-PI. With these steps, you ensure that your installation is fully isolated in Python 3.13 and will not conflict with the system’s default Python (e.g., 3.8). Or you go jump to ((### 2) Clone baygaud-PI from github below)

  ```bash
  # (1) Create and activate a Python 3.13 virtual environment (e.g., .venv313)
  [seheon@Mac project] python3.13 -m venv .venv313

  # Check the created .venv313
  [seheon@Mac project] ls -al
  ...
  ...
  drwxr-xr-x@  8 seheon  staff    256 Sep 10 15:23 .venv313
  ...
  ...

  # Activate (bash/zsh) --------
  [seheon@Mac project] source .venv313/bin/activate
  (.venv313) [seheon@Mac project] python --version
  Python 3.13.7

  # OR activate (csh/tcsh) -----
  [seheon@Mac project] source .venv313/bin/activate.csh
  (.venv313) [seheon@Mac project] python --version
  Python 3.13.7

  # (2) Clean up environment variables and disable user site-packages: This prevents accidental mixing with the system’s Python 3.8

  # In bash or zsh
  (.venv313) [seheon@Mac project] unset PYTHONPATH PYTHONHOME PIP_PREFIX
  (.venv313) [seheon@Mac project] export PYTHONNOUSERSITE=1

  # In tcsh or csh
  (.venv313) [seheon@Mac project] unsetenv PYTHONPATH PYTHONHOME PIP_PREFIX
  (.venv313) [seheon@Mac project] setenv PYTHONNOUSERSITE 1

  # (3) Upgrade pip and build tools: 
  Always use *** python3 -m pip *** (never just pip) to ensure you are inside the virtual environment.

  (.venv313) [seheon@Mac project] python3 -m ensurepip --upgrade
  (.venv313) [seheon@Mac project] python3 -m pip install --upgrade pip setuptools wheel

  # (4) Quick sanity check
  (.venv313) [seheon@Mac project] python -V                 # --> Python 3.13.x
  (.venv313) [seheon@Mac project] python -m pip -V          # --> .../venv/.../python3.13/site-packages
  (.venv313) [seheon@Mac project] which python; which pip   # --> both should point inside your venv
```

### 2) Clone baygaud-PI from github (it will take a while...)
```bash
(.venv313) [seheon@Mac project] git clone https://github.com/seheon-oh/baygaud-PI.git
```

### 3) Move into baygaud-PI directory and install

```bash
(.venv313) [seheon@Mac project] cd baygaud-PI

# Install package + pinned deps:
*** Always use python3 -m pip (not just pip) ***
(.venv313) [seheon@Mac baygaud-PI] python3 -m pip install .


# To leave the environment later:
(.venv313) [seheon@Mac baygaud-PI] deactivate 
[seheon@Mac baygaud-PI] 
```

---

## 🟢 Quick Start (venv strongly recommended)

### 1) Prepare your data & YAML

**Put your H I data cube (FITS) in a working directory. Copy and edit a YAML template:**

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

**Minimal required params:**

```yaml
# --- REQUIRED minimum you should customize ---
wdir: '/PATH/TO/YOUR/WORKDIR'
input_datacube: 'your_cube.fits'
num_hanning_passes: 0

# small ROI for a quick test (inclusive)
naxis1_s0: 0
naxis1_e0: 40
naxis2_s0: 0
naxis2_e0: 40

# thresholds
max_ngauss: 3
mom0_nrms_limit: 1
peak_sn_pass_for_ng_opt: 3
peak_sn_limit: 2
int_sn_limit: 1.0

# classification bounds
g_sigma_upper: 999 # DEFAULT WOULD BE OK IF YOU DON'T WANT TO PUT HARD LIMIT

# parallelization
num_cpus_ray: 8
```

**YOU CAN USE SOFIA MASK CUBE (OR 2D) FITS (RECOMMENDED):**
```yaml
# Optional 2D mask usage.
#  - 'Y' if a 2D signal mask is available (use _cube_mask_2d); 'N' otherwise.
_cube_mask_2d: 'N'          # [OPTIONAL]
# file name under wdir for the 2D mask FITS; ignored if _cube_mask_2d == 'N'.
_cube_mask_2d_fits: 'mask_2d.fit'        # [CONDITIONAL] set only when _cube_mask_2d == 'Y'

# Optional 3D mask usage.
#  - 'Y' if a 3D signal mask is available (use _cube_mask_3d); 'N' otherwise.
_cube_mask_3d: 'Y'          # [OPTIONAL]
# file name under wdir for the 3D mask FITS; ignored if _cube_mask_3d == 'N'.
_cube_mask_3d_fits: 'sofia2_mask_3d.fits'        # [CONDITIONAL] set only when _cube_mask_3d == 'Y'
```

**Defaults you can usually keep (RECOMMENDED):**

- `dynesty`: `sample: 'rwalk'`, `bound: 'multi'`, `nlive: 100`, `dlogz: 0.01`, `update_interval: 2.0`, `facc/fmove/max_move`  
- **Matched filter & robust RMS** parameters  
- **SG-fit bounds:** `use_phys_for_x_bounds: true`, provided scale factors  
- **Tiling/batching:** `y_chunk_size`, `gather_batch`  
- **Threading:** set all thread counts to `1` to avoid oversubscription under Ray

### 2) Run baygaud

From the source tree:

(.venv313) [seheon@Mac baygaud-PI] cd src/baygaud_pi

(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud.py my_params.yaml

```
 ____________________________________________
[____________________________________________]

 :: Running baygaud.py with config: _baygaud_params.ngc2403.yaml

2025-09-13 04:18:56,052 INFO worker.py:1951 -- Started a local Ray instance.
 _                                       _         ___ _____ 
| |__   __ _ _   _  __ _  __ _ _   _  __| |       / _ \\_   \
| '_ \ / _` | | | |/ _` |/ _` | | | |/ _` |_____ / /_)/ / /\/
| |_) | (_| | |_| | (_| | (_| | |_| | (_| |_____/ ___/\/ /_  
|_.__/ \__,_|\__, |\__, |\__,_|\__,_|\__,_|     \/   \____/  
             |___/ |___/                                     
                           v.2.0.0

+------------------------------------------------------------------------+
| Input Cube:   ngc2403.regrid.testcube.0.fits                           |
| WDIR:   ../../demo/test_cube                                           |
+------------------------------------------------------------------------+
| Key header params       |     Value | Note                             |
+------------------------------------------------------------------------+
| naxis1 (pixels)         |        41 | [0 : 40]                         |
| naxis2 (pixels)         |        41 | [0 : 40]                         |
| naxis3 (channels)       |        74 | [:]                              |
| Velocity min (km/s)     |      9.46 |                                  |
| Velocity max (km/s)     |    396.36 |                                  |
| CDELT3 (m/s)            |  +5299.95 | (+) spectral axis increasing     |
| Spec axis unit check    |      km/s | <- displayed here should be km/s |
+------------------------------------------------------------------------+
| Key baygaud params      |     Value | Note                             |
+------------------------------------------------------------------------+
| max_ngauss (number)     |         3 | Maximum Gaussian components      |
| peak-flux S/N limit     |       2.0 | Minimum peak-flux S/N            |
+------------------------------------------------------------------------+
| Runtime (Ray)           |     Value |                                  |
+------------------------------------------------------------------------+
| Ray initialized         |      True |                                  |
| Total physical cores    |        12 |                                  |
| Ray allocated cores     |        10 |                                  |
| Sampler allocated cores |         1 |                                  |
| Numba allocated threads |         1 |                                  |
| System memory (GB)      |      16.0 |                                  |
| Process memory (GB)     |       0.2 |                                  |
| (y chunk size)          |      (10) |                                  |
| (gather batch)          |      (10) |                                  |
+------------------------------------------------------------------------+

|----------------------------------------------------------------------------------------|
|---:---:---:-->   :   :   :   :   :   :   :   :   :   :   :   :   :   :   :   |   23.12% 
| 370/1600 profiles | 17.42 profiles/s | elapsed 00:00:00:21 | eta 00:00:01:10 |          
|                                                                              |          
|----------------------------------------------------------------------------------------|
```

---

## 🟢 Output layout & resume

Segment results are saved under `_segdir` (inside `wdir`):

```
segmts/
  G03.x10.ys10ye390.npy   # example: max_ngauss=3, column x=10, rows 10..390
```

If a run stops, you can **resume** by adjusting the ROI in your YAML to skip already-processed tiles:
`naxis1_s0`, `naxis1_e0`, `naxis2_s0`, `naxis2_e0`.

---


## 🟢 Classification (merge & classify)

After segments are done, merge and classify components:

```bash
# Prints usage when run without args
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_classify.py

# Recommended: specify YAML and output index (1, 2, …)
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_classify.py my_params.yaml 1
```

This creates a folder like `segmts_merged_n_classified.1/` under your `wdir`.
It contains all merged + classified results and several helpful subfolders.

### Output layout (example)

```
wdir/
└─ segmts_merged_n_classified.1/
   ├─ _baygaud_params.<target>.yaml
   ├─ baygaud_gfit_results.fits
   ├─ baygaud_gfit_results.npy
   ├─ ngfit/
   ├─ ngfit_wrt_peak_amp/
   ├─ ngfit_wrt_vlos/
   ├─ ngfit_wrt_vdisp/
   ├─ ngfit_wrt_integrated_int/
   ├─ sgfit/
   ├─ psgfit/
   ├─ cool/
   ├─ warm/
   ├─ hot/
   └─ hvc/                # may appear, depending on your YAML settings
```

### What each folder/file means

- **`ngfit/`**
  All Gaussian components from the **profile decomposition** by `baygaud.py`.
  Components are **not sorted** (original order).
  Files follow this naming pattern:
  ```
  <prefix>.G<max-ngauss>_<n>.<idx>[.e].fits
  ```

  Where:
  - `max-ngauss` = the **maximum number of Gaussians** set in the YAML (key: `max_ngauss`)
  - `n` = the **component number** (1..N for that pixel)
  - `idx` = a number from **0 to 7** (parameter index)
  - The optional `.e` means the **error map** for that parameter

  Parameter index mapping:
  - `0` → **integrated intensity** `[Jy/beam]`
  - `1` → **line-of-sight velocity (vlos)** `[km/s]`
  - `2` → **velocity dispersion (vdisp)** `[km/s]`
  - `3` → **background** `[Jy/beam]`
  - `4` → **rms** `[Jy/beam]`
  - `5` → **peak flux** `[Jy/beam]`
  - `6` → **peak-flux S/N** `[value]`
  - `7` → **N-Gauss** `[number]`

  Examples:
  - `sgfit.G5_3.5.fits` → parameter index 5 (**peak flux**) of **component 3** when `max_ngauss=5`
  - `sgfit.G4_1.1.e.fits` → **error map** of parameter 1 (**vlos**) for **component 1**, `max_ngauss=4`
- **`ngfit_wrt_peak_amp/`**
  The same components as `ngfit/`, but **sorted by peak flux** at each pixel.
  The 1st = strongest peak … up to **N-Gauss** (the number of max_gauss).

- **`ngfit_wrt_vlos/`**
  Same as above, but **sorted by line-of-sight velocity**.

- **`ngfit_wrt_vdisp/`**
  Same as above, but **sorted by velocity dispersion**.

- **`ngfit_wrt_integrated_int/`**
  Same as above, but **sorted by integrated intensity**.

- **`sgfit/`**
  **Single-Gaussian fit** of every velocity profile in the data cube
  (regardless of the optimal number of Gaussians from decomposition).

- **`psgfit/`**
  A **pruned single-Gaussian set**: shows results **only** for pixels where the
  optimal **N-Gauss = 1** (from the decomposition).

- **`cool/`, `warm/`, `hot/`, `hvc/`**
  Components grouped by **physical class**, based on your YAML **classification parameters**.
  Each folder holds the corresponding component maps (and their `.e.fits` error maps).

- **`baygaud_gfit_results.fits`** and **`baygaud_gfit_results.npy`**
  These are **combined summaries** of the `ngfit` results, saved in **FITS** and **NumPy** formats.

- **`_baygaud_params.<target>.yaml`**
  A copy of the YAML used to run the classification (for reproducibility).

### Notes

- The trailing number in `segmts_merged_n_classified.<index>/` is the **run index** you pass on the command line (e.g., `1`, `2`, …).




## 🟢 Viewer

Inspect per-spectrum decompositions interactively:

```bash
# Prints usage when run without args
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_viewer.py

# With YAML (recommended) — use the same index as classification
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3 baygaud_viewer.py my_params.yaml 1
```

Tips: hover to locate spectra; mouse wheel to zoom; choose which 2D map to display (single-Gaussian VF, dispersion, integrated intensity, N-Gauss, peak S/N, etc.).

---

## 🟢 Troubleshooting (quick notes)

- **CPU oversubscription:** keep all threading knobs at `1` when using Ray (`OMP/MKL/OPENBLAS/numba` threads).  
- **Python version:** if a dependency lags on 3.13, recreate the venv with 3.10–3.12.  
- **Long runs:** start with a **small ROI** to validate configuration and performance.

---

## 🟢 Cite

If you use baygaud-PI, please cite the main algorithm and relevant applications:

1. Oh, S. H., Staveley-Smith, L., For, B. Q. (2019), **MNRAS**, 485, 5021–5034.  
2. Oh, S.-H., Kim, S., For, B.-Q., Staveley-Smith, L. (2022), **ApJ**, 928, 177.  
3. Park, H. J., Oh, S. H., Wang, J., et al. (2022), **AJ**, 164, 82.  
4. Kim, S. J., Oh, S. H., et al. (2023), **MNRAS**.  
5. Kim, M., Oh, S. H. (2022), **JKAS**, 55, 149–172.  
6. Wang, J., Yang, D., Oh, S.-H., et al. (2023), **ApJ**.  
7. Oh, S.-H., Wang, J. (2025), **MNRAS**, 538, 1816.

---

## 🟢 Support & Issues


Happy decomposing!!!
