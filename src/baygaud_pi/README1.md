# baygaud-PI 


BAYesian GAUssian Decomposer — a tool to decompose spectral-line profiles into multiple Gaussian components using Bayesian nested sampling.

- Version: 2.0.0 (2025-09-12)

- Author: Se-Heon Oh (Department of Physics and Astronomy, Sejong University, Seoul, South Korea, seheon.oh@gmail.com)

# Overview

baygaud-PI automatically determines the optimal number of Gaussians for each line-of-sight spectrum via Bayes factors, and produces 2D maps (velocity, dispersion, integrated intensity, N-Gauss, peak S/N, and more) from an HI data cube. These outputs can be used for downstream kinematic analysis (e.g., rotation curves).

# Features

- Bayesian nested sampling (via dynesty) for robust model selection
- Automatic component counting using Bayes factors
- Ray-based parallel processing for large cubes
- Segment-wise saving and safe resume
- Post-processing: merge segments and classify components
- Viewer for per-spectrum decomposition overlays



# Features

- Python: 3.13 (latest)

- OS: Ubuntu 20.04.6 LTS (expected to work on most releases) and macOS 15.6.1 (Apple M2 Pro).

- Core packages:

	- dynesty (2.1.5)
	- ray (tested with recent 2.x)
	- astropy, spectral-cube, numpy, matplotlib, etc.

- System deps (Linux): libbz2-dev (needed by fitsio)

- Viewer (optional): python3.13-tk



# Installation

Create and activate a Python 3.10 virtual environment, then install:


# create a clean environment (example path)
python3.10 -m venv /home/you/baygaud
source /home/you/baygaud/bin/activate   # or activate.csh for csh/tcsh

# system dependency (Ubuntu)
sudo apt-get install -y libbz2-dev

# get the code
git clone https://github.com/seheon-oh/baygaud-PI.git
cd baygaud-PI

# install package and pinned deps
pip install .

# optional (developer mode)
python3 setup.py develop

# optional (viewer)
sudo apt-get install -y python3.10-tk


To leave the environment later: deactivate.


# Quick Start (Strongly Recommended: Virtual Environment)

Always create and activate a dedicated venv before installing or running baygaud-PI.

1) Clone and create a Python 3.13 venv

[seheon@Mac ~] git clone https://github.com/seheon-oh/baygaud-PI.git
[seheon@Mac ~] cd baygaud-PI
[seheon@Mac baygaud-PI] python3.13 -m venv .venv313

Activate the venv (choose one):

For bash/zsh

[seheon@Mac baygaud-PI] source .venv313/bin/activate
(.venv313) [seheon@Mac baygaud-PI] python --version

For csh/tcsh

[seheon@Mac baygaud-PI] source .venv313/bin/activate.csh
(.venv313) [seheon@Mac baygaud-PI] python --version

Tip: If any dependency doesn’t yet support 3.13 on your system, use Python 3.10–3.12 with the same steps.


2) Install the package inside the venv

(.venv313) [seheon@Mac baygaud-PI] pip install .


3) Run baygaud.py with your YAML

Move into the source tree:

(.venv313) [seheon@Mac baygaud-PI] cd src/baygaud_pi

Prepare your data (HI data cube FITS) in a working directory.

Configure parameters using the YAML template:

# copy and edit a template for your target galaxy
cp _baygaud_params.ngc2403.yaml my_params.yaml
vim my_params.yaml

The config file is annotated with REQUIRED, RECOMMENDED, and OPTIONAL tags.
In practice:

Update all fields marked REQUIRED before running.

Leave most RECOMMENDED and OPTIONAL fields at their defaults unless you know you need a change - the shipped values work for the majority of datasets and machines.

Quick checklist (REQUIRED before first run)

Data paths

wdir: absolute path to your working directory (where the FITS lives)

input_datacube: FITS filename under wdir

num_hanning_passes: number of Hanning passes already applied to your cube (use 0 if unsure)

ROI (indices to process)

naxis1_s0, naxis1_e0 (x start/end, inclusive)

naxis2_s0, naxis2_e0 (y start/end, inclusive)
Tip: start with a small ROI to validate settings and speed up a first test.

Decomposition/selection thresholds

max_ngauss

mom0_nrms_limit, peak_sn_pass_for_ng_opt, peak_sn_limit, int_sn_limit

Physical bounds (for classification)

g_vlos_lower, g_vlos_upper

g_sigma_upper (a safe high cap; the lower bound is computed for you)

Parallelization

num_cpus_ray (total CPU cores to let Ray use; don’t exceed your physical cores)

Heads-up: Keys like g_sigma_lower, naxis1/2/3, cdelt*, vel_min/max, _rms_med, _bg_med are filled at runtime — the defaults are placeholders and can be left as is.



Minimal starter (edit these, keep the rest default)

# --- REQUIRED minimum you should customize ---
wdir: '/ABSOLUTE/PATH/TO/YOUR/WORKDIR'
input_datacube: 'your_cube.fits'
num_hanning_passes: 0

# small ROI for a quick test (inclusive indices)
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

Defaults you can usually keep (RECOMMENDED)

Nested sampling (dynesty): sample: 'rwalk', bound: 'multi', nlive: 100, dlogz: 0.01, update_interval: 2.0, facc/fmove/max_move — tuned for robustness.

Matched filter & robust RMS: kernels, thresholds, and clipping parameters are conservative and work well out of the box.

SG-fit bounds: use_phys_for_x_bounds: true and the provided scale factors are sensible starting points.

Tiling/batching: y_chunk_size and gather_batch defaults balance throughput and memory on typical laptops/workstations.

Threading: all thread counts = 1 avoids BLAS/OpenMP oversubscription when using Ray. Increase only if you know your stack.

Optional settings (nice to have, not required)

2D mask map

_cube_mask: 'Y' | 'N' and _cube_mask_2d (provide the FITS only when _cube_mask: 'Y').

Bulk motion model inputs for pre-subtraction (keep blanks if not using).

Inspection helpers: _i0, _j0 to pick a single pixel for quick checks.

Component ordering rules: _sort_gauss_wrt_int etc.

Common pitfalls (and fixes)

Paths: keep quotes around paths; use absolute paths for wdir.

CPUs: num_cpus_ray should not exceed physical cores (on macOS: sysctl -n hw.physicalcpu).

Hanning passes: if unsure, set num_hanning_passes: 0.

ROI: indices are inclusive; a large ROI on a laptop will be slow — test small first.

g_sigma_lower: it’s derived from your cube’s channel width (+ smoothing) at runtime; leave the default.


Once edited, run:

(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] $ python3 baygaud.py _baygaud_params.yaml


If you later need to fine-tune speed/accuracy, adjust RECOMMENDED values gradually (one section at a time) and re-test on a small ROI.


Output layout

Segment results are saved under _segdir:

segmts/
  G03.x10.ys10ye390.npy   # example: max_ngauss=3, column x=10, rows 10..390


If the run stops unexpectedly or intentionally, you can resume by adjusting the ROI in your params to skip already-processed tiles:

naxis1_s0, naxis1_e0, naxis2_s0, naxis2_e0



Post-processing (merge + classify)

After some or all segments are done, merge and classify components:

# prints usage when run without args
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3.10 baygaud_classify.py

# usage: with YAML
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3.10 baygaud_classify.py my_params.yaml 1 [e.g., 1, 2, 3, 4 ....]


This creates a directory like segmts_merged_n_classified.1/ containing classified components (e.g., bulk, warm, hot, non_bulk, sgfit, psgfit, hvc, etc.). It also writes combined results:

baygaud_gfit_results.fits

baygaud_gfit_results.npy


Viewer

Inspect per-spectrum decompositions with the interactive viewer:

# prints usage when run without args
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3.10 baygaud_viewer.py

# with YAML (recommended)
(.venv313) [seheon@Mac baygaud-PI/src/baygaud_pi] python3.10 baygaud_viewer.py my_params.yaml 1 [<-- same index number for baygaud_classify.py above]


Tips: hover to locate spectra; use mouse wheel to zoom; choose which 2D map to display (single-Gaussian VF, dispersion, integrated intensity, N-Gauss, peak S/N, etc.).

Cite

If you use baygaud-PI, please cite the main algorithm paper and relevant applications:

Oh, S. H., Staveley-Smith, L., For, B. Q. (2019), MNRAS, 485, 5021–5034.

Oh, S.-H., Kim, S., For, B.-Q., Staveley-Smith, L. (2022), ApJ, 928, 177.

Park, H. J., Oh, S. H., Wang, J., et al. (2022), AJ, 164, 82.

Kim, S. J., Oh, S. H., et al. (2023), MNRAS (WALLABY Pilot).

Kim, M., Oh, S. H. (2022), JKAS, 55, 149–172.

Wang, J., Yang, D., Oh, S.-H., et al. (2023), ApJ.

Oh, S.-H., Wang, J. (2025), MNRAS, 538, 1816.

Support and Issues

Please open issues and pull requests on GitHub. Include your OS, Python version, and a minimal YAML snippet when reporting bugs.
