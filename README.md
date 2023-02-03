# baygaud-PI
	- Version 1.0.0 (30 Sept 2022)
	- by Se-Heon Oh (Department of Physics and Astronomy, Sejong University, Seoul, Korea)

# Pre-requisite

	- Python3.10
	- Python3 virtual environment module: venv (normally, venv is installed along with python3)
	- The latest version of dynasty 2.0.3 will be installed for bayesian analysis employing nested sampling.
	- Tested for Ubuntu 18.04 LT and macOS Monterey 12.6 on Apple M1

# Installation

1. Make a directory of python3 virtual environment for baygaud-PI. For example, 

		[seheon@sejong00] makedir /home/seheon/research/baygaud_PI
	 

2. Download baygaud-master.zip and unzip in the directory,

		[seheon@sejong00] mv baygaud-master.zip /home/seheon/research/baygaud_PI
  		[seheon@sejong00] unzip  /home/seheon/research/baygaud_PI/baygaud-master.zip

		--> Check the files,
		[seheon@sejong00] ls /home/seheon/research/baygaud_PI/baygaud-master/
		MENIFEST.in  README.md  requirements.txt  setup.cfg  setup.py  src
	

3. Setup "python3 virtual environment" in the 'baygaud' directory created.

		[seheon@sejong00] python3 -m venv /home/seheon/research/baygaud_PI
		
		--> Then, activate the virtual environment.
		
		
		[seheon@sejong00] source /home/seheon/research/baygaud_PI/bin/activate.csh
		
		--> FYI, to deactivate, just type 'deactivate'
		
		
		[seheon@sejong00] deactivate
		
		--> Now, you enter the python virtual environment, named 'baygaud'
		
		(baygaud) [seheon@sejong00]
		
		
		--> Install the python packages for baygaud-PI. For dependency, these packages are only compatible in the virtual environment that is currently created. The required package list, 'requirements.txt' is given in 'baygaud_PI' directory.

		(baygaud) [seheon@sejong00] cd baygaud-master
		(baygaud) [seheon@sejong00] ls
		MENIFEST.in  README.md  requirements.txt  setup.cfg  setup.py  src

		(baygaud) [seheon@sejong00] pip install -r requirements.txt
		
		--> Now it should install the modules required for the baygaud-PI python3 environment. It takes a while…

		--> Install python-tk for baygaud_viewer.py		
		(baygaud) [seheon@sejong00] sudo apt install python-tk

		--> After installing all the required packages for baygaud-PI, it is ready for running baygaud-PI now.


# Quick Start

1. Setting up data (HI data cube)

		--> Make a directory where the data files including the HI data cube in FITS format are located.

		|| Data directory
		[seheon@sejong00] makedir /home/seheon/research/mhongoose/ngc2403

		--> Copy the input data files (FITS) into the data directory (‘/home/seheon/research/mhongoose/ngc2403’).

		--> Then, make a directory for the baygaud 'segment output'.
		
		[seheon@sejong00] makedir /home/seheon/research/mhongoose/ngc2403/baygaud_segs_output

		--> Check the data files are in the data directory,
		[seheon@sejong00] ls /home/seheon/research/mhongoose/ngc2403

		2403_mosaic_5kms_r05_HI_mwfilt_cube.fits	2403_mosaic_5kms_r05_HI_mwfilt_mask.fits
		2403_mosaic_5kms_r05_HI_mwfilt_cube.kms.fits	baygaud_segs_output
		2403_mosaic_5kms_r05_HI_mwfilt_mask-2d.fits	

		--> For example (see _baygaud_params.py below),

		|| Data directory; segment output directory
		'wdir':'/home/seheon/research/mhongoose/ngc2403',
		'_segdir':'/home/seheon/research/mhongoose/ngc2403/baygaud_segs_output',

		|| Input HI data cube (required)
		'input_datacube':'2403_mosaic_5kms_r05_HI_mwfilt_cube.fits'

		|| 2D mask map (if not available, put blank)
		'_cube_mask_2d':'2403_mosaic_5kms_r05_HI_mwfilt_mask-2d.fits'

		|| Bulk model VF (if not available, put blank)
		'_bulk_ref_vf':'NGC_4826_NA_MOM0_THINGS.dim.mask.fits'

		|| Bulk velocity-limit 2D map (if not available, put blank)
		'_bulk_delv_limit':'NGC_4826_NA_MOM0_THINGS.dim.mask.fits'


2. Setting up baygaud-PI parameters

		--> Open  ‘_baygaud_params.py’ file using vim or other text editors. Update keywords upon your system accordingly. Find "UPDATE HERE" lines and edit them as yours. Short descriptions (recommendation) are given below.
		
		|| In RED : should be updated upon your sample galaxy
		|| In BLUE : should be updated upon your computer
		|| In GREEN : can be adjusted for improving the performance (speed issues etc.)
		
		--> For a quick test, try a small section like,
		# district
		'naxis1_s0':200,
		'naxis1_e0':204,
		'naxis2_s0':200,
		'naxis2_e0':204,
	
	
3. Running baygaud.py

		(baygaud_PI) [seheon@sejong00] python3 baygaud.py

		--> Check the running processes (using multi-cores) on the machine. 
		--> Check the output directory where the baygaud fitting results are written in binary format. 

		# Output directory in ‘_baygaud_params.py’
		
		'_segdir':'/home/seheon/research/mhongoose/ngc2403/baygaud_segs_output',

		--> In _segdir directory, all the Gaussian fit results for each sub-cube (segment, xN - ys:y3 - vel) are saved in binary format. For example,
			G03_x10.ys10ye390.npy <-- python binary format

		|| G03 : max_ngauss=3
		|| x10 : column x-number=10 ← segment info
		|| ys10ye390 : row range, ys(start)=10 ~ ye(end)=390  ← segment info

		--> In case, baygaud process stops unexpectedly for some reasons, the analysis results completed for segments are stored. So you can resume baygaud from there but you need to adjust:

		'naxis1_s0= xxx'
		'naxis1_e0= xxx'
		'naxis2_s0= xxx'
		'naxis2_e0= xxx'

		in '_baygaud_params.py' accordingly not to repeat the segments already processed.


4. Running baygaud_classify.py

		--> After all or some of the baygaud processes are completed, you can combine the segments to produce 2D FITS maps.
		
		(baygaud_PI) [seheon@sejong00] python3 baygaud_classify.py
		
		--> This routine combines the segmented baygaud output (in binary format), and produces 2D maps in ‘FITS’ format
		which include the profile decomposition results. This will also produce the combined baygaud fit results in both
		‘fits’ and ‘binary’ formats like ‘Baygaud_gfit_results.fits’ and ‘baygaud_gfit_results.npy’. Either of these files
		can be kept for backup.

		--> In the working directory, as in _baygaud_params.py above,
		
		# working directory where the input data cube is
		'wdir':'/home/seheon/research/mhongoose/ngc2403'
		
		--> A directory, 'baygaud_combined' will be created where the decomposed Gaussian components are. These Gaussian
		components (i.e., bulk, cool, wram, hot, non_bulk, psgfit, and sgfit or whatever else the one defined by the user)
		are classified by their kinematic properties setup as in_baygaud_params.py file.
		
		bulk
		cool
		hot
		ngfit
		non_bulk
		psgfit
		sgfit
		warm


5. Running baygaud_viwer.py

		--> You can view the results of Baygaud's multi-Gaussian profile analysis for individual velocity profiles using the 'baygaud_viewer.py' code. This code reads the optimal number of Gaussian profiles derived by 'baygaud_classify.py' and displays the decomposed Gaussian components overlaid on each spectral line.

When a 2D map (such as a single Gaussian velocity field, velocity dispersion, integrated intensity, N-Gauss, or S/N) extracted by Baygaud-PI (selected in the menu) is displayed, you can move your mouse cursor over the map to locate a specific spectral line.


# Cite

	1. Robust profile decomposition for large extragalactic spectral-line surveys (main algorithm paper)
		Oh, S. H., Staveley-Smith, L. & For, B. Q., 13 Mar 2019, In: Monthly Notices of the Royal Astronomical Society. 485, 4, p. 5021-5034 14 p.

	2. Kinematic Decomposition of the HI Gaseous Component in the Large Magellanic Cloud (application paper)
		Oh, S. H., Kim, S., For, B. Q. & Staveley-Smith, L., 1 Apr 2022, In: Astrophysical Journal. 928, 2, 177.
	
	3. Gas Dynamics and Star Formation in NGC 6822 (application paper)
		Park, H. J., Oh, S. H., Wang, J., Zheng, Y., Zhang, H. X. & De Blok, W. J. G., 1 Sep 2022, In: Astronomical Journal. 164, 3, 82.
		
	4. WALLABY Pilot Survey: HI gas kinematics of galaxy pairs in cluster environment (application paper)
		Kim, S. J., Oh, S. H. et al., 2023, In Monthly Notices of the Royal Astronomical Society
	
	5. GLOBAL HI PROPERTIES OF GALAXIES VIA SUPER-PROFILE ANALYSIS (application paper)
		Kim, M. & Oh, S. H., Oct 2022, In: Journal of the Korean Astronomical Society. 55, 5, p. 149-172 24 p.



