# baygaud-PI 
	- BAYesian GAUssian Decomposer: decomposing a spectral line profile into multiple Gaussian components using Bayesian analysis
	- Version 1.0.0 (30 Sept 2022)
	- by Se-Heon Oh (Department of Physics and Astronomy, Sejong University, Seoul, Korea)
	
	baygaud-PI is a new tool for decomposing a spectral line profile with multiple Gaussian components based on Bayesian nested sampling.
	One of its unique features is that it determines the optimal number of Gaussian components for a given line profile by using Bayes
	factor statistics. This allows for the extraction of various Gaussian 2D maps for the line-of-sight velocity profiles of an HI data
	cube of a galaxy, which can then be utilized in any post-processing of HI kinematic analysis, such as deriving the galaxy rotation curve.

# Prerequisite

	- Python3.10
	- Python3 virtual environment module: venv (normally, venv is installed along with python3)
	- The latest version of dynesty 2.0.3 will be installed for Bayesian analysis utilizing nested sampling.
	- Tested for Ubuntu 18.04 LT and macOS Monterey 12.6 on Apple M1

# Installation

1. Make a directory for the python3.10 virtual environment for baygaud-PI. For example, 

		[seheon@sejong00] makedir /home/seheon/research/baygaud
	

2. Set up a 'python3.10 virtual environment' in the 'baygaud' directory created.

		[seheon@sejong00] python3.10 -m venv /home/seheon/research/baygaud
		
		--> Then, activate the virtual environment.
		--> If you are using CSH or TCSH
		[seheon@sejong00] source /home/seheon/research/baygaud/bin/activate.csh
		
		--> If you are using BASH
		[seheon@sejong00] source /home/seheon/research/baygaud/bin/activate
		
		--> Now, you enter the python3.10 virtual environment, named 'baygaud'
		(baygaud) [seheon@sejong00]
		
		--> FYI, to deactivate, just type 'deactivate'
		(baygaud) [seheon@sejong00] deactivate	
	
3. Install baygaud-PI via github,

		(baygaud) [seheon@sejong00] git clone https://github.com/seheon-oh/baygaud-PI.git
		--> Enter baygaud-PI directory created, and install it via 'pip' command
		(baygaud) [seheon@sejong00] cd baygaud-PI
		
		--> Install 'libbz2-dev' (if not pre-installed yet) which is needed for 'fitsio' package in the following installation.
		(baygaud) [seheon@sejong00] sudo apt-get install libbz2-dev
		
		--> Now, install baygaud-PI using 'pip' command
		(baygaud) [seheon@sejong00] pip install .
		
		--> Now it should install the modules required for the baygaud-PI python3.10 environment.
		It takes a while…
		
		--> Note that the required python packages for baygaud-PI are only compatible
		within the virtual environment that has been created. The required package list can be found in
		'requirements.txt' in the 'baygaud_PI' directory.
		
		(Optional) Or, for developer installation (this installs the package in the same location to
		allow for changes to be reflected across the environment), use the following command,	
		(baygaud) [seheon@sejong00] python3 setup.py develop
		
		
		--> Lastly, install python3.10-tk for baygaud_viewer.py		
		(baygaud) [seheon@sejong00] sudo apt install python3.10-tk

		--> Now, it is ready for running baygaud-PI now.


# Quick Start

1. Setting up data (HI data cube)

		--> Make your own directory where the data files including the HI data cube in FITS format are located.

		--> Put the input data files (FITS) into the data directory. 

		--> As an example, a test cube ('ngc2403.regrid.testcube.0.fits') is provided in 'demo/testcube' directory:

		|| Data directory
		[seheon@sejong00] ls /home/seheon/research/code/_python/baygaud_py/baygaud_PI/demo/test_cube

		ngc2403.regrid.testcube.0.fits

		--> For example (see '_baygaud_params.py' in 'src'),

		|| Set data directory; segment output directory in _baygaud_params.py in 'src'
		'wdir':'/home/seheon/research/code/_python/baygaud_py/baygaud_PI/demo/test_cube',
		'_segdir':'segmts',
		'_combdir':'segmts_merged_n_classified'

		|| Input HI data cube (required)
		'input_datacube':'ngc2403.regrid.testcube.0.fits'

		|| 2D mask map (if not available, put blank)
		'_cube_mask':'Y', # Y(if available) or N(if not)
		'_cube_mask_2d':'2403_mosaic_5kms_r05_HI_mwfilt_mask-2d.fits'

		|| Bulk model VF (if not available, put blank)
		'_bulk_ref_vf':'NGC_4826_NA_MOM0_THINGS.dim.mask.fits'

		|| Bulk velocity-limit 2D map (if not available, put blank)
		'_bulk_delv_limit':'NGC_4826_NA_MOM0_THINGS.dim.mask.fits'


2. Setting up baygaud-PI parameters

		--> Open ‘_baygaud_params.py’ file using vim or other text editors. Update keywords upon
		your system accordingly. Find "UPDATE HERE" lines and edit them as yours. Short descriptions
		(recommendation) are given below.
		
		|| In RED : should be updated upon your sample galaxy
		|| In BLUE : should be updated upon your computer
		|| In GREEN : can be adjusted for improving the performance (speed issues etc.)
		|| In YELLOW : <--- UPDATE HERE : should be adjusted
		--> For a quick test, try a small section like,
		# district
		'naxis1_s0':20,
		'naxis1_e0':24,
		'naxis2_s0':20,
		'naxis2_e0':24,
		
![Screen Shot 2023-02-03 at 11 39 05 PM](https://user-images.githubusercontent.com/100483350/216631276-20c9c5ba-a4d7-4ab1-bffb-5f018ee54964.png)

![Screen Shot 2023-02-03 at 11 39 12 PM](https://user-images.githubusercontent.com/100483350/216631310-e4fd11d9-1e39-4726-a7cd-a34dd1dba088.png)

![Screen Shot 2023-02-03 at 11 39 20 PM](https://user-images.githubusercontent.com/100483350/216631330-681baab0-46e0-4ff8-8cb8-7ff0242400c7.png)

![Screen Shot 2023-02-03 at 11 39 28 PM](https://user-images.githubusercontent.com/100483350/216631379-95703f0f-96fc-46a4-8330-ea4f16b7d461.png)

![Screen Shot 2023-02-03 at 11 39 34 PM](https://user-images.githubusercontent.com/100483350/216631399-29b2946e-611f-4b97-ab85-71af7d05c846.png)


	
3. Running baygaud.py

		--> You can run 'baygaud.py' without arguments, which will load the 'default' baygaud parameters as given
		in '_baygaud_params.py', or you can run it with your own baygaud parameter yaml file (!!! <-- RECOMMENDED !!!).
		For example, you can copy a template file '_baygaud_params.ngc2403.yaml' in the baygaud-PI directory and
		edit it for your target galaxy as in '_baygaud_params.py' but following yaml syntax (see '_baygaud_params.ngc2403.yaml').
		
		(baygaud) [seheon@sejong00] python3.10 baygaud.py
		or 
		(baygaud) [seheon@sejong00] python3.10 baygaud.py _baygaud_params.ngc2403.yaml

		--> Check the running processes (utilizing multi-cores) on the machine.
		--> Check the output directory where the baygaud fitting results are stored in binary format.
		
		# Output directory in ‘_baygaud_params.py’
		
		_segdir: 'segmts'

		--> In the _segdir directory, all of the Gaussian fit results for each sub-cube (segment, xN--ys:ye--vel)
		are saved in binary format. For example,
		
			G03_x10.ys10ye390.npy <-- python binary format

		|| G03 : max_ngauss=3
		|| x10 : column x-number=10 ← segment info
		|| ys10ye390 : row range, ys(start)=10 ~ ye(end)=390  ← segment info

		--> In the event that the baygaud process stops unexpectedly for any reason, the completed analysis
		results for segments are saved. As a result, you can resume Baygaud from that point, but you need
		to make adjustments:

		'naxis1_s0= xxx'
		'naxis1_e0= xxx'
		'naxis2_s0= xxx'
		'naxis2_e0= xxx'

		in '_baygaud_params.py' accordingly not to repeat the segments already processed.


4. Running baygaud_classify.py

		--> After completing some or all of the Baygaud processes, you can combine the segments to create 2D FITS maps.
		--> As for 'baygaud.py', the same options are available. 'baygaud_classify.py' can be run either with
		no arguments to use the default baygaud parameters, or with a specified baygaud parameter file. 
		
		--> If you simply run the script without any arguments. This will print out the usage.
		(baygaud) [seheon@sejong00] python3.10 baygaud_classify.py
		--> 
		___________________________________________________________________________________________

 		:: baygaud_classify.py usage ::

 		usage-1: running baygaud_classify.py with baygaud_params file
 		> python3 baygaud_classify.py [ARG1: _baygaud_params.yaml] [ARG2: output-index, 1, 2, ...]
		 e.g.,
		 > python3 baygaud_classify.py _baygaud_params.ngc2403.yaml 1
		-------------------------------------------------------------------------------------------
 		usage-2: running baygaud_classify.py with the DEFAULT baygaud_params file
  	      	: the DEFAULT baygaud_params file: _baygaud_params.py
		 > python3 baygaud_classify.py [ARG1: output-index, 1, 2, ...]
 		e.g.,
 		> python3 baygaud_classify.py 1
		___________________________________________________________________________________________
		
	
		
		(baygaud) [seheon@sejong00] python3.10 baygaud_classify.py 1
		or
		(baygaud) [seheon@sejong00] python3.10 baygaud_classify.py _baygaud_params.ngc2403.yaml 1 (<-- RECOMMENDED)
		--> The latter option will be useful when running baygaud-PI for a number of galaxies,
		for which the _baygaud_params file has been specified.
		
		--> This routine merges the segmented Baygaud output (in binary format) to create 2D maps
		in FITS format that contain the profile decomposition results. It also generates combined
		baygaud fit results in both FITS and binary formats, such as 'baygaud_gfit_results.fits'
		and 'baygaud_gfit_results.npy'. Either file can be saved for backup purposes.

		--> In the working directory, as in _baygaud_params.py above,
		
		# working directory where the input data cube is
		wdir: '/home/seheon/research/code/_python/baygaud_py/baygaud_PI/demo/test_cube'
		
		--> A directory named 'segmts_merged_n_classified.1' will be created where the decomposed Gaussian components
		are stored. These Gaussian components (such as bulk, cool, warm, hot, non_bulk, psgfit, hvc and sgfit,
		or any others defined by the user) are classified based on their kinematic properties set in the
		'_baygaud_params.py' file.
		
		bulk
		cool
		hot
		ngfit
		non_bulk
		psgfit
		sgfit
		warm
		hvc


5. Running baygaud_viewer.py

		--> You can view the results of baygaud's multi-Gaussian profile analysis for individual
		velocity profiles using the 'baygaud_viewer.py' code. This code reads the optimal number
		of Gaussian profiles derived by 'baygaud_classify.py' and displays the decomposed Gaussian
		components overlaid on each spectral line.
		
		--> If you simply run the script without any arguments, this will print out the usage.
		(baygaud) [seheon@sejong00] python3.10 baygaud_viewer.py
		--> 
		___________________________________________________________________________________________

 		:: baygaud_viewer.py usage ::

 		usage-1: running baygaud_viewer.py with baygaud_params file
		 > python3 baygaud_viewer.py [ARG1: _baygaud_params.yaml] [ARG2: output-index, 1, 2, ...]
 		output-index is the postfix number of the baygaud segments merged directory.
		 i.e., 'segmts_merged_n_classified.[output-index]' in 'wdir'
 		e.g.,
 		> python3 baygaud_viewer.py _baygaud_params.ngc2403.yaml 1
		-------------------------------------------------------------------------------------------
		 usage-2: running baygaud_viewer.py with the DEFAULT baygaud_params file
 		       : the DEFAULT baygaud_params file: _baygaud_params.py
 		> python3 baygaud_viewer.py [ARG1: output-index, 1, 2, ...]
 		e.g.,
 		> python3 baygaud_viewer.py 1
		___________________________________________________________________________________________


		(baygaud) [seheon@sejong00] python3.10 baygaud_viewer.py 1
		or
		(baygaud) [seheon@sejong00] python3.10 baygaud_viewer.py _baygaud_params.ngc2403.yaml 1 (<-- RECOMMENDED)

		
![Screen Shot 2023-02-05 at 12 19 38 AM](https://user-images.githubusercontent.com/100483350/216775296-c040f123-9062-4c38-8a53-fd5e60467d8a.png)

![Screen Shot 2023-02-05 at 12 17 12 AM](https://user-images.githubusercontent.com/100483350/216775302-55a6b2b3-390e-40d1-a0b4-4a2277611616.png)

![Screen Shot 2023-02-05 at 12 16 29 AM](https://user-images.githubusercontent.com/100483350/216775309-f2831851-2554-4e8d-82e6-cceadd9f733b.png)

![Screen Shot 2023-02-05 at 12 15 55 AM](https://user-images.githubusercontent.com/100483350/216775323-2ebc07a7-f6ca-4ec1-980f-e11b03d1c328.png)


		When a 2D map (such as the single Gaussian velocity field, velocity dispersion, integrated
		intensity, N-Gauss, or peak flux S/N) extracted by baygaud-PI (selected in the menu) is displayed,
		you can move your mouse cursor over the map to locate a specific spectral line. 
		You can also zoom-in or -out a specific region by scrollong the mouse wheel.


# Cite

	1. Robust profile decomposition for large extragalactic spectral-line surveys (main algorithm paper)
		Oh, S. H., Staveley-Smith, L. & For, B. Q., 13 Mar 2019, In: Monthly Notices of the Royal Astronomical Society. 485, 4, p. 5021-5034 14 p.

	2. Kinematic Decomposition of the HI Gaseous Component in the Large Magellanic Cloud (application paper)
		Oh, Se-Heon, Kim, Shinna, For, Bi-Qing & Staveley-Smith, Lister, 1 Apr 2022, In: Astrophysical Journal. 928, 2, 177.
	
	3. Gas Dynamics and Star Formation in NGC 6822 (application paper)
		Park, H. J., Oh, S. H., Wang, J., Zheng, Y., Zhang, H. X. & de Blok, W. J. G., 1 Sep 2022, In: Astronomical Journal. 164, 3, 82.
		
	4. WALLABY Pilot Survey: HI gas kinematics of galaxy pairs in cluster environment (application paper)
		Kim, S. J., Oh, S. H. et al., 2023, In Monthly Notices of the Royal Astronomical Society
	
	5. GLOBAL HI PROPERTIES OF GALAXIES VIA SUPER-PROFILE ANALYSIS (application paper)
		Kim, M. & Oh, S. H., Oct 2022, In: Journal of the Korean Astronomical Society. 55, 5, p. 149-172 24 p.
		
	6. FEASTS: IGM cooling triggered by tidal interactions through the diffuse HI phase around NGC 4631 (application paper)
		Jing Wang, Dong Yang, Se-Heon Oh, Lister Staveley-Smith, Jie Wang, Q. Daniel Wang, Kelley M. Hess, Luis C. Ho, Ligang Hou, Yingjie Jing, Peter Kamphuis, Fujia Li, Xuchen Lin, Ziming Liu, Li Shao, Shun Wang, Ming Zhu, Astrophysical Journal (2023)



