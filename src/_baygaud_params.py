#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _baygaud_params.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import sys
import numpy as np
import yaml

#  ______________________________________________________  #
# [______________________________________________________] #
# [ global parameters
# _______________________________________________________  #
global _inputDataCube
global _is, _ie, _js, _je
global parameters
global nparams
global ngauss
global ndim
global max_ngauss
global gfit_results
global _x
global nchannels

#  ______________________________________________________  #
# [______________________________________________________] #
# [ read yaml file
# _______________________________________________________  #
def read_configfile(configfile):
    with open(configfile, "r") as file:
        _params = yaml.safe_load(file)
    return _params
#-- END OF SUB-ROUTINE____________________________________#





