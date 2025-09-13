#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _handle_ray.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|


import ray

#  _____________________________________________________________________________  #
# [_____________________________________________________________________________] #
# _materialize function
def _mat(x):
    # if x is ObjectRef then ray.get it otherwise just return it
    try:
        if isinstance(x, ray.ObjectRef):
            return ray.get(x)
    except Exception:
        pass
    return x

#-- END OF SUB-ROUTINE____________________________________________________________#



