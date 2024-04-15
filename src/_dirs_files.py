#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#|-----------------------------------------|
#| _dirs_files.py
#|-----------------------------------------|
#| by Se-Heon Oh
#| Dept. of Physics and Astronomy
#| Sejong University, Seoul, South Korea
#|-----------------------------------------|

import os

def make_dirs(dir):
    try:
        if not os.path.exists(dir):
            os.makedirs(dir)

    except OSError:
        print("Error: failed to make %s", dir)


