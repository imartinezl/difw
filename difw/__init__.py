#%%
__author__ = "Iñigo Martinez inigomlap[at]gmail.com"
__version__ = "0.0.31"

import os

on_rtd = "READTHEDOCS" in os.environ.keys()
if not on_rtd:
    from .cpab import Cpab
