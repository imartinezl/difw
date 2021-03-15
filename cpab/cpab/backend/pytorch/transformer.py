
import torch
from torch.utils.cpp_extension import load
from ...core.utility import get_dir

#%%
class _notcompiled:
    # Small class, with structure similar to the compiled modules we can default
    # to. The class will never be called but the program can compile at run time
    def __init__(self):
        def f(*args):
            return None
        self.forward = f
        self.backward = f

#%%
_dir = get_dir(__file__)
_verbose = True

try:
    cpab_cpu = load(name = 'cpab_cpu',
                    sources = [_dir + '/transformer.cpp'],
                    extra_cflags=['-O0', '-g'],
                    verbose=_verbose)
    _cpu_succes = True
    if _verbose:
        print(70*'=')
        print('succesfully compiled cpu source')
        print(70*'=')
except Exception as e:
    cpab_cpu = _notcompiled()
    _cpu_succes = False
    if _verbose:
        print(70*'=')
        print('Unsuccesfully compiled cpu source')
        print('Error was: ')
        print(e)
        print(70*'=')


