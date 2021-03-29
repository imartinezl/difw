import os

# %%


class Parameters:
    pass

    def __repr__(self):
        return str(self.__dict__)

    def __copy__(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone

    def copy(self):
        newone = type(self)()
        newone.__dict__.update(self.__dict__)
        return newone


# %%
def get_dir(file):
    """ Get directory of the input file """
    return os.path.dirname(os.path.realpath(file))


# %%

# TODO: review name: modes, method, 
class modes:
    closed_form = "closed_form"
    numeric = "numeric"

    @staticmethod
    def check_mode(mode):
        if mode is not None:
            assert mode in [modes.closed_form, modes.numeric], "Unknown mode, choose between " + Modes.closed_form + " or " + Modes.numeric

    @staticmethod
    def default(mode=None):
        if mode is not None:
            return mode
        else:
            return modes.closed_form
