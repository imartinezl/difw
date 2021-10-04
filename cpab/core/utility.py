import os

# %%


class Parameters:
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


class methods:
    closed_form = "closed_form"
    numeric = "numeric"

    @staticmethod
    def check(method):
        if method is not None:
            assert method in [methods.closed_form, methods.numeric], (
                "Unknown method, choose between " + methods.closed_form + " or " + methods.numeric
            )

    @staticmethod
    def default(method=None):
        if method is not None:
            return method
        else:
            return methods.closed_form
