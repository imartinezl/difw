.. cpab documentation master file, created by
  sphinx-quickstart on Mon Jun 28 18:23:50 2021.
  You can adapt this file completely to your liking, but it should at least
  contain the root `toctree` directive.

|

.. figure:: docs/source/_static/logo.png
  :width: 300
  :align: center


|

Finite-dimensional spaces of simple, fast, and highly-expressive diffeomorphisms derived from parametric, continuously-defined, velocity fields in Numpy and Pytorch

.. image:: https://img.shields.io/pypi/status/cpab?style=flat-square
    :target: https://pypi.python.org/pypi/cpab
    :alt: PyPI Status

.. image:: https://img.shields.io/pypi/v/cpab?style=flat-square
    :target: https://pypi.python.org/pypi/cpab
    :alt: PyPI Version

.. image:: https://img.shields.io/github/license/imartinezl/cpab?style=flat-square
    :target: https://github.com/imartinezl/cpab/blob/master/LICENSE
    :alt: License

.. image:: https://img.shields.io/github/workflow/status/imartinezl/cpab/Workflow?style=flat-square
    :target: https://github.com/imartinezl/cpab/actions
    :alt: Actions

.. image:: https://img.shields.io/pypi/dm/cpab?style=flat-square
    :target: https://pepy.tech/project/cpab

.. image:: https://img.shields.io/github/languages/top/imartinezl/cpab?style=flat-square
    :target: https://github.com/imartinezl/cpab
    :alt: Top Language

.. image:: https://img.shields.io/github/issues/imartinezl/cpab?style=flat-square
    :target: https://github.com/imartinezl/cpab
    :alt: Github Issues


Getting Started
---------------

Use the following template to run a simulation with `cpab`:

.. code-block:: python

    import cpab



Installation
------------

As the compiled **cpab** package is hosted on the Python Package Index (PyPI) you can easily install it with ``pip``.
To install **cpab**, run this command in your terminal of choice:

.. code-block:: shell-session

    $ pip install cpab

or, alternatively:

.. code-block:: shell-session

    $ python -m pip install cpab

If you want to get **cpab**'s latest version, you can refer to the
repository hosted at github:

.. code-block:: shell-session

    python -m pip install https://github.com/imartinezl/cpab/archive/master.zip

Environment Setup
-----------------

Requirements
^^^^^^^^^^^^

**cpab** builds on ``numpy``, ``torch``, ``scipy``, ``ninja``,  and ``matplotlib`` libraries.

Python 3
^^^^^^^^

To find out which version of ``python`` you have, open a terminal window and try the following command:

.. code-block:: shell-session

    $ python3 --version
    Python 3.6.9

If you have ``python3`` on your machine, then this command should respond with a version number. If you do not have ``python3`` installed, follow these `instructions <https://realpython.com/installing-python>`_.

Pip
^^^

``pip`` is the reference Python package manager. It’s used to install and update packages. In case ``pip`` is not installed in your OS, follow these `procedure <https://pip.pypa.io/en/stable/installation/>`_.


Virtual Environment
^^^^^^^^^^^^^^^^^^^

``venv`` creates a “virtual” isolated Python installation and installs packages into that virtual installation. It is always recommended to use a virtual environment while developing Python applications. To create a virtual environment, go to your project’s directory and run venv.

.. code-block:: shell-session

    $ python3 -m venv env

Before you can start installing or using packages in your virtual environment you’ll need to activate it. 

.. code-block:: shell-session

    $ source env/bin/activate


Source Code
-----------

cpab is developed on GitHub, where the code is
`always available <https://github.com/imartinezl/cpab>`_.

You can either clone the public repository:

.. code-block:: shell-session

    $ git clone git://github.com/imartinezl/cpab.git

Or, download the `tarball <https://github.com/imartinezl/cpab/tarball/main>`_:

.. code-block:: shell-session

    $ curl -OL https://github.com/imartinezl/cpab/tarball/main
    # optionally, zipball is also available (for Windows users).

Once you have a copy of the source, you can embed it in your own Python
package, or install it into your site-packages easily:


.. code-block:: shell-session

    $ cd cpab
    $ python -m pip install .

