#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools
import pathlib

here = pathlib.Path(__file__).parent.resolve()

LONG_DESCRIPTION = (here / 'README.md').read_text(encoding='utf-8')

NAME = "cpab"
DESCRIPTION = "CPAB Transformations: finite-dimensional spaces of simple, fast, and highly-expressive diffeomorphisms derived from parametric, continuously-defined, velocity fields in Numpy and Pytorch"
URL = "https://github.com/imartinezl/cpab"
AUTHOR = "Iñigo Martinez"
AUTHOR_EMAIL = "inigomlap@gmail.com"
CLASSIFIERS = [
    "Development Status :: 2 - Pre-Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
]
ENTRY_POINTS = {
    "console_scripts": [],
}
PROJECT_URLS = {
    "Bug Reports": URL + "/issues",
    "Documentation": "https://cpab.readthedocs.io",
    "Source Code": URL,
}
REQUIRES_PYTHON = ">=3.5, <4"
EXTRAS_REQUIRE = {}
KEYWORDS = [
    "diffeomorphisms",
    "tessellations",
    "transformations",
    "continuous piecewise-affine",
    "velocity fields",
    "numpy",
    "pytorch"
]
LICENSE = "MIT license"
TEST_SUITE = "tests"
REQUIREMENTS = ["numpy>=1.20", "matplotlib>=3.4.0", "scipy", "torch>=1.8.0", "ninja"]
SETUP_REQUIREMENTS = []
TEST_REQUIREMENTS = ["pytest", "pytest-cov"]

setuptools.setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    entry_points=ENTRY_POINTS,
    extras_require=EXTRAS_REQUIRE,
    include_package_data=False,
    install_requires=REQUIREMENTS,
    keywords=KEYWORDS,
    license=LICENSE,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    name=NAME,
    package_data={},
    packages = setuptools.find_packages(exclude=("tests",)),
    project_urls=PROJECT_URLS,
    python_requires=REQUIRES_PYTHON,
    setup_requires=SETUP_REQUIREMENTS,
    test_suite=TEST_SUITE,
    tests_require=TEST_REQUIREMENTS,
    url=URL,
    # version = VERSION,
    zip_safe=False,
)
