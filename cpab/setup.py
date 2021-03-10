#!/usr/bin/env python
# -*- coding: utf-8 -*-

import setuptools

# from distutils.core import setup, Extension
# module1 = Extension('libcpab', sources = ['libcpab/core/cpab_ops.cpp'])


with open("README.md", "r") as fh:
    README = fh.read()

NAME = "cpab"
DESCRIPTION = "Diffeomorphism"
URL = "https://github.com/imartinezl/diffeomorphism"
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
    "console_scripts": [
        #'tsclust=tsclust:main',
    ],
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
    "continuous piecewise-affine velocity fields",
]
LICENSE = "MIT license"
TEST_SUITE = "tests"
REQUIREMENTS = ["numpy", "torch", "scipy", "matplotlib"]
SETUP_REQUIREMENTS = []
TEST_REQUIREMENTS = ["pytest", "pytest-cov"]

setuptools.setup(
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    classifiers=CLASSIFIERS,
    description=DESCRIPTION,
    entry_points=ENTRY_POINTS,
    extras_require=EXTRAS_REQUIRE,
    # ext_modules = [module1]
    include_package_data=False,
    install_requires=REQUIREMENTS,
    keywords=KEYWORDS,
    license=LICENSE,
    long_description=README,
    name=NAME,
    package_data={},
    # packages = setuptools.find_packages(),
    project_urls=PROJECT_URLS,
    python_requires=REQUIRES_PYTHON,
    setup_requires=SETUP_REQUIREMENTS,
    test_suite=TEST_SUITE,
    tests_require=TEST_REQUIREMENTS,
    url=URL,
    # version = VERSION,
    zip_safe=False,
)
