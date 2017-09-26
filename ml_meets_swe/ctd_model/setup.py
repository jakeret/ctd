#!/usr/bin/env python

import os
from setuptools import find_packages

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


PACKAGE_PATH = os.path.abspath(os.path.join(__file__, os.pardir))

setup(
    name='ctd_model',
    version='0.1.0',
    description='cifar10 keras model',
    long_description='cifar10 keras model',
    author='Joel Akeret',
    author_email='joel.akeret@zuehlke.com',
    url='https://github.com/jakeret',
    packages=find_packages(PACKAGE_PATH, "test"),
    package_dir={'ctd_model': 'ctd_model'},
    include_package_data=True,
    install_requires=["tensorflow", "keras", "numpy"],
    license='GPLv3',
    zip_safe=False,
    keywords='ctd_model',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: Other/Proprietary License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
    tests_require=[],
 
)
