import os
import os.path as osp
import shutil
import sys
import warnings
from setuptools import find_packages, setup


def get_version():
    version_file = 'bipc/version.py'
    with open(version_file, 'r', encoding='utf-8') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


if __name__ == '__main__':
    setup(
        name='bipc',
        version=get_version(),
        packages=find_packages(),
        include_package_data=True,
        license='Apache License 2.0',
        zip_safe=False)