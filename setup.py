"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='cil-text-classification-2018',
        version='0.1',
        description='CIL Project for Text Sentiment Classification.',

        author='Zhifei Yang',
        author_email='zhiyang@student.ethz.ch',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
            'coloredlogs',
            'numpy',
            'scipy',
            'sklearn',
            # 'tensorflow',
        ],
)