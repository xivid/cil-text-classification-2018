"""Setup module for project."""

from setuptools import setup, find_packages

setup(
        name='cil-text-classification-2018',
        version='0.1',
        description='CIL Project for Text Sentiment Classification.',

        author='CIL Group 42',
        author_email='',

        packages=find_packages(exclude=[]),
        python_requires='>=3.5',
        install_requires=[
            'coloredlogs',
            'numpy',
            'scipy',
            'sklearn',
            'gensim',
            'lightgbm',
            'tensorflow',
            'keras'
        ],
)