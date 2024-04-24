from setuptools import setup, find_packages
setup(
    name = 'corner_R_analyzer',
    version = '2.0.0',
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib',
    ],
    packages = find_packages()
)