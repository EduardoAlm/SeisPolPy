from setuptools import find_packages, setup


setup(
    name='SeisPolPy',
    packages=find_packages(include=["seisPolPyfunctions"]),
    version='0.1.0',
    description='A Python Library for the PROCESSING of SEISMIC TIME SERIES',
    author='Eduardo Rodrigues de Almeida & Hamzeh Mohammadigheymasi',
    license='MIT',
    install_requires=[],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==6.2.3'],
    test_suite='tests',
)
