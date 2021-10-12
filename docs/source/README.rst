=========
SeisPolPy
=========

Overview
--------

SeisPolPy is an open-source project which aims to provide a **Python library for processing seismic time series**. 
So far, SeisPolPy provides two, well known, methods that allow the extraction of relevant information from the seismological time series (see [CIT1965]_, [CIT1986]_), as well as a new method developed by the authors ([CIT2021]_). 
This latter method (RSTFR) consists of an adaptation of already existing methods and improves them by presenting the option of taking advantage of sparsity to process the data, upon the use of the module implementing the RSTFR algorithm its asked for the article and its respective authors to be cited.

SeisPolPy main objective is to empower its users with **reliable and efficient methods** to make the task of processing seismic time series more straightforward. 


Installation
------------

SeisPolPy is currently running and tested on Linux (32 and 64 bit) with Python 3.8. 

PyPI
^^^^

For installing the package through PyPI just run::

    pip3 install SeisPolPy

or::

    python3 -m pip install SeisPolPy

Building and Installing
^^^^^^^^^^^^^^^^^^^^^^^

To build the SeisPolPy library, in the root folder, run::

    python3 -m build

which will generate the .whl and .tar.gz files and place them inside the folder **dist**.
Having generated these files, still in the **root** folder, run::

    pip3 install dist/SeisPolPy-**replacewithcurrentversion**-py3-none-any.whl

or::

    pip3 install dist/SeisPolPy-**replacewithcurrentversion**.tar.gz

.. note::
    After finishing the library installation, please download the folder **sharedClib**, `here <https://github.com/EduardoAlm/SeisPolPy/tree/main/sharedClib>`_ present, 
    and place the .so files in the folder where the SeisPolPy functions are being imported. These were created with the Cython package to improve the code efficiency, which was necessary, due to the high complexity present in the matrices operations performed in some of the implemented methods.

.. note::
    SeisPolPy is not yet available at PyPI. The library will only be uploaded upon it's release.


Changelog
---------

The changelog is presented below:

.. changelog::
    :changelog-url: https://seispolpy.readthedocs.io/en/latest/#changelog
    :github: https://github.com/EduardoAlm/SeisPolPy/releases/

Contribution
------------

We encourage everyone to contribute to SeisPolPy progress. We can't do this without you!

Contributors
------------
    - **Eduardo Almeida** - *Maintainer* - `EduardoAlm <https://github.com/EduardoAlm>`_
    - **Hamzeh Mohammadigheymasi** - *Maintainer* - `Hamzeh <https://github.com/SigProSeismology>`_
    - **Paul Crocker** - *Maintainer* - `crockercaria <https://github.com/crockercaria>`_

See also the list of all the `contributors <https://github.com/EduardoAlm/SeisPolPy/graphs/contributors>`_ that participated in this project.

License
-------

This project is licensed under the MIT License - check the `LICENSE <https://github.com/EduardoAlm/SeisPolPy/blob/main/LICENSE.md>`_ file for details.

References
----------

.. [CIT1965] Flinn, E. A. "Signal analysis using rectilinearity and direction of particle motion." Proceedings of the IEEE 53.12 (1965): 1874-1876.
.. [CIT1986] Vidale, John E. "Complex polarization analysis of particle motion." Bulletin of the Seismological society of America 76.5 (1986): 1393-1405.
.. [CIT2021] H. Mohammadigheymasi, P. Crocker, M. Fathi, E. Almeida, G. Silveira, A. Gholami,and M. Schimmel, “Sparsity­promoting approach to polarization analysis of seismicsignals in the time­frequency domain,”IEEE Transactions on Geoscience andRemote Sensing, 7 2021. [Online]. Available:https://doi.org/10.36227/techrxiv.14910063.v1
