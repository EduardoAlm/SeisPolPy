=========
SeisPolPy
=========

Overview
--------

SeisPolPy is an open-source project which aims to provide a 
**Python library for processing seismic time series**. 
So far, SeisPolPy provides three, well known, methods that allow the extraction of relevant information from the seismological time series (see [CIT1975]_, [CIT1986]_, [CIT2006]_), as well as a new method developed by the authors. This latter method (RSTFR) consists of an adaptation of already existing methods and improves them by presenting the option of taking advantage of sparsity to process the data.

SeisPolPy main objective is to empower its users with **reliable and efficient methods** to make the task of processing seismic time series more straightforward. 

Installation
------------

SeisPolPy is currently running and tested on Linux (32 and 64 bit) with Python 3.8. 

For installing the package just run::

    pip3 install SeisPolPy

After finishing the library installation, please download the folder **sharedClib**, `here <https://github.com/EduardoAlm/SeisPolPy/tree/main/sharedClib>`_ present, 
and place the *.so* files in the folder where the SeisPolPy functions are being imported.
For due to the complexity associated with matrices operations this project requires the use o Cython created shared libraries to improve its efficiency. 

Example
-------

-----

Documentation and Changelog
---------------------------

The changelog is presented below:

.. changelog::
    :changelog-url: https://seispolpy.readthedocs.io/en/latest/#changelog
    :github: https://github.com/EduardoAlm/SeisPolPy/releases/
    
The documentation can be found `here <https://seispolpy.readthedocs.io/en/latest/>`_.

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

.. [CIT1975] Flinn, E. A. "Signal analysis using rectilinearity and direction of particle motion." Proceedings of the IEEE 53.12 (1965): 1874-1876.
.. [CIT1986] Vidale, John E. "Complex polarization analysis of particle motion." Bulletin of the Seismological society of America 76.5 (1986): 1393-1405.
.. [CIT2006] Pinnegar, C. R. "Polarization analysis and polarization filtering of three-component signals with the time—frequency S transform." Geophysical Journal International 165.2 (2006): 596-606.