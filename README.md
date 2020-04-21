# NGFC-Lib <img src="./assets/images/team-logo.png" alt="team-logo" width="120" img align="right">

[![Build Status](https://travis-ci.org/NGFC-Lib/NGFC-Lib.svg?branch=master)](https://travis-ci.org/NGFC-Lib/NGFC-Lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of high-performance C and Fortran codes with a jupyter notebook framework to model natural-gas fuel cell systems. Includes software tools developed by the Institute for the Design of Advanced Energy Systems (IDAES) and machine learning tools for fuel cell optimization.

UW DIRECT capstone project Spring 2020. Collaborators: [PNNL](https://www.pnnl.gov/), [NETL](https://www.netl.doe.gov/), and [ARPA-E](https://arpa-e.energy.gov/).

<img src="./assets/images/UW-logo.png" alt="UW-logo" height="60" img align="right"> <img src="./assets/images/PNNL-logo.png" alt="PNNL-logo" height="60" img align="right" > <img src="./assets/images/NETL-logo.jpg" alt="NETL-logo" height="60" img align="right"> <img src="./assets/images/ARPA-E-logo.jpg" alt="ARPA-E-logo" height="50" img align="right"> 
<br/><br/>

## Background
This wrapper includes the following software packages and models:

1. IDAES Process Systems Engineering Framework (idaes - pse) | [Read the Docs](https://idaes-pse.readthedocs.io/en/stable/) |  [GitHub](https://github.com/IDAES/idaes-pse)
2. IDAES pyomo | [Website](http://www.pyomo.org/installation/) | [GitHub](https://github.com/IDAES/pyomo)
3. Solid Oxide Fuel Cell Multi-Physics Stack Model (SOFC-MP) | [Literature](https://doi.org/10.1016/j.jpowsour.2010.11.123) 

## Python Wrapper for C/ Fortran
Options include: 
SWIG, Cython, PyPy, Numba, f2py, ctypes, instant, PyCXX, boost.python

## SWIG - Simplified Wrapper and Interface Generation
Python allows us to write wrapper functions in C that every object in python is represented by a C structured Py0bject. 
The purpose of writing the wrapper function is to convert between Py0bject ariables and plain C.

Steps to take:
1. Make a SWIG interface file
2. Run SWIG to generate wrapper code
3. Compile and link the C code and the wrapper code.

[Example](http://github.com/UiO-IN3110)

For this project: 
We will create a jupyter notebook that takes in (name of module (header file) and local directory the module is located), then automatically creates SWIG interface file, library function (extension .so) then create python module.

Potential Problem:
C module: 
void function_name(double x, double y, double z); let z be output
then, 
%include "typemaps.i"
void function_name(double x, double y, double **OUTPUT)
For every function call with output imbedded.

## Contributors
* Mihyun Kim
* Henry Lee
* Zang Le
