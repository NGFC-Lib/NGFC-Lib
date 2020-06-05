# NGFC-Lib <img src="./assets/images/team-logo.png" alt="team-logo" width="120" img align="right">

[![Build Status](https://travis-ci.org/NGFC-Lib/NGFC-Lib.svg?branch=master)](https://travis-ci.org/NGFC-Lib/NGFC-Lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python implementation of high-performance C and Fortran codes with a jupyter notebook framework to model natural-gas fuel cell systems. Includes software tools developed by the Institute for the Design of Advanced Energy Systems (IDAES) and machine learning tools for fuel cell optimization.

UW DIRECT capstone project Spring 2020. Collaborators: [PNNL](https://www.pnnl.gov/), [NETL](https://www.netl.doe.gov/), and [ARPA-E](https://arpa-e.energy.gov/).

<img src="./assets/images/UW-logo.png" alt="UW-logo" height="60" img align="right"> <img src="./assets/images/NETL-logo.jpg" alt="NETL-logo" height="60" img align="right"> <img src="./assets/images/ARPA-E-logo.jpg" alt="ARPA-E-logo" height="60" img align="right"> <img src="./assets/images/PNNL-logo.png" alt="PNNL-logo" height="60" img align="right" > 
<br/><br/>

## Background
There are limited open sources on a solid oxide fuel cell(SOFC) system modelling. This github repo is designed by Team Cost Zero from the University of Washington to create a simple SOFC modelling to allow following uses cases:

Use Cases: 
1. Estimation of electrical output
2. Sizing heat exchanger
3. Stream temperature prediction

Based on a SOFC multi-physics stack model from PNNL, our goal is to create a user-friendly python package with a descriptive Jupyter Notebook  that will yield optimized process modelling given a real sysem operation variables. Our python package will be created based on IDAES Process Systems Engineering Framework using pyomo for calculation/optimization on thermodynamics of the process which are not known to many of process optimization modelling packages in the market. 

1. IDAES Process Systems Engineering Framework (idaes - pse) | [Read the Docs](https://idaes-pse.readthedocs.io/en/stable/) |  [GitHub](https://github.com/IDAES/idaes-pse)
2. IDAES pyomo | [Website](http://www.pyomo.org/installation/) | [GitHub](https://github.com/IDAES/pyomo)
3. Solid Oxide Fuel Cell Multi-Physics Stack Model (SOFC-MP) | [Literature](https://doi.org/10.1016/j.jpowsour.2010.11.123) 

## Model Assumptions
Again, this process modelling is a prototype and is currently designed to sufficiently model simple SOFC system. 

1. Basic Mass balance
2. Energy balance - Simple Nernst Equation
3. Use of electrical efficiency to account for polarization loss

## User Guide

1. Clone our github repository
2. Follow Jupyter Notebook (name to be determined)
3. For running error: Refer to Idaes/Pyomo

## Future Works
1. Add additional physical properties package - e.g. Liquid phase.
2. Incorporate options for more complex electrochemistry calculation - Ohmic, Nernst, activation, concentration overpotentials.
3. Explore different solvers to maximize the accuracy of the model - Currently using "binary solver" for higher convergence.
4. Incorporate a python package that will automatically translate the process into flowsheet for better visualization.

## Contributors
* Mihyun Kim
* Henry Lee
* Zang Le
