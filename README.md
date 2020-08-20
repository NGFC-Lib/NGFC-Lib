# NGFC-Lib <img src="./assets/images/team-logo.png" alt="team-logo" width="120" img align="right">

[![Build Status](https://travis-ci.org/NGFC-Lib/NGFC-Lib.svg?branch=master)](https://travis-ci.org/NGFC-Lib/NGFC-Lib)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project utilizes software tools developed by the Institute for the Design of Advanced Energy Systems (IDAES), physics-based models, and machine learning tools for fuel cell optimization.

Repo Set-up: UW DIRECT capstone project Spring 2020. Collaborators: [PNNL](https://www.pnnl.gov/), [NETL](https://www.netl.doe.gov/), and [ARPA-E](https://arpa-e.energy.gov/).

<img src="./assets/images/UW-logo.png" alt="UW-logo" height="60" img align="right"> <img src="./assets/images/NETL-logo.jpg" alt="NETL-logo" height="60" img align="right"> <img src="./assets/images/ARPA-E-logo.jpg" alt="ARPA-E-logo" height="60" img align="right"> <img src="./assets/images/PNNL-logo.png" alt="PNNL-logo" height="60" img align="right" > 
<br/><br/>

## Background
Based on a SOFC multi-physics stack model from PNNL and the IDAES Framework, our goal is to create a collection of user-friendly Jupyter Notebooks that will yield optimized process modelling given a real sysem operation variables. Our examples will be created based on IDAES Process Systems Engineering Framework using pyomo for calculation/optimization on thermodynamics of the process which are not known to many of process optimization modelling packages in the market. 

1. IDAES Process Systems Engineering Framework (idaes - pse) | [Read the Docs](https://idaes-pse.readthedocs.io/en/stable/) |  [GitHub](https://github.com/IDAES/idaes-pse)
2. IDAES pyomo | [Website](http://www.pyomo.org/installation/) | [GitHub](https://github.com/IDAES/pyomo)
3. Solid Oxide Fuel Cell Multi-Physics Stack Model (SOFC-MP) | [Literature](https://doi.org/10.1016/j.jpowsour.2010.11.123) 

## Use Cases
Our software is designed to help solid oxide fuel cell researchers with optimizing their process design and reduce their cost in simulation software. Our software is designed based on IDAES and Pyomo which are both free and open source with high credibility in the system. The detailed user guide offered from our team will simplify the modelling even further allowing researchers with or without any background in SOFC process modelling to easily proceed with process optimization. 
Not only this will allow the calcualtion on power output, flow at steady state and temperature, and flow diagram of overall plant design, but, this will also allow the users to make decision on the sizing of heat exchangers.

## Functional Specifications
User interface:  
What it does: Solve the SOFC problem and make the prediction  
Input: environmental setting and feed/exhaust amount  
Output: SOFC energy output, efficiency, Burner exhaust temperature, and Reformer entrance temperature 
Note: Streams and units of the model are set already

Parameters which user could adjust: 
1. Fuel & air utilization, moles of methane feed
2. Temperature of air & fuel in to and out of the fuel cell


## Model Assumptions
Again, this process modelling is a prototype and is currently designed to sufficiently model simple SOFC system. 

1. Basic Mass balance
2. Energy balance - Simple Nernst Equation
3. Use of electrical efficiency to account for polarization loss

## User Guide

1. Clone our github repository
2. Follow Jupyter Notebook (user_guide.ipynb) https://github.com/NGFC-Lib/NGFC-Lib/blob/master/jupyter/idaes-pse-NGFC-model/user_guide.ipynb
3. For running error: Refer to Idaes/Pyomo
Documentation on Idaes/Pyomo could be found in their website

## Future Works
1. Incorporate options for more complex electrochemistry calculation - Ohmic, Nernst, activation, concentration overpotentials.
2. Explore different solvers to maximize the accuracy of the model - Currently using "binary solver" for higher convergence.
3. Incorporate a python package that will automatically translate the process into flowsheet for better visualization.

## License
MIT License

## DIRECT Contributors
* Mihyun Kim
* Henry Lee
* Zang Le
* Erica Eggleton - lead
