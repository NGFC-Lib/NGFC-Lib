import unittest

import heatx_network
from pyomo.environ import ConcreteModel

class TestSimulationTools(unittest.TestCase):\
    
    def test_declare_heatx_unit(self):
        h = ConcreteModel()
        assert type(heatx_network.declare_heatx_unit(h)) == ConcreteModel, \
            "Warning: Incorrect type"
        return

     def test_heatx_constraints(self):
        h=ConcreteModel()
        m=ConcreteModel()
        n_H2Of=20
        enthalpy_vap = 43988 
        air_in = 973.15
        assert type(heatx_network.heatx_constraints(m, h, n_H2Of, enthalpy_vap, air_in)) == ConcreteModel, \
            "Warning: Incorrect type"
        return
