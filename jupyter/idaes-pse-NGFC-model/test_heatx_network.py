import heatx_network
from pyomo.environ import ConcreteModel


def declare_heatx_unit():
    h = ConcreteModel()
    assert type(heatx_network.declare_heatx_unit(h)) == ConcreteModel, \
        "Warning: Incorrect type"
    return
