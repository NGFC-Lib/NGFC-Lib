##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################
"""
Make sure the supercritical steam cycle example solves.
"""

__author__ = "John Eslick"

import pytest
import pyomo.environ as pyo
from idaes.examples.power_generation.supercritical_steam_cycle import main
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                              activated_equalities_generator)
from idaes.property_models import iapws95


solver_available = pyo.SolverFactory('ipopt').available()
prop_available = iapws95.iapws95_available()


@pytest.fixture(scope="module")
def initialize_model():
    return main()


def gross_power_mw(model):
    # pyo.value(m.fs.turb.power[0]) is the power consumed in Watts
    return -pyo.value(model.fs.turb.power[0])/1e6

@pytest.mark.slow
@pytest.mark.solver
@pytest.mark.skipif(not prop_available, reason="IAPWS not available")
@pytest.mark.skipif(not solver_available, reason="Solver not available")
def test_init(initialize_model):
    m, solver = initialize_model
    # check that the model solved properly and has 0 degrees of freedom
    assert(degrees_of_freedom(m)==0)
    for c in activated_equalities_generator(m):
        assert(abs(c.body() - c.lower) < 1e-4)


@pytest.mark.slow
@pytest.mark.solver
@pytest.mark.skipif(not prop_available, reason="IAPWS not available")
@pytest.mark.skipif(not solver_available, reason="Solver not available")
def test_init_value(initialize_model):
    m, solver = initialize_model
    assert gross_power_mw(m) == pytest.approx(620.8100259113626, abs=1e-3)


@pytest.mark.slow
@pytest.mark.solver
@pytest.mark.skipif(not prop_available, reason="IAPWS not available")
@pytest.mark.skipif(not solver_available, reason="Solver not available")
def test_valve_change(initialize_model):
    m, solver = initialize_model
    m.fs.turb.throttle_valve[1].valve_opening[:].value = 0.25
    solver.solve(m, tee=True)
    assert gross_power_mw(m) == pytest.approx(580.9566851114142, abs=1e-3)
