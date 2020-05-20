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

"""This is an example supercritical pulverized coal (SCPC) power plant, including 
steam cycle and boiler heat exchanger network model.  
This model doesn't represent any specific power plant, but it represents
what could be considered a typical SCPC plant, producing around ~595 MW gross.
This model is for demonstration and tutorial purposes only. Before looking at the
model, it may be useful to look at the process flow diagram (PFD).
"""

__author__ = "Miguel Zamarripa"

# Import Python libraries
from collections import OrderedDict
import argparse
import logging

# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.network import Arc, Port

# IDAES Imports
from idaes.core import FlowsheetBlock  # Flowsheet class
from idaes.core.util import model_serializer as ms  # load/save model state
from idaes.core.util.misc import svg_tag  # place numbers/text in an SVG
from idaes.property_models import iapws95  # steam properties
from idaes.unit_models.power_generation import (  # power generation unit models
    TurbineMultistage,
    FWH0D,
)
from idaes.unit_models import (  # basic IDAES unit models, and enum
    Mixer,
    HeatExchanger,
    PressureChanger,
    MomentumMixingType,  # Enum type for mixer pressure calculation selection
)
from idaes.core.util import copy_port_values as _set_port  # for model intialization
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.tables import create_stream_table_dataframe  # as Pandas DataFrame

# Callback used to construct heat exchangers with the Underwood approx. for LMTD
from idaes.unit_models.heat_exchanger import delta_temperature_underwood_callback

# Pressure changer type (e.g. adiabatic, pump, isentropic...)
from idaes.unit_models.pressure_changer import ThermodynamicAssumption
from idaes.logger import getModelLogger, getInitLogger, init_tee, condition

_log = getModelLogger(__name__, logging.INFO)


def import_steam_cycle():
    # build concrete model
    # import steam cycle model and initialize flowsheet
    import idaes.examples.power_generation.supercritical_steam_cycle as steam_cycle
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--initialize_from_file",
        help="File from which to load initialized values. If specified, the "
             "initialization proceedure will be skipped.",
        default=None,
    )
    parser.add_argument(
        "--store_initialization",
        help="If specified, store" " the initialized model values, to reload later.",
        default=None,
    )
    args = parser.parse_args()
    m, solver = steam_cycle.main(
        initialize_from_file=args.initialize_from_file,
        store_initialization=args.store_initialization,
    )
    return m, solver
    
if __name__ == "__main__":
    # import steam cycle and build concrete model
    m, solver = import_steam_cycle()
    print(degrees_of_freedom(m))
    #at this point we have a flowsheet with "steam cycle" that solves 
    # correctly, with 0 degrees of freedom.
    
    # next step is to import and build the boiler heat exchanger network
    # importing the boiler heat exchanger network from (boiler_subflowsheet_build.py)
    # will basically append all the unit models into our model ("m") 
    # model "m" has been created a few lines above
    import boiler_subflowsheet_build as blr 
        # import the models (ECON, WW, PrSH, PlSH, FSH, Spliter, Mixer, Reheater)
        # see boiler_subflowhseet_build.py for a beter description
    blr.build_boiler(m.fs)
    #initialize boiler network models (one by one)
    blr.initialize(m)
    # at this point we have both flowsheets (steam cycle + boiler network)
    # in the same model/concrete object ("m")
    # however they are disconnected. Here we want to solve them at the same time
    # this is a square problem (i.e. degrees of freedom = 0)
    print('solving square problem disconnected')
    results = solver.solve(m, tee=True)
    
    # at this point we want to connect the units in both flowsheets
    # Economizer inlet = Feed water heater 8 outlet (water)
    # HP inlet = Attemperator outlet (steam)
    # Reheater inlet (steam) = HP split 7 outlet (last stage of HP turbine)
    # IP inlet = Reheater outlet steam7
    blr.unfix_inlets(m)
    print(degrees_of_freedom(m))
    # user can save the initialization to a json file (uncomment next line)
#    MS.to_json(m, fname = 'SCPC_full.json')
#   later user can use the json file to initialize the model
#   if this is the case comment out previous MS.to_json and uncomment next line
#    MS.from_json(m, fname = 'SCPC_full.json')
    
    # deactivate constraints linking the FWH8 to HP turbine
    m.fs.boiler_pressure_drop.deactivate()
    m.fs.close_flow.deactivate()
    m.fs.turb.constraint_reheat_flow.deactivate()
    m.fs.turb.constraint_reheat_press.deactivate()
    m.fs.turb.constraint_reheat_temp.deactivate()
    m.fs.turb.inlet_split.inlet.enth_mol.unfix()
    m.fs.turb.inlet_split.inlet.pressure.unfix()
    # user can fix the boiler feed water pump pressure (uncomenting next line)
#    m.fs.bfp.outlet.pressure[:].fix(26922222.222))
    
    m.fs.FHWtoECON = Arc(source = m.fs.fwh8.desuperheat.outlet_2,
                      destination = m.fs.ECON.side_1_inlet)
    
    m.fs.Att2HP = Arc(source = m.fs.ATMP1.outlet,
                   destination = m.fs.turb.inlet_split.inlet)
    
    m.fs.HPout2RH = Arc(source = m.fs.turb.hp_split[7].outlet_1,
                     destination = m.fs.RH.side_1_inlet)
    
    m.fs.RHtoIP = Arc(source = m.fs.RH.side_1_outlet,
                   destination =m.fs.turb.ip_stages[1].inlet)
    
    pyo.TransformationFactory("network.expand_arcs").apply_to(m)
    
    #unfix boiler connections
    m.fs.ECON.side_1_inlet.flow_mol.unfix()
    m.fs.ECON.side_1_inlet.enth_mol[0].unfix()
    m.fs.ECON.side_1_inlet.pressure[0].unfix()
    m.fs.RH.side_1_inlet.flow_mol.unfix()
    m.fs.RH.side_1_inlet.enth_mol[0].unfix()
    m.fs.RH.side_1_inlet.pressure[0].unfix()
    m.fs.hotwell.makeup.flow_mol[:].setlb(-1.0)
    
#    if user has trouble with infeasible solutions, an easy test 
#    is to deactivate the link to HP turbine (m.fs.Att2HP_expanded "enth_mol and pressure" equalities) 
#    and fix inlet pressure and enth_mol to turbine (m.fs.turb.inlet_split.inlet)
#   (then double check the values from m.fs.ATMP1.outlet)
#    m.fs.Att2HP_expanded.enth_mol_equality.deactivate()
#    m.fs.Att2HP_expanded.pressure_equality.deactivate()
    m.fs.turb.inlet_split.inlet.pressure.fix(2.423e7)
#    m.fs.turb.inlet_split.inlet.enth_mol.fix(62710.01)

#   finally, since we want to maintain High Pressure (HP) inlet temperature constant (~866 K)
#   we need to fix Attemperator enthalpy outlet and unfix heat duty to Platen superheater
#   note fixing enthalpy to control temperature is only valid because pressure is also fixed
    m.fs.ATMP1.outlet.enth_mol[0].fix(62710.01)
    m.fs.PlSH.heat_duty[:].unfix()#fix(5.5e7)
#    m.fs.ATMP1.SprayWater.flow_mol[0].unfix()
    
    print(degrees_of_freedom(m))
    solver.options = {
        "tol": 1e-6,
        "linear_solver": "ma27",
        "max_iter": 40,
    }
    #square problems tend to work better without bounds
    strip_bounds = pyo.TransformationFactory('contrib.strip_var_bounds')
    strip_bounds.apply_to(m, reversible=True)
    # this is the final solve with both flowsheets connected
    results = solver.solve(m, tee=True)
