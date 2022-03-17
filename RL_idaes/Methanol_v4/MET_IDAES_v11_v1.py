##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2020, by the
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
Task: IDAES Support for ARPE-E Differentiate
Scenario: Methanol Synthesis From Syngas
Author: D. Wang, J. Bao, Y. Chen, T. Ma, B. Paul and M. Zamarripa
"""

from timeit import default_timer as timer
import os
import sys
sys.path.append(os.path.abspath("./METH_properties"))

# Import Pyomo libraries
from pyomo.environ import (Constraint,
                           Objective,
                           Var,
                           Expression,
                           ConcreteModel,
                           TransformationFactory,
                           value,
                           maximize,
                           units as pyunits)
from pyomo.environ import TerminationCondition

# Import IDAES core libraries
from idaes.core import FlowsheetBlock
from idaes.core.util import get_solver
from idaes.core.util.model_statistics import (degrees_of_freedom, fixed_variables_set)
from idaes.core.util.initialization import propagate_state
import idaes.logger as idaeslog
from pyomo.network import Arc, SequentialDecomposition

# Import required models
from idaes.generic_models.properties.core.generic.generic_property import \
    GenericParameterBlock
from idaes.generic_models.properties.core.generic.generic_reaction import \
    GenericReactionParameterBlock

import methanol_water_ideal as thermo_props
import methanol_reactions as reaction_props

from idaes.generic_models.unit_models import (
    Mixer,
    Heater,
    Compressor,
    Turbine,
    StoichiometricReactor,
    Separator as Splitter,
    Product,
    Feed,
    Flash)
from idaes.generic_models.unit_models.mixer import MomentumMixingType
from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
import idaes.core.util.unit_costing as costing
from idaes.core.util.unit_costing import initialize as init_costing

#%%
def build_model(m, list_unit, list_inlet, list_outlet):
    
    # # change upper state bound on temperature to allow convergence
    # thermo_props.config_dict["state_bounds"]["temperature"] = (198.15, 298.15, 512.15, pyunits.K)
    
    m.fs.thermo_params = GenericParameterBlock(
        default=thermo_props.config_dict)
    m.fs.reaction_params = GenericReactionParameterBlock(
        default={"property_package": m.fs.thermo_params,
                 **reaction_props.config_dict})

    # exhaust
    if 'exhaust' in list_unit:
        m.fs.exhaust = Product(default={'property_package': m.fs.thermo_params})

    # mixing feed streams
    if 'mixer_0' in list_unit: # must exist
        m.fs.mixer_0 = Mixer(
            default={"property_package": m.fs.thermo_params,
                    "momentum_mixing_type": MomentumMixingType.minimize,
                    "has_phase_equilibrium": True,
                    "inlet_list": ['inlet_1', 'inlet_2']})
    
    if 'mixer_1' in list_unit:
        m.fs.mixer_1 = Mixer(
            default={"property_package": m.fs.thermo_params,
                    "momentum_mixing_type": MomentumMixingType.minimize,
                    "has_phase_equilibrium": True,
                    "inlet_list": ['inlet_1', 'inlet_2']})

    # pre-compression
    if 'compressor_1' in list_unit:
        m.fs.compressor_1 = Compressor(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params,
                    "compressor": True,
                    "thermodynamic_assumption": ThermodynamicAssumption.isothermal
                    })

    if 'compressor_2' in list_unit:
        m.fs.compressor_2 = Compressor(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params,
                    "compressor": True,
                    "thermodynamic_assumption": ThermodynamicAssumption.isothermal
                    })

    # pre-heating
    if 'heater_1' in list_unit:
        m.fs.heater_1 = Heater(
            default={"property_package": m.fs.thermo_params,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    if 'heater_2' in list_unit:
        m.fs.heater_2 = Heater(
            default={"property_package": m.fs.thermo_params,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    # reactor
    if 'reactor_1' in list_unit:
        m.fs.reactor_1 = StoichiometricReactor(
            default={"has_heat_transfer": True,
                    "has_heat_of_reaction": True,
                    "has_pressure_change": False,
                    "property_package": m.fs.thermo_params,
                    "reaction_package": m.fs.reaction_params})

    if 'reactor_2' in list_unit:
        m.fs.reactor_2 = StoichiometricReactor(
            default={"has_heat_transfer": True,
                    "has_heat_of_reaction": True,
                    "has_pressure_change": False,
                    "property_package": m.fs.thermo_params,
                    "reaction_package": m.fs.reaction_params})

    # post-expansion
    if 'turbine_1' in list_unit:
        m.fs.turbine_1 = Turbine(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params})

    if 'turbine_2' in list_unit:
        m.fs.turbine_2 = Turbine(
            default={"dynamic": False,
                    "property_package": m.fs.thermo_params})

    # post-cooling
    if 'cooler_1' in list_unit:
        m.fs.cooler_1 = Heater(
            default={"property_package": m.fs.thermo_params,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    if 'cooler_2' in list_unit:
        m.fs.cooler_2 = Heater(
            default={"property_package": m.fs.thermo_params,
                    "has_pressure_change": False,
                    "has_phase_equilibrium": False})

    # product recovery
    if 'flash_0' in list_unit:
        m.fs.flash_0 = Flash(
            default={"property_package": m.fs.thermo_params,
                    "has_heat_transfer": True,
                    "has_pressure_change": True})

    if 'flash_1' in list_unit:
        m.fs.flash_1 = Flash(
            default={"property_package": m.fs.thermo_params,
                    "has_heat_transfer": True,
                    "has_pressure_change": True})

    # splitter
    if 'splitter_1' in list_unit:
        m.fs.splitter_1 = Splitter(default={
            "property_package": m.fs.thermo_params,
            "ideal_separation": False,
             "outlet_list": ["outlet_1", "outlet_2"]})

    if 'splitter_2' in list_unit:
        m.fs.splitter_2 = Splitter(default={
            "property_package": m.fs.thermo_params,
            "ideal_separation": False,
             "outlet_list": ["outlet_1", "outlet_2"]})

    # build arcs
    for i in range(len(list_inlet)):
        expression = 'm.fs.Arc'+str(i)+' = Arc(source = m.fs.'\
            +list_outlet[i]+', destination = m.fs.'+list_inlet[i]+')'
        exec(expression)

    TransformationFactory("network.expand_arcs").apply_to(m)
    print("Degrees of Freedom (build) = %d" % degrees_of_freedom(m))

def set_inputs(m, list_unit):

    #  feed streams, post WGS
    if 'mixer_0' in list_unit:
        m.fs.mixer_0.inlet_1.flow_mol[0].fix(637.2)  # mol/s, relative to 177 kmol/h (49.2 mol/s)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "H2"].fix(1)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "CO"].fix(1e-6)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "CH3OH"].fix(1e-6)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "CH4"].fix(1e-6)
        m.fs.mixer_0.inlet_1.mole_frac_comp[0, "H2O"].fix(1e-6)
        m.fs.mixer_0.inlet_1.enth_mol[0].fix(-142.4)  # J/mol
        m.fs.mixer_0.inlet_1.pressure.fix(30e5)  # Pa

        m.fs.mixer_0.inlet_2.flow_mol[0].fix(316.8)  # mol/s, relative to 88 kmol/h (24.4 mol/s)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "H2"].fix(1e-6)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "CO"].fix(1)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "CH3OH"].fix(1e-6)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "CH4"].fix(1e-6)
        m.fs.mixer_0.inlet_2.mole_frac_comp[0, "H2O"].fix(1e-6)
        m.fs.mixer_0.inlet_2.enth_mol[0].fix(-110676.4)  # J/mol
        m.fs.mixer_0.inlet_2.pressure.fix(30e5)  # Pa

    # units specifications
    if 'compressor_1' in list_unit:
        m.fs.compressor_1.outlet.pressure.fix(51e5)  # Pa

    if 'compressor_2' in list_unit:
        m.fs.compressor_2.outlet.pressure.fix(51e5)  # Pa

    if 'heater_1' in list_unit:
        m.fs.heater_1.outlet_temp = Constraint(
            expr=m.fs.heater_1.control_volume.properties_out[0].temperature == 488.15)

    if 'heater_2' in list_unit:
        m.fs.heater_2.outlet_temp = Constraint(
            expr=m.fs.heater_2.control_volume.properties_out[0].temperature == 488.15)

    if 'reactor_1' in list_unit:
        m.fs.reactor_1.conversion = Var(initialize=0.75, bounds=(0, 1))
        m.fs.reactor_1.conv_constraint = Constraint(
            expr=(m.fs.reactor_1.conversion * m.fs.reactor_1.inlet.flow_mol[0] *
                m.fs.reactor_1.inlet.mole_frac_comp[0, "CO"] ==
                m.fs.reactor_1.inlet.flow_mol[0] *
                m.fs.reactor_1.inlet.mole_frac_comp[0, "CO"]
                - m.fs.reactor_1.outlet.flow_mol[0] *
                m.fs.reactor_1.outlet.mole_frac_comp[0, "CO"]))
        m.fs.reactor_1.conversion.fix(0.75)
        m.fs.reactor_1.outlet_temp = Constraint(
            expr=m.fs.reactor_1.control_volume.properties_out[0].temperature == 507.15)
        m.fs.reactor_1.heat_duty.setub(0)  # rxn is exothermic, so duty is cooling only

    if 'reactor_2' in list_unit:
        m.fs.reactor_2.conversion = Var(initialize=0.75, bounds=(0, 1))
        m.fs.reactor_2.conv_constraint = Constraint(
            expr=(m.fs.reactor_2.conversion * m.fs.reactor_2.inlet.flow_mol[0] *
                m.fs.reactor_2.inlet.mole_frac_comp[0, "CO"] ==
                m.fs.reactor_2.inlet.flow_mol[0] *
                m.fs.reactor_2.inlet.mole_frac_comp[0, "CO"]
                - m.fs.reactor_2.outlet.flow_mol[0] *
                m.fs.reactor_2.outlet.mole_frac_comp[0, "CO"]))
        m.fs.reactor_2.conversion.fix(0.75)
        m.fs.reactor_2.outlet_temp = Constraint(
            expr=m.fs.reactor_2.control_volume.properties_out[0].temperature == 507.15)
        m.fs.reactor_2.heat_duty.setub(0)  # rxn is exothermic, so duty is cooling only

    if 'turbine_1' in list_unit:
        m.fs.turbine_1.deltaP.fix(-2e6)
        m.fs.turbine_1.efficiency_isentropic.fix(0.9)

    if 'turbine_2' in list_unit:
        m.fs.turbine_2.deltaP.fix(-2e6)
        m.fs.turbine_2.efficiency_isentropic.fix(0.9)

    if 'cooler_1' in list_unit:
        m.fs.cooler_1.outlet_temp = Constraint(
            expr=m.fs.cooler_1.control_volume.properties_out[0].temperature == 407.15)

    if 'cooler_2' in list_unit:
        m.fs.cooler_2.outlet_temp = Constraint(
            expr=m.fs.cooler_2.control_volume.properties_out[0].temperature == 407.15)

    if 'flash_0' in list_unit:
        m.fs.flash_0.recovery = Var(initialize=0.01, bounds=(0, 1))
        m.fs.flash_0.rec_constraint = Constraint(
            expr=(m.fs.flash_0.recovery == m.fs.flash_0.liq_outlet.flow_mol[0] *
                m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"] /
                (m.fs.flash_0.inlet.flow_mol[0] *
                m.fs.flash_0.inlet.mole_frac_comp[0, "CH3OH"])))
        m.fs.flash_0.deltaP.fix(0)  # Pa
        m.fs.flash_0.outlet_temp = Constraint(
            expr=m.fs.flash_0.control_volume.properties_out[0].temperature == 407.15)

    if 'flash_1' in list_unit:
        m.fs.flash_1.recovery = Var(initialize=0.01, bounds=(0, 1))
        m.fs.flash_1.rec_constraint = Constraint(
            expr=(m.fs.flash_1.recovery == m.fs.flash_1.liq_outlet.flow_mol[0] *
                m.fs.flash_1.liq_outlet.mole_frac_comp[0, "CH3OH"] /
                (m.fs.flash_1.inlet.flow_mol[0] *
                m.fs.flash_1.inlet.mole_frac_comp[0, "CH3OH"])))
        m.fs.flash_1.deltaP.fix(0)  # Pa
        m.fs.flash_1.outlet_temp = Constraint(
            expr=m.fs.flash_1.control_volume.properties_out[0].temperature == 407.15)

    if 'splitter_1' in list_unit:
        m.fs.splitter_1.split_fraction[0, "outlet_1"].fix(0.9999)
        
    if 'splitter_2' in list_unit:
        m.fs.splitter_2.split_fraction[0, "outlet_1"].fix(0.9999)
        
    if 'flash_0' in list_unit:
        m.fs.meth_flow = Expression(expr=(m.fs.flash_0.liq_outlet.flow_mol[0] * \
                                     m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"]))
        m.fs.efficiency = Expression(expr=(m.fs.flash_0.liq_outlet.flow_mol[0] * \
                                     m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"]/316.8))

    print("Degrees of Freedom (set inputs) = %d" % degrees_of_freedom(m))

def initialize_flowsheet(m, list_unit, list_inlet, list_outlet, iterlim):

    # Initialize and solve flowsheet
    
    # solver options
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = iterlim

    # Using the SD tool, build the network we will solve
    G = seq.create_graph(m)
    heuristic_tear_set = seq.tear_set_arcs(G, method="heuristic")

    # order = seq.calculation_order(G)
    # print('\nTear Stream:')
    # for o in heuristic_tear_set:
    #     print(o.name, ': ', o.source.name, ' to ', o.destination.name)
    # print('\nCalculation order:')
    # for o in order:
    #     for p in o:
    #         print(p.name, end=' ')
    #     print()

    tear_guesses = {
        "flow_mol": {0: 954.00},
        "mole_frac_comp": {
                (0, "CH4"): 1e-6,
                (0, "CO"): 0.33207,
                (0, "H2"): 0.66792,
                (0, "CH3OH"): 1e-6,
                (0, "H2O"): 1e-6},
        "enth_mol": {0: -36848},
        "pressure": {0: 3e6}}
    
    # automatically build stream set for flowsheet and find the tear stream
    stream_set = [arc for arc in m.fs.component_data_objects(Arc)]
    for stream in stream_set:
        if stream in heuristic_tear_set:
            seq.set_guesses_for(stream.destination, tear_guesses)

    def function(unit):

        # print('solving ', str(unit))
        unit.initialize(outlvl=idaeslog.ERROR)  # no output unless it breaks

        for stream in stream_set:
            if stream.source.parent_block() == unit:
                propagate_state(arc=stream)  # this is an outlet of the unit
            stream.destination.unfix()
    
    seq.run(m, function)
    
    for stream in stream_set:
        if stream.destination.is_fixed() is True:
            # print('Unfixing ', stream.destination.name, '...')
            stream.destination.unfix()
        
    print("\nDegrees of Freedom (after initialize) = %d" % degrees_of_freedom(m))

def add_costing(m, list_unit):

    # Expression to compute the total cooling cost (F/R cooling not assumed)
    expression = 'm.fs.cooling_cost = Expression(expr=(0.0'
    if 'cooler_1' in list_unit:
        expression += ' -m.fs.cooler_1.heat_duty[0] * 0.212e-7' 
    if 'cooler_2' in list_unit:
        expression += ' -m.fs.cooler_2.heat_duty[0] * 0.212e-7' 
    if 'flash_0' in list_unit:
        expression += ' -m.fs.flash_0.heat_duty[0] * 0.25e-7' 
    if 'flash_1' in list_unit:
        expression += ' -m.fs.flash_1.heat_duty[0] * 0.25e-7' 
    if 'reactor_1' in list_unit:
        expression += ' -m.fs.reactor_1.heat_duty[0] * 2.2e-7' 
    if 'reactor_2' in list_unit:
        expression += ' -m.fs.reactor_2.heat_duty[0] * 2.2e-7' 
    expression += '))'
    exec(expression)

    # Expression to compute the total heating cost (F/R heating not assumed)
    expression = 'm.fs.heating_cost = Expression(expr=(0.0'
    if 'heater_1' in list_unit:
        expression += ' + m.fs.heater_1.heat_duty[0] * 2.2e-7' 
    if 'heater_2' in list_unit:
        expression += ' + m.fs.heater_2.heat_duty[0] * 2.2e-7' 
    expression += '))'
    exec(expression)

    # Expression to compute the total electricity cost (utilities - credit)
    expression = 'm.fs.electricity_cost = Expression(expr=(0.0'
    if 'turbine_1' in list_unit:
        expression += ' - m.fs.turbine_1.work_isentropic[0] * 0.08e-5' 
    if 'turbine_2' in list_unit:
        expression += ' - m.fs.turbine_2.work_isentropic[0] * 0.08e-5' 
    if 'compressor_1' in list_unit:
        expression += ' + m.fs.compressor_1.work_mechanical[0] * 0.12e-5' 
    if 'compressor_2' in list_unit:
        expression += ' + m.fs.compressor_2.work_mechanical[0] * 0.12e-5' 
    expression += '))'
    exec(expression)

    # Expression to compute the total operating cost
    m.fs.operating_cost = Expression(
        expr=(3600 * 24 * 365 * (m.fs.heating_cost + m.fs.cooling_cost
                                 + m.fs.electricity_cost)))

    # Expression to compute the annualized capital cost
    expression = 'm.fs.annualized_capital_cost = Expression(expr=(0.0'

    # Computing reactor capital cost
    if 'reactor_1' in list_unit:
        m.fs.reactor_1.get_costing()
        m.fs.reactor_1.diameter.fix(2)
        m.fs.reactor_1.length.fix(4)  # for initial problem at 75% conversion
        init_costing(m.fs.reactor_1.costing)
        # Reactor length (size, and capital cost) is adjusted based on conversion
        # surrogate model which scales length linearly with conversion
        m.fs.reactor_1.length.unfix()
        m.fs.reactor_1.L_eq = Constraint(expr=m.fs.reactor_1.length ==
                                    13.2000*m.fs.reactor_1.conversion - 5.9200)
        # m.fs.reactor_1.conversion_lb = Constraint(expr=m.fs.reactor_1.conversion >= 0.75)
        # m.fs.reactor_1.conversion_ub = Constraint(expr=m.fs.reactor_1.conversion <= 0.85)
        expression += ' + m.fs.reactor_1.costing.purchase_cost'

    if 'reactor_2' in list_unit:
        m.fs.reactor_2.get_costing()
        m.fs.reactor_2.diameter.fix(2)
        m.fs.reactor_2.length.fix(4)  # for initial problem at 75% conversion
        init_costing(m.fs.reactor_2.costing)
        # Reactor length (size, and capital cost) is adjusted based on conversion
        # surrogate model which scales length linearly with conversion
        m.fs.reactor_2.length.unfix()
        m.fs.reactor_2.L_eq = Constraint(expr=m.fs.reactor_2.length ==
                                    13.2000*m.fs.reactor_2.conversion - 5.9200)
        # m.fs.reactor_2.conversion_lb = Constraint(expr=m.fs.reactor_2.conversion >= 0.75)
        # m.fs.reactor_2.conversion_ub = Constraint(expr=m.fs.reactor_2.conversion <= 0.85)
        expression += ' + m.fs.reactor_2.costing.purchase_cost'

    # Computing flash capital cost
    if 'flash_0' in list_unit:
        m.fs.flash_0.get_costing()
        m.fs.flash_0.diameter.fix(2)
        m.fs.flash_0.length.fix(4)
        init_costing(m.fs.flash_0.costing)
        expression += ' + m.fs.flash_0.costing.purchase_cost'

    if 'flash_1' in list_unit:
        m.fs.flash_1.get_costing()
        m.fs.flash_1.diameter.fix(2)
        m.fs.flash_1.length.fix(4)
        init_costing(m.fs.flash_1.costing)
        expression += ' + m.fs.flash_1.costing.purchase_cost'

    # Computing heater/cooler capital costs
    # Surrogates prepared with IDAES shell and tube hx considering IP steam and
    # assuming steam outlet is condensed
    if 'heater_1' in list_unit:
        m.fs.heater_1.cost_heater = Expression(
            expr=0.036158*m.fs.heater_1.heat_duty[0] + 63931.475,
            doc='capital cost of heater in $')
        expression += ' + m.fs.heater_1.cost_heater'

    if 'heater_2' in list_unit:
        m.fs.heater_2.cost_heater = Expression(
            expr=0.036158*m.fs.heater_2.heat_duty[0] + 63931.475,
            doc='capital cost of heater in $')
        expression += ' + m.fs.heater_2.cost_heater'

    # Surrogates prepared with IDAES shell and tube hx considering cooling
    # water assuming that water inlet T is 25 deg C and outlet T is 40 deg C
    if 'cooler_1' in list_unit:
        m.fs.cooler_1.cost_heater = Expression(
            expr=0.10230*(-m.fs.cooler_1.heat_duty[0]) + 100421.572,
            doc='capital cost of cooler in $')
        expression += ' + m.fs.cooler_1.cost_heater'

    if 'cooler_2' in list_unit:
        m.fs.cooler_2.cost_heater = Expression(
            expr=0.10230*(-m.fs.cooler_2.heat_duty[0]) + 100421.572,
            doc='capital cost of cooler in $')
        expression += ' + m.fs.cooler_2.cost_heater'

    expression += ') * 5.4 / 15)'
    exec(expression)

    # methanol price $449 us dollars per metric ton  - 32.042 g/mol
    # - 1 gr = 1e-6 MT  -- consider 1000
    # H2 $16.51 per kilogram - 2.016 g/mol
    # CO $62.00 per kilogram - 28.01 g/mol
    expression = 'm.fs.sales = Expression(expr=(0.0'
    if 'flash_0' in list_unit:
        expression += ' + m.fs.flash_0.liq_outlet.flow_mol[0] * m.fs.flash_0.liq_outlet.mole_frac_comp[0, "CH3OH"] * 32.042 * 1e-6 * 449 * 1000'

    expression += ') * 3600 *24 *365)'
    exec(expression)

    expression = 'm.fs.raw_mat_cost = Expression(expr=(0.0'
    if 'mixer_0' in list_unit:
        expression += ' + m.fs.mixer_0.inlet_1.flow_mol[0] * 16.51 * 2.016 / 1000'
        expression += ' + m.fs.mixer_0.inlet_2.flow_mol[0] * 62.00 * 28.01 / 1000'
    expression += ') * 3600 * 24 * 365)'
    exec(expression)

    m.fs.revenue = Expression(expr=(m.fs.sales - m.fs.operating_cost - \
        m.fs.annualized_capital_cost - m.fs.raw_mat_cost)/1000)

    print("\nDegrees of Freedom (add costing) = %d" % degrees_of_freedom(m))

    if 'flash_0' in list_unit:
        costing.calculate_scaling_factors(m.fs.flash_0.costing)
    if 'flash_1' in list_unit:
        costing.calculate_scaling_factors(m.fs.flash_1.costing)

    if 'reactor_1' in list_unit:
        costing.calculate_scaling_factors(m.fs.reactor_1.costing)
    if 'reactor_2' in list_unit:
        costing.calculate_scaling_factors(m.fs.reactor_2.costing)

def report(m, list_unit):

    print("\nDisplay some results:")

    if 'reactor_1' in list_unit:
        print('reactor 1 Reaction conversion (0.75): ', 
        m.fs.reactor_1.conversion.value)
    if 'reactor_2' in list_unit:
        print('reactor 2 Reaction conversion (0.75): ', 
        m.fs.reactor_2.conversion.value)
    if 'flash_0' in list_unit:
        print('Methanol recovery(%): ', value(100*m.fs.flash_0.recovery))
        print('CH3OH flow rate (mol/s): ', value(m.fs.meth_flow))
        print("methanol production rate(%): ", value(m.fs.efficiency)*100)
    print('annualized capital cost ($/year) =', value(m.fs.annualized_capital_cost))
    print('operating cost ($/year) = ', value(m.fs.operating_cost))
    print('sales ($/year) = ', value(m.fs.sales))
    print('raw materials cost ($/year) =', value(m.fs.raw_mat_cost))
    print('revenue (1000$/year)= ', value(m.fs.revenue))

def add_bounds_v1(m, list_unit):

    # Set up Optimization Problem (Maximize Revenue)
    # keep process pre-reaction fixed and unfix some post-process specs
    
    if 'reactor_1' in list_unit:
        # m.fs.reactor_1.conversion.unfix()
        # m.fs.reactor_1.conversion_lb = Constraint(expr=m.fs.reactor_1.conversion >= 0.75)
        # m.fs.reactor_1.conversion_ub = Constraint(expr=m.fs.reactor_1.conversion <= 0.85)
        m.fs.reactor_1.outlet_temp.deactivate()
        m.fs.reactor_1.outlet_t_lb = Constraint(
            expr=m.fs.reactor_1.control_volume.properties_out[0.0].temperature >= 405)
        m.fs.reactor_1.outlet_t_ub = Constraint(
            expr=m.fs.reactor_1.control_volume.properties_out[0.0].temperature <= 505) #solving process is very sensitive to this temperature

    if 'reactor_2' in list_unit:
        # m.fs.reactor_2.conversion.unfix()
        # m.fs.reactor_2.conversion_lb = Constraint(expr=m.fs.reactor_2.conversion >= 0.75)
        # m.fs.reactor_2.conversion_ub = Constraint(expr=m.fs.reactor_2.conversion <= 0.85)
        m.fs.reactor_2.outlet_temp.deactivate()
        m.fs.reactor_2.outlet_t_lb = Constraint(
            expr=m.fs.reactor_2.control_volume.properties_out[0.0].temperature >= 405)
        m.fs.reactor_2.outlet_t_ub = Constraint(
            expr=m.fs.reactor_2.control_volume.properties_out[0.0].temperature <= 505)

    # Optimize turbine work (or delta P)
    if 'turbine_1' in list_unit:
        m.fs.turbine_1.deltaP.unfix()  # optimize turbine work recovery/pressure drop
        m.fs.turbine_1.outlet_p_lb = Constraint(
            expr=m.fs.turbine_1.outlet.pressure[0] >= 10E5)
        m.fs.turbine_1.outlet_p_ub = Constraint(
            expr=m.fs.turbine_1.outlet.pressure[0] <= 51E5*0.8)

    if 'turbine_2' in list_unit:
        m.fs.turbine_2.deltaP.unfix()  # optimize turbine work recovery/pressure drop
        m.fs.turbine_2.outlet_p_lb = Constraint(
            expr=m.fs.turbine_2.outlet.pressure[0] >= 10E5)
        m.fs.turbine_2.outlet_p_ub = Constraint(
            expr=m.fs.turbine_2.outlet.pressure[0] <= 51E5*0.8)

    # # Optimize cooler outlet temperature - unfix cooler outlet temperature
    # if 'cooler_1' in list_unit: 
    #     m.fs.cooler_1.outlet_temp.deactivate()
    #     m.fs.cooler_1.outlet_t_lb = Constraint(
    #         expr=m.fs.cooler_1.control_volume.properties_out[0.0].temperature
    #         >= 407.15*0.9)
    #     m.fs.cooler_1.outlet_t_ub = Constraint(
    #         expr=m.fs.cooler_1.control_volume.properties_out[0.0].temperature
    #         <= 407.15*1.1)
    #     # m.fs.cooler_1.heat_duty.setub(0)

    # if 'cooler_2' in list_unit: 
    #     m.fs.cooler_2.outlet_temp.deactivate()
    #     m.fs.cooler_2.outlet_t_lb = Constraint(
    #         expr=m.fs.cooler_2.control_volume.properties_out[0.0].temperature
    #         >= 407.15*0.9)
    #     m.fs.cooler_2.outlet_t_ub = Constraint(
    #         expr=m.fs.cooler_2.control_volume.properties_out[0.0].temperature
    #         <= 407.15*1.1)
    #     # m.fs.cooler_2.heat_duty.setub(0)
        
    # Optimize heater properties
    if 'heater_1' in list_unit:
        m.fs.heater_1.outlet_temp.deactivate()
        m.fs.heater_1.outlet_t_lb = Constraint(
            expr=m.fs.heater_1.control_volume.properties_out[0.0].temperature
            >= 480)
        m.fs.heater_1.outlet_t_ub = Constraint(
            expr=m.fs.heater_1.control_volume.properties_out[0.0].temperature
            <= 490)
        # m.fs.heater_1.heat_duty.setlb(0)
    
    if 'heater_2' in list_unit:
        m.fs.heater_2.outlet_temp.deactivate()
        m.fs.heater_2.outlet_t_lb = Constraint(
            expr=m.fs.heater_2.control_volume.properties_out[0.0].temperature
            >= 480)
        m.fs.heater_2.outlet_t_ub = Constraint(
            expr=m.fs.heater_2.control_volume.properties_out[0.0].temperature
            <= 490)
        # m.fs.heater_2.heat_duty.setlb(0)
    
    # Optimize flash properties
    if 'flash_0' in list_unit:
        # option 3
        m.fs.flash_0.deltaP.unfix()  # allow pressure change in streams

    if 'flash_1' in list_unit:
        # option 3
        # m.fs.flash_1.deltaP.unfix()
        # option 1
        m.fs.flash_1.deltaP.unfix()  # allow pressure change in streams
        m.fs.flash_1.isothermal = Constraint(
            expr=m.fs.flash_1.control_volume.properties_out[0].temperature ==
            m.fs.flash_1.control_volume.properties_in[0].temperature)
        # option 2
        # m.fs.flash_1.outlet_temp.deactivate()
        # m.fs.flash_1.outlet_t_lb = Constraint(
        #     expr=m.fs.flash_1.control_volume.properties_out[0.0].temperature
        #     >= 400)
        # m.fs.flash_1.outlet_t_ub = Constraint(
        #     expr=m.fs.flash_1.control_volume.properties_out[0.0].temperature
        #     <= 410)

    if 'flash_2' in list_unit:
        # option 3
        # m.fs.flash_2.deltaP.unfix()
        # option 1
        m.fs.flash_2.deltaP.unfix()  # allow pressure change in streams
        m.fs.flash_2.isothermal = Constraint(
            expr=m.fs.flash_2.control_volume.properties_out[0].temperature ==
            m.fs.flash_2.control_volume.properties_in[0].temperature)
        # option 2
        # m.fs.flash_2.outlet_temp.deactivate()
        # m.fs.flash_2.outlet_t_lb = Constraint(
        #     expr=m.fs.flash_2.control_volume.properties_out[0.0].temperature
        #     >= 400)
        # m.fs.flash_2.outlet_t_ub = Constraint(
        #     expr=m.fs.flash_2.control_volume.properties_out[0.0].temperature
        #     <= 410)

    # Optimize splitter properties
    if 'splitter_1' in list_unit:
        m.fs.splitter_1.split_fraction[0, "outlet_1"].unfix()
        m.fs.splitter_1.split_fraction_lb = \
            Constraint(expr=m.fs.splitter_1.split_fraction[0, "outlet_1"] >= 0.10)
        m.fs.splitter_1.split_fraction_ub = \
            Constraint(expr=m.fs.splitter_1.split_fraction[0, "outlet_1"] <= 0.60)

    if 'splitter_2' in list_unit:
        m.fs.splitter_2.split_fraction[0, "outlet_1"].unfix()
        m.fs.splitter_2.split_fraction_lb = \
            Constraint(expr=m.fs.splitter_2.split_fraction[0, "outlet_1"] >= 0.10)
        m.fs.splitter_2.split_fraction_ub = \
            Constraint(expr=m.fs.splitter_2.split_fraction[0, "outlet_1"] <= 0.60)

    if 'flash_0' in list_unit:
        m.fs.system_efficiency = Constraint(expr=m.fs.efficiency >= 0.4)

    print("\nDegrees of Freedom (add bounds) = %d" % degrees_of_freedom(m))

def examples(i):
    
    if i == 1:
        flowsheet_name = 'methanol_base_'+str(i)
        # straight line - base example
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'reactor_1', 'turbine_1', \
            'cooler_1', 'flash_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'cooler_1.inlet', 'flash_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'cooler_1.outlet']

    if i == 2:
        flowsheet_name = 'methanol_base_'+str(i)
        # remove cooler_1
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'reactor_1', 'turbine_1', 'flash_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'flash_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet']
    
    if i == 2.5:
        flowsheet_name = 'methanol_base_'+str(i)
        # remove cooler_1, turbine_1
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'reactor_1', 'flash_0']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'flash_0.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet']

    if i == 2.75:
        flowsheet_name = 'methanol_base_'+str(i)
        # remove cooler_1, turbine_1, compressor_1
        list_unit = ['mixer_0', 'heater_1', 'reactor_1', 'flash_0']
        list_inlet = ['heater_1.inlet', 'reactor_1.inlet', 'flash_0.inlet']
        list_outlet = ['mixer_0.outlet', 'heater_1.outlet', 'reactor_1.outlet']

    if i == 3:
        # replace cooler_1 by flash_1
        flowsheet_name = 'methanol_base_'+str(i)
        list_unit = ['mixer_0', 'compressor_1', 'heater_1', 'reactor_1', 'turbine_1', \
            'flash_1', 'flash_0', 'exhaust']
        list_inlet = ['compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'flash_1.inlet', 'flash_0.inlet', 'exhaust.inlet']
        list_outlet = ['mixer_0.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet']
    
    if i == 4:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_1 by splitter_1, and recycle splitter_1.outlet_2
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'reactor_1', 'turbine_1', \
            'flash_0', 'splitter_1']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'splitter_1.inlet', 'flash_0.inlet', 'compressor_2.inlet', 'mixer_1.inlet_2']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2', 'compressor_2.outlet']
    
    if i == 5:
        # recycle flash_1.vap_outlet
        flowsheet_name = 'methanol_base_'+str(i)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'reactor_1', 'turbine_1', \
            'flash_1', 'flash_0', 'exhaust', 'splitter_1']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'flash_1.inlet', 'flash_0.inlet', 'splitter_1.inlet', 'exhaust.inlet', \
                'mixer_1.inlet_2']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet', 'splitter_1.outlet_1', \
                'splitter_1.outlet_2']
    
    if i == 6:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_0.vap_outlet with splitter_1 (no cooler)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'reactor_1', 'turbine_1', \
            'flash_0', 'splitter_1', 'exhaust']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'flash_0.inlet', 'mixer_1.inlet_2', \
                'splitter_1.inlet', 'exhaust.inlet', 'compressor_2.inlet']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'compressor_2.outlet', \
                'flash_0.vap_outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2']

    if i == 7:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_0.vap_outlet with splitter_1 (with cooler)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'cooler_1', 'reactor_1', 'turbine_1', \
            'flash_0', 'splitter_1', 'exhaust']
        list_inlet = ['mixer_1.inlet_1', 'compressor_1.inlet', 'heater_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'cooler_1.inlet', 'flash_0.inlet', \
                'splitter_1.inlet', 'exhaust.inlet', 'mixer_1.inlet_2']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'cooler_1.outlet', \
                'flash_0.vap_outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2']
        
    if i == 8:
        flowsheet_name = 'methanol_base_'+str(i)
        # recycle flash_0.vap_outlet with splitter_1 (with cooler, switch compressor_1 and heater_1)
        list_unit = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'cooler_1', 'reactor_1', 'turbine_1', \
            'flash_0', 'splitter_1', 'exhaust']
        list_inlet = ['mixer_1.inlet_1', 'heater_1.inlet', 'compressor_1.inlet', 'reactor_1.inlet', \
            'turbine_1.inlet', 'cooler_1.inlet', 'flash_0.inlet', \
                'splitter_1.inlet', 'exhaust.inlet', 'mixer_1.inlet_2']
        list_outlet = ['mixer_0.outlet', 'mixer_1.outlet', 'heater_1.outlet', 'compressor_1.outlet', \
            'reactor_1.outlet', 'turbine_1.outlet', 'cooler_1.outlet', \
                'flash_0.vap_outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2']
        
    return list_unit, list_inlet, list_outlet, flowsheet_name
        
#%%
def run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet, visualize_flowsheet = False):

    # score: initialized from 500
    start = timer()
    score = 500
    delta_scoreA = 100  #bonus/penalty option 1
    delta_scoreB = 50   #bonus/penalty option 2
    status = ['infeasible', 0.0, 0.0] # store status, time consuming, system efficiency
    costs = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # print the lists of unit, inlet and outlet
    print('\nlist of units, inlets, outlets:')
    print(list_unit)
    print(list_inlet)
    print(list_outlet)
    
    # Define model components and blocks
    m = ConcreteModel(name=flowsheet_name)
    m.fs = FlowsheetBlock(default={"dynamic": False})

    # build and initialize a flowsheet
    try:
        build_model(m, list_unit, list_inlet, list_outlet)  # build flowsheet
        set_inputs(m, list_unit)  # unit and stream specifications
        initialize_flowsheet(m, list_unit, list_inlet, list_outlet, 5)  # rigorous initialization scheme
        score = score + delta_scoreB # bonus for passing the initialization process
    except:
        print('initialization process: aborted or failed')
        return score, status, costs

    # pre solve
    solver = get_solver() # create the solver object
    try:
        # solver.options = {'tol': 1e-6, 'max_iter': 5000, 'halt_on_ampl_error': 'yes'}
        solver.options = {'tol': 1e-6, 'max_iter': 5000}
        results = solver.solve(m, tee=False)
        print('pre solve - physical operational? ', results.solver.termination_condition.value)
    except:
        print('pre solve - aborted or failed')

    # initial solve
    add_costing(m, list_unit)  # re-solve with costing equations
    m.fs.objective = Objective(expr=m.fs.revenue, sense=maximize)
    # m.fs.objective = Objective(expr=m.fs.meth_flow, sense=maximize)
    try:
        # solver.options = {'tol': 1e-6, 'max_iter': 5000, 'halt_on_ampl_error': 'yes'}
        solver.options = {'tol': 1e-6, 'max_iter': 5000}
        results = solver.solve(m, tee=False)
        print('initial solve - physical operational? ', results.solver.termination_condition.value)

        score = score + delta_scoreB # bonus for passing the initial solve
        status = [results.solver.termination_condition.value, 0.0, value(m.fs.efficiency)]
        costs = [value(m.fs.annualized_capital_cost), value(m.fs.operating_cost), 
                 value(m.fs.sales), value(m.fs.raw_mat_cost), value(m.fs.revenue)]
        if results.solver.termination_condition.value == 'optimal':
            report(m, list_unit)
    except:
        print('initial solve - aborted or failed')
    
    # save initial solve results
    status_init = status
    costs_init = costs

    # optimal solve
    add_bounds_v1(m, list_unit)
    try:
        # solver.options = {'tol': 1e-6, 'max_iter': 5000, 'halt_on_ampl_error': 'yes'}
        solver.options = {'tol': 1e-6, 'max_iter': 5000}
        results = solver.solve(m, tee=False)
        print('optimal solve - physical operational? ', results.solver.termination_condition.value)
        
        score = score + delta_scoreA
        status = [results.solver.termination_condition.value, 0.0, value(m.fs.efficiency)]
        costs = [value(m.fs.annualized_capital_cost), value(m.fs.operating_cost), 
                 value(m.fs.sales), value(m.fs.raw_mat_cost), value(m.fs.revenue)]
        if results.solver.termination_condition.value == 'optimal':
            report(m, list_unit)
    except:
        print('optimal solve - aborted or failed')

    # in case of no optimal solution
    if status[0] != 'optimal':
        if status_init[0] == 'optimal':
            status = status_init
            costs = costs_init

    # score: update according to system efficiency
    if status[0] == 'optimal':
        score = 1000 + (status[2]-0.40)/0.20*delta_scoreB
        end = timer()
        status[1] = end-start

    # visualize flowsheet
    if visualize_flowsheet == True:
        m.fs.visualize(flowsheet_name)
    
    # m.fs.flash_1.report()
    # m.fs.flash_0.report()
    
    return score, status, costs

#%%
if __name__ == "__main__":

    start = timer()
    
    for i in [2.5, 2.75, 1, 2, 3, 4, 5, 6, 7, 8]: #[2.5, 2.75], [1, 2, 3, 4, 5, 6, 7, 8]
        # 70.3, 70.3
        # 70.3, 70.3, 64.9, 95.5, 95.3, 96.1, 96.1, 96.1
        
        print('\n\n------------------------------------------------------------\n\n')
        print('Evaluate example ', i)
        list_unit, list_inlet, list_outlet, flowsheet_name = examples(i)
        score, status, costs = run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet, visualize_flowsheet = False)

        # test with pre-screening
        from RL_ENV import pre_screening
        Action_taken = True
        pres_score, pass_pre_screening = pre_screening(list_unit, list_inlet, list_outlet, Action_taken)
        print('Pass the Pre-screening: ', pass_pre_screening, ' with score of ', pres_score)
    
    end = timer()
    print('\nTime consuming: ', end-start, ' s')
