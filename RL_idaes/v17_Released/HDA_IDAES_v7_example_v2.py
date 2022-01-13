import sys
import os
import numpy as np
import pyomo.environ as pyo
from pyomo.environ import (Constraint,
                           Var,
                           ConcreteModel,
                           Expression,
                           Objective,
                           SolverFactory,
                           TransformationFactory,
                           value, maximize, Block, units, exp, log)
from pyomo.network import Arc, SequentialDecomposition
from idaes.core import FlowsheetBlock
from idaes.core.util import copy_port_values
from idaes.generic_models.unit_models import (PressureChanger,
                                        Mixer,
                                        Separator as Splitter,
                                        Heater,
                                        StoichiometricReactor,
                                        Flash,
                                        Feed,
                                        Product,
                                        HeatExchanger)
from idaes.generic_models.unit_models.pressure_changer import \
    ThermodynamicAssumption
from idaes.generic_models.unit_models.heat_exchanger import \
    delta_temperature_underwood_callback
from idaes.core.util.model_statistics import degrees_of_freedom
import idaes.logger as idaeslog
from pyomo.util.infeasible import log_infeasible_constraints

sys.path.append(os.path.abspath("./HDA_properties"))
import hda_ideal_VLE as thermo_props
import hda_reaction as reaction_props
import idaes.core.util.unit_costing as costing
from idaes.core.util.unit_costing import initialize as init_costing
from idaes.core.util.misc import get_solver

def build_HDA(m, list_unit, list_inlet, list_outlet):
    
    m.fs.thermo_params = thermo_props.HDAParameterBlock()
    m.fs.reaction_params = reaction_props.HDAReactionParameterBlock(
            default={"property_package": m.fs.thermo_params})
    
    # initialize inlet/outlet units
    if 'inlet_feed' in list_unit: # must exist
        m.fs.inlet_feed = Mixer(default={"property_package": m.fs.thermo_params, 
                                          "inlet_list": ["inlet_1", "inlet_2"]})
    
    if 'outlet_product' in list_unit:  # must exist
        m.fs.outlet_product = Flash(default={"property_package": m.fs.thermo_params,
                               "has_heat_transfer": True,
                               "has_pressure_change": True})
        
    if 'outlet_exhaust' in list_unit:
        m.fs.outlet_exhaust = Product(default={'property_package': m.fs.thermo_params})

    # initialize other units
    if 'mixer2to1_1' in list_unit:
        m.fs.mixer2to1_1 = Mixer(default={"property_package": m.fs.thermo_params, 
                                          "inlet_list": ["inlet_1", "inlet_2"]})
        
    if 'mixer2to1_2' in list_unit:
        m.fs.mixer2to1_2 = Mixer(default={"property_package": m.fs.thermo_params,
                                           "inlet_list": ["inlet_1", "inlet_2"]})
        
    if 'heater_1' in list_unit:
        m.fs.heater_1 = Heater(default={"property_package": m.fs.thermo_params,
                                        "has_pressure_change": False,
                                        "has_phase_equilibrium": True})

    if 'heater_2' in list_unit:
        m.fs.heater_2 = Heater(default={"property_package": m.fs.thermo_params,
                                        "has_pressure_change": False,
                                        "has_phase_equilibrium": True})

    if 'cooler_1' in list_unit:
        m.fs.cooler_1 = Heater(default={"property_package": m.fs.thermo_params,
                                        "has_pressure_change": False,
                                        "has_phase_equilibrium": True})

    if 'cooler_2' in list_unit:
        m.fs.cooler_2 = Heater(default={"property_package": m.fs.thermo_params,
                                        "has_pressure_change": False,
                                        "has_phase_equilibrium": True})
    
    if 'heat_exchanger_1' in list_unit:                
        m.fs.heat_exchanger_1 = HeatExchanger(
            default={"delta_temperature_callback":
                    delta_temperature_underwood_callback,
                    "shell": {"property_package": m.fs.thermo_params},
                    "tube": {"property_package": m.fs.thermo_params}})

    if 'heat_exchanger_2' in list_unit:                
        m.fs.heat_exchanger_2 = HeatExchanger(
            default={"delta_temperature_callback":
                    delta_temperature_underwood_callback,
                    "shell": {"property_package": m.fs.thermo_params},
                    "tube": {"property_package": m.fs.thermo_params}})
        
    if 'StReactor_1' in list_unit:
        m.fs.StReactor_1 = StoichiometricReactor(
                default={"property_package": m.fs.thermo_params,
                         "reaction_package": m.fs.reaction_params,
                         "has_heat_of_reaction": True,
                         "has_heat_transfer": True,
                         "has_pressure_change": False})
    
    if 'StReactor_2' in list_unit:
        m.fs.StReactor_2 = StoichiometricReactor(
                default={"property_package": m.fs.thermo_params,
                         "reaction_package": m.fs.reaction_params,
                         "has_heat_of_reaction": True,
                         "has_heat_transfer": True,
                         "has_pressure_change": False})
        
    if 'flash_1' in list_unit:
        m.fs.flash_1 = Flash(default={
            "property_package": m.fs.thermo_params,
            "has_heat_transfer": True,
            "has_pressure_change": True})
    
    if 'flash_2' in list_unit:
        m.fs.flash_2 = Flash(default={
            "property_package": m.fs.thermo_params,
            "has_heat_transfer": True,
            "has_pressure_change": True})
        
    if 'splitter1to2_1' in list_unit:
        m.fs.splitter1to2_1 = Splitter(default={
            "property_package": m.fs.thermo_params,
             "ideal_separation": False,
             "outlet_list": ["outlet_1", "outlet_2"]})
    
    if 'splitter1to2_2' in list_unit:
        m.fs.splitter1to2_2 = Splitter(default={
            "property_package": m.fs.thermo_params,
            "ideal_separation": False,
             "outlet_list": ["outlet_1", "outlet_2"]})

    if 'compressor_1' in list_unit:
        m.fs.compressor_1 = PressureChanger(default={
            "property_package": m.fs.thermo_params,
            "compressor": True,
            "thermodynamic_assumption": ThermodynamicAssumption.isothermal})
        
    if 'compressor_2' in list_unit:
        m.fs.compressor_2 = PressureChanger(default={
            "property_package": m.fs.thermo_params,
            "compressor": True,
            "thermodynamic_assumption": ThermodynamicAssumption.isothermal})
        
    if 'expander_1' in list_unit:
        m.fs.expander_1 = PressureChanger(default={
            "property_package": m.fs.thermo_params,
            "compressor": False,
            "thermodynamic_assumption": ThermodynamicAssumption.isothermal})
    
    if 'expander_2' in list_unit:
        m.fs.expander_2 = PressureChanger(default={
            "property_package": m.fs.thermo_params,
            "compressor": False,
            "thermodynamic_assumption": ThermodynamicAssumption.isothermal})
    
    # build arcs
    for i in range(len(list_inlet)):
        expression = 'm.fs.Arc'+str(i)+' = Arc(source = m.fs.'+list_outlet[i]+', destination = m.fs.'+list_inlet[i]+')'
        exec(expression)

    # define parameters
    if 'outlet_product' in list_unit:  # must exist
        m.fs.purity = Expression(
            expr=m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] / 
            (m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"]
             + m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "toluene"]))
        m.fs.Benzene_flow = Expression(
            expr=m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"])
    
    TransformationFactory("network.expand_arcs").apply_to(m)
    print("Degrees of Freedom (build) = %d" % degrees_of_freedom(m))
    
def set_HDA_inputs(m, list_unit):
    
    if 'inlet_feed' in list_unit: # must exist
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-5)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-5)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(0.30)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Vap", "methane"].fix(0.02)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-5)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Liq", "toluene"].fix(1e-5)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-5)
        m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-5)
        m.fs.inlet_feed.inlet_1.temperature.fix(303.2)
        m.fs.inlet_feed.inlet_1.pressure.fix(350000)
        
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Vap", "benzene"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Vap", "toluene"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Vap", "hydrogen"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Vap", "methane"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Liq", "benzene"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Liq", "toluene"].fix(0.30)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Liq", "hydrogen"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Liq", "methane"].fix(1e-5)
        m.fs.inlet_feed.inlet_2.temperature.fix(303.2)
        m.fs.inlet_feed.inlet_2.pressure.fix(350000)
    
    if 'heater_1' in list_unit:
        m.fs.heater_1.outlet.temperature.fix(600)

    if 'heater_2' in list_unit:
        m.fs.heater_2.outlet.temperature.fix(600)

    if 'cooler_1' in list_unit:
        m.fs.cooler_1.outlet.temperature.fix(325)

    if 'cooler_2' in list_unit:
        m.fs.cooler_2.outlet.temperature.fix(325)
        
    if 'StReactor_1' in list_unit:
        m.fs.StReactor_1.conversion = Var(initialize=0.75, bounds=(0, 1))
        m.fs.StReactor_1.conv_constraint = Constraint(
            expr=m.fs.StReactor_1.conversion*m.fs.StReactor_1.inlet.
            flow_mol_phase_comp[0, "Vap", "toluene"] ==
            (m.fs.StReactor_1.inlet.flow_mol_phase_comp[0, "Vap", "toluene"] -
             m.fs.StReactor_1.outlet.flow_mol_phase_comp[0, "Vap", "toluene"]))
    
        m.fs.StReactor_1.conversion.fix(0.75)
        m.fs.StReactor_1.heat_duty.fix(0)
    
    if 'StReactor_2' in list_unit:
        m.fs.StReactor_2.conversion = Var(initialize=0.75, bounds=(0, 1))
        m.fs.StReactor_2.conv_constraint = Constraint(
            expr=m.fs.StReactor_2.conversion*m.fs.StReactor_2.inlet.
            flow_mol_phase_comp[0, "Vap", "toluene"] ==
            (m.fs.StReactor_2.inlet.flow_mol_phase_comp[0, "Vap", "toluene"] -
             m.fs.StReactor_2.outlet.flow_mol_phase_comp[0, "Vap", "toluene"]))
    
        m.fs.StReactor_2.conversion.fix(0.75)
        m.fs.StReactor_2.heat_duty.fix(0)
        
    if 'flash_1' in list_unit:
        # m.fs.flash_1.control_volume.properties_out[0].temperature.fix(325)
        m.fs.flash_1.vap_outlet.temperature.fix(325.0)
        m.fs.flash_1.deltaP.fix(0)
        
    if 'flash_2' in list_unit:
        # m.fs.flash_2.control_volume.properties_out[0].temperature.fix(325)
        m.fs.flash_2.vap_outlet.temperature.fix(325.0)
        m.fs.flash_2.deltaP.fix(0)
    
    if 'outlet_product' in list_unit: # must exist
        # m.fs.outlet_product.control_volume.properties_out[0].temperature.fix(375)
        # m.fs.outlet_product.deltaP.fix(-200000)
        m.fs.outlet_product.vap_outlet.temperature.fix(362.0)
        m.fs.outlet_product.vap_outlet.pressure[0].fix(105000)
        
    if 'splitter1to2_1' in list_unit:
        m.fs.splitter1to2_1.split_fraction[0, "outlet_2"].fix(0.2)
        
    if 'splitter1to2_2' in list_unit:
        m.fs.splitter1to2_2.split_fraction[0, "outlet_2"].fix(0.2)
        
    if 'compressor_1' in list_unit:
        m.fs.compressor_1.outlet.pressure.fix(350000)
        
    if 'compressor_2' in list_unit:
        m.fs.compressor_2.outlet.pressure.fix(350000)
        
    if 'expander_1' in list_unit:
        m.fs.expander_1.outlet.pressure.fix(150000)
        
    if 'expander_2' in list_unit:
        m.fs.expander_2.outlet.pressure.fix(150000)
    
    print("Degrees of Freedom (set inputs) = %d" % degrees_of_freedom(m))
    
def initialize_HDA(m, list_unit, iterlim):
    
    seq = SequentialDecomposition()
    seq.options.select_tear_method = "heuristic"
    seq.options.tear_method = "Wegstein"
    seq.options.iterLim = iterlim

    def function(unit):
        unit.initialize(outlvl=idaeslog.INFO_LOW)
        
    seq.run(m, function)

    print("Degrees of Freedom (after initialize) = %d" % degrees_of_freedom(m))

def add_costing(m, list_unit):

    # operating cost: cooling + heating
    expression = 'm.fs.cooling_cost = Expression(expr=(0.0'
    if 'cooler_1' in list_unit:
        expression += ' -m.fs.cooler_1.heat_duty[0]' 
    if 'cooler_2' in list_unit:
        expression += ' -m.fs.cooler_2.heat_duty[0]' 
    if 'flash_1' in list_unit:
        expression += ' -m.fs.flash_1.heat_duty[0]' 
    if 'flash_2' in list_unit:
        expression += ' -m.fs.flash_2.heat_duty[0]' 
    expression += ') * 0.25e-7)'
    exec(expression)

    expression = 'm.fs.heating_cost = Expression(expr=(0.0'
    if 'heater_1' in list_unit:
        expression += ' + m.fs.heater_1.heat_duty[0] * 2.2e-7' 
    if 'heater_2' in list_unit:
        expression += ' + m.fs.heater_2.heat_duty[0] * 2.2e-7' 
    if 'outlet_product' in list_unit:
        expression += ' + m.fs.outlet_product.heat_duty[0] * 1.9e-7' 
    expression += '))'
    exec(expression)

    m.fs.operating_cost = Expression(
        expr=(3600 * 24 * 365 * (m.fs.heating_cost + m.fs.cooling_cost)))

    # annualized capital cost
    expression = 'm.fs.annualized_capital_cost = Expression(expr=(0.0'
    # if 'StReactor_1' in list_unit:
    #     m.fs.StReactor_1.get_costing()
    #     m.fs.StReactor_1.diameter.fix(2)
    #     init_costing(m.fs.StReactor_1.costing)
    #     costing.calculate_scaling_factors(m.fs.StReactor_1.costing)

    #     expression += ' + m.fs.StReactor_1.costing.purchase_cost'
    # if 'StReactor_2' in list_unit:
    #     m.fs.StReactor_2.get_costing()
    #     m.fs.StReactor_2.diameter.fix(2)
    #     init_costing(m.fs.StReactor_2.costing)
    #     costing.calculate_scaling_factors(m.fs.StReactor_2.costing)

    #     expression += ' + m.fs.StReactor_2.costing.purchase_cost'
    # if 'flash_1' in list_unit:
    #     m.fs.flash_1.get_costing()
    #     m.fs.flash_1.diameter.fix(2)
    #     m.fs.flash_1.length.fix(4)
    #     init_costing(m.fs.flash_1.costing)
    #     costing.calculate_scaling_factors(m.fs.flash_1.costing)

    #     expression += ' + m.fs.flash_1.costing.purchase_cost'
    # if 'flash_2' in list_unit:
    #     m.fs.flash_2.get_costing()
    #     m.fs.flash_2.diameter.fix(2)
    #     m.fs.flash_2.length.fix(4)
    #     init_costing(m.fs.flash_2.costing)
    #     costing.calculate_scaling_factors(m.fs.flash_2.costing)

    #     expression += ' + m.fs.flash_2.costing.purchase_cost'
    # if 'outlet_product' in list_unit:
    #     m.fs.outlet_product.get_costing()
    #     m.fs.outlet_product.diameter.fix(2)
    #     m.fs.outlet_product.length.fix(4)
    #     init_costing(m.fs.outlet_product.costing)
    #     costing.calculate_scaling_factors(m.fs.outlet_product.costing)

    #     expression += ' + m.fs.outlet_product.costing.purchase_cost'
    if 'heater_1' in list_unit: # fine when optimizing Benzene_flow
        m.fs.heater_1.cost_heater = Expression(expr= \
        exp(0.32325 + 0.766*log(m.fs.heater_1.heat_duty[0] * 3.4121416331))\
        *1.4*(671/500))  # 1 J/s = 3.4121 BTU/hr

        expression += ' + m.fs.heater_1.cost_heater'
    if 'heater_2' in list_unit: # fine when optimizing Benzene_flow
        m.fs.heater_2.cost_heater = Expression(expr= \
        exp(0.32325 + 0.766*log(m.fs.heater_2.heat_duty[0] * 3.4121416331))\
        *1.4*(671/500))  # 1 J/s = 3.4121 BTU/hr

        expression += ' + m.fs.heater_2.cost_heater'
    expression += ') * 5.4 / 15)'
    exec(expression)

    # sales and raw material cost
    # benzene price $498 us dollars per metric ton  - 78.11 g/mol  - 1 gr = 1e-6 MT  -- consider 1000
    # toluene price USD 637 per MT - 92.14 g/mol  -- consider price 200
    # H2 $16.51 per kilogram - 2.016 g/mol
    expression = 'm.fs.sales = Expression(expr=(0.0'
    if 'outlet_product' in list_unit:
        expression += ' + m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] * 78.11 * 1e-6 * 2500'
    expression += ') * 3600 * 24 * 365)'
    exec(expression)

    expression = 'm.fs.raw_material_cost = Expression(expr=(0.0'
    if 'inlet_feed' in list_unit:
        expression += ' + m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Vap", "hydrogen"] * 2.016 / 1000 * 16.51'
        expression += ' + m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Liq", "toluene"]  * 92.14 * 1e-6 * 200'
    expression += ') * 3600 * 24 * 365)'
    exec(expression)
    
    m.fs.revenue = Expression(expr=(m.fs.sales - m.fs.operating_cost - m.fs.annualized_capital_cost - m.fs.raw_material_cost)/1000)

    print("Degrees of Freedom (add costing) = %d" % degrees_of_freedom(m))

def add_HDA_var_bounds(m, list_unit):
    
    if 'heater_1' in list_unit:
        m.fs.heater_1.outlet.temperature.unfix()
        m.fs.heater_1.outlet.temperature[0].setlb(500)
        m.fs.heater_1.outlet.temperature[0].setub(600)
        # m.fs.heater_1.heat_duty.setlb(0) # set lower/upper bound may cause solving process failing

    if 'heater_2' in list_unit:
        m.fs.heater_2.outlet.temperature.unfix()
        m.fs.heater_2.outlet.temperature[0].setlb(500)
        m.fs.heater_2.outlet.temperature[0].setub(600)
        # m.fs.heater_2.heat_duty.setlb(0) # set lower/upper bound may cause solving process failing

    # if 'cooler_1' in list_unit:
    #     m.fs.cooler_1.outlet.temperature.unfix()
    #     m.fs.cooler_1.outlet.temperature[0].setlb(300)
    #     m.fs.cooler_1.outlet.temperature[0].setub(350)
	#     m.fs.cooler_1.heat_duty.setub(0) # set lower/upper bound may cause solving process failing

    # if 'cooler_2' in list_unit:
    #     m.fs.cooler_2.outlet.temperature.unfix()
    #     m.fs.cooler_2.outlet.temperature[0].setlb(300)
    #     m.fs.cooler_2.outlet.temperature[0].setub(350)
	#     m.fs.cooler_2.heat_duty.setub(0) # set lower/upper bound may cause solving process failing
    
    # if 'compressor_1' in list_unit:
    #     m.fs.compressor_1.outlet.pressure.unfix()
    #     m.fs.compressor_1.outlet.pressure[0].setlb(345000)
    #     m.fs.compressor_1.outlet.pressure[0].setub(355000)
	#     m.fs.compressor_1.deltaP.setlb(0) # set lower/upper bound may cause solving process failing
        
    # if 'compressor_2' in list_unit:
    #     m.fs.compressor_2.outlet.pressure.unfix()
    #     m.fs.compressor_2.outlet.pressure[0].setlb(345000)
    #     m.fs.compressor_2.outlet.pressure[0].setub(355000)
	#     m.fs.compressor_2.deltaP.setlb(0) # set lower/upper bound may cause solving process failing

    # if 'StReactor_1' in list_unit:
    #     m.fs.StReactor_1.heat_duty.unfix()
    #     m.fs.StReactor_1.outlet.temperature[0].setlb(600)
    #     m.fs.StReactor_1.outlet.temperature[0].setub(800)
        
    # if 'StReactor_2' in list_unit:
    #     m.fs.StReactor_2.heat_duty.unfix()
    #     m.fs.StReactor_2.outlet.temperature[0].setlb(600)
    #     m.fs.StReactor_2.outlet.temperature[0].setub(800)
        
    if 'flash_1' in list_unit:
        # m.fs.flash_1.control_volume.properties_out[0].temperature.unfix()
        # m.fs.flash_1.control_volume.properties_out[0].temperature.setlb(298.0)
        # m.fs.flash_1.control_volume.properties_out[0].temperature.setub(450.0)
        m.fs.flash_1.vap_outlet.temperature.unfix()
        # m.fs.flash_1.deltaP.unfix()
        m.fs.flash_1.vap_outlet.temperature[0].setlb(298.0)
        m.fs.flash_1.vap_outlet.temperature[0].setub(450.0)
        # m.fs.flash_1.vap_outlet.pressure[0].setlb(105000)
        # m.fs.flash_1.vap_outlet.pressure[0].setub(3.5e5)

    if 'flash_2' in list_unit:
        # m.fs.flash_2.control_volume.properties_out[0].temperature.unfix()
        # m.fs.flash_2.control_volume.properties_out[0].temperature.setlb(298.0)
        # m.fs.flash_2.control_volume.properties_out[0].temperature.setub(450.0)
        m.fs.flash_2.vap_outlet.temperature.unfix()
        # m.fs.flash_2.deltaP.unfix()
        m.fs.flash_2.vap_outlet.temperature[0].setlb(298.0)
        m.fs.flash_2.vap_outlet.temperature[0].setub(450.0)
        # m.fs.flash_2.vap_outlet.pressure[0].setlb(105000)
        # m.fs.flash_2.vap_outlet.pressure[0].setub(3.5e5)
    
    if 'outlet_product' in list_unit: # must exist
        # m.fs.outlet_product.control_volume.properties_out[0].temperature.unfix()
        # m.fs.outlet_product.control_volume.properties_out[0].temperature.setlb(298.0)
        # m.fs.outlet_product.control_volume.properties_out[0].temperature.setub(450.0)
        # m.fs.outlet_product.deltaP.unfix()
        # m.fs.outlet_product.vap_outlet.pressure[0].setlb(100000)
        # m.fs.outlet_product.vap_outlet.pressure[0].setub(150000)
        # m.fs.outlet_product.heat_duty.setlb(-50000)
        # m.fs.outlet_product.heat_duty.setub(50000)
        
        m.fs.outlet_product.vap_outlet.pressure.unfix()
        m.fs.outlet_product.vap_outlet.pressure[0].setlb(105000)
        m.fs.outlet_product.vap_outlet.pressure[0].setub(110000)
        m.fs.outlet_product.vap_outlet.temperature.unfix()
        m.fs.outlet_product.heat_duty.setlb(-50000)
        m.fs.outlet_product.heat_duty.setub(50000)
        # m.fs.outlet_product.vap_outlet.temperature[0].setlb(298.0)
        # m.fs.outlet_product.vap_outlet.temperature[0].setub(450.0)

        # enable these constraints actually requring more iterations to get optimal solution
        m.fs.product_flow = Constraint(expr=m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] >= 0.10)
        m.fs.product_purity = Constraint(expr=
                                         m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"] / 
                                         (m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"]
                                          + m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "toluene"]) >= 0.55) 
        
    print("Degrees of Freedom (add bounds) = %d" % degrees_of_freedom(m))

def calculate_costing(m, list_unit):

    # operating cost: cooling + heating
    cooling_cost = 0.0
    if 'cooler_1' in list_unit:
        cooling_cost += -value(m.fs.cooler_1.heat_duty[0]) * 0.25e-7
    if 'cooler_2' in list_unit:
        cooling_cost += -value(m.fs.cooler_2.heat_duty[0]) * 0.25e-7
    if 'flash_1' in list_unit:
        cooling_cost += -value(m.fs.flash_1.heat_duty[0]) * 0.25e-7
    if 'flash_2' in list_unit:
        cooling_cost += -value(m.fs.flash_2.heat_duty[0]) * 0.25e-7

    heating_cost = 0.0
    if 'heater_1' in list_unit:
        heating_cost += value(m.fs.heater_1.heat_duty[0]) * 2.2e-7
    if 'heater_2' in list_unit:
        heating_cost += value(m.fs.heater_2.heat_duty[0]) * 2.2e-7
    if 'outlet_product' in list_unit:
        heating_cost += value(m.fs.outlet_product.heat_duty[0]) * 1.9e-7

    operating_cost = 3600 * 24 * 365 * (heating_cost + cooling_cost)

    # annualized capital cost
    annualized_capital_cost = 0.0
    if 'StReactor_1' in list_unit:
        m.fs.StReactor_1.get_costing()
        m.fs.StReactor_1.diameter.fix(2)
        init_costing(m.fs.StReactor_1.costing)
        annualized_capital_cost += value(m.fs.StReactor_1.costing.purchase_cost)
    if 'StReactor_2' in list_unit:
        m.fs.StReactor_2.get_costing()
        m.fs.StReactor_2.diameter.fix(2)
        init_costing(m.fs.StReactor_2.costing)
        annualized_capital_cost += value(m.fs.StReactor_2.costing.purchase_cost)
    if 'flash_1' in list_unit:
        m.fs.flash_1.get_costing()
        m.fs.flash_1.diameter.fix(2)
        m.fs.flash_1.length.fix(4)
        init_costing(m.fs.flash_1.costing)
        annualized_capital_cost += value(m.fs.flash_1.costing.purchase_cost)
    if 'flash_2' in list_unit:
        m.fs.flash_2.get_costing()
        m.fs.flash_2.diameter.fix(2)
        m.fs.flash_2.length.fix(4)
        init_costing(m.fs.flash_2.costing)
        annualized_capital_cost += value(m.fs.flash_2.costing.purchase_cost)
    if 'outlet_product' in list_unit:
        m.fs.outlet_product.get_costing()
        m.fs.outlet_product.diameter.fix(2)
        m.fs.outlet_product.length.fix(4)
        init_costing(m.fs.outlet_product.costing)
        annualized_capital_cost += value(m.fs.outlet_product.costing.purchase_cost)
    if 'heater_1' in list_unit: # fine when optimizing Benzene_flow
        if value(m.fs.heater_1.heat_duty[0])>0:
            heater_1_cost_heater = exp(0.32325 + 0.766*log(value(m.fs.heater_1.heat_duty[0]) * 3.4121416331))\
            *1.4*(671/500)  # 1 J/s = 3.4121 BTU/hr
            annualized_capital_cost += heater_1_cost_heater
    if 'heater_2' in list_unit: # fine when optimizing Benzene_flow
        if value(m.fs.heater_2.heat_duty[0])>0:
            heater_2_cost_heater = exp(0.32325 + 0.766*log(value(m.fs.heater_2.heat_duty[0]) * 3.4121416331))\
            *1.4*(671/500)  # 1 J/s = 3.4121 BTU/hr
            annualized_capital_cost += heater_2_cost_heater
    annualized_capital_cost = annualized_capital_cost * 5.4 / 15

    # sales and raw material cost
    # benzene price $498 us dollars per metric ton  - 78.11 g/mol  - 1 gr = 1e-6 MT  -- consider 1000
    # toluene price USD 637 per MT - 92.14 g/mol  -- consider price 200
    # H2 $16.51 per kilogram - 2.016 g/mol
    sales = 0.0
    if 'outlet_product' in list_unit:
        sales += value(m.fs.outlet_product.vap_outlet.flow_mol_phase_comp[0, "Vap", "benzene"]) * 78.11 * 1e-6 * 2500
    sales = sales * 3600 * 24 * 365

    raw_material_cost = 0.0
    if 'inlet_feed' in list_unit:
        raw_material_cost += value(m.fs.inlet_feed.inlet_1.flow_mol_phase_comp[0, "Vap", "hydrogen"]) * 2.016 / 1000 * 16.51
        raw_material_cost += value(m.fs.inlet_feed.inlet_2.flow_mol_phase_comp[0, "Liq", "toluene"])  * 92.14 * 1e-6 * 200
    raw_material_cost = raw_material_cost * 3600 * 24 * 365
    
    revenue = (sales - operating_cost - annualized_capital_cost - raw_material_cost)/1000 # unit: k

    # report calculations
    print('annualized capital cost $/year =', annualized_capital_cost)
    print('operating cost $/year = ', operating_cost)
    print('sales in $/year = ', sales)
    print('raw materials in $/year =', raw_material_cost)
    print('revenue in 1000$/year= ', revenue)
	
    return annualized_capital_cost, operating_cost, sales, raw_material_cost, revenue

#%%
if __name__ == "__main__":
    
    # # Benzene flow as objective: with optimal solution
    #     # benzene purity =  0.9351902300925313
    #     # benzene flow rate = 0.2697333235506165
    #     # physical operational?  optimal
    #     # annualized capital cost $/year = 98293.22227353569
    #     # operating cost $/year =  910429.4911139094
    #     # sales in $/year =  1661069.7031161473
    #     # raw materials in $/year = 489238.403328
    #     # revenue in 1000$/year=  163.10858640070225
    # flowsheet_name = 'HDA_option_1' # RL found optimized flowsheet
    # list_unit = ['outlet_product', 'outlet_exhaust', 'mixer2to1_1', 'heater_1', 'StReactor_1', 'flash_1', 'splitter1to2_1', 'compressor_1', 'inlet_feed']
    # list_inlet = ['outlet_product.inlet', 'outlet_exhaust.inlet', 'mixer2to1_1.inlet_1', 'mixer2to1_1.inlet_2', 'heater_1.inlet', 'StReactor_1.inlet', 'flash_1.inlet', 'splitter1to2_1.inlet', 'compressor_1.inlet']
    # list_outlet = ['splitter1to2_1.outlet_2', 'flash_1.vap_outlet', 'splitter1to2_1.outlet_1', 'compressor_1.outlet', 'mixer2to1_1.outlet', 'heater_1.outlet', 'StReactor_1.outlet', 'flash_1.liq_outlet', 'inlet_feed.outlet']
    
    # # Benzene flow as objective: with optimal solution
    #     # benzene purity =  0.7446476221808527
    #     # benzene flow rate = 0.21601882138831585
    #     # physical operational?  optimal
    #     # annualized capital cost $/year = 87653.40321542676
    #     # operating cost $/year =  275920.2424455595
    #     # sales in $/year =  1330285.4641304838
    #     # raw materials in $/year = 489238.403328
    #     # revenue in 1000$/year=  477.47341514149764
    # flowsheet_name = 'HDA_option_2' # RL found optimized flowsheet (normal) (final converged one)
    # list_unit = ['outlet_product', 'outlet_exhaust', 'heater_1', 'StReactor_1', 'flash_1', 'compressor_1', 'inlet_feed']
    # list_inlet = ['outlet_product.inlet', 'outlet_exhaust.inlet', 'heater_1.inlet', 'StReactor_1.inlet', 'flash_1.inlet', 'compressor_1.inlet']
    # list_outlet = ['flash_1.liq_outlet', 'flash_1.vap_outlet', 'compressor_1.outlet', 'heater_1.outlet', 'StReactor_1.outlet', 'inlet_feed.outlet']
    
    # # Benzene flow as objective: with optimal solution
    #     # benzene purity =  0.7537113990359012
    #     # benzene flow rate = 0.21857297296783237
    #     # physical operational?  optimal
    #     # annualized capital cost $/year = 87894.57878574663
    #     # operating cost $/year =  286776.40162778954
    #     # sales in $/year =  1346014.420975911
    #     # raw materials in $/year = 489238.403328
    #     # revenue in 1000$/year=  482.1050372343749
    # flowsheet_name = 'HDA_option_3'
    # list_unit =  ['outlet_product', 'outlet_exhaust', 'mixer2to1_1', 'heater_1', 'StReactor_1', 'flash_1', 'splitter1to2_1', 'compressor_1', 'inlet_feed']
    # list_inlet =  ['outlet_product.inlet', 'outlet_exhaust.inlet', 'mixer2to1_1.inlet_1', 'mixer2to1_1.inlet_2', 'heater_1.inlet', 'StReactor_1.inlet', 'flash_1.inlet', 'splitter1to2_1.inlet', 'compressor_1.inlet']
    # list_outlet =  ['flash_1.liq_outlet', 'splitter1to2_1.outlet_2', 'splitter1to2_1.outlet_1', 'heater_1.outlet', 'inlet_feed.outlet', 'mixer2to1_1.outlet', 'StReactor_1.outlet', 'compressor_1.outlet', 'flash_1.vap_outlet']

#------------------------------------------------------------------------------

    # # Benzene flow as objective: with optimal solution
    #     # benzene purity =  0.7536934699535854
    #     # benzene flow rate = 0.2185937614129049
    #     # physical operational?  optimal
    #     # annualized capital cost $/year = 90038.31451468503
    #     # operating cost $/year =  442820.4740714909
    #     # sales in $/year =  1346142.4402203641
    #     # raw materials in $/year = 489238.403328
    #     # revenue in 1000$/year=  324.04524830618817
    # flowsheet_name = 'HDA_option_base_1'
    # list_unit = ['inlet_feed', 'mixer2to1_1', 'heater_1','StReactor_1', 'flash_1', 'splitter1to2_1', 'compressor_1', 'outlet_product', 'outlet_exhaust']
    # list_inlet = ['mixer2to1_1.inlet_1', 'mixer2to1_1.inlet_2', 'heater_1.inlet', 'StReactor_1.inlet', 'flash_1.inlet', 'outlet_product.inlet', 'splitter1to2_1.inlet', 'compressor_1.inlet', 'outlet_exhaust.inlet']
    # list_outlet = ['inlet_feed.outlet', 'compressor_1.outlet', 'mixer2to1_1.outlet', 'heater_1.outlet', 'StReactor_1.outlet','flash_1.liq_outlet', 'flash_1.vap_outlet', 'splitter1to2_1.outlet_1', 'splitter1to2_1.outlet_2']
    
    # # Benzene flow as objective: with optimal solution
    #     # benzene purity =  0.7446419305055189
    #     # benzene flow rate = 0.21602839336463295
    #     # physical operational?  optimal
    #     # annualized capital cost $/year = 87653.40702138652
    #     # operating cost $/year =  329885.8677320924
    #     # sales in $/year =  1330344.410202293
    #     # raw materials in $/year = 489238.403328
    #     # revenue in 1000$/year=  423.56673212081404
    # flowsheet_name = 'HDA_option_base_2'
    # list_unit = ['inlet_feed', 'outlet_product', 'outlet_exhaust', 'heater_1', 'StReactor_1', 'flash_1']
    # list_inlet = ['heater_1.inlet', 'StReactor_1.inlet', 'flash_1.inlet', 'outlet_product.inlet', 'outlet_exhaust.inlet']
    # list_outlet = ['inlet_feed.outlet', 'heater_1.outlet', 'StReactor_1.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet']

    # Benzene flow as objective: with optimal solution
        # benzene purity =  0.7500309907981324
        # benzene flow rate = 0.22506122412045643
        # physical operational?  optimal
        # annualized capital cost $/year = 56340.70721104864
        # operating cost $/year =  75799.38688266023
        # sales in $/year =  1385970.3199132914
        # raw materials in $/year = 489238.403328
        # revenue in 1000$/year=  764.5918224915825
    flowsheet_name = 'HDA_option_base_3'
    list_unit = ['inlet_feed', 'heater_1', 'StReactor_1', 'outlet_product']
    list_inlet = ['heater_1.inlet', 'StReactor_1.inlet', 'outlet_product.inlet']
    list_outlet = ['inlet_feed.outlet', 'heater_1.outlet', 'StReactor_1.outlet']

    # build flowsheet and run simulation
    m = ConcreteModel(name=flowsheet_name)
    m.fs = FlowsheetBlock(default={"dynamic": False})
    visualize_flowsheet = True

    # if the inputs cannot build a flowsheet
    build_HDA(m, list_unit, list_inlet, list_outlet)
    set_HDA_inputs(m, list_unit)
    initialize_HDA(m, list_unit, 10)
    # add_costing(m, list_unit)
    add_HDA_var_bounds(m, list_unit)

    # Create the solver object
    solver = SolverFactory('ipopt')
    m.fs.objective = Objective(expr=m.fs.Benzene_flow, sense=pyo.maximize)

    # Solve the model
    try:
        solver.options = {'tol': 1e-6, 'max_iter': 500, 'bound_push': 1e-6}
        results = solver.solve(m, tee=False)
        
        status = results.solver.termination_condition
        purity = value(m.fs.purity)
        flow = value(m.fs.Benzene_flow)
    except:
        print('error with the first try')

    # increase the max_iter if necessary
    if status == 'maxIterations':
        
        print('try 1000 iterations')
        try:
            solver.options = {'tol': 1e-6, 'max_iter': 1000, 'bound_push': 1e-6}
            results = solver.solve(m, tee=False)
            
            status = results.solver.termination_condition
            purity = value(m.fs.purity)
            flow = value(m.fs.Benzene_flow)

            if status == 'maxIterations':
                print('try 5000 iterations')
                solver.options = {'tol': 1e-6, 'max_iter': 5000, 'bound_push': 1e-6}
                results = solver.solve(m, tee=False)
                
                status = results.solver.termination_condition
                purity = value(m.fs.purity)
                flow = value(m.fs.Benzene_flow)
        except:
            print('error when trying more iterations')
    
    # log_infeasible_constraints(m)
    
    # print optimization results
    print('benzene purity = ', value(m.fs.purity))
    print('benzene flow rate =', value(m.fs.Benzene_flow))
    print('physical operational? ', results.solver.termination_condition)

    annualized_capital_cost, operating_cost, sales, raw_material_cost, revenue = calculate_costing(m, list_unit)
    # print('annualized capital cost $/year =', value(m.fs.annualized_capital_cost))
    # print('operating cost $/year = ', value(m.fs.operating_cost))
    # print('sales in $/year = ', value(m.fs.sales))
    # print('raw materials in $/year =', value(m.fs.raw_material_cost))
    # print('revenue in 1000$/year= ', value(m.fs.revenue))

    # visualize HDA flowsheet
    if visualize_flowsheet == False:
        m.fs.visualize(flowsheet_name)