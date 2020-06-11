from pyomo.environ import TransformationFactory
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import Mixer, Separator, GibbsReactor, \
     Heater
import idaes.generic_models.properties.activity_coeff_models.\
       methane_combustion_ideal as thermo_props
from idaes.generic_models.unit_models.separator import SplittingType
from idaes.core.util.model_statistics import degrees_of_freedom as dof


def feed_exhaust(Uf, Ua, MS, n_CH4f, n_H2ex):
    """
    Define the settings of the model.
    The reactions are as below:
    Reforming: CH4 + H2O -> CO + 3H2
    Water gas shift: CO + H2O -> CO2 + H2
    Methane combustion: CH4 + 2O2 -> CO2 + 2H2O
    Hydrogen combustion: H2 + 1/2O2 -> H2O
    Carbon monoxide combustion: CO + 1/2O2 -> CO2
    Optional keyword arguments:
    ---------------------------
    :param Uf: Fuel ultilization, mole reductant consumed in FC per mole
               of reductant total
    :param Ua: Air ultilization, mole of air consumed in FC per mole
               of air feed
    :param MS: Methane to steam ratio, mole methane per mole water
    :param n_CH4f: mole of methane feed
    :param n_H2ex: mole of hydrogen exhaust
    :return: return the mole of methane, steam, air feed, and fuel
             ultilization
    """
    Uf = Uf  # Fuel ultilization
    Ua = Ua  # Air ultilization
    MS = MS  # Methane to steam ratio (MS): mole methane per mole water
    n_CH4f = n_CH4f
    n_H2Of = n_CH4f * MS
    n_O2f = n_CH4f * Uf * 2 / Ua
    n_N2f = n_O2f * 0.79 / 0.21
    print("mole of methane feed: " + str(n_CH4f) + " mole/s")
    print("mole of steam feed: " + str(n_H2Of) + " mole/s")
    print("mole of air feed: " + str(n_N2f + n_O2f) + " mole/s")

    n_H2ex = n_H2ex
    n_COex = n_CH4f * (1 - Uf) * 4 - n_H2ex
    n_CO2ex = n_CH4f - n_COex
    n_H2Oex = n_H2Of + 2 * n_CH4f - n_H2ex
    y_H2ex = n_H2ex / (n_H2ex + n_COex + n_CO2ex + n_H2Oex)
    y_COex = n_COex / (n_H2ex + n_COex + n_CO2ex + n_H2Oex)
    y_CO2ex = n_CO2ex / (n_H2ex + n_COex + n_CO2ex + n_H2Oex)
    y_H2Oex = n_H2Oex / (n_H2ex + n_COex + n_CO2ex + n_H2Oex)

    print("Anode exhaust: ")
    print("y_H2ex: " + str(y_H2ex))
    print("y_COex: " + str(y_COex))
    print("y_CO2ex: " + str(y_CO2ex))
    print("y_H2Oex: " + str(y_H2Oex))
    print("Total mole/s: " + str(n_H2ex + n_COex + n_CO2ex + n_H2Oex))

    n_N2ex = n_N2f
    n_O2ex = n_O2f - n_CH4f * Uf * 2
    y_O2ex = n_O2ex / (n_O2ex + n_N2ex)
    y_N2ex = n_N2ex / (n_O2ex + n_N2ex)
    print("Cathode exhaust: ")
    print("y_O2ex: " + str(y_O2ex))
    print("y_N2ex: " + str(y_N2ex))
    print("Total mole/s: " + str(n_O2ex + n_N2ex))

    return [n_CH4f, n_H2Of, n_O2f, n_N2f, Uf]


def temperature_trans(air_in, fuel_in, ex_out):
    """
    Convert degree celsius to kelvin.
    Optional keyword arguments:
    ---------------------------
    :param air_in: temperature of air coming in to fuel cell(FC)
    :param fuel_in: temperature of fuel coming into (FC)/
                    temperature of reformer
    :param ex_out: temperature of exhaust coming out of FC
    :return: return the degree of kelvin
    """
    T_FC_air_in = air_in + 273.15
    T_FC_fuel_in = fuel_in + 273.15
    T_FC_ex_out = ex_out + 273.15

    return [T_FC_air_in, T_FC_fuel_in, T_FC_ex_out]


def concrete_model_base(m):
    """
    Concrete and declare all units and streams in the model.
    Optional keyword arguments:
    ---------------------------
    :param m: the model
    :return: return the model
    """
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.thermo_params = thermo_props.MethaneParameterBlock()
    # Declare all Units
    m.fs.HX1 = Heater(default={"property_package": m.fs.thermo_params})
    m.fs.HX2a = Heater(default={"property_package": m.fs.thermo_params})
    m.fs.HX2b = Heater(default={"property_package": m.fs.thermo_params})
    m.fs.Mix1 = Mixer(default={"dynamic": False,
                               "property_package": m.fs.thermo_params})
    m.fs.Mix2 = Mixer(default={"dynamic": False,
                               "property_package": m.fs.thermo_params})
    m.fs.Mix3 = Mixer(default={"dynamic": False,
                               "property_package": m.fs.thermo_params})
    m.fs.Split1 = Separator(
                 default={"dynamic": False,
                          "split_basis": SplittingType.componentFlow,
                          "property_package": m.fs.thermo_params})
    m.fs.Reformer = GibbsReactor(
                    default={"dynamic": False,
                             "property_package": m.fs.thermo_params,
                             "has_pressure_change": False,
                             "has_heat_transfer": True})
    m.fs.SOFC = GibbsReactor(default={"dynamic": False,
                                      "property_package": m.fs.thermo_params,
                                      "has_pressure_change": False,
                                      "has_heat_transfer": True})
    m.fs.Burner = GibbsReactor(default={"dynamic": False,
                                        "property_package": m.fs.thermo_params,
                                        "has_pressure_change": False,
                                        "has_heat_transfer": True})
    # Declare all Streams
    m.fs.stream0 = Arc(source=m.fs.Mix1.outlet,
                       destination=m.fs.HX1.inlet)
    m.fs.stream1 = Arc(source=m.fs.Split1.outlet_1,
                       destination=m.fs.HX2b.inlet)
    m.fs.stream2 = Arc(source=m.fs.HX1.outlet,
                       destination=m.fs.Reformer.inlet)
    m.fs.stream3 = Arc(source=m.fs.Split1.outlet_2,
                       destination=m.fs.HX2a.inlet)
    m.fs.stream4 = Arc(source=m.fs.Reformer.outlet,
                       destination=m.fs.Mix2.inlet_1)
    m.fs.stream5 = Arc(source=m.fs.HX2b.outlet,
                       destination=m.fs.Mix2.inlet_2)
    m.fs.stream6 = Arc(source=m.fs.Mix2.outlet,
                       destination=m.fs.SOFC.inlet)
    m.fs.stream7 = Arc(source=m.fs.HX2a.outlet,
                       destination=m.fs.Mix3.inlet_2)
    m.fs.stream8 = Arc(source=m.fs.SOFC.outlet,
                       destination=m.fs.Mix3.inlet_1)
    m.fs.stream9 = Arc(source=m.fs.Mix3.outlet,
                       destination=m.fs.Burner.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    return m


def declare_material_streams(m, n_CH4f, n_H2Of, n_N2f, n_O2f, Uf):
    """
    Define known Material Streams and fix flows of methane and water to mix1,
    air and oxygen to splitter1,
    Optional keyword arguments:
    ---------------------------
    :param m: model
    :param n_CH4f: mole of methane feed
    :param n_H2Of: mole of water feed
    :param n_N2f: mole of nitrogen feed
    :param n_O2f: mole of oxygen feed
    :param Uf: Fuel ultilization, mole reductant consumed in FC per mole
               of reductant total
    :return: model after declaring material streams
    """
    # Define known Material Streams
    # Fix methane flow to Mix1:
    m.fs.Mix1.inlet_1.flow_mol.fix(n_CH4f)
    m.fs.Mix1.inlet_1.mole_frac_comp[0.0, :].fix(0.0)
    m.fs.Mix1.inlet_1.mole_frac_comp[0.0, "CH4"].fix(1.0)
    m.fs.Mix1.inlet_1.temperature.fix(25 + 273.15)
    m.fs.Mix1.inlet_1.pressure.fix(101325)

    # Fix water flow to Mix1:
    m.fs.Mix1.inlet_2.flow_mol.fix(n_H2Of)
    m.fs.Mix1.inlet_2.mole_frac_comp[0.0, :].fix(0.0)
    m.fs.Mix1.inlet_2.mole_frac_comp[0.0, "H2O"].fix(1.0)
    m.fs.Mix1.inlet_2.temperature.fix(25 + 273.15)
    m.fs.Mix1.inlet_2.pressure.fix(101325)

    # Fix air flow to Split1:
    m.fs.Split1.inlet.flow_mol.fix(n_N2f + n_O2f)
    m.fs.Split1.inlet.mole_frac_comp[0.0, :].fix(0.0)
    m.fs.Split1.inlet.mole_frac_comp[0.0, "O2"].fix(0.21)
    m.fs.Split1.inlet.mole_frac_comp[0.0, "N2"].fix(0.79)
    m.fs.Split1.inlet.temperature.fix(25 + 273.15)
    m.fs.Split1.inlet.pressure.fix(101325)

    # Fix O2 flow in Split1 outlet_1:
    m.fs.Split1.outlet_1.flow_mol.fix(n_CH4f * Uf * 2)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "CH4"].fix(0.0)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "CO"].fix(0.0)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "CO2"].fix(0.0)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "H2"].fix(0.0)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "H2O"].fix(0.0)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "N2"].fix(0.0)
    m.fs.Split1.outlet_1.mole_frac_comp[0.0, "O2"].fix(1.0)

    return m


def set_constraint(m, air_in, fuel_in, ex_out):
    """
    Set the constraints of the model.
    Optional keyword arguments:
    ---------------------------
    :param m: model
    :param air_in: temperature of air coming in to fuel cell(FC)
    :param fuel_in: temperature of fuel coming into (FC)/
                    temperature of reformer
    :param ex_out: temperature of exhaust coming out of FC
    :return: model after setting constraints
    """
    [T_FC_air_in, T_FC_fuel_in, T_FC_ex_out] = \
        temperature_trans(air_in, fuel_in, ex_out)
    m.fs.Burner.heat_duty.fix(0.0)
    m.fs.Reformer.heat_duty.fix(0.0)
    m.fs.Reformer.outlet.temperature.fix(T_FC_fuel_in)
    m.fs.SOFC.outlet.temperature.fix(T_FC_ex_out)
    m.fs.HX2a.outlet.temperature.fix(T_FC_ex_out)
    m.fs.HX2b.outlet.temperature.fix(T_FC_air_in)
    dof(m)

    return m


def print_out(m, n_CH4f, LHV, EE):
    """
    Print Burner exhaust temperature, SOFC energy output,
          SOFC efficiency and Reformer entrance temperature
    Optional keyword arguments:
    ---------------------------
    :param m: model
    :param n_CH4f: mole of methane feed
    :param LHV: Methane LHV(https://www.engineeringtoolbox.com/
                            fuels-higher-calorific-values-d_169.html)
    :param EE: Electrical conversion efficiency
    """
    n_CH4f = n_CH4f
    LHV = LHV
    EE = EE
    # Solution:
    print("Burner exhaust temperature: " +
          format(m.fs.Burner.outlet.temperature[0].value-273.15, ".2f")
          + u' \u2103')
    print("SOFC energy output: " +
          format(-m.fs.SOFC.heat_duty[0].value*EE, ".2f") + " J/s")
    print("SOFC efficiency: " +
          format(-m.fs.SOFC.heat_duty[0].value*EE/(n_CH4f*LHV)*100, ".2f") +
          " %")
    # Temperature into Reformer:
    print("Reformer entrance temperature: " +
          format(m.fs.Reformer.inlet.temperature[0].value-273.15, ".2f") +
          u' \u2103')
