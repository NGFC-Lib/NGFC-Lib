from pyomo.environ import TransformationFactory
from pyomo.network import Arc
from idaes.core import FlowsheetBlock
from idaes.generic_models.unit_models import HeatExchanger, Separator, Heater
import idaes.generic_models.properties.\
       activity_coeff_models.methane_combustion_ideal as thermo_props
from idaes.generic_models.unit_models.heat_exchanger \
    import delta_temperature_amtd_callback
from idaes.core.util.model_statistics import degrees_of_freedom as dof


def declare_heatx_unit(h):
    """
    Declare all units of the heat exchanger network.
    Optional keyword arguments:
    ---------------------------
    :param h: the heat exchanger network
    :return: return the heat exchanger network
    """
    h.fs = FlowsheetBlock(default={"dynamic": False})
    h.fs.thermo_params = thermo_props.MethaneParameterBlock()
    h.fs.Split2 = Separator(default={"dynamic": False,
                                     "num_outlets": 3,
                                     "property_package": h.fs.thermo_params})
    h.fs.HX1a = HeatExchanger(
        default={"dynamic": False,
                 "delta_temperature_callback": delta_temperature_amtd_callback,
                 "shell": {"property_package": h.fs.thermo_params},
                 "tube": {"property_package": h.fs.thermo_params}})
    h.fs.HX2a = HeatExchanger(
        default={"dynamic": False,
                 "delta_temperature_callback": delta_temperature_amtd_callback,
                 "shell": {"property_package": h.fs.thermo_params},
                 "tube": {"property_package": h.fs.thermo_params}})
    h.fs.HX2b = HeatExchanger(
        default={"dynamic": False,
                 "delta_temperature_callback": delta_temperature_amtd_callback,
                 "shell": {"property_package": h.fs.thermo_params},
                 "tube": {"property_package": h.fs.thermo_params}})
    h.fs.HX1b = Heater(default={"property_package": h.fs.thermo_params})

    return h


def define_heatx_streams(m, h):
    """
    Declare all steams in the heat exchanger network
    via the SOFC model built before.
    Optional keyword arguments:
    ---------------------------
    :param m: SOFC model
    :param h: the heat exchanger network
    :return: heat exchanger network after declaring all steams
    """
    h.fs.Split2.inlet.flow_mol.fix(m.fs.Burner.outlet.flow_mol[0].value)
    h.fs.HX1a.tube_inlet.flow_mol.fix(m.fs.Mix1.outlet.flow_mol[0].value)
    h.fs.HX2a.tube_inlet.flow_mol.fix(m.fs.Split1.outlet_1.flow_mol[0].value)
    h.fs.HX2b.tube_inlet.flow_mol.fix(m.fs.Split1.outlet_2.flow_mol[0].value)

    material = ["CH4", "CO", "CO2", "H2", "H2O", "N2", "NH3", "O2"]
    for i in material:
        h.fs.Split2.inlet.mole_frac_comp[0, i].\
            fix(m.fs.Burner.outlet.mole_frac_comp[0, i].value)
        h.fs.HX1a.tube_inlet.mole_frac_comp[0, i].\
            fix(m.fs.Mix1.outlet.mole_frac_comp[0, i].value)
        h.fs.HX2a.tube_inlet.mole_frac_comp[0, i].\
            fix(m.fs.Split1.outlet_1.mole_frac_comp[0, i].value)
        h.fs.HX2b.tube_inlet.mole_frac_comp[0, i].\
            fix(m.fs.Split1.outlet_2.mole_frac_comp[0, i].value)

    h.fs.Split2.inlet.temperature.fix(m.fs.Burner.outlet.temperature[0].value)
    h.fs.Split2.inlet.pressure.fix(m.fs.Burner.outlet.pressure[0].value)

    h.fs.HX1a.tube_inlet.temperature.fix(m.fs.Mix1.outlet.temperature[0].value)
    h.fs.HX1a.tube_inlet.pressure.fix(m.fs.Mix1.outlet.pressure[0].value)

    h.fs.HX2a.tube_inlet.temperature.\
        fix(m.fs.Split1.outlet_1.temperature[0].value)
    h.fs.HX2a.tube_inlet.pressure.fix(m.fs.Split1.outlet_1.pressure[0].value)

    h.fs.HX2b.tube_inlet.temperature.\
        fix(m.fs.Split1.outlet_2.temperature[0].value)
    h.fs.HX2b.tube_inlet.pressure.fix(m.fs.Split1.outlet_2.pressure[0].value)

    # Declare all Streams
    h.fs.stream0 = Arc(source=h.fs.Split2.outlet_1,
                       destination=h.fs.HX1a.shell_inlet)
    h.fs.stream1 = Arc(source=h.fs.Split2.outlet_2,
                       destination=h.fs.HX2a.shell_inlet)
    h.fs.stream2 = Arc(source=h.fs.Split2.outlet_3,
                       destination=h.fs.HX2b.shell_inlet)
    h.fs.stream3 = Arc(source=h.fs.HX1a.shell_outlet,
                       destination=h.fs.HX1b.inlet)

    TransformationFactory("network.expand_arcs").apply_to(h)

    return h


def heatx_constraints(m, h, n_H2Of, enthalpy_vap, air_in):
    """
    Set up the constraints of the heat exchanger network.
    Optional keyword arguments:
    ---------------------------
    :param m: SOFC model
    :param h: the heat exchanger network
    :param n_H2Of: mole of water feed
    :param enthalpy_vap: Heat of water vaporization @ 25 C
    :param air_in: temperature of air coming in to fuel cell(FC)
    :return: heat exchanger network after setting constraints
    """
    T_FC_air_in = air_in + 273.15
    # heat transfer coeff of heat exchanger:
    h.fs.HX1a.overall_heat_transfer_coefficient.fix(200)
    h.fs.HX2a.overall_heat_transfer_coefficient.fix(200)
    h.fs.HX2b.overall_heat_transfer_coefficient.fix(200)

    # HX4/Evaporator:
    h.fs.HX1b.heat_duty.fix(-n_H2Of * enthalpy_vap)

    # HX cold/tube side outlet temperature:
    h.fs.HX1a.tube_outlet.temperature.fix(m.fs.HX1.outlet.temperature[0].value)
    h.fs.HX2a.tube_outlet.temperature.fix(T_FC_air_in)
    h.fs.HX2b.tube_outlet.temperature.fix(T_FC_air_in)

    # HX hot/shell side outlet temperature:
    h.fs.HX2a.shell_outlet.temperature.fix(200 + 273.15)
    h.fs.HX2b.shell_outlet.temperature.fix(200 + 273.15)

    dof(h)

    return h


def print_out_heatx(h):
    """
    Print inlet/outlet temperatures of tube/shell in preheat heat exchanger
    and air heat exchanger.
    Print exhaust split ratio.
    Optional keyword arguments:
    ---------------------------
    :param h: the heat exchanger network
    """
    print("HX1 - Steam, methane preheat heat exchanger: ")
    print("\tTube inlet temperature: \t" +
          format(h.fs.HX1a.tube_inlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tTube outlet temperature: \t" +
          format(h.fs.HX1a.tube_outlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tShell inlet temperature: \t" +
          format(h.fs.HX1a.shell_inlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tShell outlet temperature: \t" +
          format(h.fs.HX1b.outlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')

    print("HX2 - Air heat exchanger: ")
    print("\tTube inlet temperature: \t" +
          format(h.fs.HX2a.tube_inlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tTube outlet temperature: \t" +
          format(h.fs.HX2a.tube_outlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tShell inlet temperature: \t" +
          format(h.fs.HX2a.shell_inlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tShell outlet temperature: \t" +
          format(h.fs.HX2a.shell_outlet.temperature[0].value - 273.15, ".2f") +
          u' \u2103')
    print("\tAssumed heat transfer coefficient: \t" +
          format(h.fs.HX2a.overall_heat_transfer_coefficient[0].value, ".2f") +
          " W/m2.K")
    print("\tHeat exchanger area: \t\t" +
          format(h.fs.HX2a.area.value + h.fs.HX2b.area.value, ".2f") + " m2")

    print("Exhaust split ratio: ")
    print("\tSplit fraction to methane, steam heat exchanger: " + format(
        h.fs.Split2.split_fraction[0, "outlet_1"].value, ".3f"))
    print("\tSplit fraction to air heat exchanger: " +
          format(1 - h.fs.Split2.split_fraction[0, "outlet_1"].value,  ".3f"))
