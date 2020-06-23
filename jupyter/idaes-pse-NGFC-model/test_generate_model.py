import unittest

import generate_model
from pyomo.environ import ConcreteModel

class TestSimulationTools(unnitest.TestCase):

    def test_feed_exhaust(self):
        assert len(generate_model.feed_exhaust
                   (Uf=0.8, Ua=0.2, MS=2, n_CH4f=10, n_H2ex=2)) == 5, \
            "Warning: Incorrect length of list generated"
        assert type(generate_model.feed_exhaust
                    (Uf=0.8, Ua=0.2, MS=2, n_CH4f=10, n_H2ex=2)[0]) == int, \
            "Warning: Incorrect type"
        return


    def test_temperature_trans(self):
        assert len(generate_model.temperature_trans
                   (air_in=700, fuel_in=500, ex_out=800)) == 3, \
            "Warning: Incorrect length of list generated"
        assert type(generate_model.temperature_trans
                    (air_in=700, fuel_in=500, ex_out=800)[0]) == float, \
            "Warning: Incorrect type"
        return


    def test_concrete_model_base(self):
        m = ConcreteModel()
        assert type(generate_model.concrete_model_base(m)) == ConcreteModel, \
            "Warning: Incorrect type"
        return
