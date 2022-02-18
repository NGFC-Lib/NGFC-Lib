import sys
import os
import numpy as np
from RL_ENV import convertobs2list

#%%
if __name__ == "__main__":
    
    # in case of testing an observation
    class user_inputs:
        def __init__(self):
            self.list_unit_all = ['inlet_feed', 'outlet_product', 'outlet_exhaust', \
                'mixer2to1_1','heater_1', 'StReactor_1', 'flash_1', 'splitter1to2_1', \
                'compressor_1']
            self.list_inlet_all = ['outlet_product.inlet', 'outlet_exhaust.inlet',\
                    'mixer2to1_1.inlet_1', 'mixer2to1_1.inlet_2', 'heater_1.inlet',\
                    'StReactor_1.inlet', 'flash_1.inlet', 'splitter1to2_1.inlet', \
                    'compressor_1.inlet']
            self.list_outlet_all = ['inlet_feed.outlet', 'mixer2to1_1.outlet',\
                    'heater_1.outlet', 'StReactor_1.outlet', 'flash_1.liq_outlet',\
                        'flash_1.vap_outlet', 'splitter1to2_1.outlet_1',\
                            'splitter1to2_1.outlet_2','compressor_1.outlet']
    user_inputs = user_inputs()

    observation = [0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0] 
    list_unit, list_inlet, list_outlet = convertobs2list(observation, user_inputs)
    print('list_unit = ', list_unit)
    print('list_inlet = ', list_inlet)
    print('list_outlet = ', list_outlet)

