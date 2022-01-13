import sys
import os
import numpy as np

def convertobs2list(observation, user_inputs):

    # load all units, inlets and outlets (string)
    list_inlet_all = user_inputs.list_inlet_all
    list_outlet_all = user_inputs.list_outlet_all

    str_unit_out = []
    for k in range(len(list_outlet_all)):
        tmp1, tmp2 = [x.strip() for x in list_outlet_all[k].split(".")]
        str_unit_out.append(tmp1)

    str_unit_in = []
    for l in range(len(list_inlet_all)):
        tmp1, tmp2 = [x.strip() for x in list_inlet_all[l].split(".")]
        str_unit_in.append(tmp1)

    # convert to list_unit, list_inlet and list_outlet
    matrix_conn = np.reshape(observation,(len(list_inlet_all),len(list_outlet_all)))

    inlet_conn_repeat = np.sum(matrix_conn,axis=1)
    outlet_conn_repeat = np.sum(matrix_conn,axis=0)

    if np.any(inlet_conn_repeat>1) or np.any(outlet_conn_repeat>1):
        sys.exit("This is an infeasible observation")

    list_unit = []	
    list_inlet = []
    list_outlet = []
    for i in range(len(str_unit_in)):
        unit = str_unit_in[i]
        if unit not in list_unit and np.sum(matrix_conn[i, :]) == 1:
            list_unit.append(unit)
    for j in range(len(str_unit_out)):
        unit = str_unit_out[j]
        if unit not in list_unit and np.sum(matrix_conn[:, j]) == 1:
            list_unit.append(unit)

    for i in range(len(list_inlet_all)):
        if np.sum(matrix_conn[i, :]) == 1:
            list_inlet.append(list_inlet_all[i])
            for j in range(len(list_outlet_all)):
                if matrix_conn[i, j] == 1:
                    list_outlet.append(list_outlet_all[j])

    return list_unit, list_inlet, list_outlet

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

    observation = [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0] 
    list_unit, list_inlet, list_outlet = convertobs2list(observation, user_inputs)
    print('list_unit = ', list_unit)
    print('list_inlet = ', list_inlet)
    print('list_outlet = ', list_outlet)

