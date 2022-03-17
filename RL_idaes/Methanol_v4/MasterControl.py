#%% Loading required packages.
from RL_CORE import RL_call

#%%
if __name__ == "__main__":

    # User inputs and model setups
        # only one reactor
        # remove sequence: reactor_2, flash_2, splitter_2, 

    class user_inputs_17:
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'heater_2', 'reactor_1', 'reactor_2',
                            'turbine_1', 'turbine_2', 'flash_0', 'flash_1', 'exhaust', 'cooler_1', 'cooler_2', 'splitter_1', 'splitter_2']
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'reactor_1.inlet', 'reactor_2.inlet',
                            'turbine_1.inlet', 'turbine_2.inlet', 'flash_1.inlet', 'flash_0.inlet', 'exhaust.inlet', 'cooler_1.inlet', 'cooler_2.inlet', 'splitter_1.inlet', 'splitter_2.inlet']
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'reactor_1.outlet', 'reactor_2.outlet',
                            'turbine_1.outlet', 'turbine_2.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet', 'cooler_1.outlet', 'cooler_2.outlet', 
                            'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2']

    class user_inputs_16: # remove 2nd reactor
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'heater_2', 'reactor_1',
                            'turbine_1', 'turbine_2', 'flash_0', 'flash_1', 'exhaust', 'cooler_1', 'cooler_2', 'splitter_1', 'splitter_2']
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'turbine_2.inlet', 'flash_1.inlet', 'flash_0.inlet', 'exhaust.inlet', 'cooler_1.inlet', 'cooler_2.inlet', 'splitter_1.inlet', 'splitter_2.inlet']
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'turbine_2.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet', 'cooler_1.outlet', 'cooler_2.outlet', 
                            'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2']
    
    class user_inputs_15: # remove 2nd flash
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'heater_2', 'reactor_1',
                            'turbine_1', 'turbine_2', 'flash_0', 'exhaust', 'cooler_1', 'cooler_2', 'splitter_1', 'splitter_2'] #15
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'turbine_2.inlet', 'flash_0.inlet', 'exhaust.inlet', 'cooler_1.inlet', 'cooler_2.inlet', 'splitter_1.inlet', 'splitter_2.inlet'] #15
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'turbine_2.outlet', 'cooler_1.outlet', 'cooler_2.outlet', 
                            'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2'] #15

    class user_inputs_14: # remove both coolers, add 2nd flash
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'heater_2', 'reactor_1',
                            'turbine_1', 'turbine_2', 'flash_0', 'flash_1', 'exhaust', 'splitter_1', 'splitter_2'] #14
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'turbine_2.inlet', 'flash_1.inlet', 'flash_0.inlet', 'exhaust.inlet', 'splitter_1.inlet', 'splitter_2.inlet'] #14
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'turbine_2.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet',
                            'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2'] #15
    
    class user_inputs_13: # remove 2nd flash
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'compressor_2', 'heater_1', 'heater_2', 'reactor_1',
                            'turbine_1', 'turbine_2', 'flash_0', 'exhaust', 'splitter_1', 'splitter_2'] #13
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'compressor_2.inlet', 'heater_1.inlet', 'heater_2.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'turbine_2.inlet', 'flash_0.inlet', 'exhaust.inlet', 'splitter_1.inlet', 'splitter_2.inlet'] #13
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'compressor_2.outlet', 'heater_1.outlet', 'heater_2.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'turbine_2.outlet',
                            'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2'] #13

    class user_inputs_12: # remove 2nd compressor, 2nd heater, 2nd turbine, add 1st cooler, 2nd flash
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'cooler_1', 'reactor_1',
                            'turbine_1', 'flash_0', 'flash_1', 'exhaust', 'splitter_1', 'splitter_2'] #12
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'heater_1.inlet', 'cooler_1.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'flash_1.inlet', 'flash_0.inlet', 'exhaust.inlet', 'splitter_1.inlet', 'splitter_2.inlet'] #12
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', 'cooler_1.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'flash_1.liq_outlet', 'flash_1.vap_outlet',
                            'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2'] #13

    class user_inputs_11: # remove 2nd flash
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'cooler_1', 'reactor_1',
                            'turbine_1', 'flash_0', 'exhaust', 'splitter_1', 'splitter_2'] #11
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'heater_1.inlet', 'cooler_1.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'flash_0.inlet', 'exhaust.inlet', 'splitter_1.inlet', 'splitter_2.inlet'] #11
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', 'cooler_1.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2', 'splitter_2.outlet_1', 'splitter_2.outlet_2'] #11

    class user_inputs_10: # remove 2nd splitter
        def __init__(self):
            self.list_unit_all = ['mixer_0', 'mixer_1', 'compressor_1', 'heater_1', 'cooler_1', 'reactor_1',
                            'turbine_1', 'flash_0', 'exhaust', 'splitter_1'] #10
            self.list_inlet_all = ['mixer_1.inlet_1', 'mixer_1.inlet_2', 'compressor_1.inlet', 'heater_1.inlet', 'cooler_1.inlet', 'reactor_1.inlet',
                            'turbine_1.inlet', 'flash_0.inlet', 'exhaust.inlet', 'splitter_1.inlet'] #10
            self.list_outlet_all = ['mixer_0.outlet', 'mixer_1.outlet', 'compressor_1.outlet', 'heater_1.outlet', 'cooler_1.outlet', 'reactor_1.outlet',
                            'turbine_1.outlet', 'splitter_1.outlet_1', 'splitter_1.outlet_2'] #9

    # prameters for the RL-GNN-IDAES integrated framework.
    P = {'model_restore':False,'model_save':True,'model_index':1,'visualize': False,
        'threshold_learn':-1,'GNN_enable':False,'learning_rate':0.01,'reward_decay':0.5,
        'replace_target_iter':200,'memory_size':20000,'batch_size':32,
        'Episode_max_mode':'dynamic','Episode_max':5e6,'Additional_step':1e5, # Episode_max_mode: dynamic or static
        'e_greedy':0.9,'e_greedy_increment':0.5e-5}

    # Calling RL framework to train the model
    RL_call(user_inputs_10(),P)