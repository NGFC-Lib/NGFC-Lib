#%% Loading required packages.
from RL_CORE import RL_call

#%%
if __name__ == "__main__":
    # User inputs and model setups
    
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
    
    # prameters for the RL-GNN-IDAES integrated framework.
    # adjustable variables: Episode_max, model_index, GNN_enable, learning_rate, e_greedy, e_greedy_increment_ref

    P = {'model_restore':False,'model_save':False,'Episode_max_mode':'static','Episode_max':1e5,
          'threshold_learn':-1,'model_index':1,'GNN_enable':False,'learning_rate':0.01,
          'reward_decay':0.5,'e_greedy':0.9,'replace_target_iter':200,'memory_size':20000,
          'e_greedy_increment_ref':1e-5,'e_greedy_increment_type':'variable',
          'batch_size':32,'e_greedy_increment_intervals':[0,0.4,0.6,0.7,0.8,0.9,1.0],
          'e_greedy_increment_ratios':[1,1,1,1,1,1],'Additianl_step':500}

    # Calling RL framework to train the model
    RL_call(user_inputs(),P)