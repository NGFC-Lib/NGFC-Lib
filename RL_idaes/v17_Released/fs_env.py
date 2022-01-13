# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 18:13:36 2021

@author: baoj529
"""

import numpy as np
import random
# import sys

def convertobs2list(observation, user_inputs):

    # load all units, inlets and outlets (string)
    list_inlet_all = user_inputs.list_inlet_all
    list_outlet_all = user_inputs.list_outlet_all

    # convert to list_unit, list_inlet and list_outlet
    matrix_conn = np.reshape(observation,(len(list_inlet_all),len(list_outlet_all)))
    inlet_conn_repeat = np.sum(matrix_conn,axis=1)
    outlet_conn_repeat = np.sum(matrix_conn,axis=0)

    if np.any(inlet_conn_repeat>1) or np.any(outlet_conn_repeat>1):
        # reward = -1000
        # IDAES_status = ['unavailable', 0.0, 0.0]
        # return reward, IDAES_status

        list_unit = []	
        list_inlet = []
        list_outlet = []
    else:
        list_unit = []	
        list_inlet = []
        list_outlet = []

        str_unit_out = []
        str_unit_in = []
        for k in range(len(list_outlet_all)):
            tmp1, tmp2 = [x.strip() for x in list_outlet_all[k].split(".")]
            str_unit_out.append(tmp1)
        for l in range(len(list_inlet_all)):
            tmp1, tmp2 = [x.strip() for x in list_inlet_all[l].split(".")]
            str_unit_in.append(tmp1)

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

def pre_screening(list_unit, list_inlet, list_outlet, Action_taken):

    # start from initial score of 500
    pass_pre_screening = False
    minimum_score = -1000
    pres_score = 500
    delta_scoreA = 160  #penalty option 1
    delta_scoreB = 80   #penalty option 2
    delta_scoreC = 40   #penalty option 3
    delta_scoreD = -10   #penalty option 4

    # physics constraint 0: repeated connections
    if len(list_unit) == 0:
        pres_score = minimum_score
        return pres_score, pass_pre_screening

    # physics constraint 1: N(inlet) = N(outlet)
    if len(list_inlet) != len(list_outlet):
        pres_score = pres_score-delta_scoreA

    # physics constraint 2: several units essential
    list_unit_essential = ['inlet_feed', 'outlet_product', 'StReactor_1']
    if not all(elem in list_unit  for elem in list_unit_essential):
        pres_score = pres_score-delta_scoreA

    # physics constraints 3, 4, 5, 6, 7
    list_str_unit_in = []; list_str_unit_out = []
    N_list_str = min(len(list_outlet), len(list_inlet))
    for k in range(N_list_str):
        str_unit_in, str_inlet = [x.strip() for x in list_inlet[k].split(".")]
        str_unit_out, str_outlet = [x.strip() for x in list_outlet[k].split(".")]
        list_str_unit_in.append(str_unit_in)
        list_str_unit_out.append(str_unit_out)

        # physics constraint 3: unit cannot connect to itself
        if str_unit_in == str_unit_out:
            pres_score = minimum_score
            return pres_score, pass_pre_screening

        # physics constraint 4: inlet cannot directly connect outlet
        if str_unit_in == 'outlet_product' and str_unit_out == 'inlet_feed':
            pres_score = pres_score-delta_scoreA

        if str_unit_in == 'outlet_exhaust' and str_unit_out == 'inlet_feed':
            pres_score = pres_score-delta_scoreA

        # physics constraint 5: outlet_exhaust can only connect to splitter or flash
            # when connected to a splitter, it must connect to outlet_2 (less side)
        if str_unit_in == 'outlet_exhaust':
            if str_unit_out not in ['flash_1', 'flash_2', 'splitter1to2_1', 'splitter1to2_2']:
                pres_score = pres_score-delta_scoreA

            elif str_unit_out in ['splitter1to2_1', 'splitter1to2_2'] and str_outlet == 'outlet_1':
                pres_score = pres_score-delta_scoreB

        # physics constraint 6: liq_outlet cannot connect to compressor or expander
        if str_unit_out in ['flash_1', 'flash_2'] and str_outlet == 'liq_outlet': 
            if str_unit_in in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2']:
                pres_score = pres_score-delta_scoreA
                
        # physics constraint 7: heater/compressor cannot connect to cooler/expander
        if str_unit_out in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2']: 
            if str_unit_in in ['compressor_1', 'compressor_2', 'expander_1', 'expander_2']:
                pres_score = pres_score-delta_scoreB
                
        if str_unit_out in ['heater_1', 'heater_2', 'cooler_1', 'cooler_2']: 
            if str_unit_in in ['heater_1', 'heater_2', 'cooler_1', 'cooler_2']: 
                pres_score = pres_score-delta_scoreB
                
        if str_unit_out in ['StReactor_1', 'StReactor_2']: 
            if str_unit_in in ['StReactor_1', 'StReactor_2']:  
                pres_score = pres_score-delta_scoreB

    # physics constraint 8: splitter cannot connect to mixer completely
    if 'splitter1to2_1' in list_str_unit_out:
        indices_1 = [i for i,val in enumerate(list_str_unit_out) if val == 'splitter1to2_1']
        if 'mixer2to1_1' in list_str_unit_in:
            indices_3 = [i for i,val in enumerate(list_str_unit_in) if val == 'mixer2to1_1']
            if indices_1 == indices_3:
                pres_score = pres_score-delta_scoreA
        if 'mixer2to1_2' in list_str_unit_in:
            indices_4 = [i for i,val in enumerate(list_str_unit_in) if val == 'mixer2to1_2']
            if indices_1 == indices_4:
                pres_score = pres_score-delta_scoreA
    if 'splitter1to2_2' in list_str_unit_out:
        indices_2 = [i for i,val in enumerate(list_str_unit_out) if val == 'splitter1to2_2']
        if 'mixer2to1_1' in list_str_unit_in:
            indices_3 = [i for i,val in enumerate(list_str_unit_in) if val == 'mixer2to1_1']
            if indices_2 == indices_3:
                pres_score = pres_score-delta_scoreA
        if 'mixer2to1_2' in list_str_unit_in:
            indices_4 = [i for i,val in enumerate(list_str_unit_in) if val == 'mixer2to1_2']
            if indices_2 == indices_4:
                pres_score = pres_score-delta_scoreA

    # physics constraint 9: all units must be in the same cycle
    connect2inlet = all_units_downstream(list_str_unit_in, list_str_unit_out, 'inlet_feed')

    if 0 in connect2inlet:
        # print('debug-constraint 9 fails')
        pres_score = pres_score-delta_scoreA

    # physics constraint 10: there must be reactor between outlet_product/flash and inlet_feed
    list_str_unit_in_out = list_str_unit_in + list_str_unit_out
    if 'outlet_product' in list_str_unit_in_out and 'inlet_feed' in list_str_unit_in_out:
        unit_in_route = all_units_between(list_str_unit_in, list_str_unit_out, 'inlet_feed', 'outlet_product')
        # print('debug-unit_in_route 1: ', unit_in_route)
        if 'StReactor_1' not in unit_in_route and 'StReactor_2' not in unit_in_route:
            # print('constraint 10 fails: 1')
            pres_score = pres_score-delta_scoreA

    if 'outlet_exhaust' in list_str_unit_in_out and 'inlet_feed' in list_str_unit_in_out:
        unit_in_route = all_units_between(list_str_unit_in, list_str_unit_out, 'inlet_feed', 'outlet_exhaust')
        # print('debug-unit_in_route 2: ', unit_in_route)
        if 'StReactor_1' not in unit_in_route and 'StReactor_2' not in unit_in_route:
            # print('constraint 10 fails: 2')
            pres_score = pres_score-delta_scoreA

    if 'flash_1' in list_str_unit_in_out and 'inlet_feed' in list_str_unit_in_out:
        unit_in_route = all_units_between(list_str_unit_in, list_str_unit_out, 'inlet_feed', 'flash_1')
        # print('debug-unit_in_route 3: ', unit_in_route)
        if 'StReactor_1' not in unit_in_route and 'StReactor_2' not in unit_in_route:
            # print('constraint 10 fails: 3')
            pres_score = pres_score-delta_scoreA

    if 'flash_2' in list_str_unit_in_out and 'inlet_feed' in list_str_unit_in_out:
        unit_in_route = all_units_between(list_str_unit_in, list_str_unit_out, 'inlet_feed', 'flash_2')
        # print('debug-unit_in_route 4: ', unit_in_route)
        if 'StReactor_1' not in unit_in_route and 'StReactor_2' not in unit_in_route:
            # print('constraint 10 fails: 4')
            pres_score = pres_score-delta_scoreA

    # physics constraint 11: for each unit, all inlets and outlets must be selected
    for k in range(len(list_unit)):
        str_unit = list_unit[k]

        if str_unit not in ['inlet_feed', 'outlet_product', 'outlet_exhaust']:
            if str_unit == 'mixer2to1_1' or str_unit == 'mixer2to1_2':
                if list_str_unit_in.count(str_unit) < 2 or list_str_unit_out.count(str_unit) < 1:
                    # print('contraint 11 fails: 1')
                    pres_score = pres_score-delta_scoreC
            elif str_unit == 'flash_1' or str_unit == 'flash_2' or str_unit == 'splitter1to2_1' or str_unit == 'splitter1to2_2':
                if list_str_unit_in.count(str_unit) < 1 or list_str_unit_out.count(str_unit) < 2:
                    # print('contraint 11 fails: 2')
                    pres_score = pres_score-delta_scoreC
            else: #all the rest units have one inlet and one outlet
                if list_str_unit_in.count(str_unit) < 1 or list_str_unit_out.count(str_unit) < 1:
                    # print('contraint 11 fails: 3')
                    pres_score = pres_score-delta_scoreC
    
    # add penalty/reward as unit # increases
    if pres_score == 500:
        pass_pre_screening = True
    # pres_score = pres_score-delta_scoreD*len(list_unit)

    # add penalty as no action taken
    if Action_taken == False:
        pres_score = pres_score-delta_scoreB
    
    return pres_score, pass_pre_screening

def all_units_downstream(list_str_unit_in, list_str_unit_out, string_unit):

    connected = np.zeros(len(list_str_unit_out), dtype=int)

    index_1 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == string_unit]
    for i1 in index_1:
        connected[i1] = 1
        index_2 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i1]]

        for i2 in index_2:
            connected[i2] = 1
            index_3 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i2]]

            for i3 in index_3:
                connected[i3] = 1
                index_4 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i3]]

                for i4 in index_4:
                    connected[i4] = 1
                    index_5 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i4]]

                    for i5 in index_5:
                        connected[i5] = 1
                        index_6 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i5]]

                        for i6 in index_6:
                            connected[i6] = 1
                            index_7 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i6]]

                            for i7 in index_7:
                                connected[i7] = 1
                                index_8 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i7]]

                                for i8 in index_8:
                                    connected[i8] = 1
                                    index_9 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i8]]

                                    for i9 in index_9:
                                        connected[i9] = 1
                                        index_10 = [ele for ele in range(len(list_str_unit_out)) if list_str_unit_out[ele] == list_str_unit_in[i9]]

                                        for i10 in index_10:
                                            connected[i10] = 1

    return connected

def all_units_between(list_str_unit_in, list_str_unit_out, string_unit_1, string_unit_2):

    index_1 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == string_unit_2]
    for i1 in index_1:
        unit1 = list_str_unit_out[i1]
        if list_str_unit_out[i1] == string_unit_1:
            return [unit1]
        index_2 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i1]]

        for i2 in index_2:
            unit2 = list_str_unit_out[i2]
            if list_str_unit_out[i2] == string_unit_1:
                return [unit1, unit2]
            index_3 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i2]]

            for i3 in index_3:
                unit3 = list_str_unit_out[i3]
                if list_str_unit_out[i3] == string_unit_1:
                    return [unit1, unit2, unit3]
                index_4 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i3]]

                for i4 in index_4:
                    unit4 = list_str_unit_out[i4]
                    if list_str_unit_out[i4] == string_unit_1:
                        return [unit1, unit2, unit3, unit4]
                    index_5 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i4]]

                    for i5 in index_5:
                        unit5 = list_str_unit_out[i5]
                        if list_str_unit_out[i5] == string_unit_1:
                            return [unit1, unit2, unit3, unit4, unit5]
                        index_6 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i5]]

                        for i6 in index_6:
                            unit6 = list_str_unit_out[i6]
                            if list_str_unit_out[i6] == string_unit_1:
                                return [unit1, unit2, unit3, unit4, unit5, unit6]
                            index_7 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i6]]

                            for i7 in index_7:
                                unit7 = list_str_unit_out[i7]
                                if list_str_unit_out[i7] == string_unit_1:
                                    return [unit1, unit2, unit3, unit4, unit5, unit6, unit7]
                                index_8 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i7]]

                                for i8 in index_8:
                                    unit8 = list_str_unit_out[i8]
                                    if list_str_unit_out[i8] == string_unit_1:
                                        return [unit1, unit2, unit3, unit4, unit5, unit6, unit7, unit8]
                                    index_9 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i8]]

                                    for i9 in index_9:
                                        unit9 = list_str_unit_out[i9]
                                        if list_str_unit_out[i9] == string_unit_1:
                                            return [unit1, unit2, unit3, unit4, unit5, unit6, unit7, unit8, unit9]
                                        index_10 = [ele for ele in range(len(list_str_unit_in)) if list_str_unit_in[ele] == list_str_unit_out[i9]]

                                        for i10 in index_10:
                                            unit10 = list_str_unit_out[i10]
                                            if list_str_unit_out[i10] == string_unit_1:
                                                return [unit1, unit2, unit3, unit4, unit5, unit6, unit7, unit8, unit9, unit10]

    return []

#%%
class fs_gen():
    def __init__(self, n_inlets, n_outlets, n_features):
        self.num_elements =     n_outlets # n_cols
        self.MAX_Iteration =    n_inlets # n_rows
        self.n_features =       n_features # n_cols*n_rows
        self.action_space =     np.arange(self.num_elements+1)
        self.n_actions  =       len(self.action_space)
        
    def reset(self,episode):
        # Initialize the enviroment observation
        # time.sleep(0.1)
        s = np.zeros(self.n_features,dtype=float)
        #old_s=np.zeros(self.num_elements)
        #init_one=random.randint(0,self.num_elements-1)
        #s[init_one]=1
        #if episode > 1000: s[init_one]=1
        
        #init_one=random.randint(0,self.num_elements-1)
        
        #if episode > 1000: s[init_one]=1
        #s[1]=1
        #s[6]=1
        #s[22]=1
        picked_true_fs=random.randint(13,14) #episode%15
        #obj_rec = 1000.0

        self.step_counter = 0
        #self.obj_rec = obj_rec
        return s, picked_true_fs

    def update_env(self,S,old_obs_,i_step):
        # * Update enviroment to get new observation
        #print("old_obs ",old_obs_)
        #print("i_step",i_step,S)
        # old_val = old_obs_[i_step*self.num_elements+S]
        new_obs_ = np.copy(old_obs_)
        #print("i_step",i_step,S)
        new_obs_[i_step*self.num_elements+S] = 1.0
        #print("old and new_obs ",old_obs_, " ", new_obs_)
        return new_obs_

    def step(self,action,old_obs,episode,i_step,user_inputs):

        S=action
        if S < self.num_elements:
            new_obs = self.update_env(S,old_obs,i_step)
            Action_taken = True
        else:
            new_obs = np.copy(old_obs) # can be optimized to avoid repeated pre-screening
            Action_taken = False
        
        list_unit, list_inlet, list_outlet = convertobs2list(new_obs, user_inputs)
        R, pass_pre_screening = pre_screening(list_unit, list_inlet, list_outlet, Action_taken)

        episode_done = False
        if R == -1000 or i_step > self.MAX_Iteration-2:
            episode_done = True

        return new_obs, R, episode_done, pass_pre_screening