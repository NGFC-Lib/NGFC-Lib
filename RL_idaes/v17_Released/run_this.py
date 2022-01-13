#%%
from fs_env import fs_gen, convertobs2list
from RL_brain import DeepQNetwork
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from timeit import default_timer as timer
# from GNN_v1 import RL2GNN_Link, GNNmodel
import HDA_IDAES_v7 as HDA

#%%
def run_RL(env,f_rec):

	# check time consuming for training and running IDAES
	RL_start = timer()
	IDAES_consume = 0
	GNN_consume = 0

	tt_reward_hist=[]
	reward_hist=[] # debug
	new_obs=[]
	dif_obs=[]
	new_obs_reward = []
	new_obs_status = []
	new_obs_costing = []
	obs_runIdaes = []
	obs_runIdaes_reward = []
	obs_runIdaes_status = []
	obs_runIdaes_costing = []
    
	obs_fail_pre_screening = []
	obs_fail_pre_screening_reward = []

	IDAES_status = []
	IDAES_costing = []

	i_idaes = 0
	i_idaes_unique = 0
	i_feasible_unique = 0
	i_idaes_unique_hist = []
	i_feasible_unique_hist = []
	ma_horizon = 2000
	ma_reward_hist = []

	for episode in range(Episode_max):

		observation, picked_true_fs = env.reset(episode)
		i_step = 0
		tt_reward = 0
		
		while True:
		
			# RL choose action based on observation
			action, actions_value = RL.choose_action(observation)
			
			# RL take action and get next observation and reward
			observation_, reward, episode_done, pass_pre_screening = env.step(action,observation,episode,i_step,user_inputs)
						
			################# GNN start #################
			if GNN_enable:
				GNN_start = timer()
				GNN_index,GNN_label,GNN_matrix = RL2GNN_Link(observation_,user_inputs)
				b1 = GNN_index[:,0].astype(int)
				b2 = GNN_index[:,1].astype(int)
				b1 = b1.tolist()
				b2 = b2.tolist()
				if len(b1) > 1:
					GNN_class = bool(GNNmodel(b1,b2))
				else:
					GNN_class = True
				GNN_end = timer()
				GNN_consume += GNN_end-GNN_start
				# print('Time lapse for GNN: ',end - start,'s.')
			else:
				GNN_class = True
						
			################# GNN end #################
			
			################# IDAES start #################
			if GNN_class == True and pass_pre_screening == True and episode_done == True:
				i_idaes += 1

				list_observation = list(observation_)
				if list_observation in obs_runIdaes:
					# print('\nEpisode with a repeated flowsheet: ', episode, ', i_step: ', i_step, ', i_idaes: ', i_idaes)
					reward = obs_runIdaes_reward[obs_runIdaes.index(list_observation)]
					IDAES_status = obs_runIdaes_status[obs_runIdaes.index(list_observation)]
					IDAES_costing = obs_runIdaes_costing[obs_runIdaes.index(list_observation)]
				else:
					i_idaes_unique += 1
					IDAES_start = timer()

					print('\nEpisode for running idaes: ', episode, ', i_step: ', i_step, ', i_idaes: ', i_idaes, ', run Idaes: ', i_idaes_unique)
					flowsheet_name = 'flowsheet_'+str(episode)+'_'+str(i_step)
					list_unit, list_inlet, list_outlet = convertobs2list(observation_, user_inputs)
					reward, IDAES_status, IDAES_costing = HDA.run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet)

					obs_runIdaes.append(list(observation_))
					obs_runIdaes_reward.append(reward)
					obs_runIdaes_status.append(IDAES_status)
					obs_runIdaes_costing.append(IDAES_costing)

					if reward >= 1000:
						i_feasible_unique += 1
						dif_obs.append(list(observation_))
					
					IDAES_end = timer()
					IDAES_consume += IDAES_end-IDAES_start

			if GNN_class == True and pass_pre_screening == False and episode_done == True:
				if len(obs_fail_pre_screening) < 5000 and reward > -1000:
					obs_fail_pre_screening.append(list(observation_))
					obs_fail_pre_screening_reward.append(reward)
					temp_flag = False

			################# IDAES end #################
			
			reward_hist.append(reward) # debug
			tt_reward=tt_reward+reward
			RL.store_transition(observation, action, reward, observation_)
			
			if (i_step > threshold_learn) and episode_done:
				RL.learn()
			
			# swap observation
			observation = np.copy(observation_)
			
			# break while loop when end of this episode
			if episode_done:
				break
				
			i_step += 1
			
		f_rec.flush()
        
		if len(obs_fail_pre_screening) == 5000 and temp_flag == False: 
			# convert list to numpy array and save
			tmp_array = np.array(obs_fail_pre_screening)
			fmt = ",".join(["%d"] * len(tmp_array[0]))
			np.savetxt(dir_result+'obs_fail_pre_screening_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
			tmp_array = np.array(obs_fail_pre_screening_reward)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'obs_fail_pre_screening_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			temp_flag = True

		if episode % 1000 == 0: 
			print("Episode ", episode, " Reward ", tt_reward)
			print("\tepsilon=========   ",RL.epsilon)

		if episode % 10000 == 0 and len(obs_runIdaes)>0: 
			# convert list to numpy array and save
			tmp_array = np.array(obs_runIdaes)
			fmt = ",".join(["%d"] * len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_reward)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'obs_runIdaes_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_status)
			tmp_array = np.array(tmp_array[:,1:], dtype=np.float)
			fmt = ",".join(["%f"]*len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_status_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_costing)
			fmt = ",".join(["%f"]*len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_costing_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')

		if episode>Episode_max-500:
			new_obs.append(observation)
			new_obs_reward.append(reward)
			new_obs_status.append(IDAES_status)
			new_obs_costing.append(IDAES_costing)

		tt_reward_hist.append(tt_reward)
		i_idaes_unique_hist.append(i_idaes_unique)
		i_feasible_unique_hist.append(i_feasible_unique)

		if episode % ma_horizon == 0 and episode > 0:
			ma_length = len(tt_reward_hist)
			ma_reward_hist.append(sum(tt_reward_hist[ma_length-ma_horizon:ma_length])/ma_horizon)   
			
	# plot moving average reward
	plt.plot(np.arange(len(ma_reward_hist))*ma_horizon, ma_reward_hist)
	plt.ylabel('MA_Reward')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_Reward_'+str(model_index)+'.png')
	plt.show()

	# plot total rewards
	plt.plot(np.arange(len(tt_reward_hist)), tt_reward_hist)
	plt.ylabel('Reward')
	plt.xlabel('training steps')
	plt.savefig('./result/Reward_'+str(model_index)+'.png')
	plt.show()

	# plot reward hist (debug)
	plt.plot(np.arange(len(reward_hist)), reward_hist)
	plt.ylabel('Reward')
	plt.xlabel('count')
	plt.show()

	# plot unique # of IDAES and feasible cases
	plt.plot(np.arange(len(i_idaes_unique_hist)), i_idaes_unique_hist)
	plt.ylabel('Unique cases sent to IDAES')
	plt.xlabel('training steps')
	plt.savefig('./result/IDAES_unique_'+str(model_index)+'.png')
	plt.show()

	plt.plot(np.arange(len(i_feasible_unique_hist)), i_feasible_unique_hist)
	plt.ylabel('Unique feasible cases')
	plt.xlabel('training steps')
	plt.savefig('./result/IDAES_unique_feasible_'+str(model_index)+'.png')
	plt.show()

	#calculate num of inlets and outlets
	if len(dif_obs)>0:
		dif_obs =np.array(dif_obs)
		dim1,dim2 = dif_obs.shape
		inlet_num=[]
		outlet_num=[]
		for i in range(dim1):
			temp_matrix = np.reshape(dif_obs[i,:],(n_inlets,n_outlets))
			inlet_count = 0
			outlet_count = 0
			for i in range(n_inlets):
				if np.sum(temp_matrix[i,:]) == 1:
					inlet_count += 1
			for i in range(n_outlets):  
				if np.sum(temp_matrix[:,i]) == 1:
					outlet_count += 1 
			inlet_num.append(inlet_count)
			outlet_num.append(outlet_count)
		plt.plot(np.arange(len(inlet_num)), inlet_num)
		plt.ylabel('inlet_num')
		plt.xlabel('dif_num')
		plt.show()
		plt.plot(np.arange(len(outlet_num)), outlet_num)
		plt.ylabel('outlet_num')
		plt.xlabel('dif_num')
		plt.show()

	# save last 500 observations
	fmt = ",".join(["%d"] * len(new_obs[0]))
	np.savetxt(dir_result+'new_obs_'+str(model_index)+'.csv', np.vstack(new_obs),fmt=fmt,comments='')
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'new_obs_reward_'+str(model_index)+'.csv',new_obs_reward,fmt=fmt,comments='')
	tmp_array = np.array(new_obs_status)
	tmp_array = np.array(tmp_array[:,1:], dtype=np.float)
	np.savetxt(dir_result+'new_obs_status_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	tmp_array = np.array(new_obs_costing)
	np.savetxt(dir_result+'new_obs_status_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	
	# save rewards
	tmp_array = np.array(tt_reward_hist)
	np.savetxt(dir_result+'training_tt_reward_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	tmp_array = np.array(reward_hist)
	np.savetxt(dir_result+'training_reward_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')

	# end of game
	print('Game Over')

	RL_end = timer()
	print('Total time consuming: ', (RL_end-RL_start)/3600, ' hr')
	print('IDAES time consuming: ', IDAES_consume/3600, ' hr')


#%%
if __name__ == "__main__":
	
	# Model save and restore option
	flag_model_save = True
	flag_model_read = True

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
	
	user_inputs = user_inputs()
	Episode_max = 300_000 # refered as global variable if not defined in run_RL
	threshold_learn = -1 # refered as global variable if not defined in run_RL
	model_index = 1 # refered as global variable if not defined in run_RL
	GNN_enable = False # refered as global variable if not defined in run_RL

	# Set up Environment
	dir_result = './result/'
	if not os.path.exists(dir_result):
		os.makedirs(dir_result) 
	f_rec  = open(dir_result+'record_test.out', 'a+')

	n_inlets = len(user_inputs.list_inlet_all) # n_rows
	n_outlets = len(user_inputs.list_outlet_all) # n_cols
	n_features = n_inlets*n_outlets
	env = fs_gen(n_inlets, n_outlets, n_features)

	RL = DeepQNetwork(env.n_actions, env.n_features,
					  learning_rate=0.01 , #0.01,
					  reward_decay=0.5,
					  e_greedy=1,
					  replace_target_iter=200,
					  memory_size=2000,
					  e_greedy_increment=5e-6,
					  randomness = 0 
					  )

	# Load GNN model
	if GNN_enable == True:
		new_model = tf.keras.models.load_model('dgcn')

	# Call RL trianing
	run_RL(env,f_rec)   

	# Save model or restore model if interupted
	save_models_to = dir_result + 'save_model/'
	if flag_model_save:
		RL.saver.save(RL.sess, save_models_to +"model.ckpt")
	else:
		if flag_model_read:
			RL.saver.restore(RL.sess,save_models_to+"model.ckpt")
			print("here==========================================")
			run_RL(env,f_rec)
			#RL.saver.save(RL.sess, save_models_to +"model.ckpt")
	RL.plot_cost(dir_result,model_index)