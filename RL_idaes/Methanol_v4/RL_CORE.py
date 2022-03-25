#%%
from RL_ENV import fs_gen, convertobs2list
from RL_DQN import DeepQNetwork
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from timeit import default_timer as timer
# from GNN_v1 import RL2GNN_Link, GNNmodel
import MET_IDAES_v11_v1 as IDAES

#%%
def RL_run(env,RL,GNN,user_inputs,f_rec,P):

	# check time consuming for training and running IDAES
	RL_start = timer()
	IDAES_consume = 0
	GNN_consume = 0
	NN_consume = 0

	tt_reward_hist=[]
	reward_hist=[] # debug
	new_obs=[]
	new_obs_reward = []
	new_obs_status = []
	new_obs_costing = []
	obs_runIdaes = []
	obs_runIdaes_reward = []
	obs_runIdaes_status = []
	obs_runIdaes_costing = []
	
	IDAES_status = []
	IDAES_costing = []

	i_idaes = 0
	i_idaes_unique = 0
	i_feasible_unique = 0
	i_idaes_unique_hist = []
	i_feasible_unique_hist = []
	ma_horizon = 2000
	ma_reward_hist = []
	
	# Local parameters.
	Episode_max = int(P['Episode_max'] + P['Additional_step'])
	GNN_enable = P['GNN_enable']
	model_index = P['model_index']
	threshold_learn = P['threshold_learn']
	dir_result = './'+'result/'

	for episode in range(Episode_max):

		observation, picked_true_fs = env.reset(episode)
		i_step = 0
		tt_reward = 0
		
		while True:
		
			# RL choose action based on observation
			action, actions_value = RL.choose_action(observation)
			
			# RL take action and get next observation and reward
			observation_, reward, episode_done, pass_pre_screening = \
				env.step(action,observation,episode,i_step,user_inputs)
						
			################# GNN start #################
			if GNN_enable:
				GNN_start = timer()
				GNN_index,GNN_label,GNN_matrix = RL2GNN_Link(observation_,user_inputs)
				b1 = GNN_index[:,0].astype(int)
				b2 = GNN_index[:,1].astype(int)
				b1 = b1.tolist()
				b2 = b2.tolist()
				if len(b1) > 1:
					GNN_class = bool(GNNmodel(b1,b2,GNN))
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

					print('\nEpisode: '+str(episode)+'/'+str(Episode_max)+', percent: '+str(round(episode/Episode_max*100,2))+'%',', i_step: ', i_step,
							', i_idaes: ', i_idaes, ', run Idaes: ', i_idaes_unique, flush = True)
					flowsheet_name = 'flowsheet_'+str(episode)+'_'+str(i_step)
					list_unit, list_inlet, list_outlet = convertobs2list(observation_, user_inputs)
					reward, IDAES_status, IDAES_costing = \
						IDAES.run_optimization(flowsheet_name, list_unit, list_inlet, list_outlet)

					obs_runIdaes.append(list(observation_))
					obs_runIdaes_reward.append(reward)
					obs_runIdaes_status.append(IDAES_status)
					obs_runIdaes_costing.append(IDAES_costing)

					if reward >= 1000:
						i_feasible_unique += 1

					IDAES_end = timer()
					IDAES_consume += IDAES_end-IDAES_start

			################# IDAES end #################
			
			reward_hist.append(reward) # debug
			tt_reward=tt_reward+reward
			RL.store_transition(observation, action, reward, episode_done, observation_) #Jie... add "done"
			
			if (i_step > threshold_learn) and episode_done:
				NN_start = timer()
				RL.learn()
				NN_end = timer()
				NN_consume += NN_end-NN_start
			
			# swap observation
			observation = np.copy(observation_)
			
			# break while loop when end of this episode
			if episode_done:
				break
				
			i_step += 1
			
		f_rec.flush()

		if episode % 1000 == 0: 
			print("Episode: "+str(episode)+'/'+str(Episode_max)+", percent: "+str(round(episode/Episode_max*100,2))+'%', ", Reward: ", tt_reward, flush = True)
			print("\tepsilon===", RL.epsilon, flush = True)
			time_now = timer()
			print('\tTime lapse: ', (time_now-RL_start)/3600, ' hr', flush = True)
			print('\tNeural Network consume: ', NN_consume/3600, ' hr', flush = True)
			print('\tIDAES consume: ', IDAES_consume/3600, ' hr', flush = True)
			# print('\tGNN consume: ', GNN_consume/3600, ' hr', flush = True)

		if episode % 10000 == 0 and len(obs_runIdaes)>0: 
			# convert list to numpy array and save
			tmp_array = np.array(obs_runIdaes)
			fmt = ",".join(["%d"] * len(tmp_array[0]))
			np.savetxt(dir_result+'obs_runIdaes_'+str(model_index)+'.csv',np.vstack(tmp_array),fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_reward)
			fmt = ",".join(["%d"])
			np.savetxt(dir_result+'obs_runIdaes_reward_'+str(model_index)+'.csv',tmp_array,fmt=fmt,comments='')
			tmp_array = np.array(obs_runIdaes_status)
			tmp_array = np.array(tmp_array[:,1:], dtype=float)
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
	plt.figure()
	plt.plot(np.arange(len(ma_reward_hist))*ma_horizon, ma_reward_hist)
	plt.ylabel('MA_Reward (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/MA_Reward_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	# plot total rewards
	plt.figure()
	plt.plot(np.arange(len(tt_reward_hist)), tt_reward_hist)
	plt.ylabel('Reward (episode)')
	plt.xlabel('training steps')
	plt.savefig('./result/Reward_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	# plot reward hist (debug)
	plt.figure()
	plt.plot(np.arange(len(reward_hist)), reward_hist)
	plt.ylabel('Reward (step)')
	plt.xlabel('count')
	plt.savefig('./result/Step_Reward_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	# plot unique # of IDAES and feasible cases
	plt.figure()
	plt.plot(np.arange(len(i_idaes_unique_hist)), i_idaes_unique_hist)
	plt.ylabel('Unique cases sent to IDAES')
	plt.xlabel('training steps')
	plt.savefig('./result/IDAES_unique_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	plt.figure()
	plt.plot(np.arange(len(i_feasible_unique_hist)), i_feasible_unique_hist)
	plt.ylabel('Unique feasible cases')
	plt.xlabel('training steps')
	plt.savefig('./result/IDAES_unique_feasible_'+str(model_index)+'.png')
	if P['visualize'] == True:
		plt.show()

	# plot training cost
	RL.plot_cost(dir_result,model_index, P['visualize'])

	# save last 500 observations
	fmt = ",".join(["%d"] * len(new_obs[0]))
	np.savetxt(dir_result+'new_obs_'+str(model_index)+'.csv', np.vstack(new_obs),fmt=fmt,comments='')
	fmt = ",".join(["%d"])
	np.savetxt(dir_result+'new_obs_reward_'+str(model_index)+'.csv',new_obs_reward,fmt=fmt,comments='')
	tmp_array = np.array(new_obs_status)
	tmp_array = np.array(tmp_array[:,1:], dtype=float)
	np.savetxt(dir_result+'new_obs_status_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	tmp_array = np.array(new_obs_costing)
	np.savetxt(dir_result+'new_obs_status_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	
	# save rewards
	tmp_array = np.array(tt_reward_hist)
	np.savetxt(dir_result+'training_tt_reward_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')
	tmp_array = np.array(reward_hist)
	np.savetxt(dir_result+'training_reward_'+str(model_index)+'.csv',tmp_array,fmt='%10.5f',delimiter=',')

	# save computing cost
	RL_end = timer()
	rtime = np.array((1,2))
	print('Total time consuming: ', (RL_end-RL_start)/3600, ' hr', flush = True)
	print('IDAES time consuming: ', IDAES_consume/3600, ' hr', flush = True)
	rtime = np.array([(RL_end-RL_start)/3600,IDAES_consume/3600])
	np.savetxt(dir_result+'training_time_'+str(model_index)+'.csv',rtime,fmt='%10.5f',delimiter=',')

	# end of game
	print('Game Over')

#%% Calculating the maximum episode.
def calcMaxEpisode(Emax, e_max):
	a = 0
	b = 1
	if e_max >1 or e_max <= 0:
		os.exit("user-provided e_greedy must fall within 0-1")
	while True:
		e = (a+b)/2.0
		if e_max > 0.8:
			E = round((e_max-0.8)/e*100+0.2/e*50+0.2/e*20+0.4/e)
		elif e_max > 0.6:
			E = round((e_max-0.6)/e*50+0.2/e*20+0.4/e)
		elif e_max > 0.4:
			E = round((e_max-0.4)/e*20+0.4/e)
		else:
			E = round(e_max/e)
		if E<Emax:
			b = e
		else:
			a = e
		if abs(E-Emax) < 100:
			break
	return e

#%%
def RL_call(user_inputs, P):

	# Determine episilon increment
	Episode_max = P['Episode_max']
	e_greedy = P['e_greedy']
	if P['Episode_max_mode'].lower() == 'dynamic': 
		P['e_greedy_increment'] = calcMaxEpisode(Episode_max, e_greedy)
		print("Calculated epsilon increment: " + str(P['e_greedy_increment']), flush = True)

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
						n_inlets,
						n_outlets,
						learning_rate=P['learning_rate'],
						reward_decay=P['reward_decay'],
						e_greedy=P['e_greedy'],
						replace_target_iter=P['replace_target_iter'],
						memory_size=P['memory_size'],
						batch_size=P['batch_size'],
						e_greedy_increment=P['e_greedy_increment'],
						randomness = 0 
						)

	# Load GNN model
	GNN = None
	if P['GNN_enable'] == True:
		GNN = tf.keras.models.load_model('dgcn')

	# Save model or restore model if interupted
	save_models_to = dir_result + 'save_model/'
	if not P['model_restore']:
		print("==================== train RL-GNN model ====================")
		RL_run(env,RL,GNN,user_inputs,f_rec,P)
		if P['model_save']: 
			RL.saver.save(RL.sess, save_models_to +'model_'+str(P['model_index'])+'.ckpt')
	else:
		RL.saver.restore(RL.sess,save_models_to +'model_'+str(P['model_index'])+'.ckpt')
		print("==================== load saved model ====================")
		RL_run(env,RL,GNN,user_inputs,f_rec,P)
		if P['model_save']: 
			RL.saver.save(RL.sess, save_models_to +'model_'+str(P['model_index'])+'_restore.ckpt')