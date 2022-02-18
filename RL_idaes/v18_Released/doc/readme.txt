v13: (a merged version from All group members)
1. save feasible IDAES flowsheet
2. correct physics constraint 1
3. rewrite physics constraint 9
4. add physics constraints 10 and 11
5. add pyomo constraints of purity and flow rates in the function "add_HDA_bounds"

v14:
1. Avoid repeatedly running the same flowsheet
2. Save feasible cases every 1000 episodes, restart mechanism
3. add GNN classification to "run_this"

v14-FullUnitPool:
1. break the inlet_feed and outlet_product to 6 units
2. rewrite physics constraints

v15:
RL_brain: plot with model_index
HDA_IDAES_v6: won't take "maxIterations" as 1000 case
run_this: time consuming, plot and save with model_index

v16:
1. fix the bug: always all the inlets must be seleted
RL_brain: action from -1 to 8
fs_env: only if action>-0, modifiy the observation

v17:
1. structure change
run_this: moving average of rewards
fs_env: moving pre_screening from HDA_IDAES to fs_env
		as convertobs2list and pre-screening
	add physics constraint 0: avoid repeated connections (in pre-screening)
	add penalty as unit # increase (in pre-screening)
	add physics constraint 8: splitter cannot connect to mixer completely
HDA_IDAES: remove prescreening, start from 500
2. reward change
	penalty in pre_screening increased by 1.5 to 2 times
3. fixed the bug
	increase action space length from 9 to 10
	n_actions = 10
	n_features still 81
	update_env in fs_env: only update obs when S < self.num_elements
4. HDA_v7 according to HDA_v7_example_v2
	post-calculation of costings


