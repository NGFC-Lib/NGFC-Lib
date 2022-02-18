import os
import numpy as np
import pandas as pd
import tensorflow as tf
from stellargraph.mapper import PaddedGraphGenerator
from stellargraph import StellarGraph
from timeit import default_timer as timer

#%% Converting RL output data to GNN format.
def convert_Mconn2Munit(list_unit_all, list_inlet_all, list_outlet_all, matrix_conn):
	matrix_conn_unit = np.zeros((len(list_unit_all), len(list_unit_all)), dtype=int)
	
	# fill unit connection matrix
	for k in range(len(list_outlet_all)):
		str_unit_out, str_outlet = [x.strip() for x in list_outlet_all[k].split(".")]
		for l in range(len(list_inlet_all)):
			if matrix_conn[l][k] == 1:
				str_unit_in, str_inlet = [x.strip() for x in list_inlet_all[l].split(".")]
				index_1 = list_unit_all.index(str_unit_in)
				index_2 = list_unit_all.index(str_unit_out)
				matrix_conn_unit[index_1, index_2] = 1
				matrix_conn_unit[index_2, index_1] = 1
				
	return matrix_conn_unit

def convert_list2matrix(list_unit_all, list_inlet_all, list_outlet_all, list_unit, list_inlet, list_outlet):
	matrix_comb = np.zeros((1, len(list_unit_all), 1), dtype=int)
	matrix_conn = np.zeros((len(list_inlet_all), len(list_outlet_all), 1), dtype=int)
	
	# fill unit combination
	for k in range(len(list_unit_all)):
		if list_unit_all[k] in list_unit:
			matrix_comb[:,k, 0] = 1
			
	# fill connection matrix
	for k in range(len(list_inlet)):
		index_inlet = list_inlet_all.index(list_inlet[k])
		index_outlet = list_outlet_all.index(list_outlet[k])
		matrix_conn[index_inlet, index_outlet, 0] = 1
		
	return matrix_comb, matrix_conn

def RL2GNN_Link(RL_input,user_inputs):
	unit = user_inputs.list_unit_all
	inlet = user_inputs.list_inlet_all
	outlet = user_inputs.list_outlet_all
	
	nc = RL_input.shape
	if RL_input.ndim == 1:
		RL = np.zeros((1,nc[0]))
		RL[0:] = RL_input
	else:
		RL = RL_input
		
	nu = len(unit)
	ni = len(inlet)
	no = len(outlet)
	nc = RL.shape
	matrix_label_all = np.zeros((nu,nc[0]), dtype=int)
	matrix_conn_unit_all = np.zeros((nu,nu, nc[0]), dtype=int)
	for i in range(nc[0]):
		matrix_conn = np.reshape(RL[i, :], (ni,no))
		matrix_conn_unit = convert_Mconn2Munit(unit, inlet, outlet, matrix_conn)
		matrix_label = np.array(range(nu))
		matrix_label_all[:, i] = matrix_label
		matrix_conn_unit_all[:, :, i] = matrix_conn_unit
		
	if nc[0] != 1:
		print('Observation size is larger than 1.')
	else:
		matrix_conn_unit_all = matrix_conn_unit_all[:,:,0]
		matrix_label_all = matrix_label_all[:,0]
		
	indices_tmp = list(np.where(np.triu(matrix_conn_unit_all)))
	indices = np.zeros((len(indices_tmp[0]),2))
	indices[:,0] = indices_tmp[0] + 1
	indices[:,1] = indices_tmp[1] + 1
		
	return indices,matrix_label_all,matrix_conn_unit_all

#%% GNN
def GNNmodel(b1,b2,new_model): 
    c0=b1+b2
    c1=sorted(set(c0),key=c0.index)
    s_edges = pd.DataFrame(
    {"source":b1,"target":b2}
    )
    #print(s_edges)
    s_node_feature = pd.DataFrame({"x1":c1,"x2":c1,"x3":c1,"x4":c1,"x5":c1,"x6":c1,"x7":c1,"x8":c1,"x9":c1,"x10":c1},index=c1)
    #print(s_node_feature)
    graph1 = StellarGraph(s_node_feature, s_edges)
    graphlist1 = []
    graphlist1.append(graph1)
    #print(graphlist1[0].info())
    # new_model = tf.keras.models.load_model('dgcn')
    generator = PaddedGraphGenerator(graphs=graphlist1)
    test_gen = generator.flow(list(range(0,1)),targets=None)
    predictions = new_model.predict(test_gen)
    #print(predictions)
    classes = np.rint(predictions)
    graphlist1.clear()

    #print(classes)
    return classes
