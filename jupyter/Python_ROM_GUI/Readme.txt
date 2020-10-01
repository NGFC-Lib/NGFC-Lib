The universal ROM GUI can be used for SOFC-MP simulation:  
	1. For using this ROM GUI, the user needs to download the "pySOFC_v12_Release_v1.ipynb", "pySOFC_v12_Release_v1.py" and the "source" folder to the same directory.  
	2. The user also needs to install essential packages to the Jupyter Notebook, and also the windows subsystem for linux if the user plans to run simulations on Windows PC.
	3. An instruction to installing Windows subsystem for Linux is "WSL_Instruction.pdf".

Follow the step-by-step instructions in the Jupyter Notebook:
	1. The user can prepare simulation cases for either "SOFC stack only" case or "NGFC/IGFC system" case. 
	2. In the next step, the user can choose running the simulations on the local desktop PC or on the remote high-performance computing (HPC) cluster.  
	3. With the simulation results, the user can choose to build the ROMs based on either the "Kriging method" or the "DNN method".

Two examples are provided in folder "Test_NW" and "Test_WW":
	1. The first example "Test_NW" is a "SOFC stack only" case. The user prepares the folder "Test_NW" that contains the base input file ("base.dat") and "VoltageOnCurrent.dat" file.  An example routine is given in Step 1, Step 2a, Step 3a and Step 4a and Step 4b in the ROM GUI.
	2. The second example "Test_WW" is a "NGFC/IGFC system" case.  The user prepares the folder "Test_WW" that contains the base input file ("input000.dat") and "VoltageOnCurrent.dat" file.  An example routine is given in Step 1, Step 2a, Step 3b in the ROM GUI.  Surely, the user can also use the Step 3a to run the simulations and Step 4a or 4b to build the ROMs.
