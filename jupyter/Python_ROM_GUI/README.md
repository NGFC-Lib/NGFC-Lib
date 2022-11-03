# Python_ROM_GUI
The universal ROM GUI can be used for SOFC-MP simulation:  
	1. To be able to use this ROM GUI, the user need to download the "pySOFC.ipynb", "pySOFC.py" and the "source" folder to the same directory.  
	2. The user can install essential packages to the Jupyter Notebook with "InstallPackages.ipynb", and also the "windows subsystem for linux (WSL)" if the user plans to run simulations on PC.

Follow the step-by-step instructions in the Jupyter Notebook "pySOFC":
	1. The user can prepare simulation cases for either "SOFC stack only" case or "NGFC/IGFC system" case. 
	2. In the next step, the user can prefer to run the simulations on the local machine or on the high-performance computing cluster.  
	3. With the simulation results, the user can choose to build the ROMs based on either the "Kriging method" or the "DNN method".

Two examples attahced here:
	1. The first example is a "SOFC stack only" case. The user prepares the folder "Test_NW" that contains the base input file ("base.dat") and "VoltageOnCurrent.dat" file.  An example routine is given in Step 1, Step 2a, Step 3a and Step 4a and Step 4b in the ROM GUI.
	2. The other example is a "NGFC/IGFC system" case.  The user prepares the folder "Test_WW" taht contains the base input file ("input000.dat") and "VoltageOnCurrent.dat" file.  An example routin is given in Step 1, Step 2a, Step 3b in the ROM GUI.  Surely, the user can also use the Step 3a to run the simulations and Step 4a or 4b to build the ROMs.
