
##############################################################################
# The development of this flowsheet/code is funded by the ARPA-E DIFFERENTIATE project: 
# “Machine Learning for Natural Gas to Electric Power System Design” 
# Project number: DE-FOA-0002107-1625. 
# This project is a collaborative effort between the Pacific Northwest National Laboratory, 
# National Energy Technology Laboratory, and Washington University.
##############################################################################

import numpy as np
import numpy.linalg as la
import numpy.ma as ma
from numpy import array
from scipy import stats
import pandas as pd
import ipywidgets
import paramiko
import pysftp
import shutil
import getpass
import imp
import math
import sys
import copy
import os
import time
from datetime import timedelta
from smt.sampling_methods import LHS
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
plt.rcParams.update({'font.size': 30})
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import TensorBoard
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 

def sshCommand(hostname, port, username, password, command):
    sshClient = paramiko.SSHClient()                                   # create SSHClient instance
    sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
    sshClient.load_system_host_keys()
    sshClient.connect(hostname, port, username, password)
    stdin, stdout, stderr = sshClient.exec_command(command)
    for line in stdout:
        print(line.strip('\n'))

def put_r_windows(sftp, localdir, remotedir, preserve_mtime = False):
    for entry in os.listdir(localdir):
        remotepath = remotedir + "/" + entry
        localpath = os.path.join(localdir, entry)
        if not os.path.isfile(localpath):
            try:
                sftp.mkdir(remotepath)
            except OSError:     
                pass
            put_r_windows(sftp, localpath, remotepath, preserve_mtime)
        else:
            sftp.put(localpath, remotepath, preserve_mtime=preserve_mtime) 

def query_yes_no(question, default = None):
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

def dos2unix(file_path):
    # replacement strings
    WINDOWS_LINE_ENDING = b'\r\n'
    UNIX_LINE_ENDING = b'\n'

    with open(file_path, 'rb') as open_file:
        content = open_file.read()
    content = content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)
    with open(file_path, 'wb') as open_file:
        open_file.write(content)

def variable_options(display = False):
    names = [
        "Average_CellVoltage",
        "Average_CurrentDensity",
        "BackEnvironmentT",
        "BottomEnvironmentT",
        "CellFuelFlowRate",
        "CellOxidantFlowRate",
        "FrontEnvironmentT",
        "Fuel_Utilization",
        "FuelH2",
        "FuelH2O",
        "FuelCO",
        "FuelCO2",
        "FuelCH4",
        "FuelN2",
        "FuelTemperature",
        "FuelTOnTop",
        "FuelRecyclePercent",
        "FuelHTXEffectiveness",
        "FuelNGTemperature",
        "FuelNGHTXDeltaT",
        "Internal_Reforming",
        "nCells",
        "Oxidant_Recirculation",
        "OxidantRecyclePercent",
        "OxygenToCarbon_Ratio",
        "OxidantO2",
        "OxidantN2",
        "OxidantH2O",
        "OxidantCO2",
        "OxidantAr",
        "OxidantTemperature",
        "OxidantTOnTop",
        "PreReform",
        "SideEnvironmentT",
        "Simulation_Option",
        "Stack_Fuel_Utilization",
        "Stack_Oxidant_Utilization",
        "StackFuelFlowRate",
        "StackFuelFlowRateH2O",
        "StackFuelFlowRateCO",
        "StackFuelFlowRateCO2",
        "StackFuelFlowRateCH4",
        "StackFuelFlowRateH2",
        "StackFuelFlowRateN2",
        "StackOxidantFlowRate",
        "StackOxidantFlowRateO2",
        "StackOxidantFlowRateN2",
        "StackOxidantFlowRateH2O",
        "StackOxidantFlowRateCO2",
        "StackOxidantFlowRateAr",
        "StackVoltage",
        "SystemPressure",
        "TopEnvironmentT",
        "VGRRate",
        "VGRTemperature",
        "VGRH2OPassRate",
        "VGRH2PassRate",
        "VGRCO2CaptureRate",
        "VGRCOConvertRate"
    ]
    
    units = [
        "V",
        "A/m^2",
        "C",
        "C",
        "mol/s",
        "mol/s",
        "C",
        "-",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "C",
        "C",
        "%",
        "-",
        "C",
        "C",
        "-",
        "-",
        "-",
        "%",
        "-",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "C",
        "C",
        "-",
        "C",
        "-",
        "-",
        "-",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "mol/s",
        "V",
        "atm",
        "C",
        "-",
        "C",
        "-",
        "-",
        "-",
        "-"
    ]
    
    if display == True:
        print('Options of input variable:')
        for i in range(len(names)):
            print(i+1, ':', names[i]+', ['+units[i]+']', end = '\t\n')
    return names, units
        
class sys_preprocessor():  
    def NGFC_ccs(self, J,FU,AU,OCR,IR,Arec,PreReform,cellsize):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (NG)
        NG_fin[Index_H2O] = 0
        NG_fin[Index_Ar] = 0
        NG_fin[Index_CO2] = 74.0729157
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 118.516665
        NG_fin[Index_CH4] = 6896.18846
        NG_fin[Index_CO] = 0
        NG_fin[Index_H2] = 0
        NG_fin[Index_C2H6] = 237.03333
        NG_fin[Index_C3H8] = 51.851041
        NG_fin[Index_C4H10] = 29.6291663
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                    stack_mix[i] = stack_fin[i] + stack_recirc[i] #;


                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert


        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        Frec = CalcR #; //they do equal
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (lb-mol/hr)",pref_CH4)
        # print("Air cell outlet (U) (lb-mol/hr)",cell_aexit)
        # print("Fuel cell outlet (Q) (lb-mol/hr)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        #return(SOFC_Ain,stack_ain,stack_fin*Const_Convert,stack_recirc,stack_mix,pref_CH4,cell_exit,Frec,succs)
        #return(stack_fin,stack_ain/Const_Convert,Frec,succs)
        return(stack_fin,SOFC_Ain,Fresh_Ain,Frec,succs)

    def NGFC_nocc(self, J,FU,AU,OCR,IR,Arec,PreReform,cellsize):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (NG)
        NG_fin[Index_H2O] = 0
        NG_fin[Index_Ar] = 0
        NG_fin[Index_CO2] = 74.0729157
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 118.516665
        NG_fin[Index_CH4] = 6896.18846
        NG_fin[Index_CO] = 0
        NG_fin[Index_H2] = 0
        NG_fin[Index_C2H6] = 237.03333
        NG_fin[Index_C3H8] = 51.851041
        NG_fin[Index_C4H10] = 29.6291663
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 1
        splt_ain[Index_Ar] = 1
        splt_ain[Index_CO2] = 1
        splt_ain[Index_O2] = 1
        splt_ain[Index_N2] = 1
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                    stack_mix[i] = stack_fin[i] + stack_recirc[i] #;


                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert


        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        Frec = CalcR #; //they do equal
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (lb-mol/hr)",pref_CH4)
        # print("Air cell outlet (U) (lb-mol/hr)",cell_aexit)
        # print("Fuel cell outlet (Q) (lb-mol/hr)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        # return(stack_ain/Const_Convert,stack_fin,Frec,succs)      
        return(stack_fin, SOFC_Ain, Fresh_Ain, Frec, succs)

    def IGFC_ccs(self, J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (igfc) default conventional
        NG_fin[Index_H2O] = 0.0013
        NG_fin[Index_Ar] = 0.0008
        NG_fin[Index_CO2] = 0.2043
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 0.006
        NG_fin[Index_CH4] = 0.0583
        NG_fin[Index_CO] = 0.3774
        NG_fin[Index_H2] = 0.3519
        NG_fin[Index_C2H6] = 0.0
        NG_fin[Index_C3H8] = 0.0
        NG_fin[Index_C4H10] = 0.0
        if igfc=='conventional':
            NG_fin[Index_H2O] = 0.0013
            NG_fin[Index_Ar] = 0.0008
            NG_fin[Index_CO2] = 0.2043
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.006
            NG_fin[Index_CH4] = 0.0583
            NG_fin[Index_CO] = 0.3774
            NG_fin[Index_H2] = 0.3519
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='enhanced':
            NG_fin[Index_H2O] = 0.0006
            NG_fin[Index_Ar] = 0.0009
            NG_fin[Index_CO2] = 0.2423
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0064
            NG_fin[Index_CH4] = 0.1022
            NG_fin[Index_CO] = 0.3415
            NG_fin[Index_H2] = 0.3062
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='catalytic':
            NG_fin[Index_H2O] = 0.0004
            NG_fin[Index_Ar] = 0.0003
            NG_fin[Index_CO2] = 0.3465
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0069
            NG_fin[Index_CH4] = 0.3159
            NG_fin[Index_CO] = 0.0914
            NG_fin[Index_H2] = 0.2386
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        IR = 1.0
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                    stack_mix[i] = stack_fin[i] + stack_recirc[i] #;


                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert


        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        Frec = CalcR #; //they do equal
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (lb-mol/hr)",pref_CH4)
        # print("Air cell outlet (U) (lb-mol/hr)",cell_aexit)
        # print("Fuel cell outlet (Q) (lb-mol/hr)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        # return(stack_fin,stack_ain/Const_Convert,Frec,succs)
        return(stack_fin, SOFC_Ain, Fresh_Ain, Frec, succs)
 
    def NGFC_ccs_vgr(self, J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)

        recirc_VGR0 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR1 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR2 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR3 = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (NG)
        NG_fin[Index_H2O] = 0
        NG_fin[Index_Ar] = 0
        NG_fin[Index_CO2] = 74.0729157
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 118.516665
        NG_fin[Index_CH4] = 6896.18846
        NG_fin[Index_CO] = 0
        NG_fin[Index_H2] = 0
        NG_fin[Index_C2H6] = 237.03333
        NG_fin[Index_C3H8] = 51.851041
        NG_fin[Index_C4H10] = 29.6291663
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                   # stack_mix[i] = stack_fin[i] + stack_recirc[i] #;
                    recirc_VGR3[i]=stack_fin[i]*0.05

                for i in range(Nspecies):
                    stack_mix[i]=stack_fin[i]+stack_recirc[i]+recirc_VGR3[i]
                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]+recirc_VGR3[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            #cell_ref[Index_H2O] = pref_CH4[Index_H2O]-pref_CH4[Index_CH4]-2*pref_CH4[Index_C2H6]-3*pref_CH4[Index_C3H8]-4*pref_CH4[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7a) Calculate the new VGR recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------  
            for i in range(Nspecies):
                recirc_VGR0[i]=cell_exit[i]-stack_recirc[i]
                recirc_VGR1[i]=recirc_VGR0[i]
            WGSmol=WGS*recirc_VGR1[Index_CO]
            recirc_VGR1[Index_H2O] = recirc_VGR1[Index_H2O] - WGSmol
            recirc_VGR1[Index_CO2] = recirc_VGR1[Index_CO2] + WGSmol
            recirc_VGR1[Index_CO] = recirc_VGR1[Index_CO] - WGSmol
            recirc_VGR1[Index_H2] = recirc_VGR1[Index_H2] + WGSmol
            for i in range(Nspecies):
                recirc_VGR2[i]=recirc_VGR1[i]
            VGRH2O=recirc_VGR1[Index_H2O]*H2OCap
            VGRCO2=recirc_VGR1[Index_CO2]*CO2Cap
            VGRH2=recirc_VGR1[Index_H2]*H2Cap
            recirc_VGR2[Index_H2O]=recirc_VGR2[Index_H2O]-VGRH2O
            recirc_VGR2[Index_CO2]=recirc_VGR2[Index_CO2]-VGRCO2
            recirc_VGR2[Index_H2]=recirc_VGR2[Index_H2]-VGRH2
            for i in range(Nspecies):
                recirc_VGR3[i]=recirc_VGR2[i]*VGR

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert


        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        #Frec = CalcR #; //they do equal //not working for VGR
        CalcR=Frec
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (lb-mol/hr)",pref_CH4)
        # print("Air cell outlet (U) (lb-mol/hr)",cell_aexit)
        # print("Fuel cell outlet (Q) (lb-mol/hr)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        # return(stack_fin,stack_ain/Const_Convert,Frec,succs)
        return(stack_fin, SOFC_Ain, Fresh_Ain, Frec, succs)

    def IGFC_ccs_vgr(self, J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)

        recirc_VGR0 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR1 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR2 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR3 = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (igfc) default conventional
        NG_fin[Index_H2O] = 0.0013
        NG_fin[Index_Ar] = 0.0008
        NG_fin[Index_CO2] = 0.2043
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 0.006
        NG_fin[Index_CH4] = 0.0583
        NG_fin[Index_CO] = 0.3774
        NG_fin[Index_H2] = 0.3519
        NG_fin[Index_C2H6] = 0.0
        NG_fin[Index_C3H8] = 0.0
        NG_fin[Index_C4H10] = 0.0
        if igfc=='conventional':
            NG_fin[Index_H2O] = 0.0013
            NG_fin[Index_Ar] = 0.0008
            NG_fin[Index_CO2] = 0.2043
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.006
            NG_fin[Index_CH4] = 0.0583
            NG_fin[Index_CO] = 0.3774
            NG_fin[Index_H2] = 0.3519
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='enhanced':
            NG_fin[Index_H2O] = 0.0006
            NG_fin[Index_Ar] = 0.0009
            NG_fin[Index_CO2] = 0.2423
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0064
            NG_fin[Index_CH4] = 0.1022
            NG_fin[Index_CO] = 0.3415
            NG_fin[Index_H2] = 0.3062
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='catalytic':
            NG_fin[Index_H2O] = 0.0004
            NG_fin[Index_Ar] = 0.0003
            NG_fin[Index_CO2] = 0.3465
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0069
            NG_fin[Index_CH4] = 0.3159
            NG_fin[Index_CO] = 0.0914
            NG_fin[Index_H2] = 0.2386
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        IR = 1.0
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                   # stack_mix[i] = stack_fin[i] + stack_recirc[i] #;
                    recirc_VGR3[i]=stack_fin[i]*0.05

                for i in range(Nspecies):
                    stack_mix[i]=stack_fin[i]+stack_recirc[i]+recirc_VGR3[i]
                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]+recirc_VGR3[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            # cell_ref[Index_H2O] = pref_CH4[Index_H2O]-pref_CH4[Index_CH4]-2*pref_CH4[Index_C2H6]-3*pref_CH4[Index_C3H8]-4*pref_CH4[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7a) Calculate the new VGR recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------  
            for i in range(Nspecies):
                recirc_VGR0[i]=cell_exit[i]-stack_recirc[i]
                recirc_VGR1[i]=recirc_VGR0[i]
            WGSmol=WGS*recirc_VGR1[Index_CO]
            recirc_VGR1[Index_H2O] = recirc_VGR1[Index_H2O] - WGSmol
            recirc_VGR1[Index_CO2] = recirc_VGR1[Index_CO2] + WGSmol
            recirc_VGR1[Index_CO] = recirc_VGR1[Index_CO] - WGSmol
            recirc_VGR1[Index_H2] = recirc_VGR1[Index_H2] + WGSmol
            for i in range(Nspecies):
                recirc_VGR2[i]=recirc_VGR1[i]
            VGRH2O=recirc_VGR1[Index_H2O]*H2OCap
            VGRCO2=recirc_VGR1[Index_CO2]*CO2Cap
            VGRH2=recirc_VGR1[Index_H2]*H2Cap
            recirc_VGR2[Index_H2O]=recirc_VGR2[Index_H2O]-VGRH2O
            recirc_VGR2[Index_CO2]=recirc_VGR2[Index_CO2]-VGRCO2
            recirc_VGR2[Index_H2]=recirc_VGR2[Index_H2]-VGRH2
            for i in range(Nspecies):
                recirc_VGR3[i]=recirc_VGR2[i]*VGR

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert


        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        #Frec = CalcR #; //they do equal //not working for VGR
        CalcR=Frec
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (lb-mol/hr)",pref_CH4)
        # print("Air cell outlet (U) (lb-mol/hr)",cell_aexit)
        # print("Fuel cell outlet (Q) (lb-mol/hr)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        # return(stack_fin,stack_ain/Const_Convert,ref_ain,stack_amix/Const_Convert,Frec,succs)
        return(stack_fin, SOFC_Ain, Fresh_Ain, Frec, succs)
    
def LHSampling(work_path, numvar=None, numsample=None, 
               listvar=None, listmin=None, listmax=None):
    '''
    The function conducts Latin Hypercube Sampling
    '''
    print('############################################################\
            \nConducts Latin Hypercube Sampling\
            \n############################################################')
    
    # Part 0: Input variable options
    nameoptions, unitoptions = variable_options()
    
    # Part 1: creat given.dat
    filename  = work_path+'/given.dat'
    Create_Given = True
    if os.path.exists(filename):
        query = query_yes_no('"given.dat" file already exists on the local machine, do you want to overwrite it?')
        Create_Given = query
    if Create_Given == True:
        if len(listvar) != numvar or len(listmin) != numvar or len(listmax) != numvar:
            sys.exit('Code terminated: the lengths of variables/minimums/maximums not match')
        lines=["", "", "", ""]
        for i in range(numvar):
            lines[0] = lines[0] + nameoptions[listvar[i]-1] + '\t'
            lines[1] = lines[1] + str(listmin[i]) + '\t'
            lines[2] = lines[2] + str(listmax[i]) + '\t'
            lines[3] = lines[3] + str(numsample) + '\t'
        lines[0] += '\n'
        lines[1] += '\n'
        lines[2] += '\n'
        lines[3] += '\n'
        outputfilename = work_path+'/'+'given.dat'
        inp_w=open(outputfilename,"w")
        inp_w.writelines(lines)
        inp_w.close()
        print("Created given.dat")
        
    # Part 2: creat LHS.dat from given.dat
    inputfilename = work_path+'/'+'given.dat'
    outputfilename  = work_path+'/LHS.dat'
    Create_LHS = True
    if os.path.exists(outputfilename):
        query = query_yes_no('"LHS.dat" file already exists on the local machine, do you want to overwrite it?')
        Create_LHS = query
    if Create_LHS == True:
        print('Given vairables and limits:')
        name_tmp = []
        value_tmp = []
        with open(inputfilename) as f:
            i = 0
            for line in f.readlines():
                if i == 0:
                    name_tmp = line.strip().split()
                elif i > 0:
                    linestr = line.strip().split()
                    linenum = [float(lineele) for lineele in linestr]
                    value_tmp.append(linenum)
                i += 1
        # display given.dat
        givenname = name_tmp
        givenvalue = np.array(value_tmp)
        numvar = len(givenname)
        numsample = int(givenvalue[2, 0])
        for i in range(numvar):
            print(i+1, ':', givenname[i], '\n\tMin: ', givenvalue[0, i], '\tMax: ', givenvalue[1, i], 
                  '\t', int(givenvalue[2, i]), ' Samples', end = '\t\n')
        # perform Latin Hypercube sampling
        xlimits = np.transpose(givenvalue[:2, :])
        sampling = LHS(xlimits = xlimits)
        LHSvalue = sampling(numsample)
        # write LHS.dat
        lines = ["#######title########\n"]
        line = "case No."
        for i in range(numvar):
            line = line+"\t"+givenname[i]+'\t'
        line += '\n'
        lines.append(line)
        for i in range(numsample):
            line = str(i+1)+'\t'
            for j in range(numvar):
                line = line+'\t'+"{:.6g}".format(LHSvalue[i, j])+'\t'
            line += '\n'
            lines.append(line)
        inp_w=open(outputfilename,"w")
        inp_w.writelines(lines)
        inp_w.close()
        print("Created LHS.dat")
        print('End of code\n')

def createcases(work_path, source_path, inputbasefilename, 
                preprocessor_enabled = False, preprocessor_name = None, 
                igfc = None):
    '''
    The function creates cases based on LHS.dat
    '''
    print('############################################################\
            \nCreate case folders on the local machine\
            \n############################################################')
    # preprocessor_name:  "NGFC_ccs", "NGFC_nocc", "IGFC_ccs", "NGFC_ccs_vgr", "IGFC_ccs_vgr"
    # igfc:         "conventional", "enhanced", "catalytic"
    
    ## load LHS_file
    name_tmp = []
    value_tmp = []
    filename = work_path+'/LHS.dat'
    with open(filename) as f:
        i = 0
        for line in f.readlines():
            if i == 1:
                name_tmp = line.strip().split()
            elif i > 1:
                linestr = line.strip().split()
                linenum = [float(lineele) for lineele in linestr]
                value_tmp.append(linenum)
            i += 1
    value_tmp = np.array(value_tmp)
    LHSvalue = value_tmp[:,1:]
    Ncase, Nvar = LHSvalue.shape
    len_tmp = len(name_tmp)
    LHSname = np.array(name_tmp[len_tmp-Nvar:len_tmp])
    
    ## create folders and copy essential files
    path_tmp = work_path+'/Cases'
    if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)
    else:
        query = query_yes_no('"cases" folder already exists on the local machine, do you want to overwrite it?')
        if query == False:
            pass
    
    indpreprocessorfailed = []
    for i in range(Ncase):
        path_tmp = work_path+'/Cases/Case'+str(i).zfill(5)
        if not os.path.exists(path_tmp):
            os.mkdir(path_tmp)
        
        filename = 'ButlerVolmer.inp'
        source = source_path+'/'+filename
        target = path_tmp+'/'+filename
        shutil.copy2(source, target)
        filename = 'thermo.lib'
        source = source_path+'/'+filename
        target = path_tmp+'/'+filename
        shutil.copy2(source, target)
        filename = 'trans.lib'
        source = source_path+'/'+filename
        target = path_tmp+'/'+filename
        shutil.copy2(source, target)
        filename = 'VoltageOnCurrent.dat'
        source = work_path+'/'+filename
        target = path_tmp+'/'+filename
        shutil.copy2(source, target)
        
        ## generate romSOFCMP2D4ROM.inp
        outputfilename = path_tmp+'/'+'romSOFCMP2D4ROM.inp'
        lines = ["@model="+inputbasefilename+"\n"]
        for j in range(Nvar):
            line = LHSname[j]+"="+str(LHSvalue[i, j])+"\n"
            lines.append(line)
        inp_base=open(inputbasefilename,"r")
        lines_inp=inp_base.readlines()
        for j in range(len(lines_inp)):
            str00=lines_inp[j].split('=')
            str00[0]=str00[0].rstrip()
            str00[0]=str00[0].lstrip()
        inp_w=open(outputfilename,"w")
        inp_w.writelines(lines)
        inp_w.close()
        
        ## generate sofc4rom.dat
        if preprocessor_enabled == True:
            # load romSOFCMP2D4ROM.inp        
            inputfilename = path_tmp+'/'+'romSOFCMP2D4ROM.inp'
            text_file=open(inputfilename,"r")
            lines = text_file.readlines()
            df0 = pd.DataFrame(np.array([['1a', '1b', '1c']]),columns=['Name', 'Value', 'Called'])
            df1 = pd.DataFrame(columns=['Name', 'Value', 'Called'])
            for j in range(len(lines)):
                if j>0:
                    str01 = lines[j].split('=')
                    str01[0]=str01[0].rstrip()
                    str01[0]=str01[0].lstrip()
                    df0['Name']=str01[0]
                    df0['Value']=float(str01[1])
                    df0['Called']=False
                    df1=pd.concat([df1,df0],sort=False,ignore_index=True)
            
            # load inputbasefilename (base.dat or input000.dat)
            text_file=open(inputbasefilename,"r")
            lines = text_file.readlines()
            df2 = pd.DataFrame(np.array([['1a', '1b', '1c']]),columns=['Name', 'Value', 'Updated'])
            df3 = pd.DataFrame(columns=['Name', 'Value', 'Updated']) # currently, "Updated" feature not active
            for j in range(len(lines)):
                str01 = lines[j].split('=')
                if len(str01) == 2:
                    str01[0]=str01[0].rstrip()
                    str01[0]=str01[0].lstrip()
                    try:
                        df2['Name']=str01[0]
                        df2['Value']=float(str01[1])
                        df2['Updated']=False
                        df3=pd.concat([df3,df2],sort=False,ignore_index=True)
                    except:
                        pass
            
            ## Call "preprocessor" function
            # "preprocessor" input #1
            try:
                J=df1.loc[df1["Name"]=="Average_CurrentDensity","Value"].iloc[0]/10.0 # convert from A/m2 to mA/cm2
                df1.loc[df1["Name"]=="Average_CurrentDensity","Called"]=True
            except:
                try:
                    J=df3.loc[df3["Name"]=="Average_CurrentDensity","Value"].iloc[0]/10.0  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            # "preprocessor" input #2
            try:
                FU=df1.loc[df1["Name"]=="Stack_Fuel_Utilization","Value"].iloc[0]
                df1.loc[df1["Name"]=="Stack_Fuel_Utilization","Called"]=True
            except:
                try:
                     FU=df3.loc[df3["Name"]=="Stack_Fuel_Utilization","Value"].iloc[0] 
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            # "preprocessor" input #3
            try:
                AU=df1.loc[df1["Name"]=="Stack_Oxidant_Utilization","Value"].iloc[0]
                df1.loc[df1["Name"]=="Stack_Oxidant_Utilization","Called"]=True
            except:
                try:
                     AU=df3.loc[df3["Name"]=="Stack_Oxidant_Utilization","Value"].iloc[0]
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            # "preprocessor" input #4
            try:
                OCR=df1.loc[df1["Name"]=="OxygenToCarbon_Ratio","Value"].iloc[0]
                df1.loc[df1["Name"]=="OxygenToCarbon_Ratio","Called"]=True
            except:
                try:
                     OCR=df3.loc[df3["Name"]=="OxygenToCarbon_Ratio","Value"].iloc[0]
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            # "preprocessor" input #5
            try:
                IR=df1.loc[df1["Name"]=="Internal_Reforming","Value"].iloc[0]
                df1.loc[df1["Name"]=="Internal_Reforming","Called"]=True
            except:
                try:
                     IR=df3.loc[df3["Name"]=="Internal_Reforming","Value"].iloc[0]
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            # "preprocessor" input #6
            try:
                Arec=df1.loc[df1["Name"]=="Oxidant_Recirculation","Value"].iloc[0]
                df1.loc[df1["Name"]=="Oxidant_Recirculation","Called"]=True
            except:
                try:
                     Arec=df3.loc[df3["Name"]=="Oxidant_Recirculation","Value"].iloc[0]
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')        
            
            # "preprocessor" input #7
            try:
                PreReform=df1.loc[df1["Name"]=="PreReform","Value"].iloc[0]
                df1.loc[df1["Name"]=="PreReform","Called"]=True
            except:
                try:
                     PreReform=df3.loc[df3["Name"]=="PreReform","Value"].iloc[0]
                except:
                    # print('Warning: "PreReform" not defined, PreReform=0.2')  
                    PreReform=0.2
            # "preprocessor" input #8
            try:
                cellsize=df1.loc[df1["Name"]=="cellsize","Value"].iloc[0]
                df1.loc[df1["Name"]=="cellsize","Called"]=True
            except:
                try:
                     cellsize=df3.loc[df3["Name"]=="cellsize","Value"].iloc[0]
                except:
                    # print('Warning: "cellsize" not defined, cellsize=550.0')  
                    cellsize=550.0 #cm2
            
            if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                # "preprocessor" input #9
                try:
                    VGR=df1.loc[df1["Name"]=="VGRRate","Value"].iloc[0]
                    df1.loc[df1["Name"]=="VGRRate","Called"]=True
                except:
                    try:
                         VGR=df3.loc[df3["Name"]=="VGRRate","Value"].iloc[0]
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')    
                # "preprocessor" input #10
                try:
                    VGRTemperature=df1.loc[df1["Name"]=="VGRTemperature","Value"].iloc[0]
                    df1.loc[df1["Name"]=="VGRTemperature","Called"]=True
                except:
                    try:
                         VGRTemperature=df3.loc[df3["Name"]=="VGRTemperature","Value"].iloc[0]
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')  
                # "preprocessor" input #11
                try:
                    H2OCap=1-df1.loc[df1["Name"]=="VGRH2OPassRate","Value"].iloc[0]
                    df1.loc[df1["Name"]=="VGRH2OPassRate","Called"]=True
                except:
                    try:
                         H2OCap=1-df3.loc[df3["Name"]=="VGRH2OPassRate","Value"].iloc[0]
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')  
                # "preprocessor" input #12
                try:
                    CO2Cap=df1.loc[df1["Name"]=="VGRCO2CaptureRate","Value"].iloc[0]
                    df1.loc[df1["Name"]=="VGRCO2CaptureRate","Called"]=True
                except:
                    try:
                         CO2Cap=df3.loc[df3["Name"]=="VGRCO2CaptureRate","Value"].iloc[0]
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')  
                # "preprocessor" input #13
                try:
                    H2Cap=1-df1.loc[df1["Name"]=="VGRH2PassRate","Value"].iloc[0]
                    df1.loc[df1["Name"]=="VGRH2PassRate","Called"]=True
                except:
                    try:
                         H2Cap=1-df3.loc[df3["Name"]=="VGRH2PassRate","Value"].iloc[0]
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')  
                # "preprocessor" input #14
                try:
                    WGS=df1.loc[df1["Name"]=="VGRCOConvertRate","Value"].iloc[0]
                    df1.loc[df1["Name"]=="VGRCOConvertRate","Called"]=True
                except:
                    try:
                         WGS=df3.loc[df3["Name"]=="VGRCOConvertRate","Value"].iloc[0]
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined') 
            
            W = sys_preprocessor()
            if preprocessor_name == 'NGFC_ccs': # NGFC CCS
                FuelIn,AirIn,AirFresh,Frec,succ=W.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
            elif preprocessor_name == 'NGFC_nocc': # NGFC NO CCS
                FuelIn,AirIn,AirFresh,Frec,succ=W.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
            elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                FuelIn,AirIn,AirFresh,Frec,succ=W.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
            elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                FuelIn,AirIn,AirFresh,Frec,succ=W.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
            elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                FuelIn,AirIn,AirFresh,Frec,succ=W.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
            else:
                sys.exit('Code terminated: the selected "preprocessor" cannot be found')
            
            if succ == 1:
                ## write to sofc4rom.dat
                inp_base=open(inputbasefilename,"r")
                lines_inp=inp_base.readlines()
                for j in range(len(lines_inp)):
                    str00=lines_inp[j].split('=')
                    str00[0]=str00[0].rstrip()
                    str00[0]=str00[0].lstrip()
                    
                    # update according to "preprocessor" outputs
                    if str00[0]=="FuelNGH2O": lines_inp[j]="FuelNGH2O = "+str(FuelIn[0])+"\n"
                    if str00[0]=="FuelNGAr": lines_inp[j]="FuelNGAr = "+str(FuelIn[1])+"\n"
                    if str00[0]=="FuelNGCO2": lines_inp[j]="FuelNGCO2 = "+str(FuelIn[2])+"\n"
                    if str00[0]=="FuelNGO2": lines_inp[j]="FuelNGO2 = "+str(FuelIn[3])+"\n"
                    if str00[0]=="FuelNGN2": lines_inp[j]="FuelNGN2 = "+str(FuelIn[4])+"\n"
                    if str00[0]=="FuelNGCH4": lines_inp[j]="FuelNGCH4 = "+str(FuelIn[5])+"\n"
                    if str00[0]=="FuelNGCO": lines_inp[j]="FuelNGCO = "+str(FuelIn[6])+"\n"
                    if str00[0]=="FuelNGH2": lines_inp[j]="FuelNGH2 = "+str(FuelIn[7])+"\n"
                    if str00[0]=="FuelNGC2H6": lines_inp[j]="FuelNGC2H6 = "+str(FuelIn[8])+"\n"
                    if str00[0]=="FuelNGC3H8": lines_inp[j]="FuelNGC3H8 = "+str(FuelIn[9])+"\n"
                    if str00[0]=="FuelNGC4H10": lines_inp[j]="FuelNGC4H10 = "+str(FuelIn[10])+"\n"
                    if str00[0]=="StackOxidantFlowRateO2": lines_inp[j]="StackOxidantFlowRateO2 = "+str(AirIn[0])+"\n"
                    if str00[0]=="StackOxidantFlowRateN2": lines_inp[j]="StackOxidantFlowRateN2 = "+str(AirIn[1])+"\n"
                    if str00[0]=="StackOxidantFlowRateH2O": lines_inp[j]="StackOxidantFlowRateH2O = "+str(AirIn[2])+"\n"
                    if str00[0]=="StackOxidantFlowRateCO2": lines_inp[j]="StackOxidantFlowRateCO2 = "+str(AirIn[3])+"\n"
                    if str00[0]=="StackOxidantFlowRateAr": lines_inp[j]="StackOxidantFlowRateAr = "+str(AirIn[4])+"\n"
                    if str00[0]=="FuelNGRecirculationRate": lines_inp[j]="FuelNGRecirculationRate = "+str(Frec)+"\n"
                    if str00[0]=="FuelNGFlowRate": lines_inp[j]="FuelNGFlowRate = "+str(sum(FuelIn))+"\n"
                    
                    # delete four lines when "preprocessor" enabled
                    if str00[0]=="FuelRecycle": lines_inp[j]=""
                    if str00[0]=="FuelRecyclePercent": lines_inp[j]=""
                    if str00[0]=="OxidantRecycle": lines_inp[j]=""
                    if str00[0]=="OxidantRecyclePercent": lines_inp[j]=""
                            
                    # update according to LH sampling
                    for k in range(len(df1)):
                        if str00[0]==df1['Name'].iloc[k]: 
                            lines_inp[j]=str00[0]+" = "+str(df1['Value'].iloc[k])+"\n"
                            df1.loc[df1["Name"]==str00[0],'Called']=True
                
                if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                    add_inp_lines=["0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0","0", "0"]
                    add_inp_lines[0]="Stack_Fuel_Utilization = "+str(FU)+"\n"
                    add_inp_lines[1]="Stack_Oxidant_Utilization = "+str(AU)+"\n"
                    add_inp_lines[2]="Oxidant_Recirculation = "+str(Arec)+"\n"
                    add_inp_lines[3]="Internal_Reforming = "+str(IR)+"\n"
                    add_inp_lines[4]="OxygenToCarbon_Ratio = "+str(OCR)+"\n"
                    add_inp_lines[5]="Average_CurrentDensity = "+str(J*10.0)+"\n"
                    add_inp_lines[6]="PreReform = "+str(PreReform)+"\n"
                    
                    add_inp_lines[7]="VGRRate = "+str(VGR)+"\n"
                    add_inp_lines[8]="VGRTemperature  = "+str(VGRTemperature )+"\n"
                    add_inp_lines[9]="VGRH2OPassRate = "+str(1-H2OCap)+"\n"
                    add_inp_lines[10]="VGRH2PassRate = "+str(1-H2Cap)+"\n"
                    add_inp_lines[11]="VGRCO2CaptureRate = "+str(CO2Cap)+"\n"
                    add_inp_lines[12]="VGRCOConvertRate = "+str(WGS)+"\n"
                    
                    add_inp_lines[13]="FreshOxidantFlowRateO2 = "+str(AirFresh[0])+"\n"
                    add_inp_lines[14]="FreshOxidantFlowRateN2 = "+str(AirFresh[1])+"\n"
                    add_inp_lines[15]="FreshOxidantFlowRateH2O = "+str(AirFresh[2])+"\n"
                    add_inp_lines[16]="FreshOxidantFlowRateCO2 = "+str(AirFresh[3])+"\n"
                    add_inp_lines[17]="FreshOxidantFlowRateAr = "+str(AirFresh[4])+"\n"
                    
                else:
                    add_inp_lines=["0","0","0","0","0","0","0","0","0","0","0","0"]
                    add_inp_lines[0]="Stack_Fuel_Utilization = "+str(FU)+"\n"
                    add_inp_lines[1]="Stack_Oxidant_Utilization = "+str(AU)+"\n"
                    add_inp_lines[2]="Oxidant_Recirculation = "+str(Arec)+"\n"
                    add_inp_lines[3]="Internal_Reforming = "+str(IR)+"\n"
                    add_inp_lines[4]="OxygenToCarbon_Ratio = "+str(OCR)+"\n"
                    add_inp_lines[5]="Average_CurrentDensity = "+str(J*10.0)+"\n"
                    add_inp_lines[6]="PreReform = "+str(PreReform)+"\n"

                    add_inp_lines[7]="FreshOxidantFlowRateO2 = "+str(AirFresh[0])+"\n"
                    add_inp_lines[8]="FreshOxidantFlowRateN2 = "+str(AirFresh[1])+"\n"
                    add_inp_lines[9]="FreshOxidantFlowRateH2O = "+str(AirFresh[2])+"\n"
                    add_inp_lines[10]="FreshOxidantFlowRateCO2 = "+str(AirFresh[3])+"\n"
                    add_inp_lines[11]="FreshOxidantFlowRateAr = "+str(AirFresh[4])+"\n"

                extra_inp_lines = []
                for k in range(len(df1)):
                    if df1['Called'].iloc[k] == False:
                        line_tmp=str(df1['Name'].iloc[k])+" = "+str(df1['Value'].iloc[k])+"\n"
                        extra_inp_lines.append(line_tmp)
                        df1.loc[df1["Name"]==str(df1['Name'].iloc[k]),'Called']=True

                outputfilename = path_tmp+'/'+'sofc4rom.dat'
                inp_w=open(outputfilename,"w")
                inp_w.write("@model="+inputbasefilename+"\n")
                inp_w.writelines(lines_inp)
                inp_w.writelines(add_inp_lines)
                inp_w.writelines(extra_inp_lines)
                inp_w.close()
            else:
                ## create failure resutl SOFC_MP_ROM.dat
                indpreprocessorfailed.append(i)

                lines=["0", "0", "0"]
                lines[0]="#SOFC 2D Simulation Result for Reduced Order Modeling\n"
                lines[1]="#FAILED\n"
                if Frec<0:
                    lines[2]="Calcualted fuel recirculation "+str(Frec)+" is negative\n"
                if Frec>0.9:
                    lines[2]="Calcualted fuel recirculation "+str(Frec)+" is larger than 0.9\n"

                outputfilename = path_tmp+'/'+'SOFC_MP_ROM.dat'
                inp_w=open(outputfilename,"w")
                inp_w.writelines(lines)
                inp_w.close()

        else: # if "preprocessor" not enabled
            nCells = 1
            StackVoltage = 0.7082
            
            # load 'romSOFCMP2D4ROM.inp'
            inputfilename = path_tmp+'/'+'romSOFCMP2D4ROM.inp'
            text_file=open(inputfilename,"r")
            lines = text_file.readlines()
            df0 = pd.DataFrame(np.array([['1a', '1b', '1c']]),columns=['Name', 'Value', 'Called'])
            df1 = pd.DataFrame(columns=['Name', 'Value', 'Called'])
            for j in range(len(lines)):
                if j>0:
                    str01 = lines[j].split('=')
                    str01[0]=str01[0].rstrip()
                    str01[0]=str01[0].lstrip()
                    df0['Name']=str01[0]
                    df0['Value']=float(str01[1])
                    df0['Called']=False
                    df1=pd.concat([df1,df0],sort=False,ignore_index=True)
                        
            # load inputbasefile
            inp_base=open(inputbasefilename,"r")
            lines_inp=inp_base.readlines()
            for j in range(len(lines_inp)):
                str00=lines_inp[j].split('=')
                str00[0]=str00[0].rstrip()
                str00[0]=str00[0].lstrip()
                if str00[0] == 'nCells':
                    nCells = int(str00[1])
            for j in range(len(lines_inp)):
                str00=lines_inp[j].split('=')
                str00[0]=str00[0].rstrip()
                str00[0]=str00[0].lstrip()
                for k in range(len(df1)):
                    if str00[0]==df1['Name'].iloc[k]: 
                        lines_inp[j]=str00[0]+" = "+str(df1['Value'].iloc[k])+"\n"
                        df1.loc[df1["Name"]==str00[0],'Called']=True
                if str00[0]=='StackVoltage':
                    for k in range(len(df1)):
                        if df1['Name'].iloc[k]=='Average_CellVoltage':
                            StackVoltage=nCells*df1['Value'].iloc[k]
                            lines_inp[j]=str00[0]+" = "+str(StackVoltage)+"\n"
                                
            extra_inp_lines = []
            for k in range(len(df1)):
                if df1['Called'].iloc[k] == False:
                    line_tmp=str(df1['Name'].iloc[k])+" = "+str(df1['Value'].iloc[k])+"\n"
                    extra_inp_lines.append(line_tmp)
                    df1.loc[df1["Name"]==str(df1['Name'].iloc[k]),'Called']=True

            outputfilename = path_tmp+'/'+'sofc4rom.dat'
            inp_w=open(outputfilename,"w")
            inp_w.write("@model="+inputbasefilename+"\n")
            inp_w.writelines(lines_inp)
            inp_w.writelines(extra_inp_lines)
            inp_w.close()
            
    if preprocessor_enabled == True:
        print('The following cases failed for preprocessor "'+preprocessor_name+'":')
        print(*indpreprocessorfailed)
        print('End of code\n')
    else:
        print('End of code\n')

class runSimu_HPC():
    def __init__(self, local_path, HPC_path, numcase, create_HPC_path, 
                 use_scratch, vgr_enabled,  
                 hostname, username, password, port):
        self.local_path = local_path # work path on the local machine
        self.HPC_path = HPC_path # work path on the HPC
        self.create_HPC_path = create_HPC_path # if create HPC_path if not exist
        self.use_scratch = use_scratch # if use "scratch" drive
        self.vgr_enabled = vgr_enabled # if enable vgr feature
        self.numcase = numcase # number of total cases
        self.hostname = hostname # address of HPC
        self.username = username # account username
        self.password = password # account password
        self.port = port # default: 22
        self.numruncase = None # number of cases sent to HPC
        self.indruncase = None # index of cases sent to HPC
    
    def PutCaseonHPC(self):
        '''
        The function puts all the cases on the HPC
        '''
        print('############################################################\
              \nPut all the cases on the HPC\
              \n############################################################')
        
        #cinfo = {'host':'hostname', 'username':'me', 'password':'secret', 'port':2222}
        #sftp = pysftp.Connection(**cinfo)
        sftp = pysftp.Connection(self.hostname, username=self.username, password=self.password, port=self.port)
        #cnopts = pysftp.CnOpts()
        #cnopts.hostkeys = None
        #sftp = pysftp.Connection(self.hostname, username=self.username, password=self.password, cnopts = cnopts)
        
        localdir = self.local_path + '/Cases'
        remotedir = self.HPC_path  + '/Cases'
        
        if sftp.exists(self.HPC_path) == True:
            if sftp.exists(remotedir) == False: # if destination directories (cases) not exist, copy cases to HPC
                sftp.makedirs(remotedir, mode = 777)
                if os.name == 'nt':
                    put_r_windows(sftp, localdir, remotedir, preserve_mtime = True)
                else:
                    sftp.put_r(localdir, remotedir, preserve_mtime = True)
            else: # if destination directories (cases) exist, ask before copy
                query = query_yes_no('"cases" folder already exists on the HPC, do you want to overwrite it?')
                if query == True:
                    if os.name == 'nt':
                        put_r_windows(sftp, localdir, remotedir, preserve_mtime = True)
                    else:
                        sftp.put_r(localdir, remotedir, preserve_mtime = True)
                else:
                    sftp.close()
                    pass
        elif self.create_HPC_path == True:
            print('The remote path does not exist, create directories')
            sftp.makedirs(remotedir, mode = 777)
            if os.name == 'nt':
                put_r_windows(sftp, localdir, remotedir, preserve_mtime = True)
            else:
                sftp.put_r(localdir, remotedir, preserve_mtime = True)
        else:
            error('The remote path does not exist')
            
        sftp.close()

    def SubSimuonHPC(self, NumCores_eachnode = '24', allocation = 'face', 
                     partition = 'short', time_limit = '0:30:00'):
        '''
        The function submits simulations on the HPC
        '''
        print('############################################################\
              \nSubmit simulations on the HPC\
              \n############################################################')
        
        ## Step 1: determine which cases are not finished: numruncase and indruncase
        # icase_start, icase_end
        numcores = NumCores_eachnode
        numruncase = self.numcase # numcase = icase_end-icase_start+1
        indruncase = []
        indfinishedcase = []
        for i in range(self.numcase): # may consider icase_start, icase_end
            path_tmp = self.local_path+'/Cases/Case'+str(i).zfill(5)+'/SOFC_MP_ROM.dat'
            if os.path.exists(path_tmp):
                #print('Case'+str(i).zfill(5)+' already has the result "SOFC_MP_ROM.dat" on the local machine')
                numruncase = numruncase-1
                indfinishedcase.append(i)
            else:
                indruncase.append(i)
        
        print('The following cases already have "SOFC_MP_ROM.dat" on the local machine:')
        print(*indfinishedcase)
                
        # update global variables
        self.numruncase = numruncase
        self.indruncase = indruncase
        
        ## Step 2: generate ".batch" files, assign jobs to each node
        numnode = int(math.ceil(float(numruncase)/float(numcores)))
        numLastnode = numruncase%numcores
        if numLastnode == 0: numLastnode = numcores
            
        list_sbatch = []
        for i in range(numnode):
            if i<numnode-1 or numnode == 1:
                ttjobs = numcores
                if numcores>numruncase: ttjobs = numLastnode
            else:
                ttjobs = numLastnode
            
            job_start = i*numcores # may consider icase_start, icase_end
            job_end = i*numcores+ttjobs-1 # may consider icase_start, icase_end
            
            # generate individual job (.batch file) for each node
            lines=[]
            lines.append("#!/bin/csh -f\n")
            lines.append("#SBATCH --job-name=" + str(job_start) + "-" + str(job_end) + "\n")
            lines.append("#SBATCH --time=" + time_limit + "\n")
            lines.append("#SBATCH -N 1\n")
            lines.append("#SBATCH -n " + str(ttjobs) + "\n")
            lines.append("#SBATCH --output=batchsofc" + str(job_start) + "-" + str(job_end) + ".out\n")
            lines.append("#SBATCH -A " + allocation + "\n")
            lines.append("#SBATCH -p " + partition + "\n")
            lines.append("source /etc/profile.d/modules.csh\n")
            lines.append("module purge\n")
            lines.append("module load gcc/4.4.7\n")
            
            for j in range(numruncase):
                icase = indruncase[j]
                if self.vgr_enabled == True:
                    if self.use_scratch == True:
                        lines.append("(cp -rf " + self.HPC_path + 
                                     "/Cases/Case" + str(icase).zfill(5) + 
                                     " /scratch/; cd /scratch/Case" + 
                                     str(icase).zfill(5) + 
                                     "; sofcvgr sofc4rom.dat; cp /scratch/Case" + 
                                     str(icase).zfill(5) + "/* " + 
                                     self.HPC_path + "/Cases/Case" + 
                                     str(icase).zfill(5) + "/ ) &\n")
                    else:
                        lines.append("(cd " + self.HPC_path + 
                                     "/Cases/Case" + str(icase).zfill(5) + 
                                     "; sofcvgr sofc4rom.dat ) &\n")
                else:
                    if self.use_scratch == True:
                        lines.append("(cp -rf " + self.HPC_path + 
                                     "/Cases/Case" + str(icase).zfill(5) + 
                                     " /scratch/; cd /scratch/Case" + 
                                     str(icase).zfill(5) + 
                                     "; sofc sofc4rom.dat; cp /scratch/Case" + 
                                     str(icase).zfill(5) + "/* " + 
                                     self.HPC_path + "/Cases/Case" + 
                                     str(icase).zfill(5) + "/ ) &\n")
                    else:
                        lines.append("(cd " + self.HPC_path + 
                                     "/Cases/Case" + str(icase).zfill(5) + 
                                     "; sofc sofc4rom.dat ) &\n")
            lines.append("wait\n")
            outputfilename = self.local_path + '/Cases/run' + str(job_start) + "-" + str(job_end) + '.sbatch'
            inp_w=open(outputfilename,"w")
            inp_w.writelines(lines)
            inp_w.close()
            # one need to convert \r\n to \n for windows system
            if os.name == 'nt':
                dos2unix(outputfilename)
            # update .sbatch filenames
            list_sbatch.append('run' + str(job_start) + "-" + str(job_end) + '.sbatch')
            
        ## Step 3: transfer ".batch" files to HPC, submit jobs
        sshClient = paramiko.SSHClient()                                   # create SSHClient instance
        sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
        sshClient.load_system_host_keys()
        sshClient.connect(self.hostname, self.port, self.username, self.password)
        
        sftpClient = sshClient.open_sftp()
        for string in list_sbatch:
            sourcefile = self.local_path + '/Cases/' + string
            destfile = self.HPC_path + '/Cases/' + string
            sftpClient.put(sourcefile, destfile)
        sftpClient.close
        
        # Step 4: submit simulations
        query = query_yes_no('".sbatch" files have been put on the HPC, do you want to submit the simulations?')
        
        if query == True:
            command_sbatch = 'cd ' + self.HPC_path  + '/Cases'
            for string in list_sbatch:
                command_sbatch = command_sbatch + '; sbatch ' + string
            stdin, stdout, stderr = sshClient.exec_command(command_sbatch)
            for line in stdout:
                print(line.strip('\n'))
            sshClient.close()
        else:
            sshClient.close()
    
    def CheckSimuStatus(self):
        '''
        The function checks the simulation status on the HPC
        '''
        print('############################################################\
              \nChecks the simulation status on the HPC\
              \n############################################################')
        
        sshClient = paramiko.SSHClient()                                   # create SSHClient instance
        sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
        sshClient.load_system_host_keys()
        sshClient.connect(self.hostname, self.port, self.username, self.password)
        sftpClient = sshClient.open_sftp()
        
        numruncase = self.numruncase
        indruncase = self.indruncase
        
        indfinishedcase = []
        indfailedcase = []
        numfinishedcase = 0
        for icase in indruncase:
            destfile = self.HPC_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
            try:
                sftpClient.stat(destfile)
                numfinishedcase += 1
                indfinishedcase.append(icase)
            except IOError:
                indfailedcase.append(icase)
                    
        print(str(numfinishedcase)+' out of '+str(numruncase)+' cases have been done:')
        print(*indfinishedcase)
        
        sftpClient.close()
        sshClient.close()
        
    def GetReslfromHPC(self):
        '''
        The function gets simulation results from the HPC
        '''
        print('############################################################\
              \nGet simulation results from the HPC\
              \n############################################################')
        
        sshClient = paramiko.SSHClient()                                   # create SSHClient instance
        sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
        sshClient.load_system_host_keys()
        sshClient.connect(self.hostname, self.port, self.username, self.password)
        sftpClient = sshClient.open_sftp()
        
        numruncase = self.numruncase
        indruncase = self.indruncase
        
        query = False
        for icase in indruncase:
            path_tmp = self.local_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
            if os.path.exists(path_tmp):
                query = query_yes_no('certain cases already have "SOFC_MP_ROM.dat" on the local machine, do you want to overwite it?')
                break
        
        indexist = []
        indnonexist = []
        if query == True:
            for icase in indruncase:
                sourcefile = self.HPC_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
                destfile = self.local_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
                try:
                    sftpClient.get(sourcefile, destfile)
                    indexist.append(icase)
                except:
                    indnonexist.append(icase)
        else:
            for icase in indruncase:
                sourcefile = self.HPC_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
                destfile = self.local_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
                
                if os.path.exists(destfile):
                    indexistlocal.append(icase)
                else:
                    try:
                        sftpClient.get(sourcefile, destfile)
                        indexist.append(icase)
                    except:
                        indnonexist.append(icase)
        
        print('The following cases do not have "SOFC_MP_ROM.dat" on the HPC (case failed or has not converged yet):')
        print(*indnonexist)
        print('Get "SOFC_MP_ROM.dat" to the local machine for the following cases:')
        print(*indexist)
            
        sftpClient.close
        sshClient.close()

class runSimu_SubSys():
    def __init__(self, work_path, source_path, numcase, vgr_enabled,  
                 hostname, username, password, port):
        self.work_path = work_path # work path on the local machine
        self.source_path = source_path # source path on the local machine
        self.vgr_enabled = vgr_enabled # if enable vgr feature
        self.numcase = numcase # number of total cases
        self.hostname = hostname # address of sub-system
        self.username = username # account username
        self.password = password # account password
        self.port = port # port of sub-system
        self.numruncase = None # number of cases sent to sub-system
        self.indruncase = None # index of cases sent to sub-system

    def SubSimuonSS(self, MaxSimulIns = 1, time_limit = '1:00:00'):
        '''
        The function submits simulations on the sub-system
        '''
        print('############################################################\
              \nSubmit simulations on the sub-system\
              \n############################################################')
        
        # Start sshClient
        sshClient = paramiko.SSHClient()                                   # create SSHClient instance
        sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
        sshClient.load_system_host_keys()
        sshClient.connect(self.hostname, self.port, self.username, self.password)
        sftpClient = sshClient.open_sftp()

        RunningCount = 0
        RunningInd = []
        FinishedCount = 0
        FinishedInd = []
        FinishedCount_update = 0
        time_start = time.time()
        while(True):
            # Check how many processes in the background
            if self.vgr_enabled == False:
                command = 'pgrep -c sofc'
            else:
                command = 'pgrep -c sofcvgr'
            stdin, stdout, stderr = sshClient.exec_command(command)
            RunningCount = int(stdout.read())
            
            for i in range(self.numcase):
                # Check if case i is done or not
                if i in FinishedInd:
                    CaseFinished = True
                else:
                    destfile = self.work_path+'/Cases/Case'+str(i).zfill(5)+'/SOFC_MP_ROM.dat'
                    try:
                        sftpClient.stat(destfile)
                        FinishedCount += 1
                        FinishedInd.append(i)
                        if i in RunningInd:
                            RunningInd.remove(i)
                        CaseFinished = True
                    except IOError:
                        CaseFinished = False
                
                # Run case i if 1: case not done; 2: space in the queue; 3: case not running
                if CaseFinished == False and RunningCount < MaxSimulIns and (i not in RunningInd):
                    if self.vgr_enabled == False:
                        command = '(cd '+self.work_path+'/Cases/Case'+ str(i).zfill(5)                        +'; '+self.source_path+'/sofc sofc4rom.dat) &'
                        sshClient.exec_command(command)
                        # Add case i to the running case list
                        RunningInd.append(i)
                        RunningCount += 1
                    else:
                        command = '(cd '+self.work_path+'/Cases/Case'+ str(i).zfill(5)                        +'; '+self.source_path+'/sofcvgr sofc4rom.dat) &'
                        sshClient.exec_command(command)
                        # Add case i to the running case list
                        RunningInd.append(i)
                        RunningCount += 1
                
                # Break out for-loop if not space in the queue
                if RunningCount >= MaxSimulIns:
                    break
            
            # Update simulation status
            if (FinishedCount-FinishedCount_update) >= 5:
                FinishedCount_update = FinishedCount
                print("Simulation status:\nRunning: "+str(RunningCount)+"\tFinished: "+str(FinishedCount))
            
            # Break out while-loop if no running case or exceed time
            hour, min, sec = [float(i) for i in time_limit.split(':')]
            time_limit_sec = hour*3600+min*60+sec
            time_elapsed = time.time()-time_start
            if RunningCount == 0:
                print("All the simulation Done!")
                break
            if time_elapsed > time_limit_sec:
                print("Exceed time limit, simulation terminated!")
                # Kill all the background processes and break while loop
                if self.vgr_enabled == False:
                    command = 'pkill sofc'
                else:
                    command = 'pkill sofcvgr'
                stdin, stdout, stderr = sshClient.exec_command(command)
                break
        
        # End sshClient
        sftpClient.close
        sshClient.close()
    
    def CheckSimuStatus(self):
        '''
        The function checks the simulation status
        '''
        print('############################################################\
              \nChecks the simulation status\
              \n############################################################')
        
        sshClient = paramiko.SSHClient()                                   # create SSHClient instance
        sshClient.set_missing_host_key_policy(paramiko.AutoAddPolicy())    # AutoAddPolicy automatically adding the hostname and new host key
        sshClient.load_system_host_keys()
        sshClient.connect(self.hostname, self.port, self.username, self.password)
        sftpClient = sshClient.open_sftp()
        
        indfinishedcase = []
        indfailedcase = []
        numfinishedcase = 0
        for icase in range(self.numcase):
            destfile = self.work_path + '/Cases/Case'+str(icase).zfill(5)+'/SOFC_MP_ROM.dat'
            try:
                sftpClient.stat(destfile)
                numfinishedcase += 1
                indfinishedcase.append(icase)
            except IOError:
                indfailedcase.append(icase)
                    
        print(str(numfinishedcase)+' out of '+str(self.numcase)+' cases have been done:')
        print(*indfinishedcase)
        
        sftpClient.close
        sshClient.close()

class kriging():
    def __init__(self, work_path, 
                 allresultsFile = 'allResults.dat',
                 allresults_infoFile = 'allResults_info.dat',
                 inkrigingFile = 'inTraining_kriging.dat', 
                 infoFile = 'info_kriging.dat', 
                 outkrigingFile = 'outTraining_kriging.dat', 
                 inpredictionFile = 'inPrediction_kriging.dat', 
                 outpredictionFile = 'outPrediction_kriging.dat', 
                 order = 0):
        self.work_path = work_path
        self.allresultsFile = work_path + '/' + allresultsFile
        self.allresults_infoFile = work_path + '/' + allresults_infoFile
        self.inkrigingFile = work_path + '/' + inkrigingFile
        self.infoFile = work_path + '/' + infoFile
        self.outkrigingFile = work_path + '/' + outkrigingFile
        self.inpredictionFile = work_path + '/' + inpredictionFile
        self.outpredictionFile = work_path + '/' + outpredictionFile
        self.incrossvaliFile = work_path + '/inCrossVali_kriging.dat'
        self.outcrossvaliFile = work_path + '/outCrossVali_kriging.dat'
        self.order = int(order)
        self.Sname = None
        self.Yname = None
        self.S_norm = None
        self.Y_norm = None
        self.X_norm = None
        self.Xy_norm = None
        self.S = None
        self.Y = None
        self.X = None
        self.Xy = None
        self.MSE = None
        self.S_row = 0
        self.Y_row = 0
        self.S_col = 0
        self.Y_col = 0
        self.stdS = None
        self.stdY = None
        self.meanS = None
        self.meanY = None
    
    def summarize_SimuResult(self, source_path, indcase, exclude_case = 1, display_detail = False):
        '''
        The function extracts simulation results
            exclude_case = -1: all cases included
            exclude_case = 0: exclude failed cases only
            exclude_case = 1: exclude both failed and non-converged cases
        '''
        print('############################################################\
              \nSummarize simulation results\
              \n############################################################')
        
        ## Step 1: load simulation outputs to Y4kriging
        numcase4kriging = 0 # number of cases for kriging
        indcase4kriging = [] # index of cases for kriging, start from 1
        S4kriging = None # simulation inputs for kriging
        Y4kriging = None # simulation outputs for kriging
        for icase in indcase:
            # load SOFC_MP_ROM.dat to df1
            strcase = 'Case'+str(icase-1)+'Value'
            inputfilename = source_path+'/Cases/Case'+str(icase-1).zfill(5)+'/SOFC_MP_ROM.dat'
            if os.path.exists(inputfilename):
                text_input=open(inputfilename,"r")
                lines=text_input.readlines()
                
                if len(lines) == 0: 
                    continue #print('Empty case')
                if lines[1].strip() == '#FAILED': 
                    continue #print('"preprocessor" failed case')
                
                df0 = pd.DataFrame(np.array([['1a', '1b']]),columns=['Name', strcase])
                df1 = pd.DataFrame(np.array([['1a', '1b']]),columns=['Name', strcase])
                for j in range(len(lines)):
                    if j>1: # skip first two lines
                        str01 = lines[j].split('=')
                        str01[0]=str01[0].rstrip()
                        str01[0]=str01[0].lstrip()
                        
                        if len(str01) == 1: continue
                        
                        # convert variables in SOFC_MP_ROM.dat to xxx_xxx format
                        str_tmp = str01[0].strip().split()
                        str_tmp = '_'.join(str_tmp)
                        df0['Name']=str_tmp
                        df0[strcase]=float(str01[1])
                        if j==2:
                            df1["Name"]=df0["Name"]
                            df1[strcase]=df0[strcase]
                        else:
                            df1=pd.concat([df1,df0],sort=False, ignore_index=True)
                
                # exclude failed or non-converged cases
                if int(df1.loc[0, [strcase]]) >= exclude_case:
                    numcase4kriging += 1
                    indcase4kriging.append(icase)
                    if numcase4kriging == 1:
                        Y4kriging = df1
                    else:
                        Y4kriging = pd.concat([Y4kriging, df1[strcase]], sort=False, axis=1)

        ## Step 2: load simulation inputs to S4kriging
        inputfilename = source_path+'/LHS.dat'
        if os.path.exists(inputfilename):
            text_input=open(inputfilename,"r")
            lines=text_input.readlines()
            
            for j in range(len(lines)):
                if j == 1:
                    list_tmp = lines[j].strip().split()
                    list_tmp = list_tmp[2:] # 0: case; 1: No.
                    df2 = pd.DataFrame(list_tmp,columns=['Name'])
                if j > 1:
                    list_tmp = lines[j].strip().split()
                    strcase = 'Case'+str(int(list_tmp[0])-1)+'Value'
                    list_tmp = list_tmp[1:] # 0: case No.
                    df2[strcase] = list_tmp
            S4kriging = df2
        
        ## Step 3: display simulation input and output
        if exclude_case == 1:
            print('Converged simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        elif exclude_case == 0:
            print('Converged and non-converged simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        else:
            print('Simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        print(*indcase4kriging)
        print('\nSelect from the following input variables for training:')
        for i in range(S4kriging.index.size):
            print(i+1, ':', S4kriging.loc[i, 'Name'], end = '\t\n')
        print('\nSelect from the following output variables for training:')
        for i in range(Y4kriging.index.size):
            print(i+1, ':', Y4kriging.loc[i, 'Name'], end = '\t\n')
            
        if display_detail == True:
            print('\n')
            print(S4kriging)
            print('\n')
            print(Y4kriging)
        
        ## Step 4: create allResults.dat
        indS = list(S4kriging.index)
        indY = list(Y4kriging.index)
        indS = [x+1 for x in indS]
        indY = [x+1 for x in indY]
        if len(indcase4kriging) == 0 or len(indS) == 0 or len(indY) == 0:
            print('Error: No data available for training')   
        
        with open(self.allresultsFile, 'w') as f:
            for i in indS:
                f.write(S4kriging.loc[i-1, 'Name'] + '\t')
            for i in indY:
                f.write(Y4kriging.loc[i-1, 'Name'] + '\t')
            f.write('\n')
            for i in indcase4kriging:
                strcase = 'Case'+str(i-1)+'Value'
                for j in indS:
                    f.write('{:11.4E}\t'.format(float(S4kriging.loc[j-1, strcase])))
                for j in indY:
                    f.write('{:11.4E}\t'.format(float(Y4kriging.loc[j-1, strcase])))
                f.write('\n')
                
        with open(self.allresults_infoFile, 'w') as f:
            f.write('input_col\toutput_col\n')
            f.write(str(len(indS))+'\t'+str(len(indY))+'\n')
            
    def file_read(self, FileName):
        '''
        This function loads the kriginginputFile, 
        infoFile and predictioninputFile
        '''
        namearray = []
        valuearray = []
        with open(FileName) as f:
            i = 0
            for line in f.readlines():
                if i == 0:
                    namearray = line.strip().split()
                else:
                    linestr = line.strip().split()
                    linenum = [float(lineele) for lineele in linestr]
                    valuearray.append(linenum)
                i += 1
               
        return namearray, np.array(valuearray)
    
    def cal_obj(self, theta, finalized = False, order = 0):
        # Copy to local
        theta = copy.deepcopy(theta)
        [S_row, Y_row, S_col, Y_col] =         [self.S_row, self.Y_row, self.S_col, self.Y_col]
        [S_norm, Y_norm] = [self.S_norm, self.Y_norm]
        
        # calculate F
        if order == 0:
            F = np.full([S_row, 1], 1.0)
        else:
            F = np.full([S_row, S_col+1], 1.0)
            for i in range(S_col):
                for j in range(S_row):
                    F[j, i+1] = S_norm[j, i]
        
        # Calculate R
        R = np.empty([S_row, S_row])
        R_tmp = 0.0
        multiple_sites = 0.0
        
        for i in range(S_row):
            for j in range(S_row):
                for k in range(S_col):
                    
                    R_tmp = R_tmp+theta[k]*(S_norm[i, k]-S_norm[j, k])                    *(S_norm[i, k]-S_norm[j, k])
                    # Check if "multiple sites" exists or not
                    if S_norm[i, k] == S_norm[j, k] and i != j:
                        for k_multiple_sites in range(S_col):
                            multiple_sites = multiple_sites +                             np.abs(S_norm[i, k_multiple_sites] -                                    S_norm[j, k_multiple_sites])
                        if multiple_sites == 0:
                            sys.exit('Code terminated: multiple sites found')
                R[i, j] = np.exp(-R_tmp)
                R_tmp = 0.0
        #print('R: ', R)
        
        # Cholesky decomposition
        C = la.cholesky(R)
        #print('C: ', C)
        
        # calculate F hat
        Ft = la.solve(C, F)
        #Ft, resid_tmp, rank_tmp, sigma_tmp = \
        #la.lstsq(C, F, rcond = None)
        #print('Ft: ', Ft)
        
        # calculate Y hat
        Yt = la.solve(C, Y_norm)
        #Yt, resid_tmp, rank_tmp, sigma_tmp = \
        #la.lstsq(C, Y_norm, rcond = None)
        #print('Yt: ', Yt)
        #print('Yt size', Yt.shape)
        
        # QR factorization
        Q, G = la.qr(Ft, 'reduced')
        #Q, G = scipy.linalg.qr(Ft, mode = 'economic')
        #print('Q: ', Q)
        #print('G: ', G)
        
        # calculate beta
        beta = la.solve(G, np.matmul(Q.T, Yt))
        #beta, resid_tmp, rank_tmp, sigma_tmp = \
        #la.lstsq(G, np.matmul(Q.T, Yt), rcond = None)
        #print('beta: ', beta)
        
        # calculate rho, sigma
        rho = Yt-np.matmul(Ft, beta)
        #print('rho: ', rho)
        sigma2_tmp0 = np.full([1, Y_col], Y_row)
        sigma2_tmp = np.sum(rho*rho, axis = 0)/sigma2_tmp0
        #print('sigma2_tmp: ', sigma2_tmp)
        
        # calculate diag, detR
        diag = np.power(np.diag(C), 2./float(S_row))
        detR = np.prod(diag)
        #print('diag: ', diag)
        #print('detR: ', detR)
        
        # calculate obj
        obj = np.sum(sigma2_tmp)*detR
        
        if finalized == False:
            #print('obj: ', obj)
            #print('theta: ', theta)
            return obj
        else:
            gamma = np.matmul(rho.T, la.inv(C))
            sigma2 = (self.stdY*self.stdY)*sigma2_tmp
            #print('theta: ', theta)
            #print('beta: ', beta)
            #print('sigma2: ', sigma2_tmp)
            #print('G: ', G)
            #print('Ft: ', Ft)
            #print('gamma: ', gamma)
            #print('C: ', C)
            return obj, beta, sigma2, G, Ft, gamma, C
        
    def variables(self):
        print('input variables:')
        for i in range(len(self.Sname)):
            print(i+1, ':', self.Sname[i], end = '\t\n')
            
        print('\noutput variables:')
        for i in range(len(self.Yname)):
            print(i+1, ':', self.Yname[i], end = '\t\n')
    
    def variable_options(self, display = False):
        names_input = [
            "Average_CellVoltage",
            "Average_CurrentDensity",
            "BackEnvironmentT",
            "BottomEnvironmentT",
            "CellFuelFlowRate",
            "CellOxidantFlowRate",
            "FrontEnvironmentT",
            "Fuel_Utilization",
            "FuelH2",
            "FuelH2O",
            "FuelCO",
            "FuelCO2",
            "FuelCH4",
            "FuelN2",
            "FuelTemperature",
            "FuelTOnTop",
            "FuelRecyclePercent",
            "FuelHTXEffectiveness",
            "FuelNGTemperature",
            "FuelNGHTXDeltaT",
            "Internal_Reforming",
            "nCells",
            "Oxidant_Recirculation",
            "OxidantRecyclePercent",
            "OxygenToCarbon_Ratio",
            "OxidantO2",
            "OxidantN2",
            "OxidantH2O",
            "OxidantCO2",
            "OxidantAr",
            "OxidantTemperature",
            "OxidantTOnTop",
            "PreReform",
            "SideEnvironmentT",
            "Simulation_Option",
            "Stack_Fuel_Utilization",
            "Stack_Oxidant_Utilization",
            "StackFuelFlowRate",
            "StackFuelFlowRateH2O",
            "StackFuelFlowRateCO",
            "StackFuelFlowRateCO2",
            "StackFuelFlowRateCH4",
            "StackFuelFlowRateH2",
            "StackFuelFlowRateN2",
            "StackOxidantFlowRate",
            "StackOxidantFlowRateO2",
            "StackOxidantFlowRateN2",
            "StackOxidantFlowRateH2O",
            "StackOxidantFlowRateCO2",
            "StackOxidantFlowRateAr",
            "StackVoltage",
            "SystemPressure",
            "TopEnvironmentT",
            "VGRRate",
            "VGRTemperature",
            "VGRH2OPassRate",
            "VGRH2PassRate",
            "VGRCO2CaptureRate",
            "VGRCOConvertRate"
        ]

        units_input = [
            "V",
            "A/m^2",
            "C",
            "C",
            "mol/s",
            "mol/s",
            "C",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "C",
            "C",
            "%",
            "-",
            "C",
            "C",
            "-",
            "-",
            "-",
            "%",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "C",
            "C",
            "-",
            "C",
            "-",
            "-",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "V",
            "atm",
            "C",
            "-",
            "C",
            "-",
            "-",
            "-",
            "-"
        ]
        
        names_output = [
            'SimulationStatus',
            'Stack_Voltage',
            'Avg_cell_voltage',
            'Stack_Current',
            'Avg_current_density',
            'Max_current_density',
            'Min_current_density',
            'Avg_Cell_Temperature',
            'Max_Cell_Temperature',
            'Min_Cell_Temperature',
            'Delta_Cell_Temperature',
            'Outlet_Fuel_Temperature',
            'Delta_Fuel_Temperature',
            'Outlet_Air_Temperature',
            'Delta_Air_Temperature',
            'Air_Heat_Exchanger_Effectiveness',
            'Fuel_Utilization',
            'Air_Utilization',
            'Outlet_Fuel_Flowrate',
            'Outlet_Fuel_H2',
            'Outlet_Fuel_H2O',
            'Outlet_Fuel_CO',
            'Outlet_Fuel_CO2',
            'Outlet_Fuel_CH4',
            'Outlet_Fuel_N2',
            'Outlet_Air_Flowrate',
            'Outlet_Air_O2',
            'Outlet_Air_N2',
            'Outlet_Air_H2O',
            'Outlet_Air_CO2',
            'Outlet_Air_Ar',
            'Total_Power',
            'Air_Enthalpy_Change',
            'Fuel_Enthalpy_Change',
            'External_Heat',
            'Electrical_Efficiency',
            'Stack_Efficiency',
            'Air_Inlet_Temperature',
            'FSI_Temperature',
            'FSI_Flowrate',
            'FSI_H2_MF',
            'FSI_H2O_MF',
            'FSI_CO_MF',
            'FSI_CO2_MF',
            'FSI_CH4_MF',
            'FSI_N2_MF',
            'Fuel_Temperature_after_Mix',
            'Fuel_Temperature_before_Gibbs_Reactor',
            'Fuel_Heat_Exchanger_Effectiveness'
        ]

        units_output = [
            '-',
            'V',
            'V',
            'A',
            'A/m2',
            'A/m2',
            'A/m2',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            '-',
            '-',
            '-',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            'W',
            'W',
            'W',
            'W',
            '-',
            '-',
            'K',
            'K',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            'K',
            'K',
            '-'
        ]

        if display == True:
            print('Options of input variable:')
            for i in range(len(names_input)):
                print(i+1, ':', names_input[i]+', ['+units_input[i]+']', end = '\t\n')
            print('Options of output variable:')
            for i in range(len(names_output)):
                print(i+1, ':', names_output[i]+', ['+units_output[i]+']', end = '\t\n')
        
        return names_input, units_input, names_output, units_output
    
    def grid(x, y, z, resX = 100, resY = 100):
        '''
        The function Convert 3 column data to matplotlib grid
        '''
        xi = np.linspace(min(x), max(x), resX)
        yi = np.linspace(min(y), max(y), resY)
        Z = matplotlib.mlab.griddata(x, y, z, xi, yi)
        X, Y = np.meshgrid(xi, yi)
        return X, Y, Z
    
    def training(self):
        '''
        The function trains the Kriging model
        (regression model with polynomials of order 0, 1, 2)
        '''
        print('############################################################\
              \nTrain the Kriging model (order ', self.order, ')\
              \n############################################################')
        
        # # Step 0: check if outkriging.dat existing
        # if os.path.exists(self.outkrigingFile):
        #     query = query_yes_no('kriging results already exist on the local machine, do you want to overwrite it?')
        #     if query == False: return
                
        # Step 1: Load the training data S, Y
        print('Step 1: Load the training data S, Y')
        SYname, SYvalue = self.file_read(self.inkrigingFile)
        infoname, infovalue = self.file_read(self.infoFile)
        [S_row, Y_row, S_col, Y_col] = [len(SYvalue), len(SYvalue),                                         int(infovalue[0,0]), int(infovalue[0,1])]
        [self.S_row, self.Y_row, self.S_col, self.Y_col] = [S_row, Y_row, S_col, Y_col]
        
        S = copy.deepcopy(SYvalue[:, :S_col])
        Y = copy.deepcopy(SYvalue[:, S_col:])

        Sname = copy.deepcopy(SYname[:S_col])
        Yname = copy.deepcopy(SYname[S_col:])
        
        # Step 2: Normalize S, Y
        print('Step 2: Normalize S, Y')
        meanS = np.mean(S, axis = 0)
        meanY = np.mean(Y, axis = 0)
        stdS = np.std(S, axis = 0, ddof = 1) #calculate standard deviation of normal distribution
        stdY = np.std(Y, axis = 0, ddof = 1)
        stdS[stdS == 0] = 1
        stdY[stdY == 0] = 1
        S_norm = (S - np.tile(meanS, [S_row, 1]))/np.tile(stdS, [S_row, 1])
        Y_norm = (Y - np.tile(meanY, [Y_row, 1]))/np.tile(stdY, [Y_row, 1])
        
        # copy from local to global
        self.S_norm = S_norm
        self.Y_norm = Y_norm
        self.S = S
        self.Y = Y
        [self.stdS, self.stdY] = [stdS, stdY]
        self.Sname = Sname
        self.Yname = Yname
        
        # Step 3: Initial Regression model
        print('Step 3: Regression model')
        theta1 = np.ones(S_col)*10.0
        lo = np.ones(S_col)*0.1
        up = np.ones(S_col)*20.0
        
        print('\tDesign variable: ')
        print('\tlower bound: ', lo, ', upper bound: ', up,               ', initial theta: ', theta1)
        
        #call cal_obj (1st)
        obj = self.cal_obj(theta1, False, self.order)
        print('\tInitial: obj: ', obj)
        
        # Step 4: Loop optimizing the regression model
        if S_col <= 2:
            kmax = 2
        elif S_col <= 4:
            kmax = copy.deepcopy(S_col)
        else:
            kmax = 4
        p = np.array(range(0, S_col))+1
        D = np.power(2, p/(float(S_col)+2.))
        #print('p: ', p)
        #print('D: ', D)
        
        for i_opt in range(kmax):
            
            # EXPLORE
            theta1_org = copy.deepcopy(theta1)
            atbd = None
            theta_theta = copy.deepcopy(theta1)
            
            for k in range(S_col):
                
                if theta1[k] == lo[k]:
                    atbd = 1
                    theta_theta[k] = theta1[k]*np.power(D[k], 0.5)
                elif theta1[k] == up[k]:
                    atbd = 1
                    theta_theta[k] = theta1[k]/np.power(D[k], 0.5)
                else:
                    atbd = 0
                    if up[k] >= theta1[k]*D[k]:
                        theta_theta[k] = theta1[k]*D[k]
                    else:
                        theta_theta[k] = up[k]
                
                #call cal_obj (2nd)
                obj_tmp = self.cal_obj(theta_theta, False, self.order)
                
                if obj_tmp < obj:
                    obj = copy.deepcopy(obj_tmp)
                    theta1 = copy.deepcopy(theta_theta)
                else:
                    if atbd == 0:
                        if lo[k] >= theta1[k]/D[k]:
                            theta_theta[k] = lo[k]
                        else:
                            theta_theta[k] = theta1[k]/D[k]
                        
                        #call cal_obj (3rd)
                        obj_tmp = self.cal_obj(theta_theta, False, self.order)
                        
                        if obj_tmp < obj:
                            obj = copy.deepcopy(obj_tmp)
                            theta1 = copy.deepcopy(theta_theta)
                            
            print('\t', i_opt+1, ' iteration - Finish EXPLORE - obj: ', obj_tmp)
            
            # MOVE
            v = theta_theta/theta1_org
            k = np.sum(v == 1)
            
            if k == S_col:
                for i in range(S_col):
                    D[i] = np.power(D[S_col-i-1], 0.2)
            
            rept = 1
            while rept == 1:
                for i in range(S_col):
                    
                    if lo[i] >= theta1[i]*v[i]:
                        move_tmp = lo[i]
                    else:
                        move_tmp = theta1[i]*v[i]
                    
                    if up[i] >= move_tmp:
                        theta_theta[i] = move_tmp
                    else:
                        theta_theta[i] = up[i]
                        
                #call cal_obj (4th)
                obj_tmp = self.cal_obj(theta_theta, False, self.order)
                
                if obj_tmp < obj:
                    obj = copy.deepcopy(obj_tmp)
                    theta1 = copy.deepcopy(theta_theta)
                    v = v*v
                    #print('v new: ', v)
                else:
                    rept = 0
                    
                for i in range(S_col):
                    if theta_theta[i] == lo[i] or theta_theta[i] == up[i]:
                        rept = 0
            
            print('\t             - Finish MOVE - obj: ', obj_tmp)
            
            #update D
            D_tmp = np.power(D, 0.25)
            #print('D: ', D)
            #print('D_tmp', D_tmp)
            D[:(S_col-1)] = D_tmp[1:]
            D[S_col-1] = D_tmp[0]
            #print('D: ', D)
            
        # Step 5: Final Regression Model
        obj, beta, sigma2, G, Ft, gamma, C = self.cal_obj(theta1, True, self.order)
        print('\tFinal: obj: ', obj, ', theta: ', theta1)
        
        # Step 6: Write the trained model
        print('Step 4: Write the trained model')
        with open(self.outkrigingFile, 'w') as f:
            f.write('S_row\n')
            f.write(str(S_row) + '\n')
            f.write('S_col\n')
            f.write(str(S_col) + '\n')
            f.write('Y_row\n')
            f.write(str(Y_row) + '\n')
            f.write('Y_col\n')
            f.write(str(Y_col) + '\n')
            
            f.write('meanS\n')
            for value in meanS:
                f.write(str(value) + ' ')
            f.write('\n' + '\n')
            f.write('meanY\n')
            for value in meanY:
                f.write(str(value) + ' ')
            f.write('\n' + '\n')
            f.write('stdS\n')
            for value in stdS:
                f.write(str(value) + ' ')
            f.write('\n' + '\n')
            f.write('stdY\n')
            for value in stdY:
                f.write(str(value) + ' ')
            f.write('\n' + '\n')
            
            f.write('theta\n')
            for value in theta1:
                f.write(str(value) + ' ')
            f.write('\n' + '\n')
            
            f.write('beta\n')
            [row, col] = beta.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(beta[i, j]) + ' ')
                f.write(str(beta[i, col-1]) + '\n')
            f.write('\n')
            
            f.write('sigma2\n')
            for i in range(len(sigma2.T)):
                f.write(str(sigma2[0,i]) + ' ')
            f.write('\n' + '\n')
            
            f.write('G\n')
            [row, col] = G.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(G[i, j]) + ' ')
                f.write(str(G[i, col-1]) + '\n')
            f.write('\n')
            
            f.write('Ft\n')
            [row, col] = Ft.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(Ft[i, j]) + ' ')
                f.write(str(Ft[i, col-1]) + '\n')
            f.write('\n')
            
            f.write('gamma\n')
            [row, col] = gamma.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(gamma[i, j]) + ' ')
                f.write(str(gamma[i, col-1]) + '\n')
            f.write('\n')
            
            f.write('C\n')
            [row, col] = C.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(C[i, j]) + ' ')
                f.write(str(C[i, col-1]) + '\n')
            f.write('\n')
        print('End of code\n')
        
    def prediction(self):
        '''
        This function predicts the outputs and MSEs 
        based on the trained kriging model 
        (regression model with polynomials of order 0, 1, 2)
        '''
        print('############################################################\
              \nPredict Based on the trained kriging model (order ', self.order, ')\
              \n############################################################')
        
        # # Step 0: check if outprediction.dat existing
        # if os.path.exists(self.outpredictionFile):
        #     query = query_yes_no('prediction results already exist on the local machine, do you want to overwrite it?')
        #     if query == False: return
            
        # Step 1: Load the training data S, Y and prediction data Sp
        print('Step 1: Load the training data S, Y and prediction input data X')
        SYname, SYvalue = self.file_read(self.inkrigingFile)
        Xname, Xvalue = self.file_read(self.inpredictionFile)
        
        # Step 2: Load the trained model (outkrigingFile)
        print('Step 2: Load the trained model (outkrigingFile)')
        with open(self.outkrigingFile) as f:
            i = 0
            for line in f.readlines():
                if i == 2-1:
                    linestr = line
                    S_row = int(linestr)
                    #print(type(S_row))
                    #print(S_row)
                if i == 4-1:
                    linestr = line
                    S_col = int(linestr)
                    #print(type(S_col))
                    #print(S_col)
                if i == 6-1:
                    linestr = line
                    Y_row = int(linestr)
                if i == 8-1:
                    linestr = line
                    Y_col = int(linestr)
                i += 1
        
        countFt = 0
        countgamma = 0
        countC = 0
        countbeta = 0
        countG = 0
        countsigma2 = 0
        
        if self.order == 0:
            # load outkriging file with order 0: especially G, beta
            with open(self.outkrigingFile) as f:
                i = 0
                for line in f.readlines():
                    if i == 10-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        meanS = np.array(linenum)
                        #print(meanS)
                        #print(type(meanS))
                    if i == 13-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        meanY = np.array(linenum)
                    if i == 16-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        stdS = np.array(linenum)
                        #print(stdS)
                        #print(type(stdS))
                    if i == 19-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        stdY = np.array(linenum)
                    if i == 22-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        theta1_opt = np.array(linenum)
                        #print(theta1_opt)
                        #print(type(theta1_opt))
                    if i == 25-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        beta = np.array(linenum)
                    if i == 28-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        sigma2 = np.array(linenum)
                    if i == 31-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        G = np.array(linenum)
                        #print(G)
                    if i == 34-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        Ft = np.array(linenum)
                        countFt += 1
                    if i == 34-1+countFt and countFt < S_row:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        Ft = np.append(Ft, linenum)
                        countFt += 1
                    if i == 34-1+countFt+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        gamma = np.array(linenum)
                        countgamma += 1
                    if i == 34-1+countFt+2+countgamma and countgamma < Y_col:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        gamma = np.append(gamma, linenum, axis = 0)
                        countgamma += 1
                    if i == 34-1+countFt+2+countgamma+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        C = np.array(linenum)
                        countC += 1
                    if i == 34-1+countFt+2+countgamma+2+countC and countC < S_row:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        C = np.append(C, linenum, axis = 0)
                        countC += 1
                    i += 1

            theta1_opt = np.reshape(theta1_opt, (1, theta1_opt.size))
            beta = np.reshape(beta, (1, beta.size))
            sigma2 = np.reshape(sigma2, (1, sigma2.size))
            G = np.reshape(G, (G.size, 1))
            Ft = np.reshape(Ft, (Ft.size, 1))
            gamma = np.reshape(gamma, (countgamma, int(gamma.size/countgamma)))
            C = np.reshape(C, (countC, int(C.size/countC)))
            
        elif self.order == 1:
            # load outkriging file with order 1: especially G, beta
            with open(self.outkrigingFile) as f:
                i = 0
                for line in f.readlines():
                    if i == 10-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        meanS = np.array(linenum)
                    if i == 13-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        meanY = np.array(linenum)
                    if i == 16-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        stdS = np.array(linenum)
                    if i == 19-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        stdY = np.array(linenum)
                    if i == 22-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        theta1_opt = np.array(linenum)

                    if i == 25-1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        beta = np.array(linenum)
                        countbeta += 1
                    if i == 25-1+countbeta and countbeta < S_col+1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        beta = np.append(beta, linenum, axis = 0)
                        countbeta += 1

                    if i == 25-1+countbeta+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        sigma2 = np.array(linenum)
                        countsigma2 += 1

                    if i == 25-1+countbeta+2+countsigma2+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        G = np.array(linenum)
                        countG += 1
                    if i == 25-1+countbeta+2+countsigma2+2+countG and countG < S_col+1:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        G = np.append(G, linenum, axis = 0)
                        countG += 1

                    if i == 25-1+countbeta+2+countsigma2+2+countG+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        Ft = np.array(linenum)
                        countFt += 1
                    if i == 25-1+countbeta+2+countsigma2+2+countG+2+countFt and countFt < S_row:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        Ft = np.append(Ft, linenum, axis = 0)
                        countFt += 1

                    if i == 25-1+countbeta+2+countsigma2+2+countG+2+countFt+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        gamma = np.array(linenum)
                        countgamma += 1
                    if i == 25-1+countbeta+2+countsigma2+2+countG+2+countFt+2+countgamma and countgamma < Y_col:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        gamma = np.append(gamma, linenum, axis = 0)
                        countgamma += 1

                    if i == 25-1+countbeta+2+countsigma2+2+countG+2+countFt+2+countgamma+2:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        C = np.array(linenum)
                        countC += 1
                    if i == 25-1+countbeta+2+countsigma2+2+countG+2+countFt+2+countgamma+2+countC and countC < S_row:
                        linestr = line.strip().split(' ')
                        linenum = [float(lineele) for lineele in linestr]
                        C = np.append(C, linenum, axis = 0)
                        countC += 1
                    i += 1
                    
            theta1_opt = np.reshape(theta1_opt, (1, theta1_opt.size))
            beta = np.reshape(beta, (countbeta, int(beta.size/countbeta)))
            sigma2 = np.reshape(sigma2, (1, sigma2.size))
            G = np.reshape(G, (countG, int(G.size/countG)))
            Ft = np.reshape(Ft, (countFt, int(Ft.size/countFt)))
            gamma = np.reshape(gamma, (countgamma, int(gamma.size/countgamma)))
            C = np.reshape(C, (countC, int(C.size/countC)))
        
        # Design and response sites
        S = copy.deepcopy(SYvalue[:, :S_col])
        Y = copy.deepcopy(SYvalue[:, S_col:])
        X = copy.deepcopy(Xvalue)
        Sname = copy.deepcopy(SYname[:S_col])
        Yname = copy.deepcopy(SYname[S_col:])
        
        [X_row, X_col] = X.shape
        if X_col != S_col:
            sys.exit('Code terminated: # of prediction input variables \
            does not match # of given input variables')
        
        # Step 3: Normalize S, Y, X
        S_norm = (S - np.tile(meanS, [S_row, 1]))/np.tile(stdS, [S_row, 1])
        Y_norm = (Y - np.tile(meanY, [Y_row, 1]))/np.tile(stdY, [Y_row, 1])
        X_norm = (X - np.tile(meanS, [X_row, 1]))/np.tile(stdS, [X_row, 1])

        
        # Step 4: Build regression model
        print('Step 3: Regression model')
        #Calculate dx
        dx = np.empty([X_row*S_row, S_col])
        for j in range(S_col):
            for i in range(X_row*S_row):
                #print(i//S_row)
                #print(i%S_row)
                dx[i, j] = X_norm[i//S_row, j] - S_norm[i%S_row, j]
        #print('dx: ', dx)
        
        #Calculate r
        r = np.empty([X_row*S_row, 1])
        r_tmp = 0.0
        for i in range(X_row*S_row):
            for j in range(S_col):
                r_tmp = r_tmp - theta1_opt[0, j]*dx[i, j]*dx[i, j]
            r[i, 0] = np.exp(r_tmp)
            r_tmp = 0.0
        r_reshape = np.reshape(r, (X_row, S_row)).T
        #print('r: ', r)
        #print('r_reshape: ', r_reshape)
        
        #calculate f
        if self.order == 0:
            f = np.ones([X_row, 1])
        elif self.order == 1:
            f = np.ones([X_row, S_col+1])
            for i in range(S_col):
                for j in range(X_row):
                    f[j, i+1] = X_norm[j, i]
        
        #Calculate prediction Xy
        Xy_norm = np.matmul(f, beta) + np.matmul(gamma, r_reshape).T
        Xy = np.tile(meanY, [X_row, 1]) + np.tile(stdY, [X_row, 1])*Xy_norm
        print('\tFinish Prediction - Xy')
        #print('Finish Prediction - Xy: \n', Xy)
        
        #Calculate MSEs
        rt = np.matmul(la.inv(C), r_reshape)
        #print('rt: ', rt)
        u_tmp = np.matmul(Ft.T, rt)-f.T
        u = la.solve(G, u_tmp)
        #print('u: ', np.sum(u*u, axis = 0))
        
        or1_tmp = 1 + np.sum(u*u, axis = 0) - np.sum(rt*rt, axis = 0)
        # print(or1_tmp)
        or1_tmp = np.reshape(or1_tmp, (1, or1_tmp.size)).T
        # print(or1_tmp)
        or1 = np.abs(np.tile(sigma2, [X_row, 1]) * np.tile(or1_tmp, [1, Y_col]))
        print('\tFinish MSEs - or1')
        #print('Finish MSEs - or1: ', or1)
        
        # print(Xy)
        # print(or1)
        
        # Copy to Global
        [self.S_row, self.Y_row, self.S_col, self.Y_col] = [S_row, Y_row, S_col, Y_col]
        self.S_norm = S_norm
        self.Y_norm = Y_norm
        self.S = S
        self.Y = Y
        [self.stdS, self.stdY] = [stdS, stdY]
        self.X = X
        self.Xy = Xy
        self.X_norm = X_norm
        self.Xy_norm = Xy_norm
        self.MSE = or1
        self.Sname = Sname
        self.Yname = Yname
        
        # Step 5: Write the predictions
        print('Step 4: Write the predictions')
        with open(self.outpredictionFile, 'w') as f:
            for name in Xname:
                f.write(name + '\t')
            for i in range(Y_col):
                f.write('OUT' + str(i+1) + '\t')
            for i in range(Y_col):
                f.write('MSE' + str(i+1) + '\t')
            f.write('\n')
            
            for i in range(X_row):
                for j in range(S_col):
                    f.write('{:11.4E}\t'.format(X[i, j]))
                    #f.write(str(X[i, j]) + '\t')
                for j in range(Y_col):
                    f.write('{:11.4E}\t'.format(Xy[i, j]))
                    #f.write(str(Xy[i, j]) +'\t')
                for j in range(Y_col):
                    f.write('{:11.4E}\t'.format(or1[i, j]))
                    #f.write(str(or1[i, j]) +'\t')
                f.write('\n')
        print('End of code\n')
    
    def buildROM(self, indS = None, indY = None, frac4ROM = 80, filter_enabled = False, z_thres = 5):
        '''
        The function build the ROM for a certain output variable
        '''
        print('############################################################\
              \nBuild the ROM\
              \n############################################################')
        
        # create inKriging.dat
        if os.path.exists(self.allresultsFile) and os.path.exists(self.allresults_infoFile):
            ## Step 1: load all simulation results
            SYname, SYvalue = self.file_read(self.allresultsFile)
            infoname, infovalue = self.file_read(self.allresults_infoFile)
            [S_row, Y_row, S_col, Y_col] = [len(SYvalue), len(SYvalue), int(infovalue[0,0]), int(infovalue[0,1])]
            if indS == None: indS = list(range(1, S_col+1))
            if indY == None: indY = list(range(1, Y_col+1))
            
            ## Step 1.5: filter the noise and remove all failed/unconverged cases
            if SYname[S_col] == 'SimulationStatus':
                cls_enabled = True
            else:
                cls_enabled = False
            
            if cls_enabled == True:
                SYvalue_cov = SYvalue[SYvalue[:, S_col] == 1, :]
            else:
                SYvalue_cov = SYvalue
            
            if filter_enabled == True:
                SY_row_rm = []
                for j in indY:
                    tmp_data = SYvalue_cov[:, S_col+j-1]
                    while(True):
                        z = np.abs(stats.zscore(tmp_data, axis = 0))
                        result = np.where(z > z_thres)
                        index = list(result[0])
                        # line removal list
                        if len(index) == 0: break
                        SY_row_rm += index
                        SY_row_rm = list(dict.fromkeys(SY_row_rm))
                        # replace outliers with mean
                        tmp_data[SY_row_rm] = np.mean(tmp_data)
                # remove rows and columns accroding to SY_row_rm and SY_col_rm
                SYvalue_new = np.delete(SYvalue_cov, SY_row_rm, axis = 0)
                print('Noise filter: trim ' + str(len(SY_row_rm)) + ' rows from a total of ' + str(len(SYvalue_cov)) + ' rows')
            else:
                SYvalue_new = SYvalue_cov
            
            [S_row, Y_row, S_col, Y_col] = [len(SYvalue_new), len(SYvalue_new), int(infovalue[0,0]), int(infovalue[0,1])]
            S = copy.deepcopy(SYvalue_new[:, :S_col])
            Y = copy.deepcopy(SYvalue_new[:, S_col:])
            Sname = copy.deepcopy(SYname[:S_col])
            Yname = copy.deepcopy(SYname[S_col:])

            ## Step 2: compute istep, numcrossvali, rndnumberlist
            if frac4ROM >= 0:
                numtraining = int(S_row*frac4ROM/100.0)
                numcrossvali = S_row-numtraining
                if numtraining < (2**len(indS)): 
                    print('warning: data set to build the ROM is not large enough')
                if numcrossvali > 0:
                    istep = int((S_row)/numcrossvali)
                    rndnumberlist =[]
                    for i in range(1, numcrossvali+1):
                        rndnumberlist.append(i*istep-1)
                else:
                    rndnumberlist =[]
            else:
                numtraining = S_row-1000
                numcrossvali = S_row-numtraining
                rndnumberlist = list(range(numtraining, S_row))

            ## Step 3: write to inkriging.dat, info.dat and inPrediction_vali.dat 
            inpredictionFile4vali = self.work_path + '/inPrediction_vali_kriging.dat'
            f0 = open(self.outcrossvaliFile, 'w')
            f1 = open(self.inkrigingFile, 'w')
            f2 = open(inpredictionFile4vali, 'w')
            f3 = open(self.incrossvaliFile, 'w')
            for i in indS:
                f1.write(Sname[i-1] + '\t')
                f2.write(Sname[i-1] + '\t')
                f3.write(Sname[i-1] + '\t')
            for i in indY:
                f1.write(Yname[i-1] + '\t')
                f3.write(Yname[i-1] + '\t')
            f1.write('\n')
            f2.write('\n')
            f3.write('\n')
            for i in range(S_row):
                if i in rndnumberlist:
                    for j in indS:
                        f2.write('{:11.4E}\t'.format(S[i, j-1]))
                        f3.write('{:11.4E}\t'.format(S[i, j-1]))
                    for j in indY:
                        f3.write('{:11.4E}\t'.format(Y[i, j-1]))
                    f2.write('\n')
                    f3.write('\n')
                else:
                    for j in indS:
                        f1.write('{:11.4E}\t'.format(S[i, j-1]))
                    for j in indY:
                        f1.write('{:11.4E}\t'.format(Y[i, j-1]))
                    f1.write('\n')
            f1.close()
            f2.close()
            f3.close()

            # write info.dat
            with open(self.infoFile, 'w') as f:
                f.write('input_col\toutput_col\n')
                f.write(str(len(indS))+'\t'+str(len(indY))+'\n')

            ## Step 4: perform training and prediction
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_vali_kriging.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_vali_kriging.dat'
            self.training()
            if numcrossvali > 0: 
                self.prediction()
                os.remove(self.inpredictionFile)
                os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig

            ## Step 5: write to outCrossVali.dat
            Yname_new = []
            for i in indY:
                name = Yname[i-1]
                Yname_new.append(name)
                f0.write(name + '\t')
            f0.write('\n')
            for i in range(len(rndnumberlist)):
                for j in range(len(indY)):
                    tempi = rndnumberlist[i]
                    tempj = indY[j]-1
                    f0.write('{:11.4E}\t'.format(self.Xy[i, j]-Y[tempi, tempj]))
                f0.write('\n')
            f0.close()
            
            ## Step 6: write ROM prediction accuracy
            int_95 = self.percent2intervl(95) # 95% confidence interval
            trainingoutput_file = self.outkrigingFile
            trainingoutput_accuracy = trainingoutput_file.replace(".dat", "")+'_acc.dat'
            with open(trainingoutput_accuracy, 'w') as f:
                f.write('ROM Accuracy (95% confidence interval): \n')
                for i in range(len(Yname_new)):
                    f.write(Yname_new[i])
                    f.write('\t' + str(int_95[i]) + '\n')

        elif os.path.exists(self.inkrigingFile) and os.path.exists(self.infoFile):
            self.training()
            
        print('End of code\n')
        
    def percent2intervl(self, percentage, var = None):
        print('############################################################\
              \nPercentage to Confidence Interval\
              \n############################################################')
        
        # load cross validation results
        Yname, ERR = self.file_read(self.outcrossvaliFile)
        
        # find the units
        names_input, units_input, names_output, units_output = self.variable_options()
        Yunit = []
        for i in range(len(Yname)):
            tempindex = names_output.index(Yname[i])
            tempunit = units_output[tempindex]
            Yunit.append(tempunit)
            
        # compute confidence interval
        interval_all = np.zeros((len(Yname),),dtype=np.float64)
        for i in range(len(Yname)):
            err = np.sort(ERR[:, i])
            N = len(err)
            n = (N-1)*percentage/100.0 + 1
            if n == 1: 
                interval = err[0]
            elif n == N:
                interval = err[N-1]
            else:
                k = int(n)
                d = n-k
                interval = err[k-1]+d*(err[k]-err[k-1])
            
            interval_all[i] = interval
            
            if var == None:    
                print('For "' + str(Yname[i]) + '":'
                     + '[' + Yunit[i] + ']'
                     +' \n\t' 
                     + str(percentage) + '% confidence interval is '
                     + '\u00B1' + '{:11.4E}\t'.format(interval))
            elif Yname[i] == var:
                print('For "' + str(Yname[i]) + '":'
                     + '[' + Yunit[i] + ']'
                     +' \n\t' 
                     + str(percentage) + '% confidence interval is '
                     + '\u00B1' + '{:11.4E}\t'.format(interval))
            elif var not in Yname:
                print('The given variable cannot be found')
            
        print('End of code\n')
        return(interval_all)
        
    def intervl2percent(self, interval, var = None):
        print('############################################################\
              \nConfidence Interval to Percentage\
              \n############################################################')
        
        # load cross validation results
        Yname, ERR = self.file_read(self.outcrossvaliFile)
        
        # find the units
        names_input, units_input, names_output, units_output = self.variable_options()
        Yunit = []
        for i in range(len(Yname)):
            tempindex = names_output.index(Yname[i])
            tempunit = units_output[tempindex]
            Yunit.append(tempunit)
        
        # compute confidence percentage
        percentage_all = np.zeros((len(Yname),),dtype=np.float64)
        for i in range(len(Yname)):
            if var == Yname[i]:
                err = np.sort(ERR[:, i])
                N = len(err)
                if interval <= err[0]:
                    percentage = 0
                elif interval >= err[N-1]:
                    percentage = 1
                else:
                    result = np.where(err>interval)
                    index = result[0]
                    k = index[0]
                    percentage = ((interval-err[k-1])/(err[k]-err[k-1])+k-1)/float(N-1)
                
                percentage_all[i] = percentage
                
                print('For "' + str(Yname[i]) + '": ' 
                     + '[' + Yunit[i] + ']'
                     + '\n\t\u00B1' + str(interval)
                     + ' interval has a confidence of ' +  str(round(percentage*100, 2)) + '%')
            elif var not in Yname:
                print('The given variable cannot be found')

        print('End of code\n')
        return(percentage_all)
        
    def plot_contour_2D(self, xvariable, yvariable, zvariable, 
                        pltoption = 0, saveoption = False):
        '''
        The function plots 2D contour of designs and responses
        pltoption = 0: plot both training and prediction sets; 1: plot only training sets, 2: plot only prediction sets
        '''
        # check if the given variables are in the list
        if (xvariable not in self.Sname) or (yvariable not in self.Sname) or (zvariable not in self.Yname):
            sys.exit('Code terminated: variable index out of bound')
        v1 = self.Sname.index(xvariable)+1
        v2 = self.Sname.index(yvariable)+1
        v3 = self.Yname.index(zvariable)+1
        option = int(pltoption)
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_input.index(yvariable)
        yunit = units_input[tempindex]
        tempindex = names_output.index(zvariable)
        zunit = units_output[tempindex]
            
        # Generate inPrediction4contour.dat
        if option == 0 or option == 2:
            Xname, Xvalue = self.file_read(self.inpredictionFile)
            Xvalue_mean = np.mean(Xvalue, axis = 0)
            [X_row, X_col] = Xvalue.shape
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_contour_kriging.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_contour_kriging.dat'
            with open(self.inpredictionFile, 'w') as f:
                for name in Xname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(X_row):
                    for j in range(X_col):
                        if (j+1) == v1 or (j+1) == v2:
                            f.write('{:11.4E}\t'.format(Xvalue[i, j]))
                        else:
                            f.write('{:11.4E}\t'.format(Xvalue_mean[j]))
                    f.write('\n')
            self.prediction()
            os.remove(self.inpredictionFile)
            os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig

        if option == 0: # Default: plot both training and prediction sets
            x1 = self.S[:, v1-1]
            y1 = self.S[:, v2-1]
            z1 = self.Y[:, v3-1]
            x2 = self.X[:, v1-1]
            y2 = self.X[:, v2-1]
            z2 = self.Xy[:, v3-1]
            
            plt.figure(figsize=(17.5,6))
            plt.subplot(1, 2, 1)
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            C = plt.tricontour(x1, y1, z1, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x1, y1, z1, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            #plt.colorbar().set_label(label='a label',size=15,weight='bold')
            plt.xlim((min(min(x1), min(x2)), max(max(x1), max(x2))))
            plt.ylim((min(min(y1), min(y2)), max(max(y1), max(y2))))
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            
            plt.subplot(1, 2, 2)
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            C = plt.tricontour(x2, y2, z2, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x2, y2, z2, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.xlim((min(min(x1), min(x2)), max(max(x1), max(x2))))
            plt.ylim((min(min(y1), min(y2)), max(max(y1), max(y2))))
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 1: # plot training sets
            x = self.S[:, v1-1]
            y = self.S[:, v2-1]
            z = self.Y[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            plt.figure(figsize=(8,6))
            C = plt.tricontour(x, y, z, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x, y, z, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 2: # plot prediciton sets
            x = self.X[:, v1-1]
            y = self.X[:, v2-1]
            z = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            plt.figure(figsize=(8,6))
            C = plt.tricontour(x, y, z, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x, y, z, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        # save option
        if saveoption == True:
            figurename = '2D_contour.png'
            plt.savefig(figurename)
            
    def plot_contour_3D(self, xvariable, yvariable, zvariable, 
                        pltoption = 0, saveoption = False):
        '''
        The function plots 2D contour of designs and responses
        pltoption = 0: plot both training and prediction sets; 1: plot only training sets, 2: plot only prediction sets
        '''
        # check if the given variables are in the list
        if (xvariable not in self.Sname) or (yvariable not in self.Sname) or (zvariable not in self.Yname):
            sys.exit('Code terminated: variable index out of bound')
        v1 = self.Sname.index(xvariable)+1
        v2 = self.Sname.index(yvariable)+1
        v3 = self.Yname.index(zvariable)+1
        option = int(pltoption)
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_input.index(yvariable)
        yunit = units_input[tempindex]
        tempindex = names_output.index(zvariable)
        zunit = units_output[tempindex]
        
        # Generate inPrediction4contour.dat
        if option == 0 or option == 2:
            Xname, Xvalue = self.file_read(self.inpredictionFile)
            Xvalue_mean = np.mean(Xvalue, axis = 0)
            [X_row, X_col] = Xvalue.shape
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_contour_kriging.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_contour_kriging.dat'
            with open(self.inpredictionFile, 'w') as f:
                for name in Xname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(X_row):
                    for j in range(X_col):
                        if (j+1) == v1 or (j+1) == v2:
                            f.write('{:11.4E}\t'.format(Xvalue[i, j]))
                        else:
                            f.write('{:11.4E}\t'.format(Xvalue_mean[j]))
                    f.write('\n')
            self.prediction()
            os.remove(self.inpredictionFile)
            os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig
            
        if option == 0: # Default: plot both training and prediction sets
            x1 = self.S[:, v1-1]
            y1 = self.S[:, v2-1]
            z1 = self.Y[:, v3-1]
            x2 = self.X[:, v1-1]
            y2 = self.X[:, v2-1]
            z2 = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(18.5,6))
            ax = fig.add_subplot(1, 2, 1, projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x1, y1, z1, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            
            ax = fig.add_subplot(1, 2, 2, projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x2, y2, z2, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 1: # plot training sets
            x = self.S[:, v1-1]
            y = self.S[:, v2-1]
            z = self.Y[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x, y, z, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 2: # plot prediciton sets
            x = self.X[:, v1-1]
            y = self.X[:, v2-1]
            z = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x, y, z, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
        
        # save option
        if saveoption == True:
            figurename = '3D_contour.png'
            plt.savefig(figurename)
            
    def plot_box(self, xvariable, yvariable, saveoption = False):
        '''
        The function is for box plot, it can help to perform sensitivity studies
        '''
        # convert to pandam dataframe
        S = pd.DataFrame(data = self.S, columns = self.Sname, dtype = 'float')
        Y = pd.DataFrame(data = self.Y, columns = self.Yname, dtype = 'float')
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_output.index(yvariable)
        yunit = units_output[tempindex]
        
        # generate box plot data
        x = S[[xvariable]]
        y = Y[[yvariable]]
        min_x = min(x.values)
        max_x = max(x.values)
        x = round((x-min_x)/((max_x-min_x)/9), 0)*((max_x-min_x)/9)+min_x
        x = round(x, 2)
        #xy = pd.concat([x, y], axis = 1, sort = False)
        #print(x.sort_values(by = ['Average_CurrentDensity']))
        #print(xy)
        
        # box plot
        plt.figure(figsize=(18.5,6))
        sns.set_context("paper", font_scale=3)
        sns.set_style('ticks')
        bplot = sns.boxplot(y=y[yvariable], x=x[xvariable], 
                    color = 'yellow', width = 0.5)
        bplot = sns.swarmplot(y=y[yvariable], x=x[xvariable], 
                    color = 'black', alpha = 0.5)
        sns.axes_style()
        bplot.axes.set_title('Design-response sites', fontsize = 25)
        bplot.set_xlabel(xvariable+', ['+xunit+']', fontsize = 25)
        bplot.set_ylabel(yvariable+', ['+yunit+']', fontsize = 25)
        bplot.tick_params(labelsize = 25)
        plt.show()
        
        # save option
        if saveoption == True:
            figurename = 'boxplot.png'
            plt.savefig(figurename)
        
    def Generate_inprediction(self, numsample = None, listmin = None, listmax = None):
        '''
        The function generates prediction input if it doesn't exist by Latin Hypercube Sampling
        '''
        print('############################################################\
                \nGenerate prediction input\
                \n############################################################')

        # find input variable list Sname
        SYname, SYvalue = self.file_read(self.inkrigingFile)
        infoname, infovalue = self.file_read(self.infoFile)
        [S_col, Y_col] = [int(infovalue[0,0]), int(infovalue[0,1])]
        Sname = copy.deepcopy(SYname[:S_col])
        
        # check if exists
        filename  = self.inpredictionFile
        Create_handle = True
        if os.path.exists(filename):
            query = query_yes_no('Prediction input file already exists on the local machine, do you want to overwrite it?')
            Create_handle = query

        if Create_handle == True:
            numvar = len(Sname)
            listvar = Sname
            if len(listmin) != numvar or len(listmax) != numvar:
                sys.exit('Code terminated: the lengths of variables/minimums/maximums not match')

            # LHS sampling
            xlimits = np.transpose(np.vstack((listmin, listmax)))
            sampling = LHS(xlimits = xlimits)
            LHSvalue = sampling(numsample)

            # write prediction input
            with open(filename, 'w') as f:
                for name in Sname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(numsample):
                    for j in range(numvar):
                        f.write('{:11.4E}\t'.format(LHSvalue[i, j]))
                    f.write('\n')

            print("Created prediciton input file")
            print('End of code\n')
            
class DNN():
    def __init__(self, work_path, 
                 allresultsFile = 'allResults.dat',
                 allresults_infoFile = 'allResults_info.dat',
                 intrainingFile = 'inTraining_DNN.dat', 
                 infoFile = 'info_DNN.dat', 
                 outtrainingFile = 'outTraining_DNN.dat', 
                 inpredictionFile = 'inPrediction_DNN.dat', 
                 outpredictionFile = 'outPrediction_DNN.dat', 
                 incrossvaliFile = 'inCrossVali_DNN.dat',
                 outcrossvaliFile = 'outCrossVali_DNN.dat'):
        self.work_path = work_path
        self.allresultsFile = work_path + '/' + allresultsFile
        self.allresults_infoFile = work_path + '/' + allresults_infoFile
        self.intrainingFile = work_path + '/' + intrainingFile
        self.infoFile = work_path + '/' + infoFile
        self.outtrainingFile = work_path + '/' + outtrainingFile
        self.inpredictionFile = work_path + '/' + inpredictionFile
        self.outpredictionFile = work_path + '/' + outpredictionFile
        self.incrossvaliFile = work_path + '/' + incrossvaliFile
        self.outcrossvaliFile = work_path + '/' + outcrossvaliFile
        self.Sname = None
        self.Yname = None
        self.S_norm = None
        self.Y_norm = None
        self.X_norm = None
        self.Xy_norm = None
        self.S = None
        self.Y = None
        self.X = None
        self.Xy = None
        self.MSE = None
        self.S_row = 0
        self.Y_row = 0
        self.S_col = 0
        self.Y_col = 0
        self.stdS = None
        self.stdY = None
        self.meanS = None
        self.meanY = None
    
    #%% The DNN function for ROM, save the trained DNN
    def DNNROM(self,maxiteration,trainX_nrm,trainY_nrm,testX_nrm1,input_num,output_num,DNN_save_file):

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]

        learning_rate = 0.001
        training_epochs = maxiteration
        batch_size = int(X_train.shape[0]/3)
        total_len=trainX_nrm.shape[0]

        seed=88
        print("DNN ROM training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for testing data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])

        # Network Parameters
        n_hidden_1 = 32#64 
        n_hidden_2 = 200#400 
        n_hidden_3 = 200#400
        n_hidden_4 = 256#512
        n_input = input_num
        n_classes = output_num

        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
        y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1,seed=seed)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1,seed=seed)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1,seed=seed)),
            'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1,seed=seed)),
            'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1,seed=seed))
        }

        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            tf.summary.histogram("weights",weights['h1'])
            tf.summary.histogram("layer", layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.sigmoid(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.sigmoid(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
            layer_4 = tf.nn.sigmoid(layer_4)

            out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            return out_layer   

        # Construct model
        pred = multilayer_perceptron(x)
        cost = tf.reduce_mean(tf.square(pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Run the graph in the session 
        predict = np.array([])
        count_converge= [0] * training_epochs
        prev_cost=10000000.
        saver = tf.train.Saver()
        #tf.reset_default_graph()

        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(total_len/batch_size)
                for i in range(total_batch-1):
                    batch_x = X_train[i*batch_size:(i+1)*batch_size]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size]
                    _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                    if epoch==training_epochs-1:
                        predict = np.append(predict, p)
               # print  ('epoch', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost))
                val_c, val_p=sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
                test_p1=sess.run(pred, feed_dict={x: testX_nrm1})

                #count cost convergence for validation
                count_converge[epoch]=val_c
                if epoch == training_epochs-1:
                        print('break the loop at maximum iteration')
                if epoch %2000 == 0 :
                    print  ('epoch ',(epoch+1),'  training cost =','{:.5f}'.format(avg_cost),'  validation cost =', '{:.5f}'.format(val_c))
                #for validation set if no improvement then break
                if epoch %2000 ==0 and val_c>=prev_cost:
                    break
                    #print("val cost increase !!!")
                if epoch %2000 ==0:
                    prev_cost=val_c
            saver.save(sess, DNN_save_file)
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            #for k,v in zip(variables_names, values):
            # print(k, v)
            # for v in values: 
            #     print(v) 
            sess.close()

        tf.reset_default_graph()
        return(test_p1, values)

    #%% The DNN function for ROM, save the trained DNN
    def DNNROM2(self,maxiteration,trainX_nrm,trainY_nrm,testX_nrm1,input_num,output_num,DNN_save_file, DNNsize):
        
        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]
        
        learning_rate = 0.001
        training_epochs = maxiteration
        batch_size = int(X_train.shape[0]/3)
        total_len=trainX_nrm.shape[0]
        
        seed=88
        print("DNN ROM training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for testing data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])
        
        # Network Parameters
        DNNlayers=len(DNNsize)
        print('Number of layers = ',DNNlayers)
        if DNNlayers>10:
            print('Number of layers needs <=10')
            return()
        if DNNlayers>=1: n_hidden_1 = DNNsize[0]#64 
        if DNNlayers>=2: n_hidden_2 = DNNsize[1]#400 
        if DNNlayers>=3: n_hidden_3 = DNNsize[2]#400
        if DNNlayers>=4: n_hidden_4 = DNNsize[3]#512
        if DNNlayers>=5: n_hidden_5 = DNNsize[4]#512
        if DNNlayers>=6: n_hidden_6 = DNNsize[5]#512
        if DNNlayers>=7: n_hidden_7 = DNNsize[6]#512
        if DNNlayers>=8: n_hidden_8 = DNNsize[7]#512
        if DNNlayers>=9: n_hidden_9 = DNNsize[8]#512
        if DNNlayers>=10: n_hidden_10 = DNNsize[9]#512

        n_input = input_num
        n_classes = output_num
        
        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
        y = tf.placeholder("float", [None, n_classes])
        #tf.compat.v1.disable_eager_execution()
        
        # Store layers weight & bias
        if DNNlayers==1:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==2:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }    
        if DNNlayers==3:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_3, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==4:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==5:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_5, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==6:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_6, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==7:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_7, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==8:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_8, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==9:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_9, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==10:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'h10': tf.Variable(tf.random.normal([n_hidden_9, n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_10, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'b10': tf.Variable(tf.random.normal([n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        
        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            print(DNNlayers)
            if DNNlayers>=1:
                layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
                layer_1 = tf.nn.sigmoid(layer_1)
                #tf.summary.histogram("weights",weights['h1'])
                #tf.summary.histogram("layer", layer_1)
                
            if DNNlayers>=2:   
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                layer_2 = tf.nn.sigmoid(layer_2)
            if DNNlayers>=3:             
                layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
                layer_3 = tf.nn.sigmoid(layer_3)
            if DNNlayers>=4:  
                layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
                layer_4 = tf.nn.sigmoid(layer_4)
            if DNNlayers>=5:  
                layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
                layer_5 = tf.nn.sigmoid(layer_5)
            if DNNlayers>=6:  
                layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
                layer_6 = tf.nn.sigmoid(layer_6)
            if DNNlayers>=7:  
                layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
                layer_7 = tf.nn.sigmoid(layer_7)
            if DNNlayers>=8:  
                layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
                layer_8 = tf.nn.sigmoid(layer_8)
            if DNNlayers>=9:  
                layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
                layer_9 = tf.nn.sigmoid(layer_9)
            if DNNlayers>=10:  
                layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
                layer_10 = tf.nn.sigmoid(layer_10)            
            if DNNlayers==1:
                out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
            if DNNlayers==2:
                out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            if DNNlayers==3:
                out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
            if DNNlayers==4:
                out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            if DNNlayers==5:
                out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
            if DNNlayers==6:
                out_layer = tf.matmul(layer_6, weights['out']) + biases['out']
            if DNNlayers==7:
                out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
            if DNNlayers==8:
                out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
            if DNNlayers==9:
                out_layer = tf.matmul(layer_9, weights['out']) + biases['out']
            if DNNlayers==10:
                out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
            return out_layer   
        
        # Construct model
        pred = multilayer_perceptron(x)
        cost = tf.reduce_mean(tf.square(pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        # Run the graph in the session 
        predict = np.array([])
        count_converge= [0] * training_epochs
        prev_cost=10000000.
        saver = tf.train.Saver()
        #tf.reset_default_graph()
        
        config = tf.ConfigProto(device_count={"CPU": 1}, # limit to num_cpu_core CPU usage
                    inter_op_parallelism_threads = 0, 
                    intra_op_parallelism_threads = 28,
                    )
        init = tf.global_variables_initializer()
        start=time.time()
        with tf.Session(config=config) as sess:
            
            sess.run(init)
            for epoch in range(training_epochs):
                    
                    avg_cost = 0.
                    total_batch = int(total_len/batch_size)
                    for i in range(total_batch-1):
                        batch_x = X_train[i*batch_size:(i+1)*batch_size]
                        batch_y = y_train[i*batch_size:(i+1)*batch_size]
                        _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                        avg_cost += c / total_batch
                        if epoch==training_epochs-1:
                            predict = np.append(predict, p)
                # print  ('epoch', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost))
                    val_c, val_p=sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
                    test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
                    
                    #count cost convergence for validation
                    count_converge[epoch]=val_c
                    
                    if epoch %2000 == 0 :
                        end=time.time()
                        print  ('epoch ',(epoch+1),'  training cost =','{:.5f}'.format(avg_cost),'  validation cost =', '{:.5f}'.format(val_c),'  training time (s/100epochs)= ','{:.5f}'.format(end-start))
                        start=time.time()
                    #for validation set if no improvement then break
                    if epoch == training_epochs-1:
                        print('break the loop at maximum iteration')
                    if epoch %2000 ==0 and val_c>=prev_cost:
                        break
                        #print("val cost increase !!!")
                    if epoch %2000 ==0:
                        prev_cost=val_c
            #saver.save(sess, r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\DNN')
            saver.save(sess, DNN_save_file)
            
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            #for k,v in zip(variables_names, values):
            # print(k, v)
            # for v in values: 
            #     print(v) 
            sess.close()
        
        tf.reset_default_graph()
        return(test_p1, values)

    #%% The DNN function for ROM, load in a trained DNN, and continue training
    def DNNROM_restore(self,maxiteration,trainX_nrm,trainY_nrm,testX_nrm1,input_num,output_num,DNN_load_file,DNN_save_file):

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]

        learning_rate = 0.001
        training_epochs = maxiteration
        batch_size = int(X_train.shape[0]/3)
        total_len=trainX_nrm.shape[0]

        seed=88
        print("DNN ROM training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for testing data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])

        # Network Parameters
        n_hidden_1 = 32#64 
        n_hidden_2 = 200#400 
        n_hidden_3 = 200#400
        n_hidden_4 = 256#512
        n_input = input_num
        n_classes = output_num

        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
        y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1,seed=seed)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1,seed=seed)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1,seed=seed)),
            'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1,seed=seed)),
            'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1,seed=seed))
        }

        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            tf.summary.histogram("weights",weights['h1'])
            tf.summary.histogram("layer", layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.sigmoid(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.sigmoid(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
            layer_4 = tf.nn.sigmoid(layer_4)

            out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            return out_layer   

        # Construct model
        pred = multilayer_perceptron(x)
        cost = tf.reduce_mean(tf.square(pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Run the graph in the session 
        predict = np.array([])
        count_converge= [0] * training_epochs
        prev_cost=10000000.
        saver = tf.train.Saver()
        #tf.train.latest_checkpoint(r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\checkpoint')
        #init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver.restore(sess,  DNN_load_file)
            #sess.run(init)
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(total_len/batch_size)
                for i in range(total_batch-1):
                    batch_x = X_train[i*batch_size:(i+1)*batch_size]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size]
                    _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                    if epoch==training_epochs-1:
                        predict = np.append(predict, p)
               # print  ('epoch', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost))
                val_c, val_p=sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
                test_p1=sess.run(pred, feed_dict={x: testX_nrm1})

                #count cost convergence for validation
                count_converge[epoch]=val_c
                if epoch == training_epochs-1:
                        print('break the loop at maximum iteration')
                if epoch %2000 == 0 :print  ('epoch ',(epoch+1),'  training cost =','{:.5f}'.format(avg_cost),'  validation cost =', '{:.5f}'.format(val_c))
                #for validation set if no improvement then break
                if epoch %2000 ==0 and val_c>=prev_cost:
                    break
                    #print("val cost increase !!!")
                if epoch %2000 ==0:
                    prev_cost=val_c
            saver.save(sess, DNN_save_file)
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            #for k,v in zip(variables_names, values):
            # print(k, v)
            # for v in values: 
            #     print(v)    
            sess.close()

        tf.reset_default_graph()
        return(test_p1,values)

    #%% The DNN function for ROM, load in a trained DNN, and continue training
    def DNNROM_restore2(self,maxiteration,trainX_nrm,trainY_nrm,testX_nrm1,input_num,output_num,DNN_load_file,DNN_save_file, DNNsize):

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]

        learning_rate = 0.001
        training_epochs = maxiteration
        batch_size = int(X_train.shape[0]/3)
        total_len=trainX_nrm.shape[0]

        seed=88
        print("DNN ROM training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for testing data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])
        DNNlayers=len(DNNsize)
        print('Number of layers = ',DNNlayers)
        if DNNlayers>10:
            print('Number of layers needs <=10')
            return()
        if DNNlayers>=1: n_hidden_1 = DNNsize[0]#64 
        if DNNlayers>=2: n_hidden_2 = DNNsize[1]#400 
        if DNNlayers>=3: n_hidden_3 = DNNsize[2]#400
        if DNNlayers>=4: n_hidden_4 = DNNsize[3]#512
        if DNNlayers>=5: n_hidden_5 = DNNsize[4]#512
        if DNNlayers>=6: n_hidden_6 = DNNsize[5]#512
        if DNNlayers>=7: n_hidden_7 = DNNsize[6]#512
        if DNNlayers>=8: n_hidden_8 = DNNsize[7]#512
        if DNNlayers>=9: n_hidden_9 = DNNsize[8]#512
        if DNNlayers>=10: n_hidden_10 = DNNsize[9]#512

        n_input = input_num
        n_classes = output_num

        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
        y = tf.placeholder("float", [None, n_classes])
        #tf.compat.v1.disable_eager_execution()

        # Store layers weight & bias
        if DNNlayers==1:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==2:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }    
        if DNNlayers==3:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_3, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }

        if DNNlayers==4:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }

        if DNNlayers==5:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_5, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==6:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_6, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==7:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_7, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }

        if DNNlayers==8:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_8, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==9:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_9, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==10:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'h10': tf.Variable(tf.random.normal([n_hidden_9, n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_10, n_classes], 0, 0.1,seed=seed))
            }

            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'b10': tf.Variable(tf.random.normal([n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }

        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            print(DNNlayers)
            if DNNlayers>=1:
                layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
                layer_1 = tf.nn.sigmoid(layer_1)
                #tf.summary.histogram("weights",weights['h1'])
                #tf.summary.histogram("layer", layer_1)

            if DNNlayers>=2:   
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                layer_2 = tf.nn.sigmoid(layer_2)
            if DNNlayers>=3:             
                layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
                layer_3 = tf.nn.sigmoid(layer_3)
            if DNNlayers>=4:  
                layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
                layer_4 = tf.nn.sigmoid(layer_4)
            if DNNlayers>=5:  
                layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
                layer_5 = tf.nn.sigmoid(layer_5)
            if DNNlayers>=6:  
                layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
                layer_6 = tf.nn.sigmoid(layer_6)
            if DNNlayers>=7:  
                layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
                layer_7 = tf.nn.sigmoid(layer_7)
            if DNNlayers>=8:  
                layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
                layer_8 = tf.nn.sigmoid(layer_8)
            if DNNlayers>=9:  
                layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
                layer_9 = tf.nn.sigmoid(layer_9)
            if DNNlayers>=10:  
                layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
                layer_10 = tf.nn.sigmoid(layer_10)            
            if DNNlayers==1:
                out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
            if DNNlayers==2:
                out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            if DNNlayers==3:
                out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
            if DNNlayers==4:
                out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            if DNNlayers==5:
                out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
            if DNNlayers==6:
                out_layer = tf.matmul(layer_6, weights['out']) + biases['out']
            if DNNlayers==7:
                out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
            if DNNlayers==8:
                out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
            if DNNlayers==9:
                out_layer = tf.matmul(layer_9, weights['out']) + biases['out']
            if DNNlayers==10:
                out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
            return out_layer   


        # Construct model
        pred = multilayer_perceptron(x)
        cost = tf.reduce_mean(tf.square(pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Run the graph in the session 
        predict = np.array([])
        count_converge= [0] * training_epochs
        prev_cost=10000000.
        saver = tf.train.Saver()
        #tf.train.latest_checkpoint(r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\checkpoint')
        #init = tf.global_variables_initializer()
        start=time.time()
        with tf.Session() as sess:
            saver.restore(sess,  DNN_load_file)
            #sess.run(init)
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(total_len/batch_size)
                for i in range(total_batch-1):
                    batch_x = X_train[i*batch_size:(i+1)*batch_size]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size]
                    _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                    avg_cost += c / total_batch
                    if epoch==training_epochs-1:
                        predict = np.append(predict, p)
               # print  ('epoch', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost))
                val_c, val_p=sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
                test_p1=sess.run(pred, feed_dict={x: testX_nrm1})

                #count cost convergence for validation
                count_converge[epoch]=val_c
                if epoch == training_epochs-1:
                        print('break the loop at maximum iteration')
                if epoch %2000 == 0 :
                    end=time.time()
                    print  ('epoch ',(epoch+1),'  training cost =','{:.5f}'.format(avg_cost),'  validation cost =', '{:.5f}'.format(val_c),'  training time (s/100epochs)= ','{:.5f}'.format(end-start))
                    start=time.time()
                #for validation set if no improvement then break
                if epoch %2000 ==0 and val_c>=prev_cost:
                    break
                    #print("val cost increase !!!")
                if epoch %2000 ==0:
                    prev_cost=val_c
            saver.save(sess, DNN_save_file)
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            # for k,v in zip(variables_names, values):
            #     print(k, v)
            # for v in values: 
            #     print(v)    
            sess.close()

        tf.reset_default_graph()
        return(test_p1,values)

    #%% The DNN function for ROM, load in a trained DNN, and do prediction
    def DNNROM_prediction(self,testX_nrm1,input_num,output_num,DNN_load_file):

        #split_size = int(trainX_nrm.shape[0]*0.8)

        #X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        #y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]

        #learning_rate = 0.001
        #training_epochs = 0
        #batch_size = int(X_train.shape[0]/3)
        #total_len=trainX_nrm.shape[0]

        seed=88
        print("DNN ROM predicting start ...")
        #print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        #print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for class training data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])

        # Network Parameters
        n_hidden_1 = 32#64 
        n_hidden_2 = 200#400 
        n_hidden_3 = 200#400
        n_hidden_4 = 256#512
        n_input = input_num
        n_classes = output_num

        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
       # y = tf.placeholder("float", [None, n_classes])

        # Store layers weight & bias
        weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
            'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
            'out': tf.Variable(tf.random_normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
        }

        biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1], 0, 0.1,seed=seed)),
            'b2': tf.Variable(tf.random_normal([n_hidden_2], 0, 0.1,seed=seed)),
            'b3': tf.Variable(tf.random_normal([n_hidden_3], 0, 0.1,seed=seed)),
            'b4': tf.Variable(tf.random_normal([n_hidden_4], 0, 0.1,seed=seed)),
            'out': tf.Variable(tf.random_normal([n_classes], 0, 0.1,seed=seed))
        }

        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
            layer_1 = tf.nn.sigmoid(layer_1)
            tf.summary.histogram("weights",weights['h1'])
            tf.summary.histogram("layer", layer_1)

            layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
            layer_2 = tf.nn.sigmoid(layer_2)

            layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
            layer_3 = tf.nn.sigmoid(layer_3)

            layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
            layer_4 = tf.nn.sigmoid(layer_4)

            out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            return out_layer   

        # Construct model
        pred = multilayer_perceptron(x)
        #cost = tf.reduce_mean(tf.square(pred-y))
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Run the graph in the session 
        #predict = np.array([])
        #count_converge= [0] * training_epochs
        #prev_cost=10000000.
        saver = tf.train.Saver()
        #tf.train.latest_checkpoint(r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\checkpoint')
        #init = tf.global_variables_initializer()
        with tf.Session() as sess:
            saver.restore(sess,  DNN_load_file)
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            sess.close()

        tf.reset_default_graph()
        return(test_p1)

    #%% DNN classification       one layer, train DNN classifier, and save DNN
    def DNNCls(self,maxiteration,trainX_nrm,trainY_nrm,input_num_units,DNNcls_save_file):

        hidden_num_units = 500
        output_num_units = 2
        seed=88

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]
        print("DNN classification training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])   
        # print("prediction for testing data set size ", testX_nrm.shape[0]," * ",testX_nrm.shape[1])

        # define placeholders
        xc = tf.placeholder(tf.float32, [None, input_num_units])
        yc = tf.placeholder(tf.float32, [None, output_num_units])

        # set remaining variables
        epochs = maxiteration
        batch_size = int(X_train.shape[0]/2)  #1500
        learning_rate = 0.001

        ### define weights and biases of the neural network 

        weights = {
            'hidden': tf.Variable(tf.random_uniform([input_num_units, hidden_num_units],-1,1,seed=seed)),
            #'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], 0, 1,seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units],0, 0.1, seed=seed))
        }

        biases = {
            #'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'hidden': tf.Variable(tf.random_uniform([hidden_num_units], -1,1,seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        } 
        #
        hidden_layer = tf.add(tf.matmul(xc, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        tf.summary.histogram("weights_hidden",weights['hidden'])
        tf.summary.histogram("biases_hidden",biases['hidden'])
        tf.summary.histogram("layer_hidden", hidden_layer)

        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
        tf.summary.histogram("weights_output",weights['output'])
        tf.summary.histogram("biases_output",biases['output'])
        tf.summary.histogram("layer_output", output_layer)


        #
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=yc))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        pred=output_layer
        #tf.summary.scalar('cost',cost)

        init = tf.global_variables_initializer()

        #write this after all the summary
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter(cwd)
        # covert output scalar to vector https://stackoverflow.com/questions/43543594/label-scalar-into-one-hot-in-tensorr-flow-code
        def dense_to_one_hot(labels_dense, num_classes=2):
            """Convert class labels from scalars to one-hot vectors"""
            num_labels = labels_dense.shape[0]
            #index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            for ii in range(num_labels):
                labels_one_hot[ii,int(labels_dense[ii])]=1
            #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

            return labels_one_hot

        prev_cost=0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                avg_cost = 0
                total_batch = int(X_train.shape[0]/batch_size)
                for i in range(total_batch):

                    batch_x = X_train[i*batch_size:(i+1)*batch_size,]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size,]
                    batch_y = dense_to_one_hot(batch_y)
                    _, c = sess.run([optimizer, cost], feed_dict = {xc: batch_x, yc: batch_y})
                    avg_cost += c / total_batch

                #write tensorboard summary
                #summary_avg_cost = tf.Summary()
                #summary_avg_cost.value.add(tag="avg_cost", simple_value=avg_cost)
                #writer.add_summary(summary_avg_cost, epoch)
                #writer.add_summary(summary, epoch)
            # find predictions on val set #location of the catagory, can be greater than 2
                pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(yc, 1))
               # pred_temp2= tf.argmax(output_layer, 1)
                accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
                val_acc=accuracy.eval({xc: val_x, yc: dense_to_one_hot(val_y)})
               # test_acc=accuracy.eval({xc: testX_nrm, yc: dense_to_one_hot(testY_nrm)})

                #print ("Validation Accuracy:", accuracy.eval({x: val_x, y: dense_to_one_hot(val_y)})) 
                if epoch == epochs-1:
                        print('break the loop at maximum iteration')
                if epoch %2000 ==0 :print ('Epoch:', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost),
                                           "  Validation accuracy:", val_acc," ") 
                if epoch %2000 ==0 and val_acc<=prev_cost:
                    break

                #print("val cost increase !!!")
                if epoch %2000 ==0:
                    prev_cost=val_acc

            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            saver.save(sess, DNNcls_save_file)
            sess.close()
        tf.reset_default_graph()
        return(val_acc, values)

    #%% DNN classification       one layer, load in a trained DNN, and continue training
    def DNNCls_restore(self,maxiteration,trainX_nrm,trainY_nrm,input_num_units,DNNcls_load_file,DNNcls_save_file):
     #   input_num_units = 55
        hidden_num_units = 500
        output_num_units = 2
        seed=88

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]
        print("DNN classification training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])   
        # print("prediction for testing data set size ", testX_nrm.shape[0]," * ",testX_nrm.shape[1])

        # define placeholders
        xc = tf.placeholder(tf.float32, [None, input_num_units])
        yc = tf.placeholder(tf.float32, [None, output_num_units])

        # set remaining variables
        epochs = maxiteration
        batch_size = int(X_train.shape[0]/2)  #1500
        learning_rate = 0.001

        ### define weights and biases of the neural network 

        weights = {
            'hidden': tf.Variable(tf.random_uniform([input_num_units, hidden_num_units],-1,1,seed=seed)),
            #'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], 0, 1,seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units],0, 0.1, seed=seed))
        }

        biases = {
            #'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'hidden': tf.Variable(tf.random_uniform([hidden_num_units], -1,1,seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        } 
        #
        hidden_layer = tf.add(tf.matmul(xc, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        tf.summary.histogram("weights_hidden",weights['hidden'])
        tf.summary.histogram("biases_hidden",biases['hidden'])
        tf.summary.histogram("layer_hidden", hidden_layer)

        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
        tf.summary.histogram("weights_output",weights['output'])
        tf.summary.histogram("biases_output",biases['output'])
        tf.summary.histogram("layer_output", output_layer)


        #
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=yc))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        pred=output_layer
        #tf.summary.scalar('cost',cost)

        #init = tf.global_variables_initializer()

        #write this after all the summary
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter(cwd)
        # covert output scalar to vector https://stackoverflow.com/questions/43543594/label-scalar-into-one-hot-in-tensorr-flow-code
        def dense_to_one_hot(labels_dense, num_classes=2):
            """Convert class labels from scalars to one-hot vectors"""
            num_labels = labels_dense.shape[0]
            #index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            for ii in range(num_labels):
                labels_one_hot[ii,int(labels_dense[ii])]=1
            #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

            return labels_one_hot

        prev_cost=0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,  DNNcls_load_file)
            #sess.run(init)


            for epoch in range(epochs):
                avg_cost = 0
                total_batch = int(X_train.shape[0]/batch_size)
                for i in range(total_batch):

                    batch_x = X_train[i*batch_size:(i+1)*batch_size,]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size,]
                    batch_y = dense_to_one_hot(batch_y)
                    _, c = sess.run([optimizer, cost], feed_dict = {xc: batch_x, yc: batch_y})
                    avg_cost += c / total_batch

                #write tensorboard summary
                #summary_avg_cost = tf.Summary()
                #summary_avg_cost.value.add(tag="avg_cost", simple_value=avg_cost)
                #writer.add_summary(summary_avg_cost, epoch)
                #writer.add_summary(summary, epoch)
                # find predictions on val set #location of the catagory, can be greater than 2
                pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(yc, 1))
                # pred_temp2= tf.argmax(output_layer, 1)
                accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
                val_acc=accuracy.eval({xc: val_x, yc: dense_to_one_hot(val_y)})
               # test_acc=accuracy.eval({xc: testX_nrm, yc: dense_to_one_hot(testY_nrm)})

                #print ("Validation Accuracy:", accuracy.eval({x: val_x, y: dense_to_one_hot(val_y)})) 
                if epoch == epochs-1:
                        print('break the loop at maximum iteration')
                if epoch %2000 ==0 :
                    print ('Epoch:', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost),"  Validation accuracy:", val_acc," ") 
                if epoch %2000 ==0 and val_acc<=prev_cost:
                    break
                #print("val cost increase !!!")
                if epoch %2000 ==0:
                    prev_cost=val_acc

            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            saver.save(sess, DNNcls_save_file)
            sess.close()
        tf.reset_default_graph()
        return(values)

    #%% DNN classification       one layer, load in a trained DNN, and do preidction for classification
    def DNNCls_prediction(self,testX_nrm,input_num_units,DNNcls_load_file):
     #   input_num_units = 55
        hidden_num_units = 500
        output_num_units = 2
        seed=88

     #   split_size = int(trainX_nrm.shape[0]*0.8)

     #   X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
     #   y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]
        print("DNN classification prediction start ...")
     #   print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
     #   print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])   
        print("prediction for testing data set size ", testX_nrm.shape[0]," * ",testX_nrm.shape[1])

        # define placeholders
        xc = tf.placeholder(tf.float32, [None, input_num_units])
        yc = tf.placeholder(tf.float32, [None, output_num_units])

        # set remaining variables
     #   epochs = 5000
     #   batch_size = int(X_train.shape[0]/2)  #1500
     #   learning_rate = 0.001

        ### define weights and biases of the neural network 

        weights = {
            'hidden': tf.Variable(tf.random_uniform([input_num_units, hidden_num_units],-1,1,seed=seed)),
            #'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], 0, 1,seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units],0, 0.1, seed=seed))
        }

        biases = {
            #'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'hidden': tf.Variable(tf.random_uniform([hidden_num_units], -1,1,seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        } 
        #
        hidden_layer = tf.add(tf.matmul(xc, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        tf.summary.histogram("weights_hidden",weights['hidden'])
        tf.summary.histogram("biases_hidden",biases['hidden'])
        tf.summary.histogram("layer_hidden", hidden_layer)

        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
        tf.summary.histogram("weights_output",weights['output'])
        tf.summary.histogram("biases_output",biases['output'])
        tf.summary.histogram("layer_output", output_layer)


        #
      #  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=yc))

      #  optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        pred=output_layer
        #tf.summary.scalar('cost',cost)

        #init = tf.global_variables_initializer()

        #write this after all the summary
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter(cwd)
        # covert output scalar to vector https://stackoverflow.com/questions/43543594/label-scalar-into-one-hot-in-tensorr-flow-code
        def dense_to_one_hot(labels_dense, num_classes=2):
            """Convert class labels from scalars to one-hot vectors"""
            num_labels = labels_dense.shape[0]
            #index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            for ii in range(num_labels):
                labels_one_hot[ii,int(labels_dense[ii])]=1
            #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot

      #  prev_cost=0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,  DNNcls_load_file)
            #sess.run(init)
            test_p1=sess.run(pred, feed_dict={xc: testX_nrm})
            test_p0=sess.run(tf.argmax(test_p1,1))
            #saver.save(sess, r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\ClsDNN')
            sess.close()
        tf.reset_default_graph()
        return(test_p0)

    def summarize_SimuResult(self, source_path, indcase, exclude_case = 1, display_detail = False):
        '''
        The function extracts simulation results
            exclude_case = -1: all cases included
            exclude_case = 0: exclude failed cases only
            exclude_case = 1: exclude both failed and non-converged cases
        '''
        print('############################################################\
              \nSummarize simulation results\
              \n############################################################')
        
        ## Step 1: load simulation outputs to Y4kriging
        numcase4kriging = 0 # number of cases for kriging
        indcase4kriging = [] # index of cases for kriging, start from 1
        S4kriging = None # simulation inputs for kriging
        Y4kriging = None # simulation outputs for kriging
        for icase in indcase:
            # load SOFC_MP_ROM.dat to df1
            strcase = 'Case'+str(icase-1)+'Value'
            inputfilename = source_path+'/Cases/Case'+str(icase-1).zfill(5)+'/SOFC_MP_ROM.dat'
            if os.path.exists(inputfilename):
                text_input=open(inputfilename,"r")
                lines=text_input.readlines()
                
                if len(lines) == 0: 
                    continue #print('Empty case')
                if lines[1].strip() == '#FAILED': 
                    continue #print('"preprocessor" failed case')
                
                df0 = pd.DataFrame(np.array([['1a', '1b']]),columns=['Name', strcase])
                df1 = pd.DataFrame(np.array([['1a', '1b']]),columns=['Name', strcase])
                for j in range(len(lines)):
                    if j>1: # skip first two lines
                        str01 = lines[j].split('=')
                        str01[0]=str01[0].rstrip()
                        str01[0]=str01[0].lstrip()
                        
                        if len(str01) == 1: continue
                        
                        # convert variables in SOFC_MP_ROM.dat to xxx_xxx format
                        str_tmp = str01[0].strip().split()
                        str_tmp = '_'.join(str_tmp)
                        df0['Name']=str_tmp
                        df0[strcase]=float(str01[1])
                        if j==2:
                            df1["Name"]=df0["Name"]
                            df1[strcase]=df0[strcase]
                        else:
                            df1=pd.concat([df1,df0],sort=False, ignore_index=True)
                
                # exclude failed or non-converged cases
                if int(df1.loc[0, [strcase]]) >= exclude_case:
                    numcase4kriging += 1
                    indcase4kriging.append(icase)
                    if numcase4kriging == 1:
                        Y4kriging = df1
                    else:
                        Y4kriging = pd.concat([Y4kriging, df1[strcase]], sort=False, axis=1)

        ## Step 2: load simulation inputs to S4kriging
        inputfilename = source_path+'/LHS.dat'
        if os.path.exists(inputfilename):
            text_input=open(inputfilename,"r")
            lines=text_input.readlines()
            
            for j in range(len(lines)):
                if j == 1:
                    list_tmp = lines[j].strip().split()
                    list_tmp = list_tmp[2:] # 0: case; 1: No.
                    df2 = pd.DataFrame(list_tmp,columns=['Name'])
                if j > 1:
                    list_tmp = lines[j].strip().split()
                    strcase = 'Case'+str(int(list_tmp[0])-1)+'Value'
                    list_tmp = list_tmp[1:] # 0: case No.
                    df2[strcase] = list_tmp
            S4kriging = df2
        
        ## Step 3: display simulation input and output
        if exclude_case == 1:
            print('Converged simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        elif exclude_case == 0:
            print('Converged and non-converged simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        else:
            print('Simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        print(*indcase4kriging)
        print('\nSelect from the following input variables for training:')
        for i in range(S4kriging.index.size):
            print(i+1, ':', S4kriging.loc[i, 'Name'], end = '\t\n')
        print('\nSelect from the following output variables for training:')
        for i in range(Y4kriging.index.size):
            print(i+1, ':', Y4kriging.loc[i, 'Name'], end = '\t\n')
            
        if display_detail == True:
            print('\n')
            print(S4kriging)
            print('\n')
            print(Y4kriging)
        
        ## Step 4: create allResults.dat
        indS = list(S4kriging.index)
        indY = list(Y4kriging.index)
        indS = [x+1 for x in indS]
        indY = [x+1 for x in indY]
        if len(indcase4kriging) == 0 or len(indS) == 0 or len(indY) == 0:
            print('Error: No data available for training')   
        
        with open(self.allresultsFile, 'w') as f:
            for i in indS:
                f.write(S4kriging.loc[i-1, 'Name'] + '\t')
            for i in indY:
                f.write(Y4kriging.loc[i-1, 'Name'] + '\t')
            f.write('\n')
            for i in indcase4kriging:
                strcase = 'Case'+str(i-1)+'Value'
                for j in indS:
                    f.write('{:11.4E}\t'.format(float(S4kriging.loc[j-1, strcase])))
                for j in indY:
                    f.write('{:11.4E}\t'.format(float(Y4kriging.loc[j-1, strcase])))
                f.write('\n')
                
        with open(self.allresults_infoFile, 'w') as f:
            f.write('input_col\toutput_col\n')
            f.write(str(len(indS))+'\t'+str(len(indY))+'\n')
            
    def file_read(self, FileName):
        '''
        This function loads the kriginginputFile, 
        infoFile and predictioninputFile
        '''
        namearray = []
        valuearray = []
        with open(FileName) as f:
            i = 0
            for line in f.readlines():
                if i == 0:
                    namearray = line.strip().split()
                else:
                    linestr = line.strip().split()
                    linenum = [float(lineele) for lineele in linestr]
                    valuearray.append(linenum)
                i += 1
               
        return namearray, np.array(valuearray)
    
    def variables(self):
        print('input variables:')
        for i in range(len(self.Sname)):
            print(i+1, ':', self.Sname[i], end = '\t\n')
            
        print('\noutput variables:')
        for i in range(len(self.Yname)):
            print(i+1, ':', self.Yname[i], end = '\t\n')
    
    def variable_options(self, display = False):
        names_input = [
            "Average_CellVoltage",
            "Average_CurrentDensity",
            "BackEnvironmentT",
            "BottomEnvironmentT",
            "CellFuelFlowRate",
            "CellOxidantFlowRate",
            "FrontEnvironmentT",
            "Fuel_Utilization",
            "FuelH2",
            "FuelH2O",
            "FuelCO",
            "FuelCO2",
            "FuelCH4",
            "FuelN2",
            "FuelTemperature",
            "FuelTOnTop",
            "FuelRecyclePercent",
            "FuelHTXEffectiveness",
            "FuelNGTemperature",
            "FuelNGHTXDeltaT",
            "Internal_Reforming",
            "nCells",
            "Oxidant_Recirculation",
            "OxidantRecyclePercent",
            "OxygenToCarbon_Ratio",
            "OxidantO2",
            "OxidantN2",
            "OxidantH2O",
            "OxidantCO2",
            "OxidantAr",
            "OxidantTemperature",
            "OxidantTOnTop",
            "PreReform",
            "SideEnvironmentT",
            "Simulation_Option",
            "Stack_Fuel_Utilization",
            "Stack_Oxidant_Utilization",
            "StackFuelFlowRate",
            "StackFuelFlowRateH2O",
            "StackFuelFlowRateCO",
            "StackFuelFlowRateCO2",
            "StackFuelFlowRateCH4",
            "StackFuelFlowRateH2",
            "StackFuelFlowRateN2",
            "StackOxidantFlowRate",
            "StackOxidantFlowRateO2",
            "StackOxidantFlowRateN2",
            "StackOxidantFlowRateH2O",
            "StackOxidantFlowRateCO2",
            "StackOxidantFlowRateAr",
            "StackVoltage",
            "SystemPressure",
            "TopEnvironmentT",
            "VGRRate",
            "VGRTemperature",
            "VGRH2OPassRate",
            "VGRH2PassRate",
            "VGRCO2CaptureRate",
            "VGRCOConvertRate"
        ]

        units_input = [
            "V",
            "A/m^2",
            "C",
            "C",
            "mol/s",
            "mol/s",
            "C",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "C",
            "C",
            "%",
            "-",
            "C",
            "C",
            "-",
            "-",
            "-",
            "%",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "C",
            "C",
            "-",
            "C",
            "-",
            "-",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "V",
            "atm",
            "C",
            "-",
            "C",
            "-",
            "-",
            "-",
            "-"
        ]
        
        names_output = [
            'SimulationStatus',
            'Stack_Voltage',
            'Avg_cell_voltage',
            'Stack_Current',
            'Avg_current_density',
            'Max_current_density',
            'Min_current_density',
            'Avg_Cell_Temperature',
            'Max_Cell_Temperature',
            'Min_Cell_Temperature',
            'Delta_Cell_Temperature',
            'Outlet_Fuel_Temperature',
            'Delta_Fuel_Temperature',
            'Outlet_Air_Temperature',
            'Delta_Air_Temperature',
            'Air_Heat_Exchanger_Effectiveness',
            'Fuel_Utilization',
            'Air_Utilization',
            'Outlet_Fuel_Flowrate',
            'Outlet_Fuel_H2',
            'Outlet_Fuel_H2O',
            'Outlet_Fuel_CO',
            'Outlet_Fuel_CO2',
            'Outlet_Fuel_CH4',
            'Outlet_Fuel_N2',
            'Outlet_Air_Flowrate',
            'Outlet_Air_O2',
            'Outlet_Air_N2',
            'Outlet_Air_H2O',
            'Outlet_Air_CO2',
            'Outlet_Air_Ar',
            'Total_Power',
            'Air_Enthalpy_Change',
            'Fuel_Enthalpy_Change',
            'External_Heat',
            'Electrical_Efficiency',
            'Stack_Efficiency',
            'Air_Inlet_Temperature',
            'FSI_Temperature',
            'FSI_Flowrate',
            'FSI_H2_MF',
            'FSI_H2O_MF',
            'FSI_CO_MF',
            'FSI_CO2_MF',
            'FSI_CH4_MF',
            'FSI_N2_MF',
            'Fuel_Temperature_after_Mix',
            'Fuel_Temperature_before_Gibbs_Reactor',
            'Fuel_Heat_Exchanger_Effectiveness'
        ]

        units_output = [
            '-',
            'V',
            'V',
            'A',
            'A/m2',
            'A/m2',
            'A/m2',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            '-',
            '-',
            '-',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            'W',
            'W',
            'W',
            'W',
            '-',
            '-',
            'K',
            'K',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            'K',
            'K',
            '-'
        ]

        if display == True:
            print('Options of input variable:')
            for i in range(len(names_input)):
                print(i+1, ':', names_input[i]+', ['+units_input[i]+']', end = '\t\n')
            print('Options of output variable:')
            for i in range(len(names_output)):
                print(i+1, ':', names_output[i]+', ['+units_output[i]+']', end = '\t\n')
        
        return names_input, units_input, names_output, units_output

    def buildROM(self, indS = None, indY = None, frac4ROM = 80, filter_enabled = False, z_thres = 5):
        '''
        The function build the ROM for certain input/output variables
        '''
        print('############################################################\
              \nBuild the ROM\
              \n############################################################')
        
        if not os.path.exists(self.allresultsFile) or not os.path.exists(self.allresults_infoFile):
            sys.exit('Code terminated: essential files missing')
        
        ## Step -1: train the classifier
        SYname, SYvalue = self.file_read(self.allresultsFile)
        infoname, infovalue = self.file_read(self.allresults_infoFile)
        [S_row, Y_row, S_col, Y_col] = [len(SYvalue), len(SYvalue), int(infovalue[0,0]), int(infovalue[0,1])]
        if indS == None: indS = list(range(1, S_col+1))
        if indY == None: indY = list(range(1, Y_col+1))
        indS_index = [i-1 for i in indS]
        indY_index = [i-1 for i in indY]
            
        if SYname[S_col] == 'SimulationStatus':
            cls_enabled = True
        else:
            cls_enabled = False
            
        if cls_enabled == True:
            if 1 in indY: indY.remove(1) # remove SimulationStatus
            if 0 in indY_index: indY_index.remove(0)
                
            for i in range(S_row):
                if SYvalue[i, S_col] == -1: SYvalue[i, S_col] = 0
            temp = SYvalue[:, 0:S_col+1]
            S_train_cls = temp[:, indS_index]
            Y_train_cls = temp[:, S_col]
            meanS_cls = S_train_cls.mean(axis=0)
            stdS_cls = S_train_cls.std(axis=0)
            S_train_nrm_cls = (S_train_cls-meanS_cls)/stdS_cls
            
            Y_train_cls = Y_train_cls. astype(int)
            
            maxiteration = 50000
            trainingoutput_file = self.outtrainingFile
            DNNcls_load_file = trainingoutput_file.replace(".dat", "")+'_cls'
            DNNcls_save_file = DNNcls_load_file
            # Initial training
            acc_val, cls_values = self.DNNCls(maxiteration, S_train_nrm_cls, Y_train_cls, len(indS), DNNcls_save_file)
            print("Classifier accuracy: ", acc_val)
            # Restore DNN, continue training
            #cls_values = self.DNNCls_restore(maxiteration, S_train_nrm_cls, Y_train_cls, len(indS), DNNcls_load_file, DNNcls_save_file)

        ## Step 0: filter the noise and remove all failed/unconverged cases
        if cls_enabled == True:
            SYvalue_cov = SYvalue[SYvalue[:, S_col] == 1, :]
        else:
            SYvalue_cov = SYvalue
            
        if filter_enabled == True:
            SY_row_rm = []
            for j in indY:
                tmp_data = SYvalue_cov[:, S_col+j-1]
                while(True):
                    z = np.abs(stats.zscore(tmp_data, axis = 0))
                    result = np.where(z > z_thres)
                    index = list(result[0])
                    # line removal list
                    if len(index) == 0: break
                    SY_row_rm += index
                    SY_row_rm = list(dict.fromkeys(SY_row_rm))
                    # replace outliers with mean
                    tmp_data[SY_row_rm] = np.mean(tmp_data)
            # remove rows and columns accroding to SY_row_rm and SY_col_rm
            SYvalue_new = np.delete(SYvalue_cov, SY_row_rm, axis = 0)
            print('Noise filter: trim ' + str(len(SY_row_rm)) + ' rows from a total of ' + str(len(SYvalue_cov)) + ' rows')
        else:
            SYvalue_new = SYvalue_cov
            
        ## Step 1: load all simulation results
        [S_row, Y_row, S_col, Y_col] = [len(SYvalue_new), len(SYvalue_new), int(infovalue[0,0]), int(infovalue[0,1])]
        S = copy.deepcopy(SYvalue_new[:, :S_col])
        Y = copy.deepcopy(SYvalue_new[:, S_col:])
        Sname = copy.deepcopy(SYname[:S_col])
        Yname = copy.deepcopy(SYname[S_col:])

        ## Step 2: compute istep, numcrossvali, rndnumberlist
        if frac4ROM >= 0:
            numtraining = int(S_row*frac4ROM/100.0)
            numcrossvali = S_row-numtraining
            if numtraining < (2**len(indS)): 
                print('warning: data set to build the ROM is not large enough')
            if numcrossvali > 0:
                istep = int((S_row)/numcrossvali)
                rndnumberlist =[]
                restlist = list(range(S_row))
                for i in range(1, numcrossvali+1):
                    rndnumberlist.append(i*istep-1)
                restlist = [i for i in restlist if i not in rndnumberlist]
            else:
                sys.exit('Code terminated: the fraction of training dataset cannot be 100%')
        else:
            numtraining = S_row-1000
            numcrossvali = S_row-numtraining
            rndnumberlist = list(range(numtraining, S_row))
            restlist = list(range(numtraining))
            
        ## Step 3: write to info.dat, intraining.dat, info.dat and inCrossVali.dat 
        with open(self.infoFile, 'w') as f:
            f.write('input_col\toutput_col\n')
            f.write(str(len(indS))+'\t'+str(len(indY))+'\n')
        f1 = open(self.intrainingFile, 'w')
        f3 = open(self.incrossvaliFile, 'w')
        for i in indS:
            f1.write(Sname[i-1] + '\t')
            f3.write(Sname[i-1] + '\t')
        for i in indY:
            f1.write(Yname[i-1] + '\t')
            f3.write(Yname[i-1] + '\t')
        f1.write('\n')
        f3.write('\n')
        for i in range(S_row):
            if i in rndnumberlist:
                for j in indS:
                    f3.write('{:11.4E}\t'.format(S[i, j-1]))
                for j in indY:
                    f3.write('{:11.4E}\t'.format(Y[i, j-1]))
                f3.write('\n')
            else:
                for j in indS:
                    f1.write('{:11.4E}\t'.format(S[i, j-1]))
                for j in indY:
                    f1.write('{:11.4E}\t'.format(Y[i, j-1]))
                f1.write('\n')
        f1.close()
        f3.close()
        
        ## Step 4: perform training and prediction
        temp = S[restlist, :]
        S_train = temp[:, indS_index]
        temp = S[rndnumberlist, :]
        S_vali = temp[:, indS_index]
        temp = Y[restlist, :]
        Y_train = temp[:, indY_index]
        temp = Y[rndnumberlist, :]
        Y_vali = temp[:, indY_index]
        
        meanS=S_train.mean(axis=0)
        stdS=S_train.std(axis=0)
        meanY=Y_train.mean(axis=0)
        stdY=Y_train.std(axis=0)
        S_train_nrm=(S_train-meanS)/stdS
        Y_train_nrm=(Y_train-meanY)/stdY
        S_vali_nrm=(S_vali-meanS)/stdS

        maxiteration = 50000 
        trainingoutput_file = self.outtrainingFile
        DNN_load_file = trainingoutput_file.replace(".dat", "")
        DNN_save_file = DNN_load_file
        DNNsize = [32, 200, 200, 256]
        
        # Initial training
        Y_vali_nrm_pre, model_values = self.DNNROM2(maxiteration, 
                                                   S_train_nrm, Y_train_nrm, S_vali_nrm, 
                                                   len(indS), len(indY), DNN_save_file, DNNsize)
        # Restore DNN, continue training
        # Y_vali_nrm_pre, model_values = self.DNNROM_restore2(maxiteration, S_train_nrm, Y_train_nrm, S_vali_nrm, len(indS), len(indY), DNN_load_file, DNN_save_file, DNNsize)
        # Load a DNN, and prediction
        #Y_vali_nrm_load_pre = self.DNNROM_prediction(S_vali_nrm, len(indS), len(indY), DNN_load_file)
        
        ## Step 5: save built ROM
        trainingoutput_file = self.outtrainingFile
        trainingoutput_file_cls = trainingoutput_file.replace(".dat", "")+'_cls.dat'
        
        if cls_enabled == True:
            w1,w2,b1,b2 = cls_values
            with open(trainingoutput_file_cls, 'w') as f:
                f.write('w1\n')
                values_tmp = np.copy(w1)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w2\n')
                values_tmp = np.copy(w2)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('b1\n')
                values_tmp = np.copy(b1)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b2\n')
                values_tmp = np.copy(b2)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('end\n')
        
        w1,w2,w3,w4,w5,b1,b2,b3,b4,b5 = model_values
        with open(self.outtrainingFile, 'w') as f:
            f.write('w1\n')
            values_tmp = np.copy(w1)
            [row, col] = values_tmp.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(values_tmp[i, j]) + ' ')
                f.write(str(values_tmp[i, col-1]) + '\n')
            f.write('\n')
            f.write('w2\n')
            values_tmp = np.copy(w2)
            [row, col] = values_tmp.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(values_tmp[i, j]) + ' ')
                f.write(str(values_tmp[i, col-1]) + '\n')
            f.write('\n')
            f.write('w3\n')
            values_tmp = np.copy(w3)
            [row, col] = values_tmp.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(values_tmp[i, j]) + ' ')
                f.write(str(values_tmp[i, col-1]) + '\n')
            f.write('\n')
            f.write('w4\n')
            values_tmp = np.copy(w4)
            [row, col] = values_tmp.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(values_tmp[i, j]) + ' ')
                f.write(str(values_tmp[i, col-1]) + '\n')
            f.write('\n')
            f.write('w5\n')
            values_tmp = np.copy(w5)
            [row, col] = values_tmp.shape
            for i in range(row):
                for j in range(col-1):
                    f.write(str(values_tmp[i, j]) + ' ')
                f.write(str(values_tmp[i, col-1]) + '\n')
            f.write('\n')
            f.write('b1\n')
            values_tmp = np.copy(b1)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('b2\n')
            values_tmp = np.copy(b2)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('b3\n')
            values_tmp = np.copy(b3)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('b4\n')
            values_tmp = np.copy(b4)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('b5\n')
            values_tmp = np.copy(b5)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('meanS\n')
            values_tmp = np.copy(meanS)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('meanY\n')
            values_tmp = np.copy(meanY)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('stdS\n')
            values_tmp = np.copy(stdS)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('stdY\n')
            values_tmp = np.copy(stdY)
            row = len(values_tmp)
            for i in range(row):
                f.write(str(values_tmp[i]) + '\n')
            f.write('\n')
            f.write('end\n')
            
        ## Step 6: write to outCrossVali.dat
        Y_vali_pre = Y_vali_nrm_pre*stdY+meanY
        f0 = open(self.outcrossvaliFile, 'w')
        for i in indY:
            name = Yname[i-1]
            f0.write(name + '\t')
        f0.write('\n')
        for i in range(len(rndnumberlist)):
            for j in range(len(indY)):
                f0.write('{:11.4E}\t'.format(Y_vali_pre[i,j]-Y_vali[i, j]))
            f0.write('\n')
        f0.close()
        
        ## Step 7: update global variables
        [self.S_row, self.Y_row, self.S_col, self.Y_col] = [len(restlist), len(restlist), len(indS), len(indY)]
        self.S_norm = S_train_nrm
        self.Y_norm = Y_train_nrm
        self.S = S_train
        self.Y = Y_train
        [self.stdS, self.stdY, self.meanS, self.meanY] = [stdS, stdY, meanS, meanY]
        Sname_new = [ Sname[i] for i in indS_index]
        Yname_new = [ Yname[i] for i in indY_index]
        self.Sname = Sname_new
        self.Yname = Yname_new
        
        ## Step 8: write classifier accuracy and ROM prediction accuracy
        int_95 = self.percent2intervl(95) # 95% confidence interval
        trainingoutput_file = self.outtrainingFile
        trainingoutput_accuracy = trainingoutput_file.replace(".dat", "")+'_acc.dat'
        with open(trainingoutput_accuracy, 'w') as f:
            if cls_enabled == True:
                f.write('Classifier Accuracy: \n')
                f.write(str(acc_val) + '\n')
            f.write('ROM Accuracy (95% confidence interval): \n')
            for i in range(len(Yname_new)):
                f.write(Yname_new[i])
                f.write('\t' + str(int_95[i]) + '\n')
        
        print('End of code\n')
    
    def Generate_inprediction(self, numsample = None, listmin = None, listmax = None):
        '''
        The function generates prediction input if it doesn't exist by Latin Hypercube Sampling
        '''
        print('############################################################\
                \nGenerate prediction input\
                \n############################################################')

        # find input variable list Sname
        SYname, SYvalue = self.file_read(self.intrainingFile)
        infoname, infovalue = self.file_read(self.infoFile)
        [S_col, Y_col] = [int(infovalue[0,0]), int(infovalue[0,1])]
        Sname = copy.deepcopy(SYname[:S_col])
        
        # check if exists
        filename  = self.inpredictionFile
        Create_handle = True
        if os.path.exists(filename):
            query = query_yes_no('Prediction input file already exists on the local machine, do you want to overwrite it?')
            Create_handle = query

        if Create_handle == True:
            numvar = len(Sname)
            listvar = Sname
            if len(listmin) != numvar or len(listmax) != numvar:
                sys.exit('Code terminated: the lengths of variables/minimums/maximums not match')

            # LHS sampling
            xlimits = np.transpose(np.vstack((listmin, listmax)))
            sampling = LHS(xlimits = xlimits)
            LHSvalue = sampling(numsample)

            # write prediction input
            with open(filename, 'w') as f:
                for name in Sname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(numsample):
                    for j in range(numvar):
                        f.write('{:11.4E}\t'.format(LHSvalue[i, j]))
                    f.write('\n')

            print("Created prediciton input file")
            print('End of code\n')

    def prediction(self):
        '''
        This function predicts the outputs and MSEs 
        based on the trained model 
        '''
        print('############################################################\
              \nPredict Based on the trained model\
              \n############################################################')
        
        # # Step 0: check if outprediction.dat existing
        # if os.path.exists(self.outpredictionFile):
        #     query = query_yes_no('prediction results already exist on the local machine, do you want to overwrite it?')
        #     if query == False: return
    
        # Step 1: Load the training data S, Y and prediction data X
        print('Step 1: Load the training data S, Y and prediction input data X')
        SYname, SYvalue = self.file_read(self.intrainingFile)
        Xname, Xvalue = self.file_read(self.inpredictionFile)
        infoname, infovalue = self.file_read(self.infoFile)
        [S_row, Y_row, S_col, Y_col] = [len(SYvalue), len(SYvalue), int(infovalue[0,0]), int(infovalue[0,1])]
        
        # Step 1.5: Load the trained classifier
        trainingoutput_file = self.outtrainingFile
        if not os.path.exists(trainingoutput_file):
            sys.exit('Code terminated: trained model missing')
            
        trainingoutput_file_cls = trainingoutput_file.replace(".dat", "")+'_cls.dat'
        if os.path.exists(trainingoutput_file_cls):
            cls_enabled = True
        else:
            cls_enabled = False
            print('trained model has no classifier, continue')
        
        if cls_enabled == True:
            with open(trainingoutput_file_cls) as f:
                lines = f.readlines()
                i = 0
                for line in lines:
                    linestr = line.strip().split(' ')
                    if linestr[0] == 'w1': 
                        w1_s_cls = i+1
                    if linestr[0] == 'w2': 
                        w2_s_cls = i+1
                        w1_e_cls = i-2
                    if linestr[0] == 'b1': 
                        b1_s_cls = i+1
                        w2_e_cls = i-2
                    if linestr[0] == 'b2': 
                        b2_s_cls = i+1
                        b1_e_cls = i-2
                    if linestr[0] == 'end': 
                        b2_e_cls = i-2
                    i += 1

                i = 0
                for line in lines:
                    linestr = line.strip().split(' ')
                    if i == w1_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w1_cls = np.array(linenum)
                        w1_row_cls = w1_e_cls-w1_s_cls+1
                        w1_col_cls = len(w1_cls)
                    if i > w1_s_cls and i <= w1_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w1_cls = np.append(w1_cls, linenum)
                    if i == w2_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w2_cls = np.array(linenum)
                        w2_row_cls = w2_e_cls-w2_s_cls+1
                        w2_col_cls = len(w2_cls)
                    if i > w2_s_cls and i <= w2_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w2_cls = np.append(w2_cls, linenum)
                    if i == b1_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b1_cls = np.array(linenum)
                    if i > b1_s_cls and i <= b1_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b1_cls = np.append(b1_cls, linenum)
                    if i == b2_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b2_cls = np.array(linenum)
                    if i > b2_s_cls and i <= b2_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b2_cls = np.append(b2_cls, linenum)
                    i += 1

                w1_cls = np.reshape(w1_cls, (w1_row_cls, w1_col_cls))
                w2_cls = np.reshape(w2_cls, (w2_row_cls, w2_col_cls))
            
        # Step 2: Load the trained model (outtrainingFile)
        print('Step 2: Load the trained model (outtrainingFile)')
        with open(self.outtrainingFile) as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                linestr = line.strip().split(' ')
                if linestr[0] == 'w1': 
                    w1_s = i+1
                if linestr[0] == 'w2': 
                    w2_s = i+1
                    w1_e = i-2
                if linestr[0] == 'w3': 
                    w3_s = i+1
                    w2_e = i-2
                if linestr[0] == 'w4': 
                    w4_s = i+1
                    w3_e = i-2
                if linestr[0] == 'w5': 
                    w5_s = i+1
                    w4_e = i-2
                if linestr[0] == 'b1': 
                    b1_s = i+1
                    w5_e = i-2
                if linestr[0] == 'b2': 
                    b2_s = i+1
                    b1_e = i-2
                if linestr[0] == 'b3': 
                    b3_s = i+1
                    b2_e = i-2
                if linestr[0] == 'b4': 
                    b4_s = i+1
                    b3_e = i-2
                if linestr[0] == 'b5': 
                    b5_s = i+1
                    b4_e = i-2
                if linestr[0] == 'meanS': 
                    meanS_s = i+1
                    b5_e = i-2
                if linestr[0] == 'meanY': 
                    meanY_s = i+1
                    meanS_e = i-2
                if linestr[0] == 'stdS': 
                    stdS_s = i+1
                    meanY_e = i-2
                if linestr[0] == 'stdY': 
                    stdY_s = i+1
                    stdS_e = i-2
                if linestr[0] == 'end': 
                    stdY_e = i-2
                i += 1
            
            i = 0
            for line in lines:
                linestr = line.strip().split(' ')
                if i == w1_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w1 = np.array(linenum)
                    w1_row = w1_e-w1_s+1
                    w1_col = len(w1)
                if i > w1_s and i <= w1_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w1 = np.append(w1, linenum)
                if i == w2_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w2 = np.array(linenum)
                    w2_row = w2_e-w2_s+1
                    w2_col = len(w2)
                if i > w2_s and i <= w2_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w2 = np.append(w2, linenum)
                if i == w3_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w3 = np.array(linenum)
                    w3_row = w3_e-w3_s+1
                    w3_col = len(w3)
                if i > w3_s and i <= w3_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w3 = np.append(w3, linenum)
                if i == w4_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w4 = np.array(linenum)
                    w4_row = w4_e-w4_s+1
                    w4_col = len(w4)
                if i > w4_s and i <= w4_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w4 = np.append(w4, linenum)
                if i == w5_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w5 = np.array(linenum)
                    w5_row = w5_e-w5_s+1
                    w5_col = len(w5)
                if i > w5_s and i <= w5_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w5 = np.append(w5, linenum)
                if i == b1_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b1 = np.array(linenum)
                if i > b1_s and i <= b1_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b1 = np.append(b1, linenum)
                if i == b2_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b2 = np.array(linenum)
                if i > b2_s and i <= b2_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b2 = np.append(b2, linenum)
                if i == b3_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b3 = np.array(linenum)
                if i > b3_s and i <= b3_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b3 = np.append(b3, linenum)
                if i == b4_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b4 = np.array(linenum)
                if i > b4_s and i <= b4_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b4 = np.append(b4, linenum)
                if i == b5_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b5 = np.array(linenum)
                if i > b5_s and i <= b5_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b5 = np.append(b5, linenum)
                if i == meanS_s:
                    linenum = [float(lineele) for lineele in linestr]
                    meanS = np.array(linenum)
                if i > meanS_s and i <= meanS_e:
                    linenum = [float(lineele) for lineele in linestr]
                    meanS = np.append(meanS, linenum)
                if i == meanY_s:
                    linenum = [float(lineele) for lineele in linestr]
                    meanY = np.array(linenum)
                if i > meanY_s and i <= meanY_e:
                    linenum = [float(lineele) for lineele in linestr]
                    meanY = np.append(meanY, linenum)
                if i == stdS_s:
                    linenum = [float(lineele) for lineele in linestr]
                    stdS = np.array(linenum)
                if i > stdS_s and i <= stdS_e:
                    linenum = [float(lineele) for lineele in linestr]
                    stdS = np.append(stdS, linenum)
                if i == stdY_s:
                    linenum = [float(lineele) for lineele in linestr]
                    stdY = np.array(linenum)
                if i > stdY_s and i <= stdY_e:
                    linenum = [float(lineele) for lineele in linestr]
                    stdY = np.append(stdY, linenum)
                i += 1
                
            w1 = np.reshape(w1, (w1_row, w1_col))
            w2 = np.reshape(w2, (w2_row, w2_col))
            w3 = np.reshape(w3, (w3_row, w3_col))
            w4 = np.reshape(w4, (w4_row, w4_col))
            w5 = np.reshape(w5, (w5_row, w5_col))
            
        # Step 3: Normalize S, Y, X
        S = copy.deepcopy(SYvalue[:, :S_col])
        Y = copy.deepcopy(SYvalue[:, S_col:])
        X = copy.deepcopy(Xvalue)
        Sname = copy.deepcopy(SYname[:S_col])
        Yname = copy.deepcopy(SYname[S_col:])
        [X_row, X_col] = X.shape
        if X_col != S_col:
            sys.exit('Code terminated: # of prediction input variables \
            does not match # of given input variables')
        S_nrm = (S - np.tile(meanS, [S_row, 1]))/np.tile(stdS, [S_row, 1])
        Y_nrm = (Y - np.tile(meanY, [Y_row, 1]))/np.tile(stdY, [Y_row, 1])
        X_nrm = (X - np.tile(meanS, [X_row, 1]))/np.tile(stdS, [X_row, 1])
        
        # Step 3.5: perform prediction of SimulationStatus
        if cls_enabled == True:
            for j in range(X_row):
                inputX_cls = X_nrm[j,:]
                m1_cls = np.matmul(inputX_cls,w1_cls)
                m1b_cls = np.add(m1_cls,b1_cls)
                m1ba_cls = np.zeros(len(m1b_cls))
                for i in range(len(m1b_cls)):
                    m1ba_cls[i] = 1.0/(1+math.exp(-m1b_cls[i]))
                m2_cls = np.matmul(m1ba_cls,w2_cls)
                m2b_cls = np.add(m2_cls,b2_cls)
                m2ba_cls = np.zeros(len(m2b_cls))
                for i in range(len(m2b_cls)):
                    m2ba_cls[i] = m2b_cls[i]

                outputX_cls = m2ba_cls
                if j == 0:
                    Xy_cls = outputX_cls
                else:
                    Xy_cls = np.vstack((Xy_cls, outputX_cls))
            
            #convert to 0 and 1
            Xy_cls = np.argmax(Xy_cls, 1)
            # print(len(Xy_cls))
            # print(sum(Xy_cls))

            # DNNcls_load_file = trainingoutput_file.replace(".dat", "")+'_cls'
            # SimuStatus = self.DNNCls_prediction(X_nrm, S_col, DNNcls_load_file)
            # print('Compare two methods of predictions:')
            # print((Xy_cls==SimuStatus).all())

        # Step 4: perform prediction
        for j in range(X_row):
            inputX = X_nrm[j,:]
            m1 = np.matmul(inputX,w1)
            m1b = np.add(m1,b1)
            m1ba = np.zeros(len(m1b))
            for i in range(len(m1b)):
                m1ba[i] = 1.0/(1+math.exp(-m1b[i]))
            m2 = np.matmul(m1ba,w2)
            m2b = np.add(m2,b2)
            m2ba = np.zeros(len(m2b))
            for i in range(len(m2b)):
                m2ba[i] = 1.0/(1+math.exp(-m2b[i]))
            m3 = np.matmul(m2ba,w3)
            m3b = np.add(m3,b3)
            m3ba = np.zeros(len(m3b))
            for i in range(len(m3b)):
                m3ba[i] = 1.0/(1+math.exp(-m3b[i]))
            m4 = np.matmul(m3ba,w4)
            m4b = np.add(m4,b4)
            m4ba = np.zeros(len(m4b))
            for i in range(len(m4b)):
                m4ba[i] = 1.0/(1+math.exp(-m4b[i]))
            m5 = np.matmul(m4ba,w5)
            m5b = np.add(m5,b5)
            m5ba = np.zeros(len(m5b))
            for i in range(len(m5b)):
                    m5ba[i] = m5b[i]
                    
            outputX_nrm = m5ba
            outputX = m5ba*stdY+meanY
            if j == 0:
                Xy_nrm = outputX_nrm
                Xy = outputX
            else:
                Xy_nrm = np.vstack((Xy_nrm, outputX_nrm))
                Xy = np.vstack((Xy, outputX))

        print('\tFinish Prediction - Xy')

        # Copy to Global
        [self.S_row, self.Y_row, self.S_col, self.Y_col] = [S_row, Y_row, S_col, Y_col]
        self.S_norm = S_nrm
        self.Y_norm = Y_nrm
        self.S = S
        self.Y = Y
        [self.stdS, self.stdY] = [stdS, stdY]
        self.X = X
        self.Xy = Xy
        self.X_norm = X_nrm
        self.Xy_norm = Xy_nrm
        self.Sname = Sname
        self.Yname = Yname
        
        # Step 5: Write the predictions
        print('Step 4: Write the predictions')
        with open(self.outpredictionFile, 'w') as f:
            for name in Xname:
                f.write(name + '\t')
            if cls_enabled == True:
                f.write('SimulationStatus\t')
            for i in range(Y_col):
                f.write(Yname[i] + '\t')
            f.write('\n')
            
            for i in range(X_row):
                # write input variables
                for j in range(S_col):
                    f.write('{:11.4E}\t'.format(X[i, j]))
                # write simulation status
                if cls_enabled == True:
                    f.write('{:11.4E}\t'.format(Xy_cls[i]))
                # write output variables
                for j in range(Y_col):
                    f.write('{:11.4E}\t'.format(Xy[i, j]))
                f.write('\n')
        print('End of code\n')
    
    def percent2intervl(self, percentage, var = None):
        print('############################################################\
              \nPercentage to Confidence Interval\
              \n############################################################')
        
        # load cross validation results
        Yname, ERR = self.file_read(self.outcrossvaliFile)
        
        # find the units
        names_input, units_input, names_output, units_output = self.variable_options()
        Yunit = []
        for i in range(len(Yname)):
            tempindex = names_output.index(Yname[i])
            tempunit = units_output[tempindex]
            Yunit.append(tempunit)
            
        # compute confidence interval
        interval_all = np.zeros((len(Yname),),dtype=np.float64)
        for i in range(len(Yname)):
            err = np.sort(ERR[:, i])
            N = len(err)
            n = (N-1)*percentage/100.0 + 1
            if n == 1: 
                interval = err[0]
            elif n == N:
                interval = err[N-1]
            else:
                k = int(n)
                d = n-k
                interval = err[k-1]+d*(err[k]-err[k-1])
            
            interval_all[i] = interval
            
            if var == None:    
                print('For "' + str(Yname[i]) + '":'
                     + '[' + Yunit[i] + ']'
                     +' \n\t' 
                     + str(percentage) + '% confidence interval is '
                     + '\u00B1' + '{:11.4E}\t'.format(interval))
            elif Yname[i] == var:
                print('For "' + str(Yname[i]) + '":'
                     + '[' + Yunit[i] + ']'
                     +' \n\t' 
                     + str(percentage) + '% confidence interval is '
                     + '\u00B1' + '{:11.4E}\t'.format(interval))
            elif var not in Yname:
                print('The given variable cannot be found')
            
        print('End of code\n')
        return(interval_all)
        
    def intervl2percent(self, interval, var = None):
        print('############################################################\
              \nConfidence Interval to Percentage\
              \n############################################################')
        
        # load cross validation results
        Yname, ERR = self.file_read(self.outcrossvaliFile)
        
        # find the units
        names_input, units_input, names_output, units_output = self.variable_options()
        Yunit = []
        for i in range(len(Yname)):
            tempindex = names_output.index(Yname[i])
            tempunit = units_output[tempindex]
            Yunit.append(tempunit)
        
        # compute confidence percentage
        percentage_all = np.zeros((len(Yname),),dtype=np.float64)
        for i in range(len(Yname)):
            if var == Yname[i]:
                err = np.sort(ERR[:, i])
                N = len(err)
                if interval <= err[0]:
                    percentage = 0
                elif interval >= err[N-1]:
                    percentage = 1
                else:
                    result = np.where(err>interval)
                    index = result[0]
                    k = index[0]
                    percentage = ((interval-err[k-1])/(err[k]-err[k-1])+k-1)/float(N-1)
                
                percentage_all[i] = percentage
                
                print('For "' + str(Yname[i]) + '": ' 
                     + '[' + Yunit[i] + ']'
                     + '\n\t\u00B1' + str(interval)
                     + ' interval has a confidence of ' +  str(round(percentage*100, 2)) + '%')
            elif var not in Yname:
                print('The given variable cannot be found')

        print('End of code\n')
        return(percentage_all)
        
    def plot_contour_2D(self, xvariable, yvariable, zvariable, 
                        pltoption = 0, saveoption = False):
        '''
        The function plots 2D contour of designs and responses
        pltoption = 0: plot both training and prediction sets; 1: plot only training sets, 2: plot only prediction sets
        '''
        # check if the given variables are in the list
        if (xvariable not in self.Sname) or (yvariable not in self.Sname) or (zvariable not in self.Yname):
            sys.exit('Code terminated: variable index out of bound')
        v1 = self.Sname.index(xvariable)+1
        v2 = self.Sname.index(yvariable)+1
        v3 = self.Yname.index(zvariable)+1
        option = int(pltoption)
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_input.index(yvariable)
        yunit = units_input[tempindex]
        tempindex = names_output.index(zvariable)
        zunit = units_output[tempindex]
            
        # Generate inPrediction4contour.dat
        if option == 0 or option == 2:
            Xname, Xvalue = self.file_read(self.inpredictionFile)
            Xvalue_mean = np.mean(Xvalue, axis = 0)
            [X_row, X_col] = Xvalue.shape
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_contour_DNN.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_contour_DNN.dat'
            with open(self.inpredictionFile, 'w') as f:
                for name in Xname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(X_row):
                    for j in range(X_col):
                        if (j+1) == v1 or (j+1) == v2:
                            f.write('{:11.4E}\t'.format(Xvalue[i, j]))
                        else:
                            f.write('{:11.4E}\t'.format(Xvalue_mean[j]))
                    f.write('\n')
            self.prediction()
            os.remove(self.inpredictionFile)
            os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig
            

        if option == 0: # Default: plot both training and prediction sets
            x1 = self.S[:, v1-1]
            y1 = self.S[:, v2-1]
            z1 = self.Y[:, v3-1]
            x2 = self.X[:, v1-1]
            y2 = self.X[:, v2-1]
            z2 = self.Xy[:, v3-1]
            
            plt.figure(figsize=(17.5,6))
            plt.subplot(1, 2, 1)
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            C = plt.tricontour(x1, y1, z1, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x1, y1, z1, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            #plt.colorbar().set_label(label='a label',size=15,weight='bold')
            plt.xlim((min(min(x1), min(x2)), max(max(x1), max(x2))))
            plt.ylim((min(min(y1), min(y2)), max(max(y1), max(y2))))
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            
            plt.subplot(1, 2, 2)
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            C = plt.tricontour(x2, y2, z2, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x2, y2, z2, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.xlim((min(min(x1), min(x2)), max(max(x1), max(x2))))
            plt.ylim((min(min(y1), min(y2)), max(max(y1), max(y2))))
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 1: # plot training sets
            x = self.S[:, v1-1]
            y = self.S[:, v2-1]
            z = self.Y[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            plt.figure(figsize=(8,6))
            C = plt.tricontour(x, y, z, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x, y, z, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 2: # plot prediciton sets
            x = self.X[:, v1-1]
            y = self.X[:, v2-1]
            z = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            plt.figure(figsize=(8,6))
            C = plt.tricontour(x, y, z, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x, y, z, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        # save option
        if saveoption == True:
            figurename = '2D_contour.png'
            plt.savefig(figurename)
            
    def plot_contour_3D(self, xvariable, yvariable, zvariable, 
                        pltoption = 0, saveoption = False):
        '''
        The function plots 2D contour of designs and responses
        pltoption = 0: plot both training and prediction sets; 1: plot only training sets, 2: plot only prediction sets
        '''
        # check if the given variables are in the list
        if (xvariable not in self.Sname) or (yvariable not in self.Sname) or (zvariable not in self.Yname):
            sys.exit('Code terminated: variable index out of bound')
        v1 = self.Sname.index(xvariable)+1
        v2 = self.Sname.index(yvariable)+1
        v3 = self.Yname.index(zvariable)+1
        option = int(pltoption)
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_input.index(yvariable)
        yunit = units_input[tempindex]
        tempindex = names_output.index(zvariable)
        zunit = units_output[tempindex]
        
        # Generate inPrediction4contour.dat
        if option == 0 or option == 2:
            Xname, Xvalue = self.file_read(self.inpredictionFile)
            Xvalue_mean = np.mean(Xvalue, axis = 0)
            [X_row, X_col] = Xvalue.shape
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_contour_kriging.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_contour_kriging.dat'
            with open(self.inpredictionFile, 'w') as f:
                for name in Xname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(X_row):
                    for j in range(X_col):
                        if (j+1) == v1 or (j+1) == v2:
                            f.write('{:11.4E}\t'.format(Xvalue[i, j]))
                        else:
                            f.write('{:11.4E}\t'.format(Xvalue_mean[j]))
                    f.write('\n')
            self.prediction()
            os.remove(self.inpredictionFile)
            os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig
            
        if option == 0: # Default: plot both training and prediction sets
            x1 = self.S[:, v1-1]
            y1 = self.S[:, v2-1]
            z1 = self.Y[:, v3-1]
            x2 = self.X[:, v1-1]
            y2 = self.X[:, v2-1]
            z2 = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(18.5,6))
            ax = fig.add_subplot(1, 2, 1, projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x1, y1, z1, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            
            ax = fig.add_subplot(1, 2, 2, projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x2, y2, z2, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 1: # plot training sets
            x = self.S[:, v1-1]
            y = self.S[:, v2-1]
            z = self.Y[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x, y, z, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 2: # plot prediciton sets
            x = self.X[:, v1-1]
            y = self.X[:, v2-1]
            z = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x, y, z, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
        
        # save option
        if saveoption == True:
            figurename = '3D_contour.png'
            plt.savefig(figurename)
            
    def plot_box(self, xvariable, yvariable, saveoption = False):
        '''
        The function is for box plot, it can help to perform sensitivity studies
        '''
        # convert to pandam dataframe
        S = pd.DataFrame(data = self.S, columns = self.Sname, dtype = 'float')
        Y = pd.DataFrame(data = self.Y, columns = self.Yname, dtype = 'float')
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_output.index(yvariable)
        yunit = units_output[tempindex]
        
        # generate box plot data
        x = S[[xvariable]]
        y = Y[[yvariable]]
        min_x = min(x.values)
        max_x = max(x.values)
        x = round((x-min_x)/((max_x-min_x)/9), 0)*((max_x-min_x)/9)+min_x
        x = round(x, 2)
        #xy = pd.concat([x, y], axis = 1, sort = False)
        #print(x.sort_values(by = ['Average_CurrentDensity']))
        #print(xy)
        
        # box plot
        plt.figure(figsize=(18.5,6))
        sns.set_context("paper", font_scale=3)
        sns.set_style('ticks')
        bplot = sns.boxplot(y=y[yvariable], x=x[xvariable], 
                    color = 'yellow', width = 0.5)
        bplot = sns.swarmplot(y=y[yvariable], x=x[xvariable], 
                    color = 'black', alpha = 0.5)
        sns.axes_style()
        bplot.axes.set_title('Design-response sites', fontsize = 25)
        bplot.set_xlabel(xvariable+', ['+xunit+']', fontsize = 25)
        bplot.set_ylabel(yvariable+', ['+yunit+']', fontsize = 25)
        bplot.tick_params(labelsize = 25)
        plt.show()
        
        # save option
        if saveoption == True:
            figurename = 'boxplot.png'
            plt.savefig(figurename)

class PhyDNN():
    def __init__(self, work_path, 
                 allresultsFile = 'allResults.dat',
                 allresults_infoFile = 'allResults_info.dat',
                 intrainingFile = 'inTraining_Phy.dat', 
                 infoFile = 'info_Phy.dat', 
                 outtrainingFile = 'outTraining_Phy.dat', 
                 inpredictionFile = 'inPrediction_Phy.dat', 
                 outpredictionFile = 'outPrediction_Phy.dat', 
                 incrossvaliFile = 'inCrossVali_Phy.dat',
                 outcrossvaliFile = 'outCrossVali_Phy.dat'):
        self.work_path = work_path
        self.allresultsFile = work_path + '/' + allresultsFile
        self.allresults_infoFile = work_path + '/' + allresults_infoFile
        self.intrainingFile = work_path + '/' + intrainingFile
        self.infoFile = work_path + '/' + infoFile
        self.outtrainingFile = work_path + '/' + outtrainingFile
        self.inpredictionFile = work_path + '/' + inpredictionFile
        self.outpredictionFile = work_path + '/' + outpredictionFile
        self.incrossvaliFile = work_path + '/' + incrossvaliFile
        self.outcrossvaliFile = work_path + '/' + outcrossvaliFile
        self.Sname = None
        self.Yname = None
        self.S_norm = None
        self.Y_norm = None
        self.X_norm = None
        self.Xy_norm = None
        self.S = None
        self.Y = None
        self.X = None
        self.Xy = None
        self.MSE = None
        self.S_row = 0
        self.Y_row = 0
        self.S_col = 0
        self.Y_col = 0
        self.stdS = None
        self.stdY = None
        self.meanS = None
        self.meanY = None

    def NGFC_ccs(self, J,FU,AU,OCR,IR,Arec,PreReform,cellsize):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        cell_exhaust = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (NG)
        NG_fin[Index_H2O] = 0
        NG_fin[Index_Ar] = 0
        NG_fin[Index_CO2] = 74.0729157
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 118.516665
        NG_fin[Index_CH4] = 6896.18846
        NG_fin[Index_CO] = 0
        NG_fin[Index_H2] = 0
        NG_fin[Index_C2H6] = 237.03333
        NG_fin[Index_C3H8] = 51.851041
        NG_fin[Index_C4H10] = 29.6291663
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                    stack_mix[i] = stack_fin[i] + stack_recirc[i] #;


                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
                cell_exhaust[i] = cell_exit[i] - stack_recirc[i]
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert
            cell_exhaust[i] /= Const_Convert
            cell_aexhaust[i] /= Const_Convert
            cell_exit[i] /= Const_Convert
            cell_aexit[i] /= Const_Convert
            pref_CH4[i] /= Const_Convert


        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        Frec = CalcR #; //they do equal
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (mol/s)",pref_CH4)
        # print("Air cell outlet (U) (mol/s)",cell_aexit)
        # print("Fuel cell outlet (Q) (mol/s)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        #return(SOFC_Ain,stack_ain,stack_fin*Const_Convert,stack_recirc,stack_mix,pref_CH4,cell_exit,Frec,succs)
        #return(stack_fin,stack_ain/Const_Convert,Frec,succs)
        #return(stack_fin,SOFC_Ain,Fresh_Ain,Frec,succs)
        return(cell_exit, cell_aexit, pref_CH4, succs)

    def NGFC_nocc(self, J,FU,AU,OCR,IR,Arec,PreReform,cellsize):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)
        cell_exhaust = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (NG)
        NG_fin[Index_H2O] = 0
        NG_fin[Index_Ar] = 0
        NG_fin[Index_CO2] = 74.0729157
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 118.516665
        NG_fin[Index_CH4] = 6896.18846
        NG_fin[Index_CO] = 0
        NG_fin[Index_H2] = 0
        NG_fin[Index_C2H6] = 237.03333
        NG_fin[Index_C3H8] = 51.851041
        NG_fin[Index_C4H10] = 29.6291663
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 1
        splt_ain[Index_Ar] = 1
        splt_ain[Index_CO2] = 1
        splt_ain[Index_O2] = 1
        splt_ain[Index_N2] = 1
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3
        # print(FU_REF1,FU_REF2,FU_REF3,FU_REF,FU)
        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                    stack_mix[i] = stack_fin[i] + stack_recirc[i] #;


                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]
                cell_exhaust[i] = cell_exit[i] - stack_recirc[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert
            cell_exhaust[i] /= Const_Convert
            cell_aexhaust[i] /= Const_Convert
            cell_exit[i] /= Const_Convert
            cell_aexit[i] /= Const_Convert
            pref_CH4[i] /= Const_Convert

        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        Frec = CalcR #; //they do equal
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (mol/s)",pref_CH4)
        # print("Air cell outlet (U) (mol/s)",cell_aexit)
        # print("Fuel cell outlet (Q) (mol/s)",cell_exit)

        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        #return(stack_ain/Const_Convert,stack_fin,Frec,succs)
        #return(stack_fin, SOFC_Ain, Fresh_Ain, Frec, succs)
        return(cell_exit, cell_aexit, pref_CH4, succs)

    def IGFC_ccs(self, J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)
        cell_exhaust = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (igfc) default conventional
        NG_fin[Index_H2O] = 0.0013
        NG_fin[Index_Ar] = 0.0008
        NG_fin[Index_CO2] = 0.2043
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 0.006
        NG_fin[Index_CH4] = 0.0583
        NG_fin[Index_CO] = 0.3774
        NG_fin[Index_H2] = 0.3519
        NG_fin[Index_C2H6] = 0.0
        NG_fin[Index_C3H8] = 0.0
        NG_fin[Index_C4H10] = 0.0
        if igfc=='conventional':
            NG_fin[Index_H2O] = 0.0013
            NG_fin[Index_Ar] = 0.0008
            NG_fin[Index_CO2] = 0.2043
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.006
            NG_fin[Index_CH4] = 0.0583
            NG_fin[Index_CO] = 0.3774
            NG_fin[Index_H2] = 0.3519
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='enhanced':
            NG_fin[Index_H2O] = 0.0006
            NG_fin[Index_Ar] = 0.0009
            NG_fin[Index_CO2] = 0.2423
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0064
            NG_fin[Index_CH4] = 0.1022
            NG_fin[Index_CO] = 0.3415
            NG_fin[Index_H2] = 0.3062
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='catalytic':
            NG_fin[Index_H2O] = 0.0004
            NG_fin[Index_Ar] = 0.0003
            NG_fin[Index_CO2] = 0.3465
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0069
            NG_fin[Index_CH4] = 0.3159
            NG_fin[Index_CO] = 0.0914
            NG_fin[Index_H2] = 0.2386
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        IR = 1.0
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                    stack_mix[i] = stack_fin[i] + stack_recirc[i] #;


                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]
                cell_exhaust[i] = cell_exit[i] - stack_recirc[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert
            cell_exhaust[i] /= Const_Convert
            cell_aexhaust[i] /= Const_Convert
            cell_exit[i] /= Const_Convert
            cell_aexit[i] /= Const_Convert
            pref_CH4[i] /= Const_Convert

        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        Frec = CalcR #; //they do equal
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (mol/s)",pref_CH4)
        # print("Air cell outlet (U) (mol/s)",cell_aexit)
        # print("Fuel cell outlet (Q) (mol/s)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        #return(stack_fin,stack_ain/Const_Convert,Frec,succs)
        #return(stack_fin,SOFC_Ain,Fresh_Ain,Frec,succs)
        return(cell_exit, cell_aexit, pref_CH4, succs)
 
    def NGFC_ccs_vgr(self, J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)
        cell_exhaust = np.arange(Nspecies,dtype=np.float64)

        recirc_VGR0 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR1 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR2 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR3 = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (NG)
        NG_fin[Index_H2O] = 0
        NG_fin[Index_Ar] = 0
        NG_fin[Index_CO2] = 74.0729157
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 118.516665
        NG_fin[Index_CH4] = 6896.18846
        NG_fin[Index_CO] = 0
        NG_fin[Index_H2] = 0
        NG_fin[Index_C2H6] = 237.03333
        NG_fin[Index_C3H8] = 51.851041
        NG_fin[Index_C4H10] = 29.6291663
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                   # stack_mix[i] = stack_fin[i] + stack_recirc[i] #;
                    recirc_VGR3[i]=stack_fin[i]*0.05

                for i in range(Nspecies):
                    stack_mix[i]=stack_fin[i]+stack_recirc[i]+recirc_VGR3[i]
                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]+recirc_VGR3[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            #cell_ref[Index_H2O] = pref_CH4[Index_H2O]-pref_CH4[Index_CH4]-2*pref_CH4[Index_C2H6]-3*pref_CH4[Index_C3H8]-4*pref_CH4[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7a) Calculate the new VGR recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------  
            for i in range(Nspecies):
                recirc_VGR0[i]=cell_exit[i]-stack_recirc[i]
                recirc_VGR1[i]=recirc_VGR0[i]
            WGSmol=WGS*recirc_VGR1[Index_CO]
            recirc_VGR1[Index_H2O] = recirc_VGR1[Index_H2O] - WGSmol
            recirc_VGR1[Index_CO2] = recirc_VGR1[Index_CO2] + WGSmol
            recirc_VGR1[Index_CO] = recirc_VGR1[Index_CO] - WGSmol
            recirc_VGR1[Index_H2] = recirc_VGR1[Index_H2] + WGSmol
            for i in range(Nspecies):
                recirc_VGR2[i]=recirc_VGR1[i]
            VGRH2O=recirc_VGR1[Index_H2O]*H2OCap
            VGRCO2=recirc_VGR1[Index_CO2]*CO2Cap
            VGRH2=recirc_VGR1[Index_H2]*H2Cap
            recirc_VGR2[Index_H2O]=recirc_VGR2[Index_H2O]-VGRH2O
            recirc_VGR2[Index_CO2]=recirc_VGR2[Index_CO2]-VGRCO2
            recirc_VGR2[Index_H2]=recirc_VGR2[Index_H2]-VGRH2
            for i in range(Nspecies):
                recirc_VGR3[i]=recirc_VGR2[i]*VGR
                cell_exhaust[i] = recirc_VGR2[i] - recirc_VGR3[i]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert
            cell_exhaust[i] /= Const_Convert
            cell_aexhaust[i] /= Const_Convert
            cell_exit[i] /= Const_Convert
            cell_aexit[i] /= Const_Convert
            pref_CH4[i] /= Const_Convert

        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        #Frec = CalcR #; //they do equal //not working for VGR
        CalcR=Frec
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (mol/s)",pref_CH4)
        # print("Air cell outlet (U) (mol/s)",cell_aexit)
        # print("Fuel cell outlet (Q) (mol/s)",cell_exit)
        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        #return(stack_fin,stack_ain/Const_Convert,Frec,succs)
        #return(stack_fin,SOFC_Ain,Fresh_Ain,Frec,succs)
        return(cell_exit, cell_aexit, pref_CH4, succs)

    def IGFC_ccs_vgr(self, J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc):
        Nspecies = 11

        MW_fuel = np.arange(Nspecies,dtype=np.float64) ##molucular weight
        NG_fin = np.arange(Nspecies,dtype=np.float64)  ##hardcode, fuel species, in 
        NG_mfin = np.arange(Nspecies,dtype=np.float64) ##fuel species from NG_fin[] turned to fractions
        std_ain = np.arange(Nspecies,dtype=np.float64) ##standard air in
        splt_ain = np.arange(Nspecies,dtype=np.float64) ##air separation split? why not sum==1?
        ref_ain = np.arange(Nspecies,dtype=np.float64)  ##recirculation fuel species? what unit?
        mix_refin = np.arange(Nspecies,dtype=np.float64) ##goes to Reformer, see the graph. Comes from three sources: part of NG, Steam, and air after split.
        mix_cpox=np.arange(Nspecies,dtype=np.float64)  ##intermediate fuel species assuming all complete oxidized?
        mix_refout=np.arange(Nspecies,dtype=np.float64) ##fuel output after hydrocarbon reforming? ExtReform part of NG
        stack_recirc = np.arange(Nspecies,dtype=np.float64) ##contains onl H2O, Ar, CO2, N2, CO, and H2. NO CH4. In iteration loop
        stack_mix = np.arange(Nspecies,dtype=np.float64) ##= stack_fin[] + stack_recirc[]
        pref_HH = np.arange(Nspecies,dtype=np.float64)  ##After PreReformer step 1: taking care of high hydrocarbon: all high hydrocarbon hone
        pref_CH4 = np.arange(Nspecies,dtype=np.float64) ##After PreReformer step 2: taking care of PreReforming: only CH4, by PreReform
                                                              ##this leads to output SOFC_Fin[]
        cell_ref = np.arange(Nspecies,dtype=np.float64) ##an assumed fuel composition at the stack inlet in the iteration loop. No more CH4. 
        cell_use = np.arange(Nspecies,dtype=np.float64) ##
        cell_exit = np.arange(Nspecies,dtype=np.float64)
        NG_in = np.arange(Nspecies,dtype=np.float64)
        vartemp = np.arange(Nspecies,dtype=np.float64)
        tester = np.arange(Nspecies,dtype=np.float64)
        pref_CH4OLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD = np.arange(Nspecies,dtype=np.float64)
        stack_recircOLD[:]=0.0
        ##air part
        stack_ain = np.arange(Nspecies,dtype=np.float64)
        stack_amix = np.arange(Nspecies,dtype=np.float64)
        stack_arecirc = np.arange(Nspecies,dtype=np.float64)
        stack_arecircOLD = np.arange(Nspecies,dtype=np.float64)
        cell_aexit = np.arange(Nspecies,dtype=np.float64)
        cell_aexhaust = np.arange(Nspecies,dtype=np.float64)
        cell_exhaust = np.arange(Nspecies,dtype=np.float64)

        recirc_VGR0 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR1 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR2 = np.arange(Nspecies,dtype=np.float64)
        recirc_VGR3 = np.arange(Nspecies,dtype=np.float64)

        SOFC_Ain = np.arange(5,dtype=np.float64)
        Fresh_Ain = np.arange(5,dtype=np.float64)
        stack_fin = np.arange(Nspecies,dtype=np.float64) ##The NG part before PreReformer: sum of two parts, pure NG (IR part) and mix_refout (ExtReform part)
        #% Read Independent Variables
        # J=400
        # FU=0.9
        # AU=0.378
        # OCR=2.6
        # IR=0.6
        # Arec=0.5
        # PreReform=0.2

        # cellsize = 550 # cell area (cm2)
        #% Assign General Fixed Values 
        R=8.3145
        F=96485
        Pi=3.14159265359

        #% index
        Index_H2O = 0
        Index_Ar = 1
        Index_CO2 = 2
        Index_O2 = 3
        Index_N2 = 4
        Index_CH4 = 5
        Index_CO = 6
        Index_H2 = 7
        Index_C2H6 = 8
        Index_C3H8 = 9
        Index_C4H10 = 10

        #%
        # Molecular Weights
        MW_fuel[Index_H2O] = 18.01488 # H2O
        MW_fuel[Index_Ar] = 39.948 # Ar
        MW_fuel[Index_CO2] = 44.009 # CO2
        MW_fuel[Index_O2] = 31.998 # O2
        MW_fuel[Index_N2] = 28.0134 # N2
        MW_fuel[Index_CH4] = 16.04276 # CH4
        MW_fuel[Index_CO] = 28.01 # CO
        MW_fuel[Index_H2] = 2.01588 # H2
        MW_fuel[Index_C2H6] = 30.07 # C2H6
        MW_fuel[Index_C3H8] = 44.1 # C3H8
        MW_fuel[Index_C4H10] = 58.12 # C4H10
        #%
        #-- Define Fixed Assumptions for Operation
        max_steam = 0.99 #-- Maximum fuel recirculation fraction

        #%
        #-- Define the inlet fuel feed composition (igfc) default conventional
        NG_fin[Index_H2O] = 0.0013
        NG_fin[Index_Ar] = 0.0008
        NG_fin[Index_CO2] = 0.2043
        NG_fin[Index_O2] = 0
        NG_fin[Index_N2] = 0.006
        NG_fin[Index_CH4] = 0.0583
        NG_fin[Index_CO] = 0.3774
        NG_fin[Index_H2] = 0.3519
        NG_fin[Index_C2H6] = 0.0
        NG_fin[Index_C3H8] = 0.0
        NG_fin[Index_C4H10] = 0.0
        if igfc=='conventional':
            NG_fin[Index_H2O] = 0.0013
            NG_fin[Index_Ar] = 0.0008
            NG_fin[Index_CO2] = 0.2043
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.006
            NG_fin[Index_CH4] = 0.0583
            NG_fin[Index_CO] = 0.3774
            NG_fin[Index_H2] = 0.3519
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='enhanced':
            NG_fin[Index_H2O] = 0.0006
            NG_fin[Index_Ar] = 0.0009
            NG_fin[Index_CO2] = 0.2423
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0064
            NG_fin[Index_CH4] = 0.1022
            NG_fin[Index_CO] = 0.3415
            NG_fin[Index_H2] = 0.3062
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        if igfc=='catalytic':
            NG_fin[Index_H2O] = 0.0004
            NG_fin[Index_Ar] = 0.0003
            NG_fin[Index_CO2] = 0.3465
            NG_fin[Index_O2] = 0
            NG_fin[Index_N2] = 0.0069
            NG_fin[Index_CH4] = 0.3159
            NG_fin[Index_CO] = 0.0914
            NG_fin[Index_H2] = 0.2386
            NG_fin[Index_C2H6] = 0.0
            NG_fin[Index_C3H8] = 0.0
            NG_fin[Index_C4H10] = 0.0
        #%
        #-- Define the standard air composition
        std_ain[Index_H2O] = 0.0104
        std_ain[Index_Ar] = 0.0094
        std_ain[Index_CO2] = 0.0003
        std_ain[Index_O2] = 0.2077
        std_ain[Index_N2] = 0.7722
        std_ain[Index_CH4] = 0
        std_ain[Index_CO] = 0
        std_ain[Index_H2] = 0
        std_ain[Index_C2H6] = 0
        std_ain[Index_C3H8] = 0
        std_ain[Index_C4H10] = 0
        #%
        #-- Define the air separation splits
        splt_ain[Index_H2O] = 0
        splt_ain[Index_Ar] = 0.0673
        splt_ain[Index_CO2] = 0
        splt_ain[Index_O2] = 0.9691
        splt_ain[Index_N2] = 0.0005
        splt_ain[Index_CH4] = 0
        splt_ain[Index_CO] = 0
        splt_ain[Index_H2] = 0
        splt_ain[Index_C2H6] = 0
        splt_ain[Index_C3H8] = 0
        splt_ain[Index_C4H10] = 0
        #%
        zb = -1 #make Brian's 1-based code to 0-based

        #%
        #  (0) Initial Calculations                                                               |
        #-- Define useful paramters
        IR = 1.0
        ExtReform = 1.0 - IR #-- External reformation fraction
        Stoichs = 1.0 / AU #-- Stoichs air
        current = J * cellsize / 1000 # '-- Current (A)

        #-- Calculate the air and fuel needs
        fuelneed = current / 2 / F #-- H2 equiv (mol/s)
        airneed = current / 4 / F #     '-- O2 (mol/s)

        #-- Define iteration parameters
        itermax = 5000 # Total allowed iterations
        ERRTOTAL = 100 #     ' Error value
        ERRTOLER = 1e-8 #    ' Error convergence target

        #-- Define calculation flags
        Flag1 = 1 #  ' 0=no output, 1=write output to spreadsheet

        #%
        #  (1F) External Reformer Calculations                                                    |
        #-- Fuel composition
        NG_fin_sum = 0
        for i in range(Nspecies):
            NG_fin_sum += NG_fin[i]

        #%
        for i in range(Nspecies):
           # print(i,NG_fin[i],NG_fin_sum,NG_fin[i]/NG_fin_sum)
            #a=NG_fin[i]/NG_fin_sum
            NG_mfin[i]=NG_fin[i]/NG_fin_sum
            #print(NG_mfin[i],i)

        #NG_mfin=NG_fin/NG_fin_sum
        fueleqv = NG_mfin[Index_H2] + NG_mfin[Index_CO] + 4 * NG_mfin[Index_CH4] + 7 * NG_mfin[Index_C2H6] + 10 * NG_mfin[Index_C3H8] + 13 * NG_mfin[Index_C4H10]
        NG_flowrate = fuelneed / fueleqv #//fuelneed=mol/s, so NG_flowrate = mol/s

        #//why Const_Convert=3600 * 2.20462 / 1000, making it SLPM (should only *60), 3600=hour in seconds, NOT mole volume=22.4 (litter/mole).  
        #//  2.20462=1/0.454, from kilogram to lbs. /1000 is to make it kilogram because NW_fuel[] are in gram?
        #//  
        #//  but FU_REF1 and FU_REF2 are both very local, only to calculate FU_REF
        #//  FU_ stands for fuel utlization?
        Const_Convert = 3600 * 2.20462 / 1000
        FU_REF1 = NG_flowrate * Const_Convert * fueleqv # //equivalent fuel in lbs/h

        #//FU_REF2: sum (molecular weight * composition) * flowrate 
        FU_REF2 = 0.0;
        for i in range(Nspecies):
            FU_REF2 = FU_REF2 + NG_mfin[i] * MW_fuel[i]


        #//what is 2.0? 0.44? and 0.4?
        #//      0.44 related to CO2 molucular weight 44?
        #//      0.4 ??
        FU_REF2 = 2.0 * NG_flowrate * Const_Convert * FU_REF2 * 0.44 * ExtReform / 0.4 / MW_fuel[Index_O2]
        FU_REF3 = fuelneed / FU * Const_Convert

        #//FU_REF = no unit
        #// the effective FU?
        #//                                  0.44 * ExtReform * Sum(NG_mfin[]*NW_fuel[])
        #//                        fueleqv - -------------------------------------------
        #//                                                0.4 NW_fuel[O2]
        #//  = FU * NG*Flowrate * (--------------------------------------------------------)
        #//                                            fuelneed
        FU_REF = (FU_REF1 - FU_REF2) / FU_REF3

        #            SOFCMP2D4ROM.debugwrite.WriteLine("FU_REF = (FU_REF1 - FU_REF2) / FU_REF3: " + FU_REF.ToString() + "=" + FU_REF1.ToString() + "-" + FU_REF2.ToString() + "/" + FU_REF3.ToString());

        #//NG_in[] = NG_mfin[] mass composition * flowrate * C / FU_REF?
        for i in range(Nspecies):
            NG_in[i] = NG_mfin[i] * (NG_flowrate * Const_Convert) / FU_REF  # //in lbs/h unit?


        #//NG_massflow: sum(inlet  * molecular weight)
        NG_massflow = 0
        for i in range(Nspecies):
            NG_massflow += NG_in[i] * MW_fuel[i];
        #//'-- Reformer air composition
        O2_flowrate = (NG_massflow * 0.44 * ExtReform * 1 / 0.4) / MW_fuel[Index_O2]
        ref_ain[Index_O2] = O2_flowrate

        #//what does it do?
        for i in range(1,Nspecies+1):
            if i != 4: #//zb+4=3=Index_O2
                ref_ain[zb + i] = splt_ain[zb + i] * (ref_ain[Index_O2] / splt_ain[Index_O2]) / std_ain[Index_O2] * std_ain[zb + i]
        #//basically ref_air[]= splt_ain[]  * (std_ain[]/std_ain[O2])   * (ref_ain[O2]/splt_ain[O2]) or
        #ref_air[]= ref_ain[O2] * (splt_ain[]/splt_ain[O2]) * (std_ain[]/std_ain[O2])


        #           //'-- Reformer Mix

        #//debugging8
        c1 = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        c2 = ref_ain[Index_H2O]
        c3 = (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            SOFCMP2D4ROM.debugwrite.WriteLine("For water: original " + c1.ToString() + " air separator " + c2.ToString() + " added " + c3.ToString());
        #//end of debugging8

        mix_refin[Index_H2O] = NG_mfin[Index_H2O] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[Index_H2O] + (NG_flowrate * Const_Convert) / FU_REF * ExtReform
        #            //VB code: mix_refin(zb + 1) = NG_mfin(zb + 1) * (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform + ref_ain(zb + 1) + (NG_flowrate * 3600# * 2.20462 / 1000#) / FU_REF * ExtReform

        #            //i=1 is for H2O, already done
        #            //the below makes more sense than the one with H2O. See the question to Brian
        #            //
        for i in range(2,Nspecies+1):
            mix_refin[zb + i] = NG_mfin[zb + i] * (NG_flowrate * Const_Convert) / FU_REF * ExtReform + ref_ain[zb + i] # //unit=lbs/h?


        #            //'-- After CPOX
        #            //what is fueloxid? fuel oxidide fraction?
        #            //CPOX: partial oxidization?
        fueloxid = 0;
        if ExtReform == 0:
            fueloxid = 0
        else:
        #                //NG_in[] already with proper flow rate unit, so we can simply +
        #                //  CratCH4:  C=1, H=1/4, so CH4=1+4/4=2
        #                //  CratC2H6: 2*1 + 6/4 = 3.5
        #                //  C3H8:     =3*1+8/4=5
        #                //  C4H10:    4*1+10/4=6.5
        #                /*old code, using Ctot, not necessary at all
        #    Ctot = NG_in[Index_CH4] + NG_in[Index_C2H6] + NG_in[Index_C3H8] + NG_in[Index_C4H10]
        #    CratCH4 = NG_in[Index_CH4] / Ctot
        #    CratC2H6 = NG_in[Index_C2H6] / Ctot
        #    CratC2H8 = NG_in[Index_C3H8] / Ctot
         #               double CratC4H10 = NG_in[Index_C4H10] / Ctot;
        #                fueloxid = O2_flowrate / (2 * CratCH4 + 3.5 * CratC2H6 + 5 * CratC2H8 + 6.5 * CratC4H10) / (Ctot * ExtReform)
        #                */
            fueloxid = O2_flowrate / (2 * NG_in[Index_CH4] + 3.5 * NG_in[Index_C2H6] + 5 * NG_in[Index_C3H8] + 6.5 * NG_in[Index_C4H10]) / ExtReform

        #% GetMix_CPoxFromMix_Refin(mix_refin, out mix_cpox, out mix_refout, fueloxid)

        mix_cpox = np.arange(Nspecies,dtype=np.float64)
        mix_cpox[Index_H2O] = mix_refin[Index_H2O] + (2 * mix_refin[Index_CH4] + 3 * mix_refin[Index_C2H6] + 4 * mix_refin[Index_C3H8] + 5 * mix_refin[Index_C4H10]) * fueloxid;
        mix_cpox[Index_CO2] = mix_refin[Index_CO2] + (mix_refin[Index_CH4] + 2 * mix_refin[Index_C2H6] + 3 * mix_refin[Index_C3H8] + 4 * mix_refin[Index_C4H10]) * fueloxid

        mix_cpox[Index_Ar] = mix_refin[Index_Ar]
        mix_cpox[Index_N2] = mix_refin[Index_N2]
        mix_cpox[Index_CO] = mix_refin[Index_CO]
        mix_cpox[Index_H2] = mix_refin[Index_H2]

        mix_cpox[Index_CH4] = mix_refin[Index_CH4] * (1 - fueloxid)
        mix_cpox[Index_C2H6] = mix_refin[Index_C2H6] * (1 - fueloxid)
        mix_cpox[Index_C3H8] = mix_refin[Index_C3H8] * (1 - fueloxid)
        mix_cpox[Index_C4H10] = mix_refin[Index_C4H10] * (1 - fueloxid)

        mix_cpox[Index_O2] = (2 * (mix_refin[Index_CH4] - mix_cpox[Index_CH4]) + 3.5 * (mix_refin[Index_C2H6] - mix_cpox[Index_C2H6]) + 5 * (mix_refin[Index_C3H8] - mix_cpox[Index_C3H8]) + 6.5 * (mix_refin[Index_C4H10] - mix_cpox[Index_C4H10])) - mix_refin[Index_O2]

        mix_cpox[Index_O2] = max(mix_cpox[Index_O2], 0)

        #            //'-- Reformer Exit (get rid of higher hydrocarbons)
        #            //'-------------------------------------------------
        #            //Kevin, why CH4 = 0? All go to CO and H2 and H2O
        mix_refout = np.arange(Nspecies,dtype=np.float64)
        #            //No change species
        mix_refout[Index_Ar] = mix_cpox[Index_Ar]
        mix_refout[Index_CO2] = mix_cpox[Index_CO2]
        mix_refout[Index_O2] = mix_cpox[Index_O2]
        mix_refout[Index_N2] = mix_cpox[Index_N2]

        #            //the actual reformer, see the equations below
        #            //  CH4 +    H2O ->  CO + 3H2
        #            //  C2H6 +  2H2O -> 2CO + 5H2
        #            //  C3H8 +  3H2O -> 3CO + 7H2
        #            //  C4H10 + 4H2O -> 4CO + 9H2
        mix_refout[Index_H2O] = mix_cpox[Index_H2O] - (mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10])
        mix_refout[Index_CO] = mix_cpox[Index_CO] + mix_cpox[Index_CH4] + 2 * mix_cpox[Index_C2H6] + 3 * mix_cpox[Index_C3H8] + 4 * mix_cpox[Index_C4H10] # //added mix_cpox[Index_CO]=0
        mix_refout[Index_H2] = mix_cpox[Index_H2] + 3 * mix_cpox[Index_CH4] + 5 * mix_cpox[Index_C2H6] + 7 * mix_cpox[Index_C3H8] + 9 * mix_cpox[Index_C4H10] #//added mix_cpox[Index_H2]=0

        #            //SOFCMP2D4ROM.debugwrite.WriteLine("mix_refout[Index_H2]=" + mix_refout[Index_H2].ToString()); proven work!

        #            //0-out all species with C
        mix_refout[Index_CH4] = 0;
        mix_refout[Index_C2H6] = 0;
        mix_refout[Index_C3H8] = 0;
        mix_refout[Index_C4H10] = 0;

        #%
        #            SOFCMP2D4ROM.debugwrite.WriteLine("IR=" + IR.ToString() + " ExtReform=" + ExtReform.ToString() + " PreReform=" + PreReform.ToString());
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t mix_refout[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + mix_refout[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + mix_refout[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + mix_refout[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + mix_refout[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + mix_refout[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + mix_refout[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + mix_refout[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + mix_refout[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + mix_refout[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + mix_refout[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + mix_refout[Index_C4H10].ToString("E4"));

        #            //'-- Mix to SOFC
        #            //'--------------
        #            //Kevin: or going to Pre-Reformer?
        for i in range(Nspecies):
            stack_fin[i] = mix_refout[i] + NG_mfin[i] * (NG_flowrate * Const_Convert / FU_REF) * (1.0 - ExtReform)

        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_fin[]:");
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_fin[Index_H2O].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_fin[Index_Ar].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_fin[Index_CO2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_fin[Index_O2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_fin[Index_N2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_fin[Index_CH4].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_fin[Index_CO].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_fin[Index_H2].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_fin[Index_C2H6].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_fin[Index_C3H8].ToString("E4"));
        #            SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_fin[Index_C4H10].ToString("E4"));

        #%//'-------------------------------------------------------------------------------------------
        #            //'|  (1A) Air Inlet                                                                         |
        #            //'-------------------------------------------------------------------------------------------
        air_flowrate = airneed / std_ain[Index_O2]
        for i in range(Nspecies):
            stack_ain[i] = Stoichs * air_flowrate * 3600 * std_ain[i] * 2.20462 / 1000


        #            // *** START ITERATIVE LOOP ***
        #            double Steam1, Steam2;
        Steam1=0.0
        Steam2=0.0
        #           //double Frec; //fuel recirculation ratio
        AddedSteam = 0;
        Frec = 0.05;
        OCRValue=0.0
        #%
        itermax=5000
        for iter in range(1,itermax):

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [2] Calculate the fuel inlet composition to get OCR ratio                              |
        #                //'-------------------------------------------------------------------------------------------

            if iter == 1:  #          // This is the first iteration needing initialization
                for i in range(Nspecies):
                    stack_recirc[i] = stack_fin[i] * 0.05  #; //                  ' Initial condition set to 5% of fuel inlet
                   # stack_mix[i] = stack_fin[i] + stack_recirc[i] #;
                    recirc_VGR3[i]=stack_fin[i]*0.05

                for i in range(Nspecies):
                    stack_mix[i]=stack_fin[i]+stack_recirc[i]+recirc_VGR3[i]
                AddedSteam = 0   #;  //                                        ' Initial condition set to zero
                Frec = 0.05 #;                    //                          ' Initial condition set to 5%
                cell_exit[Index_H2O] = stack_fin[Index_H2O] #; //                    ' Initial condition set to fuel inlet
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O]) #;

                Steam2 = 0;
                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam;
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O];

                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1;


                stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam;

            else: # //Else                                                       ' This is the second + iteration
                Steam1 = OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])- (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_recirc[Index_H2O])

                if cell_exit[Index_H2O] == 0:
                    Steam2 = max_steam
                else:
                    Steam2 = (OCR * (stack_mix[Index_CO2] + stack_mix[Index_CH4] + stack_mix[Index_CO] + 2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - (2 * stack_mix[Index_CO2] + stack_mix[Index_CO] + stack_fin[Index_H2O])) / cell_exit[Index_H2O]


                if Steam2 > max_steam:
                    Frec = max_steam
                else:
                    Frec = Steam2

                if Steam2 < max_steam:
                    AddedSteam = 0
                else:
                    AddedSteam = Steam1
            for i in range(Nspecies):
                stack_mix[i] = stack_fin[i] + stack_recirc[i]+recirc_VGR3[i]

            stack_mix[Index_H2O] = stack_mix[Index_H2O] + AddedSteam # //need to ask Brian

        #                //'MsgBox "Steam1: " & Steam1 & "Steam2: " & Steam2 & "AddedSteam: " & AddedSteam
        #                //'
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  [3] Calculate the fuel inlet composition after prereforming higher hydrocarbons        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - NOT THIS ONE
        #               //  C2H6 +  2H2O -> 2CO + 5H2
        #                //  C3H8 +  3H2O -> 3CO + 7H2
        #               //  C4H10 + 4H2O -> 4CO + 9H2
            pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_Ar] = stack_mix[Index_Ar]
            pref_HH[Index_CO2] = stack_mix[Index_CO2]
            pref_HH[Index_O2] = stack_mix[Index_O2]
            pref_HH[Index_N2] = stack_mix[Index_N2]
            pref_HH[Index_CH4] = stack_mix[Index_CH4]
            pref_HH[Index_CO] = stack_mix[Index_CO] + (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10])
            pref_HH[Index_H2] = stack_mix[Index_H2] + (5 * stack_mix[Index_C2H6] + 7 * stack_mix[Index_C3H8] + 9 * stack_mix[Index_C4H10])
            pref_HH[Index_C2H6] = 0
            pref_HH[Index_C3H8] = 0
            pref_HH[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (4) Calculate the fuel inlet composition after prereforming CH4                        |
        #                //'-------------------------------------------------------------------------------------------
        #                //  CH4 +    H2O ->  CO + 3H2 - only by ratio=PreReform 
            pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_Ar] = pref_HH[Index_Ar]
            pref_CH4[Index_CO2] = pref_HH[Index_CO2]
            pref_CH4[Index_O2] = pref_HH[Index_O2]
            pref_CH4[Index_N2] = pref_HH[Index_N2]
            pref_CH4[Index_CH4] = pref_HH[Index_CH4] * (1 - PreReform)
            pref_CH4[Index_CO] = pref_HH[Index_CO] + PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_H2] = pref_HH[Index_H2] + 3 * PreReform * pref_HH[Index_CH4]
            pref_CH4[Index_C2H6] = pref_HH[Index_C2H6]
            pref_CH4[Index_C3H8] = pref_HH[Index_C3H8]
            pref_CH4[Index_C4H10] = pref_HH[Index_C4H10]

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (5) Reform the CH4 in stack                                                            |
        #                //'-------------------------------------------------------------------------------------------
        #                //Question: why cell_ref[H2O]!=pref_CH4[H2O]?
        #                //  pref_HH[Index_H2O] = stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]);
        #                //  pref_CH4[Index_H2O] = pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * pref_HH[Index_CH4];
        #                //  =stack_mix[Index_H2O] - (2 * stack_mix[Index_C2H6] + 3 * stack_mix[Index_C3H8] + 4 * stack_mix[Index_C4H10]) - PreReform * stack_mix[Index_CH4];
        #                //  There is a difference between - PreReform * stack_mix[Index_CH4] and - stack_mix[Index_CH4]
        #                //Explanation: whether CH4 is reformed in PreReformer or in the stack, it consumes the same amount of water
        #                //  cell_use[Index_H2O]=pref_CH4[Index_H2O]-((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - PreReform * pref_HH[Index_CH4] - ((1-PreReform) * pref_HH[Index_CH4])
        #                //                     =pref_HH[Index_H2O] - pref_HH[Index_CH4]
            cell_ref[Index_H2O] = stack_mix[Index_H2O] - stack_mix[Index_CH4] - 2 * stack_mix[Index_C2H6] - 3 * stack_mix[Index_C3H8] - 4 * stack_mix[Index_C4H10]
            # cell_ref[Index_H2O] = pref_CH4[Index_H2O]-pref_CH4[Index_CH4]-2*pref_CH4[Index_C2H6]-3*pref_CH4[Index_C3H8]-4*pref_CH4[Index_C4H10]
            cell_ref[Index_Ar] = pref_CH4[Index_Ar]
            cell_ref[Index_CO2] = pref_CH4[Index_CO2]
            cell_ref[Index_O2] = pref_CH4[Index_O2]
            cell_ref[Index_N2] = pref_CH4[Index_N2]
            cell_ref[Index_CH4] = 0
            cell_ref[Index_CO] = pref_CH4[Index_CO] + pref_CH4[Index_CH4] + 2 * pref_CH4[Index_C2H6] + 3 * pref_CH4[Index_C3H8] + 4 * pref_CH4[Index_C4H10]
            cell_ref[Index_H2] = pref_CH4[Index_H2] + 3 * pref_CH4[Index_CH4] + 5 * pref_CH4[Index_C2H6] + 7 * pref_CH4[Index_C3H8] + 9 * pref_CH4[Index_C4H10]
            cell_ref[Index_C2H6] = 0
            cell_ref[Index_C3H8] = 0
            cell_ref[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (6) Calculate the fuel outlet composition                                              |
        #                //'-------------------------------------------------------------------------------------------
        #                //FU: per-pass value, because applying on stack_fin[] which are fresh
            cell_use[Index_H2O] = -(stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_Ar] = 0
            cell_use[Index_CO2] = -(stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_O2] = 0
            cell_use[Index_N2] = 0
            cell_use[Index_CH4] = 0
            cell_use[Index_CO] = (stack_fin[Index_CO] + stack_fin[Index_CH4] + 2 * stack_fin[Index_C2H6] + 3 * stack_fin[Index_C3H8] + 4 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_H2] = (stack_fin[Index_H2] + 3 * stack_fin[Index_CH4] + 5 * stack_fin[Index_C2H6] + 7 * stack_fin[Index_C3H8] + 9 * stack_fin[Index_C4H10]) * FU
            cell_use[Index_C2H6] = 0
            cell_use[Index_C3H8] = 0
            cell_use[Index_C4H10] = 0

        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7) Calculate the new recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                cell_exit[i] = cell_ref[i] - cell_use[i]
                stack_recirc[i] = cell_exit[i] * Frec
            #print(cell_ref,"cell_ref")
            #print(cell_use,"cell_use")
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (7a) Calculate the new VGR recirc composition                                               |
        #                //'-------------------------------------------------------------------------------------------  
            for i in range(Nspecies):
                recirc_VGR0[i]=cell_exit[i]-stack_recirc[i]
                recirc_VGR1[i]=recirc_VGR0[i]
            WGSmol=WGS*recirc_VGR1[Index_CO]
            recirc_VGR1[Index_H2O] = recirc_VGR1[Index_H2O] - WGSmol
            recirc_VGR1[Index_CO2] = recirc_VGR1[Index_CO2] + WGSmol
            recirc_VGR1[Index_CO] = recirc_VGR1[Index_CO] - WGSmol
            recirc_VGR1[Index_H2] = recirc_VGR1[Index_H2] + WGSmol
            for i in range(Nspecies):
                recirc_VGR2[i]=recirc_VGR1[i]
            VGRH2O=recirc_VGR1[Index_H2O]*H2OCap
            VGRCO2=recirc_VGR1[Index_CO2]*CO2Cap
            VGRH2=recirc_VGR1[Index_H2]*H2Cap
            recirc_VGR2[Index_H2O]=recirc_VGR2[Index_H2O]-VGRH2O
            recirc_VGR2[Index_CO2]=recirc_VGR2[Index_CO2]-VGRCO2
            recirc_VGR2[Index_H2]=recirc_VGR2[Index_H2]-VGRH2
            for i in range(Nspecies):
                recirc_VGR3[i]=recirc_VGR2[i]*VGR
                cell_exhaust[i] = recirc_VGR2[i] - recirc_VGR3[i]
                
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  (9) Calculate the new air composition with recirculation                               |
        #                //'-------------------------------------------------------------------------------------------
            for i in range(Nspecies):
                stack_amix[i] = stack_ain[i] + stack_arecirc[i]
                cell_aexit[i] = stack_amix[i]

            cell_aexit[Index_O2] = stack_amix[Index_O2] - stack_ain[Index_O2] * AU
            for i in range(Nspecies):
                stack_arecirc[i] = cell_aexit[i] * Arec
                cell_aexhaust[i] = cell_aexit[i] - stack_arecirc[i]

        #                //NOT YET write the following: Frec, stack_mix[i] = stack_fin[i] + stack_recirc[i];
        #                SOFCMP2D4ROM.debugwrite.WriteLine("Iteration " + iter.ToString() + " of " + itermax.ToString());
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t Frec=" + Frec.ToString());

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t cell_ref[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + cell_ref[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + cell_ref[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + cell_ref[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + cell_ref[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + cell_ref[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + cell_ref[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + cell_ref[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + cell_ref[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + cell_ref[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + cell_ref[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + cell_ref[Index_C4H10].ToString("E4"));

        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t stack_recirc[]:");
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2O:\t" + stack_recirc[Index_H2O].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t Ar:\t" + stack_recirc[Index_Ar].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO2:\t" + stack_recirc[Index_CO2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t O2:\t" + stack_recirc[Index_O2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t N2:\t" + stack_recirc[Index_N2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CH4:\t" + stack_recirc[Index_CH4].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t CO:\t" + stack_recirc[Index_CO].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t H2:\t" + stack_recirc[Index_H2].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C2H6:\t" + stack_recirc[Index_C2H6].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C3H8:\t" + stack_recirc[Index_C3H8].ToString("E4"));
        #                SOFCMP2D4ROM.debugwrite.WriteLine("\t\t C4H10:\t" + stack_recirc[Index_C4H10].ToString("E4"));

            oo = pref_CH4[Index_H2O] + pref_CH4[Index_CO] + pref_CH4[Index_CO2] * 2.0
            cc = pref_CH4[Index_CO] + pref_CH4[Index_CO2] + pref_CH4[Index_CH4]
            OCRValue = oo / cc
        #                SOFCMP2D4ROM.debugwrite.WriteLine("OCR value " + OCR.ToString() + " vs. calculated " + OCRValue.ToString());
        #                //'-------------------------------------------------------------------------------------------
        #                //'|  Check for convergence                                                                  |
        #                //'-------------------------------------------------------------------------------------------
            if iter == 1:
                ERRTOTAL = 100
                for i in range(Nspecies):
                    stack_recirc[i] = stack_recircOLD[i];
            else:
                ERRSUM = 0;
                for i in range(Nspecies):
                    ERRSUM = ERRSUM + pow(stack_recirc[i] - stack_recircOLD[i], 2)
                    ERRSUM = ERRSUM + pow(stack_arecirc[i] - stack_arecircOLD[i], 2)
                    stack_recircOLD[i] = stack_recirc[i]
                    stack_arecircOLD[i] = stack_arecirc[i]

                ERRTOTAL = math.sqrt(ERRSUM)

            #print("Iteration=",iter,": Frec=",Frec,"; OCR=",OCRValue,"; Error=",ERRTOTAL,"; Target error=",ERRTOLER)           
            if ERRTOTAL < ERRTOLER:
                break

        #                //'                                        *** END ITERATIVE LOOP ***
        #            } //iter
        #%        
        #            SOFCMP2D4ROM.debugwrite.WriteLine("DONE Iterations");
        #            //'                                        *** END ITERATIVE LOOP ***
        #            //MsgBox "Iterations Required: " & iter

        #            //convert to mole/s
        for i in range(Nspecies):
            stack_fin[i] /= Const_Convert
            cell_exhaust[i] /= Const_Convert
            cell_aexhaust[i] /= Const_Convert
            cell_exit[i] /= Const_Convert
            cell_aexit[i] /= Const_Convert
            pref_CH4[i] /= Const_Convert

        #%
        #            //'-------------------------------------------------------------------------------------------
        #            //'|  Final Results for SOFC-MP: 1-cell gas flow rates in mol/s                              |
        #            //'-------------------------------------------------------------------------------------------
        #            //'-- Air
        SOFC_Ain[0] = stack_amix[Index_O2] / Const_Convert #;  //' O2
        SOFC_Ain[1] = stack_amix[Index_N2] / Const_Convert #;  //' N2
        SOFC_Ain[2] = stack_amix[Index_H2O] / Const_Convert #;  //' H2O
        SOFC_Ain[3] = stack_amix[Index_CO2] / Const_Convert #; //' CO2
        SOFC_Ain[4] = stack_amix[Index_Ar] / Const_Convert #; //' Ar'

        #            //Calculting Frec directly
        FaradayEC = 96487.0
        ooFromCurrent = (cellsize * J * 0.001) / (2.0 * FaradayEC) #; //this is for O atom
        ooNG = stack_fin[Index_H2O] + stack_fin[Index_CO2] * 2.0 + stack_fin[Index_O2] * 2.0 + stack_fin[Index_CO]
        ccNG = stack_fin[Index_CO2] + stack_fin[Index_CH4] + stack_fin[Index_CO] + 2.0 * stack_fin[Index_C2H6] + 3.0 * stack_fin[Index_C3H8] + 4.0 * stack_fin[Index_C4H10]

        CalcR = (ccNG * OCR - ooNG) / ooFromCurrent
        #Frec = CalcR #; //they do equal //not working for VGR
        CalcR=Frec
        #            SOFCMP2D4ROM.debugwrite.WriteLine("calcR=" + CalcR.ToString());

        #            //calculating air side
        o2Consumed4Current = (cellsize * J * 0.001) / (4.0 * FaradayEC) #; //this is for O2
        o2_fresh = o2Consumed4Current / AU
        o2_stack = (o2_fresh - Arec * o2Consumed4Current) / (1.0 - Arec)

        fresh_factor = o2_fresh / std_ain[Index_O2]

        ar_fresh = fresh_factor * std_ain[Index_Ar]
        h2o_fresh = fresh_factor * std_ain[Index_H2O]
        co2_fresh = fresh_factor * std_ain[Index_CO2]
        n2_fresh = fresh_factor * std_ain[Index_N2]

        ar_stack = ar_fresh / (1.0 - Arec)
        h2o_stack = h2o_fresh / (1.0 - Arec)
        co2_stack = co2_fresh / (1.0 - Arec)
        n2_stack = n2_fresh / (1.0 - Arec)

        Fresh_Ain[0] = o2_fresh
        Fresh_Ain[1] = n2_fresh
        Fresh_Ain[2] = h2o_fresh
        Fresh_Ain[3] = co2_fresh
        Fresh_Ain[4] = ar_fresh
        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, ROMdemo() result (O2, N2, H2O, CO2, Ar)="
        #                   + SOFC_Ain[0].ToString() + ","
         #                  + SOFC_Ain[1].ToString() + ","
          #                 + SOFC_Ain[2].ToString() + ","
           #                + SOFC_Ain[3].ToString() + ","
            #               + SOFC_Ain[4].ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result stack (O2, N2, H2O, CO2, Ar)="
         #                  + o2_stack.ToString() + ","
          #                 + n2_stack.ToString() + ","
           #                + h2o_stack.ToString() + ","
            #               + co2_stack.ToString() + ","
             #              + ar_stack.ToString());

        #            SOFCMP2D4ROM.debugwrite.WriteLine("Air side, calculated result fresh (O2, N2, H2O, CO2, Ar)="
         #                  + o2_fresh.ToString() + ","
          #                 + n2_fresh.ToString() + ","
           #                + h2o_fresh.ToString() + ","
            #               + co2_fresh.ToString() + ","
             #              + ar_fresh.ToString());
        #        }

        #% Print outputs
        # print("Fresh air in (J)",Fresh_Ain)
        # print("Stack air in (T)",SOFC_Ain)
            
        # print("Fuel in (F)",stack_fin)



        # print("Fuel recy (R) (lb-mol/hr)",stack_recirc)
        # print("Air recy (V) (lb-mol/hr)",stack_arecirc)

        # The outputs used for SOFC-MP ROM
        # print("Fuel cell inlet (P) (mol/s)",pref_CH4)
        # print("Air cell outlet (U) (mol/s)",cell_aexit)
        # print("Fuel cell outlet (Q) (mol/s)",cell_exit)

        #The outputs used for SOFC-MP ROM
        if Frec>0.9 or Frec<=0:
            succs=0
        else:
            succs=1
        #return(stack_fin,stack_ain/Const_Convert,ref_ain,stack_amix/Const_Convert,Frec,succs)
        #return(stack_fin,SOFC_Ain,Fresh_Ain,Frec,succs)
        return(cell_exit, cell_aexit, pref_CH4, succs)
    
    def DNNROM_4cls(self, maxiteration,trainX_nrm,trainY_nrm,testX_nrm1,testX_nrm2,input_num,output_num,DNNsize):

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]

        learning_rate = 0.001
        training_epochs= maxiteration
        batch_size = int(X_train.shape[0]/3)
        total_len=trainX_nrm.shape[0]

        seed=88
        print("DNN ROM training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for class training data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])
        print("prediction for final testing data set size ", testX_nrm2.shape[0]," * ",testX_nrm2.shape[1])
        
        # Network Parameters
        DNNlayers=len(DNNsize)
        print('Number of layers = ',DNNlayers)
        if DNNlayers>10:
            print('Number of layers needs <=10')
            return()
        if DNNlayers>=1: n_hidden_1 = DNNsize[0]#64 
        if DNNlayers>=2: n_hidden_2 = DNNsize[1]#400 
        if DNNlayers>=3: n_hidden_3 = DNNsize[2]#400
        if DNNlayers>=4: n_hidden_4 = DNNsize[3]#512
        if DNNlayers>=5: n_hidden_5 = DNNsize[4]#512
        if DNNlayers>=6: n_hidden_6 = DNNsize[5]#512
        if DNNlayers>=7: n_hidden_7 = DNNsize[6]#512
        if DNNlayers>=8: n_hidden_8 = DNNsize[7]#512
        if DNNlayers>=9: n_hidden_9 = DNNsize[8]#512
        if DNNlayers>=10: n_hidden_10 = DNNsize[9]#512
        n_input = input_num
        n_classes = output_num
        
        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
        y = tf.placeholder("float", [None, n_classes])
        #tf.compat.v1.disable_eager_execution()
        
        # Store layers weight & bias
        if DNNlayers==1:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==2:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }    
        if DNNlayers==3:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_3, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==4:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==5:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_5, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==6:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_6, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==7:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_7, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==8:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_8, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==9:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_9, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==10:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'h10': tf.Variable(tf.random.normal([n_hidden_9, n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_10, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'b10': tf.Variable(tf.random.normal([n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        
        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            print(DNNlayers)
            if DNNlayers>=1:
                layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
                layer_1 = tf.nn.sigmoid(layer_1)
                #tf.summary.histogram("weights",weights['h1'])
                #tf.summary.histogram("layer", layer_1)
                
            if DNNlayers>=2:   
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                layer_2 = tf.nn.sigmoid(layer_2)
            if DNNlayers>=3:             
                layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
                layer_3 = tf.nn.sigmoid(layer_3)
            if DNNlayers>=4:  
                layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
                layer_4 = tf.nn.sigmoid(layer_4)
            if DNNlayers>=5:  
                layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
                layer_5 = tf.nn.sigmoid(layer_5)
            if DNNlayers>=6:  
                layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
                layer_6 = tf.nn.sigmoid(layer_6)
            if DNNlayers>=7:  
                layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
                layer_7 = tf.nn.sigmoid(layer_7)
            if DNNlayers>=8:  
                layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
                layer_8 = tf.nn.sigmoid(layer_8)
            if DNNlayers>=9:  
                layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
                layer_9 = tf.nn.sigmoid(layer_9)
            if DNNlayers>=10:  
                layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
                layer_10 = tf.nn.sigmoid(layer_10)            
            if DNNlayers==1:
                out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
            if DNNlayers==2:
                out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            if DNNlayers==3:
                out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
            if DNNlayers==4:
                out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            if DNNlayers==5:
                out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
            if DNNlayers==6:
                out_layer = tf.matmul(layer_6, weights['out']) + biases['out']
            if DNNlayers==7:
                out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
            if DNNlayers==8:
                out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
            if DNNlayers==9:
                out_layer = tf.matmul(layer_9, weights['out']) + biases['out']
            if DNNlayers==10:
                out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
            return out_layer   
        
        # Construct model
        pred = multilayer_perceptron(x)
        cost = tf.reduce_mean(tf.square(pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        # Run the graph in the session 
        predict = np.array([])
        count_converge= [0] * training_epochs
        prev_cost=10000000.
        #saver = tf.train.Saver()
        #tf.reset_default_graph()
        
        config = tf.ConfigProto(device_count={"CPU": 1}, # limit to num_cpu_core CPU usage
                    inter_op_parallelism_threads = 0, 
                    intra_op_parallelism_threads = 28,
                    )
        init = tf.global_variables_initializer()
        start=time.time()
        with tf.Session(config=config) as sess:
            
            sess.run(init)
            for epoch in range(training_epochs):
                    
                    avg_cost = 0.
                    total_batch = int(total_len/batch_size)
                    for i in range(total_batch-1):
                        batch_x = X_train[i*batch_size:(i+1)*batch_size]
                        batch_y = y_train[i*batch_size:(i+1)*batch_size]
                        _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                        avg_cost += c / total_batch
                        if epoch==training_epochs-1:
                            predict = np.append(predict, p)
                    # print  ('epoch', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost))
                    val_c, val_p=sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
                    test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
                    test_p2=sess.run(pred, feed_dict={x: testX_nrm2})
                    
                    #count cost convergence for validation
                    count_converge[epoch]=val_c
                    
                    if epoch %2000 == 0 :
                        end=time.time()
                        print  ('epoch ',(epoch+1),'  training cost =','{:.5f}'.format(avg_cost),'  validation cost =', '{:.5f}'.format(val_c),'  training time (s/100epochs)= ','{:.5f}'.format(end-start))
                        start=time.time()
                    #for validation set if no improvement then break
                    if epoch == training_epochs-1:
                        print('break the loop at maximum iteration')
                    if epoch %2000 ==0 and val_c>=prev_cost:
                        break
                        #print("val cost increase !!!")
                    if epoch %2000 ==0:
                        prev_cost=val_c
            #saver.save(sess, r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\DNN')
            #saver.save(sess, DNN_save_file)
            
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            test_p2=sess.run(pred, feed_dict={x: testX_nrm2})
            
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            #for k,v in zip(variables_names, values):
            # print(k, v)
            # for v in values: 
            #     print(v) 
            sess.close()
        
        tf.reset_default_graph()
        return(test_p1,test_p2, values)

    def DNNCls(self, maxiteration,trainX_nrm,trainY_nrm,testX_nrm,testY_nrm,input_num_units):
        
        hidden_num_units = 500
        output_num_units = 2
        seed=88

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]
        print("DNN classification training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])   
        print("prediction for final testing data set size ", testX_nrm.shape[0]," * ",testX_nrm.shape[1])

        # define placeholders
        xc = tf.placeholder(tf.float32, [None, input_num_units])
        yc = tf.placeholder(tf.float32, [None, output_num_units])

        # set remaining variables
        epochs = maxiteration
        batch_size = int(X_train.shape[0]/2)  #1500
        learning_rate = 0.001

        ### define weights and biases of the neural network 
        weights = {
            'hidden': tf.Variable(tf.random_uniform([input_num_units, hidden_num_units],-1,1,seed=seed)),
            #'hidden': tf.Variable(tf.random_normal([input_num_units, hidden_num_units], 0, 1,seed=seed)),
            'output': tf.Variable(tf.random_normal([hidden_num_units, output_num_units],0, 0.1, seed=seed))
        }

        biases = {
            #'hidden': tf.Variable(tf.random_normal([hidden_num_units], seed=seed)),
            'hidden': tf.Variable(tf.random_uniform([hidden_num_units], -1,1,seed=seed)),
            'output': tf.Variable(tf.random_normal([output_num_units], seed=seed))
        } 
        #
        hidden_layer = tf.add(tf.matmul(xc, weights['hidden']), biases['hidden'])
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        tf.summary.histogram("weights_hidden",weights['hidden'])
        tf.summary.histogram("biases_hidden",biases['hidden'])
        tf.summary.histogram("layer_hidden", hidden_layer)

        output_layer = tf.matmul(hidden_layer, weights['output']) + biases['output']
        tf.summary.histogram("weights_output",weights['output'])
        tf.summary.histogram("biases_output",biases['output'])
        tf.summary.histogram("layer_output", output_layer)
        
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output_layer, labels=yc))

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        pred=output_layer

        init = tf.global_variables_initializer()

        #write this after all the summary
        #merged = tf.summary.merge_all()
        #writer = tf.summary.FileWriter(cwd)
        #saver = tf.train.Saver()
        # covert output scalar to vector https://stackoverflow.com/questions/43543594/label-scalar-into-one-hot-in-tensorr-flow-code
        def dense_to_one_hot(labels_dense, num_classes=2):
            """Convert class labels from scalars to one-hot vectors"""
            num_labels = labels_dense.shape[0]
            #index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            for ii in range(num_labels):
                labels_one_hot[ii,int(labels_dense[ii])]=1
            #labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
            return labels_one_hot
        
        prev_cost=0
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(epochs):
                avg_cost = 0
                total_batch = int(X_train.shape[0]/batch_size)
                for i in range(total_batch):

                    batch_x = X_train[i*batch_size:(i+1)*batch_size,]
                    batch_y = y_train[i*batch_size:(i+1)*batch_size,]
                    batch_y = dense_to_one_hot(batch_y)
                    _, c = sess.run([optimizer, cost], feed_dict = {xc: batch_x, yc: batch_y})
                    avg_cost += c / total_batch

                #write tensorboard summary
                #summary_avg_cost = tf.Summary()
                #summary_avg_cost.value.add(tag="avg_cost", simple_value=avg_cost)
                #writer.add_summary(summary_avg_cost, epoch)
                #writer.add_summary(summary, epoch)
                #find predictions on val set #location of the catagory, can be greater than 2
                pred_temp = tf.equal(tf.argmax(output_layer, 1), tf.argmax(yc, 1))
                accuracy = tf.reduce_mean(tf.cast(pred_temp, "float"))
                val_acc=accuracy.eval({xc: val_x, yc: dense_to_one_hot(val_y)})
                test_acc=accuracy.eval({xc: testX_nrm, yc: dense_to_one_hot(testY_nrm)})

                #print ("Validation Accuracy:", accuracy.eval({x: val_x, y: dense_to_one_hot(val_y)})) 
                if epoch %2000 ==0 :print ('Epoch:', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost),"  Validation accuracy:", val_acc,"  Test accuracy:",test_acc) 
                
                if epoch == epochs-1:
                        print('break the loop at maximum iteration')
                if epoch %2000 ==0 and val_acc<=prev_cost:
                    break
                #print("val cost increase !!!")
                if epoch %2000 ==0:
                    prev_cost=val_acc
            test_p1=sess.run(pred, feed_dict={xc: testX_nrm})
            test_p0=sess.run(tf.argmax(test_p1,1))
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            #saver.save(sess, DNNcls_save_file)
            sess.close()
        tf.reset_default_graph()
        return(val_acc,test_acc,test_p0, values)

    def DNNROM(self, maxiteration,trainX_nrm,trainY_nrm,testX_nrm1,input_num,output_num,DNNsize):

        split_size = int(trainX_nrm.shape[0]*0.8)

        X_train, val_x = trainX_nrm[:split_size],trainX_nrm[split_size:]
        y_train, val_y = trainY_nrm[:split_size], trainY_nrm[split_size:]

        learning_rate = 0.001
        training_epochs = maxiteration
        batch_size = int(X_train.shape[0]/3)
        total_len=trainX_nrm.shape[0]

        seed=88
        print("DNN ROM training start ...")
        print("training data set size  ", X_train.shape[0]," * ",X_train.shape[1])
        print("validation data set size", val_x.shape[0]," * ",val_x.shape[1])
        print("prediction for testing data set size", testX_nrm1.shape[0]," * ",testX_nrm1.shape[1])
        
        # Network Parameters
        DNNlayers=len(DNNsize)
        print('Number of layers = ',DNNlayers)
        if DNNlayers>10:
            print('Number of layers needs <=10')
            return()
        if DNNlayers>=1: n_hidden_1 = DNNsize[0]#64 
        if DNNlayers>=2: n_hidden_2 = DNNsize[1]#400 
        if DNNlayers>=3: n_hidden_3 = DNNsize[2]#400
        if DNNlayers>=4: n_hidden_4 = DNNsize[3]#512
        if DNNlayers>=5: n_hidden_5 = DNNsize[4]#512
        if DNNlayers>=6: n_hidden_6 = DNNsize[5]#512
        if DNNlayers>=7: n_hidden_7 = DNNsize[6]#512
        if DNNlayers>=8: n_hidden_8 = DNNsize[7]#512
        if DNNlayers>=9: n_hidden_9 = DNNsize[8]#512
        if DNNlayers>=10: n_hidden_10 = DNNsize[9]#512
        n_input = input_num
        n_classes = output_num
        
        # tf Graph input
        x = tf.placeholder("float", [None, n_input],name="x")
        y = tf.placeholder("float", [None, n_classes])
        #tf.compat.v1.disable_eager_execution()
        
        # Store layers weight & bias
        if DNNlayers==1:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_1, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==2:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }    
        if DNNlayers==3:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_3, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==4:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_4, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==5:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_5, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==6:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_6, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==7:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_7, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
            
        if DNNlayers==8:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_8, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==9:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_9, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        if DNNlayers==10:
            weights = {
                'h1': tf.Variable(tf.random.normal([n_input, n_hidden_1], 0, 0.1,seed=seed)),
                'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2], 0, 0.1,seed=seed)),
                'h3': tf.Variable(tf.random.normal([n_hidden_2, n_hidden_3], 0, 0.1,seed=seed)),
                'h4': tf.Variable(tf.random.normal([n_hidden_3, n_hidden_4], 0, 0.1,seed=seed)),
                'h5': tf.Variable(tf.random.normal([n_hidden_4, n_hidden_5], 0, 0.1,seed=seed)),
                'h6': tf.Variable(tf.random.normal([n_hidden_5, n_hidden_6], 0, 0.1,seed=seed)),
                'h7': tf.Variable(tf.random.normal([n_hidden_6, n_hidden_7], 0, 0.1,seed=seed)),
                'h8': tf.Variable(tf.random.normal([n_hidden_7, n_hidden_8], 0, 0.1,seed=seed)),
                'h9': tf.Variable(tf.random.normal([n_hidden_8, n_hidden_9], 0, 0.1,seed=seed)),
                'h10': tf.Variable(tf.random.normal([n_hidden_9, n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_hidden_10, n_classes], 0, 0.1,seed=seed))
            }
            
            biases = {
                'b1': tf.Variable(tf.random.normal([n_hidden_1], 0, 0.1,seed=seed)),
                'b2': tf.Variable(tf.random.normal([n_hidden_2], 0, 0.1,seed=seed)),
                'b3': tf.Variable(tf.random.normal([n_hidden_3], 0, 0.1,seed=seed)),
                'b4': tf.Variable(tf.random.normal([n_hidden_4], 0, 0.1,seed=seed)),
                'b5': tf.Variable(tf.random.normal([n_hidden_5], 0, 0.1,seed=seed)),
                'b6': tf.Variable(tf.random.normal([n_hidden_6], 0, 0.1,seed=seed)),
                'b7': tf.Variable(tf.random.normal([n_hidden_7], 0, 0.1,seed=seed)),
                'b8': tf.Variable(tf.random.normal([n_hidden_8], 0, 0.1,seed=seed)),
                'b9': tf.Variable(tf.random.normal([n_hidden_9], 0, 0.1,seed=seed)),
                'b10': tf.Variable(tf.random.normal([n_hidden_10], 0, 0.1,seed=seed)),
                'out': tf.Variable(tf.random.normal([n_classes], 0, 0.1,seed=seed))
            }
        
        # Create model
        def multilayer_perceptron(x):
            # Hidden layer with RELU activation
            print(DNNlayers)
            if DNNlayers>=1:
                layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
                layer_1 = tf.nn.sigmoid(layer_1)
                #tf.summary.histogram("weights",weights['h1'])
                #tf.summary.histogram("layer", layer_1)
                
            if DNNlayers>=2:   
                layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
                layer_2 = tf.nn.sigmoid(layer_2)
            if DNNlayers>=3:             
                layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
                layer_3 = tf.nn.sigmoid(layer_3)
            if DNNlayers>=4:  
                layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
                layer_4 = tf.nn.sigmoid(layer_4)
            if DNNlayers>=5:  
                layer_5 = tf.add(tf.matmul(layer_4, weights['h5']), biases['b5'])
                layer_5 = tf.nn.sigmoid(layer_5)
            if DNNlayers>=6:  
                layer_6 = tf.add(tf.matmul(layer_5, weights['h6']), biases['b6'])
                layer_6 = tf.nn.sigmoid(layer_6)
            if DNNlayers>=7:  
                layer_7 = tf.add(tf.matmul(layer_6, weights['h7']), biases['b7'])
                layer_7 = tf.nn.sigmoid(layer_7)
            if DNNlayers>=8:  
                layer_8 = tf.add(tf.matmul(layer_7, weights['h8']), biases['b8'])
                layer_8 = tf.nn.sigmoid(layer_8)
            if DNNlayers>=9:  
                layer_9 = tf.add(tf.matmul(layer_8, weights['h9']), biases['b9'])
                layer_9 = tf.nn.sigmoid(layer_9)
            if DNNlayers>=10:  
                layer_10 = tf.add(tf.matmul(layer_9, weights['h10']), biases['b10'])
                layer_10 = tf.nn.sigmoid(layer_10)            
            if DNNlayers==1:
                out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
            if DNNlayers==2:
                out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
            if DNNlayers==3:
                out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
            if DNNlayers==4:
                out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
            if DNNlayers==5:
                out_layer = tf.matmul(layer_5, weights['out']) + biases['out']
            if DNNlayers==6:
                out_layer = tf.matmul(layer_6, weights['out']) + biases['out']
            if DNNlayers==7:
                out_layer = tf.matmul(layer_7, weights['out']) + biases['out']
            if DNNlayers==8:
                out_layer = tf.matmul(layer_8, weights['out']) + biases['out']
            if DNNlayers==9:
                out_layer = tf.matmul(layer_9, weights['out']) + biases['out']
            if DNNlayers==10:
                out_layer = tf.matmul(layer_10, weights['out']) + biases['out']
            return out_layer   
        
        # Construct model
        pred = multilayer_perceptron(x)
        cost = tf.reduce_mean(tf.square(pred-y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        
        # Run the graph in the session 
        predict = np.array([])
        count_converge= [0] * training_epochs
        prev_cost=10000000.
        #saver = tf.train.Saver()
        #tf.reset_default_graph()
        
        config = tf.ConfigProto(device_count={"CPU": 1}, # limit to num_cpu_core CPU usage
                    inter_op_parallelism_threads = 0, 
                    intra_op_parallelism_threads = 28,
                    )
        init = tf.global_variables_initializer()
        start=time.time()
        with tf.Session(config=config) as sess:
            
            sess.run(init)
            for epoch in range(training_epochs):
                    
                    avg_cost = 0.
                    total_batch = int(total_len/batch_size)
                    for i in range(total_batch-1):
                        batch_x = X_train[i*batch_size:(i+1)*batch_size]
                        batch_y = y_train[i*batch_size:(i+1)*batch_size]
                        _, c, p = sess.run([optimizer, cost, pred], feed_dict={x: batch_x, y: batch_y})
                        avg_cost += c / total_batch
                        if epoch==training_epochs-1:
                            predict = np.append(predict, p)
                # print  ('epoch', (epoch+1), 'cost =', '{:.5f}'.format(avg_cost))
                    val_c, val_p=sess.run([cost, pred], feed_dict={x: val_x, y: val_y})
                    test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
                    
                    #count cost convergence for validation
                    count_converge[epoch]=val_c
                    
                    if epoch %2000 == 0 :
                        end=time.time()
                        print  ('epoch ',(epoch+1),'  training cost =','{:.5f}'.format(avg_cost),'  validation cost =', '{:.5f}'.format(val_c),'  training time (s/100epochs)= ','{:.5f}'.format(end-start))
                        start=time.time()
                    #for validation set if no improvement then break
                    if epoch == training_epochs-1:
                        print('break the loop at maximum iteration')
                    if epoch %2000 ==0 and val_c>=prev_cost:
                        break
                        #print("val cost increase !!!")
                    if epoch %2000 ==0:
                        prev_cost=val_c
            #saver.save(sess, r'E:\SOFC\ARPA-E\Work2020\codes\DNN_rom\DNN')
            #saver.save(sess, DNN_save_file)
            
            test_p1=sess.run(pred, feed_dict={x: testX_nrm1})
            
            variables_names =[v.name for v in tf.trainable_variables()]
            values = sess.run(variables_names)
            #for k,v in zip(variables_names, values):
            # print(k, v)

            sess.close()
        
        tf.reset_default_graph()
        return(test_p1, values)

    def summarize_SimuResult(self, source_path, indcase, exclude_case = 1, display_detail = False):
        '''
        The function extracts simulation results
            exclude_case = -1: all cases included
            exclude_case = 0: exclude failed cases only
            exclude_case = 1: exclude both failed and non-converged cases
        '''
        print('############################################################\
              \nSummarize simulation results\
              \n############################################################')
        
        ## Step 1: load simulation outputs to Y4kriging
        numcase4kriging = 0 # number of cases for kriging
        indcase4kriging = [] # index of cases for kriging, start from 1
        S4kriging = None # simulation inputs for kriging
        Y4kriging = None # simulation outputs for kriging
        for icase in indcase:
            # load SOFC_MP_ROM.dat to df1
            strcase = 'Case'+str(icase-1)+'Value'
            inputfilename = source_path+'/Cases/Case'+str(icase-1).zfill(5)+'/SOFC_MP_ROM.dat'
            if os.path.exists(inputfilename):
                text_input=open(inputfilename,"r")
                lines=text_input.readlines()
                
                if len(lines) == 0: 
                    continue #print('Empty case')
                if lines[1].strip() == '#FAILED': 
                    continue #print('"preprocessor" failed case')
                
                df0 = pd.DataFrame(np.array([['1a', '1b']]),columns=['Name', strcase])
                df1 = pd.DataFrame(np.array([['1a', '1b']]),columns=['Name', strcase])
                for j in range(len(lines)):
                    if j>1: # skip first two lines
                        str01 = lines[j].split('=')
                        str01[0]=str01[0].rstrip()
                        str01[0]=str01[0].lstrip()
                        
                        if len(str01) == 1: continue
                        
                        # convert variables in SOFC_MP_ROM.dat to xxx_xxx format
                        str_tmp = str01[0].strip().split()
                        str_tmp = '_'.join(str_tmp)
                        df0['Name']=str_tmp
                        df0[strcase]=float(str01[1])
                        if j==2:
                            df1["Name"]=df0["Name"]
                            df1[strcase]=df0[strcase]
                        else:
                            df1=pd.concat([df1,df0],sort=False, ignore_index=True)
                
                # exclude failed or non-converged cases
                if int(df1.loc[0, [strcase]]) >= exclude_case:
                    numcase4kriging += 1
                    indcase4kriging.append(icase)
                    if numcase4kriging == 1:
                        Y4kriging = df1
                    else:
                        Y4kriging = pd.concat([Y4kriging, df1[strcase]], sort=False, axis=1)

        ## Step 2: load simulation inputs to S4kriging
        inputfilename = source_path+'/LHS.dat'
        if os.path.exists(inputfilename):
            text_input=open(inputfilename,"r")
            lines=text_input.readlines()
            
            for j in range(len(lines)):
                if j == 1:
                    list_tmp = lines[j].strip().split()
                    list_tmp = list_tmp[2:] # 0: case; 1: No.
                    df2 = pd.DataFrame(list_tmp,columns=['Name'])
                if j > 1:
                    list_tmp = lines[j].strip().split()
                    strcase = 'Case'+str(int(list_tmp[0])-1)+'Value'
                    list_tmp = list_tmp[1:] # 0: case No.
                    df2[strcase] = list_tmp
            S4kriging = df2
        
        ## Step 3: display simulation input and output
        if exclude_case == 1:
            print('Converged simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        elif exclude_case == 0:
            print('Converged and non-converged simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        else:
            print('Simulation results are summarized from '+ str(numcase4kriging)+' cases:')
        print(*indcase4kriging)
        print('\nSelect from the following input variables for training:')
        for i in range(S4kriging.index.size):
            print(i+1, ':', S4kriging.loc[i, 'Name'], end = '\t\n')
        print('\nSelect from the following output variables for training:')
        for i in range(Y4kriging.index.size):
            print(i+1, ':', Y4kriging.loc[i, 'Name'], end = '\t\n')
            
        if display_detail == True:
            print('\n')
            print(S4kriging)
            print('\n')
            print(Y4kriging)
        
        ## Step 4: create allResults.dat
        indS = list(S4kriging.index)
        indY = list(Y4kriging.index)
        indS = [x+1 for x in indS]
        indY = [x+1 for x in indY]
        if len(indcase4kriging) == 0 or len(indS) == 0 or len(indY) == 0:
            print('Error: No data available for training')   
        
        with open(self.allresultsFile, 'w') as f:
            for i in indS:
                f.write(S4kriging.loc[i-1, 'Name'] + '\t')
            for i in indY:
                f.write(Y4kriging.loc[i-1, 'Name'] + '\t')
            f.write('\n')
            for i in indcase4kriging:
                strcase = 'Case'+str(i-1)+'Value'
                for j in indS:
                    f.write('{:11.4E}\t'.format(float(S4kriging.loc[j-1, strcase])))
                for j in indY:
                    f.write('{:11.4E}\t'.format(float(Y4kriging.loc[j-1, strcase])))
                f.write('\n')
                
        with open(self.allresults_infoFile, 'w') as f:
            f.write('input_col\toutput_col\n')
            f.write(str(len(indS))+'\t'+str(len(indY))+'\n')
            
    def file_read(self, FileName):
        '''
        This function loads the kriginginputFile, 
        infoFile and predictioninputFile
        '''
        namearray = []
        valuearray = []
        with open(FileName) as f:
            i = 0
            for line in f.readlines():
                if i == 0:
                    namearray = line.strip().split()
                else:
                    linestr = line.strip().split()
                    linenum = [float(lineele) for lineele in linestr]
                    valuearray.append(linenum)
                i += 1
               
        return namearray, np.array(valuearray)
    
    def variables(self):
        print('input variables:')
        for i in range(len(self.Sname)):
            print(i+1, ':', self.Sname[i], end = '\t\n')
            
        print('\noutput variables:')
        for i in range(len(self.Yname)):
            print(i+1, ':', self.Yname[i], end = '\t\n')
    
    def variable_options(self, display = False):
        names_input = [
            "Average_CellVoltage",
            "Average_CurrentDensity",
            "BackEnvironmentT",
            "BottomEnvironmentT",
            "CellFuelFlowRate",
            "CellOxidantFlowRate",
            "FrontEnvironmentT",
            "Fuel_Utilization",
            "FuelH2",
            "FuelH2O",
            "FuelCO",
            "FuelCO2",
            "FuelCH4",
            "FuelN2",
            "FuelTemperature",
            "FuelTOnTop",
            "FuelRecyclePercent",
            "FuelHTXEffectiveness",
            "FuelNGTemperature",
            "FuelNGHTXDeltaT",
            "Internal_Reforming",
            "nCells",
            "Oxidant_Recirculation",
            "OxidantRecyclePercent",
            "OxygenToCarbon_Ratio",
            "OxidantO2",
            "OxidantN2",
            "OxidantH2O",
            "OxidantCO2",
            "OxidantAr",
            "OxidantTemperature",
            "OxidantTOnTop",
            "PreReform",
            "SideEnvironmentT",
            "Simulation_Option",
            "Stack_Fuel_Utilization",
            "Stack_Oxidant_Utilization",
            "StackFuelFlowRate",
            "StackFuelFlowRateH2O",
            "StackFuelFlowRateCO",
            "StackFuelFlowRateCO2",
            "StackFuelFlowRateCH4",
            "StackFuelFlowRateH2",
            "StackFuelFlowRateN2",
            "StackOxidantFlowRate",
            "StackOxidantFlowRateO2",
            "StackOxidantFlowRateN2",
            "StackOxidantFlowRateH2O",
            "StackOxidantFlowRateCO2",
            "StackOxidantFlowRateAr",
            "StackVoltage",
            "SystemPressure",
            "TopEnvironmentT",
            "VGRRate",
            "VGRTemperature",
            "VGRH2OPassRate",
            "VGRH2PassRate",
            "VGRCO2CaptureRate",
            "VGRCOConvertRate"
        ]

        units_input = [
            "V",
            "A/m^2",
            "C",
            "C",
            "mol/s",
            "mol/s",
            "C",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "C",
            "C",
            "%",
            "-",
            "C",
            "C",
            "-",
            "-",
            "-",
            "%",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "C",
            "C",
            "-",
            "C",
            "-",
            "-",
            "-",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "mol/s",
            "V",
            "atm",
            "C",
            "-",
            "C",
            "-",
            "-",
            "-",
            "-"
        ]
        
        names_output = [
            'SimulationStatus',
            'Stack_Voltage',
            'Avg_cell_voltage',
            'Stack_Current',
            'Avg_current_density',
            'Max_current_density',
            'Min_current_density',
            'Avg_Cell_Temperature',
            'Max_Cell_Temperature',
            'Min_Cell_Temperature',
            'Delta_Cell_Temperature',
            'Outlet_Fuel_Temperature',
            'Delta_Fuel_Temperature',
            'Outlet_Air_Temperature',
            'Delta_Air_Temperature',
            'Air_Heat_Exchanger_Effectiveness',
            'Fuel_Utilization',
            'Air_Utilization',
            'Outlet_Fuel_Flowrate',
            'Outlet_Fuel_H2',
            'Outlet_Fuel_H2O',
            'Outlet_Fuel_CO',
            'Outlet_Fuel_CO2',
            'Outlet_Fuel_CH4',
            'Outlet_Fuel_N2',
            'Outlet_Air_Flowrate',
            'Outlet_Air_O2',
            'Outlet_Air_N2',
            'Outlet_Air_H2O',
            'Outlet_Air_CO2',
            'Outlet_Air_Ar',
            'Total_Power',
            'Air_Enthalpy_Change',
            'Fuel_Enthalpy_Change',
            'External_Heat',
            'Electrical_Efficiency',
            'Stack_Efficiency',
            'Air_Inlet_Temperature',
            'FSI_Temperature',
            'FSI_Flowrate',
            'FSI_H2_MF',
            'FSI_H2O_MF',
            'FSI_CO_MF',
            'FSI_CO2_MF',
            'FSI_CH4_MF',
            'FSI_N2_MF',
            'Fuel_Temperature_after_Mix',
            'Fuel_Temperature_before_Gibbs_Reactor',
            'Fuel_Heat_Exchanger_Effectiveness'
        ]

        units_output = [
            '-',
            'V',
            'V',
            'A',
            'A/m2',
            'A/m2',
            'A/m2',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            'K',
            '-',
            '-',
            '-',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            'W',
            'W',
            'W',
            'W',
            '-',
            '-',
            'K',
            'K',
            'mol/s',
            '-',
            '-',
            '-',
            '-',
            '-',
            '-',
            'K',
            'K',
            '-'
        ]

        if display == True:
            print('Options of input variable:')
            for i in range(len(names_input)):
                print(i+1, ':', names_input[i]+', ['+units_input[i]+']', end = '\t\n')
            print('Options of output variable:')
            for i in range(len(names_output)):
                print(i+1, ':', names_output[i]+', ['+units_output[i]+']', end = '\t\n')
        
        return names_input, units_input, names_output, units_output

    def buildROM(self, frac4ROM = 80, preprocessor_name = None, igfc = None,
                 filter_enabled = True, z_thres = 5, inputbasefilename = None):
        '''
        The function build the ROM for certain input/output variables
        '''
        print('############################################################\
              \nBuild the ROM\
              \n############################################################')
        
        if not os.path.exists(self.allresultsFile) or not os.path.exists(self.allresults_infoFile):
            sys.exit('Code terminated: essential files missing')
        
        ################## Step 1: train the classifier ##################
        SYname, SYvalue = self.file_read(self.allresultsFile)
        infoname, infovalue = self.file_read(self.allresults_infoFile)
        [S_row, Y_row, S_col, Y_col] = [len(SYvalue), len(SYvalue), int(infovalue[0,0]), int(infovalue[0,1])]
        Sname = copy.deepcopy(SYname[:S_col])
        Yname = copy.deepcopy(SYname[S_col:])
        Svalue = copy.deepcopy(SYvalue[:, :S_col])
        Yvalue = copy.deepcopy(SYvalue[:, S_col:])
        
        ## 1.1 determine indS, indY
        indS = list(range(1, S_col+1))
        indY = []
        for i in range(Y_col):
            Y_tmp = Yvalue[:, i]
            if len(np.unique(Y_tmp))>5:
                indY.append(i+1)
        
        indS_index = [i-1 for i in indS]
        indY_index = [i-1 for i in indY]
        
        ## 1.2 determine if enabling classifier or not
        if Yname[0] == 'SimulationStatus':
            cls_enabled = True
        else:
            cls_enabled = False

        ## 1.3-- call "preprocessor", train classifier, etc.
        if cls_enabled == True:
            ## 1.3 split dataset into 3 sets
            if frac4ROM >= 0:
                size_tmp1 = int(S_row*frac4ROM/100.0)
                size_tmp2 = int(size_tmp1*50.0/100.0)
                size_tmp3 = int(S_row*(1-frac4ROM/100.0))
            else:
                size_tmp1 = int(S_row*0.8)
                size_tmp2 = int(size_tmp1*50.0/100.0)
                size_tmp3 = int(S_row*0.2)
            
            ## 1.4 change all SimulationStatus = -1 to 0
            for i in range(S_row):
                if Yvalue[i, 0] == -1: Yvalue[i, 0] = 0
                    
            Sname_4cls = [ Sname[i] for i in indS_index]
            Yname_4cls = [ Yname[i] for i in indY_index]
            
            S_4cls_ROM_train_tmp = Svalue[:size_tmp2, :]
            Y_4cls_ROM_train_tmp = Yvalue[:size_tmp2, :]
            S_4cls_ROM_train_tmp = S_4cls_ROM_train_tmp[Y_4cls_ROM_train_tmp[:, 0] == 1, :]
            Y_4cls_ROM_train_tmp = Y_4cls_ROM_train_tmp[Y_4cls_ROM_train_tmp[:, 0] == 1, :]
            S_4cls_ROM_train = S_4cls_ROM_train_tmp[:, indS_index]
            Y_4cls_ROM_train = Y_4cls_ROM_train_tmp[:, indY_index]
            
            S_4cls_ROM_vali_tmp = Svalue[size_tmp2:size_tmp1, :]
            Y_4cls_ROM_vali_tmp = Yvalue[size_tmp2:size_tmp1, :]
            S_4cls_ROM_vali_cls_train = S_4cls_ROM_vali_tmp[:, indS_index]
            Y_4cls_ROM_vali = Y_4cls_ROM_vali_tmp[:, indY_index]
            Y_4cls_cls_train = Y_4cls_ROM_vali_tmp[:, 0]
            
            S_4cls_vali = Svalue[S_row-size_tmp3:, indS_index]
            Y_4cls_vali = Yvalue[S_row-size_tmp3:, 0]

            ## 1.5 normalize dataset
            meanS=S_4cls_ROM_train.mean(axis=0)
            stdS=S_4cls_ROM_train.std(axis=0)

            meanY=Y_4cls_ROM_train.mean(axis=0)
            stdY=Y_4cls_ROM_train.std(axis=0)

            S_4cls_ROM_train_nrm=(S_4cls_ROM_train-meanS)/stdS
            Y_4cls_ROM_train_nrm=(Y_4cls_ROM_train-meanY)/stdY
            S_4cls_ROM_vali_cls_train_nrm=(S_4cls_ROM_vali_cls_train-meanS)/stdS
            S_4cls_vali_nrm=(S_4cls_vali-meanS)/stdS  
            
            ## 1.6 call DNN rom
            maxiteration = 50000
            DNNsize = [64, 200, 200, 256]
            Y_4cls_ROM_vali_cls_train_nrm_pred, Y_4cls_vali_nrm_pred, cls_ROM_values =             self.DNNROM_4cls(maxiteration, S_4cls_ROM_train_nrm, Y_4cls_ROM_train_nrm,                              S_4cls_ROM_vali_cls_train_nrm, S_4cls_vali_nrm, len(indS), len(indY), DNNsize)

            ## 1.7 call preprocessor
            succs_cls_training = np.zeros((S_4cls_ROM_vali_cls_train_nrm.shape[0],1),dtype=np.float64)
            succs_cls_testing = np.zeros((S_4cls_vali_nrm.shape[0],1),dtype=np.float64)
            
            # load inputbasefilename (base.dat or input000.dat)
            if inputbasefilename != None:
                text_file=open(inputbasefilename,"r")
                lines = text_file.readlines()
                df2 = pd.DataFrame(np.array([['1a', '1b', '1c']]),columns=['Name', 'Value', 'Updated'])
                df3 = pd.DataFrame(columns=['Name', 'Value', 'Updated']) # currently, "Updated" feature not active
                for j in range(len(lines)):
                    str01 = lines[j].split('=')
                    if len(str01) == 2:
                        str01[0]=str01[0].rstrip()
                        str01[0]=str01[0].lstrip()
                        try:
                            df2['Name']=str01[0]
                            df2['Value']=float(str01[1])
                            df2['Updated']=False
                            df3=pd.concat([df3,df2],sort=False,ignore_index=True)
                        except:
                            pass

            # find index of preprocessor inputs
            try: 
                index1 = Sname_4cls.index("Average_CurrentDensity")
            except:
                index1 = -1
                try:
                    J_fix = df3.loc[df3["Name"]=="Average_CurrentDensity","Value"].iloc[0]/10.0  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index2 = Sname_4cls.index("Stack_Fuel_Utilization")
            except:
                index2 = -1
                try:
                    FU_fix = df3.loc[df3["Name"]=="Stack_Fuel_Utilization","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index3 = Sname_4cls.index("Stack_Oxidant_Utilization")
            except:
                index3 = -1
                try:
                    AU_fix = df3.loc[df3["Name"]=="Stack_Oxidant_Utilization","Value"].iloc[0]/10.0  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index4 = Sname_4cls.index("OxygenToCarbon_Ratio")
            except:
                index4 = -1
                try:
                    OCR_fix = df3.loc[df3["Name"]=="OxygenToCarbon_Ratio","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index5 = Sname_4cls.index("Internal_Reforming")
            except:
                index5 = -1
                try:
                    IR_fix = df3.loc[df3["Name"]=="Internal_Reforming","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index6 = Sname_4cls.index("Oxidant_Recirculation")
            except:
                index6 = -1
                try:
                    Arec_fix = df3.loc[df3["Name"]=="Oxidant_Recirculation","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index7= Sname_4cls.index("PreReform")
            except:
                index7 = -1
                try:
                    PreReform_fix = df3.loc[df3["Name"]=="PreReform","Value"].iloc[0]  
                except:
                    # sys.exit('Code terminated: "preprocessor" input not defined')
                    PreReform_fix=0.2 #[]
            try: 
                index8= Sname_4cls.index("cellsize")
            except:
                index8 = -1
                try:
                    cellsize_fix = df3.loc[df3["Name"]=="cellsize","Value"].iloc[0]  
                except:
                    # sys.exit('Code terminated: "preprocessor" input not defined')
                    cellsize_fix=550 #[cm2]
                
            if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                try: 
                    index9 = Sname_4cls.index("VGRRate")
                except:
                    index9 = -1
                    try:
                        VGR_fix = df3.loc[df3["Name"]=="VGRRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index11 = Sname_4cls.index("VGRH2OPassRate")
                except:
                    index11 = -1
                    try:
                        H2OCap_fix = 1-df3.loc[df3["Name"]=="VGRH2OPassRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index12 = Sname_4cls.index("VGRCO2CaptureRate")
                except:
                    index12 = -1
                    try:
                        CO2Cap_fix = df3.loc[df3["Name"]=="VGRCO2CaptureRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index13 = Sname_4cls.index("VGRH2PassRate")
                except:
                    index13 = -1
                    try:
                        H2Cap_fix = 1-df3.loc[df3["Name"]=="VGRH2PassRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index14 = Sname_4cls.index("VGRCOConvertRate")
                except:
                    index14 = -1
                    try:
                        WGS_fix = df3.loc[df3["Name"]=="VGRCOConvertRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                    
            # find value of preprocessor inputs
            for i in range(S_4cls_ROM_vali_cls_train_nrm.shape[0]):
                if index1 == -1:
                    J = J_fix
                else:
                    J = S_4cls_ROM_vali_cls_train[i,index1]/10.0 # mA/cm2
                if index2 == -1:
                    FU = FU_fix
                else:
                    FU = S_4cls_ROM_vali_cls_train[i,index2]
                if index3 == -1:
                    AU = AU_fix
                else:
                    AU = S_4cls_ROM_vali_cls_train[i,index3]
                if index4 == -1:
                    OCR = OCR_fix
                else:
                    OCR = S_4cls_ROM_vali_cls_train[i,index4]
                if index5 == -1:
                    IR = IR_fix
                else:
                    IR = S_4cls_ROM_vali_cls_train[i,index5]
                if index6 == -1:
                    Arec = Arec_fix
                else:
                    Arec = S_4cls_ROM_vali_cls_train[i,index6]
                if index7 == -1:
                    PreReform = PreReform_fix
                else:
                    PreReform = S_4cls_ROM_vali_cls_train[i,index7]
                if index8 == -1:
                    cellsize = cellsize_fix # cm2
                else:
                    cellsize = S_4cls_ROM_vali_cls_train[i,index8]
                    
                if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                    if index9 == -1:
                        VGR = VGR_fix
                    else:
                        VGR = S_4cls_ROM_vali_cls_train[i,index9]
                    if index11 == -1:
                        H2OCap = H2OCap_fix
                    else:
                        H2OCap = 1-S_4cls_ROM_vali_cls_train[i,index11]
                    if index12 == -1:
                        CO2Cap = CO2Cap_fix
                    else:
                        CO2Cap = S_4cls_ROM_vali_cls_train[i,index12]
                    if index13 == -1:
                        H2Cap = H2Cap_fix
                    else:
                        H2Cap = 1-S_4cls_ROM_vali_cls_train[i,index13]
                    if index14 == -1:
                        WGS = WGS_fix
                    else:
                        WGS = S_4cls_ROM_vali_cls_train[i,index14]

                if i%1000 == 0: print(i," cls_training")
                if preprocessor_name == None or preprocessor_name == 'NGFC_ccs':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'NGFC_nocc':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
                elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
                elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
                else:
                    sys.exit('Code terminated: the selected "preprocessor" cannot be found')
                succs_cls_training[i,0] = succ
            mean_succs = succs_cls_training.mean(axis=0)
            std_succs = succs_cls_training.std(axis=0)
            succs_cls_training_nrm = (succs_cls_training-mean_succs)/std_succs

            for i in range(S_4cls_vali_nrm.shape[0]):
                if index1 == -1:
                    J = J_fix
                else:
                    J = S_4cls_vali[i,index1]/10.0 # mA/cm2
                if index2 == -1:
                    FU = FU_fix
                else:
                    FU = S_4cls_vali[i,index2]
                if index3 == -1:
                    AU = AU_fix
                else:
                    AU = S_4cls_vali[i,index3]
                if index4 == -1:
                    OCR = OCR_fix
                else:
                    OCR = S_4cls_vali[i,index4]
                if index5 == -1:
                    IR = IR_fix
                else:
                    IR = S_4cls_vali[i,index5]
                if index6 == -1:
                    Arec = Arec_fix
                else:
                    Arec = S_4cls_vali[i,index6]
                if index7 == -1:
                    PreReform = PreReform_fix
                else:
                    PreReform = S_4cls_vali[i,index7]
                if index8 == -1:
                    cellsize = cellsize_fix # cm2
                else:
                    cellsize = S_4cls_vali[i,index8]
                    
                if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                    if index9 == -1:
                        VGR = VGR_fix
                    else:
                        VGR = S_4cls_vali[i,index9]
                    if index11 == -1:
                        H2OCap = H2OCap_fix
                    else:
                        H2OCap = 1-S_4cls_vali[i,index11]
                    if index12 == -1:
                        CO2Cap = CO2Cap_fix
                    else:
                        CO2Cap = S_4cls_vali[i,index12]
                    if index13 == -1:
                        H2Cap = H2Cap_fix
                    else:
                        H2Cap = 1-S_4cls_vali[i,index13]
                    if index14 == -1:
                        WGS = WGS_fix
                    else:
                        WGS = S_4cls_vali[i,index14]

                if i%1000 == 0: print(i," cls_testing")
                if preprocessor_name == None or preprocessor_name == 'NGFC_ccs':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'NGFC_nocc':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
                elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
                elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
                else:
                    sys.exit('Code terminated: the selected "preprocessor" cannot be found')
                succs_cls_testing[i,0] = succ
            mean_succs=succs_cls_testing.mean(axis=0)
            std_succs=succs_cls_testing.std(axis=0)
            succs_cls_testing_nrm=(succs_cls_testing-mean_succs)/std_succs
            
            ## 1.8 prepare classification data
            data_cls_training_y = Y_4cls_cls_train
            data_cls_training_x = np.concatenate((S_4cls_ROM_vali_cls_train_nrm,Y_4cls_ROM_vali_cls_train_nrm_pred),axis=1)

            data_cls_testing_x = np.concatenate((S_4cls_vali_nrm, Y_4cls_vali_nrm_pred),axis=1)
            data_cls_testing_y = Y_4cls_vali

            ## 1.9 perform classification with all inputs + all outputs + mbm decision
            data_cls_training_x_with_mbm = np.concatenate((data_cls_training_x,succs_cls_training_nrm),axis=1)
            data_cls_testing_x_with_mbm = np.concatenate((data_cls_testing_x,succs_cls_testing_nrm),axis=1)
            maxiteration = 50000
            acc_val_mbm,acc_test_mbm,test_prediction_mbm, cls_values = self.DNNCls(maxiteration, data_cls_training_x_with_mbm, data_cls_training_y, data_cls_testing_x_with_mbm, data_cls_testing_y, len(indS)+len(indY)+1)
            
            # ## 1.10 show classifier accuracy
            print('Classifier accuracy with vali-data: ', acc_val_mbm)
            print('Classifier accuracy with test-data: ', acc_test_mbm)
            # print(test_prediction_mbm)
            
            ## 1.11 write classifier as text file
            trainingoutput_file = self.outtrainingFile
            trainingoutput_file_cls = trainingoutput_file.replace(".dat", "")+'_cls.dat'
            trainingoutput_file_cls_ROM = trainingoutput_file.replace(".dat", "")+'_cls_ROM.dat'
            
            print('length of cls_values: ', len(cls_values))
            
            w1,w2,b1,b2 = cls_values
            with open(trainingoutput_file_cls, 'w') as f:
                f.write('w1\n')
                values_tmp = np.copy(w1)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w2\n')
                values_tmp = np.copy(w2)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('b1\n')
                values_tmp = np.copy(b1)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b2\n')
                values_tmp = np.copy(b2)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('end\n')
            
            print('length of cls_ROM_values: ', len(cls_ROM_values))
            
            w1,w2,w3,w4,w5,b1,b2,b3,b4,b5 = cls_ROM_values
            with open(trainingoutput_file_cls_ROM, 'w') as f:
                f.write('w1\n')
                values_tmp = np.copy(w1)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w2\n')
                values_tmp = np.copy(w2)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w3\n')
                values_tmp = np.copy(w3)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w4\n')
                values_tmp = np.copy(w4)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w5\n')
                values_tmp = np.copy(w5)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('b1\n')
                values_tmp = np.copy(b1)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b2\n')
                values_tmp = np.copy(b2)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b3\n')
                values_tmp = np.copy(b3)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b4\n')
                values_tmp = np.copy(b4)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b5\n')
                values_tmp = np.copy(b5)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('meanS\n')
                values_tmp = np.copy(meanS)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('meanY\n')
                values_tmp = np.copy(meanY)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('stdS\n')
                values_tmp = np.copy(stdS)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('stdY\n')
                values_tmp = np.copy(stdY)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('end\n')

        ################## Step 2: train the ROM ##################
        ## 2.1 determine indS, indY and determine if enabling ROM training
        indS = list(range(1, S_col+1))
        indY = []
        Yname_4indY = ["Outlet_Fuel_Flowrate", "Outlet_Fuel_H2", 
                       "Outlet_Fuel_H2O", "Outlet_Fuel_CO", 
                       "Outlet_Fuel_CO2", "Outlet_Fuel_CH4", 
                       "Outlet_Fuel_N2", "Outlet_Air_Flowrate", 
                       "Outlet_Air_O2", "Outlet_Air_N2", 
                       "Outlet_Air_H2O", "Outlet_Air_CO2", 
                       "Outlet_Air_Ar", "FSI_Flowrate", "FSI_H2_MF", 
                       "FSI_H2O_MF", "FSI_CO_MF", "FSI_CO2_MF", 
                       "FSI_CH4_MF", "FSI_N2_MF"]
        
        ROM_enabled = False
        for i in range(Y_col):
            Yname_tmp = Yname[i]
            if Yname_tmp in Yname_4indY:
                indY.append(i+1)
        if len(indY) == len(Yname_4indY):
            ROM_enabled = True # if any element in Yname_4indY is missing, disable ROM training
        else:
            print('certain disired variable is missing')
        
        indS_index = [i-1 for i in indS]
        indY_index = [i-1 for i in indY]
        
        ## 2.2- call preprocessor, prepare training data, train the ROM model, etc.
        if ROM_enabled == True:
            ## 2.2 prepare training data (simulation results)
            if cls_enabled == True: # filter non-converged
                SYvalue_cov = SYvalue[SYvalue[:, S_col] == 1, :]
            else:
                SYvalue_cov = SYvalue
            if filter_enabled == True: # filter noise
                SY_row_rm = []
                for j in indY:
                    tmp_data = SYvalue_cov[:, S_col+j-1]
                    while(True):
                        z = np.abs(stats.zscore(tmp_data, axis = 0))
                        result = np.where(z > z_thres)
                        index = list(result[0])
                        # line removal list
                        if len(index) == 0: break
                        SY_row_rm += index
                        SY_row_rm = list(dict.fromkeys(SY_row_rm))
                        # replace outliers with mean
                        tmp_data[SY_row_rm] = np.mean(tmp_data)
                # remove rows and columns accroding to SY_row_rm and SY_col_rm
                SYvalue_new = np.delete(SYvalue_cov, SY_row_rm, axis = 0)
                print('Noise filter: trim ' + str(len(SY_row_rm)) + ' rows from a total of ' + str(len(SYvalue_cov)) + ' rows')
            else:
                SYvalue_new = SYvalue_cov

            [S_row, Y_row, S_col, Y_col] = [len(SYvalue_new), len(SYvalue_new), int(infovalue[0,0]), int(infovalue[0,1])]
            Svalue_new = copy.deepcopy(SYvalue_new[:, :S_col])
            Yvalue_new = copy.deepcopy(SYvalue_new[:, S_col:])

            # compute istep, numcrossvali, rndnumberlist
            if frac4ROM >= 0:
                numtraining = int(S_row*frac4ROM/100.0)
                numcrossvali = S_row-numtraining
                if numtraining < (2**len(indS)): 
                    print('warning: "frac4ROM" is too low')
                if numcrossvali > 0:
                    istep = int((S_row)/numcrossvali)
                    rndnumberlist =[]
                    restlist = list(range(S_row))
                    for i in range(1, numcrossvali+1):
                        rndnumberlist.append(i*istep-1)
                    restlist = [i for i in restlist if i not in rndnumberlist]
                else:
                    sys.exit('Code terminated: the fraction of training dataset cannot be 100%')
            else:
                numtraining = S_row-1000
                numcrossvali = S_row-numtraining
                rndnumberlist = list(range(numtraining, S_row))
                restlist = list(range(numtraining))
        
            # split to training and validation data
            Sname_4ROM = [ Sname[i] for i in indS_index]
            Yname_4ROM = [ Yname[i] for i in indY_index]

            temp = Svalue_new[restlist, :]
            S_4ROM_train = temp[:, indS_index]
            temp = Svalue_new[rndnumberlist, :]
            S_4ROM_vali = temp[:, indS_index]
            temp = Yvalue_new[restlist, :]
            Y_4ROM_train = temp[:, indY_index]
            temp = Yvalue_new[rndnumberlist, :]
            Y_4ROM_vali = temp[:, indY_index]

            ## 2.3 prepare training data ("preprocessor" results)
            preprocessor_result_train = np.zeros((len(restlist),len(indY)),dtype=np.float64)
            preprocessor_result_vali = np.zeros((len(rndnumberlist),len(indY)),dtype=np.float64)

            # load inputbasefilename (base.dat or input000.dat)
            if inputbasefilename != None:
                text_file=open(inputbasefilename,"r")
                lines = text_file.readlines()
                df2 = pd.DataFrame(np.array([['1a', '1b', '1c']]),columns=['Name', 'Value', 'Updated'])
                df3 = pd.DataFrame(columns=['Name', 'Value', 'Updated']) # currently, "Updated" feature not active
                for j in range(len(lines)):
                    str01 = lines[j].split('=')
                    if len(str01) == 2:
                        str01[0]=str01[0].rstrip()
                        str01[0]=str01[0].lstrip()
                        try:
                            df2['Name']=str01[0]
                            df2['Value']=float(str01[1])
                            df2['Updated']=False
                            df3=pd.concat([df3,df2],sort=False,ignore_index=True)
                        except:
                            pass
            
            # find index of preprocessor inputs
            try: 
                index1 = Sname_4ROM.index("Average_CurrentDensity")
            except:
                index1 = -1
                try:
                    J_fix = df3.loc[df3["Name"]=="Average_CurrentDensity","Value"].iloc[0]/10.0  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index2 = Sname_4ROM.index("Stack_Fuel_Utilization")
            except:
                index2 = -1
                try:
                    FU_fix = df3.loc[df3["Name"]=="Stack_Fuel_Utilization","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index3 = Sname_4ROM.index("Stack_Oxidant_Utilization")
            except:
                index3 = -1
                try:
                    AU_fix = df3.loc[df3["Name"]=="Stack_Oxidant_Utilization","Value"].iloc[0]/10.0  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index4 = Sname_4ROM.index("OxygenToCarbon_Ratio")
            except:
                index4 = -1
                try:
                    OCR_fix = df3.loc[df3["Name"]=="OxygenToCarbon_Ratio","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index5 = Sname_4ROM.index("Internal_Reforming")
            except:
                index5 = -1
                try:
                    IR_fix = df3.loc[df3["Name"]=="Internal_Reforming","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index6 = Sname_4ROM.index("Oxidant_Recirculation")
            except:
                index6 = -1
                try:
                    Arec_fix = df3.loc[df3["Name"]=="Oxidant_Recirculation","Value"].iloc[0]  
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined')
            try: 
                index7= Sname_4ROM.index("PreReform")
            except:
                index7 = -1
                try:
                    PreReform_fix = df3.loc[df3["Name"]=="PreReform","Value"].iloc[0]  
                except:
                    # sys.exit('Code terminated: "preprocessor" input not defined')
                    PreReform_fix=0.2 #[]
            try: 
                index8= Sname_4ROM.index("cellsize")
            except:
                index8 = -1
                try:
                    cellsize_fix = df3.loc[df3["Name"]=="cellsize","Value"].iloc[0]  
                except:
                    # sys.exit('Code terminated: "preprocessor" input not defined')
                    cellsize_fix=550 #[cm2]
                
            if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                try: 
                    index9 = Sname_4ROM.index("VGRRate")
                except:
                    index9 = -1
                    try:
                        VGR_fix = df3.loc[df3["Name"]=="VGRRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index11 = Sname_4ROM.index("VGRH2OPassRate")
                except:
                    index11 = -1
                    try:
                        H2OCap_fix = 1-df3.loc[df3["Name"]=="VGRH2OPassRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index12 = Sname_4ROM.index("VGRCO2CaptureRate")
                except:
                    index12 = -1
                    try:
                        CO2Cap_fix = df3.loc[df3["Name"]=="VGRCO2CaptureRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index13 = Sname_4ROM.index("VGRH2PassRate")
                except:
                    index13 = -1
                    try:
                        H2Cap_fix = 1-df3.loc[df3["Name"]=="VGRH2PassRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')
                try: 
                    index14 = Sname_4ROM.index("VGRCOConvertRate")
                except:
                    index14 = -1
                    try:
                        WGS_fix = df3.loc[df3["Name"]=="VGRCOConvertRate","Value"].iloc[0]  
                    except:
                        sys.exit('Code terminated: "preprocessor" input not defined')

            # call preprocessor for trianing data
            for i in range(S_4ROM_train.shape[0]):
                if index1 == -1:
                    J = J_fix
                else:
                    J = S_4ROM_train[i,index1]/10.0 # mA/cm2
                if index2 == -1:
                    FU = FU_fix
                else:
                    FU = S_4ROM_train[i,index2]
                if index3 == -1:
                    AU = AU_fix
                else:
                    AU = S_4ROM_train[i,index3]
                if index4 == -1:
                    OCR = OCR_fix
                else:
                    OCR = S_4ROM_train[i,index4]
                if index5 == -1:
                    IR = IR_fix
                else:
                    IR = S_4ROM_train[i,index5]
                if index6 == -1:
                    Arec = Arec_fix
                else:
                    Arec = S_4ROM_train[i,index6]
                if index7 == -1:
                    PreReform = PreReform_fix
                else:
                    PreReform = S_4ROM_train[i,index7]
                if index8 == -1:
                    cellsize = cellsize_fix # cm2
                else:
                    cellsize = S_4ROM_train[i,index8]
                    
                if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                    if index9 == -1:
                        VGR = VGR_fix
                    else:
                        VGR = S_4ROM_train[i,index9]
                    if index11 == -1:
                        H2OCap = H2OCap_fix
                    else:
                        H2OCap = 1-S_4ROM_train[i,index11]
                    if index12 == -1:
                        CO2Cap = CO2Cap_fix
                    else:
                        CO2Cap = S_4ROM_train[i,index12]
                    if index13 == -1:
                        H2Cap = H2Cap_fix
                    else:
                        H2Cap = 1-S_4ROM_train[i,index13]
                    if index14 == -1:
                        WGS = WGS_fix
                    else:
                        WGS = S_4ROM_train[i,index14]
                    
                if preprocessor_name == None or preprocessor_name == 'NGFC_ccs':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'NGFC_nocc':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
                elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
                elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
                else:
                    sys.exit('Code terminated: the selected "preprocessor" cannot be found')

                preprocessor_result_train[i,0] = np.sum(FuelOut)
                preprocessor_result_train[i,1] = FuelOut[7]/np.sum(FuelOut)
                preprocessor_result_train[i,2] = FuelOut[0]/np.sum(FuelOut)
                preprocessor_result_train[i,3] = FuelOut[6]/np.sum(FuelOut)
                preprocessor_result_train[i,4] = FuelOut[2]/np.sum(FuelOut)
                preprocessor_result_train[i,5] = FuelOut[5]/np.sum(FuelOut)
                preprocessor_result_train[i,6] = FuelOut[4]/np.sum(FuelOut)
                preprocessor_result_train[i,7] = np.sum(AirOut)
                preprocessor_result_train[i,8] = AirOut[3]/np.sum(AirOut)
                preprocessor_result_train[i,9] = AirOut[4]/np.sum(AirOut)
                preprocessor_result_train[i,10] = AirOut[0]/np.sum(AirOut)
                preprocessor_result_train[i,11] = AirOut[2]/np.sum(AirOut)
                preprocessor_result_train[i,12] = AirOut[1]/np.sum(AirOut)
                preprocessor_result_train[i,13] = np.sum(FuelIn)
                preprocessor_result_train[i,14] = FuelIn[7]/np.sum(FuelIn)
                preprocessor_result_train[i,15] = FuelIn[0]/np.sum(FuelIn)
                preprocessor_result_train[i,16] = FuelIn[6]/np.sum(FuelIn)
                preprocessor_result_train[i,17] = FuelIn[2]/np.sum(FuelIn)
                preprocessor_result_train[i,18] = FuelIn[5]/np.sum(FuelIn)
                preprocessor_result_train[i,19] = FuelIn[4]/np.sum(FuelIn)
                
                # # plot preprocessor results vs simulation results
                # tempy1 = Y_4ROM_train[i,:].flatten()
                # tempy2 = preprocessor_result_train[i,:].flatten()
                # tempx = list(range(1, len(indY)+1))

                # fig, ax = plt.subplots(figsize=(8,6))
                # ax.plot(tempx, tempy1, 'ro-', linewidth = 2, 
                #         markersize = 12, label = 'Simulation')
                # ax.plot(tempx, tempy2, 'bd--', linewidth = 2, 
                #         markersize = 12, label = 'Preprocessor')
                # plt.legend(loc='upper left')
                # ax.set(title = 'Results comparison of case '+str(i))
                # FigureName = self.work_path + '/Case ' + str(i) +'.png'
                # plt.savefig(FigureName)
                # plt.show()
                
            # call preprocessor for validation data
            for i in range(S_4ROM_vali.shape[0]):
                if index1 == -1:
                    J = J_fix
                else:
                    J = S_4ROM_vali[i,index1]/10.0 # mA/cm2
                if index2 == -1:
                    FU = FU_fix
                else:
                    FU = S_4ROM_vali[i,index2]
                if index3 == -1:
                    AU = AU_fix
                else:
                    AU = S_4ROM_vali[i,index3]
                if index4 == -1:
                    OCR = OCR_fix
                else:
                    OCR = S_4ROM_vali[i,index4]
                if index5 == -1:
                    IR = IR_fix
                else:
                    IR = S_4ROM_vali[i,index5]
                if index6 == -1:
                    Arec = Arec_fix
                else:
                    Arec = S_4ROM_vali[i,index6]
                if index7 == -1:
                    PreReform = PreReform_fix
                else:
                    PreReform = S_4ROM_vali[i,index7]
                if index8 == -1:
                    cellsize = cellsize_fix # cm2
                else:
                    cellsize = S_4ROM_vali[i,index8]
                    
                if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                    if index9 == -1:
                        VGR = VGR_fix
                    else:
                        VGR = S_4ROM_vali[i,index9]
                    if index11 == -1:
                        H2OCap = H2OCap_fix
                    else:
                        H2OCap = 1-S_4ROM_vali[i,index11]
                    if index12 == -1:
                        CO2Cap = CO2Cap_fix
                    else:
                        CO2Cap = S_4ROM_vali[i,index12]
                    if index13 == -1:
                        H2Cap = H2Cap_fix
                    else:
                        H2Cap = 1-S_4ROM_vali[i,index13]
                    if index14 == -1:
                        WGS = WGS_fix
                    else:
                        WGS = S_4ROM_vali[i,index14]

                if preprocessor_name == None or preprocessor_name == 'NGFC_ccs':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'NGFC_nocc':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
                elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
                elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
                else:
                    sys.exit('Code terminated: the selected "preprocessor" cannot be found')

                preprocessor_result_vali[i,0] = np.sum(FuelOut)
                preprocessor_result_vali[i,1] = FuelOut[7]/np.sum(FuelOut)
                preprocessor_result_vali[i,2] = FuelOut[0]/np.sum(FuelOut)
                preprocessor_result_vali[i,3] = FuelOut[6]/np.sum(FuelOut)
                preprocessor_result_vali[i,4] = FuelOut[2]/np.sum(FuelOut)
                preprocessor_result_vali[i,5] = FuelOut[5]/np.sum(FuelOut)
                preprocessor_result_vali[i,6] = FuelOut[4]/np.sum(FuelOut)
                preprocessor_result_vali[i,7] = np.sum(AirOut)
                preprocessor_result_vali[i,8] = AirOut[3]/np.sum(AirOut)
                preprocessor_result_vali[i,9] = AirOut[4]/np.sum(AirOut)
                preprocessor_result_vali[i,10] = AirOut[0]/np.sum(AirOut)
                preprocessor_result_vali[i,11] = AirOut[2]/np.sum(AirOut)
                preprocessor_result_vali[i,12] = AirOut[1]/np.sum(AirOut)
                preprocessor_result_vali[i,13] = np.sum(FuelIn)
                preprocessor_result_vali[i,14] = FuelIn[7]/np.sum(FuelIn)
                preprocessor_result_vali[i,15] = FuelIn[0]/np.sum(FuelIn)
                preprocessor_result_vali[i,16] = FuelIn[6]/np.sum(FuelIn)
                preprocessor_result_vali[i,17] = FuelIn[2]/np.sum(FuelIn)
                preprocessor_result_vali[i,18] = FuelIn[5]/np.sum(FuelIn)
                preprocessor_result_vali[i,19] = FuelIn[4]/np.sum(FuelIn)

            ## 2.4 prepare training data (differences betweeen simulation and preprocessor results)
            err_4ROM_train = preprocessor_result_train - Y_4ROM_train
            err_4ROM_vali = preprocessor_result_vali - Y_4ROM_vali

            meanS=S_4ROM_train.mean(axis=0)
            stdS=S_4ROM_train.std(axis=0)
            meanY=Y_4ROM_train.mean(axis=0)
            stdY=Y_4ROM_train.std(axis=0)
            meanerr=err_4ROM_train.mean(axis=0)
            stderr=err_4ROM_train.std(axis=0)

            S_4ROM_train_nrm=(S_4ROM_train-meanS)/stdS
            S_4ROM_vali_nrm=(S_4ROM_vali-meanS)/stdS
            Y_4ROM_train_nrm=(Y_4ROM_train-meanY)/stdY
            err_4ROM_train_nrm=(err_4ROM_train-meanerr)/stderr

            ## 2.4 write to info.dat, intraining.dat, info.dat and inCrossVali.dat 
            with open(self.infoFile, 'w') as f:
                f.write('input_col\toutput_col\n')
                f.write(str(len(indS))+'\t'+str(len(indY))+'\n')
            f1 = open(self.intrainingFile, 'w')
            f3 = open(self.incrossvaliFile, 'w')
            for i in range(len(indS)):
                f1.write(Sname_4ROM[i] + '\t')
                f3.write(Sname_4ROM[i] + '\t')
            for i in range(len(indY)):
                f1.write(Yname_4ROM[i] + '\t')
                f3.write(Yname_4ROM[i] + '\t')
            f1.write('\n')
            f3.write('\n')
            for i in range(len(restlist)):
                for j in range(len(indS)):
                    f1.write('{:11.4E}\t'.format(S_4ROM_train[i, j]))
                for j in range(len(indY)):
                    f1.write('{:11.4E}\t'.format(Y_4ROM_train[i, j]))
                f1.write('\n')
            for i in range(len(rndnumberlist)):
                for j in range(len(indS)):
                    f3.write('{:11.4E}\t'.format(S_4ROM_vali[i, j]))
                for j in range(len(indY)):
                    f3.write('{:11.4E}\t'.format(Y_4ROM_vali[i, j]))
                f3.write('\n')
            f1.close()
            f3.close()

            # # write simulation results and "preprocessor" results
            # traininginput_file = self.intrainingFile
            # traininginput_file_simu = traininginput_file.replace(".dat", "")+'_simu.dat'
            # traininginput_file_wrap = traininginput_file.replace(".dat", "")+'_wrap.dat'

            # f1 = open(traininginput_file_simu, 'w')
            # f3 = open(traininginput_file_wrap, 'w')
            # for i in range(len(indS)):
            #     f1.write(Sname_4ROM[i] + '\t')
            #     f3.write(Sname_4ROM[i] + '\t')
            # for i in range(len(indY)):
            #     f1.write(Yname_4ROM[i] + '\t')
            #     f3.write(Yname_4ROM[i] + '\t')
            # f1.write('\n')
            # f3.write('\n')
            # for i in range(len(restlist)):
            #     for j in range(len(indS)):
            #         f1.write('{:11.4E}\t'.format(S_4ROM_train[i, j]))
            #         f3.write('{:11.4E}\t'.format(S_4ROM_train[i, j]))
            #     for j in range(len(indY)):
            #         f1.write('{:11.4E}\t'.format(Y_4ROM_train[i, j]))
            #         f3.write('{:11.4E}\t'.format(preprocessor_result_train[i, j]))
            #     f1.write('\n')
            #     f3.write('\n')
            # f1.close()
            # f3.close()

            ## 2.5 perform training and prediction
            maxiteration = 50000
            DNNsize = [32, 200, 200, 256]
            err_4ROM_vali_nrm_pre, ROM_values = self.DNNROM(maxiteration, S_4ROM_train_nrm, err_4ROM_train_nrm, S_4ROM_vali_nrm, len(indS), len(indY), DNNsize)

            ## 2.6 save built ROM model
            print('length of ROM_values: ', len(ROM_values))
            
            w1,w2,w3,w4,w5,b1,b2,b3,b4,b5 = ROM_values
            with open(self.outtrainingFile, 'w') as f:
                f.write('w1\n')
                values_tmp = np.copy(w1)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w2\n')
                values_tmp = np.copy(w2)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w3\n')
                values_tmp = np.copy(w3)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w4\n')
                values_tmp = np.copy(w4)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('w5\n')
                values_tmp = np.copy(w5)
                [row, col] = values_tmp.shape
                for i in range(row):
                    for j in range(col-1):
                        f.write(str(values_tmp[i, j]) + ' ')
                    f.write(str(values_tmp[i, col-1]) + '\n')
                f.write('\n')
                f.write('b1\n')
                values_tmp = np.copy(b1)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b2\n')
                values_tmp = np.copy(b2)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b3\n')
                values_tmp = np.copy(b3)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b4\n')
                values_tmp = np.copy(b4)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('b5\n')
                values_tmp = np.copy(b5)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('meanS\n')
                values_tmp = np.copy(meanS)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('meanY\n')
                values_tmp = np.copy(meanY)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('stdS\n')
                values_tmp = np.copy(stdS)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('stdY\n')
                values_tmp = np.copy(stdY)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('meanerr\n')
                values_tmp = np.copy(meanerr)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('stderr\n')
                values_tmp = np.copy(stderr)
                row = len(values_tmp)
                for i in range(row):
                    f.write(str(values_tmp[i]) + '\n')
                f.write('\n')
                f.write('end\n')

            ## 2.7 write to ourCrossVali.dat
            err_4ROM_vali_pre = err_4ROM_vali_nrm_pre*stderr+meanerr
            Y_4ROM_vali_pre = preprocessor_result_vali-err_4ROM_vali_pre
            f0 = open(self.outcrossvaliFile, 'w')
            for i in range(len(indY)):
                name = Yname_4ROM[i]
                f0.write(name + '\t')
            f0.write('\n')
            for i in range(len(rndnumberlist)):
                for j in range(len(indY)):
                    f0.write('{:11.4E}\t'.format(Y_4ROM_vali_pre[i,j]-Y_4ROM_vali[i, j]))
                f0.write('\n')
            f0.close()

            ## 2.8 update global variables
            [self.S_row, self.Y_row, self.S_col, self.Y_col] = [len(restlist), len(restlist), len(indS), len(indY)]
            self.S_norm = S_4ROM_train_nrm
            self.Y_norm = Y_4ROM_train_nrm
            self.S = S_4ROM_train
            self.Y = Y_4ROM_train
            [self.stdS, self.stdY, self.meanS, self.meanY] = [stdS, stdY, meanS, meanY]
            self.Sname = Sname_4ROM
            self.Yname = Yname_4ROM
        
        ################## Step 3: write accuracy ##################
        int_95 = self.percent2intervl(95) # 95% confidence interval
        trainingoutput_file = self.outtrainingFile
        trainingoutput_accuracy = trainingoutput_file.replace(".dat", "")+'_acc.dat'
        with open(trainingoutput_accuracy, 'w') as f:
            if cls_enabled == True:
                f.write('Classifier Accuracy: \n')
                f.write(str(acc_test_mbm) + '\n')
            if ROM_enabled == True:
                f.write('ROM Accuracy (95% confidence interval): \n')
                for i in range(len(Yname_4ROM)):
                    f.write(Yname_4ROM[i])
                    f.write('\t' + str(int_95[i]) + '\n')
                
        print('End of code\n')
    
    def Generate_inprediction(self, numsample = None, listmin = None, listmax = None):
        '''
        The function generates prediction input if it doesn't exist by Latin Hypercube Sampling
        '''
        print('############################################################\
                \nGenerate prediction input\
                \n############################################################')

        # find input variable list Sname
        SYname, SYvalue = self.file_read(self.intrainingFile)
        infoname, infovalue = self.file_read(self.infoFile)
        [S_col, Y_col] = [int(infovalue[0,0]), int(infovalue[0,1])]
        Sname = copy.deepcopy(SYname[:S_col])
        
        # check if exists
        filename  = self.inpredictionFile
        Create_handle = True
        if os.path.exists(filename):
            query = query_yes_no('Prediction input file already exists on the local machine, do you want to overwrite it?')
            Create_handle = query

        if Create_handle == True:
            numvar = len(Sname)
            listvar = Sname
            if len(listmin) != numvar or len(listmax) != numvar:
                sys.exit('Code terminated: the lengths of variables/minimums/maximums not match')

            # LHS sampling
            xlimits = np.transpose(np.vstack((listmin, listmax)))
            sampling = LHS(xlimits = xlimits)
            LHSvalue = sampling(numsample)

            # write prediction input
            with open(filename, 'w') as f:
                for name in Sname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(numsample):
                    for j in range(numvar):
                        f.write('{:11.4E}\t'.format(LHSvalue[i, j]))
                    f.write('\n')

            print("Created prediciton input file")
            print('End of code\n')

    def prediction(self, preprocessor_name = None, igfc = None):
        '''
        This function predicts the outputs and MSEs 
        based on the trained model 
        '''
        print('############################################################\
              \nPredict Based on the trained model\
              \n############################################################')
        
        # # Step 0: check if outprediction.dat existing
        # if os.path.exists(self.outpredictionFile):
        #     query = query_yes_no('prediction results already exist on the local machine, do you want to overwrite it?')
        #     if query == False: return
    
        ############# Step 1: Load the training data S, Y and prediction data X #############
        print('Step 1: Load the training data S, Y and prediction input data X')
        SYname, SYvalue = self.file_read(self.intrainingFile)
        Xname, Xvalue = self.file_read(self.inpredictionFile)
        infoname, infovalue = self.file_read(self.infoFile)
        [S_row, Y_row, S_col, Y_col] = [len(SYvalue), len(SYvalue), int(infovalue[0,0]), int(infovalue[0,1])]
        
        S = copy.deepcopy(SYvalue[:, :S_col])
        Y = copy.deepcopy(SYvalue[:, S_col:])
        X = copy.deepcopy(Xvalue)
        Sname = copy.deepcopy(SYname[:S_col])
        Yname = copy.deepcopy(SYname[S_col:])
        [X_row, X_col] = X.shape
        if X_col != S_col:
            sys.exit('Code terminated: # of prediction input variables \
            does not match # of given input variables')
                
        ############# Step 2: Load the trained models for classifier #############
        trainingoutput_file = self.outtrainingFile
        if not os.path.exists(trainingoutput_file):
            sys.exit('Code terminated: trained model missing')
            
        trainingoutput_file_cls = trainingoutput_file.replace(".dat", "")+'_cls.dat'
        trainingoutput_file_cls_ROM = trainingoutput_file.replace(".dat", "")+'_cls_ROM.dat'
        if os.path.exists(trainingoutput_file_cls) or os.path.exists(trainingoutput_file_cls_ROM):
            cls_enabled = True
        else:
            cls_enabled = False
            print('trained model has no classifier, continue')
        
        if cls_enabled == True:
            with open(trainingoutput_file_cls) as f:
                lines = f.readlines()
                i = 0
                for line in lines:
                    linestr = line.strip().split(' ')
                    if linestr[0] == 'w1': 
                        w1_s_cls = i+1
                    if linestr[0] == 'w2': 
                        w2_s_cls = i+1
                        w1_e_cls = i-2
                    if linestr[0] == 'b1': 
                        b1_s_cls = i+1
                        w2_e_cls = i-2
                    if linestr[0] == 'b2': 
                        b2_s_cls = i+1
                        b1_e_cls = i-2
                    if linestr[0] == 'end': 
                        b2_e_cls = i-2
                    i += 1

                i = 0
                for line in lines:
                    linestr = line.strip().split(' ')
                    if i == w1_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w1_cls = np.array(linenum)
                        w1_row_cls = w1_e_cls-w1_s_cls+1
                        w1_col_cls = len(w1_cls)
                    if i > w1_s_cls and i <= w1_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w1_cls = np.append(w1_cls, linenum)
                    if i == w2_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w2_cls = np.array(linenum)
                        w2_row_cls = w2_e_cls-w2_s_cls+1
                        w2_col_cls = len(w2_cls)
                    if i > w2_s_cls and i <= w2_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        w2_cls = np.append(w2_cls, linenum)
                    if i == b1_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b1_cls = np.array(linenum)
                    if i > b1_s_cls and i <= b1_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b1_cls = np.append(b1_cls, linenum)
                    if i == b2_s_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b2_cls = np.array(linenum)
                    if i > b2_s_cls and i <= b2_e_cls:
                        linenum = [float(lineele) for lineele in linestr]
                        b2_cls = np.append(b2_cls, linenum)
                    i += 1

                w1_cls = np.reshape(w1_cls, (w1_row_cls, w1_col_cls))
                w2_cls = np.reshape(w2_cls, (w2_row_cls, w2_col_cls))
            
            with open(trainingoutput_file_cls_ROM) as f:
                lines = f.readlines()
                i = 0
                for line in lines:
                    linestr = line.strip().split(' ')
                    if linestr[0] == 'w1': 
                        w1_s = i+1
                    if linestr[0] == 'w2': 
                        w2_s = i+1
                        w1_e = i-2
                    if linestr[0] == 'w3': 
                        w3_s = i+1
                        w2_e = i-2
                    if linestr[0] == 'w4': 
                        w4_s = i+1
                        w3_e = i-2
                    if linestr[0] == 'w5': 
                        w5_s = i+1
                        w4_e = i-2
                    if linestr[0] == 'b1': 
                        b1_s = i+1
                        w5_e = i-2
                    if linestr[0] == 'b2': 
                        b2_s = i+1
                        b1_e = i-2
                    if linestr[0] == 'b3': 
                        b3_s = i+1
                        b2_e = i-2
                    if linestr[0] == 'b4': 
                        b4_s = i+1
                        b3_e = i-2
                    if linestr[0] == 'b5': 
                        b5_s = i+1
                        b4_e = i-2
                    if linestr[0] == 'meanS': 
                        meanS_s = i+1
                        b5_e = i-2
                    if linestr[0] == 'meanY': 
                        meanY_s = i+1
                        meanS_e = i-2
                    if linestr[0] == 'stdS': 
                        stdS_s = i+1
                        meanY_e = i-2
                    if linestr[0] == 'stdY': 
                        stdY_s = i+1
                        stdS_e = i-2
                    if linestr[0] == 'end': 
                        stdY_e = i-2
                    i += 1

                i = 0
                for line in lines:
                    linestr = line.strip().split(' ')
                    if i == w1_s:
                        linenum = [float(lineele) for lineele in linestr]
                        w1 = np.array(linenum)
                        w1_row = w1_e-w1_s+1
                        w1_col = len(w1)
                    if i > w1_s and i <= w1_e:
                        linenum = [float(lineele) for lineele in linestr]
                        w1 = np.append(w1, linenum)
                    if i == w2_s:
                        linenum = [float(lineele) for lineele in linestr]
                        w2 = np.array(linenum)
                        w2_row = w2_e-w2_s+1
                        w2_col = len(w2)
                    if i > w2_s and i <= w2_e:
                        linenum = [float(lineele) for lineele in linestr]
                        w2 = np.append(w2, linenum)
                    if i == w3_s:
                        linenum = [float(lineele) for lineele in linestr]
                        w3 = np.array(linenum)
                        w3_row = w3_e-w3_s+1
                        w3_col = len(w3)
                    if i > w3_s and i <= w3_e:
                        linenum = [float(lineele) for lineele in linestr]
                        w3 = np.append(w3, linenum)
                    if i == w4_s:
                        linenum = [float(lineele) for lineele in linestr]
                        w4 = np.array(linenum)
                        w4_row = w4_e-w4_s+1
                        w4_col = len(w4)
                    if i > w4_s and i <= w4_e:
                        linenum = [float(lineele) for lineele in linestr]
                        w4 = np.append(w4, linenum)
                    if i == w5_s:
                        linenum = [float(lineele) for lineele in linestr]
                        w5 = np.array(linenum)
                        w5_row = w5_e-w5_s+1
                        w5_col = len(w5)
                    if i > w5_s and i <= w5_e:
                        linenum = [float(lineele) for lineele in linestr]
                        w5 = np.append(w5, linenum)
                    if i == b1_s:
                        linenum = [float(lineele) for lineele in linestr]
                        b1 = np.array(linenum)
                    if i > b1_s and i <= b1_e:
                        linenum = [float(lineele) for lineele in linestr]
                        b1 = np.append(b1, linenum)
                    if i == b2_s:
                        linenum = [float(lineele) for lineele in linestr]
                        b2 = np.array(linenum)
                    if i > b2_s and i <= b2_e:
                        linenum = [float(lineele) for lineele in linestr]
                        b2 = np.append(b2, linenum)
                    if i == b3_s:
                        linenum = [float(lineele) for lineele in linestr]
                        b3 = np.array(linenum)
                    if i > b3_s and i <= b3_e:
                        linenum = [float(lineele) for lineele in linestr]
                        b3 = np.append(b3, linenum)
                    if i == b4_s:
                        linenum = [float(lineele) for lineele in linestr]
                        b4 = np.array(linenum)
                    if i > b4_s and i <= b4_e:
                        linenum = [float(lineele) for lineele in linestr]
                        b4 = np.append(b4, linenum)
                    if i == b5_s:
                        linenum = [float(lineele) for lineele in linestr]
                        b5 = np.array(linenum)
                    if i > b5_s and i <= b5_e:
                        linenum = [float(lineele) for lineele in linestr]
                        b5 = np.append(b5, linenum)
                    if i == meanS_s:
                        linenum = [float(lineele) for lineele in linestr]
                        meanS = np.array(linenum)
                    if i > meanS_s and i <= meanS_e:
                        linenum = [float(lineele) for lineele in linestr]
                        meanS = np.append(meanS, linenum)
                    if i == meanY_s:
                        linenum = [float(lineele) for lineele in linestr]
                        meanY = np.array(linenum)
                    if i > meanY_s and i <= meanY_e:
                        linenum = [float(lineele) for lineele in linestr]
                        meanY = np.append(meanY, linenum)
                    if i == stdS_s:
                        linenum = [float(lineele) for lineele in linestr]
                        stdS = np.array(linenum)
                    if i > stdS_s and i <= stdS_e:
                        linenum = [float(lineele) for lineele in linestr]
                        stdS = np.append(stdS, linenum)
                    if i == stdY_s:
                        linenum = [float(lineele) for lineele in linestr]
                        stdY = np.array(linenum)
                    if i > stdY_s and i <= stdY_e:
                        linenum = [float(lineele) for lineele in linestr]
                        stdY = np.append(stdY, linenum)
                    i += 1
                
            del w1_s, w1_e, w2_s, w2_e, w3_s, w3_e, w4_s, w4_e, w5_s, w5_e, 
            b1_s, b1_e, b2_s, b2_e, b3_s, b3_e, b4_s, b4_e, b5_s, b5_e, 
            meanS_s, meanS_e, meanY_s, meanY_e, stdS_s, stdS_e, stdY_s, stdY_e
                
            w1 = np.reshape(w1, (w1_row, w1_col))
            w2 = np.reshape(w2, (w2_row, w2_col))
            w3 = np.reshape(w3, (w3_row, w3_col))
            w4 = np.reshape(w4, (w4_row, w4_col))
            w5 = np.reshape(w5, (w5_row, w5_col))
            
            ############# Step 3: ROM prediction for classifier #############
            X_nrm = (X - np.tile(meanS, [X_row, 1]))/np.tile(stdS, [X_row, 1])
            
            for j in range(X_row):
                inputX = X_nrm[j,:]
                m1 = np.matmul(inputX,w1)
                m1b = np.add(m1,b1)
                m1ba = np.zeros(len(m1b))
                for i in range(len(m1b)):
                    m1ba[i] = 1.0/(1+math.exp(-m1b[i]))
                m2 = np.matmul(m1ba,w2)
                m2b = np.add(m2,b2)
                m2ba = np.zeros(len(m2b))
                for i in range(len(m2b)):
                    m2ba[i] = 1.0/(1+math.exp(-m2b[i]))
                m3 = np.matmul(m2ba,w3)
                m3b = np.add(m3,b3)
                m3ba = np.zeros(len(m3b))
                for i in range(len(m3b)):
                    m3ba[i] = 1.0/(1+math.exp(-m3b[i]))
                m4 = np.matmul(m3ba,w4)
                m4b = np.add(m4,b4)
                m4ba = np.zeros(len(m4b))
                for i in range(len(m4b)):
                    m4ba[i] = 1.0/(1+math.exp(-m4b[i]))
                m5 = np.matmul(m4ba,w5)
                m5b = np.add(m5,b5)
                m5ba = np.zeros(len(m5b))
                for i in range(len(m5b)):
                        m5ba[i] = m5b[i]

                outputX_nrm = m5ba
                outputX = m5ba*stdY+meanY
                if j == 0:
                    Xy_nrm_4cls = outputX_nrm
                    Xy_4cls = outputX
                else:
                    Xy_nrm_4cls = np.vstack((Xy_nrm_4cls, outputX_nrm))
                    Xy_4cls = np.vstack((Xy_4cls, outputX))
            
            ############# Step 4: preprocessor prediction (SimulationStatus) for classifier #############
            succs_Xy = np.zeros((X.shape[0],1),dtype=np.float64)
            try: # find index of preprocessor inputs
                index1 = Xname.index("Average_CurrentDensity")
                index2 = Xname.index("Stack_Fuel_Utilization")
                index3 = Xname.index("Stack_Oxidant_Utilization")
                index4 = Xname.index("OxygenToCarbon_Ratio")
                index5 = Xname.index("Internal_Reforming")
                index6 = Xname.index("Oxidant_Recirculation")
            except:
                sys.exit('Code terminated: "preprocessor" input not defined') 
            try: 
                index7= Xname.index("PreReform")
            except:
                index7 = -1
                PreReform_fix=0.2 #[]
            try: 
                index8= Xname.index("cellsize")
            except:
                index8 = -1
                cellsize_fix=550 #[cm2]
            if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                try: 
                    index9 = Xname.index("VGRRate")
                    index11 = Xname.index("VGRH2OPassRate")
                    index12 = Xname.index("VGRCO2CaptureRate")
                    index13 = Xname.index("VGRH2PassRate")
                    index14 = Xname.index("VGRCOConvertRate")
                except:
                    sys.exit('Code terminated: "preprocessor" input not defined') 
                    
            # find value of preprocessor inputs
            for i in range(X.shape[0]):
                J = X[i,index1]/10.0 # mA/cm2
                FU = X[i,index2]
                AU = X[i,index3]
                OCR = X[i,index4]
                IR = X[i,index5]
                Arec = X[i,index6]
                if index7 == -1:
                    PreReform = PreReform_fix
                else:
                    PreReform = X[i,index7]
                if index8 == -1:
                    cellsize = cellsize_fix # cm2
                else:
                    cellsize = X[i,index8]
                    
                if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                    VGR = X[i,index9]
                    H2OCap = 1-X[i,index11]
                    CO2Cap = X[i,index12]
                    H2Cap = 1-X[i,index13]
                    WGS = X[i,index14]

                if preprocessor_name == None or preprocessor_name == 'NGFC_ccs':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'NGFC_nocc':
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
                elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
                elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                    FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
                elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                    FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
                else:
                    sys.exit('Code terminated: the selected "preprocessor" cannot be found')
                succs_Xy[i,0] = succ
            mean_succs = succs_Xy.mean(axis=0)
            std_succs = succs_Xy.std(axis=0)
            succs_Xy_nrm = (succs_Xy-mean_succs)/std_succs
            
            ############# Step 5: perform prediction of SimulationStatus #############
            X_nrm_4cls = np.concatenate((X_nrm, Xy_nrm_4cls, succs_Xy_nrm),axis=1)
            
            for j in range(X_row):
                inputX_cls = X_nrm_4cls[j,:]
                m1_cls = np.matmul(inputX_cls,w1_cls)
                m1b_cls = np.add(m1_cls,b1_cls)
                m1ba_cls = np.zeros(len(m1b_cls))
                for i in range(len(m1b_cls)):
                    m1ba_cls[i] = 1.0/(1+math.exp(-m1b_cls[i]))
                m2_cls = np.matmul(m1ba_cls,w2_cls)
                m2b_cls = np.add(m2_cls,b2_cls)
                m2ba_cls = np.zeros(len(m2b_cls))
                for i in range(len(m2b_cls)):
                    m2ba_cls[i] = m2b_cls[i]

                outputX_cls = m2ba_cls
                if j == 0:
                    Xy_cls = outputX_cls
                else:
                    Xy_cls = np.vstack((Xy_cls, outputX_cls))
            #convert to 0 and 1
            Xy_cls = np.argmax(Xy_cls, 1)
            
        ############# Step 6: Load the trained model for ROM #############
        print('Step 6: Load the trained model (outtrainingFile)')
        with open(self.outtrainingFile) as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                linestr = line.strip().split(' ')
                if linestr[0] == 'w1': 
                    w1_s = i+1
                if linestr[0] == 'w2': 
                    w2_s = i+1
                    w1_e = i-2
                if linestr[0] == 'w3': 
                    w3_s = i+1
                    w2_e = i-2
                if linestr[0] == 'w4': 
                    w4_s = i+1
                    w3_e = i-2
                if linestr[0] == 'w5': 
                    w5_s = i+1
                    w4_e = i-2
                if linestr[0] == 'b1': 
                    b1_s = i+1
                    w5_e = i-2
                if linestr[0] == 'b2': 
                    b2_s = i+1
                    b1_e = i-2
                if linestr[0] == 'b3': 
                    b3_s = i+1
                    b2_e = i-2
                if linestr[0] == 'b4': 
                    b4_s = i+1
                    b3_e = i-2
                if linestr[0] == 'b5': 
                    b5_s = i+1
                    b4_e = i-2
                if linestr[0] == 'meanS': 
                    meanS_s = i+1
                    b5_e = i-2
                if linestr[0] == 'meanY': 
                    meanY_s = i+1
                    meanS_e = i-2
                if linestr[0] == 'stdS': 
                    stdS_s = i+1
                    meanY_e = i-2
                if linestr[0] == 'stdY': 
                    stdY_s = i+1
                    stdS_e = i-2
                if linestr[0] == 'meanerr': 
                    meanerr_s = i+1
                    stdY_e = i-2
                if linestr[0] == 'stderr': 
                    stderr_s = i+1
                    meanerr_e = i-2
                if linestr[0] == 'end': 
                    stderr_e = i-2
                i += 1
            
            i = 0
            for line in lines:
                linestr = line.strip().split(' ')
                if i == w1_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w1 = np.array(linenum)
                    w1_row = w1_e-w1_s+1
                    w1_col = len(w1)
                if i > w1_s and i <= w1_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w1 = np.append(w1, linenum)
                if i == w2_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w2 = np.array(linenum)
                    w2_row = w2_e-w2_s+1
                    w2_col = len(w2)
                if i > w2_s and i <= w2_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w2 = np.append(w2, linenum)
                if i == w3_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w3 = np.array(linenum)
                    w3_row = w3_e-w3_s+1
                    w3_col = len(w3)
                if i > w3_s and i <= w3_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w3 = np.append(w3, linenum)
                if i == w4_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w4 = np.array(linenum)
                    w4_row = w4_e-w4_s+1
                    w4_col = len(w4)
                if i > w4_s and i <= w4_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w4 = np.append(w4, linenum)
                if i == w5_s:
                    linenum = [float(lineele) for lineele in linestr]
                    w5 = np.array(linenum)
                    w5_row = w5_e-w5_s+1
                    w5_col = len(w5)
                if i > w5_s and i <= w5_e:
                    linenum = [float(lineele) for lineele in linestr]
                    w5 = np.append(w5, linenum)
                if i == b1_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b1 = np.array(linenum)
                if i > b1_s and i <= b1_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b1 = np.append(b1, linenum)
                if i == b2_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b2 = np.array(linenum)
                if i > b2_s and i <= b2_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b2 = np.append(b2, linenum)
                if i == b3_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b3 = np.array(linenum)
                if i > b3_s and i <= b3_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b3 = np.append(b3, linenum)
                if i == b4_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b4 = np.array(linenum)
                if i > b4_s and i <= b4_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b4 = np.append(b4, linenum)
                if i == b5_s:
                    linenum = [float(lineele) for lineele in linestr]
                    b5 = np.array(linenum)
                if i > b5_s and i <= b5_e:
                    linenum = [float(lineele) for lineele in linestr]
                    b5 = np.append(b5, linenum)
                if i == meanS_s:
                    linenum = [float(lineele) for lineele in linestr]
                    meanS = np.array(linenum)
                if i > meanS_s and i <= meanS_e:
                    linenum = [float(lineele) for lineele in linestr]
                    meanS = np.append(meanS, linenum)
                if i == meanY_s:
                    linenum = [float(lineele) for lineele in linestr]
                    meanY = np.array(linenum)
                if i > meanY_s and i <= meanY_e:
                    linenum = [float(lineele) for lineele in linestr]
                    meanY = np.append(meanY, linenum)
                if i == stdS_s:
                    linenum = [float(lineele) for lineele in linestr]
                    stdS = np.array(linenum)
                if i > stdS_s and i <= stdS_e:
                    linenum = [float(lineele) for lineele in linestr]
                    stdS = np.append(stdS, linenum)
                if i == stdY_s:
                    linenum = [float(lineele) for lineele in linestr]
                    stdY = np.array(linenum)
                if i > stdY_s and i <= stdY_e:
                    linenum = [float(lineele) for lineele in linestr]
                    stdY = np.append(stdY, linenum)
                    
                # two more variables meanerr, stderr
                if i == meanerr_s:
                    linenum = [float(lineele) for lineele in linestr]
                    meanerr = np.array(linenum)
                if i > meanerr_s and i <= meanerr_e:
                    linenum = [float(lineele) for lineele in linestr]
                    meanerr = np.append(meanerr, linenum)
                if i == stderr_s:
                    linenum = [float(lineele) for lineele in linestr]
                    stderr = np.array(linenum)
                if i > stderr_s and i <= stderr_e:
                    linenum = [float(lineele) for lineele in linestr]
                    stderr = np.append(stderr, linenum)
                i += 1
                
            w1 = np.reshape(w1, (w1_row, w1_col))
            w2 = np.reshape(w2, (w2_row, w2_col))
            w3 = np.reshape(w3, (w3_row, w3_col))
            w4 = np.reshape(w4, (w4_row, w4_col))
            w5 = np.reshape(w5, (w5_row, w5_col))
        
        ############# Step 7: perform prediction of other variables #############
        # Normalize S, Y, X again
        S_nrm = (S - np.tile(meanS, [S_row, 1]))/np.tile(stdS, [S_row, 1])
        Y_nrm = (Y - np.tile(meanY, [Y_row, 1]))/np.tile(stdY, [Y_row, 1])
        X_nrm = (X - np.tile(meanS, [X_row, 1]))/np.tile(stdS, [X_row, 1])
        
        for j in range(X_row):
            inputX = X_nrm[j,:]
            m1 = np.matmul(inputX,w1)
            m1b = np.add(m1,b1)
            m1ba = np.zeros(len(m1b))
            for i in range(len(m1b)):
                m1ba[i] = 1.0/(1+math.exp(-m1b[i]))
            m2 = np.matmul(m1ba,w2)
            m2b = np.add(m2,b2)
            m2ba = np.zeros(len(m2b))
            for i in range(len(m2b)):
                m2ba[i] = 1.0/(1+math.exp(-m2b[i]))
            m3 = np.matmul(m2ba,w3)
            m3b = np.add(m3,b3)
            m3ba = np.zeros(len(m3b))
            for i in range(len(m3b)):
                m3ba[i] = 1.0/(1+math.exp(-m3b[i]))
            m4 = np.matmul(m3ba,w4)
            m4b = np.add(m4,b4)
            m4ba = np.zeros(len(m4b))
            for i in range(len(m4b)):
                m4ba[i] = 1.0/(1+math.exp(-m4b[i]))
            m5 = np.matmul(m4ba,w5)
            m5b = np.add(m5,b5)
            m5ba = np.zeros(len(m5b))
            for i in range(len(m5b)):
                m5ba[i] = m5b[i]
                    
            outputX_nrm = m5ba
            outputX = m5ba*stderr+meanerr
            if j == 0:
                err_nrm = outputX_nrm
                err = outputX
            else:
                err_nrm = np.vstack((err_nrm, outputX_nrm))
                err = np.vstack((err, outputX))
                
        ############# Step 8: preprocessor prediction for ROM #############
        preprocessor_result = np.zeros((X.shape[0], 20),dtype=np.float64)

        # find index of preprocessor inputs
        try: 
            index1 = Xname.index("Average_CurrentDensity")
            index2 = Xname.index("Stack_Fuel_Utilization")
            index3 = Xname.index("Stack_Oxidant_Utilization")
            index4 = Xname.index("OxygenToCarbon_Ratio")
            index5 = Xname.index("Internal_Reforming")
            index6 = Xname.index("Oxidant_Recirculation")
        except:
            sys.exit('Code terminated: "preprocessor" input not defined') 
        try: 
            index7= Xname.index("PreReform")
        except:
            index7 = -1
            PreReform_fix=0.2 #[]
        try: 
            index8= Xname.index("cellsize")
        except:
            index8 = -1
            cellsize_fix=550 #[cm2]
        if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
            try: 
                index9 = Xname.index("VGRRate")
                index11 = Xname.index("VGRH2OPassRate")
                index12 = Xname.index("VGRCO2CaptureRate")
                index13 = Xname.index("VGRH2PassRate")
                index14 = Xname.index("VGRCOConvertRate")
            except:
                sys.exit('Code terminated: "preprocessor" input not defined') 
                                  
        for i in range(X.shape[0]):
            J = X[i,index1]/10.0 # mA/cm2
            FU = X[i,index2]
            AU = X[i,index3]
            OCR = X[i,index4]
            IR = X[i,index5]
            Arec = X[i,index6]
            if index7 == -1:
                PreReform = PreReform_fix
            else:
                PreReform = X[i,index7]
            if index8 == -1:
                cellsize = cellsize_fix # cm2
            else:
                cellsize = X[i,index8]

            if preprocessor_name == 'NGFC_ccs_vgr' or preprocessor_name == 'IGFC_ccs_vgr':
                VGR = X[i,index9]
                H2OCap = 1-X[i,index11]
                CO2Cap = X[i,index12]
                H2Cap = 1-X[i,index13]
                WGS = X[i,index14]

            if preprocessor_name == None or preprocessor_name == 'NGFC_ccs':
                FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
            elif preprocessor_name == 'NGFC_nocc':
                FuelOut, AirOut, FuelIn,succ=self.NGFC_nocc(J,FU,AU,OCR,IR,Arec,PreReform,cellsize)
            elif preprocessor_name == 'IGFC_ccs': # IGFC: conventional, Enhanced, Catalytic
                FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs(J,FU,AU,OCR,IR,Arec,PreReform,cellsize,igfc)
            elif preprocessor_name == 'NGFC_ccs_vgr': # NGFC CCS VGR
                FuelOut, AirOut, FuelIn,succ=self.NGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize)
            elif preprocessor_name == 'IGFC_ccs_vgr': # IGFC VGR: conventional, Enhanced, Catalytic
                FuelOut, AirOut, FuelIn,succ=self.IGFC_ccs_vgr(J,FU,AU,OCR,IR,Arec,PreReform,VGR,H2OCap,CO2Cap,H2Cap,WGS,cellsize,igfc)
            else:
                sys.exit('Code terminated: the selected "preprocessor" cannot be found')

            preprocessor_result[i,0] = np.sum(FuelOut)
            preprocessor_result[i,1] = FuelOut[7]
            preprocessor_result[i,2] = FuelOut[0]
            preprocessor_result[i,3] = FuelOut[6]
            preprocessor_result[i,4] = FuelOut[2]
            preprocessor_result[i,5] = FuelOut[5]
            preprocessor_result[i,6] = FuelOut[4]
            preprocessor_result[i,7] = np.sum(AirOut)
            preprocessor_result[i,8] = AirOut[3]
            preprocessor_result[i,9] = AirOut[4]
            preprocessor_result[i,10] = AirOut[0]
            preprocessor_result[i,11] = AirOut[2]
            preprocessor_result[i,12] = AirOut[1]
            preprocessor_result[i,13] = np.sum(FuelIn)
            preprocessor_result[i,14] = FuelIn[7]
            preprocessor_result[i,15] = FuelIn[0]
            preprocessor_result[i,16] = FuelIn[6]
            preprocessor_result[i,17] = FuelIn[2]
            preprocessor_result[i,18] = FuelIn[5]
            preprocessor_result[i,19] = FuelIn[4]
        
        ############# Step 9: Final prediction for ROM #############
        Xy = preprocessor_result - err
        Xy_nrm = (Xy - np.tile(meanY, [X_row, 1]))/np.tile(stdY, [X_row, 1])
        
        # Copy to Global
        [self.S_row, self.Y_row, self.S_col, self.Y_col] = [S_row, Y_row, S_col, Y_col]
        self.S_norm = S_nrm
        self.Y_norm = Y_nrm
        self.S = S
        self.Y = Y
        [self.stdS, self.stdY] = [stdS, stdY]
        self.X = X
        self.Xy = Xy
        self.X_norm = X_nrm
        self.Xy_norm = Xy_nrm
        self.Sname = Sname
        self.Yname = Yname
        
        ############# Step 10: Write the predictions #############
        print('Step 10: Write the predictions')
        with open(self.outpredictionFile, 'w') as f:
            for name in Xname:
                f.write(name + '\t')
            if cls_enabled == True:
                f.write('SimulationStatus\t')
            for i in range(Y_col):
                f.write(Yname[i] + '\t')
            f.write('\n')
            
            for i in range(X_row):
                # write input variables
                for j in range(S_col):
                    f.write('{:11.4E}\t'.format(X[i, j]))
                # write simulation status
                if cls_enabled == True:
                    f.write('{:11.4E}\t'.format(Xy_cls[i]))
                # write output variables
                for j in range(Y_col):
                    f.write('{:11.4E}\t'.format(Xy[i, j]))
                f.write('\n')
        print('End of code\n')
    
    def percent2intervl(self, percentage, var = None):
        print('############################################################\
              \nPercentage to Confidence Interval\
              \n############################################################')
        
        # load cross validation results
        Yname, ERR = self.file_read(self.outcrossvaliFile)
        
        # find the units
        names_input, units_input, names_output, units_output = self.variable_options()
        Yunit = []
        for i in range(len(Yname)):
            tempindex = names_output.index(Yname[i])
            tempunit = units_output[tempindex]
            Yunit.append(tempunit)
            
        # compute confidence interval
        interval_all = np.zeros((len(Yname),),dtype=np.float64)
        for i in range(len(Yname)):
            err = np.sort(ERR[:, i])
            N = len(err)
            n = (N-1)*percentage/100.0 + 1
            if n == 1: 
                interval = err[0]
            elif n == N:
                interval = err[N-1]
            else:
                k = int(n)
                d = n-k
                interval = err[k-1]+d*(err[k]-err[k-1])
            
            interval_all[i] = interval
            
            if var == None:    
                print('For "' + str(Yname[i]) + '":'
                     + '[' + Yunit[i] + ']'
                     +' \n\t' 
                     + str(percentage) + '% confidence interval is '
                     + '\u00B1' + '{:11.4E}\t'.format(interval))
            elif Yname[i] == var:
                print('For "' + str(Yname[i]) + '":'
                     + '[' + Yunit[i] + ']'
                     +' \n\t' 
                     + str(percentage) + '% confidence interval is '
                     + '\u00B1' + '{:11.4E}\t'.format(interval))
            elif var not in Yname:
                print('The given variable cannot be found')
            
        print('End of code\n')
        return(interval_all)
        
    def intervl2percent(self, interval, var = None):
        print('############################################################\
              \nConfidence Interval to Percentage\
              \n############################################################')
        
        # load cross validation results
        Yname, ERR = self.file_read(self.outcrossvaliFile)
        
        # find the units
        names_input, units_input, names_output, units_output = self.variable_options()
        Yunit = []
        for i in range(len(Yname)):
            tempindex = names_output.index(Yname[i])
            tempunit = units_output[tempindex]
            Yunit.append(tempunit)
        
        # compute confidence percentage
        percentage_all = np.zeros((len(Yname),),dtype=np.float64)
        for i in range(len(Yname)):
            if var == Yname[i]:
                err = np.sort(ERR[:, i])
                N = len(err)
                if interval <= err[0]:
                    percentage = 0
                elif interval >= err[N-1]:
                    percentage = 1
                else:
                    result = np.where(err>interval)
                    index = result[0]
                    k = index[0]
                    percentage = ((interval-err[k-1])/(err[k]-err[k-1])+k-1)/float(N-1)
                
                percentage_all[i] = percentage
                
                print('For "' + str(Yname[i]) + '": ' 
                     + '[' + Yunit[i] + ']'
                     + '\n\t\u00B1' + str(interval)
                     + ' interval has a confidence of ' +  str(round(percentage*100, 2)) + '%')
            elif var not in Yname:
                print('The given variable cannot be found')

        print('End of code\n')
        return(percentage_all)
        
    def plot_contour_2D(self, xvariable, yvariable, zvariable, 
                        pltoption = 0, saveoption = False):
        '''
        The function plots 2D contour of designs and responses
        pltoption = 0: plot both training and prediction sets; 1: plot only training sets, 2: plot only prediction sets
        '''
        # check if the given variables are in the list
        if (xvariable not in self.Sname) or (yvariable not in self.Sname) or (zvariable not in self.Yname):
            sys.exit('Code terminated: variable index out of bound')
        v1 = self.Sname.index(xvariable)+1
        v2 = self.Sname.index(yvariable)+1
        v3 = self.Yname.index(zvariable)+1
        option = int(pltoption)
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_input.index(yvariable)
        yunit = units_input[tempindex]
        tempindex = names_output.index(zvariable)
        zunit = units_output[tempindex]
            
        # Generate inPrediction4contour.dat
        if option == 0 or option == 2:
            Xname, Xvalue = self.file_read(self.inpredictionFile)
            Xvalue_mean = np.mean(Xvalue, axis = 0)
            [X_row, X_col] = Xvalue.shape
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_contour_DNN.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_contour_DNN.dat'
            with open(self.inpredictionFile, 'w') as f:
                for name in Xname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(X_row):
                    for j in range(X_col):
                        if (j+1) == v1 or (j+1) == v2:
                            f.write('{:11.4E}\t'.format(Xvalue[i, j]))
                        else:
                            f.write('{:11.4E}\t'.format(Xvalue_mean[j]))
                    f.write('\n')
            self.prediction()
            os.remove(self.inpredictionFile)
            os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig
            

        if option == 0: # Default: plot both training and prediction sets
            x1 = self.S[:, v1-1]
            y1 = self.S[:, v2-1]
            z1 = self.Y[:, v3-1]
            x2 = self.X[:, v1-1]
            y2 = self.X[:, v2-1]
            z2 = self.Xy[:, v3-1]
            
            plt.figure(figsize=(17.5,6))
            plt.subplot(1, 2, 1)
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            C = plt.tricontour(x1, y1, z1, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x1, y1, z1, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            #plt.colorbar().set_label(label='a label',size=15,weight='bold')
            plt.xlim((min(min(x1), min(x2)), max(max(x1), max(x2))))
            plt.ylim((min(min(y1), min(y2)), max(max(y1), max(y2))))
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            
            plt.subplot(1, 2, 2)
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            C = plt.tricontour(x2, y2, z2, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x2, y2, z2, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.xlim((min(min(x1), min(x2)), max(max(x1), max(x2))))
            plt.ylim((min(min(y1), min(y2)), max(max(y1), max(y2))))
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 1: # plot training sets
            x = self.S[:, v1-1]
            y = self.S[:, v2-1]
            z = self.Y[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            plt.figure(figsize=(8,6))
            C = plt.tricontour(x, y, z, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x, y, z, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 2: # plot prediciton sets
            x = self.X[:, v1-1]
            y = self.X[:, v2-1]
            z = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            plt.figure(figsize=(8,6))
            C = plt.tricontour(x, y, z, 10, linewidths = 0.5, colors = 'k')
            Cf = plt.tricontourf(x, y, z, 20, alpha = 0.75)
            #plt.clabel(C, inline = True, fontsize = 10)
            plt.colorbar(orientation = 'vertical', shrink = 1).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.xticks(fontsize = 12)
            plt.yticks(fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        # save option
        if saveoption == True:
            figurename = '2D_contour.png'
            plt.savefig(figurename)
            
    def plot_contour_3D(self, xvariable, yvariable, zvariable, 
                        pltoption = 0, saveoption = False):
        '''
        The function plots 2D contour of designs and responses
        pltoption = 0: plot both training and prediction sets; 1: plot only training sets, 2: plot only prediction sets
        '''
        # check if the given variables are in the list
        if (xvariable not in self.Sname) or (yvariable not in self.Sname) or (zvariable not in self.Yname):
            sys.exit('Code terminated: variable index out of bound')
        v1 = self.Sname.index(xvariable)+1
        v2 = self.Sname.index(yvariable)+1
        v3 = self.Yname.index(zvariable)+1
        option = int(pltoption)
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_input.index(yvariable)
        yunit = units_input[tempindex]
        tempindex = names_output.index(zvariable)
        zunit = units_output[tempindex]
        
        # Generate inPrediction4contour.dat
        if option == 0 or option == 2:
            Xname, Xvalue = self.file_read(self.inpredictionFile)
            Xvalue_mean = np.mean(Xvalue, axis = 0)
            [X_row, X_col] = Xvalue.shape
            inpredictionFile_orig = self.inpredictionFile
            outpredictionFile_orig = self.outpredictionFile
            self.inpredictionFile = self.work_path + '/inPrediction_contour_kriging.dat'
            self.outpredictionFile = self.work_path + '/outPrediction_contour_kriging.dat'
            with open(self.inpredictionFile, 'w') as f:
                for name in Xname:
                    f.write(name + '\t')
                f.write('\n')
                for i in range(X_row):
                    for j in range(X_col):
                        if (j+1) == v1 or (j+1) == v2:
                            f.write('{:11.4E}\t'.format(Xvalue[i, j]))
                        else:
                            f.write('{:11.4E}\t'.format(Xvalue_mean[j]))
                    f.write('\n')
            self.prediction()
            os.remove(self.inpredictionFile)
            os.remove(self.outpredictionFile)
            self.inpredictionFile = inpredictionFile_orig
            self.outpredictionFile = outpredictionFile_orig
            
        if option == 0: # Default: plot both training and prediction sets
            x1 = self.S[:, v1-1]
            y1 = self.S[:, v2-1]
            z1 = self.Y[:, v3-1]
            x2 = self.X[:, v1-1]
            y2 = self.X[:, v2-1]
            z2 = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(18.5,6))
            ax = fig.add_subplot(1, 2, 1, projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x1, y1, z1, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            
            ax = fig.add_subplot(1, 2, 2, projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x2, y2, z2, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 1: # plot training sets
            x = self.S[:, v1-1]
            y = self.S[:, v2-1]
            z = self.Y[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x, y, z, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Training sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
            
        elif option == 2: # plot prediciton sets
            x = self.X[:, v1-1]
            y = self.X[:, v2-1]
            z = self.Xy[:, v3-1]
            xname = self.Sname[v1-1]
            yname = self.Sname[v2-1]
            zname = self.Yname[v3-1]
            
            fig = plt.figure(figsize=(8,6))
            ax = plt.axes(projection = '3d')
            ax.tick_params(labelsize=12)
            surf = ax.plot_trisurf(x, y, z, color = 'k', cmap = plt.get_cmap('rainbow'))
            fig.colorbar(surf, orientation = 'vertical', shrink = 0.8).ax.tick_params(labelsize=12)
            plt.xlabel(xname+', ['+xunit+']', fontsize = 12)
            plt.ylabel(yname+', ['+yunit+']', fontsize = 12)
            plt.title('Prediction sets: '+zname+', ['+zunit+']', fontsize = 12)
            plt.show()
        
        # save option
        if saveoption == True:
            figurename = '3D_contour.png'
            plt.savefig(figurename)
            
    def plot_box(self, xvariable, yvariable, saveoption = False):
        '''
        The function is for box plot, it can help to perform sensitivity studies
        '''
        # convert to pandam dataframe
        S = pd.DataFrame(data = self.S, columns = self.Sname, dtype = 'float')
        Y = pd.DataFrame(data = self.Y, columns = self.Yname, dtype = 'float')
        
        # find the units for x,y,z variables
        names_input, units_input, names_output, units_output = self.variable_options()
        tempindex = names_input.index(xvariable)
        xunit = units_input[tempindex]
        tempindex = names_output.index(yvariable)
        yunit = units_output[tempindex]
        
        # generate box plot data
        x = S[[xvariable]]
        y = Y[[yvariable]]
        min_x = min(x.values)
        max_x = max(x.values)
        x = round((x-min_x)/((max_x-min_x)/9), 0)*((max_x-min_x)/9)+min_x
        x = round(x, 2)
        #xy = pd.concat([x, y], axis = 1, sort = False)
        #print(x.sort_values(by = ['Average_CurrentDensity']))
        #print(xy)
        
        # box plot
        plt.figure(figsize=(18.5,6))
        sns.set_context("paper", font_scale=3)
        sns.set_style('ticks')
        bplot = sns.boxplot(y=y[yvariable], x=x[xvariable], 
                    color = 'yellow', width = 0.5)
        bplot = sns.swarmplot(y=y[yvariable], x=x[xvariable], 
                    color = 'black', alpha = 0.5)
        sns.axes_style()
        bplot.axes.set_title('Design-response sites', fontsize = 25)
        bplot.set_xlabel(xvariable+', ['+xunit+']', fontsize = 25)
        bplot.set_ylabel(yvariable+', ['+yunit+']', fontsize = 25)
        bplot.tick_params(labelsize = 25)
        plt.show()
        
        # save option
        if saveoption == True:
            figurename = 'boxplot.png'
            plt.savefig(figurename)



