import time
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
import os, sys
from shutil import copyfile
import ast
import Gibbs_V4 as Gibbs_Module

#Script assumes polyFTS returns operators: Hamiltonian, Pressure, and Species Chemical Potentials

''' Initialize instance of Gibbs class. '''
program = 'polyFTS' # or 'MD'
jobtype = 'CL' # CL,SCFT,MF
barostat = True # whether to apply a barostat to a system
number_species = 1
GM = Gibbs_Module.Gibbs_System(program,number_species)
GM.SetJobType(jobtype)
GM.SetRunTemplate('template.in',['__CTOT__','__Phi1__','__Phi2__'])
GM.SetDt([0.01,0.1,0.1])
if barostat: # good for troubleshooting
	GM.SetRunTemplate('template.in',['__CTOT__','__Phi1__','__Phi2__'])
	GM.Barostat = barostat 
	GM.TargetP = 0.113*1.384688387**3 # kbT/lo**3
	GM.SetDt([0.01])


# For MF need to specify the interactions. Currently only for two-species: uPP,uSS,uPS
# 	Values are in polyFTS units. 
# TODO: Turn into a matrix, make MF model general.
GM.SetInteractions([380.1713095,50.95952805,173.502])
GM.SetInteractionRange([2.708173622,2.253219669,2.491178966])
GM.SetUseRPA(False) #RPA unstable inside spinodal, default to false

''' Set total species number and total volume. '''
CTot = 0.04 # number concentration
NumberFrac_Species = [1.0]#,0.]
NumberDens_Species = [CTot*x for x in NumberFrac_Species]

GM.SetSpeciesCTotal(NumberDens_Species)
GM.SetSpeciesDOP([10,1]) # the degree of polymerization, topology doesn't matter. 

''' Set the initial guesses for the coexistence pts of BoxI. '''
# auto-converts internally to either polyFTS or MD suitable parameters
fI  = 1.
CI1 = 1.0*CTot
#CI2 = 0.1*CTot
# calculate BoxII pts
#fII = 1.-fI
#CII1 = (NumberDens_Species[0]-CI1*fI)/fII
#CII2 = (NumberDens_Species[1]-CI2*fI)/fII

#VarInit = [fI,fII,CI1,CII1,CI2,CII2]
if barostat:
	VarInit = [1.,CI1]#,CI2]

GM.SetInitialGuess(VarInit)

def uPS(Temp):
	''' Polymer-Solvent Interaction Parameter as a function of temperature. '''
	# Returns _uPS in polyFTS units
	a0 = 2.099E-1
	a1 = 3.723E-4
	a2 = -9.637E-7
	a3 = 7.374E-10
	aPS = 3.45 # angstrom
	L   = 1.384688 #angstroms - polyFTS length Scale

		
	_uPS = a0 + a1*Temp + a2*Temp*Temp + a3*Temp**3
	_uPS = _uPS*44.54662*(aPS**3)/L**3
	_aPS = aPS/L
	
	return _uPS

# Sanity check
print("uPS at 200C: {}".format(uPS(200.)))

''' Begin Running Gibbs Simulations. '''

if jobtype == "CL" and not barostat:
	Tolerance 	= 1e-6 # MF Tolerance
	MFMaxIter	= 1e6
	nsteps 		= 20000 # Number CL steps to run for
	prefac 		= 1000./1.384688**3 # conversion to monomers/nm**3
	N 			= 6 # length of the polymer chain
	Temp 		= 25.0 # temperature to run at
	_uPS 		= uPS(Temp)
	
	GM.SetSpeciesDOP([N,1])
	GM.SetInteractions([380.1713095,50.95952805,_uPS])
	GM.SetUseOneNode(True)
	GM.SetNPolyFTSBlocks(200)
	GM.SetNPolyFTSBlocksMin(200)
	GM.SetNPolyFTSBlocksMax(200)
	GM.SetOperatorRelTol(0.001)
	GM.SetVolFracBounds([0.1,0.9])	
	GM.PSInteraction = _uPS
	
	print("DOP: {}".format(N))
	
	CoexptsMF = open('Coexpts_MF.data','w')
	CoexptsMF.write("# T  uPS  fI  fII  CI_1  CII_1  CI_2  CII_2\n")
	Coexpts = open('Coexpts_CL.data','w')
	Coexpts.write("# T  uPS  fI  fII  CI_1  CII_1  CI_2  CII_2\n")

	GM.DvalsCurrent = [[1.]]*4 # initialize Dvals
	
	''' First run a MF simulation to get MF coexistence points. '''
	GM.SetJobType('MF')
	GM.GibbsLogFileName = 'gibbs_MF.dat'
	GM.GibbsErrorFileName = 'error_MF.dat'
	GM.SetDt([0.01/N,0.1/N,0.1/N])
	while max(np.abs(GM.DvalsCurrent)) > Tolerance:
		if GM.Iteration >= MFMaxIter:
			break # end-loop
		else:
			GM.TakeGibbsStep()
			
	Fvals = GM.ValuesCurrent
	print("MF Converged...")
	print(Fvals)
	CoexptsMF.write('{} {} {} {} {} {} {} {} \n'.format(Temp,_uPS,Fvals[0],Fvals[1],prefac*Fvals[2],prefac*Fvals[3],prefac*Fvals[4],prefac*Fvals[5]))
	CoexptsMF.flush()	
	
	''' Start from MF solution and run CL simulation. '''
	print("Running CL...")
	GM.SetJobType('CL')
	GM.GibbsLogFileName = 'gibbs_CL.dat'
	GM.GibbsErrorFileName = 'error_CL.dat'
	GM.Iteration = 1 # reset iteration
	GM.SetInitialGuess(Fvals) # set initial variables from MF simulation
	GM.SetDt([0.01/N,0.1/N,0.1/N])
	
	for step in range(nsteps): # run CL simulation
		GM.TakeGibbsStep()
		print("Step: {0} RunTime: {1:3.3e} min.".format(step,GM.StepRunTime))
	
	GM.SetDt([0.01/N,0.1/N,0.1/N]) # decrease time step once warmed up for better stats	
	for step in range(1000): # run CL simulation
		GM.TakeGibbsStep()
		print("Step: {0} RunTime: {1:3.3e} min.".format(step,GM.StepRunTime))
	
	Fvals = GM.ValuesCurrent
	GM.Iteration = 1 # reset iteration
	GM.SetInitialGuess(VarInit) # set next simulation to start from prior
	GM.ValuesCurrent = []
	GM.OperatorsLast = []
	
	Coexpts.write('{} {} {} {} {} {} {} {} \n'.format(Temp,_uPS,Fvals[0],Fvals[1],prefac*Fvals[2],prefac*Fvals[3],prefac*Fvals[4],prefac*Fvals[5]))
	Coexpts.flush()	


if jobtype == "CL" and barostat:
	Tolerance 	= 1e-6 # MF Tolerance
	MFMaxIter	= 1e6
	nsteps 		= 20000 # Number CL steps to run for
	prefac 		= 1000./1.384688**3 # conversion to monomers/nm**3
	N 			= 20 # length of the polymer chain
	Temp 		= 100.0 # temperature to run at
	_uPS 		= uPS(Temp)
	
	GM.SetSpeciesDOP([N,1])
	GM.SetInteractions([380.1713095,50.95952805,_uPS])
	GM.SetUseOneNode(True)
	GM.SetNPolyFTSBlocks(100)
	GM.SetNPolyFTSBlocksMin(100)
	GM.SetNPolyFTSBlocksMax(100)
	GM.SetOperatorRelTol(0.001)
	GM.SetVolFracBounds([0.01,1.02])	
	GM.PSInteraction = _uPS
	
	print("DOP: {}".format(N))
	
	CoexptsMF = open('Coexpts_MF.data','w')
	CoexptsMF.write("# T  uPS  CI_1 CI_2  CTot\n")
	Coexpts = open('Coexpts_CL.data','w')
	Coexpts.write("# T  uPS  CI_1 CI_2  CTot\n")	
	
	''' Start from MF solution and run CL simulation. '''
	print("Running CL...")
	GM.SetJobType('CL')
	GM.GibbsLogFileName = 'gibbs_CL.dat'
	GM.GibbsErrorFileName = 'error_CL.dat'
	GM.Iteration = 1 # reset iteration
	GM.SetInitialGuess(VarInit) # set initial variables from MF simulation
	GM.SetDt([0.01])
	
	for step in range(nsteps): # run CL simulation
		GM.TakeGibbsStep()
		print("Step: {0} RunTime: {1:3.3e} min.".format(step,GM.StepRunTime))
	
		Fvals = GM.ValuesCurrent
		print(Fvals)
		Coexpts.write('{} {} {} \n'.format(Temp,_uPS,Fvals[1]))
		Coexpts.flush()	
