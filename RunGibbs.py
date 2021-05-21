import time
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
import os, sys
from shutil import copyfile
import ast
import Gibbs_V3 as Gibbs_Module

#Script assumes polyFTS returns operators: Hamiltonian, Pressure, and Species Chemical Potentials

''' Initialize instance of Gibbs class. '''
program = 'polyFTS' # or 'MD'
jobtype = 'MF' # CL,SCFT,MF
number_species = 2  
GM = Gibbs_Module.Gibbs_System(program,number_species)
GM.SetJobType(jobtype)
GM.SetRunTemplate('template.in',['__CTOT__','__Phi1__','___Phi2__'])
GM.SetDt([0.01,0.02,0.02])

# For MF need to specify the interactions. Currently only for two-species: uPP,uSS,uPS
# 	Values are in polyFTS units. 
# TODO: Turn into a matrix, make MF model general.
GM.SetInteractions([423.6405479,50.95952805,173.502])
GM.SetInteractionRange([2.708173622,2.253219669,2.491178966])
GM.SetUseRPA(False) #RPA unstable inside spinodal, default to false

def uPS(Temp):
	''' Polymer-Solvent Interaction Parameter as a function of temperature. '''
	# Returns _uPS in polyFTS units
	a0 = 1.4086E-1
	a1 = 1.3857E2
	a2 = -5.9751E4
	a3 = 7.5812E6
	aPS = 3.45 # angstrom
	L   = 1.384688 #angstroms - polyFTS length Scale

	_Temp = Temp + 273 # convert to Kelvin
	_uPS = a0 + a1/_Temp + a2/_Temp/_Temp + a3/_Temp**3
	_uPS = 0.235
	_uPS = _uPS*44.54662*(aPS**3)/L**3
	_aPS = aPS/L
	
	return _uPS
	
def CalcC(_xfrac):
	''' Calculate the total segment concentration as function of monomer fraction. '''
	# Returns _uPS in polyFTS units
	a0 = 33.237
	a1 = -29.479
	a2 = 11.882
	a3 = 0.
	L   = 0.1384688 #nms - polyFTS length Scale

		
	_Ctot = a0 + a1*_xfrac + a2*_xfrac*_xfrac + a3*_xfrac**3
	_Ctot = _Ctot*L**3
	
	return _Ctot
	
''' Set total species number and total volume. '''
_xfrac = 0.1 # mole fraction of 0.0909
CTot = CalcC(_xfrac) # number concentration
NumberFrac_Species = [0.1,0.9]
NumberDens_Species = [CTot*x for x in NumberFrac_Species]
GM.SetSpeciesCTotal(NumberDens_Species)
GM.SetSpeciesDOP([20,1]) # the degree of polymerization, topology doesn't matter. 

''' Set the initial guesses for the coexistence pts of BoxI. '''
# auto-converts internally to either polyFTS or MD suitable parameters
fI  = 0.51
CI1 = 0.01*CTot
CI2 = 0.99*CTot
# calculate BoxII pts
fII = 1.-fI
CII1 = (NumberDens_Species[0]-CI1*fI)/fII
CII2 = (NumberDens_Species[1]-CI2*fI)/fII

VarInit = [fI,fII,CI1,CII1,CI2,CII2]
GM.SetInitialGuess(VarInit)

# Sanity check
print("uPS at 200C: {}".format(uPS(200.)))
print("CTot at x=0.10: {}".format(CalcC(0.10)))

''' Begin Running Gibbs Simulations. '''

if jobtype == "MF":
	Tolerance = 1e-6
	MF_MaxIterations = 5e5
	prefac = 1000./1.384688**3 # conversion to monomers/nm**3
	print(prefac)
	MW_List = [2,3,4,5,6,7,8,10,15,20,40,50]
	Xfrac_List = [0.090909]
	
	for i,N in enumerate(MW_List): # Loop over molecular weight
		
		for j,_xfrac in enumerate(Xfrac_List): # loop over c_total
		
			''' Set total species number and total volume. '''
			CTot = CalcC(_xfrac) # number concentration
			NumberFrac_Species = [_xfrac,(1.-_xfrac)]
			NumberDens_Species = [CTot*x for x in NumberFrac_Species]
			GM.SetSpeciesCTotal(NumberDens_Species)
			GM.SetSpeciesDOP([N,1]) # the degree of polymerization, topology doesn't matter. 

			''' Set the initial guesses for the coexistence pts of BoxI. '''
			# auto-converts internally to either polyFTS or MD suitable parameters
			fI  = 0.51
			CI1 = 0.005*CTot
			CI2 = 0.995*CTot
			# calculate BoxII pts
			fII = 1.-fI
			CII1 = (NumberDens_Species[0]-CI1*fI)/fII
			CII2 = (NumberDens_Species[1]-CI2*fI)/fII

			VarInit = [fI,fII,CI1,CII1,CI2,CII2]
			GM.SetInitialGuess(VarInit)
						
			
		
			GM.SetDt([0.025*1/N,0.1*1/N,0.1*1/N])
			#Temp_List = np.linspace(140,550,25)
			Temp_List = [200]#np.linspace(50,600,551)
			GM.Write2Log("DOP:   {}\n".format(N))
			GM.Write2Log("Xfrac: {}\n".format(_xfrac))
			GM.Write2Log('CTot:  {}\n'.format(CTot))
			GM.SetSpeciesDOP([N,1])
			Coexpts = open('Coexpts_N_{0:2d}_xfrac_{1:2.2f}.data'.format(N,_xfrac),'w')
			Coexpts.write("# T  uPS  fI  fII  CI_1  CII_1  CI_2  CII_2\n")
			for i,Temp in enumerate(Temp_List): # Generate Temperature-Rho Phase Diagram
				GM.Write2Log("Iter: {} Temp.: {}\n".format(i,Temp))
				_uPS = uPS(Temp)
				GM.SetInteractions([374.577,50.95952805,_uPS])
				GM.DvalsCurrent = [1.]
				while max(np.abs(GM.DvalsCurrent)) > Tolerance:
					if GM.Iteration > MF_MaxIterations:
						GM.Write2Log("Reached Max Iterations: Breaking Out!\n")
						break

					GM.TakeGibbsStep()
				

				Fvals = GM.ValuesCurrent
				GM.Iteration = 1 # reset iteration
				GM.SetInitialGuess(VarInit) # set next simulation to start from prior
				GM.ValuesCurrent = []
				GM.ValuesCurrent = []
				GM.ValuesCurrent = []
				GM.OperatorsLast = []
				
				Coexpts.write('{} {} {} {} {} {} {} {} \n'.format(Temp,_uPS,Fvals[0],Fvals[1],prefac*Fvals[2],prefac*Fvals[3],prefac*Fvals[4],prefac*Fvals[5]))
				Coexpts.flush()
	
	

	
