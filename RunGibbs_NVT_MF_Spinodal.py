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
jobtype = 'MF' # CL,SCFT,MF
phaseBoundary = 'spinodal'
ensemble = 'NVT'
number_species = 2  
GM = Gibbs_Module.Gibbs_System(program,number_species)
GM.SetJobType(jobtype)
GM.SetRunTemplate('template.in',['__CTOT__','__Phi1__','___Phi2__'])
GM.SetDt([0.01,0.02,0.02])
GM.SetEnsemble(ensemble)
GM.SetPhaseBoundary(phaseBoundary)
# For MF need to specify the interactions. Currently only for two-species: uPP,uSS,uPS
# 	Values are in polyFTS units. 
# TODO: Turn into a matrix, make MF model general.
GM.SetInteractions([423.6405479,50.95952805,173.502])
GM.SetInteractionRange([2.708173622,2.253219669,2.491178966])
GM.SetUseRPA(False) #RPA unstable inside spinodal, default to false

def uPS(Temp):
	''' Polymer-Solvent Interaction Parameter as a function of temperature. '''
	# Returns _uPS in polyFTS units
	a0 = 0.23087
	a1 = 18.67411
	a2 = -7642.03
	a3 = 533942.9
	aPS = 3.45 # angstrom
	L   = 1.384688 #angstroms - polyFTS length Scale

	_Temp = Temp + 273 # convert to Kelvin
	_uPS = a0 + a1/_Temp + a2/_Temp/_Temp + a3/_Temp**3
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

def MinTemp(_Temp,_uPSTarget):
	''' Function to backout Temp given a target uPS '''
	
	_Obj = np.abs(_uPSTarget - uPS(_Temp))
	
	return _Obj
	
	
''' Set total species number and total volume. '''
_xfrac = 0.1 # mole fraction of 0.0909
CTot = CalcC(_xfrac) # number concentration
NumberFrac_Species = [0.1,0.9]
NumberDens_Species = [CTot*x for x in NumberFrac_Species]
GM.SetSpeciesCTotal(NumberDens_Species)
GM.SetSpeciesDOP([20,1]) # the degree of polymerization, topology doesn't matter. 

''' Set the initial guesses for the coexistence pts of BoxI. '''
# auto-converts internally to either polyFTS or MD suitable parameters
fI  = 0.50
CI1 = 0.10*CTot
CI2 = 0.90*CTot
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

if jobtype == "MF" and phaseBoundary == 'binodal':
	Tolerance = 1e-7
	MF_MaxIterations = 1e6
	prefac = 1000./1.384688**3 # conversion to monomers/nm**3
	print(prefac)
	MW_List = [6]#[10,20]
	Xfrac_List = [0.10]#[0.02,0.04,0.06,0.08,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60]
	
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
			CI1 = 0.09*CTot
			CI2 = 0.91*CTot
			# calculate BoxII pts
			fII = 1.-fI
			CII1 = (NumberDens_Species[0]-CI1*fI)/fII
			CII2 = (NumberDens_Species[1]-CI2*fI)/fII

			VarInit = [fI,fII,CI1,CII1,CI2,CII2]
			GM.SetInitialGuess(VarInit)
						
			
		
			GM.SetDt([0.01*1,0.1*1/N,0.1*1/N])
			Temp_List = np.linspace(130,150,200)
			GM.Write2Log("DOP:   {}\n".format(N))
			GM.Write2Log("Xfrac: {}\n".format(_xfrac))
			GM.Write2Log('CTot:  {}\n'.format(CTot))
			GM.SetSpeciesDOP([N,1])
			Coexpts = open('Coexpts_N_{0:2d}_xfrac_{1:2.2f}.data'.format(N,_xfrac),'w')
			Coexpts.write("# T  uPS  fI  fII  CI_1  CII_1  CI_2  CII_2\n")
			for i,Temp in enumerate(Temp_List): # Generate Temperature-Rho Phase Diagram
				GM.Write2Log("Iter: {} Temp.: {}\n".format(i,Temp))
				_uPS = uPS(Temp)
				GM.SetInteractions([423.6405479,50.95952805,_uPS])
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
	
''' Begin Running Spinodal Calculations. '''
phaseBoundary = 'spinodal'
GM.SetPhaseBoundary(phaseBoundary)

# conversion factors
Mw1 = 44.052 # g/mole
Mw2 = 18.016 # g/mole
Nav = 6.022E23 # molecules/mole

prefac1 = Mw1/Nav 
prefac2 = Mw2/Nav

if jobtype == "MF" and phaseBoundary == 'spinodal':
	FindZero = True # use sp.optimize.minimize to find where eigenvalue goes to 0 precisely
	GM.FindEigenValuesNumerically = False # Use linalg to find eigenvalues numerically of stability matrix
	Tolerance = 1e-6
	MF_MaxIterations = 5e5
	prefac = 1000./1.384688**3 # conversion to monomers/nm**3
	print(prefac)
	MW_List = [1,2,3,4,5,6,10]
	Xfrac_List = [0.00001,0.000025,0.00005,0.000075,0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009]
	xfrac_temp = np.linspace(0.01,0.9,540).tolist()
	Xfrac_List.extend(xfrac_temp)
	
	TotalWeightFracMatrix = np.zeros((len(Xfrac_List),int(3*len(MW_List))))
	headerSummary = "# "
	for i,N in enumerate(MW_List): # Loop over molecular weight
		
		Coexpts = open('Spinodal_N_{}.data'.format(str(N).zfill(4)),'w')
		Coexpts.write("# T  uPS minEig fI  fII  CI_1  CII_1  CI_2  CII_2\n")
		
		for j,_xfrac in enumerate(Xfrac_List): # loop over c_total
			#print("XFRACTION: {}".format(_xfrac))
			''' Set total species number and total volume. '''
			CTot = CalcC(_xfrac) # number concentration
			NumberFrac_Species = [_xfrac,(1.-_xfrac)]
			NumberDens_Species = [CTot*x for x in NumberFrac_Species]
			GM.SetSpeciesCTotal(NumberDens_Species)
			GM.SetSpeciesDOP([N,1]) # the degree of polymerization, topology doesn't matter. 

			''' Set the initial guesses for the coexistence pts of BoxI. '''
			# auto-converts internally to either polyFTS or MD suitable parameters
			fI  = 0.50
			CI1 = _xfrac*CTot
			CI2 = (1.-_xfrac)*CTot
			# calculate BoxII pts
			fII = 1.-fI
			CII1 = (NumberDens_Species[0]-CI1*fI)/fII
			CII2 = (NumberDens_Species[1]-CI2*fI)/fII
			wtfracI1 = CI1*prefac1/(CI1*prefac1+CI2*prefac2)
			wtfracI2 = CI2*prefac2/(CI1*prefac1+CI2*prefac2)
			
			VarInit = [fI,fII,CI1,CII1,CI2,CII2]
			GM.SetInitialGuess(VarInit)
							
			GM.SetDt([0.0025*1/N,0.01*1/N,0.01*1/N])
			Temp_List = np.linspace(50,650,1200)
			GM.Write2Log("DOP:   {}\n".format(N))
			GM.Write2Log("Xfrac: {}\n".format(_xfrac))
			GM.Write2Log('CTot:  {}\n'.format(CTot))
			
			
			''' Find lower spinodal '''
			SpinodalLoc = False
			MinEigVal = 0.
			MinEigValList = []
			InteractionList = []
			LowerSpinodal = []
			GM.Iteration = 1
			for Tindex,Temp in enumerate(Temp_List): # Generate Temperature-Rho Phase Diagram

				
				#GM.Write2Log("Iter: {} Temp.: {}\n".format(i,Temp))
				_uPS = uPS(Temp)
				#print(Temp)
				GM.SetInteractions([423.6405479,50.95952805,_uPS])
				GM.DvalsCurrent = [1.]			
					
				GM.TakeGibbsStep()
				EigVals = GM.EigenValues
				MinEigVal = GM.MinEigVal
				#print(MinEigVal)
				MinEigValList.append(MinEigVal)
				InteractionList.append(GM.Interactions)
				
				#print(MinEigValList)
				if MinEigValList[-1] < 0. and len(MinEigValList) > 1 and MinEigValList[-2] >= 0. and SpinodalLoc == False:
					GM.Write2Log("Hit Spinodal!\n")
					SpinodalLoc = True
						
				#if MinEigValList[-1] >= 0. and len(MinEigValList) > 1 and MinEigValList[-2] < 0. and SpinodalLoc == False:
				#	GM.Write2Log("Hit Spinodal!\n")
				#	SpinodalLoc = True
				
				if SpinodalLoc:
					print('Iteration {}'.format(i))
					print('xFrac: {}'.format(_xfrac))
					print('Writing T: {}'.format(Temp))
					print('MinEig {}'.format(MinEigVal))
					SpinodalLoc = False
					
					if FindZero:
						for _i,_val in enumerate(EigVals):
							if _val == GM.MinEigVal:
								_LambdaIndex = _i
						_indexes = [2]
						_x = GM.Interactions[_indexes[0]]
						_opt = sp.optimize.minimize(GM.MinimizeEigenValue,_x,args=(_indexes,_LambdaIndex,CI1,CI2,N,1,GM.Interactions,GM.InteractionRange),tol=1e-5,method='Nelder-Mead' )
						print('Minimization _uPS: {}'.format(_opt.x))
						print('Success: {}'.format(_opt.success))
						print('Termination: {}'.format(_opt.status))
						print('Description: {}'.format(_opt.message))
						print('Nfev: {}'.format(_opt.nfev))
						#print('Tol: {}'.format(_opt.maxcv))
						_uPSAvg = _opt.x[0]
						MinEigVal = np.min(GM.MFStabilityEigenValues(CI1,CI2,N,1,GM.Interactions,GM.InteractionRange))
						
						# Backout Temperature
						_opt = sp.optimize.minimize(MinTemp,Temp,args=(_uPSAvg),tol=1e-7,method='Nelder-Mead' )
						TempAvg = _opt.x[0]
					else:					
						_uPSAvg = (InteractionList[-1][2]+InteractionList[-2][2])/2.					
						TempAvg = (Temp + Temp_List[i-1])/2.
						
					Fvals = GM.ValuesCurrent
					GM.Iteration = 1 # reset iteration
					
					Coexpts.write('{} {} {} {} {} {} {} {} {} \n'.format(TempAvg,_uPSAvg,MinEigVal,Fvals[0],Fvals[1],prefac*CI1,wtfracI1,wtfracI2,prefac*CII2))
					Coexpts.flush()
					TotalWeightFracMatrix[j,i*3] = wtfracI1
					TotalWeightFracMatrix[j,i*3+1] = TempAvg
					InteractionList = []
					MinEigValList = []
					break
			
			''' Find upper spinodal '''
			SpinodalLoc = False
			MinEigVal = 0.
			MinEigValList = []
			InteractionList = []
			UpperSpinodal = []
			for Tindex,Temp in enumerate(Temp_List): # Generate Temperature-Rho Phase Diagram
				
				#GM.Write2Log("Iter: {} Temp.: {}\n".format(i,Temp))
				_uPS = uPS(Temp)
				#print(Temp)
				GM.SetInteractions([423.6405479,50.95952805,_uPS])
				GM.DvalsCurrent = [1.]			
					
				GM.TakeGibbsStep()
				EigVals = GM.EigenValues
				MinEigVal = GM.MinEigVal
				#print(MinEigVal)
				MinEigValList.append(MinEigVal)
				InteractionList.append(GM.Interactions)
				
				#if MinEigValList[-1] < 0. and len(MinEigValList) > 1 and MinEigValList[-2] >= 0. and SpinodalLoc == False:
				#	GM.Write2Log("Hit Spinodal!\n")
				#	SpinodalLoc = True
					
				if MinEigValList[-1] >= 0. and len(MinEigValList) > 1 and MinEigValList[-2] < 0. and SpinodalLoc == False:
					GM.Write2Log("Hit Spinodal!\n")
					SpinodalLoc = True
				
				if SpinodalLoc:
					print('Iteration {}'.format(i))
					print('xFrac: {}'.format(_xfrac))
					print('Writing T: {}'.format(Temp))
					print('MinEig {}'.format(MinEigVal))
					SpinodalLoc = False
					
					if FindZero:
						for _i,_val in enumerate(EigVals):
							if _val == GM.MinEigVal:
								_LambdaIndex = _i
						_indexes = [2]
						_x = GM.Interactions[_indexes[0]]
						_opt = sp.optimize.minimize(GM.MinimizeEigenValue,_x,args=(_indexes,_LambdaIndex,CI1,CI2,N,1,GM.Interactions,GM.InteractionRange),tol=1e-5,method='Nelder-Mead' )
						print('Minimization _uPS: {}'.format(_opt.x))
						print('Success: {}'.format(_opt.success))
						print('Termination: {}'.format(_opt.status))
						print('Description: {}'.format(_opt.message))
						print('Nfev: {}'.format(_opt.nfev))
						#print('Tol: {}'.format(_opt.maxcv))
						_uPSAvg = _opt.x[0]
						MinEigVal = np.min(GM.MFStabilityEigenValues(CI1,CI2,N,1,GM.Interactions,GM.InteractionRange))
						
						# Backout temperature
						_opt = sp.optimize.minimize(MinTemp,Temp,args=(_uPSAvg),tol=1e-7,method='Nelder-Mead' )
						TempAvg = _opt.x[0]
					
					else:					
						_uPSAvg = (InteractionList[-1][2]+InteractionList[-2][2])/2.
						TempAvg = (Temp + Temp_List[i-1])/2.
					
					#print("TEMPERATURE")
					#print(Temp)
					#print(Temp_List[i-1])
					#print(Temp_List[i-2])
					
					
					Fvals = GM.ValuesCurrent
					GM.Iteration = 1 # reset iteration
					
					Coexpts.write('{} {} {} {} {} {} {} {} {} \n'.format(TempAvg,_uPSAvg,MinEigVal,Fvals[0],Fvals[1],prefac*CI1,wtfracI1,wtfracI2,prefac*CII2))
					Coexpts.flush()
					TotalWeightFracMatrix[j,i*3+2] = TempAvg
					InteractionList = []
					MinEigValList = []
					break

		headerSummary += " wI_1_N_{}  LowT  UpperT ".format(N)
	
	np.savetxt('WeightFracSummary.data',TotalWeightFracMatrix,header=headerSummary)
	
