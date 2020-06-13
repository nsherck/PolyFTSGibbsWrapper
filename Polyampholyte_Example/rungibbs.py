import time, re
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
import os, sys
from shutil import copyfile
import ast
sys.path.append('/home/mnguyen/bin/scripts/')
sys.path.append('../')
import Gibbs_V4 as Gibbs_Module

#Script assumes polyFTS returns operators: Hamiltonian, Pressure, and Species Chemical Potentials

''' Initialize instance of Gibbs class. '''
program = 'polyFTS' # or 'MD'
jobtype = 'CL' # CL,SCFT,MF
ensemble = 'NVT'
number_species = 3 
GM = Gibbs_Module.Gibbs_System(program,number_species)
GM.SetJobType(jobtype)
GM.SetEnsemble(ensemble)
GM.SetRunTemplate('template.in',[['__C1__','__PhiP1__','__PhiCIA1__','__PhiCIC1__'],['__C2__','__PhiP2__','__PhiCIA2__','__PhiCIC2__']],TwoModelTemplate=True)
GM.SetDt([0.01,0.1,0.1,0.1])

# specific for charged system
GM.SetCharges([0.1,-1,1])
GM.SetChargedPairs([[0,1],[1,2]])
GM.SetDtCpair([0.1,0.1])
#GM.SetDtCMax(0.1)

if ensemble == "NPT":
	GM.TargetP = 0.2416177877


''' Set total species number and total volume. '''
C1 = 0.15
C2 = 0.0825
C3 = 0.0675
GM.SetSpeciesCTotal([C1,C2,C3])

''' Set the initial guesses for the coexistence pts of BoxI. '''
# auto-converts internally to either polyFTS or MD suitable parameters
fI  = 0.51
CI1 = 0.008
CI2 = 0.075+0.008*0.1
CI3 = 0.075
# calculate BoxII pts
fII = 1.-fI
CII1 = (C1-CI1*fI)/fII
CII2 = (C2-CI2*fI)/fII
CII3 = (C3-CI3*fI)/fII
VarInit = [fI,fII,CI1,CII1,CI2,CII2,CI3,CII3]

nstepsEquil	=200 # Number CL steps to run for
nstepsProd	=20

#initialize
GM.DvalsCurrent = [1.] * (GM.Nspecies+2) # initialize Dvals	
GM.SetUseOneNode(True)

GM.SetNPolyFTSBlocksInit(500)
GM.SetNPolyFTSBlocks(200)
GM.SetNPolyFTSBlocksMin(100)
GM.SetNPolyFTSBlocksMax(200)
GM.SetOperatorRelTol(0.001)
GM.SetVolFracBounds([0.1,0.9])
GM.GibbsLogFileName = 'gibbs_CL.dat'
GM.GibbsErrorFileName = 'error_CL.dat'	
GM.Iteration = 1
GM.SetInitialGuess(VarInit)
GM.ValuesCurrent = VarInit

print("="*5,'Equilibration','='*5)
for step in range(nstepsEquil):
    print('== step {}'.format(step+1))
    print('fI fII\n{}'.format(GM.ValuesCurrent[0:2]))
    print ("CtotI phi1I phi2I phi3I CtotII phi1II phi2II phi3II\n{}".format(GM.GetPolyFTSParameters()))
    GM.TakeGibbsStep()
    print('dts\n{}'.format(GM.Dt[1:]))
    print('\nCpair in box 1\n{}'.format(GM.Cpair1))
    print("Step: {0} RunTime: {1:3.3e} min.".format(step,GM.StepRunTime))
GM.SetNPolyFTSBlocks(400)
GM.SetNPolyFTSBlocksMin(100)
GM.SetNPolyFTSBlocksMax(500)
for step in range(nstepsProd): # run CL simulation
    GM.TakeGibbsStep()
    print("Step: {0} RunTime: {1:3.3e} min.".format(step,GM.StepRunTime))
Fvals = GM.ValuesCurrent
print('ValuesCurrent: {}'.format())
