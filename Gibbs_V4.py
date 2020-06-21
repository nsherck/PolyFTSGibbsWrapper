#!/usr/bin/env python

# Scott Danielsen, sdanielsen@mrl.ucsb.edu, 11/30/2017, (Joint chemical potentials for charge neutrality)

import os
import sys
import re
import subprocess
import numpy as np
import math
import stats
import subprocess as prcs
from subprocess import call
import time
from shutil import copyfile, rmtree
import copy



class Gibbs_System():
    ''' Methods and tools for calculating phase behavior using the Gibb's Ensemble approach. '''
    
    def __init__(self,_Program,_Nspecies=1):
        ''' 
            Program: 
                (1) polyFTS - generally applicable to systems with macromolecules 
                (2) MD      - can be used for small molecules 
                
            Nspecies_:
                (1) integer that specifies the number of species to track, default 1
            
            Requirements/Assumptions:
                Polymers are listed first
                Any number of charged species
                ChargedPairs: specify N-1 pairs if have N charged species
                Pair that is made up of cation (+) with valency z+ and anion (-) with valency z- has the formula:
                    z+ (-) + z- (-)
                A pair is made up of exactly two charged species
                A charged species involves in no more than two pairs
                    
                
                
        '''
        
        self.Program             = _Program
        self.Nspecies           = _Nspecies
        self.SpeciesMoleFrac    = []
        self.CTotal                = float(1.)
        self.Ensemble            = 'NVT' # whether to run Gibbs in NVT, NPT, etc.
        self.Barostat            = '' # whether to barostat the pressure, good for troubleshooting
        self.TargetP            = 1.             
        self.JobType            = '' 
        self.SpeciesDOP             = []
        self.SpeciesCTotal         = []
        self.VarsInit            = []
        self.ValuesCurrent        = [] # list ==current values of the fI,CIi
        self.ValuesLast            = [] # the last steps values of the fI,CIi 
        self.OperatorsCurrent    = [] 
        self.OperatorsLast        = [] 
        self.DvalsCurrent        = [1.]
        self.DvalsLast            = []
        self.SaveFolders         = True # default by true
        self.Iteration             = 1 # initial iteration is 1
        self.RunTemplateName    = '' # the program template run file
        self.RunTemplateVars    = [] # variable names to replace in run template
        self.TwoModelTemplate   = False # True if using template to run two models simultaneously
        self.RunTemplateVarsBox1    = [] # variable names to replace in run template for box 1
        self.RunTemplateVarsBox2    = [] # variable names to replace in run template for box 2
        self.GibbsLogFile       = ''
        self.GibbsLogFileName   = 'gibbs.dat'
        self.GibbsErrorFile     = ''
        self.GibbsErrorFileName = 'error.dat'
        self.Converged             = False
        self.Dt                    = [0.1,0.1,0.1] #[volume frac, concentrations of charged pairs (# = NchargedSpecies-1), concentrations of neutral species]
        self.InteractionRange   = []
        self.Interactions         = []
        self.UseRPA                = False
        self.NPolyFTSBlocksInit = 500 # number of blocks in 1st iteration
        self.NPolyFTSBlocks     = 200
        self.NPolyFTSBlocksmin  = 200
        self.NPolyFTSBlocksmax  = 10000
        self.OperatorRelTol        = 0.005
        self.UseOneNode            = False
        self.PolyFTSExec         = '~/bin/PolyFTS20200605/bin/Release/PolyFTSPLL.x'
        self.VolFracBounds         = [0.1,0.9] # lower,upper
        self.StepRunTime         = -1. # amount of time to run simulations
        self.PSInteraction        = -999
        self.ReRun                = False # Boolean that is True if program should rerun the current Gibbs step before moving on
        self.UseReRun            = False # Turn on/off ReRun capability
        self.ReRunHist          = [] # list of reruns
        self.LogFileName        = 'Gibbs.log'
        self.LogFile            = None
        self.Break                = False
 
        # charged parameters
        self.NchargedSpecies   = 0
        self.NeffSpecies       =  _Nspecies # number of "effective" species (ChargedPairs + neutral species)
        self.Charges           = np.zeros(_Nspecies) # charges of all species, for polymer, this is the average charge/monomer 
        self.Valencies         = np.zeros(_Nspecies) # valencies of all species
        self.ChargedSpecies    = [] # indices of charged species
        self.NeutralSpecies    = range(_Nspecies) # indices of neutral species
        self.ChargedPairs      = [] # pairs of indices of species that make up effective neutral species
        self.Cpair1             = [] # concentrations of charged pairs in box 1
        self.Cpair2             = []
        self.MuPair1             = []
        self.MuPair1Err          = []
        self.MuPair2             = []
        self.MuPair2Err          = []
        self.dMuPair            = []
        self.DtCpair            = [] # time step to update pair concentration

        self.DtCMax             = None # maximum Dt to update the concentration
        self.UseAdaptiveDtC     = False
        self.SetLogFile(self.LogFileName) # initialize LogFile
        
        
    def SetChargedSpecies(self):
        ''' Set indices and number of charged species and indices of neutral species'''
        self.ChargedSpecies = np.where(self.Charges != 0.)[0]
        self.NeutralSpecies = np.where(self.Charges == 0.)[0]
        self.NchargedSpecies = len(self.ChargedSpecies)
        
    def SetCharges(self,_Charges):
        ''' Set charges of all species in the system '''
        if len(_Charges) != self.Nspecies:
            raise Exception('Must provide charges to all species')
        self.Charges = np.array(_Charges)
        self.Valencies = np.abs(self.Charges)
        return self.SetChargedSpecies()
       
    def SetChargedPairs(self,_ChargedPairs):
        ''' Set pairs of species that make up effective neutral species 
            and set the effective number of species'''

        if len(_ChargedPairs) != self.NchargedSpecies - 1:
            raise Exception('Number of charged pairs must be number of charged species - 1')
            
        for i1,i2 in _ChargedPairs:
            if self.Charges[i1] * self.Charges[i2] >= 0.:
                raise Exception('Species in a ChargedPair must be of opposite charges')
        self.ChargedPairs = _ChargedPairs
        self.NeffSpecies = int(len(self.ChargedPairs) + (self.Nspecies - self.NchargedSpecies))

    def GetTotalCharge(self, Box):
        '''Get total charge in each box'''
        q = 0
        for i in range(self.Nspecies):
            qi = self.Charges[i]
            Ci = self.ValuesCurrent[2*(i+1)+(Box-1)]
            q += qi*Ci
        return q

    def CheckTotalCharge(self, ChargeTol = 0.0005):
        '''Check for electroneutrality'''
        for Box in [1,2]:
            q = self.GetTotalCharge(Box)
            self.Write2Log('Total charge in box {}: {}\n'.format(Box,q))
    
    def CheckTotalSpeciesC(self, ConcTol = 0.01):
        for i in range(self.Nspecies):
            Ci_tot = self.ValuesCurrent[0]*self.ValuesCurrent[(i+1)*2] + self.ValuesCurrent[1]*self.ValuesCurrent[(i+1)*2+1]
            RelErr = np.abs((self.SpeciesCTotal[i]-Ci_tot)/self.SpeciesCTotal[i])
            if RelErr > ConcTol:
                self.Write2Log('\nCalculated total concentration of species {} is off from SpeciesCTotal by a relative error of {}'.format(i+1,RelErr))        
       
    def GetEffMu(self, Pair, Mu1, Mu2, Mu1Err, Mu2Err):
        ''' Get the combined chemical potential of a charged pair'''
        i1,i2 = Pair
        z1, z2 = [self.Valencies[i1],self.Valencies[i2]]
        Mu = z2 * Mu1 + z1 * Mu2
        MuErr = np.sqrt((z2*Mu1Err)**2 + (z1*Mu2Err)**2)
        return Mu, MuErr
    
    def UpdateMuPair(self):
        ''' Update the chemical potentials of charged pairs in two boxes and their differences'''
        for Box in [1,2]:
            MuPair = []
            MuPairErr = []
            for indx,[i,j] in enumerate(self.ChargedPairs):
                Mui = self.OperatorsCurrent[8+i*4+(Box-1)*2]
                Muj = self.OperatorsCurrent[8+j*4+(Box-1)*2]
                MuiErr = self.OperatorsCurrent[9+i*4+(Box-1)*2]
                MujErr = self.OperatorsCurrent[9+j*4+(Box-1)*2] 
                Mu, MuErr = self.GetEffMu([i,j],Mui,Muj,MuiErr,MujErr)
                MuPair.append(Mu)
                MuPairErr.append(MuErr)
            if Box == 1:
                self.MuPair1 = np.array(MuPair)
                self.MuPair1Err = np.array(MuPairErr)
            elif Box == 2:
                self.MuPair2 = np.array(MuPair)
                self.MuPair2Err = np.array(MuPairErr)
        self.dMuPair =  self.MuPair1 - self.MuPair2  
        
    def GetEffC(self, Box):
        ''' Get the concentration of a charged pair
            Box: 1 for box 1, 2 for box 2 '''        
        
        Cpairs = np.zeros(len(self.ChargedPairs))        
        PairIds = [] # indices of pairs that involve species i
        zjs = [] # number of species i in pairs that it involves
        for i in self.ChargedSpecies:
            PairId = []
            zj = []
            for indx, Pair in enumerate(self.ChargedPairs):
                    if i in Pair:
                        PairId.append(indx)
                        j = int(np.array(Pair)[np.where(np.array(Pair) != i)[0]])
                        zj.append(self.Valencies[j])
            PairIds.append(PairId)
            zjs.append(zj)
            if len(PairId) == 1: # this species only involves in one pair, so its concentration directly gives the pair concentration
                Cpair = self.ValuesCurrent[2*(i+1)+(Box-1)]/zj[0]
                if Cpairs[PairId[0]] != 0.: # This pair concentration has alredy been calculated, check if consistent with prev. value
                    relErr = np.abs((Cpair-Cpairs[PairId[0]])/Cpairs[PairId[0]])
                    if relErr >= 5.0e-3:
                        self.Write2Log('\nInconsistent values for concentration of pair {} in box {}'.format(PairId[0],Box))
                        raise Exception('Inconsistent values for concentration of pair {} in box {}'.format(PairId[0],Box))
                else:
                    Cpairs[PairId[0]] = Cpair        
        # calculate concentration of pairs consisting of 2 species in more than one pair, assume no species involve in more than 2 pairs
        for  i,ii in enumerate(self.ChargedSpecies):
            for k,indx in enumerate(PairIds[i]):
                if Cpairs[indx] == 0.: 
                    otherPairIds = np.array(PairIds[i])[np.where(np.array(PairIds[i]) != indx)[0]]
                    otherCpairs = Cpairs[otherPairIds]
                    otherPairZjs = np.array(zjs[i])[np.where(np.array(PairIds[i]) != indx)[0]]
                    Ci_in_other = np.multiply(otherCpairs,otherPairZjs)
                    Ci = self.ValuesCurrent[2*(i+1)+(Box-1)]
                    Cpair = (Ci - np.sum(Ci_in_other))/zjs[i][k]
                    Cpairs[indx] = Cpair
        return Cpairs
    
    def UpdateCpair(self):
        '''Update concentrations of all charged pairs in two boxes'''
        self.Cpair1 = self.GetEffC(1)
        self.Cpair2 = self.GetEffC(2)
    
    def GetChargedC(self, i):
        ''' Get the concentration of an individual charged species 
            i: index of charged species i'''                        
        C = 0
        for indx, Pair in enumerate(self.ChargedPairs):
            if i in Pair:
                j = int(np.array(Pair)[np.where(np.array(Pair) != i)[0]])
                zj = self.Valencies[j] # valency of the j species in pair
                Cpair = self.Cpair1[indx]
                C += zj * Cpair
        return C
    
    def GetChargedDMu(self,i):
        ''' Get the driving force (dMu) of an individual charged species 
            by summing over driving force of charged pairs that have species i
            i: index of charged species i'''
        dMu = 0
        for indx, Pair in enumerate(self.ChargedPairs):
            if i in Pair:
                j = int(np.array(Pair)[np.where(np.array(Pair) != i)[0]])
                zj = self.Valencies[j] # valency of the j species in pair
                dMu_pair = self.dMuPair[indx]
                dMu += zj * dMu_pair 
        return dMu
    
    def GetDtBounds(self, i, Ci_high = None, Ci_low = None):
        ''' Get upper and lower bounds of time stepper
            i : index of the species
            Ci_low, Ci_high: upper and lower bounds of Ci1 in the next time step
            Currently, concentrations of all charged pairs are updating with the same time step
            '''
        CTotal = self.SpeciesCTotal[i]
        Ci1 = self.ValuesCurrent[(i+1)*2]

        if self.Valencies[i] > 0:
            dMui = self.GetChargedDMu(i)
        else:
            dMui = self.DvalsCurrent[i+2] 

        #default bounds
        if Ci_high == None or Ci_high > CTotal:
            Ci_high = CTotal
        if Ci_low == None or Ci_low < 0:
            Ci_low = 1.e-10       
        Dt_low = np.abs((Ci1 - Ci_high/self.ValuesCurrent[0]) / dMui)
        Dt_high = np.abs((Ci1 - Ci_low/self.ValuesCurrent[0]) / dMui)
        self.Write2Log('\n{} {} {}'.format(i+1,Dt_low,Dt_high))
        return Dt_low, Dt_high
    
    def SetDtCMax(self,_DtCMax):
        ''' Set Maximum time step to update concentration '''
        self.DtCMax = _DtCMax
        
    def UpdateChargedDtC(self):
        '''Update Dt for concentration update using adaptive time step
           Currently, concentrations of all charged pairs are updating with the same time step'''        
        self.Write2Log(' ==get DtC bounds==\n species bound1 bound2')
        Dt_bound_charged = []
        Dt = np.array(self.Dt)
        DtCMax = 10.
        if self.DtCMax:
            Dt_bound_charged = [self.DtCMax]  
            DtCMax = self.DtCMax
            
        for i in range(self.Nspecies):
            if i in self.ChargedSpecies:
                Dt_low, Dt_high = self.GetDtBounds(i)
                Dt_bound_charged.extend([Dt_low, Dt_high])
            elif i in self.NeutralSpecies:    
                Dt_low, Dt_high = self.GetDtBounds(i)
                Dt[i+1] = min(Dt_low, Dt_high, DtCMax)            
        Dt_charged = np.min(Dt_bound_charged)
        self.SetDtCpair([Dt_charged] * len(self.ChargedPairs))
        self.SetDt(Dt)
            
    def SetLogFile(self,_LogFileName):
        ''' Create Log File For the Run '''
        try:    
            self.LogFile = open(str(_LogFileName),'w')
        except:
            pass
            
            self.LogFile.close()
            
    def Write2Log(self,_text):
        ''' Write out 2 Log file '''
        self.LogFile = open(self.LogFileName,'a')
        self.LogFile.write(str(_text))
        self.LogFile.close()
        
    def SetJobType(self,_JobType):
        ''' The Jobtype to run.
            
            program == polyFTS:
                (1) CL
                (2) SCFT
                (3) MF - an analytical SCFT model 
        
        '''
        self.JobType = str(_JobType)
        
    def SetEnsemble(self,_Ensemble):
        ''' The Ensemble to Use.
            
            program == polyFTS:
                (1) NVT
                (2) NPT
                (3) semi-grand, etc 
        
        '''
        self.Ensemble = str(_Ensemble)
        
    def SetSpeciesCTotal(self,_SpeciesCTotal):
        ''' A list of the number concentration of each species. '''
        self.SpeciesCTotal = _SpeciesCTotal
        if self.Nspecies != len(_SpeciesCTotal):
            self.Write2Log('WARNING: The number of species does not match the length of the list of number densities!\n')
        self.Nspecies = len(_SpeciesCTotal)
        self.CTotal = np.sum(_SpeciesCTotal) # important for NPT
        self.SpeciesMoleFrac = _SpeciesCTotal/self.CTotal # important for NPT
        
    def SetDt(self,_Dt):
        ''' A list of the Gibbs updates. '''
        self.Dt = _Dt

    def SetDtCpair(self,_Dt):
        ''' A list of the Gibbs updates. '''
        self.DtCpair = _Dt
    
    def SetVolFracBounds(self,_VolFracBounds):
        ''' Sets the minimum and upper bounds on the volume fraction, can be useful to keep simulation in stable region at outset. '''
        self.VolFracBounds = list(_VolFracBounds)
    
    def SetUseOneNode(self,_UseOneNode):
        ''' Whether to run on a single node, or to submit to multiple nodes. '''
        self.UseOneNode = bool(_UseOneNode)
    
    def SetNPolyFTSBlocksInit(self,_NPolyFTSBlocksInit):
        ''' Sets the number of polyFTS blocks to run in the first iteration '''
        self.NPolyFTSBlocksInit = int(_NPolyFTSBlocksInit)
        
    def SetNPolyFTSBlocks(self,_NPolyFTSBlocks):
        ''' Sets the number of polyFTS blocks to run. '''
        self.NPolyFTSBlocks = int(_NPolyFTSBlocks)
        
    def SetNPolyFTSBlocksMin(self,_NPolyFTSBlocksMin):
        ''' Sets the minimum number of polyFTS blocks. '''
        self.NPolyFTSBlocksmin = int(_NPolyFTSBlocksMin)
        
    def SetNPolyFTSBlocksMax(self,_NPolyFTSBlocksMax):
        ''' Sets the maximum number of polyFTS blocks. '''
        self.NPolyFTSBlocksmax = int(_NPolyFTSBlocksMax)
        
    def SetOperatorRelTol(self,_OperatorRelTol):
        ''' Sets the Operator Relative Tolerance (i.e. stderr/|value|).
            If greater, slows down by (stderr/|value|/RelTol)**2, if 1/3 speeds up by 0.5.
            Default set to 0.005.
        '''
        self.OperatorRelTol = float(_OperatorRelTol)
    def SetUseRPA(self,_UseRPA):
        ''' Whether to include RPA correction, unstable inside spinodal. '''
        self.UseRPA = _UseRPA

    def SetInteractions(self,_Interactions):
        ''' A list of the Gaussian Interactions. '''
        self.Interactions = _Interactions
        
    def SetInteractionRange(self,_InteractionRange):
        ''' A list of the Range of the Gaussian Interactions. '''
        self.InteractionRange = _InteractionRange
    
    def SetSpeciesDOP(self,_SpeciesDOP):
        ''' A list of the DOP of each species. '''
        self.SpeciesDOP = _SpeciesDOP
        
    def SetInitialGuess(self,_InitValues):
        ''' Initial Guess for the phase coexsistence. 
                fI - volume fraction of phase ID
                CIi - number density for each species in phase I
                [fI,CI1,CI2,...,CIns]
        '''
        self.VarsInit = copy.deepcopy(_InitValues)
        
    def SetRunTemplate(self,_RunTemplateName,_DummyVariables,TwoModelTemplate=False):
        ''' Run Template name and replacement variables. '''
        self.RunTemplateName = str(_RunTemplateName)
        if not TwoModelTemplate:
            self.RunTemplateVars = _DummyVariables
        else:
            self.RunTemplateVarsBox1 = _DummyVariables[0]
            self.RunTemplateVarsBox2 = _DummyVariables[1]
            self.TwoModelTemplate = True

    def RunJob(self,RunPath_,SubmitFilePath_):
        ''' submit jobs to the queue '''
        os.chdir(RunPath_)

        if self.UseOneNode:
            call_1 = "{} run.in > run.out".format(self.PolyFTSExec)    
        else:
            call_1 = "qsub {}".format(SubmitFilePath_)
        
        p1 = prcs.Popen(call_1, stdout=prcs.PIPE, shell=True)    
        (output, err) = p1.communicate()
        p_status = p1.wait()
            
        with open(os.path.join(RunPath_,"cgsweep_submit.log"),'w') as logout:
            logout.write(output.decode("utf-8"))
        
        ID = output.decode("utf-8")
        
        os.chdir('..')
        
        return p1,ID
    
    def CheckQ(self,JobIDs,_RunPaths):
        ''' Checks the cluster queue 2 see if job finished. Currently only tested with Knot.'''
        tStart = time.time()
        
        fn = 'logQ.out'
        with open(fn,'w') as g:
            g.write('')
        
        if self.UseOneNode:
            status = []
            temp = [1]
            while sum(temp) > 0:
                temp = []
                for rpath in enumerate(_RunPaths):
                    with open(os.path.join(rpath[1],'run.out')) as f:
                        if 'TOTAL Runtime:' in f.read():
                            temp.append(0)
                        else:
                            temp.append(1)

                if self.JobType == 'SCFT':
                    time.sleep(1)
                else:
                    time.sleep(20)
                    
        else:        
            status = ['r']
            while 'r' in status or 'q' in status:
                status = []
                time.sleep(20) # check every 20 seconds
                for i,ID in enumerate(JobIDs):
                    call_1 = "qstat {}".format(ID)
                    p1 = prcs.Popen(call_1, stdout=prcs.PIPE, shell=True)    
                    (output, err) = p1.communicate()
                    p_status = p1.wait()
                    if 'Unknown' in output: # check if job found
                        status.append('')
                    elif '-' in output: # check for dashes
                        try:
                            status.append(output.replace('-','').split()[-2].lower())
                        except:
                            g.write('BREAKING!\n')
                            break
                    else: # temporary default for now
                        status.append('') 
                    with open(fn,'a') as g:
                        g.write('{}\n {}\n'.format(output,err))

        tStop = time.time()
        runTime = (tStop - tStart)/60. # in minutes
        
        return status,runTime

    def SCFTModel(self,rhoA,rhoB,N_A,N_B,Int,a_list,IncludeRPA):
        ''' SCFT Model. '''
        #A = 
        P = rhoA/N_A + rhoB/N_B + 0.5*Int[0]*rhoA**2 + 0.5*Int[1]*rhoB**2 + Int[2]*rhoA*rhoB
        MuA = np.log(rhoA/N_A) + Int[0]*rhoA*N_A + Int[2]*rhoB*N_A # self interaction and ideal chain not included 
        MuB = np.log(rhoB/N_B) + Int[1]*rhoB*N_B + Int[2]*rhoA*N_B # self interaction and thermal wavelength not included 
        
        if IncludeRPA:
        # NOTE: RPA inside the spinodal is unstable to fluctuations
            if N_B > 1:
                self.Write2Log("WARNING: RPA model only treats species B as a solvent, i.e. N_B == 1!")
            _kmin = 0.
            _kmax = 4*2*np.pi/min(a_list)
            deltak = 0.001 # the resolution for the loop integrals
            _nkgrid = int(_kmax/deltak) #
            _C = [rhoA,rhoB]
            _N = N_A
            UseCGC = False
                
            Piex,muPex,muSex  =  RPA_PS.RPA(a_list,Int,_C,_N,UseCGC,_kmin,_kmax,_nkgrid)
            
            P = P+Piex
            MuA = MuA+muPex
            MuB = MuB+muSex
        
        return P,MuA,MuB

    def GenerateRunDirectory(self):
        ''' Generates the new run directories. '''
        if not self.TwoModelTemplate:
            try:
                os.mkdir('model1')
            except:     
                #rmtree('model1')
                #os.mkdir('model1')
                pass
            
            if self.Barostat != True:
                try:
                    os.mkdir('model2')
                except:
                    #rmtree('model2')
                    #os.mkdir('model2')
                    pass
        else:
            try:
                os.mkdir('model_1_2')
            except:     
                pass
            
        if not self.UseOneNode:
            copyfile(os.path.join(os.getcwd(),'submit_template.sh'),os.path.join(os.getcwd(),'model1','submit_template.sh'))
            if not self.Barostat:                                                                                                        
                copyfile(os.path.join(os.getcwd(),'submit_template.sh'),os.path.join(os.getcwd(),'model2','submit_template.sh'))
            
    def WritePolyFTSInput(self,dummyvarnames,variables,modelpath,TwoModelTemplate=False):
        ''' Writes polyFTS run files. 
            if use'''
        with open(self.RunTemplateName,'r') as myfile:
            ini=myfile.read()
            if not TwoModelTemplate:
                for indx, var in enumerate(variables): 
                    ini=re.sub(dummyvarnames[indx],str(var),ini)
            else:
                for j, vars in enumerate(variables): 
                    varnames = dummyvarnames[j]
                    for indx, var in enumerate(vars): 
                        ini=re.sub(varnames[indx],str(var),ini)
            if self.Iteration > 1:
                ini=re.sub('__READFIELDS__','Yes',ini)
                ini=re.sub('__NUMBLOCKS__',str(self.NPolyFTSBlocks),ini)
            elif self.Iteration < 1:
                ini=re.sub('__READFIELDS__','No',ini)
                ini=re.sub('__NUMBLOCKS__',str(self.NPolyFTSBlocksInit),ini)
            else:
                ini=re.sub('__READFIELDS__','No',ini)
                ini=re.sub('__NUMBLOCKS__',str(self.NPolyFTSBlocksInit),ini)
           
            ini = re.sub('__PS__',str(self.PSInteraction),ini)
            
            runfile = open(os.path.join(modelpath,"run.in"),"w")
            runfile.write(ini)
            runfile.close()

    def WriteStats(self):
        ''' Writes out Gibbs step statistics. '''
        
        if self.Iteration == 1: # initialize file
            try:
                os.remove(self.GibbsLogFileName)
            except:
                pass
            self.GibbsLogFile = open(self.GibbsLogFileName,"w")
            header_temp = "# step  "
            header_temp += "fI  fII  "
            for i in range(self.Nspecies):
                header_temp += 'CI_{}  CII_{}  '.format(i+1,i+1)        
            header_temp += "HI  stderr  HII  stderr  PI  stderr  PII  stderr  "
            if self.NchargedSpecies > 0:
                for i in range(len(self.ChargedPairs)):
                    header_temp += 'muI_pair{}  stderr  muII_pair{}  stderr  '.format(i+1,i+1)
            for i in self.NeutralSpecies:
                header_temp += 'muI_{}  stderr  muII_{}  stderr  '.format(i+1,i+1)                    
            
            header_temp +="\n"
            self.GibbsLogFile.write(header_temp)
            
            try:
                os.remove(self.GibbsErrorFileName)
            except:
                pass
            self.GibbsErrorFile = open(self.GibbsErrorFileName,"w")
            header_temp = "#  step  dH  dP  "
            if self.NchargedSpecies > 0:
                for i in range(len(self.ChargedPairs)):
                    header_temp += 'dmuPair_{}  '.format(i+1)
            for i in self.NeutralSpecies:
                header_temp += 'dmu_{}  '.format(i+1)
            
            header_temp +="\n"
            self.GibbsErrorFile.write(header_temp)
            step = 0
        else: 
            step = self.Iteration-1
    
    
        temp = "{}  ".format(step)
        for val in self.ValuesCurrent:
            temp += "{}  ".format(val)
        for val in self.OperatorsCurrent[:8]:
            temp += "{}  ".format(val)
        if self.Iteration != 1:
            for i in range(len(self.ChargedPairs)):
                temp += "{}  {}  {}  {}  ".format(self.MuPair1[i],self.MuPair1Err[i],self.MuPair2[i],self.MuPair2Err[i])
            for i in self.NeutralSpecies:
                indx = 8 + i*4 
                temp += "{}  {}  {}  {}  ".format(self.OperatorsCurrent[indx],self.OperatorsCurrent[indx+1],self.OperatorsCurrent[indx+2],self.OperatorsCurrent[indx+3])
        temp +="\n"
        self.GibbsLogFile.write(temp)
        self.GibbsLogFile.flush()
        
        temp = "{}  ".format(step)
        for val in self.DvalsCurrent[:2]:
            temp += "{}  ".format(val)
        if self.Iteration == 1:
            self.dMuPair = [1]*len(self.ChargedPairs)
        for i in range(len(self.ChargedPairs)):
            temp += "{}  ".format(self.dMuPair[i])
        for i in self.NeutralSpecies:
            temp += "{}  ".format(self.DvalsCurrent[i+2])
        temp +="\n"
        self.GibbsErrorFile.write(temp)
        self.GibbsErrorFile.flush()
    
    def GetOperatorStats(self,RunPath_,Box=None):
        ''' Get operator statistics. TODO: Generalize to MD. 
            If system has charged species, modify the chemical potentials
            If Box!=None: Get operator statistics of a specified box'''
        
        
        operator_statistics = []
        if Box == None:      
            file_name = 'operators.dat'
        else:
            if Box not in [1,2]:
                raise Exception('Values of Box must be 1 or 2')
            file_name = 'model{}_operators.dat'.format(Box)
        if self.JobType == 'CL' and self.Program == 'polyFTS':
            number_columns = self.Nspecies*2+4
            datafile = open(os.path.join(RunPath_,file_name),'r')
            for c in range(number_columns)[::2]: #Skip step column and imaginary columns
                try:
                    warmup, Data, nwarmup = stats.autoWarmupMSER(datafile,c+1)
                except:
                    break
                (nsamples,(min,max),Val,Valerr,kappa,unbiasedvar,autocor)=stats.doStats(warmup,Data,False)
                operator_statistics.append([Val,Valerr,nsamples])
                                  
        elif self.JobType == 'SCFT' and self.Program == 'polyFTS': 
            number_columns = self.Nspecies+2
            data = np.loadtxt(os.path.join(RunPath_,file_name))[-1]
            for c in range(number_columns):
                operator_statistics.append([data[c+1],0.,1])
                          
        return operator_statistics        
    
    def UpdateParameters(self,Operator_List):
        ''' Update the parameters based on differences in Pressures and Chemical potentials between 
                BoxI and BoxII. 
            
            program == polyFTS:
                dUCurrent == [dH, dP, dMu_1, ...., dMu_Nspecies]
                
            program == MD:
                dUCurrent == [dU, dP, dMu_1, ...., dMu_Nspecies]
        '''
        
        self.DvalsLast = copy.deepcopy(self.DvalsCurrent) # copy to old before updating
        self.ValuesLast = copy.deepcopy(self.ValuesCurrent) # copy new to old before update
        self.OperatorsLast = copy.deepcopy(self.OperatorsCurrent)

        dUCurrent = []
        dUErrCurrent = []
        self.dUCurrent = []
        self.dUErrCurrent = []
        BoxIOperators = []
        BoxIIOperators = []            
        temp_Operators = []
        temp_error = []
        nparam = 2 + self.Nspecies
        if self.Barostat: 
            BoxIIOperators = [[0.,0.],[self.TargetP,0.]]        
        for i in range(nparam):
            BoxIOperators.append([Operator_List[0][i][0],Operator_List[0][i][1]])
            if not self.Barostat:
                BoxIIOperators.append([Operator_List[1][i][0],Operator_List[1][i][1]])
                temp_Operators.extend([Operator_List[0][i][0],Operator_List[0][i][1],Operator_List[1][i][0],Operator_List[1][i][1]])                                                                                                                          
                if self.Ensemble == 'NPT' and i == 1: # set the pressure in boxII to target pressure, only barostat boxI, @ equilibium PI==PII
                    dUCurrent.append(Operator_List[0][i][0] - Operator_List[1][i][0])
                    dUdP = (Operator_List[0][i][0] - self.TargetP)
                    dUErrCurrent.append(np.sqrt(Operator_List[0][i][1]**2))
                else:
                    dUCurrent.append(Operator_List[0][i][0] - Operator_List[1][i][0])
                    dUErrCurrent.append(np.sqrt(Operator_List[0][i][1]**2 + Operator_List[1][i][1]**2))
                if self.JobType == 'CL':
                    temp_error.extend([(Operator_List[0][i][1]/Operator_List[0][i][0]),(Operator_List[1][i][1]/Operator_List[1][i][0])])
            elif self.Barostat: # if barostating a single system
                if i == 1:
                    temp_Operators.extend([Operator_List[0][i][0],self.TargetP])
                    dUCurrent.append(Operator_List[0][i][0] - self.TargetP)
                    
                else: 
                    temp_Operators.extend([Operator_List[0][i][0],0.])
                    dUCurrent.append(Operator_List[0][i][0] - 0.)
                
                if i > 1:
                    BoxIIOperators.append([0.,0.])
                    
                dUErrCurrent.append(np.sqrt(Operator_List[0][i][1]**2 ))
        
        if self.JobType == 'CL' and not self.Barostat: # increase/decrease number of CL blocks depending on operator relative error
            max_relative_error = max(temp_error) # pick out the maximum relative error in the operator
            self.Write2Log("Max relative error: {0:3.3e}\n".format(max_relative_error))
            if max_relative_error > self.OperatorRelTol: # slow down
                scale = (max_relative_error/self.OperatorRelTol)**2
                self.NPolyFTSBlocks = self.NPolyFTSBlocks = np.minimum(self.NPolyFTSBlocksmax,scale*self.NPolyFTSBlocks)
                self.Write2Log('Slowing down: blocks {0} scale {1:3.3e}\n'.format(self.NPolyFTSBlocks,scale))
                if self.UseReRun and self.ReRunHist[-1] != True:
                    self.ReRun = True # rerun the current Gibbs step
                    self.Write2Log('Re-Running Gibbs Iteration {}\n'.format(self.Iteration))
                else:
                    self.ReRun = False
            elif max_relative_error < self.OperatorRelTol/3.: # speed up
                self.NPolyFTSBlocks = np.maximum(self.NPolyFTSBlocksmin,0.5*self.NPolyFTSBlocks)
                self.Write2Log('Speeding up: {}\n'.format(self.NPolyFTSBlocks))
                self.ReRun = False
            else:
                self.ReRun = False
                pass
        
        if self.JobType == 'CL' and max_relative_error > 0.09 and not self.Barostat:
            self.Break = True
            self.Write2Log('Breaking out of Gibbs!\n')
        
        if self.ReRun != True or self.UseReRun != True: # only update if not rerunning
            self.DvalsCurrent = copy.deepcopy(dUCurrent)
            self.OperatorsCurrent = copy.deepcopy(temp_Operators)
            
            if self.NchargedSpecies == 0: # neutral system
                if self.Barostat != True:
                    for indx, var in enumerate(self.ValuesCurrent[::2]):
                        if indx == 0: # volume fraction 
                            if self.Ensemble == 'NVT':
                                self.ValuesCurrent[indx*2] = var + self.Dt[indx]*self.DvalsCurrent[indx+1]
                                
                                if self.ValuesCurrent[indx*2] < self.VolFracBounds[0]:
                                    self.ValuesCurrent[indx*2] = 0.10
                                if self.ValuesCurrent[indx*2] > self.VolFracBounds[1]:
                                    self.ValuesCurrent[indx*2] = 0.90
                                self.ValuesCurrent[indx*2+1] = 1.-self.ValuesCurrent[indx*2] # update phase II
                            elif self.Ensemble == 'NPT':
                                CTot_new = self.CTotal - self.Dt[0]*dUdP # if P-Ptarget > 0, lower CTot, else increase
                                _NumberDens_Species = [CTot_new*_x for _x in self.SpeciesMoleFrac] # update number species
                                self.SetSpeciesCTotal(_NumberDens_Species) # update CTotal and SpeciesCTotals
                                
                                # update volume fractions
                                self.ValuesCurrent[indx*2] = var + self.Dt[indx]*self.DvalsCurrent[indx+1]
                                
                                if self.ValuesCurrent[indx*2] < self.VolFracBounds[0]:
                                    self.ValuesCurrent[indx*2] = 0.10
                                if self.ValuesCurrent[indx*2] > self.VolFracBounds[1]:
                                    self.ValuesCurrent[indx*2] = 0.90
                                self.ValuesCurrent[indx*2+1] = 1.-self.ValuesCurrent[indx*2] # update phase II
                                
                        
                        else: #update species
                            self.ValuesCurrent[indx*2] = var - np.minimum(var,self.ValuesCurrent[indx*2+1])*self.Dt[indx]*self.DvalsCurrent[indx+1]                
                                                
                            if self.ValuesCurrent[indx*2]  < 0.:
                                self.Write2Log('Value for operator {} < 0; iterating...\n'.format(indx))
                                self.ValuesCurrent[indx*2] = var/2.#temp_val
                            if self.ValuesCurrent[indx*2]  > self.SpeciesCTotal[indx-1]/self.ValuesCurrent[0]:
                                self.Write2Log('Value for operator {} > [max]; setting to [max]...\n'.format(indx))
                                self.ValuesCurrent[indx*2] = self.SpeciesCTotal[indx-1]/self.ValuesCurrent[0] - 1e-5
                                                           
                            #update phase II for species i
                            self.ValuesCurrent[indx*2+1] = (self.SpeciesCTotal[indx-1] - self.ValuesCurrent[indx*2]*self.ValuesCurrent[0])/(1.-self.ValuesCurrent[0])
                
                elif self.Barostat:
                    CTot = 0.
                    for indx, var in enumerate(self.ValuesCurrent):
                        if indx == 0: # the dP, update the CTot value 
                            pass
                        else:
                            CTot += var
                    # get new CTOT
                    CTot_new = CTot - self.Dt[0]*self.DvalsCurrent[1] # if P-Ptarget > 0, lower CTot, else increase
                
                    for indx in range(self.Nspecies):
                        self.ValuesCurrent[indx+1] = CTot_new * (self.SpeciesCTotal[indx]/np.sum(self.SpeciesCTotal))

            else: #charged system
                self.UpdateCpair()
                self.UpdateMuPair()
                self.CheckTotalCharge()
                self.CheckTotalSpeciesC()
                DtC_tmp = []
                if self.UseAdaptiveDtC:
                    self.UpdateChargedDtC()

                if self.Barostat != True:
                    for indx, var in enumerate(self.ValuesCurrent[::2]):                     
                        if indx == 0: # volume fraction
                            self.Write2Log('Volume fraction time step: {}\n'.format(self.Dt[0]))
                            if self.Ensemble == 'NVT':
                                self.ValuesCurrent[indx*2] = var + self.Dt[indx]*self.DvalsCurrent[indx+1]
                                if self.ValuesCurrent[indx*2] < self.VolFracBounds[0]:
                                    self.ValuesCurrent[indx*2] = 0.10
                                if self.ValuesCurrent[indx*2] > self.VolFracBounds[1]:
                                    self.ValuesCurrent[indx*2] = 0.90
                                self.ValuesCurrent[indx*2+1] = 1.-self.ValuesCurrent[indx*2] # update phase II
                            elif self.Ensemble == 'NPT':
                                CTot_new = self.CTotal - self.Dt[0]*dUdP # if P-Ptarget > 0, lower CTot, else increase
                                _NumberDens_Species = [CTot_new*_x for _x in self.SpeciesMoleFrac] # update number species
                                self.SetSpeciesCTotal(_NumberDens_Species) # update CTotal and SpeciesCTotals
                                
                                # update volume fractions
                                self.ValuesCurrent[indx*2] = var + self.Dt[indx]*self.DvalsCurrent[indx+1]
                                
                                if self.ValuesCurrent[indx*2] < self.VolFracBounds[0]:
                                    self.ValuesCurrent[indx*2] = 0.10
                                if self.ValuesCurrent[indx*2] > self.VolFracBounds[1]:
                                    self.ValuesCurrent[indx*2] = 0.90
                                self.ValuesCurrent[indx*2+1] = 1.-self.ValuesCurrent[indx*2] # update phase II
                        else:  #update species
                            # update neutral species first   
                            if indx - 1 in self.NeutralSpecies: 
                                if self.UseAdaptiveDtC:  
                                    Dt_tmp = self.Dt[indx]
                                else:
                                    Dt_tmp = np.minimum(var,self.ValuesCurrent[indx*2+1])* self.Dt[indx]
                                DtC_tmp.append(Dt_tmp)
                                self.ValuesCurrent[indx*2] = var - self.Dt[indx] * self.DvalsCurrent[indx+1] 
                                if self.ValuesCurrent[indx*2]  < 0.:
                                    self.Write2Log('Value for operator {} < 0; iterating...\n'.format(indx))
                                    self.ValuesCurrent[indx*2] = var/2.
                                if self.ValuesCurrent[indx*2]  > self.SpeciesCTotal[indx-1]/self.ValuesCurrent[0]:       
                                    self.Write2Log('Value for operator {} > [max]; setting to [max]...\n'.format(indx)) 
                                    self.ValuesCurrent[indx*2] = self.SpeciesCTotal[indx-1]/self.ValuesCurrent[0] - 1e-5 

                    # update charged pair concentration
                    for indx, [i,j] in enumerate(self.ChargedPairs):
                        Cpair1 = self.Cpair1[indx]
                        Cpair2 = self.Cpair2[indx]
                        if self.UseAdaptiveDtC:
                            Dt_tmp = self.DtCpair[indx]
                        else:
                            Dt_tmp = np.min([Cpair1,Cpair2]) * self.DtCpair[indx]
                        DtC_tmp.append(Dt_tmp)
                        dMuPair = self.MuPair1[indx] - self.MuPair2[indx]
                        Cpair1 = Cpair1 - Dt_tmp * dMuPair
                        self.Cpair1[indx] = Cpair1
                    # update individual charged species concentration in box 1
                    for i in self.ChargedSpecies:                                         
                        C = self.GetChargedC(i)
                        var = self.ValuesCurrent[(i+1)*2]
                        self.ValuesCurrent[(i+1)*2] = C                                   
                        if self.ValuesCurrent[(i+1)*2]  < 0.:
                            self.Write2Log('Value for operator {} < 0; iterating...\n'.format(i+1))
                            self.ValuesCurrent[(i+1)*2] = var/2.#temp_val
                        if self.ValuesCurrent[(i+1)*2]  > self.SpeciesCTotal[i]/self.ValuesCurrent[0]:         
                            self.Write2Log('Value for operator {} > [max]; setting to [max]...\n'.format(i+1)) 
                            self.ValuesCurrent[(i+1)*2] = self.SpeciesCTotal[i]/self.ValuesCurrent[0] - 1e-5 
                    # update phase II for species i
                    for i in range(self.Nspecies):
                        Ci1 = self.ValuesCurrent[(i+1)*2]
                        Ci = self.SpeciesCTotal[i]
                        f1 = self.ValuesCurrent[0]
                        self.ValuesCurrent[(i+1)*2+1] = (Ci - Ci1 * f1)/(1. - f1)
                                              
                    # update time step for volume fraction
                    Dtv = np.abs(np.min(DtC_tmp))/10.
                    Dt_new = np.array(self.Dt)
                    Dt_new[0] = Dtv
                    self.SetDt(Dt_new)
                                                                                                 
        if self.UseReRun != True: # just to ensure if not using rerun, set self.ReRun = False.
            self.ReRun = False
            self.ReRunHist.append(False)
        elif self.ReRun == True:
            self.ReRunHist.append(True)
        else:
            self.ReRunHist.append(False)
    
    def GetPolyFTSParameters(self):
        if not self.Barostat:
            vars_model1 = self.ValuesCurrent[2::2]
            vars_model2 = self.ValuesCurrent[3::2]
            
            model1 = [sum(vars_model1)]
            model2 = [sum(vars_model2)]
            
            for i in range(self.Nspecies):
                model1.append(vars_model1[i]/model1[0])
                model2.append(vars_model2[i]/model2[0])
        else:
            vars_model1 = self.ValuesCurrent[1:]
            
            model1 = [sum(vars_model1)]
            
            for i in range(self.Nspecies):
                model1.append(vars_model1[i]/model1[0])
            
            model2 = [None]
            
        return [model1,model2]
        
    def TakeGibbsStep(self):
        ''' Runs one simulation. '''
                
        if self.Converged == False:

            # update the parameters, check if boxes switched; i.e. concentrated box became dilute box and vice-versa
            if self.Program == 'polyFTS' and self.JobType != 'MF' and self.Barostat != True:
                JobID_List = []
                RunPaths = []
                p_status_list = []
                self.GenerateRunDirectory() # build directories
                
                if self.Iteration == 1:
                    self.ValuesCurrent = copy.deepcopy(self.VarsInit)
                    self.WriteStats() # values for this step
                
                out = self.GetPolyFTSParameters() # convert to polyFTS parameters
                VarModel1 = out[0]
                VarModel2 = out[1]
                self.Write2Log('\nIteration: {}\n'.format(self.Iteration))
                
                self.ReRun = True # always set true at start of Gibbs Step
                while self.ReRun: # rerun until self.ReRun not True
                    if not self.TwoModelTemplate:
                        model1path = os.path.join(os.getcwd(),'model1')
                        RunPaths.append(model1path)
                        self.WritePolyFTSInput(self.RunTemplateVars,VarModel1,model1path)
                        time.sleep(2)#wait 
                        SubmitFilePath = os.path.join(model1path,'submit_template.sh')
                        p_status, ID = self.RunJob(model1path,SubmitFilePath)
                        JobID_List.append(ID)
                        p_status_list.append(p_status)
                        
                        model2path = os.path.join(os.getcwd(),'model2')
                        RunPaths.append(model2path)
                        self.WritePolyFTSInput(self.RunTemplateVars,VarModel2,model2path)
                        time.sleep(2)#wait 
                        SubmitFilePath = os.path.join(model2path,'submit_template.sh')
                        p_status, ID = self.RunJob(model2path,SubmitFilePath)
                        JobID_List.append(ID)
                        p_status_list.append(p_status)
            
                        status,runTime = self.CheckQ(JobID_List,RunPaths) # wait for jobs to finish before continueing
                    else:
                        modelpath = os.path.join(os.getcwd(),'model_1_2')
                        RunPaths.append(modelpath)
                        self.WritePolyFTSInput([self.RunTemplateVarsBox1,self.RunTemplateVarsBox2],[VarModel1,VarModel2],modelpath,TwoModelTemplate=True)
                        time.sleep(2)#wait 
                        SubmitFilePath = os.path.join(modelpath,'submit_template.sh')
                        p_status, ID = self.RunJob(modelpath,SubmitFilePath)
                        JobID_List.append(ID)
                        p_status_list.append(p_status)
                        status,runTime = self.CheckQ(JobID_List,RunPaths) # wait for jobs to finish before continueing
                        
                    ''' Wait for jobs to finish current iteration. '''
                    exit_codes = [p.wait() for p in p_status_list]
                    status,runTime = self.CheckQ(JobID_List,RunPaths) # wait for jobs to finish before continueing
                    self.StepRunTime = runTime
                    
                    ''' Update parameters in preparation for next iteration. '''
                    Operator_List = []
                    if not self.TwoModelTemplate:
                        for RunPath in RunPaths:
                            Operator_List.append(self.GetOperatorStats(RunPath))
                    else:
                        Operator_List.append(self.GetOperatorStats(RunPaths[0],Box=1))
                        Operator_List.append(self.GetOperatorStats(RunPaths[0],Box=2))
                    
                    self.UpdateParameters(Operator_List)
                
                self.Iteration += 1
                self.WriteStats()

                    
            elif self.Program == 'polyFTS' and self.JobType == 'MF' and self.Barostat != True:
                # run a analytical mean-field model
                
                    if self.Iteration == 1:
                        self.ValuesCurrent = self.VarsInit
                        self.WriteStats() # values for this step    
                
                    out = self.GetPolyFTSParameters() # convert to polyFTS parameters
                    VarModel1 = out[0]
                    VarModel2 = out[1]
                    
                    Int = self.Interactions # uPP,uSS,uPS
                    a_list = self.InteractionRange # aPP,aSS,aPS
                    IncludeRPA = self.UseRPA
                    
                    # Data Analysis:
                    #model 1
                    rho1_A = VarModel1[0]*VarModel1[1]
                    rho1_B = VarModel1[0]*VarModel1[2]
                    model1 = self.SCFTModel(rho1_A,rho1_B,self.SpeciesDOP[0],self.SpeciesDOP[1],Int,a_list,IncludeRPA)
                    P1 = model1[0]
                    mu1_A = model1[1]
                    mu1_B = model1[2]
                    
                    #model 2
                    rho2_A = VarModel2[0]*VarModel2[1]
                    rho2_B = VarModel2[0]*VarModel2[2]
                    model2 = self.SCFTModel(rho2_A,rho2_B,self.SpeciesDOP[0],self.SpeciesDOP[1],Int,a_list,IncludeRPA)
                    P2 = model2[0]
                    mu2_A = model2[1]
                    mu2_B = model2[2]
                    
                    Operator_List = [[[0.,0.],[P1,0.],[mu1_A,0.],[mu1_B,0.]],[[0.,0.],[P2,0.],[mu2_A,0.],[mu2_B,0.]]]                    
                    self.UpdateParameters(Operator_List)
                    self.Iteration += 1
                    self.WriteStats()
            
            elif self.Program == 'polyFTS' and self.JobType != 'MF' and self.Barostat == True:
                # using a barostat
                JobID_List = []
                RunPaths = []
                p_status_list = []
                self.GenerateRunDirectory() # build directories
                
                if self.Iteration == 1:
                    self.ValuesCurrent = copy.deepcopy(self.VarsInit)
                    self.WriteStats() # values for this step
                
                out = self.GetPolyFTSParameters() # convert to polyFTS parameters
                VarModel1 = out[0]
                VarModel2 = out[1]
                print(self.Iteration)
                
                model1path = os.path.join(os.getcwd(),'model1')
                RunPaths.append(model1path)
                self.WritePolyFTSInput(self.RunTemplateVars,VarModel1,model1path)
                time.sleep(2)#wait 
                SubmitFilePath = os.path.join(model1path,'submit_template.sh')
                p_status, ID = self.RunJob(model1path,SubmitFilePath)
                JobID_List.append(ID)
                p_status_list.append(p_status)
    
                status,runTime = self.CheckQ(JobID_List,RunPaths) # wait for jobs to finish before continueing
                
                ''' Wait for jobs to finish current iteration. '''
                exit_codes = [p.wait() for p in p_status_list]
                status,runTime = self.CheckQ(JobID_List,RunPaths) # wait for jobs to finish before continueing
                self.StepRunTime = runTime
                
                ''' Update parameters in preparation for next iteration. '''
                Operator_List = []
                for RunPath in RunPaths:
                    Operator_List.append(self.GetOperatorStats(RunPath))
                
                self.UpdateParameters(Operator_List)
                self.Iteration += 1
                self.WriteStats()        
