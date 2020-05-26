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
		
		
		'''
		
		self.Program 			= _Program
		self.Nspecies   		= _Nspecies
		self.JobType    		= '' 
		self.SpeciesDOP		 	= []
		self.SpeciesCTotal 		= []
		self.VarsInit			= []
		self.ValuesCurrent		= [] # list ==current values of the fI,CIi
		self.ValuesLast			= [] # the last steps values of the fI,CIi 
		self.OperatorsCurrent	= [] 
		self.OperatorsLast		= [] 
		self.DvalsCurrent	    = [1.]
		self.DvalsLast			= []
		self.SaveFolders 		= True # default by true
		self.Iteration 			= 1 # initial iteration is 1
		self.RunTemplateName	= '' # the program template run file
		self.RunTemplateVars    = [] # variable names to replace in run template
		self.GibbsLogFile   	= ''
		self.GibbsLogFileName   = 'gibbs.dat'
		self.GibbsErrorFile 	= ''
		self.GibbsErrorFileName = 'error.dat'
		self.Converged 			= False
		self.Dt					= [0.1,0.1,0.1]
		self.InteractionRange   = []
		self.Interactions 		= []
		self.UseRPA				= False
		self.NPolyFTSBlocks 	= 200
		self.NPolyFTSBlocksmin  = 200
		self.NPolyFTSBlocksmax  = 10000
		self.OperatorRelTol		= 0.005
		self.UseOneNode			= False
		self.PolyFTSExec	 	= '~/PolyFTS_2020.02.24/bin/Release/PolyFTS.x'
		self.VolFracBounds 		= [0.1,0.9] # lower,upper
		self.StepRunTime 		= -1. # amount of time to run simulations
		self.PSInteraction		= -999
		self.ReRun				= False # Boolean that is True if program should rerun the current Gibbs step before moving on
		self.UseReRun			= True # Turn on/off ReRun capability
		self.ReRunHist          = [] # list of reruns
		self.LogFileName		= 'Gibbs.log'
		self.LogFile			= None
		self.Break				= False
		
		self.SetLogFile(self.LogFileName) # initialize LogFile
	
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
		
	def SetSpeciesCTotal(self,_SpeciesCTotal):
		''' A list of the number concentration of each species. '''
		self.SpeciesCTotal = _SpeciesCTotal
		if self.Nspecies != len(_SpeciesCTotal):
			self.Write2Log('WARNING: The number of species does not match the length of the list of number densities!\n')
		self.NSpecies = len(_SpeciesCTotal)
	
	def SetDt(self,_Dt):
		''' A list of the Gibbs updates. '''
		self.Dt = _Dt
	
	def SetVolFracBounds(self,_VolFracBounds):
		''' Sets the minimum and upper bounds on the volume fraction, can be useful to keep simulation in stable region at outset. '''
		self.VolFracBounds = list(_VolFracBounds)
	
	def SetUseOneNode(self,_UseOneNode):
		''' Whether to run on a single node, or to submit to multiple nodes. '''
		self.UseOneNode = bool(_UseOneNode)
	
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
		
	def SetRunTemplate(self,_RunTemplateName,_DummyVariables):
		''' Run Template name and replacement variables. '''
		self.RunTemplate = str(_RunTemplateName[0])
		self.RunTemplateVars = _DummyVariables

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
		try:
			os.mkdir('model1')
		except: 	
			#rmtree('model1')
			#os.mkdir('model1')
			pass
			
		try:
			os.mkdir('model2')
		except:
			#rmtree('model2')
			#os.mkdir('model2')
			pass
			
		if not self.UseOneNode:
			copyfile(os.path.join(os.getcwd(),'submit_template.sh'),os.path.join(os.getcwd(),'model1','submit_template.sh'))
			copyfile(os.path.join(os.getcwd(),'submit_template.sh'),os.path.join(os.getcwd(),'model2','submit_template.sh'))
			
	def WritePolyFTSInput(self,dummyvarnames,variables,modelpath):
		''' Writes polyFTS run files. '''

		with open('template.in','r') as myfile:
			ini=myfile.read()
			for indx, var in enumerate(variables): 
				ini=re.sub(dummyvarnames[indx],str(var),ini)
			
			if self.Iteration > 1:
				ini=re.sub('__READFIELDS__','Yes',ini)
				ini=re.sub('__NUMBLOCKS__',str(self.NPolyFTSBlocks),ini)
			elif self.Iteration < 1:
				ini=re.sub('__READFIELDS__','No',ini)
				ini=re.sub('__NUMBLOCKS__','4000',ini)
			else:
				ini=re.sub('__READFIELDS__','No',ini)
				ini=re.sub('__NUMBLOCKS__','4000',ini)
			
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
			for i in range(self.NSpecies):
				header_temp += 'CI_{}  CII_{}  '.format(i+1,i+1)		
			header_temp += "HI  stderr  HII  stderr  PI  stderr  PII  stderr  "
			for i in range(self.NSpecies):
				header_temp += 'muI_{}  stderr  muII_{}  stderr  '.format(i+1,i+1)
			header_temp +="\n"
			self.GibbsLogFile.write(header_temp)
			
			try:
				os.remove(self.GibbsErrorFileName)
			except:
				pass
			self.GibbsErrorFile = open(self.GibbsErrorFileName,"w")
			header_temp = "#  step  dH  dP  "
			for i in range(self.NSpecies):
				header_temp += 'dmu_{}  '.format(i+1)
			header_temp +="\n"
			self.GibbsErrorFile.write(header_temp)
			step = 0
		else: 
			step = self.Iteration-1
	
	
		temp = "{}  ".format(step)
		for val in self.ValuesCurrent:
			temp += "{}  ".format(val)
		for val in self.OperatorsCurrent:
			temp += "{}  ".format(val)
		temp +="\n"
		self.GibbsLogFile.write(temp)
		self.GibbsLogFile.flush()
		
		temp = "{}  ".format(step)
		for val in self.DvalsCurrent:
			temp += "{}  ".format(val)
		temp +="\n"
		self.GibbsErrorFile.write(temp)
		self.GibbsErrorFile.flush()
	
	def GetOperatorStats(self,RunPath_):
		''' Get operator statistics. TODO: Generalize to MD. '''
		
		
		operator_statistics = []
		
		if self.JobType == 'CL' and self.Program == 'polyFTS':
			number_columns = self.Nspecies*2+4
			datafile = open(os.path.join(RunPath_,'operators.dat'),'r')
			for c in range(number_columns)[::2]: #Skip step column and imaginary columns
				try:
					warmup, Data, nwarmup = stats.autoWarmupMSER(datafile,c+1)
				except:
					break
				(nsamples,(min,max),Val,Valerr,kappa,unbiasedvar,autocor)=stats.doStats(warmup,Data,False)
				operator_statistics.append([Val,Valerr,nsamples])
		
		elif self.JobType == 'SCFT' and self.Program == 'polyFTS': 
			number_columns = self.Nspecies+2
			data = np.loadtxt(os.path.join(RunPath_,'operators.dat'))[-1]
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
		temp_Operators = []
		temp_error = []
		nparam = 2+self.Nspecies
		for i in range(nparam):
			temp_Operators.extend([Operator_List[0][i][0],Operator_List[0][i][1],Operator_List[1][i][0],Operator_List[1][i][1]])
			dUCurrent.append(Operator_List[0][i][0] - Operator_List[1][i][0])
			dUErrCurrent.append(np.sqrt(Operator_List[0][i][1]**2 + Operator_List[1][i][1]**2))
			if self.JobType == 'CL':
				temp_error.extend([(Operator_List[0][i][1]/Operator_List[0][i][0]),(Operator_List[1][i][1]/Operator_List[1][i][0])])
			
		if self.JobType == 'CL': # increase/decrease number of CL blocks depending on operator relative error
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
		
		if self.JobType == 'CL' and max_relative_error > 0.09:
			self.Break = True
			self.Write2Log('Breaking out of Gibbs!\n')
		
		if self.ReRun != True or self.UseReRun != True: # only update if not rerunning
			self.DvalsCurrent = copy.deepcopy(dUCurrent)
			self.OperatorsCurrent = copy.deepcopy(temp_Operators)
			
			for indx, var in enumerate(self.ValuesCurrent[::2]):
				if indx == 0: # volume fraction 
					self.ValuesCurrent[indx*2] = var + self.Dt[indx]*self.DvalsCurrent[indx+1]
					
					if self.ValuesCurrent[indx*2] < self.VolFracBounds[0]:
						self.ValuesCurrent[indx*2] = 0.10
					if self.ValuesCurrent[indx*2] > self.VolFracBounds[1]:
						self.ValuesCurrent[indx*2] = 0.90
					self.ValuesCurrent[indx*2+1] = 1.-self.ValuesCurrent[indx*2] # update phase II
				
				else: #update species
					self.ValuesCurrent[indx*2] = var - np.minimum(var,self.ValuesCurrent[indx*2+1])*self.Dt[indx]*self.DvalsCurrent[indx+1]				
										
					if self.ValuesCurrent[indx*2]  < 0.:
					#	self.ValuesCurrent[indx*2] = 0.+1e-6
						#temp_val = self.ValuesCurrent[indx*2]
						#dteff = self.Dt[indx]
						#cnt = 0.
						self.Write2Log('Value for operator {} < 0; iterating...\n'.format(indx))
						#while temp_val < 0.:
						#	dteff = dteff/2.
							#print("dteff {}".format(dteff))
							#print("temp_value {}".format(temp_val))
						#	temp_val = var - np.minimum(var,self.ValuesCurrent[indx*2+1])*dteff*self.DvalsCurrent[indx+1]	
						#	cnt += 1
						self.ValuesCurrent[indx*2] = var/2.#temp_val
						#self.NPolyFTSBlocks = int(self.NPolyFTSBlocks*4) 
						#print('After {0} iter. found new dteff {1:3.3e}'.format(cnt,dteff))
						#print('Increasing number blocks to {}'.format(self.NPolyFTSBlocks))
					if self.ValuesCurrent[indx*2]  > self.SpeciesCTotal[indx-1]/self.ValuesCurrent[0]:
						self.Write2Log('Value for operator {} > [max]; setting to [max]...\n'.format(indx))
						self.ValuesCurrent[indx*2] = self.SpeciesCTotal[indx-1]/self.ValuesCurrent[0] - 1e-5
						#self.NPolyFTSBlocks = int(self.NPolyFTSBlocks*4) 
						#print('Increasing number blocks to {}'.format(self.NPolyFTSBlocks))
					
					#update phase II for species i
					self.ValuesCurrent[indx*2+1] = (self.SpeciesCTotal[indx-1] - self.ValuesCurrent[indx*2]*self.ValuesCurrent[0])/(1.-self.ValuesCurrent[0])
	
		if self.UseReRun != True: # just to ensure if not using rerun, set self.ReRun = False.
			self.ReRun = False
			self.ReRunHist.append(False)
		elif self.ReRun == True:
			self.ReRunHist.append(True)
		else:
			self.ReRunHist.append(False)
	
	def GetPolyFTSParameters(self):
		vars_model1 = self.ValuesCurrent[2::2]
		vars_model2 = self.ValuesCurrent[3::2]
		
		model1 = [sum(vars_model1)]
		model2 = [sum(vars_model2)]
		
		for i in range(self.NSpecies):
			model1.append(vars_model1[i]/model1[0])
			model2.append(vars_model2[i]/model2[0])
		
		return [model1,model2]
		
	def TakeGibbsStep(self):
		''' Runs one simulation. '''
				
		if self.Converged == False:

			# update the parameters, check if boxes switched; i.e. concentrated box became dilute box and vice-versa
			if self.Program == 'polyFTS' and self.JobType != 'MF':
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
				self.Write2Log('Iteration: {}\n'.format(self.Iteration))
				
				self.ReRun = True # always set true at start of Gibbs Step
				while self.ReRun: # rerun until self.ReRun not True
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

					
			elif self.Program == 'polyFTS' and self.JobType == 'MF':
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
					