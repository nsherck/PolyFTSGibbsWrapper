import time
import numpy as np
import scipy as sp
import scipy.stats
import math
import subprocess as prcs
import os, sys
from shutil import copyfile
import ast
import re
import subprocess
from subprocess import call

DOP = 6
#Temp = [140.,150.,160.,170.,180.,190.,200.,210.,220.,230.,240.,250.,260.,270.,280.,290.,300.,310.,320.,330.,
#		340.,350.,360.,370.,380.,390.,400.,410.,420.,430.,440.,450.,460.,470.,480.,490.,500.,510.,520.,520.,540.,550.]

Temp = [200.,250.,300.,350.,400.,450.]

file2move = ['Gibbs_V2.py','stats.py']
RunDirectorys = []
for T in Temp:
	RunDir = 'Temp_{}'.format(T)
	RunDirectorys.append(RunDir)
	os.mkdir('Temp_{}'.format(T))
	for file in file2move:
		copyfile(file,os.path.join(os.getcwd(),RunDir,file))
	
	# Move submit_Gibbs.sh
	with open('submit_Gibbs.sh', 'r') as myfile:
		ini = myfile.read()
		ini = re.sub('__DOP__',str(DOP),ini)
		ini = re.sub('__TEMP__',str(T),ini)

	outfile = open(os.path.join(os.getcwd(),RunDir,'submit_Gibbs.sh'),'w')
	outfile.write(ini)
	outfile.close()
	
	# Move RunGibbs.py
	with open('RunGibbsTemplate.py', 'r') as myfile:
		ini = myfile.read()
		ini = re.sub('__DOP__',str(DOP),ini)
		ini = re.sub('__TEMP__',str(T),ini)

	outfile = open(os.path.join(os.getcwd(),RunDir,'RunGibbs.py'),'w')
	outfile.write(ini)
	outfile.close()
	
	# Move template.in for polyFTS
	with open('template.in', 'r') as myfile:
		ini = myfile.read()
		ini = re.sub('__DOP__',str(DOP),ini)
		#ini = re.sub('__TEMP__',str(T),ini)

	outfile = open(os.path.join(os.getcwd(),RunDir,'template.in'),'w')
	outfile.write(ini)
	outfile.close()
	
	# run job
	pathsubmit = os.path.join(os.getcwd(),RunDir,'submit_Gibbs.sh')
	os.chdir(os.path.join(os.getcwd(),RunDir)) # submit from directory
	call_1 = "qsub {}".format(pathsubmit)
	p1 = prcs.Popen(call_1, stdout=prcs.PIPE, shell=True)	
	(output, err) = p1.communicate()
	p_status = p1.wait()
	os.chdir('..') #move backup
	print('Status for Run: {}'.format(T))
	print(p_status)