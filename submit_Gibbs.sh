#!/bin/bash
######################################
# Replace everything inside <...> with
# suitable settings for your jobs
######################################
#PBS -q batch
#PBS -l nodes=1:ppn=1
#PBS -l walltime=200:00:00
#PBS -V
#PBS -j oe
#PBS -N N6_T_300.0
#PBS -M nsherck@mrl.ucsb.edu
######################################
cd $PBS_O_WORKDIR

source ~/.bashrc

python RunGibbs.py > runGibbs.out

# Force good exit code here - e.g., for job dependency
exit 0

