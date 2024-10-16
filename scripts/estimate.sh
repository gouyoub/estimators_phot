#!/bin/bash

#SBATCH -e stdout_estimate.txt
#SBATCH -o stdout_estimate.txt
cd ../
python estimate_cl_2pt_format.py inifiles/FS2_3x2_firstchain.cfg