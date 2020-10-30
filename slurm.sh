#!/bin/bash

####### select resources (check https://ubccr.freshdesk.com/support/solutions/articles/13000076253-requesting-specific-hardware-in-batch-jobs)
#SBATCH --mem=32000
#SBATCH --constraint=V100

####### make sure no other jobs are assigned to your nodes

####### further customizations
#SBATCH --job-name="PointNet"
#SBATCH --output=out/%j.stdout
#SBATCH --error=err/%j.stderr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --time=01:00:00

date
hostname
eval "$(/util/common/python/py37/anaconda-2020.02/bin/conda shell.bash hook)"
conda activate tensorflow-2.3-gpu
python train.py
date
