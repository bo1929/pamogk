#!/bin/bash
# -= Resources =-
#
#SBATCH --job-name=pamogk
#SBATCH --account=mdbf
#SBATCH --ntasks-per-node=16
#SBATCH --qos=long_mdbf
#SBATCH --partition=long_mdbf
#SBATCH --time=23:59:00
#SBATCH --output=/cta/users/aosman/pamogk/experiments/out.out
#SBATCH --mem=32G

# Set stack size to unlimited
python /cta/users/aosman/pamogk/experiments/pamogk_exp_comm.py > outlog.out
