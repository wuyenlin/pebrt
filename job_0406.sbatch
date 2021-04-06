#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=11:30:00
#SBATCH --job-name=vit
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1          
#SBATCH --gres=gpu:turing:1
#SBATCH --mem=4096

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24
module list
srun python3 main.py --lift
