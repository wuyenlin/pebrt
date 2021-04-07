#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=16:00:00
#SBATCH --job-name=finetune_lift
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1          
#SBATCH --gres=gpu:turing:1
#SBATCH --mem=3584

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
nvidia-smi -L
srun python3 finetune.py --lift --resume ../checkpoint/0406.bin
