#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --job-name=mpi_job_test      
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --ntasks=1          
#SBATCH --cpus-per-task=1          
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=12         
#SBATCH --mem=8192

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module use /opt/insy/modulefiles
module load cuda/11.1 cudnn/11.1-8.0.5.39
module list
srun python3 -m torch.distributed.launch --nproc_per_node=2 --use_env main.py
