#!/bin/bash

#SBATCH --qos=short
#SBATCH --job-name=mpi_job_test      
#SBATCH --mail-type=BEGIN,END,FAIL  
#SBATCH --cpus-per-task=1          
#SBATCH --nodes=2                   
#SBATCH --ntasks-per-node=12         
#SBATCH --distribution=cyclic:cyclic
#SBATCH --mem=4096  

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
echo "Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES"
echo "Number of Tasks Allocated      = $SLURM_NTASKS"
echo "Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK"

module load cuda/11.0 cudnn
module list
srun 