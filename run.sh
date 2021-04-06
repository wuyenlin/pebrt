#!/bin/bash

module use /opt/insy/modulefiles
#module load cuda/11.1 cudnn/11.1-8.0.5.39
module load cuda/10.0 cudnn/10.0-7.4.2.24
module list

#CMD="python3 -m torch.distributed.launch --nproc_per_node=2 --use_env main.py --distributed"
CMD="python3 main.py"
echo $CMD
$CMD



