#!/bin/bash
#CMD="python3 main.py"
#CMD="python3 main.py --resume ./petr/epoch_20_h36m.bin"
CMD='python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 main.py'
echo $CMD
$CMD

