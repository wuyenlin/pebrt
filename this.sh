#!/bin/bash
CMD="python3 angle.py --lift"
#CMD="python3 finetune.py --lift --resume ./checkpoint/ft_5.bin"
echo $CMD
$CMD
