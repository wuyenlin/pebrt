#!/bin/bash
CMD="python3 main.py --lift"
#CMD="python3 finetune.py --lift --resume ./checkpoint/ft_4.bin"
echo $CMD
$CMD
