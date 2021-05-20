#!/bin/bash
#CMD="python3 main.py --lift"
CMD="python3 finetune.py --lift --resume ./world_checkpoint/ft_2.bin"
echo $CMD
$CMD
