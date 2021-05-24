#!/bin/bash
CMD="python3 angle.py --lift --resume ./angle_checkpoint/epoch_5.bin"
#CMD="python3 angle.py --lift --resume ./angle_checkpoint/trans.bin"
echo $CMD
$CMD
