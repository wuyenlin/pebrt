#!/bin/bash
#CMD="python3 main.py"
CMD="python3 main.py --resume ./petr/epoch_20_h36m.bin"
#CMD="python3 main.py --eval --resume ./petr/epoch_10_h36m.bin"
# CMD="python3 main.py --eval --resume ./checkpoint/ft_5.bin"
# CMD="python3 main.py  --eval --resume ./petr/ft_5_h36m.bin"
echo $CMD
$CMD
