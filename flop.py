#!/usr/bin/python3
from common.petr import *

import torch
from ptflops import get_model_complexity_info



if __name__ == "__main__":
    model = PETR(lift=True)
    model = model.cuda()
    macs, params = get_model_complexity_info(model, (3, 256, 256), as_strings=True,
                                            print_per_layer_stat=False, verbose=False)
                                            # print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))