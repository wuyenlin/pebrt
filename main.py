#!/usr/bin/python3

import numpy as np
import os
import torch
from transformer import Transformer

def main():
    src = torch.rand(64, 16, 512)
    tgt = torch.rand(64, 16, 512)
    out = Transformer()(src, tgt)
    print(out.shape)

if __name__ == "__main__":
    main()
