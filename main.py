#!/usr/bin/python3

import argparse
import datetime, time
import random

import numpy as np
import os
import torch
from transformer import Transformer

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)

    return parser

def main(args):
    src = torch.rand(64, 16, 512)
    tgt = torch.rand(64, 16, 512)
    out = Transformer()(src, tgt)
    print(out.shape)

    for epoch in range(args.epochs):
        model.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("DETR training and evaluation script". parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)
