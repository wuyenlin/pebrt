#!/usr/bin/python3

import argparse
import datetime, time
import random

import numpy as np
import os
import torch
from train import *

def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--lr", default=1e-4, type=float)

    return parser

def main(args):
    pass

if __name__ == "__main__":
    main(args)
