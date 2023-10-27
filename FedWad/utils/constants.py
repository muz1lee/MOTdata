import socket
import os
import re
import sys
import copy
import math
import time
import random
import datetime
import shutil
import logging

import numpy as np
np.seterr(divide = 'ignore')
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
from torch import nn
import torch.nn.functional as F

PROJECT_NAME = "fedselect"

hostname = socket.gethostname()
# set the path according to the environment
if hostname.startswith('muz1lee.local'):
    DEFAULT_SAVE_ROOT = '/Users/muz1lee/Desktop/代码/fedselect/save/'
    DEFAULT_DATA_DIR = '/Users/muz1lee/Desktop/代码/fedselect/datasets/'
elif hostname.startswith('amax'):
    DEFAULT_SAVE_ROOT = '/Users/muz1lee/save/'
    DEFAULT_DATA_DIR = '/Users/muz1lee/datasets/'


def set_cuda(x):
    if "CUDA_VISIBLE_DEVICES" in os.environ and eval(os.environ["CUDA_VISIBLE_DEVICES"]) != -1:
        x = x.cuda()
    return x