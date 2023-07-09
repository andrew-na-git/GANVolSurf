from __future__ import print_function
import argparse
import datetime
import numpy as np
from scipy.stats import norm
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import cm
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from torch import nn, optim, Tensor
from torch.utils.data import Dataset, DataLoader, random_split

# script parameters
batch_size = 128
log_interval = 100

# run on GPU if possible
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")