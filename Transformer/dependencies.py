import os, sys
import pandas as pd
import io
import numpy as np
import math
from typing import Optional
from dataclasses import dataclass
from math import sqrt


import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib import pyplot
import seaborn as sns

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.transformer import MultiheadAttention, _get_activation_fn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
