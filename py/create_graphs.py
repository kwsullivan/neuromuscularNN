import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from create_datasets import get_emg_array


# file = '/Users/kevinsullivan/Downloads/N2001A03BB/N2001A03BB51/N2001A03BB51.bin'
file = '../sims10/train/neuro40-004.dat'
# file = '/Users/kevinsullivan/Desktop/sim/trunk/data/run005/patient/emg/emg1.dat'
file = '/Users/kevinsullivan/Desktop/6050data/S003N5001/S003N500101/S003N500101.dat'
emg_array = get_emg_array(file, 200, 125200)

print(range(len(emg_array)/31250))
plt.plot(np.arange(int(len(emg_array)/31250)), emg_array)
plt.plot(emg_array)
plt.show()
