import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_emg_array(filename, start, end):
    with open(filename, 'rb') as fh:
        loaded_array = np.frombuffer(fh.read(), dtype=np.uint8)
    return loaded_array[start:end]

csv_file = './emg_data_labels_10.csv'

emg_data = pd.read_csv(csv_file)

for index, row in emg_data.iterrows():
    print(row['file_name'])

# file_name = emg_data.iloc[11,0]
# neuropathy = emg_data.iloc[11, 2]
# myopathy =  emg_data.iloc[11, 3]
# sim_output = './emgOutput/sims10/'

# file_path = emg_data.iloc[0, 1]

# print(file_path)

# file_name = os.path.join(   sim_output,
#                             file_path,
#                             emg_data.iloc[0, 0])

# print(file_name)
# emg_array = get_emg_array(file_name)
# print(emg_array)


## USE THIS


# for i in range(len(emg_dataset)):
#     sample = emg_dataset[i]
#     plt.plot(sample['emg_array'])
#     plt.show()