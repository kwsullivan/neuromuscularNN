import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EmgArrayDataset(Dataset):

    def __init__(self, csv_file, root_dir):
        self.severity = pd.read_csv(csv_file)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.severity)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        file_name = os.path.join(   self.root_dir,
                                    self.severity.iloc[idx, 0])
        data = get_emg_array(file_name, 200, 125200)
        label = self.severity.iloc[idx][2]
        sample = {'data' : data, 'label' : label}
        return sample

def get_emg_array(filename, start, end):
    with open(filename, 'rb') as fh:
        loaded_array = np.frombuffer(fh.read(), dtype=np.uint8)
    return loaded_array[start:end]

###############################################################
# emg_data = pd.read_csv('../emg_data_labels_10.csv')

# for i in range(0, 60):
#     file_name = emg_data.iloc[i,0]
#     neuropathy = emg_data.iloc[i, 3]
#     myopathy =  emg_data.iloc[i, 4]
    
#     labels = np.array([float(neuropathy), float(myopathy)])
#     print(labels[1])
###############################################################


# file_path = emg_data.iloc[0, 1]

# print(file_path)

# file_name = os.path.join(   sim_output,
#                             file_path,
#                             emg_data.iloc[0, 0])

# print(file_name)
# emg_array = get_emg_array(file_name)
# print(emg_array)



#emg_dataset = EmgArrayDataset(dataset_file='../datasets/emg_dataset10.pt')


# for i in range(len(emg_dataset)):
#     data, labels = emg_dataset[i]
#     print(labels)

# train_set = DataLoader(emg_dataset, batch_size=4, shuffle=True)

# for curr in train_set:
#     print(curr)


# WRITE DATASETS TO FILE
#######################################################################################

# level='basic'
# data='train'


# csv_file = f"../csv/emg_data_labels_{level}.csv"
# sim_dir = f"../databuilder/{data}/{level}"
# file_array = ['healthy/', 'neuro50/']
# dataset_file = f"../datasets/emg_dataset_{level}_{data}.pt"



# emg_data = pd.read_csv(csv_file)
# emg_out = []
# for idx in range(emg_data.shape[0]):
#     file_name = os.path.join(sim_dir,
#                             emg_data.iloc[idx, 0])
#     data = get_emg_array(file_name, 200, 125200)
#     # neuropathy = emg_data.iloc[idx, 2]
#     # emg_out.append({ 'data': data, 'neuropathy': neuropathy })
#     emg_out.append(data)

# for curr in emg_out:
#     print(curr[0:5])
#     print(' -- ')

# torch.save(emg_out, dataset_file)
