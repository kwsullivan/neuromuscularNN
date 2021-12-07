import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EmgArrayDataset(Dataset):

    def __init__(self, dataset_file):
        self.emg_data = torch.load(dataset_file)

    def __len__(self):
        return len(self.emg_data)

    def __getitem__(self, idx):

        data = torch.tensor(self.emg_data[idx]['data']).float()
        neuropathy = torch.tensor(self.emg_data[idx]['neuropathy'])
        myopathy = torch.tensor(self.emg_data[idx]['myopathy'])

        return data, neuropathy, myopathy

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

csv_file='../csv/emg_data_labels_10.csv'
sim_dir='../emgOutput/sims10/test/'
dataset_file='../datasets/emg_dataset10_test.pt'

emg_data = pd.read_csv(csv_file)
emg_out = []
for idx in range(emg_data.shape[0]):
    file_name = os.path.join(sim_dir,
                            emg_data.iloc[idx, 0])

    data = get_emg_array(file_name, 200, 125200)
    neuropathy = emg_data.iloc[idx, 3]
    myopathy = emg_data.iloc[idx, 4]
    # labels = np.array([labels])
    # labels = labels.astype('float').reshape(-1, 2)
    emg_out.append({ 'data': data, 'neuropathy': neuropathy, 'myopathy': myopathy})
    
torch.save(emg_out, dataset_file)

