import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class EmgArrayDataset(Dataset):

    def __init__(self, csv_file, sim_dir, transform=None):
        self.emg_data = pd.read_csv(csv_file)
        self.sim_dir = sim_dir

    def __len__(self):
        return len(self.emg_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        file_path = self.emg_data.iloc[idx, 1]
        file_name = os.path.join(self.sim_dir,
                                file_path,
                                self.emg_data.iloc[idx, 0])
        
        data = get_emg_array(file_name, 100, 500)
        data = torch.tensor(data)
        
        labels = self.emg_data.iloc[idx, 3:]
        labels = np.array([labels])
        labels = labels.astype('float').reshape(-1, 2)
        
        
        labels = torch.tensor(labels)
        
        sample = { 'data': data, 'labels': labels }

        return sample


def get_emg_array(filename, start, end):
    with open(filename, 'rb') as fh:
        loaded_array = np.frombuffer(fh.read(), dtype=np.uint8)
    return loaded_array[start:end]

###############################################################
# emg_data = pd.read_csv('./emg_data_labels_10.csv')

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



emg_dataset = EmgArrayDataset(csv_file='./emg_data_labels_10.csv', sim_dir='./emgOutput/sims10/')

train_set = DataLoader(emg_dataset, batch_size=4, shuffle=True)

# for i in range(len(emg_dataset)):
#     sample = emg_dataset[i]

#     print(i, sample['labels'])

# for data in training_set:
#     print(data)

# for data in training_set:
#     print(data)