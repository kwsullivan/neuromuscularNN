import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from create_datasets import EmgArrayDataset
from torch.utils.data import Dataset, DataLoader

class Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(400, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=3, stride=2)
        self.fc1 = nn.Linear(4*50*50, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.fc1(x)
        return x


train = EmgArrayDataset(csv_file='../csv/emg_data_labels_basic.csv', root_dir='../databuilder/test/basic/')
# test = EmgArrayDataset(dataset_file='../datasets/emg_dataset10_test.pt')
trainset = DataLoader(train, batch_size=8, shuffle=False)
# testset = DataLoader(test, batch_size, shuffle=True)

# for data in trainset:
#     print(data)
model = CNN()
model(trainset).shape

'''
optimizer = optim.Adam(net.parameters(), lr=0.00001)
correct = 0
total = 0

EPOCHS = 6

for epoch in range(EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(trainset, 0):
        X, y, z = data
        net.zero_grad()
        output = net(X)
        loss = F.nll_loss(output, y.long())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        print('[%d, %5d] loss %.3f' %
        (epoch + 1, i + 1, running_loss))
        running_loss = 0.0
print('Finished Training')


dataiter = iter(testset)
X, y, z = dataiter.next()
#print('Ground Truth: ', ' '.join('%5f' % [y[j]] for j in range(10)))
print('Ground Truth')
for j in range(batch_size):
    print(y[j])
outputs = net(X)

_, predicted = torch.max(outputs, 1)

#print('Predicted', ' '.join('%5f' % [predicted[j]] for j in range(10)))
print('Predicted')
for j in range(batch_size):
    print(predicted[j].float())

'''