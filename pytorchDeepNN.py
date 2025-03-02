import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from classification import load_data
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
learning_rate = 0.0001
batch_size = 32

# Architecture
num_features = 520
num_classes = 3


EPOCHS = 100
X, y_train, X_test, testingIDs, X_train  = load_data() # loads data 

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train = torch.argmax(y_train, dim = 1) # reverts one_hot for labels
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train = torch.cat((X_train[:, :519], X_train[:, 519+1:]), dim=1) # gets rid of nans
X_test = torch.cat((X_test[:, :519], X_test[:, 519+1:]), dim=1) 

X_train_features = X_train[:, :-1]
class CustomDataset(Dataset):
    def __init__(self, features, labels=None):  # Make labels optional
        self.features = features
        self.labels = labels
   
    def __len__(self):
        return len(self.features)
   
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx], 0  


train_dataset = CustomDataset(X_train_features, y_train)
test_dataset = CustomDataset(X_test)  

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Label batch dimensions:', labels.shape)
    break

class DeepNeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x  

model = DeepNeuralNetwork(num_features=num_features, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training!
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)
        
        logits = model(features)
        loss = criterion(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f'Epoch {epoch+1}/{EPOCHS}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    model.eval() #training accuracy
    with torch.no_grad():
        correct, total = 0, 0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Training Accuracy: {accuracy:.2f}%')

model.eval()
predictions = [] #saving predictions 
with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
        predictions.extend(predicted.cpu().numpy())

df = pd.DataFrame({'candidateID': testingIDs, 'disease': predictions})
df.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")