import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from classification_5 import load_data
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random_seed = 123
learning_rate = 0.0001
batch_size = 24
num_features = 1035
num_classes = 3
EPOCHS = 40 #80
TESTING = True

if TESTING:
    X_train, y_train, X_test, Y_test = load_data(testing=TESTING)  # loads data    
else:
    X_train, y_train, X_test, testingIDs = load_data(testing=TESTING)  # loads data

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_train = torch.argmax(y_train, dim = 1) # reverts one_hot for labels
X_test = torch.tensor(X_test, dtype=torch.float32)


X_train_features = X_train

class CustomDataset(Dataset):
    def __init__(self, features, labels=None):  
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

#MLP 
class MLPModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
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

model = MLPModel(input_size=num_features, num_classes=num_classes).to(device)  # Adjusting for removed feature

criterion = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#training!
for epoch in range(EPOCHS):
    model.train()  
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%")

model.eval()  
correct = 0
total = 0

predictions = [] #saving predictions 
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        predictions.extend(predicted.cpu().numpy())

if TESTING:
    accuracy = np.sum(Y_test.argmax(axis=1) == np.array(predictions)) / len(Y_test)
    print(f"Testing: {accuracy}")

    yhat_baseline = np.zeros_like(Y_test)
    yhat_baseline[:, Y_test.sum(axis=0).argmax(axis=0)] = 1
    accuracy = np.sum(np.all(Y_test == yhat_baseline, axis=1)) / len(Y_test)
    print(f"Baseline: {accuracy}")
else:
    df = pd.DataFrame({'candidateID': testingIDs, 'disease': predictions})
    df.to_csv('submissionMLP.csv', index=False)
    print("Predictions saved to submission.csv")