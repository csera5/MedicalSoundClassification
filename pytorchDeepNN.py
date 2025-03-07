import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch
from classification_5 import load_data
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


random_seed = 123
learning_rate = 0.0001
batch_size = 24
weight_decay = 0 

# Architecture
num_features = 1035
num_classes = 3

TESTING = True

EPOCHS = 80

if TESTING:
    X_train, y_train, X_test, Y_test  = load_data(testing=TESTING) # loads data  
else:  
    X_train, y_train, X_test, testingIDs  = load_data(testing=TESTING) # loads data

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train = torch.argmax(y_train, dim=1)
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train_features = X_train
print(X_train_features)

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

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

class DeepNeuralNetwork(nn.Module):
    def __init__(self, num_features, num_classes):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0)  

# Training!
best_val_accuracy = 0.0  

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

    model.eval()  
    with torch.no_grad():
        correct, total = 0, 0
        for features, targets in test_loader:
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        val_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}/{EPOCHS}, Validation Accuracy: {val_accuracy:.2f}%')

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), 'best_model.pth')

model.load_state_dict(torch.load('best_model.pth'))

model.eval()
predictions = []  
with torch.no_grad():
    for features, _ in test_loader:
        features = features.to(device)
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)
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
    df.to_csv('submission.csv', index=False)
    print("Predictions saved to submission.csv")