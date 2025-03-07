import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from spectrograph import load_sound_data

TESTING = True

if TESTING:
    X_train, Y_train, X_test, Y_test = load_sound_data(testing=TESTING)
else:
    X_train, Y_train, X_test, testingIDs = load_sound_data(testing=TESTING)

scaler = StandardScaler()
X_train = torch.tensor(scaler.fit_transform(X_train), dtype=torch.float32)
X_test = torch.tensor(scaler.transform(X_test), dtype=torch.float32)
Y_train = torch.tensor(np.argmax(Y_train, axis=1), dtype=torch.long)  

X_train = torch.nan_to_num(X_train)
X_test = torch.nan_to_num(X_test)

batch_size = 32 #24, 16
epochs = 50 #50, 60, 45
train_dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class DiseaseClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, num_classes)  
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DiseaseClassifier(input_dim=X_train.shape[1], num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  

#trianing!
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(1) == batch_Y).sum().item()
    
    accuracy = correct / len(Y_train)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

model.eval()
with torch.no_grad():
    X_test = X_test.to(device)
    test_preds = model(X_test).argmax(1).cpu().numpy()

if TESTING:
    accuracy = np.sum(Y_test.argmax(axis=1) == np.array(test_preds)) / len(Y_test)
    print(f"Testing: {accuracy}")

    yhat_baseline = np.zeros_like(Y_test)
    yhat_baseline[:, Y_test.sum(axis=0).argmax(axis=0)] = 1
    accuracy = np.sum(np.all(Y_test == yhat_baseline, axis=1)) / len(Y_test)
    print(f"Baseline: {accuracy}")
else:
    df = pd.DataFrame({'candidateID': testingIDs, 'disease': test_preds})
    df.to_csv('submission3.csv', index=False)
    print("Predictions saved to submission3.csv")