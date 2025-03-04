from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from ellys_data_loader_2 import load_data
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np

##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
learning_rate = 0.001
batch_size = 64

# Architecture
num_features = 523
num_classes = 3

EPOCHS = 100
K = 5  # Number of folds for cross-validation

X_train, y_train, X_test, testingIDs = load_data()  # loads data

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train = torch.argmax(y_train, dim=1)  # reverts one_hot for labels
X_test = torch.tensor(X_test, dtype=torch.float32)

print(X_train.shape)
print(X_test.shape)

X_train_features = X_train
X_test_features = X_test

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return a tuple (feature, label)
        return self.features[idx], self.labels[idx]

# Model Definition
class SoftmaxRegression(torch.nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftmaxRegression, self).__init__()
        self.linear = torch.nn.Linear(num_features, num_classes)
        
        self.linear.weight.detach().zero_()
        self.linear.bias.detach().zero_()
        
    def forward(self, x):
        logits = self.linear(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

# Function to compute accuracy
def compute_accuracy(model, data_loader):
    correct_pred, num_examples = 0, 0
    
    for features, targets in data_loader:
        features = features.view(-1, num_features).to(device)
        targets = targets.to(device)
        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
        
    return correct_pred.float() / num_examples * 100

# Initialize KFold
kf = KFold(n_splits=K, shuffle=True, random_state=random_seed)

# Perform K-Fold Cross Validation
fold_accuracies = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
    print(f"\nTraining Fold {fold + 1}/{K}...")
    
    # Split data into train and validation sets for this fold
    X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
    y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
    
    # Create DataLoader for this fold
    train_dataset = CustomDataset(X_train_fold, y_train_fold)
    val_dataset = CustomDataset(X_val_fold, y_val_fold)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model and optimizer for this fold
    model = SoftmaxRegression(num_features=num_features, num_classes=num_classes).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    torch.manual_seed(random_seed)
    
    # Training loop for the current fold
    for epoch in range(EPOCHS):
        model.train()  # Set the model to training mode
        
        for batch_idx, (features, targets) in enumerate(train_loader):
            batch_size = features.size(0)
            features = features.view(batch_size, num_features).to(device)  
            targets = targets.to(device)
            
            # Forward pass and backward pass
            logits, probas = model(features)
            cost = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            cost.backward()
            
            # Update model parameters
            optimizer.step()

            # Logging
            if batch_idx % 50 == 0:
                print(f"Epoch: {epoch + 1}/{EPOCHS} | Batch {batch_idx}/{len(train_loader)} | Cost: {cost.item():.4f}")

    # Validation phase for this fold
    model.eval()  # Set the model to evaluation mode
    val_accuracy = compute_accuracy(model, val_loader)
    print(f"Validation Accuracy for Fold {fold + 1}: {val_accuracy:.2f}%")
    fold_accuracies.append(val_accuracy)

# Compute average validation accuracy over all folds
avg_val_accuracy = np.mean(fold_accuracies)
print(f"\nAverage Validation Accuracy across all {K} folds: {avg_val_accuracy:.2f}%")

# After K-fold cross-validation, train on the full training set and evaluate on the test set
# Final model training on all data
final_model = SoftmaxRegression(num_features=num_features, num_classes=num_classes).to(device)
final_optimizer = torch.optim.SGD(final_model.parameters(), lr=learning_rate)

train_dataset = CustomDataset(X_train_features, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Training on full dataset
for epoch in range(EPOCHS):
    final_model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):
        batch_size = features.size(0)
        features = features.view(batch_size, num_features).to(device)
        targets = targets.to(device)
        
        logits, probas = final_model(features)
        cost = F.cross_entropy(logits, targets)
        final_optimizer.zero_grad()
        cost.backward()
        final_optimizer.step()

    if epoch % 50 == 0:
        print(f"Final Model - Epoch {epoch}/{EPOCHS} | Cost: {cost.item():.4f}")

# Test phase on final model
test_loader = DataLoader(dataset=CustomDataset(X_test_features, y_train), batch_size=batch_size, shuffle=False)
final_model.eval()

predictions = []

for features, targets in test_loader:
    features = features.view(features.size(0), num_features).to(device)
    targets = targets.to(device)

    with torch.no_grad():
        logits, probas = final_model(features)
        _, predicted_labels = torch.max(probas, 1)

    predictions.extend(predicted_labels.cpu().numpy())

# Save predictions to a CSV file
df = pd.DataFrame({
    'candidateID': testingIDs,
    'disease': predictions
})

df.to_csv('submissions.csv', index=False)
print("Predictions saved to submissions.csv")
