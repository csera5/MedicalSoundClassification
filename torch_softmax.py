from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from classification import load_data
import matplotlib.pyplot as plt
import pandas as pd

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
num_features = 520
num_classes = 3


EPOCHS = 50
X, y_train, X_test, testingIDs, X_train  = load_data() # loads data 

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train = torch.argmax(y_train, dim = 1) # reverts one_hot for labels
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train = torch.cat((X_train[:, :519], X_train[:, 519+1:]), dim=1) # gets rid of nans
X_test = torch.cat((X_test[:, :519], X_test[:, 519+1:]), dim=1) 

X_train_features = X_train[:, :-1]
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

# Modify train_loader and test_loader
train_dataset = CustomDataset(X_train_features, y_train)
test_dataset = CustomDataset(X_test, y_train)  

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset, 
                         batch_size=batch_size, 
                         shuffle=False)

# Check the dimensions
for images, labels in train_loader:  
    print('Image batch dimensions:', images.shape)
    print('Label batch dimensions:', labels.shape)
    break

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

model = SoftmaxRegression(num_features=num_features,
                          num_classes=num_classes)

model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  


torch.manual_seed(random_seed)


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

for epoch in range(EPOCHS):
    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.view(-1, num_features).to(device) 
        targets = targets.to(device)
            
        ### FORWARD AND BACK PROP
        logits, probas = model(features)
        
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        cost.backward()
        
        ### UPDATE MODEL PARAMETERS
        optimizer.step()
        
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %03d/%03d | Cost: %.4f' 
                   %(epoch+1, EPOCHS, batch_idx, 
                     len(train_dataset)//batch_size, cost))
            
    with torch.set_grad_enabled(False):
        print('Epoch: %03d/%03d training accuracy: %.2f%%' % (
              epoch+1, EPOCHS, 
              compute_accuracy(model, train_loader)))

print('Final train accuracy: %.2f%%' % (compute_accuracy(model, train_loader)))  

predictions = []

for features, targets in test_loader:
    features = features.view(features.size(0), num_features).to(device)  # Flatten each sample to 521 features
    targets = targets.to(device)
    logits, probas = model(features)
    _, predicted_labels = torch.max(probas, 1)
    predictions.extend(predicted_labels.cpu().numpy()) 


df = pd.DataFrame({
    'candidateID': testingIDs,  
    'disease': predictions
})

# Save the DataFrame to a CSV file
df.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")
