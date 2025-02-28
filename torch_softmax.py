from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch
from classification import load_data


##########################
### SETTINGS
##########################

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
random_seed = 123
learning_rate = 0.001
num_epochs = 100
batch_size = 256

# Architecture
num_features = 523
num_classes = 3


EPOCHS = 100
X, y_train, X_test, testingIDs, X_train  = load_data() # loads data 

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train = torch.argmax(y_train, dim = 1) # reverts one_hot for labels
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train = torch.cat((X_train[:, :519], X_train[:, 519+1:]), dim=1) # gets rid of nans
X_test = torch.cat((X_test[:, :519], X_test[:, 519+1:]), dim=1) 


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
train_dataset = CustomDataset(X_train, y_train)
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


