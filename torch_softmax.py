import torch
import torch.nn as nn
import torch.optim as optim
from classification import load_data
import pandas as pd

EPOCHS = 100
X_train, y_train, X_test, testingIDs = load_data() # loads data 


X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_train = torch.argmax(y_train, dim = 1)
X_test = torch.tensor(X_test, dtype=torch.float32)

X_train = torch.cat((X_train[:, :519], X_train[:, 519+1:]), dim=1) # gets rid of nans
X_test = torch.cat((X_test[:, :519], X_test[:, 519+1:]), dim=1) 


class SoftmaxRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]
output_dim = len(torch.unique(y_train))
model = SoftmaxRegression(input_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# Evaluation
with torch.no_grad():
    model.eval()
    test_outputs = model(X_test)
    _, y_pred_tensor = torch.max(test_outputs, 1)
    y_pred = y_pred_tensor.numpy()

# Calculate accuracy
print(y_pred)

df = pd.DataFrame({'candidateID': testingIDs, 'disease': y_pred})
df.to_csv('submission.csv', index = False) # write to csv file
