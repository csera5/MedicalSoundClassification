import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz.12.2.1-win64/bin/'
# cry no work

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

dummy_input = torch.randn(1, 784)  # Batch size of 1, 784 input features

from torchviz import make_dot

model = SimpleNet()
output = model(dummy_input)
dot = make_dot(output, params=dict(model.named_parameters()))

# Save or display the generated graph
dot.format = 'png'
dot.render('simple_net')