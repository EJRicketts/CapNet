import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    # inherits from nn.Module
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 25)
        self.fc4 = nn.Linear(25, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
print(torch.__version__)
# Initiate network, load model, put in evaluation mode
net = Net()
net = torch.load("NNetModel")
net.eval()

# Sample data for fracture and non fracture
data = [1.25E-01, 1.67E-01, 5.00E-01, 1.33E-05, 1.00E+00]
data = torch.FloatTensor(data)
dataFrac = [1.25E-01, 1.67E-01, 5.00E-01, 1.33E-01, 6.67E-01]
dataFrac = torch.FloatTensor(dataFrac)

# Run inference
out = net(data)
outFrac = net(dataFrac)

# Print output (0 or 1)
print(torch.argmax(out).numpy())
print(torch.argmax(outFrac).numpy())
