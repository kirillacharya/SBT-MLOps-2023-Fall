import torch
import torch.optim as optim
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(784, 512)
        self.bn = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.rlu = nn.ReLU()
        self.bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc3(self.bn(self.fc2(self.rlu(self.bn(self.fc1(self.flatten(x)))))))
        return x

model = Model()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
torch.save({
    'model_state_dict': model.state_dict(),
    'opt_state_dict': opt.state_dict(),
    'epoch_num': 8, 
    'loss': 5}, 
    'model.pt')