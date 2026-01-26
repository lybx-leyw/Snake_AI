import torch.nn as nn

class policy_value(nn.Module):
    def __init__(self):
        super(policy_value,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.fc1 = nn.Sequential(
            nn.Linear(6*6*16,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(6*6*16,64),
            nn.ReLU(),
            nn.Linear(64,4),
            nn.ReLU(),
            nn.Linear(4,1),
        )
    
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        policy = self.fc1(x)
        value = self.fc2(x)
        return policy,value