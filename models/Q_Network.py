"""
q-learning 网络设计说明：
    1.我们假设小蛇能看到自己周围6*6大小的所有物品（事实上，我们地图大小也就6*6）
    2.首先，小蛇不需要对局部关系有太强的敏感度，因此我们仅使用大小为3的卷积核。
    3.因为小蛇需要精确知道自己身体的形状和目标奖励的位置，因此我们设置padding为1
    4.对于小蛇，我们将它的身体表示为1，头部表示为2，奖励表示为3，空白用0表示
"""
import torch
import torch.nn as nn

class q_network(nn.Module):
    def __init__(self):
        super(q_network,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16,out_channels=16,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(6*6*16,64),
            nn.ReLU(),
            nn.Linear(64,4)
        )
    
    def forward(self,x):
        x = self.conv(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)  
        return x

'''
# 随机生成测试
Q = q_network()
x = torch.randn(10,1,6,6)
x = Q(x)
x,_ = x.topk(1)
print(x)

x = torch.randn(1,1,6,6)
x = Q(x)
x,_ = x.topk(1)
print(x)

x = torch.randn(1,1,6,6)
x = Q(x)
x,_ = x.topk(1)
print(x)
'''