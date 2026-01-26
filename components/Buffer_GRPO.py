"""
经验回放类：
    1.采样并储存到经验回放区
    2.读取经验
"""
from collections import deque
import random

class replay():
    def __init__(self,maxlen=100000,train_samples=2048):
        self.buffers = deque(maxlen=int(maxlen))
        self.train_samples = train_samples
    
    def pull_memories(self,state,action,reward,log_prob,done):
        # 按步赋值，奖励直接转化为相对奖励
        mask = ( done != 2 )
        state = state[mask]
        action = action[mask]
        reward = reward[mask]
        log_prob = log_prob[mask]
        done = done[mask]

        if len(reward)>1:
            reward_std = reward.std(dim=-1) + 1e-6
        else:
            reward_std = 1
        reward = (reward - reward.mean(dim=-1))/reward_std
        for i in range(state.size(0)):
            self.buffers.appendleft([state[i:i+1],action[i:i+1],reward[i:i+1],log_prob[i:i+1],done[i:i+1]])
            
              
    def get_memories(self):
        # 最左边是最新的一步
        if len(self.buffers) < self.train_samples:
            return list(self.buffers)
        else:
            return random.sample(self.buffers,self.train_samples)