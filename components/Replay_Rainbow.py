"""
经验回放类：
    1.采样并储存到经验回放区
    2.读取经验
    L^DQN = 1/2(-Q(new)+r+γQmax(old))^2 
"""
import torch
from collections import deque

class replay():
    def __init__(self,maxlen=10000,train_samples=256):
        self.normal = deque(maxlen=int(0.6*maxlen))
        self.good = deque(maxlen=int(0.2*maxlen))
        self.dead = deque(maxlen=int(0.2*maxlen))
        self.train_samples = train_samples
    
    def pull_memory(self,state,action,reward,next_state,done,td_error):
        mask = ( done != 2 )
        if mask.sum() == 0:
            return
        else:
            state = state[mask]
            action = action[mask]
            reward = reward[mask]
            next_state = next_state[mask]
            done = done[mask]
            td_error = td_error[mask]
            for i in range(state.size(0)):
                if reward[i] < 0:
                    self.dead.appendleft([state[i:i+1],action[i:i+1],reward[i:i+1],next_state[i:i+1],done[i:i+1],td_error[i:i+1]])
                elif reward[i] > 0.1:
                    self.good.appendleft([state[i:i+1],action[i:i+1],reward[i:i+1],next_state[i:i+1],done[i:i+1],td_error[i:i+1]])
                else:
                    self.normal.appendleft([state[i:i+1],action[i:i+1],reward[i:i+1],next_state[i:i+1],done[i:i+1],td_error[i:i+1]])
    
    def randget_memory(self):
        
        def get_memory(buffer,n):
            if len(buffer) < n:
                return list(buffer)
            else:
                weights = torch.cat([td for _,_,_,_,_,td in buffer])
                probs = torch.softmax(weights, dim=-1)
                indices = torch.multinomial(probs, num_samples=n, replacement=False)

                return [buffer[idx] for idx in indices.tolist()]
        
        normal_n = int(0.6*self.train_samples)
        good_n = int(0.2*self.train_samples)
        dead_n = int(0.2*self.train_samples)
        
        memory1 = get_memory(self.normal,normal_n)
        memory2 = get_memory(self.good,good_n)
        memory3 = get_memory(self.dead,dead_n)
        
        return memory1+memory2+memory3