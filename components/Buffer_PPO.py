"""
经验回放类：
    1.采样并储存到经验回放区
    2.读取经验
"""
from collections import deque

class replay():
    def __init__(self,paras,maxlen=2048):
        self.paras = paras
        self.buffers = [deque(maxlen=int(maxlen))
                        for _ in range(paras)]
    
    def pull_memories(self,state,action,reward,log_prob,value,next_value,done):
        if self.paras != state.size(0):
            print("Error:缓冲区数量与并行局数不同")
            return
        for i in range(state.size(0)):
            if done[i] == 2:
                continue
            self.buffers[i].appendleft([state[i:i+1],action[i:i+1],reward[i:i+1],log_prob[i:i+1],value[i:i+1],next_value[i:i+1],done[i:i+1]])
              
    def get_memories(self):
        # 最左边是最新的一步
        return list(self.buffers)
    
    def clear_memories(self):
        for i in range(self.paras):
            self.buffers[i].clear()