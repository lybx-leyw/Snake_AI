"""
游戏环境 搭建目标：
    1.地图大小固定为6*6，使用torch张量形式储存
    2.对于小蛇，我们将它的身体表示为1，头部表示为2，奖励表示为3，空白用0表示
    3.目标函数：
        a)移动函数
        b)胜负判断
        c)并行策略
"""
0
import torch
import msvcrt
import time
import numpy as np
import random
from collections import deque
from matplotlib import pyplot as plt 
    
class game_env():
    def __init__(self,paras=1,size=6,printLOG=False,logNAME="logs.txt",Watch=False):
        # 基本地图参数
        self.map_num = paras
        self.map_size = size
        self.maps = torch.zeros(paras,1,size,size)
        self.log = printLOG
        self.logName = logNAME
        self.Watch = Watch

        # 地图状态
        self.snake = []
        self.reward = []

        self.snake_length = torch.zeros(paras)
        self.snake_steps = torch.zeros(paras)
        self.snake_over = torch.zeros(paras)
        self.train_reward = torch.zeros(paras)
        self.features = torch.zeros(paras,24)

        self.sum_length = paras*3
        self.sum_steps = paras*3

        # 初始化
        self.reset()

    def reset(self):
        # 状态初始化
        self.snake_length = torch.ones(self.map_num)*3
        self.snake_steps = torch.zeros(self.map_num)
        self.snake_over = torch.zeros(self.map_num)
        self.train_reward = torch.zeros(self.map_num)
        self.features = torch.zeros(self.map_num,24)
        self.sum_length = self.map_num*3
        self.sum_steps = self.map_num*3
        
        # 蛇图初始位置
        initial_place = [2*self.map_size+3,3*self.map_size+3,4*self.map_size+3]
        self.snake = [deque(initial_place) for _ in range(self.map_num)]
        self.maps[:,0,2,3] = 2
        self.maps[:,0,3,3] = 1
        self.maps[:,0,4,3] = 1
        self.reward = [random.randint(0, self.map_size*self.map_size-1)
                      for _ in range(self.map_num)]
        
        for i in range(self.map_num):
            while self.reward[i] in self.snake[i]:
                self.reward[i] = random.randint(0, self.map_size*self.map_size-1)
        
        for i in range(self.map_num):
            row = self.reward[i]//6
            col = self.reward[i]%6
            self.maps[i,0,row,col] = 3
        
        # log清空
        with open(self.logName,"w") as file:
            file.close()
        
        self.all_watch()

    def move(self, direction, index=0):
        self.train_reward[index] = 0
        if self.snake_over[index] == 1:
            self.snake_over[index] = 2
            return False
        if self.snake_over[index] == 2:
            return False
        
        head = self.snake[index][0]
        move = self.getMove(direction)
        newHead = head + move
        
        if newHead == self.snake[index][1]:
            self.train_reward[index] = -2
            direction = (direction+2)%4
            move = self.getMove(direction)
            newHead = head + move
        
        # 处理穿墙bug
        if direction == 0 and head < self.map_size:  # 上
            self.snake_over[index] = 1
            self.train_reward[index] = -5
            return False
        elif direction == 1 and head >= self.map_size * (self.map_size - 1):  # 下
            self.snake_over[index] = 1
            self.train_reward[index] = -5
            return False
        elif direction == 2 and head % self.map_size == 0:  # 左
            self.snake_over[index] = 1
            self.train_reward[index] = -5
            return False
        elif direction == 3 and head % self.map_size == self.map_size - 1:  # 右
            self.snake_over[index] = 1
            self.train_reward[index] = -5
            return False
        
        # 处理碰到蛇身
        if newHead in self.snake[index]:
            self.snake_over[index] = 1
            self.train_reward[index] = -5
            return False
            
        self.snake[index].appendleft(newHead)
        self.maps[index,0,newHead//6,newHead%6] = 2
        self.maps[index,0,head//6,head%6] = 1
        
        if self.reward[index] == newHead:
            self.snake_length[index] += 1
            self.sum_length += 1 
            self.train_reward[index] = 10
            while True:
                new_reward = random.randint(0, self.map_size * self.map_size - 1)
                if new_reward not in self.snake[index]:
                    self.reward[index] = new_reward
                    new_row = new_reward//self.map_size
                    new_col = new_reward%self.map_size
                    self.maps[index,0,new_row,new_col] = 3
                    break
        else:
            tail_place = self.snake[index][int(self.snake_length[index].item())]
            tail_row = tail_place//6
            tail_col = tail_place%6
            self.maps[index,0,tail_row,tail_col] = 0
            self.snake[index].pop()
        
        self.snake_steps[index] += 1
        self.sum_steps += 1
        self.train_reward[index] += 0.1
        return True

    def move_all(self, directions):
        if self.log is True:
            self.print_log(index=0)
        for i in range(self.map_num):
            direction = directions[i]
            self.move((int(direction)), i)
        if self.Watch is True:
            self.all_watch()

    def showMap(self, index=0, LINE=True):
        block_size = 8
        image = np.ones((self.map_size * block_size + 1, 
                       self.map_size * block_size + 1), dtype=np.float32)
        
        # 填充蛇身
        color = 0
        if self.snake_over[index] == 1:
            color = 0.4
        for place in self.snake[index]:
            row = place // self.map_size * block_size
            col = place % self.map_size * block_size
            image[row:row + block_size, col:col + block_size] = color
        
        # 填充奖励
        row = self.reward[index] // self.map_size * block_size
        col = self.reward[index] % self.map_size * block_size
        image[row:row + block_size, col:col + block_size] = 0.8
        
        # 绘制网格线
        if LINE is True:
            for r in range(self.map_size + 1):
                image[r * block_size, :] = 0.5
            for c in range(self.map_size + 1):
                image[:, c * block_size] = 0.5
            
        # 绘制图像
        if not hasattr(self, 'fig') or self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 6))
            self.ax.axis('off')
            self.im = self.ax.imshow(image,'gray')
            plt.tight_layout()
            plt.ion()
        else:
            self.im.set_data(image)
            
        plt.draw()
        plt.pause(0.001)
    
    def print_log(self,index=0):
        with open(self.logName,"a") as file:
            file.write(f"current length:{self.snake_length[index]}, "
                       f"steps:{self.snake_steps[index]}\n" 
                       f"{self.maps[index,0]}\n\n")
            file.close()
    
    def getMove(self, direction):
        if direction == 0:  # 上
            return -self.map_size
        elif direction == 1:  # 下
            return self.map_size
        elif direction == 2:  # 左
            return -1
        elif direction == 3:  # 右
            return 1
        else:
            return 1
    
    def watch(self, index):
        head = self.snake[index][0]
        row = head // self.map_size
        col = head % self.map_size
        directions = [
            (-1, 0, row, 0),                    # 上
            (1, 0, self.map_size - row - 1, 3),   # 下
            (0, -1, col, 6),                    # 左
            (0, 1, self.map_size - col - 1, 9),    # 右
            (-1, -1, min(row, col), 6),    # 左上
            (-1, 1, min(row, self.map_size - col - 1), 15),  # 右上
            (1, -1, min(self.map_size - row - 1, col), 18),  # 左下
            (1, 1, min(self.map_size - row - 1, self.map_size - col - 1),21)  
                    # 右下
                    ]
        for _,(row_offset, col_offset, max_steps, feature_offset) in enumerate(directions):
            food_distance = self.map_size
            body_distance = self.map_size
            for step in range(1, max_steps + 1):
                target_row = row + row_offset * step
                target_col = col + col_offset * step
                pos = target_row * self.map_size + target_col
            
            if food_distance == 0 and pos == self.reward[index]:
                food_distance = step
            if body_distance == 0 and pos in self.snake[index]:
                body_distance = step
            if food_distance > 0 and body_distance > 0:
                break
            self.features[index][feature_offset] = food_distance      
            self.features[index][feature_offset + 1] = body_distance  
            self.features[index][feature_offset + 2] = max_steps    

    def all_watch(self):
        for i in range(self.map_num):
            self.watch(i)

def keyRunning():
    Game = game_env(paras=2)
    lastTime = time.time()
    direction = 0
    while(1):
        if Game.snake_over[0] == 1:
            print("Game Over!")
            break
        Game.showMap()
        curTime = time.time()
        if curTime - lastTime >= 0.2:
            Game.move_all([direction,direction])
            lastTime = curTime
        if msvcrt.kbhit():
            key = msvcrt.getch()
            key = key.decode('utf-8').lower()
            if key == 'w': 
                direction = 0
            elif key == 's': 
                direction = 1
            elif key == 'a':  
                direction = 2
            elif key == 'd':  
                direction = 3
            else:
                print("invalid move!")
                direction = 0
        Game.showMap()

def main():
    keyRunning()