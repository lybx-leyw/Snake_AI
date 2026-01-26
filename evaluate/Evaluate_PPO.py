import torch
import torch.nn as nn
import models.PV_Network as pv_net
import components.Game_Env as env
import msvcrt
import random
import numpy as np
import time

def evaluate_show(noise=0.0,path="snake_ppo_eva.pkl"):

    model = pv_net.policy_value()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    if isinstance(state_dict,nn.Module):
        model = state_dict
    else:
        model.load_state_dict(state_dict)

    Game = env.game_env(paras=1,printLOG=True,logNAME="logs_ppo_eva.txt")
    while True:
        lastTime = time.time()
        direction = 0
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch()
                key = key.decode('utf-8').lower()
                if key == 'k':
                    print("kill!")
                    break
                elif key == 'o':
                    print("over!")
                    return
            if Game.snake_over[0] == 1:
                print("dead!")
                break
            Game.showMap()
            curTime = time.time()
            if curTime - lastTime >= 0.2:
                direction,_ = model(Game.maps)
                if torch.rand(1) < noise:
                    direction = torch.randint(0,4,(1,))
                else:
                    direction = torch.argmax(direction,dim=-1)
                Game.move_all(direction)
                lastTime = curTime
            Game.showMap()
        Game.reset()

def evaluate_calc(path="snake_ppo_eva.pkl",index=97,Show=True):

    model = pv_net.policy_value()
    state_dict = torch.load(path,map_location=torch.device('cpu'))
    if isinstance(state_dict,nn.Module):
        model = state_dict
    else:
        model.load_state_dict(state_dict)

    Game = env.game_env(paras=1000,printLOG=False)
    if Show == True:
        Game.showMap(index=index)
        print("按任意键开始...")
        msvcrt.getch()

    direction = torch.zeros(Game.map_num)
    while True:
        if Show == True:
            Game.showMap(index=index)
        if Game.snake_over.sum() == 2*Game.map_num or Game.sum_steps//Game.map_num>80:
            print("evaluate_over!")
            break
        direction,_ = model(Game.maps)
        direction = torch.argmax(input=direction,dim=1)
        Game.move_all(direction)
        if Show == True:
            Game.showMap(index=index)
    
    max_length,max_index = torch.max(input=Game.snake_length,dim=-1)
    print(f"平均蛇长:{Game.sum_length/Game.map_num}")
    print(f"最大蛇长:{max_length},索引:{max_index}")
    return max_index

def evaluate_ppo():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if int(input("输入1计算模型性能:")) == 1:
        evaluate_calc(path="best\\snake_ppo.pkl",index=479,Show=True)

    if int(input("输入1演示模型性能:")) == 1:
        evaluate_show(noise=0.0,path="best\\snake_ppo112.pkl")   
    print("评估完毕")     