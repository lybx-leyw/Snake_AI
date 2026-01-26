import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import models.Q_Network as q_net
import components.Game_Env as env
import components.Replay as rep

def train_DQN():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("当前使用设备为:CPU，速度可能更慢哦")
    else:
        print("使用设备:GPU")

    # 环境搭建
    Game = env.game_env(paras=200)
    Q_net = q_net.q_network().to(device)
    state_dict = torch.load(f"snake_q.pkl")
    if isinstance(state_dict,nn.Module):
        Q_net = state_dict
    else:
        Q_net.load_state_dict(state_dict)
    target_Q_net = q_net.q_network().to(device)
    target_Q_net.load_state_dict(Q_net.state_dict())

    # 参数表
    max_train_samples=1024
    max_memories = 50000
    Replay = rep.replay(maxlen=max_memories,train_samples=max_train_samples)
    explore_rate = 0.02
    lr = 0.001
    max_steps = Game.map_num*40
    optimizer = optim.Adam(Q_net.parameters(), lr=lr)
    games = 0

    with open("train_log.txt","a",encoding='utf-8') as file:
        file.write("训练日志\n""本次训练参数：\n"
                   f"\tparas:{Game.map_num}\n"
                   f"\t学习率lr:{lr}\n"
                   f"\tmax_steps:{max_steps//Game.map_num}\n"
                   f"\t经验回放类大小:{max_memories}\n"
                   f"\t一次采样数：{max_train_samples}\n")
        file.close()

        while (1):
            # 若游戏结束或达到限定步数，开始新的一局
            if Game.sum_steps > max_steps or Game.snake_over.sum() == Game.map_num*2:
                games += 1
                if games%20 == 0:
                    target_Q_net.load_state_dict(Q_net.state_dict())
                    explore_rate = max(0.99*explore_rate,1e-2)
                    torch.save(Q_net.state_dict(), f"history\\snake_q{465+games//20}.pkl")
                    torch.save(Q_net.state_dict(), f"history\\snake_q.pkl")

                    # 打印信息
                    print(f"第{games//20}次更新, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                        f"小蛇平均存活步数为:{Game.sum_steps/Game.map_num:.2f}, "
                        f"探索率为:{explore_rate:.2f}")
                    with open("train_log.txt","a",encoding='utf-8') as file:
                        file.write(f"第{games//20}局, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                                    f"小蛇平均存活步数为:{Game.sum_steps/Game.map_num:.2f}, "
                                    f"探索率为:{explore_rate:.2f}\n")
                        file.close()

                if Game.sum_length/Game.map_num > 35:
                    print("训练完毕！")
                    break

                Game.reset()


            current_S = Game.maps.clone()
            current_Length = Game.snake_length.clone()

            # 先根据当前状态采样
            current_Q = Q_net(Game.maps.clone().to(device))
            # 选择最优动作
            current_A = torch.argmax(input=current_Q,dim=1)

            # 贪婪算法
            explore_mask = torch.rand(Game.map_num)<explore_rate
            explore_map = torch.randint(0,4,(Game.map_num,))
            current_A[explore_mask] = explore_map[explore_mask].to(device)

            '''
            print(current_Q.shape) #[50,4]
            print(current_A)
            '''

            # 走下一步（其实也是模拟）
            Game.move_all(current_A)

            current_R = Game.train_reward.clone()
            current_Done = Game.snake_over.clone()
            '''
            print(current_R)
            '''
            # 记录经验
            Replay.pull_memory(current_S,current_A.clone(),current_R,Game.maps.clone(),current_Done)

            # 提取经验进行优化
            for i in range(3):
                samples = Replay.randget_memory()
                all_s = torch.cat([s for s,_,_,_,_ in samples]).to(device)
                all_ns = torch.cat([ns for _,_,_,ns,_ in samples]).to(device)
                all_r = torch.cat([r for _,_,r,_,_ in samples]).to(device)
                all_a = torch.cat([a for _,a,_,_,_ in samples]).to(device)
                all_d = torch.cat([done for _,_,_,_,done in samples]).to(device)

                mask = (all_d == 1)

                # 批量计算Q值
                train_current_q = Q_net(all_s)
                train_current_Q = torch.gather(train_current_q, dim=1, index=all_a.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    # 按照在线网络评估最优动作
                    train_next_q = Q_net(all_ns)
                    _,train_next_a = torch.max(train_next_q, dim=1)

                    # 使用目标网络评估下一步动作
                    train_next_q = target_Q_net(all_ns)
                    train_next_Q = torch.gather(train_next_q, dim=1, index=train_next_a.unsqueeze(1)).squeeze(1)

                    # 计算目标Q值
                    train_target_Q = all_r + train_next_Q*0.9

                train_target_Q[mask] = all_r[mask]
                '''
                print(train_target_Q)
                '''
                L_DQN = F.mse_loss(train_target_Q,train_current_Q)

                '''
                print(L_DQN)
                '''

                # 反向传播
                optimizer.zero_grad()
                L_DQN.backward()

                # 梯度下降
                optimizer.step()

                # 打印信息
                '''
                print(f"loss:{L_DQN.item()}")
                '''