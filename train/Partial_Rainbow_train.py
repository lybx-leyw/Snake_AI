import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque
import models.Q_Network as q_net
import components.Game_Env as env
import components.Replay_Rainbow as rep

def train_DQN_part_rainbow():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("当前使用设备为:CPU，速度可能更慢哦")
    else:
        print("使用设备:CUDA")

    # 环境搭建
    Game = env.game_env(paras=200)
    Q_net = q_net.q_network().to(device)
    try:
        state_dict = torch.load(f"snake_q.pkl")
        if isinstance(state_dict,nn.Module):
            Q_net = state_dict
        else:
            Q_net.load_state_dict(state_dict)
    except:
        print("从头开始训练")
    target_Q_net = q_net.q_network().to(device)
    target_Q_net.load_state_dict(Q_net.state_dict())

    # 参数表
    max_train_samples=512
    max_memories = 50000
    Replay = rep.replay(maxlen=max_memories,train_samples=max_train_samples)
    lr = 0.001
    tau = 0.01
    max_steps = 30
    optimizer = optim.Adam(Q_net.parameters(), lr=lr)
    games = 0
    temperature = 1
    deque_length = deque(maxlen=100)

    # 冻结图像处理层
    '''
    for name, param in Q_net.named_parameters():
        if 'conv' in name or 'fc.0' in name:
            param.requires_grad = False
            print(f"冻结层: {name}")
    '''
    with open("train_log.txt","a",encoding='utf-8') as file:
        file.write("训练日志\n""本次训练参数：\n"
                   f"\tparas:{Game.map_num}\n"
                   f"\t学习率lr:{lr}\n"
                   f"\tmax_steps:{max_steps}\n"
                   f"\t经验回放类大小:{max_memories}\n"
                   f"\t一次采样数：{max_train_samples}\n")
        file.close()

        while (1):
            # 若游戏结束或达到限定步数，开始新的一局
            if Game.sum_steps/Game.map_num > 40 or Game.snake_over.sum() == Game.map_num*2:
                games += 1
                for target_param, current_param in zip(target_Q_net.parameters(), Q_net.parameters()):
                    target_param.data.copy_(tau*current_param.data+(1-tau)*target_param.data)
                if games%20 == 0:
                    temperature = max(0.995*temperature,1e-2)
                    torch.save(Q_net.state_dict(), f"history\\snake_q{games//20}.pkl")
                    torch.save(Q_net.state_dict(), f"history\\snake_q.pkl")

                    # 打印信息
                    print(f"第{games//20}次更新, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                        f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}, "
                        f"当前小蛇近100局平均长度为:{sum(deque_length)/len(deque_length):.2f}, "
                        f"温度为:{temperature:.4f}")
                    with open("train_log.txt","a",encoding='utf-8') as file:
                        file.write(f"第{games//20}次更新, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                                    f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}, "
                                    f"当前小蛇近100局平均长度为:{sum(deque_length)/len(deque_length):.2f}, "
                                    f"温度为:{temperature:.4f}, "
                                    f"模型序号:{games//20}\n")
                        file.close()

                deque_length.appendleft(Game.sum_length/Game.map_num)
                ''''''
                print(f"第{games}局, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                        f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}, "
                        f"当前小蛇近100局平均长度为:{sum(deque_length)/len(deque_length):.2f}, "
                        f"温度为:{temperature:.4f}")
                ''''''
                if Game.snake_steps.sum()/Game.map_num > 35:
                    print("训练完毕！")
                    break

                Game.reset()
                continue

            current_S = Game.maps.clone()
            current_Length = Game.snake_length.clone()

            # 先根据当前状态采样
            current_Q = Q_net(Game.maps.clone().to(device))

            # 加权探索
            epsilon = 0.05
            if random.random() < epsilon:
                current_A = torch.randint(0, 4, (Game.map_num,), device=device)
            else:
                probs = F.softmax(current_Q/temperature, dim=1)
                current_A = torch.multinomial(probs, num_samples=1).squeeze(-1)

            '''
            print(current_Q.shape) #[50,4]
            print(current_A)
            '''

            # 走下一步（其实也是模拟）
            Game.move_all(current_A)

            current_Done = Game.snake_over.clone()
            current_R = Game.train_reward.clone()
            '''
            print(current_R)
            '''
            # 记录经验
            td_q_values = torch.gather(current_Q, dim=1, index=current_A.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                td_target_q_values = target_Q_net(Game.maps.clone().to(device))
                td_target_q_values = torch.gather(td_target_q_values, dim=1, index=current_A.unsqueeze(1)).squeeze(1)
            td_error = abs(td_q_values-td_target_q_values)
            Replay.pull_memory(current_S,current_A.clone(),current_R,Game.maps.clone(),current_Done,td_error.clone())

            # 预热
            if len(Replay.good) + len(Replay.dead) + len(Replay.normal) < 0.02*max_memories:
                continue

            # 提取经验进行优化
            for i in range(3):
                samples = Replay.randget_memory()
                all_s = torch.cat([s for s,_,_,_,_,_ in samples]).clone().to(device)
                all_ns = torch.cat([ns for _,_,_,ns,_,_ in samples]).clone().to(device)
                all_r = torch.cat([r for _,_,r,_,_,_ in samples]).clone().to(device)
                all_a = torch.cat([a for _,a,_,_,_,_ in samples]).clone().to(device)
                all_d = torch.cat([done for _,_,_,_,done,_ in samples]).clone().to(device)

                # 数据增强
                transform_type = random.choice(['rot90', 'rot180', 'rot270', 'transpose', 'flip_h', 'flip_v', 'none'])
                def transform_action(actions, transform_type):
                    if transform_type == 'none':
                        return actions
                    if transform_type == 'rot90':
                        # 上(0) -> 右(3), 右(3) -> 下(1), 下(1) -> 左(2), 左(2) -> 上(0)
                        action_map = {0: 3, 1: 2, 2: 0, 3: 1}
                    elif transform_type == 'rot180':
                        # 上(0) -> 下(1), 下(1) -> 上(0), 左(2) -> 右(3), 右(3) -> 左(2)
                        action_map = {0: 1, 1: 0, 2: 3, 3: 2}
                    elif transform_type == 'rot270':
                        # 上(0) -> 左(2), 左(2) -> 下(1), 下(1) -> 右(3), 右(3) -> 上(0)
                        action_map = {0: 2, 1: 3, 2: 1, 3: 0}
                    elif transform_type == 'transpose':
                        # 上(0) -> 左(2), 左(2) -> 上(0), 下(1) -> 右(3), 右(3) -> 下(1)
                        action_map = {0: 2, 1: 3, 2: 0, 3: 1}
                    elif transform_type == 'flip_h':
                        # 左(2) -> 右(3), 右(3) -> 左(2), 上(0)和下(1)保持不变
                        action_map = {0: 0, 1: 1, 2: 3, 3: 2}
                    elif transform_type == 'flip_v':
                        # 上(0) -> 下(1), 下(1) -> 上(0), 左(2)和右(3)保持不变
                        action_map = {0: 1, 1: 0, 2: 2, 3: 3}
                    else:
                        return actions

                    transformed_actions = torch.tensor([action_map[a.item()] if a.item() in action_map else a.item()
                                                    for a in actions], device=device)
                    return transformed_actions

                # 对状态进行变换
                if transform_type == 'rot90':
                    all_s = torch.rot90(all_s, k=1, dims=[-2, -1])
                    all_ns = torch.rot90(all_ns, k=1, dims=[-2, -1])
                elif transform_type == 'rot180':
                    all_s = torch.rot90(all_s, k=2, dims=[-2, -1])
                    all_ns = torch.rot90(all_ns, k=2, dims=[-2, -1])
                elif transform_type == 'rot270':
                    all_s = torch.rot90(all_s, k=3, dims=[-2, -1])
                    all_ns = torch.rot90(all_ns, k=3, dims=[-2, -1])
                elif transform_type == 'transpose':
                    all_s = all_s.transpose(-2, -1)
                    all_ns = all_ns.transpose(-2, -1)
                elif transform_type == 'flip_h':
                    all_s = torch.flip(all_s, dims=[-1])
                    all_ns = torch.flip(all_ns, dims=[-1])
                elif transform_type == 'flip_v':
                    all_s = torch.flip(all_s, dims=[-2])
                    all_ns = torch.flip(all_ns, dims=[-2])

                all_a = transform_action(all_a, transform_type)

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
                L_DQN = F.smooth_l1_loss(train_target_Q, train_current_Q)

                '''
                print(L_DQN)
                '''

                # 反向传播
                optimizer.zero_grad()
                L_DQN.backward()

                # 梯度下降
                torch.nn.utils.clip_grad_norm_(Q_net.parameters(), max_norm=0.5)
                optimizer.step()

                # 打印信息
                '''
                print(f"loss:{L_DQN.item()}")
                '''