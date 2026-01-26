import torch
import torch.nn as nn
import torch.optim as optim
import models.Q_Network as q_net
import components.Game_Env as env
import components.Buffer_GRPO_all as rep

def train_GRPO_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("当前使用设备为:CPU，速度可能更慢哦")
    else:
        print("使用设备:CUDA")

    # 1.环境搭建
        # 游戏环境
    Game = env.game_env(paras=1000) # 1000个并行环境
        # 在线网络
    online_net = q_net.q_network().to(device)
    try:
        state_dict = torch.load(f"../history_grpo_all/snake_grpo_all.pkl")
        if isinstance(state_dict,nn.Module):
            online_net = state_dict.to(device)
        else:
            online_net.load_state_dict(state_dict)
    except:
        print("没有预训练模型，从头开始训练")
        # 参考网络
    R_net = q_net.q_network().to(device)
    R_net.load_state_dict(online_net.state_dict())

    # 2.参数表
    lr = 0.001

    clip_param = 0.2
    grpo_epoch = 10
    beta = 0.01
    gamma = 0.95

    update_freq = 20

    max_steps = Game.map_num * 50

    temperature_start = 1.29
    temperature_end = 0.05
    temperature_decay = 0.995

    T = 2000

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    # 训练状态变量
    games = 194*update_freq
    temperature = temperature_start
    Replay = rep.replay(maxlen=max_steps//Game.map_num,paras=Game.map_num)

    with open("../logs/train_log_grpo_all.txt", "a", encoding='utf-8') as file:
        pass

    while (1):
        # 若游戏结束或达到限定步数，结束采样，开始训练
        if Game.sum_steps > max_steps or Game.snake_over.sum() == Game.map_num*2:

            A = [[]for _ in range(Game.map_num)]
            # 计算每一条轨迹的累计优势
            for i in range(Game.map_num):
                # 读取每一条轨迹的A_s
                A_s = torch.cat([A_t for _,_,A_t,_,_ in Replay.buffers[i]]).to(device)

                # 利用A_s组合获得每个时间步的A
                for t in range(len(A_s)):
                    left = max(0,t-T+1)
                    right = t
                    t_A = torch.zeros(1,1).to(device)
                    for j in range(left,right+1):
                        index = j-left
                        t_A += ((gamma)**index)*A_s[j].to(device)
                    A[i].append(t_A.clone())

            '''
            print(A[5])
            break
            '''
            # 将所需量转化为张量
            all_samples = Replay.get_memories()
            all_s = []
            all_a = []
            all_prob = []
            all_d = []
            all_A = []
            for i in range(Game.map_num):
                samples = all_samples[i]
                all_s.append(torch.cat([s for s,_,_,_,_ in samples]).to(device))
                all_a.append(torch.cat([a for _,a,_,_,_ in samples]).to(device))
                all_prob.append(torch.cat([log_prob for _,_,_,log_prob,_ in samples]).to(device))
                all_d.append(torch.cat([done for _,_,_,_,done in samples]).to(device))
                all_A.append(torch.tensor(A[i]))

            # 合并本局经验
            all_s = torch.cat(all_s).to(device)
            all_a = torch.cat(all_a).to(device)
            all_prob = torch.cat(all_prob).to(device)
            all_d = torch.cat(all_d).to(device)
            all_A = torch.cat(all_A).to(device)

            for _ in range(grpo_epoch):
                online_Q = online_net(all_s.clone().to(device))
                S_probs = torch.softmax(online_Q, dim=1)
                online_probs = torch.gather(input=S_probs,dim=1,index=all_a.unsqueeze(1)).squeeze(1)
                online_log_probs = torch.log(online_probs)

                # 计算Radio
                Radio = torch.exp(online_log_probs-all_prob)
                # print(Radio)

                # clip剪切
                Radio_clipped = torch.clamp(Radio, 1-clip_param, 1+clip_param)
                Lclip = -torch.min(Radio*all_A,Radio_clipped*all_A)
                # print(Lclip)

                # Dkl惩罚
                Dkl = torch.exp(all_prob-online_log_probs)-all_prob+online_log_probs-1
                # print(Dkl)

                # 组合损失
                Lfinal = (Lclip + beta*Dkl).mean()
                # print(Lfinal)

                optimizer.zero_grad()
                Lfinal.backward()
                optimizer.step()

            games += 1
            Replay.clear_memories()

            if games%update_freq == 0:
                temperature *= temperature_decay
                temperature = max(temperature,temperature_end)
                R_net.load_state_dict(online_net.state_dict())
                torch.save(online_net.state_dict(), f"history\\snake_grpo_all{games//update_freq}.pkl")
                torch.save(online_net.state_dict(), f"history\\snake_grpo_all.pkl")
                with open("../logs/train_log_grpo_all.txt", "a", encoding='utf-8') as file:
                    file.write(f"第{games//update_freq}次更新, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                            f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}\n")

            print(f"第{games}局, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                    f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}, "
                    f"温度:{temperature:.2f}")

            Game.reset()
            continue

        current_S = Game.maps.clone()

        # 用在线网络模拟整场游戏
        current_Q = online_net(current_S.clone().to(device))

        # 探索算法
        probs = torch.softmax(current_Q, dim=1)
        current_A = torch.multinomial(probs/temperature, 1).squeeze(1)

        # 记录其余需要记录的量
        with torch.no_grad():
            refer_Q = R_net(current_S.clone().to(device))
            refer_probs = torch.softmax(input=refer_Q, dim=1)
            refer_probs = torch.gather(input=probs,dim=1,index=current_A.unsqueeze(1)).squeeze(1)
            refer_log_probs = torch.log(refer_probs)

        # 走下一步获取当前步的reward
        Game.move_all(current_A)
        current_R = Game.train_reward.clone()
        current_Done = Game.snake_over.clone()

        # 记录经验
        Replay.pull_buffers(current_S,current_A.clone(),current_R,refer_log_probs.clone(),current_Done)