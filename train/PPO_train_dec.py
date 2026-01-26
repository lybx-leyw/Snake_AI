import torch
import torch.nn as nn
import torch.optim as optim
import models.PV_Network as pv_net
import components.Game_Env as env
import components.Buffer_PPO as rep

def train_PPO_dec():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cpu"):
        print("当前使用设备为:CPU，速度可能更慢哦")
    else:
        print("使用设备:CUDA")

    # 1.环境搭建
        # 游戏环境
    Game = env.game_env(paras=500) #只能按顺序储存，这意味着需要paras个缓存区
        # 在线网络
    online_net = pv_net.policy_value().to(device)
    try:
        state_dict = torch.load(f"../history_ppo/snake_ppo.pkl")
        if isinstance(state_dict,nn.Module):
            online_net = state_dict.to(device)
        else:
            online_net.load_state_dict(state_dict)
    except:
        print("没有预训练模型，从头开始训练")
        # 参考网络
    R_net = pv_net.policy_value().to(device)
    R_net.load_state_dict(online_net.state_dict())

    # 2.参数表
    lr = 0.001

    clip_param = 0.2
    ppo_epoch = 10
    entropy_coef = 0.01
    value_loss_coef = 0.5

    gamma = 0.99
    gae_lambda = 0.95

    update_freq = 10

    max_steps = Game.map_num * 100

    temperature_start = 1.29
    temperature_end = 0.05
    temperature_decay = 0.99

    optimizer = optim.Adam(online_net.parameters(), lr=lr)

    T = 2000

    # 训练状态变量
    games = 143*update_freq
    temperature = temperature_start
    Replay = rep.replay(maxlen=max_steps//Game.map_num,paras=Game.map_num)

    with open("../logs/train_log_ppo.txt", "a", encoding='utf-8') as file:
        pass

    while (1):
        # 若游戏结束或达到限定步数，结束采样，开始训练
        if Game.sum_steps > max_steps or Game.snake_over.sum() == Game.map_num*2:
            '''
            # 打印一条轨迹看看
            print(Replay.buffers[6])
            Replay.clear_memories()
            print(Replay.buffers[6])
            break
            '''
            A = [[]for _ in range(Game.map_num)]
            # 计算每一条轨迹的GAE
            for i in range(Game.map_num):
                delta = []
                for t in range(len(Replay.buffers[i])):
                    # 解包
                    s_t,a_t,r_t,prob_t,V_t,v_nt,done = Replay.buffers[i][t]
                    s_t = s_t.to(device)
                    a_t = a_t.to(device)
                    r_t = r_t.to(device)
                    prob_t = prob_t.to(device)
                    V_t = V_t.to(device)
                    done = done.to(device)
                    if done != 1:
                        delta_t = r_t + gamma*v_nt - V_t
                    else:
                        delta_t = r_t
                        # print(r_t)
                    delta.append(delta_t)

                # 利用delta组合获得每个时间步的GAE
                for t in range(len(delta)):
                    left = max(0,t-T+1)
                    right = t
                    A_t = torch.zeros(1,1).to(device)
                    for j in range(left,right+1):
                        index = j-left
                        A_t += ((gae_lambda*gamma)**index)*delta[j].to(device)
                    A[i].append(A_t.clone())
            '''
            print(A[8])
            break
            '''
            # 将所需量转化为张量
            all_samples = Replay.get_memories()
            all_s = []
            all_a = []
            all_r = []
            all_prob = []
            all_v = []
            all_nv = []
            all_d = []
            all_A = []
            for i in range(Game.map_num):
                samples = all_samples[i]
                all_s.append(torch.cat([s for s,_,_,_,_,_,_ in samples]).to(device))
                all_a.append(torch.cat([a for _,a,_,_,_,_,_ in samples]).to(device))
                all_r.append(torch.cat([r for _,_,r,_,_,_,_ in samples]).to(device))
                all_prob.append(torch.cat([log_prob for _,_,_,log_prob,_,_,_ in samples]).to(device))
                all_v.append(torch.cat([v for _,_,_,_,v,_,_ in samples]).to(device))
                all_nv.append(torch.cat([nv for _,_,_,_,_,nv,_ in samples]).to(device))
                all_d.append(torch.cat([done for _,_,_,_,_,_,done in samples]).to(device))
                all_A.append(torch.tensor(A[i]))
            all_s = torch.cat(all_s).to(device)
            all_a = torch.cat(all_a).to(device)
            all_r = torch.cat(all_r).to(device)
            all_prob = torch.cat(all_prob).to(device)
            all_v = torch.cat(all_v).to(device)
            all_nv = torch.cat(all_nv).to(device)
            all_d = torch.cat(all_d).to(device)
            all_A = torch.cat(all_A).to(device)
            '''
            print(all_A.shape)
            print(all_A)
            print(all_nv.shape)
            '''

            for _ in range(ppo_epoch):
                online_P,online_V = online_net(all_s.clone().to(device))
                S_probs = torch.softmax(online_P, dim=1)
                online_probs = torch.gather(input=S_probs,dim=1,index=all_a.unsqueeze(1)).squeeze(1)
                online_log_probs = torch.log(online_probs)

                # 计算Radio
                Radio = torch.exp(online_log_probs-all_prob)
                # print(Radio)

                # clip剪切
                Radio_clipped = torch.clamp(Radio, 1-clip_param, 1+clip_param)
                Lclip = -torch.min(Radio*all_A,Radio_clipped*all_A)
                # print(Lclip)

                # 计算Lvf
                td_error = online_V.squeeze(1) - ( all_r + all_nv.squeeze(1)*(1-all_d) )
                Lvf = value_loss_coef*(td_error)**2
                # print(Lvf)

                # 计算熵奖励
                S_bonus = entropy_coef*(S_probs*torch.log(S_probs+1e-10)).sum(dim=1)
                # print(S_bonus)

                # print(Lclip.shape,Lvf.shape,S_bonus.shape)

                # 组合成最终损失函数
                Lfinal = (Lclip+Lvf-S_bonus).mean()
                # print(Lfinal)

                optimizer.zero_grad()
                Lfinal.backward()

                # 梯度下降
                optimizer.step()

            games += 1
            temperature *= temperature_decay
            temperature = max(temperature,temperature_end)
            if games%update_freq == 0:
                R_net.load_state_dict(online_net.state_dict())
                torch.save(online_net.state_dict(), f"history\\snake_ppo{games//update_freq}.pkl")
                torch.save(online_net.state_dict(), f"history\\snake_ppo.pkl")
                with open("../logs/train_log_ppo.txt", "a", encoding='utf-8') as file:
                    file.write(f"第{games//update_freq}次更新, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                            f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}\n")

            print(f"第{games}局, 当前小蛇平均长度为:{Game.sum_length/Game.map_num:.2f}, "
                    f"小蛇平均存活步数为:{Game.snake_steps.sum()/Game.map_num:.2f}, "
                    f"温度:{temperature:.2f}")

            Replay.clear_memories()
            Game.reset()
            continue

        current_S = Game.maps.clone()
        current_Length = Game.snake_length.clone()

        # 用在线网络模拟整场游戏
        current_P,_ = online_net(current_S.clone().to(device))

        # 探索算法
        probs = torch.softmax(current_P, dim=1)
        current_A = torch.multinomial(probs/temperature, 1).squeeze(1)

        # 记录其余需要记录的量
        with torch.no_grad():
            refer_P,refer_V = R_net(current_S.clone().to(device))
            dif_refer_probs = torch.gather(input=probs,dim=1,index=current_A.unsqueeze(1)).squeeze(1)
            refer_log_probs = torch.log(dif_refer_probs)

        # 走下一步获取当前步的reward
        Game.move_all(current_A)
        current_R = Game.train_reward.clone()
        current_Done = Game.snake_over.clone()
        with torch.no_grad():
            _,refer_nV = R_net(current_S.clone().to(device))

        # 记录经验
        Replay.pull_memories(current_S,current_A.clone(),0.1*current_R,refer_log_probs.clone(),refer_V.clone(),refer_nV.clone(),current_Done)