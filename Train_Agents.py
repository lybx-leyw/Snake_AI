from train.DQN_train import train_DQN
from train.Partial_Rainbow_train import train_DQN_part_rainbow
from train.PPO_train import train_PPO
from train.PPO_train_dec import train_PPO_dec
from train.GRPO_train_sigstep import train_GRPO_sig
from train.GRPO_train_allsteps import train_GRPO_all

choice = int(input("请输入你想要进行的训练脚本号:\n"
                   "1:double DQN\n"
                   "2:rainbow DQN\n"
                   "3.PPO,4.PPO_dec\n"
                   "5.GRPO_sigstep\n"
                   "6.GRPO_allsteps")
             )
if choice == 1:
    train_DQN()
elif choice == 2:
    train_DQN_part_rainbow()
elif choice == 3:
    train_PPO()
elif choice == 4:
    train_PPO_dec()
elif choice == 5:
    train_GRPO_sig()
elif choice == 6:
    train_GRPO_all()
else:
    print("非法输入")