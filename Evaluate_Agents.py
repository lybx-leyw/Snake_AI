from evaluate.Evaluate_GRPO import evaluate_grpo
from evaluate.Evaluate_PPO import evaluate_ppo
from evaluate.Evaluate_Q import evaluate_q

choice = int(input("请输入你想要测试的模型号:(1:grpo,2:ppo,3:dqn)"))
if choice == 1:
    evaluate_grpo()
elif choice == 2:
    evaluate_ppo()
elif choice == 3:
    evaluate_q()
else:
    print("非法输入")