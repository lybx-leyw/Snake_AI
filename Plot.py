import re
import matplotlib.pyplot as plt
"""
plot脚本：提取文件数据，绘画可视化折线图
"""

"""
在我的训练日志中，所有包含训练过程的数据都含有多个数字
我们只想绘制平均长度-更新次数的图像，并在Conclusion中详细标注我们每一次更新的次数
"""

# 需要知道我们的平均长度是在文件中的第几个数字
def pause(path,index):
    x_plot = []
    y_plot = []
    with open(path,"r",encoding='utf-8') as file:
        x_index = 0
        for _,line in enumerate(file,1):
            numbers = re.findall(r'-?\d+\.?\d*',line.strip())
            if len(numbers) < 3:
                continue
            else:
                x_index += 1
                x_plot.append(x_index)
                y_plot.append(numbers[index])
    return x_plot,y_plot

def draw(x_plot,y_plot,save_path):
    plt.figure(figsize=(18, 18), dpi=100, facecolor='white')
    plt.plot(x_plot, y_plot, 'b-', linewidth=1, label='avg_length', alpha=0.8)
    plt.title('avg_length_curve', fontsize=8, fontweight='bold')
    plt.xlabel('update_cnt', fontsize=7)
    plt.ylabel('avg_length', fontsize=7)
    plt.grid(True, alpha=0.3, linestyle='--', color='gray')
    plt.legend()
    plt.savefig(save_path,bbox_inches='tight',dpi=100)
    plt.close()

'''
x_plot,y_plot = pause(path="train_log_partial_rainbow.txt",index=1)
draw(x_plot,y_plot,"docs\\partial_rainbow.png")
'''
'''
x_plot,y_plot = pause(path="train_log_ppo_1.txt",index=1)
draw(x_plot,y_plot,"docs\\ppo_1.png")
'''
'''
x_plot,y_plot = pause(path="train_log_ppo_10.txt",index=1)
draw(x_plot,y_plot,"docs\\ppo_10.png")
'''
'''
x_plot,y_plot = pause(path="train_log_ppo.txt",index=1)
draw(x_plot,y_plot,"docs\\ppo_dec.png")
'''
'''
x_plot,y_plot = pause(path="train_log_double.txt",index=1)
draw(x_plot,y_plot,"docs\\double.png")
'''
'''
x_plot,y_plot = pause(path="train_log_grpo.txt",index=1)
draw(x_plot,y_plot,"docs\\grpo.png")
'''
'''
x_plot,y_plot = pause(path="train_log_grpo_all.txt",index=1)
draw(x_plot,y_plot,"docs\\grpo_all.png")
'''