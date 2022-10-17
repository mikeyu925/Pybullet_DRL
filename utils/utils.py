from parameter import args
import numpy as np
import torch
from matplotlib import pyplot as plt
import pybullet as p

def connetPybulletServer(isOpenGUI):
    if isOpenGUI == True:
        return p.connect(p.GUI)
    else:
        return p.connect(p.DIRECT)

def fanin_init(size, fanin=None):
    # 一种比较合理的初始化网络参数https://arxiv.org/abs/1502.01852
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    x = torch.Tensor(size).uniform_(-v, v)
    return x.type(torch.FloatTensor)

def trapezoidal_function(time):
    """
    Get the value of the trace curve at the time moment.
    Total simulation time is 0.5s.
    得到该时刻的轨迹曲线值。总仿真时间为0.5s。
    :param time:
    :return:
    """
    height = args.height  # 1000
    if time <= 0.075:
        return (height / 0.075) * time
    elif 0.075 < time and time <= 0.375:
        return height
    elif 0.375 < time and time <= 0.45:
        return (-height / 0.075) * time + 6 * height
    elif 0.45 < time and time <= 0.5:
        return 0

def sin_curve(time):
    return np.sin(time) * 100

def line_curve(time,x):
    if isinstance(time, list):
        return [x for _ in range(len(time))]
    else:
        return x

def step_function(time):
    if isinstance(time,list):
        n =len(time)
        y = [ 0 for _ in range((int)(n/5))]
        for _ in range(n - ((int)(n/5))):
            y.append(100)
        return y

def  horizontal_line(time,height):
        n =len(time)
        y = [height for _ in range(n)]
        return y

def square_line(time,height):
    x1 = (int)(len(time) / 4)
    x2 = (int)(len(time) / 2)
    x3 = (int)(len(time) * 3 / 4)
    y = [height for i in range(x1)]
    for i in range(x1,x2):
        y.append(0)
    for i in range(x2,x3):
        y.append(height)
    for i in range(x3,len(time)):
        y.append(0)
    return y

def show_resutlt(x,target,real):
    plt.plot(x, target, 'r-',linewidth=0.5)
    plt.plot(x,real, 'b-',linewidth=0.5)
    plt.xlabel('time')
    plt.ylabel('position')
    plt.legend(["target_pos", "real_pos"], loc='lower right')
    plt.show()

def action_limit(action):
    # 对积分部分进行再一次限幅
    action[1] = action[1] / args.action_bound * args.i_limit
    # limit = 0.3
    # if action[1] >= limit:
    #     action[1] = limit
    # elif action[1] <= -limit:
    #     action[1] = -limit
    return action



