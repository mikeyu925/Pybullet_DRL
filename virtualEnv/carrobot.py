import pybullet as p
import pybullet_data
import math
import time


def loadCarRobotURDF():
    # 加载机器人，并设置加载的机器人的位姿
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0 , -math.pi / 2,math.pi / 2]) # 从 欧拉角 转换得到 四元数
    robot_id = p.loadURDF("/home/ywh/anaconda3/pkgs/pybullet-3.21-py37he8f5f7f_1/lib/python3.7/site-packages/pybullet_data/miniBox.urdf",startPos,startOrientation)

    return robot_id