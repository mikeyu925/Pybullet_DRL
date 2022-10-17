import math

import utils
from utils.camera import  *
import numpy as np
import cv2

'''
[r g b] => [x y z]
'''

'''
摄像头固定在车头
'''


if __name__ == '__main__':
    serve_id = p.connect(p.GUI)  # 对物理引擎进行链接
    p.setGravity(0,0,-10) # 设置重力加速度

    # 配置渲染机制
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)# 关闭渲染
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 取消渲染时候周围的控制面板

    # 加载场景及模型
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 使用相对路径读入库中现成的模型
    planeId = p.loadURDF("plane.urdf") # 加载urdf模型

    # 加载机器人，并设置加载的机器人的位姿
    startPos = [-0.25, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0 , 0, 0]) # 从 欧拉角 转换得到 四元数 np.pi/20
    robot_id = p.loadURDF("husky/husky.urdf",startPos,startOrientation)
    # robot_id = p.loadURDF("racecar/racecar.urdf")
    p.resetBasePositionAndOrientation(robot_id, startPos, startOrientation)
    # 开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(0)

    # # 创建一面墙
    wall_right_id = utils.createWall([9, 1, 1], [0, 10, 1])
    wall_left_id = utils.createWall([9, 1, 1], [0, -10, 1])
    wall_down_id = utils.createWall([1, 10, 1], [10, 0, 1])
    wall_top_id = utils.createWall([1, 10, 1], [-10, 0, 1])

    line = utils.createTrickLine([0,0,0],4)
    print(line)
    # 在仿真界面自定义参数滑块，分别为速度，驱动力、转向
    base_speed_Slider = p.addUserDebugParameter("Base Speed", -50, 50, 0)  # 参数名称 最小值 最大值 起始值
    front_left_diif_Slider = p.addUserDebugParameter("Front Left Diff Speed", -10, 10, 0)
    front_right_diif_Slider = p.addUserDebugParameter("Front Right Diff Speed", -10, 10, 0)
    rear_left_diif_Slider = p.addUserDebugParameter("Rear Left Diff Speed", -10, 10, 0)
    rear_right_diif_Slider = p.addUserDebugParameter("Rear Right Diff Speed", -10, 10, 0)

    # utils.showJointInformation("赛车",robot_id)

    while True:
        p.stepSimulation()

        imgInfo = setCameraOnRaceCarAndGetImg(robot_id,64,64,serve_id)
        [width, height, rgbaImg, depthImg, segImg] = imgInfo
        rgbaImg = np.array(rgbaImg)
        # rgbaImg = np.rot90(rgbaImg,1) # 逆时针旋转90度
        # rgbaImg = np.rot90(rgbaImg, -1)  # 顺时针旋转90度
        liner_regression(rgbaImg)

        base_speed = p.readUserDebugParameter(base_speed_Slider)
        front_left_speed = p.readUserDebugParameter(front_left_diif_Slider) + base_speed
        front_right_speed = p.readUserDebugParameter(front_right_diif_Slider) + base_speed
        rear_left_speed = p.readUserDebugParameter(rear_left_diif_Slider) + base_speed
        rear_right_speed = p.readUserDebugParameter(rear_right_diif_Slider) + base_speed

        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=[2,3,4,5],
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[front_left_speed,front_right_speed,rear_left_speed,rear_right_speed],
            forces=[100,100,100,100]
        )

        sleep(1 / 240)

    p.disconnect(serve_id)