import utils
from utils.camera import  *
'''
[r g b] => [x y z]
'''

'''
摄像头固定在车头
'''


if __name__ == '__main__':
    serve_id = p.connect(p.GUI)  # 对物理引擎进行链接
    p.setGravity(0,0,-10) # 设置中坜加速度

    # 配置渲染机制
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)# 关闭渲染
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 取消渲染时候周围的控制面板

    # 加载场景及模型
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 使用相对路径读入库中现成的模型
    planeId = p.loadURDF("plane.urdf") # 加载urdf模型

    # 加载机器人，并设置加载的机器人的位姿
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0 , 0, math.pi / 2]) # 从 欧拉角 转换得到 四元数
    robot_id = p.loadURDF("/home/ywh/anaconda3/pkgs/pybullet-3.21-py37he8f5f7f_1/lib/python3.7/site-packages/pybullet_data/miniBox.urdf",startPos,startOrientation)
    p.resetBasePositionAndOrientation(robot_id, startPos, startOrientation)
    # 开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(0)

    # 创建一面墙
    wall_right_id = utils.createWall([9, 1, 1], [0, 10, 1])
    wall_left_id = utils.createWall([9, 1, 1], [0, -10, 1])
    wall_down_id = utils.createWall([1, 10, 1], [10, 0, 1])
    wall_top_id = utils.createWall([1, 10, 1], [-10, 0, 1])

    # 在仿真界面自定义参数滑块，分别为速度，驱动力、转向
    base_speed_Slider = p.addUserDebugParameter("Base Speed", -10, 10, 0)  # 参数名称 最小值 最大值 起始值
    left_diif_Slider = p.addUserDebugParameter("Left Diff Speed", -10, 10, 0)
    right_diif_Slider = p.addUserDebugParameter("Right Diff Speed", -10, 10, 0)

    while True:
        p.stepSimulation()

        setCameraOnMiniCarAndGetImg(robot_id,360,360,serve_id)
        base_speed = p.readUserDebugParameter(base_speed_Slider)
        left_diff_speed = p.readUserDebugParameter(left_diif_Slider)
        right_diif_speed = p.readUserDebugParameter(right_diif_Slider)

        for i in range(p.getNumJoints(robot_id)):
            if i == 2:  # 如果是轮子的关节，则为马达配置参数，否则禁用马达  2 右轮 3 左轮
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=2,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=base_speed + right_diif_speed,
                    force=100
                )
            elif i == 3:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=3,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=base_speed + left_diff_speed,
                    force=100
                )

        sleep(1 / 240)

    p.disconnect(serve_id)