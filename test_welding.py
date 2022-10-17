from utils.camera import  *
'''
[r g b] => [x y z]
'''

if __name__ == '__main__':
    server_id = p.connect(p.GUI)  # 对物理引擎进行链接
    p.setGravity(0,0,-10) # 设置重力加速度

    # 配置渲染机制
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)     # 关闭渲染
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 取消渲染时候周围的控制面板
    # 加载场景及模型
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 使用相对路径读入库中现成的模型
    planeId = p.loadURDF("plane.urdf") # 加载urdf模型
    # 加载机器人，并设置加载的机器人的位姿
    startPos = [0, 0, 0]
    startOrientation = p.getQuaternionFromEuler([0, -math.pi / 2, 0])  # 从 欧拉角 转换得到 四元数
    robot_id = p.loadURDF("/home/ywh/anaconda3/pkgs/pybullet-3.21-py37he8f5f7f_1/lib/python3.7/site-packages/pybullet_data/装配体10.SLDASM/urdf/装配体10.SLDASM.urdf",startPos,startOrientation)


    # 开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(0)


    # 在仿真界面自定义参数滑块，分别为速度，驱动力、转向  # 参数名称 最小值 最大值 起始值
    x_speed_Slider = p.addUserDebugParameter("x", -1, 1, 0)
    y_speed_Slider = p.addUserDebugParameter("y", -1, 1, 0)
    z_speed_Slider = p.addUserDebugParameter("z", -1, 1, 0)

    # virtualEnv.resetWeldingRobot(robot_id)# 参数名称 最小值 最大值 起始值
    targetLineID = p.addUserDebugLine([-0.2+0.043,0,0.05], [0.2+0.043,0,0.05], lineColorRGB=[1, 0, 0], lineWidth=1)
    targetVal = 0.0

    cnt = 0
    while True:
        p.stepSimulation()

        x_speed = p.readUserDebugParameter(x_speed_Slider)
        y_speed = p.readUserDebugParameter(y_speed_Slider)
        z_speed = p.readUserDebugParameter(z_speed_Slider)
        setGunCameraAndGetImg(robot_id,320,320,server_id)
        for i in range(p.getNumJoints(robot_id)):
            if i == 0:  #
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=0,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=x_speed,
                    force=10
                )
            elif i == 1:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=1,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=y_speed,
                    force=10
                )
            elif i == 2:
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=2,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=z_speed,
                    force=100
                )
            # 元组形式 获取关节状态信息
        if cnt % 100 == 0:
            print("----------------------------------------")
            print("x:" + str(p.getJointState(robot_id,0)[0]))
            print("y:" + str(p.getJointState(robot_id, 1)[0]))
            print("z:" + str(p.getJointState(robot_id, 2)[0]))
            print("++++++++++++++++++++++++++++++++++++++++")
        cnt = (cnt + 1) % 1000
        sleep(1 / 240)

    # p.disconnect(serve_id)