import pybullet as p
import pybullet_data
from time import sleep
from pid_controller import PID_Controller
from virtualEnv.weldingEnv import *
from virtualEnv.enclosingwall import *
'''
[r g b] => [x y z]
'''
if __name__ == '__main__':
    serverID = p.connect(p.GUI)  # 对物理引擎进行链接
    p.setGravity(0,0,-10) # 设置重力加速度

    # 配置渲染机制
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)     # 关闭渲染
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 取消渲染时候周围的控制面板
    # 加载场景及模型
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 使用相对路径读入库中现成的模型
    planeId = p.loadURDF("plane.urdf") # 加载urdf模型
    desk = loadCube([1,1,1],[0,0,0.5])
    # 加载机器人，并设置加载的机器人的位姿
    robot_id = virtualEnv.loadWeldingRobotURDF([0, 0, 1])
    # 开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(0)

    # 在仿真界面自定义参数滑块，分别为速度，驱动力、转向  # 参数名称 最小值 最大值 起始值
    x_speed_Slider = p.addUserDebugParameter("x", -1, 1, 0)
    y_speed_Slider = p.addUserDebugParameter("y", -1, 1, 0)
    z_speed_Slider = p.addUserDebugParameter("z", -1, 1, 0)

    # virtualEnv.resetWeldingRobot(robot_id)# 参数名称 最小值 最大值 起始值
    targetLineID = p.addUserDebugLine([-0.2+0.043,0,1.05], [0.2+0.043,0,1.05], lineColorRGB=[1, 0, 0], lineWidth=1)
    targetVal = 0.0
    resetWeldingRobot(robot_id,initial_pos)

    p.setJointMotorControl2(
        bodyUniqueId=robot_id,
        jointIndex=0,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=-0.05,
        force=100
    )
    controller = PID_Controller(4,0.3,8)  # pid
    pre_x = initial_pos[0]
    pre_y = initial_pos[1]
    sleep(3)
    while True:
        p.stepSimulation()
        # print(p.getJointState(robot_id,1)[0])
        y_pos = p.getJointState(robot_id,1)[0]
        x_pos = p.getJointState(robot_id,0)[0]
        # print("x: "+ str(x_pos) + "y: " + str(y_pos))
        p.addUserDebugLine([pre_x, -pre_y, 1.05],
                           [x_pos, -y_pos, 1.05], lineColorRGB=[0, 0, 1],
                           lineWidth=1)
        pre_x = x_pos
        pre_y = y_pos

        error = y_pos - targetVal
        out = controller.update_up(0,0,0,-error)  # out max = 0.2

        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=out,
            force=100
        )
        if abs(x_pos - (-0.157)) <= 0.01:
            break
        sleep(1 / 240)

    p.disconnect(serverID)