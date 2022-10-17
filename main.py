import math
import pybullet as p
import time
import pybullet_data
import virtualEnv
from utils import renderConfig

def createTrick():
    pass

if __name__ == '__main__':
    p.connect(p.GUI)  # 对物理引擎进行链接
    p.setGravity(0,0,-10) # 设置中坜加速度
    # 配置渲染机制
    renderConfig.resetBaseRender()

    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 使用相对路径读入库中现成的模型
    planeId = p.loadURDF("plane.urdf") # 加载urdf模型
    robot_id = virtualEnv.loadWeldingRobotURDF()
    renderConfig.setRenderOpen()  # 开启渲染


    for i in range(True):
        p.stepSimulation()
        time.sleep(1. / 240.)


    # 断开连接
    p.disconnect()