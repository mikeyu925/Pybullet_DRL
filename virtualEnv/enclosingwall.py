import utils
import pybullet as p
import pybullet_data
from time import sleep

def loadEnclosingWall():
    # 创建四面墙
    wall_right_id = utils.createWall([9, 1, 1], [0, 10, 1])
    wall_left_id = utils.createWall([9, 1, 1], [0, -10, 1])
    wall_down_id = utils.createWall([1, 10, 1], [10, 0, 1])
    wall_top_id = utils.createWall([1, 10, 1], [-10, 0, 1])

    return [wall_right_id,wall_left_id,wall_down_id,wall_top_id]

'''
xyz:正方体xyz三个方向的长度
center_pos:中心点坐标
'''
def loadCube(xyz,center_pos):
    for (i,v) in enumerate(xyz):
        xyz[i] = v / 2
    return utils.createWall(xyz, center_pos)

if __name__ == '__main__':
    p.connect(p.GUI)  # 对物理引擎进行链接
    p.setGravity(0,0,0) # 设置重力加速度

    # 配置渲染机制
    p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)     # 关闭渲染
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 取消渲染时候周围的控制面板
    # 加载场景及模型
    p.setAdditionalSearchPath(pybullet_data.getDataPath()) # 使用相对路径读入库中现成的模型
    planeId = p.loadURDF("plane.urdf") # 加载urdf模型

    loadCube([1,1,0.5],[0,0,0.5])
    p.addUserDebugLine([-0.5, 0, 1], [0.5, 0, 1], lineColorRGB=[1, 0, 0], lineWidth=1)
    # 开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(0)

    while True:
        p.stepSimulation()

        sleep(1 / 240)
    p.disconnect(serve_id)