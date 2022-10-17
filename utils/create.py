import pybullet as p
import pybullet_data
from time import sleep
import math

'''
一些注意事项：
在仿真环境中一个各自的长度是1
halfExtents=halfExtent 中指定的是一半的长度

'''

lineDefaultWidth = 0.04
lineDefaultHeight = 0.02


def createWall(halfExtent,pos):
    # 创建一面墙
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=halfExtent
    )
    collison_box_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=halfExtent
    )
    wall_id = p.createMultiBody(
        baseMass=1000,
        baseCollisionShapeIndex=collison_box_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=pos
    )
    return wall_id


def createTrickLine(start_pos,length,width=lineDefaultWidth,height=lineDefaultHeight):
    # 创建条黑线
    halfExtent = [length/2,width/2,height/2]
    centerpos = [start_pos[0] + length / 2,start_pos[1],start_pos[2] + height / 2]
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=halfExtent,
        rgbaColor=[0,0,0,100]
    )
    collison_box_id = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=halfExtent
    )
    wall_id = p.createMultiBody(
        baseMass=1000,
        baseCollisionShapeIndex=collison_box_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=centerpos
    )
    return wall_id


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

    block = createWall([1,1,1],[0,0,1])

    p.addUserDebugLine([0,0,1],[2,0,1],lineColorRGB=[1, 0, 0],lineWidth=3)

    # 开启渲染
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.setRealTimeSimulation(0)




    while True:
        p.stepSimulation()

        sleep(1 / 240)

    # p.disconnect(serve_id)