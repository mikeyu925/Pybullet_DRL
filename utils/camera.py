import pybullet as p
import time
import pkgutil
import pybullet_data
import numpy as np
import math
from time import sleep
import cv2
from matplotlib import pyplot as plt
MINICAR_BASE_RADIUS = 0.5
MINICAR_BASE_THICKNESS = 0.2

RACECAR_BASE_RADIUS = 0.5
RACECAR_BASE_THICKNESS = 0.2

def rgbaToRgb(rgbaImg):
    return rgbaImg[:,:,3]


'''
需要将图像向右旋转90度
'''
def liner_regression(rgbaImg):
    grayImg = cv2.cvtColor(rgbaImg,cv2.COLOR_BGR2GRAY)
    # Otsu's thresholding
    ret2, binaryImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 腐蚀膨胀，将直线减少
    # kernel = np.ones((3, 3), dtype=np.uint8)
    # dilate = cv2.dilate(binaryImg, kernel, iterations=1)
    # dilate = cv2.transpose(dilate)  # 顺时针旋转90度
    # ss = np.hstack((binaryImg, dilate))
    # cv2.imshow("res",ss)
    # cv2.waitKey(0)
    # 计算是线的坐标[x,y]
    dilate = binaryImg
    dilateX = []
    dilateY = []
    for i in range(len(dilate)):
        for j in range(len(dilate[0])):
            if dilate[i][j] == 0:
                dilateX.append(i)
                dilateY.append(j)
    # print(dilateX)
    # print(len(dilateX))
    # list转numpy
    if len(dilateX) == 0 :
        return
    X = np.array(dilateX).astype(np.float32)
    Y = np.array(dilateY).astype(np.float32)
    # 线性回归
    ro = np.polyfit(X, Y, deg=1)  # deg为拟合的多项式的次数（线性回归就选41021）
    # ry = np.polyval(ro, X)

    print(ro)

    if abs(ro[0]) <= 0.02:
        # dis = ro[1]
        theta = 90
    else:
        # dis = -ro[1] / ro[0]
        theta = np.arctan(ro[0]) * 2 / np.pi * 90 + 90

    print("与水平夹角为:" + str(theta))
    # print("与y=0相交于:" + str(dis))
    # plt.scatter(X, Y)
    # plt.plot(X, ry, c='r')
    # plt.show()


    print("------------")

def setCameraOnRaceCarAndGetImg(robot_id: int, width: int = 224, height: int = 224, physicsClientId: int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
    ty_vec = np.array([matrix[1], matrix[4], matrix[7]])  # 变换后的y轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴
    # print("___________________")
    # print(tx_vec)
    # print(ty_vec)
    # print(tz_vec)
    # print("+++++++++++++++++++")
    basePos = np.array(basePos)
    # 摄像头的位置
    # BASE_RADIUS 为 0.5，是机器人底盘的半径。BASE_THICKNESS 为 0.2 是机器人底盘的厚度。
    # 相机的位置相当于向机器人自身坐标系 [x + r, y, z  + 0.5 * h]
    cameraPos = basePos + RACECAR_BASE_RADIUS * tx_vec + 0.5 * RACECAR_BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec
    # p.addUserDebugLine(basePos,basePos + [0,0,1],lineColorRGB=[0, 0, 1],
    #                    lineWidth=1)
    # p.addUserDebugLine(cameraPos,cameraPos + [0,0,1],lineColorRGB=[1, 0, 0],
    #                    lineWidth=1)
    # p.addUserDebugLine(targetPos,targetPos + [0,0,-1],lineColorRGB=[0, 1, 0],
    #                    lineWidth=1)
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,  # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,  # 摄像头焦距下限
        farVal=20,  # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )

    return width, height, rgbImg, depthImg, segImg


def printImgInfo(imginfo):
    [w, h, rgbaImg, depthImg, segImg] = imginfo
    print("图片大小: " + str(w) + "x" +str(h))
    print("rgba: " + str(len(rgbaImg)) + "x" +  str(len(rgbaImg[0])) + "x" + str(len(rgbaImg[0][0])))
    print("depth: " + str(len(depthImg)))
    print("seg: " + str(len(segImg)))

def setCameraOnMiniCarAndGetImg(robot_id: int, width: int = 224, height: int = 224, physicsClientId: int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
    ty_vec = np.array([matrix[1], matrix[4], matrix[7]])  # 变换后的y轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴
    # print("___________________")
    # print(tx_vec)
    # print(ty_vec)
    # print(tz_vec)
    # print("+++++++++++++++++++")
    basePos = np.array(basePos)
    # 摄像头的位置
    # BASE_RADIUS 为 0.5，是机器人底盘的半径。BASE_THICKNESS 为 0.2 是机器人底盘的厚度。
    # 相机的位置相当于向机器人自身坐标系 [x + r, y, z  + 0.5 * h]
    cameraPos = basePos + MINICAR_BASE_RADIUS * tx_vec + 0.5 * MINICAR_BASE_THICKNESS * tz_vec
    targetPos = cameraPos + 1 * tx_vec
    # p.addUserDebugLine(basePos,basePos + [0,0,1],lineColorRGB=[0, 0, 1],
    #                    lineWidth=1)
    # p.addUserDebugLine(cameraPos,cameraPos + [0,0,1],lineColorRGB=[1, 0, 0],
    #                    lineWidth=1)
    # p.addUserDebugLine(targetPos,targetPos + [0,0,-1],lineColorRGB=[0, 1, 0],
    #                    lineWidth=1)
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=45.0,  # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,  # 摄像头焦距下限
        farVal=20,  # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )

    return width, height, rgbImg, depthImg, segImg

def setGunCameraAndGetImg(robot_id: int, width: int = 224, height: int = 224, physicsClientId: int = 0):
    """
    给合成摄像头设置图像并返回robot_id对应的图像
    摄像头的位置为miniBox前头的位置
    """
    basePos, baseOrientation = p.getBasePositionAndOrientation(robot_id, physicsClientId=physicsClientId)
    # 从四元数中获取变换矩阵，从中获知指向(左乘(1,0,0)，因为在原本的坐标系内，摄像机的朝向为(1,0,0))
    matrix = p.getMatrixFromQuaternion(baseOrientation, physicsClientId=physicsClientId)
    tx_vec = np.array([matrix[0], matrix[3], matrix[6]])  # 变换后的x轴
    ty_vec = np.array([matrix[1], matrix[4], matrix[7]])  # 变换后的y轴
    tz_vec = np.array([matrix[2], matrix[5], matrix[8]])  # 变换后的z轴

    relative_x = p.getJointState(robot_id,0)[0]
    relative_y = -p.getJointState(robot_id, 1)[0]
    relative_z = -p.getJointState(robot_id, 2)[0]

    basePos = np.array(basePos)
    # 摄像头的位置
    # 相机的位置相当于向机器人自身坐标系 [x + r, y, z  + 0.5 * h]
    cameraPos = basePos + [relative_x,relative_y,relative_z]
    targetPos = basePos + [relative_x,relative_y,0]
    # p.addUserDebugLine(basePos,basePos + [0,0,1],lineColorRGB=[0, 0, 1],
    #                    lineWidth=1)
    # p.addUserDebugLine(cameraPos,cameraPos + [0,0,1],lineColorRGB=[1, 0, 0],
    #                    lineWidth=1)
    # p.addUserDebugLine(targetPos,targetPos + [0,0,-1],lineColorRGB=[0, 1, 0],
    #                    lineWidth=1)

    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=cameraPos,
        cameraTargetPosition=targetPos,
        cameraUpVector=tz_vec,
        physicsClientId=physicsClientId
    )

    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=90.0,  # 摄像头的视线夹角
        aspect=1.0,
        nearVal=0.01,  # 摄像头焦距下限
        farVal=20,  # 摄像头能看上限
        physicsClientId=physicsClientId
    )

    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=width, height=height,
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        physicsClientId=physicsClientId
    )

    return width, height, rgbImg, depthImg, segImg

if __name__ == '__main__':

    serverId = p.connect(p.GUI)


    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.setRealTimeSimulation(0)

    p.loadURDF("plane.urdf", [0, 0, -1])
    startPos = [0, 0, 1]
    startOrientation = p.getQuaternionFromEuler([0 , 0, math.pi / 2]) # 从 欧拉角 转换得到 四元数
    robot_id = p.loadURDF("/home/ywh/anaconda3/pkgs/pybullet-3.21-py37he8f5f7f_1/lib/python3.7/site-packages/pybullet_data/miniBox.urdf",startPos,startOrientation)
    p.resetBasePositionAndOrientation(robot_id, startPos, startOrientation)

    pixelWidth = 320
    pixelHeight = 220
    camTargetPos = [0, 0, 0]
    camDistance = 4
    pitch = -40
    roll = 0
    upAxisIndex = 2

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    while True:
      p.stepSimulation()

      for yaw in range(0, 360, 10):
        start = time.time()
        p.stepSimulation()
        stop = time.time()
        print("stepSimulation %f" % (stop - start))

        # [row,pitch,yaw]为相机朝向
        viewMatrix = p.computeViewMatrixFromYawPitchRoll(
            camTargetPos, # 相机目标点
            camDistance, # 距离目标点的距离
            yaw,  # 朝向
            pitch,
            roll,
            upAxisIndex
        )

        projectionMatrix =  p.computeProjectionMatrixFOV(
            fov=60.0,               # 摄像头的视线夹角
            aspect=1,
            nearVal=0.01,            # 摄像头焦距下限
            farVal=20,               # 摄像头能看上限
            physicsClientId=serverId
        )

        start = time.time()
        img_arr = p.getCameraImage(pixelWidth,
                                   pixelHeight,
                                   viewMatrix=viewMatrix,
                                   projectionMatrix=projectionMatrix,
                                   shadow=1,
                                   lightDirection=[1, 1, 1])
        stop = time.time()
        print("renderImage %f" % (stop - start))
        sleep(1 / 240)

    p.disconnect(serverId)
