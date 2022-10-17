from parameter import args
import numpy as np
import gym
from pid_controller import PID_Info,PID_Controller
import math
import pybullet as p
import pybullet_data
import virtualEnv
from virtualEnv.enclosingwall import loadCube


initial_pos = [0.20,0.03,0.00]


def loadWeldingRobotURDF(startPos):
    # [ a,b,c] 相当于分别绕着x,y,z轴转 a,b,c 度
    startOrientation = p.getQuaternionFromEuler([0, -math.pi / 2, 0])  # 从 欧拉角 转换得到 四元数
    robot_id = p.loadURDF("/home/ywh/anaconda3/pkgs/pybullet-3.21-py37he8f5f7f_1/lib/python3.7/site-packages/pybullet_data/装配体10.SLDASM/urdf/装配体10.SLDASM.urdf",startPos,startOrientation)
    return robot_id

def resetWeldingRobot(robot_id,pos):
    x,y,z = pos
    p.resetJointState(robot_id,0,x)
    p.resetJointState(robot_id,1,y)
    p.resetJointState(robot_id,2,z)

def constantFunc():
    return 0.0

class WeldingBaseEnv(gym.Env):

    def __init__(self,tagetFuncType):
        super(WeldingBaseEnv, self).__init__()
        np.random.seed(args.seed)
        # 加载pybullet 虚拟仿真环境
        self.serverID = p.connect(p.GUI)  # 对物理引擎进行链接
        # self.serverID = p.connect(p.DIRECT)  # 关闭物理引擎渲染，加快训练
        p.setGravity(0, 0, -10)  # 设置重力加速度

        # 配置渲染机制
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # 关闭渲染
        # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) # 取消渲染时候周围的控制面板
        # 加载场景及模型
        p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 使用相对路径读入库中现成的模型
        p.loadURDF("plane.urdf")  # 加载urdf模型
        loadCube([1, 1, 1], [0, 0, 0.5])
        # 加载机器人，并设置加载的机器人的位姿
        self.robot = virtualEnv.loadWeldingRobotURDF([0, 0, 1])
        # 开启渲染
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setRealTimeSimulation(0)
        # 重置robot位置
        resetWeldingRobot(self.robot, initial_pos)
        p.addUserDebugLine( [-0.2 + 0.043, 0, 1.05],
                          [0.2 + 0.043, 0, 1.05],
                          lineColorRGB=[1, 0, 0],
                          lineWidth=1)  # 绘制目标轨迹
        if tagetFuncType == 1 :
            self.targetFunc = constantFunc()

        # 环境的一些状态信息
        self.last_last_error = 0.0
        self.last_error = 0.0
        self.now_error = 0.0
        self.now_pos = 0.0
        self.last_speed = 0.0
        self.now_speed = 0.0
        # 环境状态： [上上次误差、上次误差、当前误差、当前位置、上次速度、当前速度]
        self.state = np.array([self.last_last_error,self.last_error,self.now_error,self.now_pos,self.last_speed,self.now_speed])
        # 环境状态维度
        self.state_dim = len(self.state)
        # 奖励函数的一些常量信息
        self.k = 0.1
        self.c = -100

        # PID控制器
        self.pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd)
        self.control_info = PID_Info("Welding Trick",100)  # 一些控制信息
        self.control_info.set_start_time(0.0)
        self.error_limit = 0.065 # 当误差超过 error_limit 时，给一个负的奖励
        self.quick_error_limit = 0.02 # 当误差在 [0.02,0.065]区间内时，误差缩减速度越快，奖励越高
        self.pidout_limit = 0.7  # PID输出的限制
        self._out_limit = 0.1 # _out 的输出限制
        self.stable_threadhold = 0.01  # 当误差小于 stable_threadhold时候，不再进行_out补偿

    def reset(self):
        # 环境的一些状态信息
        p.addUserDebugLine( [-0.2 + 0.043, 0, 1.05],
                          [0.2 + 0.043, 0, 1.05],
                          lineColorRGB=[1, 0, 0],
                          lineWidth=1)  # 绘制目标轨迹

        self.now_pos = 0  # 当前随机一个位置
        self.now_error = 0  # 当前误差
        self.last_error = 0  # 上次误差
        self.last_last_error = 0 # 上上次误差

        self.last_speed = 0.0
        self.now_speed  = 0.0
        self.state = np.array([self.last_last_error, self.last_error, self.now_error, self.now_pos, self.last_speed, self.now_speed])

        resetWeldingRobot(self.robot, initial_pos)

        # PID控制器  还需要重置，因为要初始化pid的本身的误差信息
        self.pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd)
        self.control_info = PID_Info("Trick Line", 100)  # 一些控制信息
        self.control_info.set_start_time(0.0)

        return self.state

    def step(self,action):
        """
        :param action: [_kp, _ki, _kd, _out]
        :return: state : [self.state, reward, done]
        """
        # 获取当前环境的状态
        last_last_error, last_error, now_error, now_pos, last_speed, now_speed = self.state
        _kp, _ki, _kd, _out = action

        if _out >= self._out_limit:
             _out = self._out_limit
        elif _out <= -self._out_limit:
            _out = -self._out_limit

        # 计算误差 = 目标值 - 当前值
        error = (0 - now_pos)
        # 如果已经出于稳态，则不再进行补偿
        if abs(error) <= self.stable_threadhold:
            _out = 0
        # PID输出量直接设为速度
        out_speed = self.pid.update_up(_kp, _ki, _kd, error) + _out
        # print(" _kp : " + str(_kp) +" _ki : " + str(_ki) + " _kd : " + str(_kd) + " pid_out : " + str(out_speed))
        # PID输出量限幅
        if out_speed >= self.pidout_limit :
             out_speed = self.pidout_limit
        elif out_speed <= -self.pidout_limit :
             out_speed = -self.pidout_limit

        p.setJointMotorControl2(
            bodyUniqueId=self.robot,
            jointIndex=1,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=out_speed,
            force=10
        )

        # 更新状态
        self.last_last_error = self.last_error
        self.last_error = self.now_error
        self.now_error = error
        self.now_pos = p.getJointState(self.robot,1)[0]
        self.last_speed = now_speed
        self.now_speed = out_speed

        # 获得即时奖励
        reward = self._get_immediate_reward(error)
        # 是否到达终止状态 如果顺利到达终止态 额外给予奖励
        done,reward_ = self._isDone()

        self.state = np.array([self.last_last_error,self.last_error,self.now_error,self.now_pos,self.last_speed,self.now_speed])

        return self.state, reward + reward_, done

    def _get_immediate_reward(self,error):
        """
        获取当前的即时奖励
        :param error:
        :return: int (reward)
        """
        if math.fabs(error) >= self.error_limit:
            return self.c
        elif math.fabs(error) >= self.quick_error_limit:
            last_abs = math.fabs(self.last_error)
            now_abs = math.fabs(error)
            return math.exp(last_abs - now_abs) - math.exp(now_abs)
            # return math.fabs(self.last_error) - math.fabs(error)
        else:
            return math.exp(self.k - math.fabs(error))

    def _isDone(self):
        """
        判断是否到达了终止状态
        :return:
        """
        if math.fabs(self.now_error) >= self.error_limit:  # 误差过大提前结束
            return True , -100
        if abs(p.getJointState(self.robot,0)[0] - (-0.157)) <= 0.01:  # 到达终止状态
            r = math.exp(2-self.control_info.AEI)
            if self.control_info.reach_stable == True:
                r += 10 / self.control_info.AT
            if self.control_info.reach_top == True:
                r += math.exp(self.control_info.PO / self.control_info.height * 100)
            return True , r
        return False , 0


    def disconnect(self):
        p.disconnect(self.serverID)

    def getRobotPos(self):
        pos = []
        pos.append(p.getJointState(self.robot, 0)[0])
        pos.append(p.getJointState(self.robot, 1)[0])
        pos.append(p.getJointState(self.robot, 2)[0])
        return pos

