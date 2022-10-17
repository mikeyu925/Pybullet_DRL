#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/18 18:35   xxx      1.0         None
"""
from parameter import args
from utils.utils import trapezoidal_function,step_function,square_line
import numpy as np
import gym
import matplotlib.pyplot as plt
from pid_controller import PID_Controller,PID_Info
import math


class SquareEnv(gym.Env):
    def __init__(self):
        super(SquareEnv, self).__init__()
        np.random.seed(args.seed)
        # 环境的一些状态信息
        self.last_last_error = 0.0
        self.last_error = 0.0
        self.now_error = 0.0
        self.now_pos = 0.0
        self.last_speed = 0.0
        self.now_speed = 0.0
        # 计数器
        self.cnt = 0
        # 时间步长
        self.times = np.arange(args.start_time, args.end_time, args.dt)

        # 环境状态： [上上次误差、上次误差、当前误差、当前位置、上次速度、当前速度]
        self.state = np.array([self.last_last_error,self.last_error,self.now_error,self.now_pos,self.last_speed,self.now_speed])
        # 环境状态维度
        self.state_dim = len(self.state)

        # 奖励函数的一些常量信息
        self.k = 1
        self.c = -300

        #TODO 待修改  一些阈值常量
        self.error_limit = 130
        self.quick_error_limit = 50

        self.speed_limie = 100
        # PID控制器
        self.pid = PID_Controller(kp=args.kp, ki=args.ki, kd=args.kd)
        self.control_info = PID_Info("Trick Line",100)  # 一些控制信息
        self.control_info.set_start_time(0.0)

        self.height = 100
        self.line_y = step_function(list(self.times))
        self.square_line = square_line(self.times,self.height)

    def reset(self):
        # 环境的一些状态信息 TODO 是全部初始化为0还是？
        # self.now_pos = random() * 10  # 当前随机一个位置
        # self.now_error = self.now_pos - sin_curve(0)  # 当前误差
        # self.last_error = min(self.now_error * 1.3,self.error_limit)  # 上次误差
        # self.last_last_error = min(self.now_error * 1.6,self.error_limit) # 上上次误差

        self.now_pos = 0  # 当前随机一个位置
        self.now_error = 0  # 当前误差
        self.last_error = 0  # 上次误差
        self.last_last_error = 0 # 上上次误差

        self.last_speed = 0.0
        self.now_speed  = 0.0
        self.state = np.array([self.last_last_error, self.last_error, self.now_error, self.now_pos, self.last_speed, self.now_speed])
        # 计数器
        self.cnt = 0

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
        # 计算误差
        # error = sin_curve(self.times[self.cnt]) - now_pos

        error = self.square_line[self.cnt] - now_pos
        # 如果已经出于稳态，则不再进行补偿
        if error <= self.height * 0.08:
            _out = 0

        out_speed = self.pid.update_up(_kp, _ki, _kd, error) + _out
        # 更新状态
        self.last_last_error = self.last_error
        self.last_error = self.now_error
        self.now_error = error
        self.now_pos += out_speed * args.dt
        self.last_speed = now_speed
        self.now_speed = out_speed

        # 获得即时奖励
        reward = self._get_immediate_reward(error)
        # 是否到达终止状态
        done,reward_ = self._isDone()

        self.cnt += 1
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
            return math.fabs(self.last_error) - math.fabs(error)
        else:
            return math.exp(self.k - math.fabs(error))

    def _isDone(self):
        """
        判断是否到达了终止状态
        :return:
        """
        if math.fabs(self.now_error) >= self.error_limit:  # 提前结束
            return True , 0
        if self.cnt == len(self.times) - 1:  # 到达终止状态
            r = math.exp(2-self.control_info.AEI)
            if self.control_info.reach_stable == True:
                r += 10 / self.control_info.AT
            if self.control_info.reach_top == True:
                r += math.exp(self.control_info.PO / self.control_info.height * 100)
            return True , r
        return False , 0