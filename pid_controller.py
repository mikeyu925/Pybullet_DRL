#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
@Modify Time      @Author    @Version    @Desciption
------------      -------    --------    -----------
2022/6/2 9:53   xxx      1.0         None
"""

import math
from parameter import args

class PID_Controller(object):
    # def __init__(self, kp, ki, kd, args):
    def __init__(self, kp, ki, kd):
        super(PID_Controller).__init__()
        self.kp = kp
        self.ki = ki
        self.kd = kd
        # self.dt = args.dt  # 要这个dt是为了干什么？？？

        self.CumulativeError = 0.0
        self.LastError = None
        self.integralThreshold = 10
        self.CumulativeThreshold = 200

    def update(self, _kp,_ki,_kd,error):
        """
        compute the out of  PID .
        :param error:
        :return:
        """
        p = (self.kp + _kp) * error
        i = (self.ki + _ki) * self.CumulativeError
        if self.LastError is None:
            d = 0.0
        else:
            d = (self.kd + _kd) * (error - self.LastError)

        self.CumulativeError += error  # TODO 是否进行限制
        self.LastError = error

        return p + i + d

    def update_up(self, _kp,_ki,_kd,error):
        """
        compute the out of  PID .
        :param error:
        :return:
        """
        p = (self.kp + _kp) * error
        i = (self.ki + _ki) * self.CumulativeError
        if self.LastError is None:
            d = 0.0
        else:
            d = (self.kd + _kd) * (error - self.LastError)
        # 误差在指定区间内才累计误差
        if abs(error) <= self.integralThreshold:
            self.CumulativeError += error
        # 限制积分的累计误差
        if self.CumulativeError > self.CumulativeThreshold:
            self.CumulativeError = self.CumulativeThreshold
        elif self.CumulativeError < -self.CumulativeThreshold:
            self.CumulativeError = -self.CumulativeThreshold

        self.LastError = error
        return p + i + d

class Point():
    def __init__(self,x ,y):
        self.x = x
        self.y = y

class PID_Info():
    def __init__(self,name,height):
        self.name = name
        self.height = height
        self.AEI = 0 # 绝对误差积分
        self.dt = args.dt

        self.AT = float('inf')   # 调节时间
        self.reach_stable = False  # 是否到达稳态标志位
        self.start_time = -1      # 开始判断是否达到稳态的起始时间
        self.cnt_threadhold = 80  # 当 cnt >= cnt_threadhold 时说明到达稳态
        self.cnt =  0
        self.error_threadhold = 0.02 * self.height
        self.diff = 0.005 * self.height
        self.stable_idx = -1

        self.PO = 0  # 超调量
        self.reach_top = False

    def set_start_time(self,t):
        self.start_time = t

    def check_stable(self,error,now,t,idx):
        self.AEI += math.fabs(error) * self.dt

        if error < self.PO:
            self.reach_top = True
            self.PO = error
            self.top_point = Point(t, now)

        if t >= self.start_time and math.fabs(error) <= self.error_threadhold:
            self.cnt += 1
        else:
            self.cnt = 0
        if self.cnt >= self.cnt_threadhold and self.reach_stable == False:
            self.AT = t - (int)(self.cnt_threadhold / 2) * 0.01
            self.reach_stable = True
            self.stable_idx = (int)((idx - self.cnt_threadhold + idx) / 2);
            self.stable_point = Point(t,now)

    def showPIDControlInfo(self):
        print(self.name + "相关信息如下：")
        print("     绝对误差积分:" + str(self.AEI))
        print("     调节时间:" + str(self.AT))
        print("     超调量:" + str(-self.PO / self.height * 100))