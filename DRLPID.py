from models import *
import numpy as np
from virtualEnv.SimulateEnv.line_env import LineEnv
from virtualEnv.SimulateEnv.sin_env import  SinEnv
from virtualEnv.SimulateEnv.square_env import SquareEnv
from utils.utils import action_limit,show_resutlt,horizontal_line
from pid_controller import PID_Info,PID_Controller
from matplotlib import  pyplot as plt
import pybullet as p
import pybullet_data
from time import sleep
import virtualEnv
from pid_controller import PID_Controller
from virtualEnv import loadCube, resetWeldingRobot, initial_pos


def ClassicPID(kp,ki,kd):
    pidinfo = PID_Info("Classic PID",100)
    x = list(np.arange(args.start_time, args.end_time, args.dt))
    y = horizontal_line(x,100)
    pidinfo.set_start_time(x[0])
    p1 = []
    now1 = 0
    pid1 = PID_Controller(kp,ki,kd)
    for i,sim_time in enumerate(x):
        error1 = y[i] - now1
        # print(error1)
        pidinfo.check_stable(error1,now1,sim_time,i) # 更新相关信息
        now1 += pid1.update_up(-2, 0.5, 0, error1) * args.dt
        # now1 += pid1.update(0, 0, 0, error1) * args.dt
        p1.append(now1)
    pidinfo.showPIDControlInfo()

    # plt.plot(x, y, 'r-', linewidth=0.5)
    plt.plot(x, p1, 'b-', linewidth=0.5)

    # plt.scatter(x[pidinfo.stable_idx], p1[pidinfo.stable_idx], color=(0.7, 0., 0.6))
    # plt.scatter(pidinfo.top_point.x, pidinfo.top_point.y,color=(0.,0.5,0.))

    plt.xlabel('time')
    plt.ylabel('position')
    # plt.legend(["target_pos", "real_pos"], loc='lower right')


def Sin_Trick():
    pass

class run_base(object):
    def __init__(self):
        self.var_min = 0.01
        self.lineenv = LineEnv()  # 环境
        self.sinenv = SinEnv()
        self.squareenv = SquareEnv()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 训练模式
        if args.run_type == "train":
            result_dir = os.path.join('results',
                                      'ChooseModel_{}_RunType_{}'.format(
                                          args.choose_model,args.run_type))
            self.result_dir = os.path.join(current_dir, result_dir)
        else:
            # Need to load the train model.  需要加载训练的模型
            result_dir = os.path.join('results',
                                      'ChooseModel_{}_RunType_{}'.format(
                                          args.choose_model,"train"))
            self.result_dir = os.path.join(current_dir, result_dir)

        os.makedirs(self.result_dir, exist_ok=True)

        if args.choose_model == "dynamic_pid":
            self.max_episodes = 800
            self.action_dim = args.action_dim  # 动作维度  默认4
            self.action_bound = args.action_bound  # 动作范围界限  默认10

            self.state_dim = self.lineenv.state_dim   # 状态维度
            self.agent = DDPG(self.state_dim, self.action_dim, self.action_bound)  # 创建一个Agent

    def train(self,times):
        pass

    def test(self,times):
        pass

    def save_result_txt(self, xs, ys, yaxis='radValues'):
        """
        # Save the data to .txt file.
        :param xs:
        :param ys:
        :param yaxis:
        """
        filename = os.path.join(self.result_dir, yaxis + '.txt')
        if isinstance(xs, np.ndarray) and isinstance(ys, list):
            if not os.path.exists(filename):
                with open(file=filename, mode="a+") as file:
                    file.write("times {}".format(yaxis))
                    file.write("\r\n")
            else:
                print("{} has already existed. added will be doing.".format(filename))
                with open(file=filename, mode="a+") as file:
                    file.write("times, {}".format(yaxis))
                    file.write("\r\n")
        else:
            pass

        with open(file=filename, mode="a+") as file:
            for index, data in enumerate(zip(xs, ys)):
                file.write("{} {}".format(str(data[0]), str(data[1])))
                file.write("\r\n")

    def show_rewards(self,steps,rewards):
        plt.plot(steps, rewards, 'b-', linewidth=1)
        plt.xlabel('step')
        plt.ylabel('reward')
        plt.show()

class run_pid_parameter(run_base):
    def __init__(self):
        super(run_pid_parameter, self).__init__()
        self.action_dim = args.action_dim  # 动作维度  默认4
        self.action_bound = args.action_bound  # 动作范围界限  默认10

    def train(self,times,mode):
        if mode == "response":
            steps = []
            epRewards = []
            maxEpReward = 1500.0  # set the base max reward.  Default: 4000
            en = args.exploration_noise  # 5
            for i in range(self.max_episodes):
                state = self.lineenv.reset()  # 获取环境初始状态
                ep_reward = 0 # 累计奖励
                # 绘图需要的一些数据
                x,target,real = [],[],[]
                for j, sim_time in enumerate(times):
                    action = self.agent.select_action(state) # 获得行为
                    # 给行为添加探索的噪音 exploration_noise: 10， clip相当于做了一个边界处理
                    action = np.clip(np.random.normal(action, en), -self.action_bound, self.action_bound)
                    action = action_limit(action)

                    next_state, reward, done = self.lineenv.step(action)   # 获得下一个状态
                    self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                    state = next_state  # 更新状态
                    ep_reward += reward  # 累加奖励

                    x.append(sim_time)
                    # target.append(utils.sin_curve(sim_time))
                    target.append(self.lineenv.horizontal_line[j])
                    real.append(state[3])

                    if done:
                        break
                    if sim_time == times[-1]: # 到达了times的最后
                        break

                en = max([ en * 0.999995, 0.01]) # TODO 感觉有点问题
                self.agent.update()  # 更新网络
                # 打印: 第i段经历, 奖励 , 探索噪音, time
                print("Episode: {}, Reward {}, Explore {} Steps {}".format(i, ep_reward, args.exploration_noise, j))
                epRewards.append(ep_reward)
                steps.append(i)

                # utils.show_resutlt(x, target, real)

                if ep_reward >= maxEpReward:  # 保存最大的 reward 的模型
                    print("Get More Episode Reward {}".format(maxEpReward))
                    maxEpReward = ep_reward  # 更新最大reward值
                    self.agent.save() # 保存模型
                    show_resutlt(x, target, real)

            self.save_result_txt(xs=steps,ys=epRewards,yaxis="epRewards")
            self.show_rewards(steps,epRewards)
        elif mode == "sin":
            steps = []
            epRewards = []
            maxEpReward = 1000.0  # set the base max reward.  Default: 4000
            en = args.exploration_noise  # 5
            for i in range(self.max_episodes):
                state = self.sinenv.reset()  # 获取环境初始状态
                ep_reward = 0  # 累计奖励
                # 绘图需要的一些数据
                x, target, real = [], [], []
                for j, sim_time in enumerate(times):
                    action = self.agent.select_action(state)  # 获得行为
                    # 给行为添加探索的噪音 exploration_noise: 10， clip相当于做了一个边界处理
                    action = np.clip(np.random.normal(action, en), -self.action_bound, self.action_bound)
                    action = action_limit(action)

                    next_state, reward, done = self.sinenv.step(action)  # 获得下一个状态
                    self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                    state = next_state  # 更新状态
                    ep_reward += reward  # 累加奖励

                    x.append(sim_time)
                    target.append(self.sinenv.sin_line[j])
                    real.append(state[3])

                    if done:
                        break
                    if sim_time == times[-1]:  # 到达了times的最后
                        break

                en = max([en * 0.999995, 0.01])  # TODO 感觉有点问题
                self.agent.update()  # 更新网络
                # 打印: 第i段经历, 奖励 , 探索噪音, time
                print("Episode: {}, Reward {}, Explore {} Steps {}".format(i, ep_reward, args.exploration_noise, j))
                epRewards.append(ep_reward)
                steps.append(i)

                # utils.show_resutlt(x, target, real)

                if ep_reward >= maxEpReward:  # 保存最大的 reward 的模型
                    print("Get More Episode Reward {}".format(maxEpReward))
                    maxEpReward = ep_reward  # 更新最大reward值
                    self.agent.save()  # 保存模型
                    # utils.show_resutlt(x, target, real)
                    plt.plot(x, target, 'r-', linewidth=0.5)
                    plt.plot(x, real, 'b-', linewidth=0.5)
                    plt.xlabel('time')
                    plt.ylabel('position')
                    plt.legend(["Sin", "DRL_PID"], loc='lower right')
                    plt.show()

            self.save_result_txt(xs=steps, ys=epRewards, yaxis="epRewards")
            self.show_rewards(steps, epRewards)

    def test(self,times,mode):
        x, target, real = [], [], []
        p, i, d = [], [], []
        if mode == "response":
            self.agent.load()
            state = self.lineenv.reset()  # 获取环境初始状态
            ep_reward = 0  # 累计奖励

            pidinfo = PID_Info("DRL PID", 100)
            pidinfo.set_start_time(0.0)
            # 绘图需要的一些数据
            for j, sim_time in enumerate(times):
                action = self.agent.select_action(state)  # 获得行为
                # 给行为添加探索的噪音 exploration_noise: 10， clip相当于做了一个边界处理
                action = np.clip(np.random.normal(action, self.var_min), -self.action_bound, self.action_bound)
                action = action_limit(action)
                p.append(action[0])
                i.append(action[1])
                d.append(action[2])
                self.lineenv.control_info.check_stable(state[2],state[3],sim_time,j)
                next_state, reward, done = self.lineenv.step(action)  # 获得下一个状态
                self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                state = next_state  # 更新状态
                ep_reward += reward  # 累加奖励

                pidinfo.check_stable(self.lineenv.horizontal_line[j] - state[3], state[3], sim_time, j)  # 更新相关信息
                x.append(sim_time)
                target.append(self.lineenv.horizontal_line[j])  # 目标值 target
                real.append(state[3])  # 当前 position

                if done:
                    break
                if sim_time == times[-1]:  # 到达了times的最后
                    break

            pidinfo.showPIDControlInfo()
            plt.figure(1)
            plt.subplot(1,2,1)

            plt.plot(x, real, 'g-', linewidth=0.5)
            ClassicPID(args.kp,args.ki,args.kd)
            plt.plot(x, target, 'r-', linewidth=0.5)

            plt.xlabel('time')
            plt.ylabel('position')
            plt.legend(["DRL_PID","PID"], loc='lower right')

            plt.subplot(1, 2, 2)
            plt.plot(x, p, 'g-', linewidth=0.5)
            plt.plot(x, i, 'r-', linewidth=0.5)
            plt.plot(x, d, 'b-', linewidth=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend(["p","i","d"], loc='lower right')
            plt.show()

        elif mode == "sin":
            self.agent.load()
            state = self.sinenv.reset()  # 获取环境初始状态
            ep_reward = 0  # 累计奖励

            for j, sim_time in enumerate(times):
                action = self.agent.select_action(state)  # 获得行为
                # 给行为添加探索的噪音 exploration_noise: 10， clip相当于做了一个边界处理
                action = np.clip(np.random.normal(action, self.var_min), -self.action_bound, self.action_bound)
                action = action_limit(action)
                p.append(action[0])
                i.append(action[1])
                d.append(action[2])
                self.lineenv.control_info.check_stable(state[2], state[3], sim_time, j)
                next_state, reward, done = self.sinenv.step(action)  # 获得下一个状态
                self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                state = next_state  # 更新状态
                ep_reward += reward  # 累加奖励

                x.append(sim_time)
                target.append(self.sinenv.sin_line[j])  # 目标值 target
                real.append(state[3])  # 当前 position

                if done:
                    break
                if sim_time == times[-1]:  # 到达了times的最后
                    break

            self.lineenv.control_info.showPIDControlInfo()
            plt.figure(1)
            plt.subplot(1,2,1)
            plt.plot(x, real, 'g-', linewidth=0.5)
            plt.plot(x, target, 'r-', linewidth=0.5)

            plt.xlabel('time')
            plt.ylabel('position')
            plt.legend(["DRL_PID"], loc='lower right')
            # plt.legend(["DRL_PID","PID"], loc='lower right')
            plt.subplot(1, 2, 2)
            plt.plot(x, p, 'g-', linewidth=0.5)
            plt.plot(x, i, 'r-', linewidth=0.5)
            plt.plot(x, d, 'b-', linewidth=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend(["p","i","d"], loc='lower right')
            plt.show()
            plt.show()

        elif mode == "square":
            self.agent.load()
            state = self.squareenv.reset()  # 获取环境初始状态
            ep_reward = 0  # 累计奖励

            for j, sim_time in enumerate(times):
                action = self.agent.select_action(state)  # 获得行为
                # 给行为添加探索的噪音 exploration_noise: 10， clip相当于做了一个边界处理
                action = np.clip(np.random.normal(action, self.var_min), -self.action_bound, self.action_bound)
                action = action_limit(action)
                p.append(action[0])
                i.append(action[1])
                d.append(action[2])
                self.lineenv.control_info.check_stable(state[2], state[3], sim_time, j)
                next_state, reward, done = self.squareenv.step(action)  # 获得下一个状态
                self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                state = next_state  # 更新状态
                ep_reward += reward  # 累加奖励

                x.append(sim_time)
                target.append(self.squareenv.square_line[j])  # 目标值 target
                real.append(state[3])  # 当前 position

                if done:
                    break
                if sim_time == times[-1]:  # 到达了times的最后
                    break

            self.lineenv.control_info.showPIDControlInfo()

            plt.figure(1)
            plt.subplot(1,2,1)
            plt.plot(x, real, 'g-', linewidth=0.5)
            plt.plot(x, target, 'r-', linewidth=0.5)

            plt.xlabel('time')
            plt.ylabel('position')
            plt.legend(["DRL_PID"], loc='lower right')
            # plt.legend(["DRL_PID","PID"], loc='lower right')
            # plt.show()
            plt.subplot(1, 2, 2)
            plt.plot(x, p, 'g-', linewidth=0.5)
            plt.plot(x, i, 'r-', linewidth=0.5)
            plt.plot(x, d, 'b-', linewidth=0.5)
            plt.xlabel('time')
            plt.ylabel('value')
            plt.legend(["p","i","d"], loc='lower right')
            plt.show()

def main():
    # Get Time [0,args.dt,2 * args.dt,..., ]
    times = np.arange(args.start_time, args.end_time, args.dt)[:args.max_ep_steps]
    runner = run_pid_parameter()
    if args.run_type == "train":
        if args.choose_model == "dynamic_pid":
            runner.train(times,args.trick_type)
    elif args.run_type == "test": # --run_type test --kp 4 --ki 1 --kd 4
        if args.choose_model == "dynamic_pid":
            runner.test(times,args.trick_type)
        elif args.choose_model == "welding":
            runner.weldingTest()

if __name__ == '__main__':
    main()