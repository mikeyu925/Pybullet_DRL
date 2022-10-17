from time import sleep
from models import *
from virtualEnv.weldingEnv import *
from utils.utils import action_limit,show_resutlt,horizontal_line
from pid_controller import PID_Info,PID_Controller
from matplotlib import  pyplot as plt

class weldingRun_base(object):
    def __init__(self):
        self.var_min = 0.01
        self.env = WeldingBaseEnv(1)
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
            self.max_episodes = 600
            self.action_dim = args.action_dim  # 动作维度  默认4
            self.action_bound = args.action_bound  # 动作范围界限  默认1

            self.state_dim = self.env.state_dim   # 状态维度
            self.agent = DDPG(self.state_dim, self.action_dim, self.action_bound)  # 创建一个Agent

    def train(self):
        pass

    def test(self):
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

class run_pid_parameter(weldingRun_base):
    def __init__(self):
        super(run_pid_parameter, self).__init__()
        self.action_dim = args.action_dim  # 动作维度  默认4
        self.action_bound = args.action_bound  # 动作范围界限  默认10

    def train(self):
        epRewards = []
        maxEpReward = 1500.0  # set the base max reward.  Default: 4000
        en = args.exploration_noise  # 5
        for i in range(self.max_episodes): # 最多训练800个episode
            state = self.env.reset()  # 获取环境初始状态
            ep_reward = 0 # 累计奖励
            pre_pos = self.env.getRobotPos() # 获得初始位置作为上一个值
            while(True):
                p.stepSimulation()
                action = self.agent.select_action(state) # 获得行为

                action = np.clip(np.random.normal(action, en), -self.action_bound, self.action_bound) # 给行为添加探索的噪音 ， clip相当于做了一个边界处理
                action = action_limit(action)
                # x 轴方向移动速度
                p.setJointMotorControl2(
                    bodyUniqueId=self.env.robot,
                    jointIndex=0,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=-0.05,
                    force=10
                )
                next_state, reward, done = self.env.step(action)   # 获得下一个状态
                self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

                state = next_state  # 更新状态
                ep_reward += reward  # 累加奖励

                if done:
                    break

                now_pos = self.env.getRobotPos()  # 获得当前位置
                # print("prepos : " + str(pre_pos) + "  nowpos : " + str(now_pos))
                p.addUserDebugLine([pre_pos[0], -pre_pos[1], 1.05],[now_pos[0], -now_pos[1], 1.05], lineColorRGB=[0, 0, 1], lineWidth=1)
                pre_pos = now_pos

                sleep(1 / 240)

            p.removeAllUserDebugItems()

            en = max([ en * 0.999995, 0.01])
            self.agent.update()  # 更新网络
            # 打印: 第i段经历, 奖励 , 探索噪音, time
            print("Episode: {}, Reward {}, Explore {} lastPos:{}".format(i, ep_reward, en, self.env.getRobotPos()[0]))
            epRewards.append(ep_reward)

            if ep_reward >= maxEpReward:  # 保存最大的 reward 的模型
                print("Get More Episode Reward {}".format(ep_reward))
                maxEpReward = ep_reward  # 更新最大reward值
                self.agent.save() # 保存模型

        self.env.disconnect()

    def test(self):
        self.agent.load()
        state = self.env.reset()  # 获取环境初始状态
        ep_reward = 0  # 累计奖励
        # pidinfo = PID_Info("DRL PID", 100)
        # pidinfo.set_start_time(0.0)
        pre_pos = self.env.getRobotPos()  # 获得初始位置作为上一个值
        sleep(3)
        while (True):
            p.stepSimulation()
            action = self.agent.select_action(state)  # 获得行为
            action = np.clip(np.random.normal(action, self.var_min), -self.action_bound,self.action_bound)  # 给行为添加探索的噪音 ， clip相当于做了一个边界处理
            action = action_limit(action)
            print(action)
            # x 轴方向移动速度
            p.setJointMotorControl2(
                bodyUniqueId=self.env.robot,
                jointIndex=0,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=-0.05,
                force=10
            )
            next_state, reward, done = self.env.step(action)  # 获得下一个状态
            self.agent.replay_buffer.push((state, next_state, action, reward, np.float64(done)))  # 放入经验回放池

            state = next_state  # 更新状态
            ep_reward += reward  # 累加奖励

            now_pos = self.env.getRobotPos()  # 获得当前位置
            p.addUserDebugLine([pre_pos[0], -pre_pos[1], 1.05], [now_pos[0], -now_pos[1], 1.05], lineColorRGB=[0, 0, 1],
                               lineWidth=1)
            pre_pos = now_pos

            if done:
                break

            sleep(1 / 240)
        sleep(5)
        self.env.disconnect()
        print("final score : " + str(ep_reward))

def main():
    runner = run_pid_parameter()
    if args.run_type == "train":   # --run_type train --kp 4 --ki 0.3 --kd 8 --exploration_noise 3 --batch_size 128
        if args.choose_model == "dynamic_pid":
            runner.train()
    elif args.run_type == "test": # --run_type test --kp 4 --ki 0.3 --kd 8
        if args.choose_model == "dynamic_pid":
            runner.test()

if __name__ == '__main__':
    main()