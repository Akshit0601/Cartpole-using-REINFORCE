from __future__ import annotations

import random
from reset import reset_sim
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import gymnasium as gym
import rclpy
from threading import Thread
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rclpy.callback_groups import ReentrantCallbackGroup,MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from std_srvs.srv import Empty
from agent import *

from time import sleep
# seed = 1

# torch.manual_seed(seed)
# random.seed(seed)
# np.random.seed(seed)

class train(Node):

    def __init__(self):
        super().__init__('trainer')
        self.call_grp = ReentrantCallbackGroup()
        # self.create_subscription(JointState,'/joint_states',self.obs_state,10,callback_group=self.call_grp)
        # self.step = self.create_publisher(Float64MultiArray,'/effort_controller/commands',10)

        # self.unpause = self.create_client(Empty,'/unpause_physics')

        self.obs_state = np.zeros(4)

        self.agent = REINFORCE(4, 1)
        # shared_weights = torch.load('/home/akshit/rl/src/weights/shared_weights_5.pth')
        # mean_weights = torch.load('/home/akshit/rl/src/weights/mean_5.pth')
        # std_weights = torch.load('/home/akshit/rl/src/weights/std_5.pth')
        # self.agent.net.shared_net.load_state_dict(shared_weights)
        # self.agent.net.policy_mean_net.load_state_dict(mean_weights)
        # self.agent.net.policy_stddev_net.load_state_dict(std_weights)
        

        self.msg = Float64MultiArray()
        self.reset_msg = Float64MultiArray()
        self.reset_msg.data = [0.0]

        self.req = Empty.Request()

        self.main_loop = self.create_service(Empty,"/init",self.train)
        self.exec = self.create_service(Empty,'/start',self.test_mujoco)
        self.create_service(Empty,"/start_gazebo",self.test)


        # self.rate = self.create_rate(500)




    def obs_state(self,data:JointState):
        self.obs_state[0] = data.position[0]
        self.obs_state[1] = self.normalise(data.position[1])
        # print(data.position[1])
        #between [0,6.28]
        self.obs_state[2] = data.velocity[0]
        self.obs_state[3] = data.velocity[1]
        # print(self.obs_state)
    
    def normalise(self,angle):
        if angle>=0:
            angle = angle%6.28
        else:
            angle = 6.28 - abs(angle)%6.28
        return angle
    
    def test(self, req:Empty.Request,  response: Empty.Response):
        for i in range(150):
            thread = Thread(target=reset_sim)
            
            thread.start()
            thread.join()
            sleep(0.5)
            self.unpause.call_async(Empty.Request())

            while not (self.obs_state[1] > 0.40 and self.obs_state[1] < 5.7):
                action = self.agent.sample_action(self.obs_state)

                
                self.msg.data = [action.item()]
                self.step.publish(self.msg)

                self.rate.sleep()
            thread = None
        return response

    def test_mujoco(self, req:Empty.Request,  response: Empty.Response):
        env = gym.make("InvertedPendulum-v4",render_mode = 'human')
        
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)
        shared_weights = torch.load('weights/shared_weights_6_mujoco.pth')
        mean_weights = torch.load('weights/mean_mujoco.pth')
        std_weights = torch.load('weights/std6_mujoco.pth')
        self.agent.net.shared_net.load_state_dict(shared_weights)
        self.agent.net.policy_mean_net.load_state_dict(mean_weights)
        self.agent.net.policy_stddev_net.load_state_dict(std_weights) 

        
        done = False
         # Records episode-reward
        for i in range(10):
            obs,info = wrapped_env.reset(seed=1)
            done = False

            while not done:
                action = self.agent.sample_action(obs)
                obs, reward, terminated, truncated, info = wrapped_env.step(action)
                # self.get_logger().info(str(obs,reward))
                # print(obs,reward,episode)
                # self.agent.rewards.append(reward)

                done = terminated
                if done:
                    print(info)  
                sleep(0.01)
            np.save('std3.npy',np.array(self.agent.std_devs)) 
        wrapped_env.close()

        return response   

        





    def train(self, req: Empty.Request, response:Empty.Response):
        env = gym.make("InvertedPendulum-v4",render_mode = 'human')
        wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward

        total_num_episodes = int(5e3)  # Total number of episodes
        rewards_over_seeds = []
        std_over_seeds = []
        # for seed in [1,2,3,5,8]:  # Fibonacci seeds
        #     # set seed
        #     torch.manual_seed(seed)
        #     random.seed(seed)
        #     np.random.seed(seed)
        #     reward_over_episodes = []

        #     # Reinitialize agent every seed
        #     for episode in range(total_num_episodes):
        #         # self.obs_state, info = wrapped_env.reset(seed=seed)
        #         thread = Thread(target=reset_sim)
                

        #         done = False 
        #         terminated = False
        #         truncated = False
        #         self.get_logger().info(str(episode))
        #         time_step = 0
        #         thread.start()
        #         thread.join()
        #         sleep(0.5)
        #         self.unpause.call_async(Empty.Request())
        #         while not done and rclpy.ok():

        #             action = self.agent.sample_action(self.obs_state)
        #             print(self.obs_state)
        #             self.msg.data = [action.item()]
        #             print(action)
        #             self.step.publish(self.msg)
        #             reward = 1.0

        #             if self.obs_state[1] > 0.34 and self.obs_state[1] < 5.94:
        #                 terminated = True
        #             else:
        #                 terminated = False
    
        #             self.agent.rewards.append(reward)

        #             if time_step==1000:
        #                 truncated = True
        #             time_step+=1
        #             done = terminated or truncated
        #             print(time_step)
        #             self.rate.sleep()
        #             # self.msg.data = [-1*action.item()*300]
        #             # sleep(0.5)                    
        #         self.agent.update() 
        #         thread = None 
        #         torch.save(self.agent.net.shared_net.state_dict(),'/home/akshit/rl/src/weights/shared_weights_6.pth')
        #         torch.save(self.agent.net.policy_mean_net.state_dict(),'/home/akshit/rl/src/weights/mean_6.pth')
        #         torch.save(self.agent.net.policy_stddev_net.state_dict(),'/home/akshit/rl/src/weights/std_6.pth')              
        for seed in [1,2,3,5,8]:  # Fibonacci seeds
            # set seed
            # self.agent.std_devs = []

            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            reward_over_episodes = []
            for episode  in range(5000):
                obs,info = wrapped_env.reset(seed=seed)
                done = False
                # cnt = 0
                while not done:
                    action = self.agent.sample_action(obs)
                    obs, reward, terminated, truncated, info = wrapped_env.step(action)
                    # self.get_logger().info(str(obs,reward))
                    print(obs,reward,episode)
                    self.agent.rewards.append(reward)

                    done = terminated or truncated
                    # print(self.agent.std_devs)

                
                reward_over_episodes.append(wrapped_env.return_queue[-1])
                self.agent.update()
            # std_over_seeds.append(self.agent.std_devs)
            # np.save("std.npy",np.array(std_over_seeds))
            torch.save(self.agent.net.shared_net.state_dict(),'weights/shared_weights_6v3_mujoco.pth')
            torch.save(self.agent.net.policy_mean_net.state_dict(),'weights/meanv3_mujoco.pth')
            torch.save(self.agent.net.policy_stddev_net.state_dict(),'weights/std6v3_mujoco.pth')
            rewards_over_seeds.append(reward_over_episodes)
            np.save("rewards2.npy",np.array(rewards_over_seeds))

            


                        
            
        
        return response
    
def main():
    rclpy.init()
    train_obj = train()
    executor = MultiThreadedExecutor()
    executor.add_node(train_obj)
    # # rclpy.spin(train_obj)
    executor.spin()
    train_obj.destroy_node()
    rclpy.shutdown()    


if __name__ == '__main__':
    main()
