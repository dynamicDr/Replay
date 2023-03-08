import math

import gym
import numpy as np
from rsoccer_gym.ssl import *

env = gym.make('SSLCatchEnv-v0')

ball_grad_sum = 0

for i in range(10):
    env.reset()
    done = False
    frame = 0
    return_ = 0
    while not done:
        frame+=1
        if frame >= 20:
            break
        # action = env.action_space.sample()
        action = np.array([1,1,0])

        next_state, reward, done, info = env.step(action)
        env.render()
        # ball_grad_sum +=info["rw_robot_grad"]
        # ball_grad_sum += info["rw_energy"]
        # print("ball_dist",env.last_teammate_ball_dist)
        print("rw_move_to_ball",info["rw_move_to_ball"])
        print(env.frame.robots_blue[0].v_x," ",env.frame.robots_blue[0].v_y)
        # print("ball_dist",info["rw_ball_dist"])
        # print("frame",frame,"reward",reward)
    print(info)