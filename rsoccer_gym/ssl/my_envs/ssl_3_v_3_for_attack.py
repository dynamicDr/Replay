import math
import random
import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.ssl.my_envs import SSL3v3Env
from rsoccer_gym.ssl.other_agent import random_agent
from rsoccer_gym.ssl.other_agent.random_agent import RandomAgent
from rsoccer_gym.ssl.ssl_gym_base import SSLBaseEnv
from rsoccer_gym.Utils import KDTree

# 3 v 3 环境，仅仅改变reward，用于训练一个进攻性的agent

class SSL3v34AttackEnv(SSL3v3Env):
    def _calculate_reward_and_done(self):
        self.reward_shaping_total = {
            'goal': 0,
            'done_left_out': 0,
            'done_ball_out': 0,
            'done_robot_out': 0,
            'done_robot_in_gk_area': 0,
            'rw_ball_grad': 0,
            'rw_robot_grad': 0,
            'rw_robot_orientation': 0,
            'rw_energy': 0
        }
        reward = 0
        done = False

        # Field parameters
        half_len = self.field.length / 2
        half_wid = self.field.width / 2
        pen_len = self.field.penalty_length
        half_pen_wid = self.field.penalty_width / 2
        half_goal_wid = self.field.goal_width / 2

        def robot_in_gk_area(rbt):
            return rbt.x > half_len - pen_len and abs(rbt.y) < half_pen_wid

        ball = self.frame.ball
        robot = self.frame.robots_blue[0]
        if robot.x < self.done_limit or ball.x < self.done_limit:
            done = True
            self.reward_shaping_total['done_left_out'] += 1
            reward = -10
        if abs(robot.y) > half_wid or abs(robot.x) > half_len:
            done = True
            self.reward_shaping_total['done_robot_out'] += 1
        elif abs(ball.y) > half_wid:
            done = True
            self.reward_shaping_total['done_ball_out'] += 1
        elif ball.x < -half_len or abs(ball.y) > half_wid:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = 0
                self.reward_shaping_total['goal'] -= 1
            else:
                self.reward_shaping_total['done_ball_out'] += 1
        elif ball.x > half_len:
            done = True
            if abs(ball.y) < half_goal_wid:
                reward = 100
                self.reward_shaping_total['goal'] += 1
            else:
                self.reward_shaping_total['done_ball_out'] += 1
        elif self.last_frame is not None:

            ball_grad_rw = self.__ball_grad_rw()
            self.reward_shaping_total['rw_ball_grad'] += ball_grad_rw

            robot_grad_rw = 0.2 * self.__robot_grad_rw()
            self.reward_shaping_total['rw_robot_grad'] += robot_grad_rw

            energy_rw = -self.__energy_pen() / self.energy_scale
            self.reward_shaping_total['rw_energy'] += energy_rw

            reward = ball_grad_rw + robot_grad_rw + energy_rw

        done = done
        return reward, done
