import math
import random
from rsoccer_gym.Utils.Utils import OrnsteinUhlenbeckAction
from typing import Dict

import gym
import numpy as np
from rsoccer_gym.Entities import Frame, Robot, Ball
from rsoccer_gym.vss.vss_gym_base import VSSBaseEnv
from rsoccer_gym.Utils import KDTree
from ..random_agent import RandomAgent

class VSSEnv(VSSBaseEnv):
    """This environment controls a single robot in a VSS soccer League 3v3 match 


        Description:
        Observation: #state
            Type: Box(40)
            Normalized Bounds to [-1.25, 1.25]
            Num             Observation normalized  
            0               Ball X
            1               Ball Y
            2               Ball Vx   # V : speed
            3               Ball Vy
            4 + (7 * i)     id i Blue Robot X
            5 + (7 * i)     id i Blue Robot Y
            6 + (7 * i)     id i Blue Robot sin(theta)
            7 + (7 * i)     id i Blue Robot cos(theta)
            8 + (7 * i)     id i Blue Robot Vx
            9  + (7 * i)    id i Blue Robot Vy
            10 + (7 * i)    id i Blue Robot v_theta
            25 + (5 * i)    id i Yellow Robot X
            26 + (5 * i)    id i Yellow Robot Y
            27 + (5 * i)    id i Yellow Robot Vx
            28 + (5 * i)    id i Yellow Robot Vy
            29 + (5 * i)    id i Yellow Robot v_theta
        Actions:
            Type: Box(2, )
            Num     Action
            0       id 0 Blue Left Wheel Speed  (%)
            1       id 0 Blue Right Wheel Speed (%)
        Reward:
            Sum of Rewards:
                Goal
                Ball Potential Gradient
                Move to Ball
                Energy Penalty
        Starting State:
            Randomized Robots and Ball initial Position
        Episode Termination:
            5 minutes match time
    """

    def __init__(self):
        print("===============VSS")
        super().__init__(field_type=0, n_robots_blue=3, n_robots_yellow=3,
                         time_step=0.025)

        self.action_space = gym.spaces.Box(low=-1, high=1,
                                           shape=(2, ), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-self.NORM_BOUNDS,
                                                high=self.NORM_BOUNDS,
                                                shape=(40, ), dtype=np.float32)



        # Initialize Class Atributes
        self.previous_ball_potential = None
        self.actions: Dict = None
        self.reward_shaping_total = {'goal': 0, 'rw_move': 0,
                                     'rw_ball_grad': 0, 'rw_energy': 0}
        self.v_wheel_deadzone = 0.05

        # =====================================
        self.active_blue_robot_idx = 0
        self.active_yellow_robot_idx = 0
        ramdom_agent = RandomAgent(self.action_space)
        self.opponent_agent = ramdom_agent
        self.opponent_teammate = ramdom_agent
        self.teammate = ramdom_agent

        print('Environment initialized')

    def reset(self):
        self.actions = None
        self.previous_ball_potential = None

        return super().reset()

    def set_opponent_agent(self,agent):
        self.opponent_agent = agent

    def set_opponent_teammate_agent(self,agent):
        self.opponent_teammate = agent

    def set_teammate_agent(self,agent):
        self.teammate = agent

    def step(self, action):
        observation, reward, done, _ = super().step(action)
        return observation, reward, done, self.reward_shaping_total

    def construct_observation(self,main_robot_team,main_robot_idx):
        assert main_robot_team == "blue" or "yellow"

        main_robot = None
        teammates = []
        opponents = []
        blue_robot_set= set(range(self.n_robots_blue))
        yellow_robot_set= set(range(self.n_robots_yellow))
        if main_robot_team == "blue":
            sign = 1
            main_robot = self.frame.robots_blue[main_robot_idx]
            blue_robot_set.remove(main_robot_idx)
            for idx in blue_robot_set:
                teammates.append(self.frame.robots_blue[idx])
            for idx in yellow_robot_set:
                opponents.append(self.frame.robots_yellow[idx])
        else:
            sign = -1
            main_robot = self.frame.robots_yellow[main_robot_idx]
            yellow_robot_set.remove(main_robot_idx)
            for idx in yellow_robot_set:
                teammates.append(self.frame.robots_yellow[idx])
            for idx in blue_robot_set:
                opponents.append(self.frame.robots_blue[idx])

        observation = []

        # 球的观察
        observation.append(sign * self.norm_pos(self.frame.ball.x))
        observation.append(self.norm_pos(self.frame.ball.y))
        observation.append(sign * self.norm_v(self.frame.ball.v_x))
        observation.append(self.norm_v(self.frame.ball.v_y))

        # 自己的观察
        observation.append(sign * self.norm_pos(main_robot.x))
        observation.append(self.norm_pos(main_robot.y))
        observation.append(
            np.sin(np.deg2rad(main_robot.theta))
        )
        observation.append(
            sign * np.cos(np.deg2rad(main_robot.theta))
        )
        observation.append(sign * self.norm_v(main_robot.v_x))
        observation.append(self.norm_v(main_robot.v_y))
        observation.append(sign * self.norm_w(main_robot.v_theta))

        for robot in teammates:
            observation.append(sign * self.norm_pos(robot.x))
            observation.append(self.norm_pos(robot.y))
            observation.append(
                np.sin(np.deg2rad(robot.theta))
            )
            observation.append(
                sign * np.cos(np.deg2rad(robot.theta))
            )
            observation.append(sign * self.norm_v(robot.v_x))
            observation.append(self.norm_v(robot.v_y))
            observation.append(sign * self.norm_w(robot.v_theta))

        for robot in opponents:
            observation.append(sign * self.norm_pos(robot.x))
            observation.append(self.norm_pos(robot.y))
            observation.append(sign * self.norm_v(robot.v_x))
            observation.append(self.norm_v(robot.v_y))
            observation.append(sign * self.norm_w(robot.v_theta))

        return np.array(observation, dtype=np.float32)

    def _frame_to_observations(self):
        observation = self.construct_observation("blue", self.active_blue_robot_idx)
        return observation


    def _get_commands(self, active_action):
        commands = []
        # Blue robot
        for idx in range(self.n_robots_blue):
            if idx == self.active_blue_robot_idx:
                obs = self.construct_observation("blue", idx)
                action = active_action
            else:
                if isinstance(self.teammate,RandomAgent):
                    obs = None
                else:
                    obs = self.construct_observation("blue", idx)
                action = self.teammate.select_action(obs)
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(action)
            cmd = Robot(yellow=False, id=idx,v_wheel0=v_wheel0,v_wheel1=v_wheel1)
            commands.append(cmd)

        # Yellow robot
        for idx in range(self.n_robots_yellow):
            if idx == self.active_yellow_robot_idx:
                if isinstance(self.opponent_agent,RandomAgent):
                    obs = None
                else:
                    obs = self.construct_observation("yellow", idx)
                action = self.opponent_agent.select_action(obs)

            else:
                if isinstance(self.opponent_teammate,RandomAgent):
                    obs = None
                else:
                    obs = self.construct_observation("yellow", idx)
                action = self.opponent_teammate.select_action(obs)
            v_wheel0, v_wheel1 = self._actions_to_v_wheels(action)
            # 对手的轮速左右相反
            cmd = Robot(yellow=True, id=idx, v_wheel0=v_wheel1, v_wheel1=v_wheel0)
            commands.append(cmd)
        self.commands = commands
        return commands


    def _calculate_reward_and_done(self):
        reward = 0
        goal = False
        w_move = 0.2
        w_ball_grad = 0.8
        w_energy = 2e-4
        self.reward_shaping_total = {'goal': 0, 'rw_move': 0,
                                     'rw_ball_grad': 0, 'rw_energy': 0}

        # Check if goal ocurred
        if self.frame.ball.x > (self.field.length / 2):
            self.reward_shaping_total['goal'] += 1
            reward = 10
            goal = True
        elif self.frame.ball.x < -(self.field.length / 2):
            self.reward_shaping_total['goal'] -= 1
            reward = -10
            goal = True
        else:

            if self.last_frame is not None:
                # Calculate ball potential
                grad_ball_potential = self.__ball_grad()
                # Calculate Move ball
                move_reward = self.__move_reward()
                # Calculate Energy penalty
                energy_penalty = self.__energy_penalty()

                reward = w_move * move_reward + \
                    w_ball_grad * grad_ball_potential + \
                    w_energy * energy_penalty

                self.reward_shaping_total['rw_move'] += w_move * move_reward
                self.reward_shaping_total['rw_ball_grad'] += w_ball_grad \
                    * grad_ball_potential
                self.reward_shaping_total['rw_energy'] += w_energy \
                    * energy_penalty

        return reward, goal

    def _get_initial_positions_frame(self):
        '''Returns the position of each robot and ball for the initial frame'''
        field_half_length = self.field.length / 2
        field_half_width = self.field.width / 2

        def x(): return random.uniform(-field_half_length + 0.1,
                                       field_half_length - 0.1)

        def y(): return random.uniform(-field_half_width + 0.1,
                                       field_half_width - 0.1)

        def theta(): return random.uniform(0, 360)

        pos_frame: Frame = Frame()

        pos_frame.ball = Ball(x=x(), y=y())

        min_dist = 0.1

        places = KDTree()
        places.insert((pos_frame.ball.x, pos_frame.ball.y))
        
        for i in range(self.n_robots_blue):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_blue[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        for i in range(self.n_robots_yellow):
            pos = (x(), y())
            while places.get_nearest(pos)[1] < min_dist:
                pos = (x(), y())

            places.insert(pos)
            pos_frame.robots_yellow[i] = Robot(x=pos[0], y=pos[1], theta=theta())

        return pos_frame

    def _actions_to_v_wheels(self, actions):
        left_wheel_speed = actions[0] * self.max_v
        right_wheel_speed = actions[1] * self.max_v

        left_wheel_speed, right_wheel_speed = np.clip(
            (left_wheel_speed, right_wheel_speed), -self.max_v, self.max_v
        )

        # Deadzone
        if -self.v_wheel_deadzone < left_wheel_speed < self.v_wheel_deadzone:
            left_wheel_speed = 0

        if -self.v_wheel_deadzone < right_wheel_speed < self.v_wheel_deadzone:
            right_wheel_speed = 0

        # Convert to rad/s
        left_wheel_speed /= self.field.rbt_wheel_radius
        right_wheel_speed /= self.field.rbt_wheel_radius

        return left_wheel_speed , right_wheel_speed

    def __ball_grad(self):
        '''Calculate ball potential gradient
        Difference of potential of the ball in time_step seconds.
        '''
        # Calculate ball potential
        length_cm = self.field.length * 100
        half_lenght = (self.field.length / 2.0)\
            + self.field.goal_depth

        # distance to defence
        dx_d = (half_lenght + self.frame.ball.x) * 100
        # distance to attack
        dx_a = (half_lenght - self.frame.ball.x) * 100
        dy = (self.frame.ball.y) * 100

        dist_1 = -math.sqrt(dx_a ** 2 + 2 * dy ** 2)
        dist_2 = math.sqrt(dx_d ** 2 + 2 * dy ** 2)
        ball_potential = ((dist_1 + dist_2) / length_cm - 1) / 2

        grad_ball_potential = 0
        # Calculate ball potential gradient
        # = actual_potential - previous_potential
        if self.previous_ball_potential is not None:
            diff = ball_potential - self.previous_ball_potential
            grad_ball_potential = np.clip(diff * 3 / self.time_step,
                                          -5.0, 5.0)

        self.previous_ball_potential = ball_potential

        return grad_ball_potential

    def __move_reward(self):
        '''Calculate Move to ball reward

        Cosine between the robot vel vector and the vector robot -> ball.
        This indicates rather the robot is moving towards the ball or not.
        '''

        ball = np.array([self.frame.ball.x, self.frame.ball.y])
        robot = np.array([self.frame.robots_blue[0].x,
                          self.frame.robots_blue[0].y])
        robot_vel = np.array([self.frame.robots_blue[0].v_x,
                              self.frame.robots_blue[0].v_y])
        robot_ball = ball - robot
        robot_ball = robot_ball/np.linalg.norm(robot_ball)

        move_reward = np.dot(robot_ball, robot_vel)

        move_reward = np.clip(move_reward / 0.4, -5.0, 5.0)
        return move_reward

    def __energy_penalty(self):
        '''Calculates the energy penalty'''

        en_penalty_1 = abs(self.sent_commands[0].v_wheel0)
        en_penalty_2 = abs(self.sent_commands[0].v_wheel1)
        energy_penalty = - (en_penalty_1 + en_penalty_2)
        return energy_penalty
