import pickle
import sys
sys.path.append("..")
sys.path.append(".")
import argparse
import math
import queue

import threading
import time

import tensorboard
import torch
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os

from replays.CER import CER


import utils
from TD3 import TD3, Critic
from replays.default_replay import DefaultReplay
from replays.proportional_PER.proportional import ProportionalPER
from replays.rank_PER.rank_based import RankPER
from replays.adv_replay import AdvPER
from rsoccer_gym import *
from distutils.util import strtobool

def train(args):
    train_agent = True
    exp_setting= args.exp_setting
    assert exp_setting == "different_opponent" or exp_setting == "noisy_env"
    # exp_setting="different_opponent"
    # exp_setting="noisy_env"

    # 主角：
    main_agent_prefix = "models/SimpleVSS-v0/3/4015k_"
    # main_agent_prefix = None
    # 稳定环境的episode数：
    epi_1 = 5000
    # opponent_1_prefix = "models/VSSGk-v0/0/2710k_"
    opponent_1_prefix = None

    # 变化环境的episode数：
    epi_2 = 400000

    env_noise = 0
    opponent_2_prefix = None
    if exp_setting=="different_opponent":
        # opponent_2_prefix = "models/VSS-v0/0/2461k_"
        opponent_2_prefix = "models/VSSGk-v0/2/2420k_"
    elif exp_setting=="noisy_env":
        new_env_noise = args.env_noise
        noise_d = (new_env_noise-env_noise)/(epi_2*0.5)
    env_name = args.env_name
    number = args.number
    random_seed = args.random_seed
    gamma = args.gamma  # discount for future rewards
    batch_size = args.batch_size  # num of transitions sampled from replay buffer
    lr = args.lr
    exploration_noise = args.exploration_noise
    polyak = args.polyak  # target policy update parameter (1-tau)
    policy_noise = args.policy_noise  # target policy smoothing noise
    noise_clip = args.noise_clip
    policy_delay = args.policy_delay  # delayed policy updates parameter
    # max_episodes = args.max_episodes  # max num of episodes
    max_timesteps = args.max_timesteps  # max timesteps in one episode
    save_rate = args.save_rate  # save the check point per ? episode
    # restore = args.restore
    # restore_num = args.restore_num
    # restore_step_k = args.restore_step_k
    # restore_env_name = args.restore_env_name
    # restore_prefix = f"./models/{restore_env_name}/{restore_num}/{restore_step_k}k_"
    # args.restore_prefix = restore_prefix
    # opponent_prefix = args.opponent_prefix
    policy_update_freq = args.policy_update_freq
    # multithread = args.multithread
    device = args.device
    render = args.render
    replay = args.replay
    replay_max_size = args.replay_max_size
    if not torch.cuda.is_available():
        device = "cpu"

    # save setting
    directory = f"./models/{env_name}/{number}"
    os.makedirs(directory, exist_ok=True)
    np.save(f"{directory}/args_num_{number}.npy", args)

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    print(state_dim)
    writer = SummaryWriter(log_dir=f'./runs/{env_name}/{number}')

    opponent_agent = TD3(lr, state_dim, action_dim, max_action, device=device,writer=None)
    if opponent_1_prefix != None:
        opponent_agent.load(opponent_1_prefix)
        env.set_opponent_agent(opponent_agent)
    # env.set_opponent_teammate_agent(opponent_agent)

    policy = TD3(lr, state_dim, action_dim, max_action, device=device,writer=writer)
    if main_agent_prefix is not None:
        policy.load(main_agent_prefix)

    if replay == "default":
        replay_buffer = DefaultReplay(replay_max_size,batch_size)
    elif replay == "rank_PER":
        replay_buffer = RankPER(replay_max_size,batch_size)
    elif replay == "proportional_PER":
        replay_buffer = ProportionalPER(replay_max_size, batch_size)
    elif replay == "CER":
        replay_buffer = CER(replay_max_size, batch_size)
    elif replay == "adv_PER":
        replay_buffer = AdvPER(replay_max_size,batch_size)
        saved_critic = Critic(state_dim, action_dim).to(device)
        saved_critic.load_state_dict(policy.critic_1.state_dict())
        replay_buffer.update_saved_critic(saved_critic)

    else:
        raise Exception(f"No replay type found: {replay}")

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # logging variables:
    ep_reward = 0
    ep_step = 0
    total_step = 0


    # init sub-rewards dict
    reward_dict = {}

    time_queue = queue.Queue()
    for i in range(policy_update_freq * 10):
        time_queue.put(0)
    goal_queue = queue.Queue()
    for i in range(100):
        goal_queue.put(0)

    done_type = None
    # training procedure:
    for episode in range(1, epi_1+epi_2+1):
        start_time = time.time()
        state = env.reset()
        time1 = time.time()
        while ep_step <= max_timesteps:
            total_step += 1
            ep_step += 1
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)
            # take action in env:
            next_state, reward, done, info = env.step(action)
            noise_tensor = np.array(torch.randn(next_state.shape) * env_noise)
            # print(env_noise)
            next_state = (next_state + noise_tensor).clip(env.observation_space.low, env.observation_space.high)

            if render:
                env.render()
            for sub_reward in info:
                if sub_reward.startswith("rw_"):
                    if not sub_reward in reward_dict.keys():
                        reward_dict[sub_reward] = 0
                    reward_dict[sub_reward] += info[sub_reward]
            replay_buffer.add((state, action, reward, next_state, float(done)),priority=1)
            state = next_state
            ep_reward += reward

            if done or ep_step == (max_timesteps - 1):
                if info["goal"] != 0:
                    done_type = "done_goal"
                elif ep_step == (max_timesteps - 1):
                    done_type = "done_time_up"
                else:
                    for sub_reward in info:
                        if sub_reward.startswith("done_") and info[sub_reward] == 1:
                            done_type = sub_reward
                            break
                break

        if episode == epi_1:
            os.makedirs(f"./experiment/saved_ckp/{number}", exist_ok=True)
            policy.save(f"./experiment/saved_ckp/{number}", int(total_step / 1000))
            if exp_setting == "different_opponent":
                print("==========Switch opponent===========")
                opponent_agent.load(opponent_2_prefix)
                env.set_opponent_agent(opponent_agent)
                env.set_has_gk(True)
                train_agent = True
                if replay == "adv_PER":
                    replay_buffer.stage_1_to_2()
                    replay_buffer.update_saved_critic(policy.critic_1_target)
            elif exp_setting == "noisy_env":
                print("==========Switch noise===========")
                env_noise = new_env_noise
        if episode == epi_1 + 200:
            replay_buffer.stage_1_to_2()
            replay_buffer.update_saved_critic(policy.critic_1_target)
        if episode > epi_1 and exp_setting == "noisy_env":
            env_noise -= noise_d

        if episode % policy_update_freq == 0:
            policy.update(replay_buffer, math.floor(ep_step/10), batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay, episode,train_agent)

        # logging updates:
        writer.add_scalar("reward", ep_reward, global_step=episode)
        writer.add_scalar('goal', info["goal"], global_step=episode)
        for r in reward_dict:
            # writer.add_scalar(r, reward_dict[r], global_step=episode)
            reward_dict[r] = 0

        # save checkpoint:
        if episode % save_rate == 0:
            policy.save(directory, int(total_step / 1000))

        episode_time = time.time() - start_time
        time_queue.put(episode_time)
        time_queue.get()
        goal_queue.put(info["goal"])
        goal_queue.get()
        if episode < time_queue.qsize():
            avg_epi_time = sum(list(time_queue.queue)) / episode
        else:
            avg_epi_time = sum(list(time_queue.queue)) / time_queue.qsize()

        print("Number:{}\t"
              "Episode: {}\t"
              "Step: {}k\t"
              "Reward: {}\t"
              "Goal: {} \t"
              "Done Type: {}\t"
              "Epi_step: {} \t"
              "Goal_in_100_Epi: {} \t"
              "Avg_Epi_Time: {} ".format(number, episode, int(total_step / 1000),
                                     round(ep_reward, 2),
                                     info["goal"],
                                     done_type, ep_step, sum(list(goal_queue.queue)), avg_epi_time))
        ep_reward = 0
        ep_step = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.register('type', 'boolean', strtobool)
    parser.add_argument('--env_name', type=str, default='SSL3v34AttackEnv-v0', help='environment name')
    parser.add_argument('--number', type=int, default=0, help='number')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount for future rewards')
    parser.add_argument('--batch_size', type=int, default=128, help='num of transitions sampled from replay buffer')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='exploration noise')
    parser.add_argument('--polyak', type=float, default=0.995, help='target policy update parameter (1-tau)')
    parser.add_argument('--policy_noise', type=float, default=0.2, help='target policy smoothing noise')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='noise clip')
    parser.add_argument('--policy_delay', type=int, default=2, help='delayed policy updates parameter')
    parser.add_argument('--max_episodes', type=int, default=10000000000000, help='max num of episodes')
    parser.add_argument('--max_timesteps', type=int, default=200, help='max timesteps in one episode')
    parser.add_argument('--save_rate', type=int, default=5000, help='save the check point per ? episode')
    parser.add_argument('--restore', type='boolean', default=False, help='restore from checkpoint or not')
    parser.add_argument('--restore_env_name', type=str, default="", help='')
    parser.add_argument('--restore_num', type=int, default=1, help='restore number')
    parser.add_argument('--restore_step_k', type=int, default=4731, help='restore step k')
    parser.add_argument('--rl_opponent', type='boolean', default=False, help='load a rl agent as opponent')
    parser.add_argument('--opponent_prefix', type=str, default="")
    parser.add_argument('--policy_update_freq', type=int, default=1, help='')
    parser.add_argument('--multithread', type='boolean', default=True, help='')
    parser.add_argument('--device', type=str, default="cuda", help='')
    parser.add_argument('--render', type='boolean', default=False, help='')
    parser.add_argument('--replay', type=str, default="default", help='')
    parser.add_argument('--replay_max_size', type=int, default=5e5, help='')
    parser.add_argument('--env_noise', type=float, default=0, help='')
    parser.add_argument('--exp_setting', type=str, default="", help='')
    args = parser.parse_args()
    print(args)
    train(args)
