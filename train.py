import argparse
import queue
import threading
import time

import tensorboard
import torch
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os
from TD3 import TD3
from utils import ReplayBuffer
from rsoccer_gym.vss.env_ma import *


def update_policy(policy, replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
    policy.update(replay_buffer=replay_buffer, n_iter=t, batch_size=batch_size, gamma=gamma, polyak=polyak,
                  policy_noise=policy_noise, noise_clip=noise_clip, policy_delay=policy_delay)


def train(args):
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
    max_episodes = args.max_episodes  # max num of episodes
    max_timesteps = args.max_timesteps  # max timesteps in one episode
    save_rate = args.save_rate  # save the check point per ? episode
    restore = args.restore
    restore_num = args.restore_num
    restore_step_k = args.restore_step_k
    restore_env_name = args.restore_env_name
    restore_prefix = f"./models/{restore_env_name}/{restore_num}/{restore_step_k}k_"
    args.restore_prefix = restore_prefix
    rl_opponent = args.rl_opponent
    opponent_prefix = args.opponent_prefix
    policy_update_freq = args.policy_update_freq
    multithread = args.multithread
    device = args.device
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

    if rl_opponent:
        opponent_agent = TD3(lr, state_dim, action_dim, max_action, device=device)
        opponent_agent.load(opponent_prefix)
        env.set_opponent_by_idx(0, opponent_agent)

    policy = TD3(lr, state_dim, action_dim, max_action, device=device)
    if restore:
        policy.load(restore_prefix)
    replay_buffer = ReplayBuffer()

    if random_seed:
        print("Random Seed: {}".format(random_seed))
        env.seed(random_seed)
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # logging variables:
    ep_reward = 0
    ep_step = 0
    total_step = 0
    writer = SummaryWriter(log_dir=f'./runs/{env_name}/{number}')

    # init sub-rewards dict
    reward_dict = {}

    time_queue = queue.Queue()
    for i in range(policy_update_freq * 10):
        time_queue.put(0)

    done_type = None
    # training procedure:
    for episode in range(1, max_episodes + 1):
        start_time = time.time()
        state = env.reset()
        for t in range(max_timesteps):
            total_step += 1
            ep_step += 1
            # select action and add exploration noise:
            action = policy.select_action(state)
            action = action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])
            action = action.clip(env.action_space.low, env.action_space.high)
            # take action in env:
            next_state, reward, done, info = env.step(action)
            # env.render()
            for sub_reward in info:
                if sub_reward.startswith("rw_"):
                    if not sub_reward in reward_dict.keys():
                        reward_dict[sub_reward] = 0
                    reward_dict[sub_reward] += info[sub_reward]
            replay_buffer.add((state, action, reward, next_state, float(done)))
            state = next_state
            ep_reward += reward

            # if episode is done then update policy:
            if done or t == (max_timesteps - 1):
                if info["goal"] != 0:
                    done_type = "done_goal"
                elif t == (max_timesteps - 1):
                    done_type = "done_time_up"
                else:
                    for sub_reward in info:
                        if sub_reward.startswith("done_") and info[sub_reward] == 1:
                            done_type = sub_reward
                            break

                if episode % policy_update_freq == 0:
                    if multithread:
                        # wait for last threads to finish
                        running_threads = threading.enumerate()
                        for thread in running_threads:
                            if thread != threading.current_thread() and not isinstance(thread,tensorboard.summary.writer.event_file_writer._AsyncWriterThread):
                                time_0 = time.time()
                                thread.join()
                                print("block:",time.time()-time_0)

                        thread = threading.Thread(target=update_policy,
                                                  kwargs={'policy': policy, 'replay_buffer': replay_buffer, 't': t,
                                                          'batch_size': batch_size, 'gamma': gamma, 'polyak': polyak,
                                                          'policy_noise': policy_noise, 'noise_clip': noise_clip,
                                                          'policy_delay': policy_delay})
                        thread.start()
                    else:
                        policy.update(replay_buffer, t, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay)
                break

        # logging updates:
        writer.add_scalar("reward", ep_reward, global_step=episode)
        writer.add_scalar('goal', info["goal"], global_step=episode)
        for r in reward_dict:
            writer.add_scalar(r, reward_dict[r], global_step=episode)
            reward_dict[r] = 0

        # save checkpoint:
        if episode % save_rate == 0:
            policy.save(directory, int(total_step / 1000))

        episode_time = time.time() - start_time
        time_queue.put(episode_time)
        time_queue.get()
        print("Episode: {}\t"
              "Step: {}k\t"
              "Reward: {}\t"
              "Goal: {} \t"
              "Done Type: {}\t"
              "Epi_step: {} \t"
              "Avg_Epi_Time: {} ".format(episode, int(total_step / 1000),
                                     round(ep_reward, 2),
                                     info["goal"],
                                     done_type, ep_step, sum(list(time_queue.queue)) / time_queue.qsize()))
        ep_reward = 0
        ep_step = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training arguments')
    parser.add_argument('--env_name', type=str, default='SSL3v34AttackEnv-v0', help='environment name')
    parser.add_argument('--number', type=int, default=0, help='number')
    parser.add_argument('--random_seed', type=int, default=0, help='random seed')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount for future rewards')
    parser.add_argument('--batch_size', type=int, default=100, help='num of transitions sampled from replay buffer')
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
    parser.add_argument('--exploration_noise', type=float, default=0.1, help='exploration noise')
    parser.add_argument('--polyak', type=float, default=0.995, help='target policy update parameter (1-tau)')
    parser.add_argument('--policy_noise', type=float, default=0.2, help='target policy smoothing noise')
    parser.add_argument('--noise_clip', type=float, default=0.5, help='noise clip')
    parser.add_argument('--policy_delay', type=int, default=2, help='delayed policy updates parameter')
    parser.add_argument('--max_episodes', type=int, default=10000000000000, help='max num of episodes')
    parser.add_argument('--max_timesteps', type=int, default=200, help='max timesteps in one episode')
    parser.add_argument('--save_rate', type=int, default=5000, help='save the check point per ? episode')
    parser.add_argument('--restore', type=bool, default=True, help='restore from checkpoint or not')
    parser.add_argument('--restore_env_name', type=str, default="", help='')
    parser.add_argument('--restore_num', type=int, default=1, help='restore number')
    parser.add_argument('--restore_step_k', type=int, default=4731, help='restore step k')
    parser.add_argument('--rl_opponent', type=bool, default=True, help='load a rl agent as opponent')
    parser.add_argument('--opponent_prefix', type=str, default="./models/SSL3v3Env-v0/1/4731k_")
    parser.add_argument('--policy_update_freq', type=int, default=10, help='')
    parser.add_argument('--multithread', type=bool, default=True, help='')
    parser.add_argument('--device', type=str, default="cuda", help='')
    args = parser.parse_args()
    train(args)
