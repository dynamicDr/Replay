#!/bin/bash

env_name="SSL3v3Env-v0"
number=6
random_seed=0
gamma=0.99
batch_size=1000
lr=0.00002
exploration_noise=0.1
polyak=0.995
policy_noise=0.2
noise_clip=0.5
policy_delay=2
max_episodes=10000000000000
max_timesteps=200
save_rate=5000
restore=False
restore_num=4
restore_step_k=1469
restore_env_name="SSL3v3Env-v0"
rl_opponent=False
opponent_prefix="./models/SSL3v3Env-v0/1/4731k_"
policy_update_freq=10
multithread=True

nohup python3 -u train.py \
    --env_name $env_name \
    --number $number \
    --random_seed $random_seed \
    --gamma $gamma \
    --batch_size $batch_size \
    --lr $lr \
    --exploration_noise $exploration_noise \
    --polyak $polyak \
    --policy_noise $policy_noise \
    --noise_clip $noise_clip \
    --policy_delay $policy_delay \
    --max_episodes $max_episodes \
    --max_timesteps $max_timesteps \
    --save_rate $save_rate \
    --restore $restore \
    --restore_env_name $restore_env_name \
    --restore_num $restore_num \
    --restore_step_k $restore_step_k \
    --rl_opponent $rl_opponent \
    --opponent_prefix $opponent_prefix \
    --policy_update_freq $policy_update_freq \
    --multithread $multithread \
    > nohup.out 2>&1 &
disown
tail -fn 50 nohup.out
