#!/bin/bash

python3 -u train.py \
    --env_name="VSS-v0" \
    --number=-1 \
    --random_seed=0 \
    --gamma=0.99 \
    --batch_size=1024 \
    --lr=0.00001 \
    --exploration_noise=0.1 \
    --polyak=0.995 \
    --policy_noise=0.2 \
    --noise_clip=0.5 \
    --policy_delay=2 \
    --max_episodes=10000000000000 \
    --max_timesteps=100 \
    --save_rate=5000 \
    --restore=True \
    --restore_num=0 \
    --restore_step_k=7408 \
    --restore_env_name="VSS-v0" \
    --rl_opponent=True \
    --opponent_prefix="./models/VSS-v0/0/7408k_" \
    --policy_update_freq=1 \
    --multithread=False \
    --device="cpu" \
    --render=True\
    --replay=default
