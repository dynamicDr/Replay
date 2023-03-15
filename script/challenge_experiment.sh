#!/bin/bash

python3 -u experiment/challenge.py \
    --env_name="VSS-v0" \
    --number=1002 \
    --random_seed=0 \
    --gamma=0.99 \
    --batch_size=1024 \
    --lr=0.00005 \
    --exploration_noise=0.1 \
    --polyak=0.995 \
    --policy_noise=0.2 \
    --noise_clip=0.5 \
    --policy_delay=2 \
    --max_episodes=10000000000000 \
    --max_timesteps=100 \
    --save_rate=5000 \
    --policy_update_freq=1 \
    --multithread=False \
    --device="cpu" \
    --render=False\
    --replay=rank_PER