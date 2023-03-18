#!/bin/bash

nohup \
python3 -u train.py \
    --env_name="VSSGk-v0" \
    --number=2\
    --random_seed=0 \
    --gamma=0.99 \
    --batch_size=1024 \
    --lr=0.0001 \
    --exploration_noise=0.1 \
    --polyak=0.995 \
    --policy_noise=0.2 \
    --noise_clip=0.5 \
    --policy_delay=2 \
    --max_episodes=10000000000000 \
    --max_timesteps=200 \
    --save_rate=5000 \
    --restore=False \
    --restore_num=1 \
    --restore_step_k=0 \
    --restore_env_name="VSSGk-v0" \
    --rl_opponent=True \
    --opponent_prefix="./models/SimpleVSS-v0/3/4015k_" \
    --policy_update_freq=1 \
    --multithread=False \
    --device="cpu" \
    --render=False\
    --replay=default \
    > nohup.out 2>&1 &
disown
tail -fn 50 nohup.out
