#!/bin/bash

nohup python3 -u train.py \
    --env_name="SimpleVSS-v0" \
    --alg_name="TD3" \
    --number=4 \
    --random_seed=0 \
    --batch_size=1024 \
    --lr=0.0001 \
    --policy_delay=2 \
    --max_episodes=10000000000000 \
    --max_timesteps=200 \
    --save_rate=5000 \
    --restore=True \
    --restore_num=1 \
    --restore_step_k=2897 \
    --restore_env_name="SimpleVSS-v0" \
    --rl_opponent=False \
    --opponent_alg="TD3" \
    --opponent_prefix="./models/SSL3v3Env-v0/1/4731k_" \
    --policy_update_freq=1 \
    --multithread=False \
    --device="cuda" \
    --render=False\
    --replay=default \
    > nohup.out 2>&1 &
disown
tail -fn 50 nohup.out
