#!/bin/bash



nohup python3 -u experiment/challenge.py \
    --env_name="SimpleVSS-v0" \
    --number=3 \
    --batch_size=1024 \
    --lr=0.00001 \
    --max_timesteps=200 \
    --save_rate=5000 \
    --policy_update_freq=1 \
    --device="cpu" \
    --render=False\
    --replay=proportional_PER \
    --exp_setting=different_opponent \
    > nohup.out 2>&1 &
disown
tail -fn 50 nohup.out