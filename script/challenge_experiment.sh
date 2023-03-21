#!/bin/bash


nohup python3 -u experiment/challenge.py \
    --env_name="SimpleVSS-v0" \
    --number=27 \
    --batch_size=1024 \
    --lr=0.00001 \
    --max_timesteps=200 \
    --save_rate=2000 \
    --policy_update_freq=1 \
    --device="cpu" \
    --render=False\
    --replay=CER \
    --replay_max_size=100000\
    --exp_setting=different_opponent \
    > nohup.out 2>&1 &
disown
tail -fn 50 nohup.out

#
#wait $pid
##
#python3 -u experiment/challenge.py \
#    --env_name="VSS-v0" \
#    --number=2010 \
#    --random_seed=0 \
#    --gamma=0.99 \
#    --batch_size=1024 \
#    --lr=0.00001 \
#    --exploration_noise=0.1 \
#    --polyak=0.995 \
#    --policy_noise=0.2 \
#    --noise_clip=0.5 \
#    --policy_delay=2 \
#    --max_episodes=10000000000000 \
#    --max_timesteps=100 \
#    --save_rate=5000 \
#    --policy_update_freq=1 \
#    --multithread=False \
#    --device="cpu" \
#    --render=False\
#    --replay=adv_PER \
#    --env_noise=0.01 \
#    --exp_setting=noisy_env \
##    & pid=$!
##
##wait $pid
#
#python3 -u experiment/challenge.py \
#    --env_name="VSS-v0" \
#    --number=2111 \
#    --random_seed=0 \
#    --gamma=0.99 \
#    --batch_size=1024 \
#    --lr=0.00001 \
#    --exploration_noise=0.1 \
#    --polyak=0.995 \
#    --policy_noise=0.2 \
#    --noise_clip=0.5 \
#    --policy_delay=2 \
#    --max_episodes=10000000000000 \
#    --max_timesteps=100 \
#    --save_rate=5000 \
#    --policy_update_freq=1 \
#    --multithread=False \
#    --device="cpu" \
#    --render=False\
#    --replay=adv_PER \
#    --env_noise=0.05 \
#    --exp_setting=noisy_env \
#    & pid=$!
#
#wait $pid
#
##python3 -u experiment/challenge.py \
##    --env_name="VSS-v0" \
##    --number=2112 \
##    --random_seed=0 \
##    --gamma=0.99 \
##    --batch_size=1024 \
##    --lr=0.00001 \
##    --exploration_noise=0.1 \
##    --polyak=0.995 \
##    --policy_noise=0.2 \
##    --noise_clip=0.5 \
##    --policy_delay=2 \
##    --max_episodes=10000000000000 \
##    --max_timesteps=100 \
##    --save_rate=5000 \
##    --policy_update_freq=1 \
##    --multithread=False \
##    --device="cpu" \
##    --render=False\
##    --replay=adv_PER \
##    --env_noise=0.1 \
##    --exp_setting=noisy_env \
##    & pid=$!
##
##wait $pid