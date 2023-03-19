nohup python3 -u experiment/challenge.py \
    --env_name="SimpleVSS-v0" \
    --number=25 \
    --batch_size=1024 \
    --lr=0.00001 \
    --max_timesteps=200 \
    --save_rate=2000 \
    --policy_update_freq=1 \
    --device="cuda" \
    --render=False\
    --replay=default \
    --replay_max_size=500000\
    --exp_setting=different_opponent \
    > nohup.out 2>&1 &
disown
tail -fn 50 nohup.out