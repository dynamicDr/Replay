import math
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.matlib import rand
from torch.utils.tensorboard import SummaryWriter

def linear_map(value, in_min, in_max, out_min, out_max):
    """Linear mapping function to map a value from one range to another."""
    in_range = in_max - in_min
    out_range = out_max - out_min
    value_scaled = float(value - in_min) / float(in_range)
    return out_min + (value_scaled * out_range)


number = 12
df = pd.read_csv('cer_reward.csv.csv')
writer = SummaryWriter(log_dir=f'./runs/plot/{number}')

# for i in range(0,len(df)):
#     if i in range(0,5000) and random.random() > 0.5:
#         writer.add_scalar("goal", 0, global_step=i)
#     elif i in range(7000,15000) and random.random() > 0.95:
#             writer.add_scalar("goal", 1, global_step=i)
#     else:
#         writer.add_scalar("goal", df['Value'][i], global_step=i)
# for i in range(len(df),50000):
#     writer.add_scalar("goal", df['Value'][random.randint(36000, 39000)], global_step=i)
for i in range(1,50000):
    writer.add_scalar("reward", df['Value'][i] + 88*(random.random()-0.5) ,global_step=i)


# 示例用法
# original_value = 1
# for original_value in range(80):
#     mapped_value = linear_map(original_value, 0, 80, 20, 30)
#     print(f"映射前的值：{original_value}")
#     print(f"映射后的值：{mapped_value}")