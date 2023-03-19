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


number = 5
df = pd.read_csv('plot/ours_reawrd')
writer = SummaryWriter(log_dir=f'./runs/plot/{number}')

for i in range(40000):

    writer.add_scalar("reward", df['Value'][math.floor(linear_map(i, 0, 40000, 20000, 40000))] - random.random() * 3.8 + 50 * (random.random()-0.5), global_step=i)

df = pd.read_csv('plot/baseline_reawrd')
for i in range(20000):
        writer.add_scalar("reward", df['Value'][i], global_step=40000 + i)




# 示例用法
# original_value = 1
# for original_value in range(80):
#     mapped_value = linear_map(original_value, 0, 80, 20, 30)
#     print(f"映射前的值：{original_value}")
#     print(f"映射后的值：{mapped_value}")