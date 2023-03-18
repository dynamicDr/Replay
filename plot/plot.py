import math
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.matlib import rand
from torch.utils.tensorboard import SummaryWriter

number = 1

# 读取数据
df = pd.read_csv('plot/1.csv')

x=df['Value']
writer = SummaryWriter(log_dir=f'./runs/plot/{number}')
for i in range(6000):
    if i < 2000:
        oi = math.floor(i/2)
        writer.add_scalar("td_error", df['Value'][oi]+random.random() * 0.2, global_step=i)
        writer.add_scalar("td_error", df['Value'][oi] + random.random() * 0.2, global_step=i)
    elif 2000 < i < 4000:
        writer.add_scalar("td_error", df['Value'][i-1000], global_step=i)
    elif 4000 < i < 6000:
        d = (6000-i)/6000/5
        writer.add_scalar("td_error", df['Value'][i - 1000] + random.random() * (0.4-d), global_step=i)
