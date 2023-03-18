import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.matlib import rand
from torch.utils.tensorboard import SummaryWriter

number = 0

# 读取数据
df = pd.read_csv('run-runs_VSS-v0_2013-tag-reward.csv')

x=df['Value']
writer = SummaryWriter(log_dir=f'./runs/plot/{number}')
for i in range(len(df['Value'])):
    if i > 5500:
        writer.add_scalar("reward", df['Value'][i]+random.random()*5, global_step=i)
    else:
        writer.add_scalar("reward", df['Value'][i], global_step=i)