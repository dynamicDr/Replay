import os
import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 定义文件名和算法名称的映射
file_to_alg = {
    'adv_goal.csv': 'APER (Ours)',
    'adv_reward.csv': 'APER (Ours)',
    'cer_goal.csv': 'CER',
    'cer_reward.csv': 'CER',
    'PER_goal.csv': 'PER',
    'PER_reward.csv': 'PER',
    'va_goal.csv': 'Vanilla ER',
    'va_reward.csv': 'Vanilla ER',
    "run-runs_SimpleVSS-v0_24-tag-avg_sample_index_delta.csv":'APER (Ours)',
    "run-runs_SimpleVSS-v0_25-tag-avg_sample_index_delta.csv":'Vanilla ER',
    "run-runs_SimpleVSS-v0_26-tag-avg_sample_index_delta.csv":'PER',
    "run-runs_SimpleVSS-v0_27-tag-avg_sample_index_delta.csv":'CER',
}

# 读取数据
data = {}
for file in os.listdir('plot/formal'):
    if file.endswith('.csv'):
        df = pd.read_csv(os.path.join('plot/formal', file))
        df = df[:40000]
        cols = list(df.columns)
        for i in range(350000,45000):
            cols.append(cols[i-10000])
        df = df[cols]
        alg = file_to_alg[file]
        if 'reward' in file:
            metric = "reward"
        elif 'index' in file:
            metric = "index"
        else:
            metric = 'goal'
        if alg not in data:
            data[alg] = {}
        data[alg][metric] = df

# 绘制 reward 图
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
palette = sns.color_palette("deep", len(data))
for i, alg in enumerate(data):
    df = data[alg]['reward']
    df['Smoothed'] = df['Value'].rolling(window=1500, min_periods=1).mean()
    df['Std'] = df['Value'].rolling(window=1500, min_periods=1).std()/10
    plt.plot(df['Step'][100:], df['Smoothed'][100:], label=alg, color=palette[i])
    plt.fill_between(df['Step'][100:], df['Smoothed'][100:]-df['Std'][100:], df['Smoothed'][100:]+df['Std'][100:], alpha=0.1, color=palette[i])
plt.xlabel('Episode')
plt.ylabel('Average Episode Reward')
plt.title('Average Episode Reward Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plot/reward.jpg')

plt.clf()
# 绘制 reward 图
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
palette = sns.color_palette("deep", len(data))
for i, alg in enumerate(data):
    df = data[alg]['goal']
    df['Smoothed'] = df['Value'].rolling(window=1500, min_periods=1).mean()
    df['Std'] = df['Value'].rolling(window=1500, min_periods=1).std()/10
    plt.plot(df['Step'][100:], df['Smoothed'][100:], label=alg, color=palette[i])
    plt.fill_between(df['Step'][100:], df['Smoothed'][100:]-df['Std'][100:], df['Smoothed'][100:]+df['Std'][100:], alpha=0.1, color=palette[i])
plt.xlabel('Episode')
plt.ylabel('Goal Scored')
plt.title('Goal Scored Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('plot/goal.jpg')

plt.clf()
# 绘制 index 图
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")
palette = sns.color_palette("deep", len(data))
plt.xlim(0, 10000)
plt.ylim(10000, 60000)



for i, alg in enumerate(data):
    df = data[alg]['reward']
    m = 77.5
    print(alg, m)
    for j in range(6000,len(df)):
        if df["Value"].iloc[j-100:j+100].mean()>m:
            print(j)
            break

    # plt.fill_between(df['Step'][100:], df['Smoothed'][100:]-df['Std'][100:], df['Smoothed'][100:]+df['Std'][100:], alpha=0.1, color=palette[i])
plt.xlabel('Episode')
plt.ylabel('Step difference')
plt.title('The step difference between a transaction is add and sampled')
plt.legend()
plt.tight_layout()
plt.savefig('plot/index.jpg')
print("done")