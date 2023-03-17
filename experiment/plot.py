import matplotlib.pyplot as plt

with open('before.txt', 'r') as f:
    before_data = [abs(float(line.strip())) for line in f]
    before_data = before_data[-10000:]

with open('after.txt', 'r') as f:
    after_data = [abs(float(line.strip())) for line in f]
    after_data = after_data[-10000:]

# 创建散点图
plt.scatter(range(len(before_data)), before_data, label='before')
plt.legend()
plt.title('Comparison of Before and After Data')
plt.savefig("before.jpg")
plt.clf()


plt.scatter(range(len(after_data)), after_data, label='after')
plt.legend()
plt.title('Comparison of Before and After Data')
plt.savefig("after.jpg")