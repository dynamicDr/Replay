import psutil

process = psutil.Process()
memory_info = process.memory_info()
print(memory_info.rss)