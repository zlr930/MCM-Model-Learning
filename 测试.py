import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用'宋体'或'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 初始化参数
TASK_COUNT = 20  # 任务数量
AGV_COUNT = 5  # AGV数量
AGV_MAX_LOAD = 5  # AGV最大负载
AGV_SPEED = 0.1  # 移动速度，单位：米/秒
AGV_DOCK_POSITION = [9, 9]  # AGV停靠位置
POPULATION_SIZE = 50  # 种群大小
MAX_ITERATIONS = 100  # 最大迭代次数
m = 0.05  # 自适应系数
# 假设的平衡系数
a1, a2, a3 = 1, 1, 1  # 这些系数需要根据问题的具体情况进行调整

# 随机生成任务的货位坐标和收益
tasks = np.random.randint(0, 20, size=(TASK_COUNT, 2))
task_profits = np.random.randint(10, 100, size=TASK_COUNT)
# print(tasks)

# 绘图初始任务坐标
# plt.figure(figsize=(8, 6))
# plt.scatter(tasks[:, 0], tasks[:, 1], color='blue', label='任务位置')
# plt.scatter(AGV_DOCK_POSITION[0], AGV_DOCK_POSITION[1], color='red', label='AGV初始位置')
# plt.xlabel('X 坐标')
# plt.ylabel('Y 坐标')
# plt.title('任务位置和AGV初始位置')
# plt.legend()
# plt.grid(True)
# plt.show()

# 初始化种群，采用双层染色体编码
def initialize_population(population_size, task_count, agv_count):
    population = []
    for _ in range(population_size):
        # 第一层编码：任务的排列
        task_encoding = np.random.permutation(task_count)
        # 第二层编码：每个任务对应的AGV编号
        agv_encoding = np.random.randint(0, agv_count, size=task_count)
        population.append((task_encoding, agv_encoding))
    return population

population = initialize_population(POPULATION_SIZE, TASK_COUNT, AGV_COUNT)
print(population)
