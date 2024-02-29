import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用'宋体'或'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 初始化参数
TASK_COUNT = 30  # 任务数量
AGV_COUNT = 5  # AGV数量
AGV_MAX_LOAD = 3  # AGV最大负载
AGV_SPEED = 0.1  # 移动速度，单位：米/秒
AGV_DOCK_POSITION = [9, 9]  # AGV停靠位置
POPULATION_SIZE = 50  # 种群大小
MAX_ITERATIONS = 100  # 最大迭代次数
m = 0.05  # 自适应系数

# 随机生成任务的货位坐标和收益
tasks = np.random.randint(0, 20, size=(TASK_COUNT, 2))
task_profits = np.random.randint(10, 100, size=TASK_COUNT)


# 初始化种群
def initialize_population():
    population = []
    for _ in range(POPULATION_SIZE):
        # 随机分配任务到AGVs，这里简化为每个AGV分配等量任务
        chromosome = [random.sample(range(TASK_COUNT), int(TASK_COUNT / AGV_COUNT)) for _ in range(AGV_COUNT)]
        population.append(chromosome)
    return population


# 适应度函数 - 简化版本
def fitness_function(chromosome):
    # 这里仅返回一个随机适应度值，实际应用中应根据问题具体定义适应度
    return random.random()


# 遗传算法主体
def genetic_algorithm():
    population = initialize_population()
    best_solution = None
    best_fitness = -np.inf

    for _ in range(MAX_ITERATIONS):
        # 评估种群中每个个体的适应度，并选择最佳解
        for individual in population:
            fitness = fitness_function(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual

    # 输出每辆AGV的任务分配情况
    print("每辆AGV的任务分配情况：")
    for agv_index, tasks_assigned in enumerate(best_solution):
        print(f"AGV {agv_index + 1}: 任务 {tasks_assigned}")


genetic_algorithm()

# 绘图
plt.figure(figsize=(8, 6))
plt.scatter(tasks[:, 0], tasks[:, 1], color='blue', label='任务位置')
plt.scatter(AGV_DOCK_POSITION[0], AGV_DOCK_POSITION[1], color='red', label='AGV初始位置')
plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.title('任务位置和AGV初始位置')
plt.legend()
plt.grid(True)
plt.show()