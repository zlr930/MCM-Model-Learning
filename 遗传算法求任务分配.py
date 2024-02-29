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
        # 生成包含所有任务的序列，并随机打乱
        all_tasks = list(range(TASK_COUNT))
        random.shuffle(all_tasks)

        # 均匀地分配任务给每个AGV
        chromosome = []
        tasks_per_agv = TASK_COUNT // AGV_COUNT  # 假设TASK_COUNT能被AGV_COUNT整除
        for i in range(AGV_COUNT):
            start_index = i * tasks_per_agv
            end_index = (i + 1) * tasks_per_agv
            chromosome.append(all_tasks[start_index:end_index])
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
        for individual in population:
            fitness = fitness_function(individual)
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual
    # 输出每辆AGV的任务分配情况
    print("每辆AGV的任务分配情况：")
    print(best_solution)
    for agv_index, tasks_assigned in enumerate(best_solution):
        print(f"AGV {agv_index + 1}: 任务 {tasks_assigned}")
    return best_solution


best_solution=genetic_algorithm()

# 绘图
# plt.figure(figsize=(8, 6))
# plt.scatter(tasks[:, 0], tasks[:, 1], color='blue', label='任务位置')
# plt.scatter(AGV_DOCK_POSITION[0], AGV_DOCK_POSITION[1], color='red', label='AGV初始位置')
# plt.xlabel('X 坐标')
# plt.ylabel('Y 坐标')
# plt.title('任务位置和AGV初始位置')
# plt.legend()
# plt.grid(True)
# plt.show()


# 假设已经有了最佳解，这里直接使用一个示例
best_solution = genetic_algorithm()

# 绘图
plt.figure(figsize=(10, 8))
plt.scatter(tasks[:, 0], tasks[:, 1], color='blue', label='任务位置')
plt.scatter(AGV_DOCK_POSITION[0], AGV_DOCK_POSITION[1], color='red', label='AGV初始位置')

colors = ['green', 'cyan', 'magenta', 'yellow', 'black']

for agv_index, task_indices in enumerate(best_solution):
    agv_path = [AGV_DOCK_POSITION] + [tasks[ti] for ti in task_indices]
    agv_path = np.array(agv_path)
    plt.plot(agv_path[:, 0], agv_path[:, 1], color=colors[agv_index], marker='o', label=f'AGV {agv_index+1} 路径')

plt.xlabel('X 坐标')
plt.ylabel('Y 坐标')
plt.title('AGV任务分配与路径')
plt.legend()
plt.grid(True)
plt.show()