import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 指定支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用'宋体'或'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 初始化参数
TASK_COUNT = 30  # 任务数量
AGV_COUNT= 3  # AGV数量
AGV_MAX_LOAD = 5  # AGV最大负载
AGV_SPEED = 0.1  # 移动速度，单位：米/秒
AGV_DOCK_POSITION = [(9, 9),(9, 9),(9, 9),(9, 9),(9, 9)]  # AGV停靠位置
POPULATION_SIZE = 200  # 种群大小
MAX_ITERATIONS = 5000  # 最大迭代次数
m = 0.05  # 自适应系数
# 假设的平衡系数
a1, a2, a3 = -1, -1, 1  # 这些系数需要根据问题的具体情况进行调整

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

def calculate_distance(point1, point2):
    """计算两点之间的欧氏距离"""
    return np.linalg.norm(np.array(point1) - np.array(point2))

def fitness_function(chromosome, tasks, task_profits, agv_speed, agv_dock_position, a1, a2, a3):
    # 解包染色体结构
    task_order, agv_assignments = chromosome

    total_distance = 0
    total_profit = 0

    # 假设任务位置是通过索引从 tasks 中获取
    for task_idx, agv_idx in zip(task_order, agv_assignments):
        task_position = tasks[task_idx]  # 获取当前任务位置
        agv_position = agv_dock_position[agv_idx]  # 获取当前AGV的初始位置

        # 计算从AGV初始位置到任务位置的距离
        distance = calculate_distance(agv_position, task_position)
        total_distance += distance
        # 假设AGV在完成任务后返回到其初始位置
        total_distance += calculate_distance(task_position, agv_position)

        # 累加任务收益
        total_profit += task_profits[task_idx]

    # 计算总时间，假设速度是恒定的
    total_time = total_distance / agv_speed

    # 计算适应度值
    fitness = a1 * total_distance + a2 * total_time + a3 * total_profit
    return fitness
def adaptive_crossover_mutation_rates(k, K, m):
    Pc = 1 / (1 + m * np.exp(k / K))
    Pm = 1 / (1 - m * np.exp(k / K))
    return Pc, Pm


def single_point_crossover(parent1, parent2, Pc):
    # 确保交叉操作同时应用于任务顺序和AGV编号两个数组
    if np.random.rand() < Pc:  # 根据交叉概率决定是否进行交叉
        # 对于任务顺序数组
        crossover_point_task = np.random.randint(1, len(parent1[0]))  # 随机选择交叉点
        child1_task = np.concatenate((parent1[0][:crossover_point_task], parent2[0][crossover_point_task:]))
        child2_task = np.concatenate((parent2[0][:crossover_point_task], parent1[0][crossover_point_task:]))

        # 对于AGV编号数组
        crossover_point_agv = np.random.randint(1, len(parent1[1]))  # 随机选择交叉点
        child1_agv = np.concatenate((parent1[1][:crossover_point_agv], parent2[1][crossover_point_agv:]))
        child2_agv = np.concatenate((parent2[1][:crossover_point_agv], parent1[1][crossover_point_agv:]))

        # 组合生成子代
        child1 = (child1_task, child1_agv)
        child2 = (child2_task, child2_agv)
        return child1, child2
    else:
        # 如果不进行交叉，子代与父代相同
        return parent1, parent2

def crossover(population, Pc):
    new_population = []
    for i in range(0, len(population), 2):  # 间隔两个染色体进行遍历
        parent1 = population[i]
        parent2 = population[i+1] if i+1 < len(population) else population[0]
        child1, child2 = single_point_crossover(parent1, parent2, Pc)
        new_population.extend([child1, child2])
    return new_population
def mutate_gene(gene, max_value):
    """对单个基因执行变异操作，返回变异后的基因值"""
    # 假设基因的变异是随机分配到另一个AGV
    return random.randint(0, max_value - 1)


def mutate_task_order(task_order):
    """对任务顺序数组执行变异操作，通过随机交换两个任务的位置"""
    if len(task_order) > 1:  # 至少需要两个任务才能交换
        idx1, idx2 = np.random.choice(len(task_order), 2, replace=False)
        # 执行交换
        task_order[idx1], task_order[idx2] = task_order[idx2], task_order[idx1]
    return task_order


def mutation(chromosome, Pm, max_value):
    """对染色体执行变异操作，同时处理任务顺序和AGV编号"""
    task_order, agv_assignments = chromosome

    # 变异任务顺序
    if random.random() < Pm:  # 根据变异概率决定是否进行变异
        task_order = mutate_task_order(list(task_order))  # 确保输入为列表以便执行交换

    # 变异AGV编号
    mutated_agv_assignments = []
    for agv in agv_assignments:
        if random.random() < Pm:  # 根据变异概率决定是否进行变异
            mutated_agv = mutate_gene(agv, max_value)
            mutated_agv_assignments.append(mutated_agv)
        else:
            mutated_agv_assignments.append(agv)

    return (np.array(task_order), np.array(mutated_agv_assignments))  # 保持numpy数组格式


def apply_mutation(population, Pm, max_value):
    """对整个种群执行变异操作"""
    new_population = []
    for chromosome in population:
        # 对每个染色体执行变异
        mutated_chromosome = mutation(chromosome, Pm, max_value)
        new_population.append(mutated_chromosome)
    return new_population

def simulated_annealing_acceptance_criteria(current_fitness, new_fitness, temperature):
# 模拟退火接受差解
    if new_fitness > current_fitness:
        return True
    else:
        # 模拟退火的概率接受准则
        probability = np.exp((new_fitness - current_fitness) / temperature)
        return np.random.rand() < probability


def genetic_algorithm(tasks, task_profits, agv_count, population_size, max_iterations, m, agv_speed, agv_dock_positions, a1, a2, a3):
    # 初始化种群
    population = initialize_population(population_size, len(tasks), agv_count)
    best_solution = None
    best_fitness = -np.inf
    temperature = 100  # 模拟退火的初始温度
    cooling_rate = 0.95  # 冷却率

    for iteration in range(max_iterations):
        # 计算自适应交叉和变异率
        Pc, Pm = adaptive_crossover_mutation_rates(iteration, max_iterations, m)

        # 生成新的种群通过交叉和变异操作
        population = crossover(population, Pc)
        population = apply_mutation(population, Pm, agv_count - 1)

        # 评估新种群的适应度并选择
        new_population = []
        for chromosome in population:
            fitness = fitness_function(chromosome, tasks, task_profits, agv_speed, agv_dock_positions, a1, a2, a3)
            if fitness > best_fitness or simulated_annealing_acceptance_criteria(best_fitness, fitness, temperature):
                new_population.append(chromosome)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = chromosome

        population = new_population
        temperature *= cooling_rate  # 更新温度

        # 打印当前最佳适应度值和对应解（可选）
        print(f"Iteration {iteration}: Best Fitness = {best_fitness}")

    return best_solution


# 调用遗传算法
best_solution = genetic_algorithm(tasks, task_profits, AGV_COUNT, POPULATION_SIZE, MAX_ITERATIONS, m, AGV_SPEED,
                                  AGV_DOCK_POSITION, a1, a2, a3)
print("Best Solution:", best_solution)



