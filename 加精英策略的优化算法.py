import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

population_size = 100  # 种群大小
task_count = 80  # 搬运任务数量
agv_count = 5  # AGV数量
task_positions = [[112,9],[96,28],[57,16],[19,42],[93,36],[63,42],[64,4],[89,15],[93,42],[40,36],[102,10],[57,22],[107,28],[27,48],[63,42],[90,48],[27,41],[56,48],[90,9],[107,28],[63,32],[79,41],[89,15],[47,43],[103,3],[78,36],[95,21],[76,48],[63,36],[83,3],[57,16],[48,22],[103,22],[83,42],[46,28],[46,48],[85,22],[89,32],[93,42],[48,10],[13,48],[49,3],[41,16],[111,21],[54,9],[50,36],[102,36],[90,9],[66,15],[96,28],[30,42],[65,28],[67,48],[109,16],[57,22],[42,4],[39,28],[27,41],[8,42],[98,16],[48,10],[100,48],[77,29],[100,4],[63,42],[65,10],[105,16],[85,10],[13,42],[57,28],[114,4],[48,16],[95,21],[57,28],[26,48],[90,4],[68,22],[46,28],[104,42],[99,10]]
agv_start_positions = [[3, 1], [5, 1], [7, 1], [9, 1], [11, 1]]
picking_stations = [[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27,21],[27,21],[27,21],[27,21],[27,21],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9]]
n = 6  # AGV一次最多运送的任务数
v = 0.1  # AGV的平均速度
max_generations=10000
Pm=0.1  #变异系数
Pc=0.6  #交叉系数
initial_temperature=1000
cooling_rate=0.95
elite_count=1


def filter_duplicate_tasks_with_picking_stations(task_positions, picking_stations):
    """
    删除task_positions中的重复任务位置，并同步删除picking_stations中对应的数据。

    :param task_positions: 包含各个任务的位置的列表。
    :param picking_stations: 与task_positions维度相同的拣货站列表。
    :return: 一个元组，包含去重后的任务位置列表、去重后的拣货站列表及任务数量。
    """
    if len(task_positions) != len(picking_stations):
        raise ValueError("task_positions 和 picking_stations 列表的长度必须相同。")

    unique_task_positions = []
    filtered_picking_stations = []

    for i, item in enumerate(task_positions):
        if item not in unique_task_positions:
            unique_task_positions.append(item)
            try:
                filtered_picking_stations.append(picking_stations[i])
            except IndexError:
                raise IndexError(f"尝试访问picking_stations的索引{i}超出范围。")

    task_count = len(unique_task_positions)

    return unique_task_positions, filtered_picking_stations, task_count


def initialize_population(population_size, task_count, agv_count):

    population = []
    for _ in range(population_size):
        # 创建乱序的任务ID列表
        task_ids = np.random.permutation(task_count)
        # 初始化染色体
        chromosome = np.zeros((task_count, 2), dtype=int)
        # 为了均匀分配AGV，创建一个AGV编号的列表，并随机打乱
        agv_ids = np.tile(np.arange(agv_count), (task_count + agv_count - 1) // agv_count)[:task_count]
        np.random.shuffle(agv_ids)
        for i, task_id in enumerate(task_ids):
            # 使用乱序的任务ID填充染色体的第一列
            chromosome[i, 0] = task_id
            # 第二列保存分配给该任务的AGV编号（基因值），尝试均匀分配
            chromosome[i, 1] = agv_ids[i]
        population.append(chromosome)
    return population

def get_agv_tasks_from_chromosome(chromosome, agv_count):
    # 初始化一个字典来保存每个AGV的任务
    agv_tasks = {i: [] for i in range(agv_count)}

    # 假设`chromosome`中的每个元素结构为 [task_id, agv_id]
    for gene in chromosome:
        task_id, agv_id = gene  # 将基因解包为task_id和agv_id
        agv_tasks[agv_id].append(task_id)

    # 将任务字典转换为每个AGV的任务列表
    tasks_list = [agv_tasks[i] for i in range(agv_count)]

    return tasks_list
def manhattan_distance(point1, point2):
    """计算两点之间的曼哈顿距离。"""
    return (abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]))*0.5


def calculate_agv_time(genes, unique_task_positions, agv_start_positions, filtered_picking_stations, n, v):
    agv_times = []  # 存储每辆AGV完成任务的总时间

    # 遍历每辆AGV及其分配的任务
    for agv_id, tasks in enumerate(genes):
        agv_time = 0  # 初始化当前AGV的时间
        print(agv_id)
        current_position = agv_start_positions[agv_id]  # AGV的起始位置

        # 分批处理任务
        for i in range(0, len(tasks), n):
            batch = tasks[i:i+n]  # 当前批次的任务

            # 遍历批次中的任务
            for task_id in batch:
                # 移动到任务点，并更新时间和位置
                task_position = unique_task_positions[task_id]
                agv_time += manhattan_distance(current_position, task_position) / v
                current_position = task_position

                # 移动到对应拣货台，并更新时间和位置
                picking_station = filtered_picking_stations[task_id]
                agv_time += manhattan_distance(current_position, picking_station) / v
                current_position = picking_station

            # 如果这不是最后一批任务，返回起始位置
            if i + n < len(tasks):
                agv_time += manhattan_distance(current_position, agv_start_positions[agv_id]) / v
                current_position = agv_start_positions[agv_id]

        # 所有任务完成后返回起始位置
        agv_time += manhattan_distance(current_position, agv_start_positions[agv_id]) / v

        # 将当前AGV的总时间添加到列表中
        agv_times.append(agv_time)

    # 返回所有AGV的完成任务总时间列表
    return agv_times
def fitness_function(genes, unique_task_positions, agv_start_positions, filtered_picking_stations, n, v):
    # 计算每辆AGV完成其分配任务的总时间
    agv_total_times = calculate_agv_time(genes, unique_task_positions, agv_start_positions, filtered_picking_stations, n, v)

    # 找到最长的任务完成时间
    max_time = max(agv_total_times)

    # 适应度值为最长完成时间（目的是最小化最长完成时间）
    fitness = max_time

    return fitness
def dynamic_crossover(parent1, parent2, agv_count):
    size = parent1.shape[0]
    child1, child2 = np.full_like(parent1, -1), np.full_like(parent2, -1)

    crossover_points = sorted(random.sample(range(size), random.randint(1, size - 1)))  # 选择1到size-1个随机交叉点

    last_cp = 0
    take_from_p1 = bool(random.getrandbits(1))  # 随机选择从哪个父代开始取段
    for cp in crossover_points + [size]:
        if take_from_p1:
            child1[last_cp:cp], child2[last_cp:cp] = parent1[last_cp:cp], parent2[last_cp:cp]
        else:
            child1[last_cp:cp], child2[last_cp:cp] = parent2[last_cp:cp], parent1[last_cp:cp]
        take_from_p1 = not take_from_p1  # 下一个段换另一个父代
        last_cp = cp

    # 检查并处理子代中的重复任务ID
    for child in [child1, child2]:
        tasks, counts = np.unique(child[:, 0], return_counts=True)
        for task, count in zip(tasks, counts):
            if task != -1 and count > 1:  # 如果任务ID不是-1且出现多于一次
                indexes = np.where(child[:, 0] == task)[0]
                for i in indexes[1:]:  # 保留第一个，其余设置为-1
                    child[i, 0] = -1

    # 调用reassign_tasks函数重新分配设置为-1的任务
    reassign_tasks(child1, parent2, agv_count)
    reassign_tasks(child2, parent1, agv_count)

    return child1, child2

def reassign_tasks(child, parent, agv_count):
    # print("原始parent:", parent)
    # print("原始child:", child)
    assigned_tasks_in_child = {task for task, _ in child if task != -1}
    all_tasks_from_parent = {task for task, _ in parent}
    tasks_to_assign = all_tasks_from_parent - assigned_tasks_in_child

    # print("已分配的任务:", assigned_tasks_in_child)
    # print("需要分配的任务:", tasks_to_assign)

    for i, (task, agv) in enumerate(child):
        if task == -1 and tasks_to_assign:
            task_to_assign = tasks_to_assign.pop()
            agv_id = np.random.randint(0, agv_count)
            child[i] = [task_to_assign, agv_id]
            # print(f"分配任务 {task_to_assign} 到位置 {i}, AGV编号: {agv_id}")

    # print("更新后的child:", child)
# 注意: 这个函数不需要返回child，因为它直接修改了传入的child数组
def perform_crossover(parent1, parent2, Pc, agv_count):
    # 初始化子代染色体，它们最开始与父代相同
    child1, child2 = parent1.copy(), parent2.copy()

    # 执行交叉操作的决定
    if random.random() < Pc:
        # 执行交叉
        child1, child2 = dynamic_crossover(parent1, parent2, agv_count)

    return child1, child2


import numpy as np


def mutate_chromosome_with_shift(chromosome, agv_count, Pm):
    # 检查是否执行变异
    if np.random.rand() > Pm:
        return chromosome  # 如果不执行变异，返回原染色体

    num_genes = len(chromosome)
    # 随机选择插入位和基因位
    gene_pos = np.random.randint(num_genes)
    insert_pos = np.random.randint(num_genes)

    # 提取要移动的基因，并为其选择新的AGVID（基于AGV的总数，不包括当前AGVID）
    gene_to_move = chromosome[gene_pos].copy()
    new_agvid = np.random.choice([x for x in range(agv_count) if x != gene_to_move[1]])
    gene_to_move[1] = new_agvid

    # 创建一个新的染色体数组以包含变异
    new_chromosome = chromosome.tolist()

    # 将基因位的基因移动到插入位
    if gene_pos < insert_pos:
        # 删除原位置的基因，插入到新位置
        del new_chromosome[gene_pos]
        new_chromosome.insert(insert_pos, gene_to_move)
    else:
        # 先插入到新位置，再删除原位置的基因
        new_chromosome.insert(insert_pos, gene_to_move)
        del new_chromosome[gene_pos + 1]  # 因为插入已经发生，所以原位置向后移动了1位

    return np.array(new_chromosome)
# unique_task_positionss,task_counts=filter_duplicate_tasks(unique_task_positions)
# print(unique_task_positionss,task_counts)
def roulette_wheel_selection(fitness_values):
    #加入轮盘赌函数
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    selected_index = np.random.choice(len(fitness_values), p=selection_probs)
    return selected_index

def elitism(population, fitness_values, elite_count):
    #加入精英策略
    elite_indices = np.argsort(fitness_values)[:elite_count]
    elites = [population[index] for index in elite_indices]
    return elites


def accept_new_individual(candidate_fitness, best_fitness, temperature):
    """
    根据模拟退火的概率接受准则来决定是否接受新个体。

    :param candidate_fitness: 新候选个体的适应度。
    :param best_fitness: 当前最佳个体的适应度。
    :param temperature: 当前的模拟退火温度。
    :return: 布尔值，表示是否接受新个体。
    """
    # 如果新候选个体的适应度更好，则始终接受
    if candidate_fitness < best_fitness:
        return True
    # 如果新候选个体的适应度较差，根据模拟退火的概率接受准则决定是否接受
    else:
        delta_fitness = candidate_fitness - best_fitness
        acceptance_probability = np.exp(-delta_fitness / temperature)
        return np.random.rand() < acceptance_probability

def genetic_algorithm_SA(population_size, task_count, agv_count, unique_task_positions, filtered_picking_stations,
                         agv_start_positions, n, v, max_generations, Pm, Pc, initial_temperature, cooling_rate,
                         elite_count):
    # 初始化种群
    population = initialize_population(population_size, task_count, agv_count)
    # 计算适应度
    fitness_values = np.array([fitness_function(get_agv_tasks_from_chromosome(individual, agv_count), unique_task_positions,
                                                agv_start_positions, filtered_picking_stations, n, v) for individual in population])

    temperature = initial_temperature
    best_fitness_history = []
    # 保持对当前最佳适应度的追踪
    best_fitness = min(fitness_values)


    for generation in tqdm(range(max_generations), desc="Processing Generations"):
        # 应用精英策略保留一定数量的最优个体
        elites = elitism(population, fitness_values, elite_count)
        # 更新新种群，包括精英个体
        new_population = elites[:]

        # 使用轮盘赌方法选择父代
        parents_indices = [roulette_wheel_selection(fitness_values) for _ in range(population_size - elite_count * 2)]

        # 循环直到新种群填满
        while len(new_population) < population_size:
            # 随机选择两个父代进行交叉和变异
            idx1, idx2 = np.random.choice(parents_indices, 2, replace=False)
            parent1, parent2 = population[idx1], population[idx2]
            child1, child2 = perform_crossover(parent1, parent2, Pc, agv_count)
            print(child1)
            print(child2)
            child1 = mutate_chromosome_with_shift(child1, agv_count, Pm)
            child2 = mutate_chromosome_with_shift(child2, agv_count, Pm)
            # 利用模拟退火决定是否接受新个体
            if accept_new_individual(fitness_function(get_agv_tasks_from_chromosome(child1, agv_count), unique_task_positions,
                                                      agv_start_positions, filtered_picking_stations, n, v), best_fitness, temperature):
                new_population.append(child1)
            # 确保有足够的空间添加第二个子代
            if len(new_population) < population_size and accept_new_individual(
                    fitness_function(get_agv_tasks_from_chromosome(child2, agv_count), unique_task_positions,
                                     agv_start_positions, filtered_picking_stations, n, v), best_fitness, temperature):
                new_population.append(child2)

        # 更新种群和适应度
        population = new_population[:population_size]
        fitness_values = np.array([fitness_function(get_agv_tasks_from_chromosome(individual, agv_count),
                                                    unique_task_positions, agv_start_positions, filtered_picking_stations, n, v) for
                                   individual in population])
        # 更新最佳适应度值
        current_best_fitness = min(fitness_values)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness

        # 记录最佳适应度
        best_fitness_history.append(best_fitness)

        # 降低温度
        temperature *= cooling_rate

    # 绘制适应度历史
    plt.plot(best_fitness_history)
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Best Fitness Over Generations')
    plt.show()

    best_index = np.argmin(fitness_values)
    return population[best_index], fitness_values[best_index], best_fitness_history


# 执行函数
unique_task_positions, filtered_picking_stations, task_count=filter_duplicate_tasks_with_picking_stations(task_positions, picking_stations)

best_solution, best_fitness, best_fitness_values = genetic_algorithm_SA(population_size, task_count, agv_count, unique_task_positions, filtered_picking_stations,
                         agv_start_positions, n, v, max_generations, Pm, Pc, initial_temperature, cooling_rate,elite_count)

# 输出结果
print("最优AGV任务分配结果:", best_solution)
print("最优适应度值:", best_fitness)
print("每次迭代后的最优适应度值:", best_fitness_values)

# population = initialize_population(population_size, task_count, agv_count)
# print(population)
# for individual in population:
#     task_assment=get_agv_tasks_from_chromosome(individual,agv_count)
#     print(task_assment)
#     times=calculate_agv_time(task_assment, task_positions, agv_start_positions, picking_stations, n, v)
#     print(times)
#     fitness_value=fitness_function(task_assment, task_positions, agv_start_positions, picking_stations, n, v)
#     print(fitness_value)