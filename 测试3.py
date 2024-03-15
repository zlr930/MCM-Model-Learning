import matplotlib.pyplot as plt
import numpy as np
import random

population_size = 10  # 种群大小
task_count = 10  # 搬运任务数量
agv_count = 3  # AGV数量
task_positions=[[3, 8],[ 18, 11],[20, 5], [6, 11], [8, 17], [ 20, 14], [6,  5], [ 8, 17], [ 18, 11], [ 15, 14]]
picking_stations=[[8,1],[ 8, 1],[8, 1], [13, 1], [13, 1], [13, 1],[ 18, 1],[18, 1], [18, 1], [18, 1]]
agv_start_positions=[[3,0],[3,0],[3,0]]
agv_speed=0.1
max_generations=10
Pc=0.6
Pm=0.3
initial_temperature=100
cooling_rate = 0.95

def initialize_population(population_size, task_count, agv_count):
    """
    初始化种群，每个染色体同时保存基因位和基因值。
    任务ID乱序，且尝试均匀分配AGV。

    :param population_size: 种群大小
    :param task_count: 搬运任务的数量
    :param agv_count: 物流AGV的数量
    :return: 初始化后的种群
    """
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

# 示范如何使用该函数
# population_size = 10  # 种群大小
# task_count = 10  # 搬运任务数量
# agv_count = 3  # AGV数量

population = initialize_population(population_size, task_count, agv_count)

# 打印出初始化的某个染色体以验证任务ID是乱序的且AGV分配尽量均匀
# print(population[0])
# 初始化种群并打印第一个染色体示例
# population = initialize_population(population_size, task_count, agv_count)
# m=len(population)
# for i in range(m):
#     print(f"染色体{i}：", population[i])


def get_agv_tasks_from_chromosome(chromosome, agv_count):
    # 确保字典覆盖所有可能的AGV编号
    agv_tasks = {agv_id: [] for agv_id in range(agv_count)}

    for task_id, agv_id in chromosome:
        # 检查AGV编号是否在预期范围内
        if agv_id in agv_tasks:
            agv_tasks[agv_id].append(task_id)
        else:
            # 如果AGV编号不在预期范围内，打印错误信息
            print(f"错误: 染色体中存在无效的AGV编号 {agv_id}")

    # 按AGV编号顺序提取任务列表
    all_agv_tasks = [agv_tasks[agv_id] for agv_id in sorted(agv_tasks)]

    return all_agv_tasks

# chromosome_data = np.array([[9, 2],[8, 1],[7, 0],[6, 2],[5, 1],[4, 1],[3, 1],[2, 2],[1, 0],[0, 0]])
#
# # 获取并打印每一辆AGV的搬运任务
# task_assignments = get_agv_tasks_from_chromosome(chromosome_data)
# print(task_assignments)

def manhattan_distance(point1, point2):
    """计算两点之间的曼哈顿距离。"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

def calculate_task_time(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed):
    """
    根据提供的逻辑计算每辆AGV完成其分配任务的总时间。

    :param task_assignments: 每辆AGV分配的任务列表的列表。
    :param task_positions: 每个任务的坐标位置列表。
    :param picking_stations: 每个任务对应的拣选台坐标位置列表。
    :param agv_start_positions: 每辆AGV的起始坐标位置列表。
    :param agv_speed: AGV的速度。
    :return: 每辆AGV完成其分配任务的总时间列表。
    """

    agv_total_times = []  # 存储每辆AGV的总时间
    print(task_assignments)

    # 遍历每辆AGV及其任务分配
    for agv_index, assignments in enumerate(task_assignments):
        total_time = 0
        current_position = agv_start_positions[agv_index]  # AGV的当前位置初始化为起始位置
        # print(assignments)

        # 遍历AGV的任务分配
        for i, task_id in enumerate(assignments):
            # print("任务ID:", task_id)
            task_position = task_positions[task_id]
            picking_station = picking_stations[task_id]

            # 计算完成当前任务的时间
            if i == 0:
                # 第一个任务的特殊处理
                distance = manhattan_distance(current_position, task_position) + manhattan_distance(task_position,
                                                                                                    picking_station)
                total_time += distance / agv_speed
            else:
                previous_picking_station = picking_stations[assignments[i - 1]]
                if picking_station == previous_picking_station:
                    # 服务于同一个拣选台，完成时间为0
                    continue
                else:
                    if task_position == task_positions[assignments[i - 1]]:
                        # 同一个任务点，不同拣选台
                        distance = manhattan_distance(previous_picking_station, picking_station)
                    else:
                        # 不同任务点
                        distance = manhattan_distance(current_position, task_position) + manhattan_distance(
                            task_position, picking_station)
                    total_time += distance / agv_speed

            current_position = picking_station  # 更新当前位置为拣选台位置

        # 计算最后一个拣选台返回到起始位置的时间
        if assignments:
            last_task_position = task_positions[assignments[-1]]
            distance_back_to_start = manhattan_distance(current_position, last_task_position) + manhattan_distance(
                last_task_position, agv_start_positions[agv_index])
            total_time += distance_back_to_start / agv_speed

        agv_total_times.append(total_time)

    return agv_total_times

# agv_total_times=calculate_task_time(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed)
# print(agv_total_times)

# def adaptive_crossover_mutation_rates(k, K, m):
#     """计算自适应交叉率和变异率"""
#     Pc = 1 / (1 + m * np.exp(k / K))
#     Pm = 1 / (1 - m * np.exp(k / K))
#     return Pc, Pm


def dynamic_crossover(parent1, parent2):
    # 确定染色体长度
    size = parent1.shape[0]
    # 初始化子代染色体
    child1, child2 = np.empty_like(parent1), np.empty_like(parent2)

    # 随机选择多个交叉点
    crossover_points = sorted(random.sample(range(size), 3))

    # 根据交叉点分段，并交替选择段落
    segments_to_take_from_parent1 = [True, False] * (3 // 2)
    for segment_index, take_from_parent1 in enumerate(segments_to_take_from_parent1):
        start = crossover_points[segment_index]
        end = crossover_points[segment_index + 1] if segment_index + 1 < len(crossover_points) else size
        if take_from_parent1:
            child1[start:end] = parent1[start:end]
            child2[start:end] = parent2[start:end]
        else:
            child1[start:end] = parent2[start:end]
            child2[start:end] = parent1[start:end]

    # 对未被选中的任务执行重新分配逻辑
    reassign_tasks(child1,parent2)
    reassign_tasks(child2,parent1)

    return child1, child2

def perform_crossover(population, Pc):
    new_population = []
    for i in range(0, len(population), 2):
        if i+1 < len(population):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = dynamic_crossover(parent1, parent2)
            # 只添加子代到新种群如果交叉率被满足
            if random.random() < Pc:
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])
        else:
            # 如果种群大小是奇数，直接添加最后一个染色体
            new_population.append(population[i])
    return new_population

def reassign_tasks(child, parent):
    # 找出child中所有已分配的任务ID
    assigned_task_ids = set(child[:, 0])

    # 找出child中缺失的任务ID（在parent中存在，但在child中不存在的）
    missing_task_ids = [task[0] for task in parent if task[0] not in assigned_task_ids]

    # 统计每个AGV在child中的任务数量
    agv_task_count = {agv_id: np.sum(child[:, 1] == agv_id) for agv_id in set(parent[:, 1])}

    # 对于每个缺失的任务ID，找到当前任务最少的AGV，并分配给它
    for task_id in missing_task_ids:
        # 找到当前任务最少的AGV
        min_tasks_agv = min(agv_task_count, key=agv_task_count.get)
        # 分配任务给该AGV，并更新任务计数
        agv_task_count[min_tasks_agv] += 1
        # 找到分配给此AGV的下一个任务位置，这里用-1标记未分配任务的位置
        for i in range(len(child)):
            if child[i, 0] == -1:  # 未分配任务的标记
                child[i] = [task_id, min_tasks_agv]
                break

    return child  # 明确返回更新后的子代染色体

# k = 1  # 当前代数
# K = 100  # 最大代数
# # 初始化种群
# population = initialize_population(population_size, task_count, agv_count)
# m=len(population)
# for i in range(m):
#     print(f"初始化染色体{i}：", population[i])
#
# # 计算交叉率
# Pc, _ = adaptive_crossover_mutation_rates(k, K, m)
#
# # 执行种群的交叉操作
# new_population = perform_crossover(population, Pc)
#
# # 打印新种群的染色体，验证交叉操作
# for i, chromosome in enumerate(new_population):
#     print(f"Chromosome {i+1}:\n{chromosome}\n")

def mutate_chromosome_with_shift(chromosome, agv_count):
    num_genes = len(chromosome)
    # 随机选择插入位和基因位
    gene_pos = np.random.randint(num_genes)
    insert_pos = np.random.randint(num_genes)

    # 提取要移动的基因，并为其选择新的AGVID（基于AGV的总数，不包括当前AGVID）
    gene_to_move = chromosome[gene_pos].copy()
    new_agvid = np.random.choice([x for x in range(agv_count) if x != gene_to_move[1]])
    gene_to_move[1] = new_agvid

    # 将基因位的基因移动到插入位
    if gene_pos < insert_pos:
        # 如果基因位在插入位之前，先删除原基因位，再在新位置插入基因
        chromosome = np.delete(chromosome, gene_pos, axis=0)
        chromosome = np.insert(chromosome, insert_pos, gene_to_move, axis=0)
    else:
        # 如果基因位在插入位之后或等于插入位，先在新位置插入基因，再删除原基因位
        chromosome = np.insert(chromosome, insert_pos, gene_to_move, axis=0)
        chromosome = np.delete(chromosome, gene_pos + 1, axis=0)

    return chromosome


def mutate_population_with_shift(population, agv_count, Pm):
    """
    对整个种群执行变异操作。

    :param population: 种群，每个染色体包含任务ID和对应的AGV ID。
    :param agv_count: AGV的数量，用于生成新的AGV ID。
    :param Pm: 变异概率 Pm。
    :return: 经过变异操作的种群。
    """
    for i in range(len(population)):
        if np.random.rand() < Pm:  # 根据变异概率决定是否进行变异
            population[i] = mutate_chromosome_with_shift(population[i],agv_count)
    return population

# k = 1 # 当前代数
# K = 100  # 最大代数
# # 初始化种群
# population = initialize_population(population_size, task_count, agv_count)
# m=len(population)
# for i in range(m):
#     print(f"初始化染色体{i}：", population[i])
#
# # # 计算交叉率
# # Pc,Pm = adaptive_crossover_mutation_rates(k, K, m)
# # print(Pm)
# # print(Pc)
#
# # 执行种群的交叉操作
# new_population = mutate_population_with_shift(population, agv_count, Pm)
#
# # 打印新种群的染色体，验证交叉操作
# for i, chromosome in enumerate(new_population):
#     print(f"变异后Chromosome {i+1}:\n{chromosome}\n")
def fitness_function(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed):
    # 计算每辆AGV完成其分配任务的总时间
    agv_total_times = calculate_task_time(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed)

    # 找到最长的任务完成时间
    max_time = max(agv_total_times)

    # 适应度值为最长完成时间（目的是最小化最长完成时间）
    fitness = max_time

    return fitness
# fitness=fitness_function(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed)
# print(fitness)


def genetic_algorithm_with_sa(population_size, task_count, agv_count, task_positions, picking_stations,
                              agv_start_positions, agv_speed, max_generations, Pm, Pc, initial_temperature,
                              cooling_rate):
    # 初始化种群
    population = initialize_population(population_size, task_count, agv_count)
    print(population)

    # 用于记录每代的最优适应度值
    best_fitness_values = []

    # 初始温度
    temperature = initial_temperature
    # 计算当前种群中每个染色体的适应度
    for generation in range(max_generations):
        # print(population)
        for chromosome in population:
            print(chromosome)
            fitness_values=[]
            task_assignments=get_agv_tasks_from_chromosome(chromosome,agv_count)
            # print("AGV分配：",task_assignments)
            fitness=fitness_function(task_assignments, task_positions, picking_stations,agv_start_positions, agv_speed)
            fitness_values.append(fitness)


        # 记录当前代的最佳适应度值
        current_best_fitness = min(fitness_values)
        best_fitness_values.append(current_best_fitness)

        # 生成新的种群
        new_population = perform_crossover(population, Pc)
        new_population = mutate_population_with_shift(new_population, agv_count, Pm)

        # 计算新种群中每个染色体的适应度并应用模拟退火逻辑
        new_fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome,agv_count), task_positions, picking_stations,
                             agv_start_positions, agv_speed) for chromosome in new_population]

        for i in range(len(population)):
            # 模拟退火接受条件
            if new_fitness_values[i] < fitness_values[i] or np.random.rand() < np.exp((fitness_values[i] - new_fitness_values[i]) / temperature):
                population[i] = new_population[i]

        # 冷却过程
        temperature *= cooling_rate

        # 可选：在此处输出当前代和其最佳适应度值等信息

    # 获取最终的最优解
    final_fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome,agv_count), task_positions, picking_stations,
                         agv_start_positions, agv_speed) for chromosome in population]
    best_index = np.argmin(final_fitness_values)
    best_chromosome = population[best_index]

    # 返回最优解、最优适应度值以及每代的最优适应度值列表
    return best_chromosome, final_fitness_values[best_index], best_fitness_values

# 执行遗传算法结合模拟退火模型
best_solution, best_fitness, best_fitness_values = genetic_algorithm_with_sa(population_size, task_count, agv_count,
                                                                             task_positions, picking_stations,
                                                                             agv_start_positions, agv_speed,
                                                                             max_generations, Pm, Pc,
                                                                             initial_temperature, cooling_rate)

# 输出结果
print("最优AGV任务分配结果:", best_solution)
print("最优适应度值:", best_fitness)
print("每次迭代后的最优适应度值:", best_fitness_values)
