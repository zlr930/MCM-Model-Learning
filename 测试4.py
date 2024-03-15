import matplotlib.pyplot as plt
import numpy as np
import random

population_size = 10  # 种群大小
task_count = 10  # 搬运任务数量
agv_count = 3  # AGV数量
genes = [[4, 3, 1, 0], [6, 7, 5], [2, 8, 9]]
task_positions = [[3, 8], [18, 11], [20, 5], [6, 11], [8, 17], [20, 14], [6, 5], [8, 17], [18, 11], [15, 14]]
agv_start_positions = [[3, 0], [3, 0], [3, 0]]
picking_stations = [[8, 1], [8, 1], [8, 1], [13, 1], [13, 1], [13, 1], [18, 1], [18, 1], [18, 1], [18, 1]]
n = 2  # AGV一次最多运送的任务数
v = 1  # AGV的平均速度
max_generations=500
Pm=0.1  #变异系数
Pc=0.6  #交叉系数
initial_temperature=1000
cooling_rate=0.95

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

def manhattan_distance(point1, point2):
    """计算两点之间的曼哈顿距离。"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])


def calculate_agv_time(genes, task_positions, agv_start_positions, picking_stations, n, v):
    agvs_times = []  # 初始化存储每辆AGV完成任务总时间的列表

    # 遍历每辆AGV及其任务序列
    for agv_id, tasks in enumerate(genes):
        start_position = agv_start_positions[agv_id]  # 当前AGV的起始位置
        current_position = start_position
        agv_time = 0  # 初始化当前AGV完成任务的总时间

        # 循环直到该AGV的所有任务都被处理
        while tasks:
            tasks_this_round = tasks[:n]  # 本轮处理的任务，最多n个
            tasks = tasks[n:]  # 更新剩余的任务列表

            # 遍历本轮的任务
            for task_index, task in enumerate(tasks_this_round):
                # 任务点到下一个任务点的距离
                task_position = task_positions[task]
                agv_time += manhattan_distance(current_position, task_position) / v
                current_position = task_position

            print("任务列表:", tasks_this_round)
            print("拣货台列表长度:", len(picking_stations))
            print(picking_stations)
            # 确保所有的task都在picking_stations的索引范围内
            for task in tasks_this_round:
                if task >= len(picking_stations) or task < 0:
                    print(f"无效的任务ID: {task}")


            # 从当前位置到本轮所有任务对应拣货台的距离，选择最短距离的拣货台先去拣货
            picking_stations_this_round = [picking_stations[task] for task in tasks_this_round]
            print(picking_stations_this_round)
            while picking_stations_this_round:
                # 计算到所有拣货台的距离并排序
                distances_to_picking_stations = [(manhattan_distance(current_position, station), station) for station in
                                                 picking_stations_this_round]
                distances_to_picking_stations.sort(key=lambda x: x[0])

                # 前往距离最近的拣货台
                nearest_picking_station = distances_to_picking_stations[0][1]
                agv_time += distances_to_picking_stations[0][0] / v
                current_position = nearest_picking_station

                # 从待访问列表中移除已访问的拣货台
                picking_stations_this_round.remove(nearest_picking_station)

            # 任务点返回逻辑
            for task in reversed(tasks_this_round):
                task_position = task_positions[task]
                agv_time += manhattan_distance(current_position, task_position) / v
                current_position = task_position

            # 如果任务已全部处理，计算返回起始点的时间
            if not tasks:
                agv_time += manhattan_distance(current_position, start_position) / v

        # 每辆AGV完成任务的总时间加入到列表中
        agvs_times.append(agv_time)

    return agvs_times
def fitness_function(genes, task_positions, agv_start_positions, picking_stations, n, v):
    # 计算每辆AGV完成其分配任务的总时间
    agv_total_times = calculate_agv_time(genes, task_positions, agv_start_positions, picking_stations, n, v)

    # 找到最长的任务完成时间
    max_time = max(agv_total_times)

    # 适应度值为最长完成时间（目的是最小化最长完成时间）
    fitness = max_time

    return fitness
def dynamic_crossover(parent1, parent2):
    # 确定染色体长度
    size = parent1.shape[0]

    # 初始化子代染色体，并使用-1填充
    child1, child2 = np.empty_like(parent1), np.empty_like(parent2)
    child1.fill(-1)  # 使用-1初始化child1
    child2.fill(-1)  # 使用-1初始化child2

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
    # 确保reassign_tasks函数能正确处理-1作为未分配标记
    reassign_tasks(child1, parent2)
    reassign_tasks(child2, parent1)

    return child1, child2

def reassign_tasks(child, parent):
    # 找出child中所有已分配的任务ID
    assigned_task_ids = set(task_id for task_id in child[:, 0] if task_id != -1)

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
        # 找到分配给此AGV的下一个任务位置
        for i in range(len(child)):
            if child[i, 0] == -1:  # 未分配任务的标记
                child[i] = [task_id, min_tasks_agv]
                break

    # 确保没有任务遗漏，所有位置都被正确填充
    return child
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

    for i in range(len(population)):
        if np.random.rand() < Pm:  # 根据变异概率决定是否进行变异
            population[i] = mutate_chromosome_with_shift(population[i],agv_count)
    return population

def genetic_algorithm_with_sa(population_size, task_count, agv_count, task_positions, picking_stations,
                              agv_start_positions, v, max_generations, Pm, Pc, initial_temperature,
                              cooling_rate):
    # 初始化种群
    population = initialize_population(population_size, task_count, agv_count)
    # 用于记录每代的最优适应度值
    best_fitness_values = []
    # 初始温度
    temperature = initial_temperature
    for generation in range(max_generations):
        for chromosome in population:
            fitness_values = []
            task_assignments = get_agv_tasks_from_chromosome(chromosome, agv_count)
            # print(task_assignments)
            # agvs_times=calculate_agv_time(task_assignments, task_positions, agv_start_positions, picking_stations, n, v)
            # print(f"AGV 完成所有任务的总时间为：{agvs_times} ")
            fitness_value=fitness_function(task_assignments, task_positions, agv_start_positions, picking_stations, n, v)
            # print("适应度值为："+str(fitness_value))
            fitness_values.append(fitness_value)
        # 记录当前代的最佳适应度值
        current_best_fitness = min(fitness_values)
        best_fitness_values.append(current_best_fitness)

        # 生成新的种群
        new_population = perform_crossover(population, Pc)
        new_population = mutate_population_with_shift(new_population, agv_count, Pm)
        # print(new_population)

        new_fitness_values = [fitness_function(task_assignments, task_positions, agv_start_positions, picking_stations, n, v) for chromosome in new_population]
        for i in range(len(population)):
            if i < len(new_fitness_values) and i < len(fitness_values):
                if new_fitness_values[i] < fitness_values[i] or np.random.rand() < np.exp(
                        (fitness_values[i] - new_fitness_values[i]) / temperature):
                    population[i] = new_population[i]
                    fitness_values[i] = new_fitness_values[i]  # 更新适应度值

        # 冷却过程
        temperature *= cooling_rate

        # 可选：在此处输出当前代和其最佳适应度值等信息

    # 获取最终的最优解
    final_fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome, agv_count), task_positions, agv_start_positions, picking_stations, n, v) for chromosome in population]
    best_index = np.argmin(final_fitness_values)
    best_chromosome = population[best_index]

    # 返回最优解、最优适应度值以及每代的最优适应度值列表
    return best_chromosome, final_fitness_values[best_index], best_fitness_values

# 执行遗传算法结合模拟退火模型
best_solution, best_fitness, best_fitness_values = genetic_algorithm_with_sa(population_size, task_count, agv_count,
                                                                             task_positions, picking_stations,
                                                                             agv_start_positions, v,
                                                                             max_generations, Pm, Pc,
                                                                             initial_temperature, cooling_rate)

# 输出结果
print("最优AGV任务分配结果:", best_solution)
print("最优适应度值:", best_fitness)
print("每次迭代后的最优适应度值:", best_fitness_values)






