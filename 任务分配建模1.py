import matplotlib.pyplot as plt
import numpy as np
import random

population_size = 100  # 种群大小
task_count = 60  # 搬运任务数量
agv_count = 5  # AGV数量
task_positions = [[3,5],[5,11],[4,17],[6,20],[24,2],[15,8],[25,8],[22,11],[13,17],[25,20],[6,5],[26,14],[18,11],[22,8],[5,11],[11,14],[15,5],[5,8],[9,2],[11,14],[24,5],[22,11],[3,5],[19,17],[28,5],[18,8],[1,11],[28,5],[21,14],[1,14],[9,5],[26,11],[19,17],[18,8],[14,11],[7,11],[16,14],[2,2],[22,17],[5,14],[19,5],[16,17],[14,14],[11,14],[14,11],[8,17],[7,8],[1,11],[7,8],[13,5],[11,20],[17,20],[28,17],[7,8],[11,11],[17,2],[28,14],[28,8],[21,14],[3,20]]
agv_start_positions = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
picking_stations = [[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1]]
n = 6  # AGV一次最多运送的任务数
v = 0.1  # AGV的平均速度
max_generations=5000
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

            # print("任务列表:", tasks_this_round)
            # print("拣货台列表长度:", len(picking_stations))
            # print(picking_stations)
            # 确保所有的task都在picking_stations的索引范围内
            for task in tasks_this_round:
                if task >= len(picking_stations) or task < 0:
                    print(f"无效的任务ID: {task}")


            # 从当前位置到本轮所有任务对应拣货台的距离，选择最短距离的拣货台先去拣货
            picking_stations_this_round = [picking_stations[task] for task in tasks_this_round]
            # print(picking_stations_this_round)
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


def perform_crossover(population, Pc, agv_count):
    new_population = []
    for i in range(0, len(population), 2):
        if i+1 < len(population):
            parent1, parent2 = population[i], population[i+1]
            child1, child2 = dynamic_crossover(parent1, parent2, agv_count)
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
    population = initialize_population(population_size, task_count, agv_count)
    best_fitness_values = []
    temperature = initial_temperature
    overall_best_chromosome = None
    overall_best_fitness = float('inf')

    for generation in range(max_generations):
        fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome, agv_count), task_positions,
                                           agv_start_positions, picking_stations, n, v) for chromosome in population]

        current_best_fitness = min(fitness_values)
        current_best_index = fitness_values.index(current_best_fitness)
        if current_best_fitness < overall_best_fitness:
            overall_best_fitness = current_best_fitness
            overall_best_chromosome = population[current_best_index]

        best_fitness_values.append(current_best_fitness)

        new_population = perform_crossover(population, Pc, agv_count)
        new_population = mutate_population_with_shift(new_population, agv_count, Pm)

        new_fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome, agv_count), task_positions,
                                               agv_start_positions, picking_stations, n, v) for chromosome in new_population]

        for i in range(len(population)):
            if new_fitness_values[i] < fitness_values[i] or np.random.rand() < np.exp(
                    (fitness_values[i] - new_fitness_values[i]) / temperature):
                population[i] = new_population[i]
                fitness_values[i] = new_fitness_values[i]

        temperature *= cooling_rate
    print(best_fitness_values)
    print(overall_best_chromosome)
    print(overall_best_fitness)

    # 绘图
    plt.plot(best_fitness_values)
    plt.title('Best Fitness Value over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness Value')
    plt.show()

    # 返回最优AGV任务分配结果、最优适应度值以及每次迭代后的最优适应度值
    return overall_best_chromosome, overall_best_fitness, best_fitness_values


# 执行函数
best_solution, best_fitness, best_fitness_values = genetic_algorithm_with_sa(population_size, task_count, agv_count, task_positions, picking_stations,agv_start_positions, v, max_generations, Pm, Pc, initial_temperature, cooling_rate)

# 输出结果
print("最优AGV任务分配结果:", best_solution)
print("最优适应度值:", best_fitness)
print("每次迭代后的最优适应度值:", best_fitness_values)