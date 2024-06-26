import matplotlib.pyplot as plt
import matplotlib

# 设置字体为Songti SC，字体大小为12，字体权重为轻
#matplotlib.rcParams['font.family'] = 'Heiti TC'
matplotlib.rcParams['font.family'] = 'Microsoft YaHei'
#matplotlib.rcParams['font.size'] = 15
matplotlib.rcParams['font.weight'] = 'light'  # 尝试使用 'normal' 如果 'light' 效果不理想
matplotlib.rcParams['axes.unicode_minus'] = False  # 确保能够显示负号
import numpy as np
import random
from tqdm import tqdm

population_size = 100  # 种群大小
task_count = 80  # 搬运任务数量
agv_count = 5  # AGV数量
task_positions = [[112,9],[96,28],[57,16],[19,42],[93,36],[63,42],[64,4],[89,15],[93,42],[40,36],[102,10],[57,22],[107,28],[27,48],[63,42],[90,48],[27,41],[56,48],[90,9],[107,28],[63,32],[79,41],[89,15],[47,43],[103,3],[78,36],[95,21],[76,48],[63,36],[83,3],[63,16],[48,22],[103,22],[83,42],[46,28],[46,48],[85,22],[89,32],[93,42],[48,10],[13,48],[49,3],[41,16],[111,21],[54,9],[50,36],[102,36],[98,9],[66,15],[96,28],[30,42],[65,28],[67,48],[109,16],[57,22],[42,4],[39,28],[27,41],[8,42],[98,16],[48,10],[100,48],[77,29],[100,4],[63,42],[65,10],[105,16],[85,10],[13,42],[57,28],[114,4],[48,16],[95,21],[57,28],[26,48],[90,4],[68,22],[46,28],[104,42],[99,10]]
agv_start_positions = [[3, 1], [5, 1], [7, 1], [9, 1], [11, 1],[13,1],[15,1],[17,1],[19,1],[21,1],[23,1],[25,1],[27,1]]
picking_stations = [[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27,21],[27,21],[27,21],[27,21],[27,21],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9]]
n = 6  # AGV一次最多运送的任务数
v = 0.5  # AGV的平均速度
max_generations=30000
Pm=0.05  #变异系数
Pc=0.6  #交叉系数
initial_temperature=1000
cooling_rate=0.95

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
    return (abs(point1[0] - point2[0]) + abs(point1[1] - point2[1]))*0.5


def calculate_agv_time(genes, task_positions, agv_start_positions, picking_stations, n, v):
    agv_times = []  # 存储每辆AGV完成任务的总时间

    # 遍历每辆AGV及其分配的任务
    for agv_id, tasks in enumerate(genes):
        agv_time = 0  # 初始化当前AGV的时间
        #print(agv_id)
        current_position = agv_start_positions[agv_id]  # AGV的起始位置

        # 分批处理任务
        for i in range(0, len(tasks), n):
            batch = tasks[i:i+n]  # 当前批次的任务

            # 遍历批次中的任务
            for task_id in batch:
                # 移动到任务点，并更新时间和位置
                task_position = task_positions[task_id]
                agv_time += manhattan_distance(current_position, task_position) / v
                current_position = task_position

                # 移动到对应拣货台，并更新时间和位置
                picking_station = picking_stations[task_id]
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

task_positionss,picking_stationss,task_counts=filter_duplicate_tasks_with_picking_stations(task_positions, picking_stations)
print(task_positionss,picking_stationss,task_counts)

def genetic_algorithm_with_sa(population_size, task_counts, agv_count, task_positionss, picking_stationss,
                              agv_start_positions, v, max_generations, Pm, Pc, initial_temperature,
                              cooling_rate):
    population = initialize_population(population_size, task_counts, agv_count)
    best_fitness_values = []
    temperature = initial_temperature
    overall_best_chromosome = None
    overall_best_fitness = float('inf')
    overall_best_agv_times = []  # 用于保存最优适应度值对应的每辆AGV完成任务的时间列表
    overall_best_agv_tasks = []  # 新增，用于保存最优适应度值对应的每辆AGV的搬运任务及顺序

    for generation in tqdm(range(max_generations), desc="Processing Generations"):
        fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome, agv_count), task_positionss,
                                           agv_start_positions, picking_stationss, n, v) for chromosome in population]

        current_best_fitness = min(fitness_values)
        current_best_index = fitness_values.index(current_best_fitness)
        if current_best_fitness < overall_best_fitness:
            overall_best_fitness = current_best_fitness
            overall_best_chromosome = population[current_best_index]
            # 保存当前最优解对应的每辆AGV的完成时间列表和搬运任务及顺序
            overall_best_agv_times = calculate_agv_time(get_agv_tasks_from_chromosome(overall_best_chromosome, agv_count),
                                                        task_positionss, agv_start_positions, picking_stationss, n, v)
            # 获取并保存每辆AGV的搬运任务及顺序
            overall_best_agv_tasks = get_agv_tasks_from_chromosome(overall_best_chromosome, agv_count)

        best_fitness_values.append(current_best_fitness)

        new_population = perform_crossover(population, Pc, agv_count)
        new_population = mutate_population_with_shift(new_population, agv_count, Pm)

        new_fitness_values = [fitness_function(get_agv_tasks_from_chromosome(chromosome, agv_count), task_positionss,
                                               agv_start_positions, picking_stationss, n, v) for chromosome in new_population]

        for i in range(len(population)):
            if new_fitness_values[i] < fitness_values[i] or np.random.rand() < np.exp(
                    (fitness_values[i] - new_fitness_values[i]) / temperature):
                population[i] = new_population[i]
                fitness_values[i] = new_fitness_values[i]

        temperature *= cooling_rate

    plt.plot(best_fitness_values)
    plt.title('算法迭代图')
    plt.xlabel('迭代次数')
    plt.ylabel('最长完成时间 (s)')
    plt.show()

    # 返回最优解、最优适应度值、最优适应度值历史记录、最优解对应的每辆AGV的完成时间列表以及搬运任务及顺序
    return overall_best_chromosome, overall_best_fitness, best_fitness_values, overall_best_agv_times, overall_best_agv_tasks

# 执行函数
best_solution, best_fitness, best_fitness_values, best_agv_times, best_agv_tasks = genetic_algorithm_with_sa(
    population_size, task_counts, agv_count, task_positionss, picking_stationss,
    agv_start_positions, v, max_generations, Pm, Pc, initial_temperature, cooling_rate)

# 输出结果
print("最优AGV任务分配结果:", best_solution)
print("最优适应度值:", best_fitness)
print("每次迭代后的最优适应度值:", best_fitness_values)
print("对应的每辆AGV完成任务的时间列表:", best_agv_times)
print("最优适应度下每辆AGV的搬运任务及顺序:", best_agv_tasks)
