import matplotlib.pyplot as plt
import numpy as np
import random

population_size = 100  # 种群大小
task_count = 10  # 搬运任务数量
agv_count = 5  # AGV数量
task_positions=[[3, 8],[ 18, 11],[20, 5], [6, 11], [8, 17], [ 20, 14], [6,  5], [ 8, 17], [ 18, 11], [ 15, 14]]
picking_stations=[[8,1],[ 8, 1],[8, 1], [13, 1], [13, 1], [13, 1],[ 18, 1],[18, 1], [18, 1], [18, 1]]
agv_start_positions=[[3,0],[3,0]]
agv_speed=0.1
generations=700
m=0.05


def initialize_population(population_size, task_count, agv_count):

    """
    初始化种群
    :param population_size: 种群大小
    :param task_count: 搬运任务的数量
    :param agv_count: 物流AGV的数量
    :return: 初始化后的种群
    """
    population = []
    for _ in range(population_size):
        # 对于每个染色体，每个任务随机分配给一个AGV
        # 基因值在[0, agv_count-1]范围内，每个值代表一个AGV的编号
        chromosome = np.random.randint(0, agv_count, size=task_count)
        population.append(chromosome)
    return population

# # 初始化种群
# population = initialize_population(population_size, task_count, agv_count)
#
# # 打印初始化种群
# for i, chromosome in enumerate(population):
#     print(f"染色体 {i+1}: {chromosome}")

def get_agv_task_lists(chromosome):
    agv_tasks = {}
    for task_id, agv_id in enumerate(chromosome):
        # 确保 agv_id 转换为整数，以免出现类型错误
        agv_id = int(agv_id)  # 这里 agv_id 已经是整数，这一步实际上是多余的，但为了清晰说明问题保留
        if agv_id not in agv_tasks:
            agv_tasks[agv_id] = []
        agv_tasks[agv_id].append(task_id)

    # 将所有AGV的任务列表组织到一个大列表中
    all_agv_tasks = list(agv_tasks.values())

    return all_agv_tasks
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

    # 遍历每辆AGV及其任务分配
    for agv_index, assignments in enumerate(task_assignments):
        total_time = 0
        current_position = agv_start_positions[agv_index]  # AGV的当前位置初始化为起始位置

        # 遍历AGV的任务分配
        for i, task_id in enumerate(assignments):
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

def adaptive_crossover_mutation_rates(k, K, m):
    """计算自适应交叉率和变异率"""
    Pc = 1 / (1 + m * np.exp(k / K))
    Pm = 1 / (1 - m * np.exp(k / K))
    return Pc, Pm


def direct_copy_crossover(parent1, parent2):
    size = len(parent1)  # 假设两个父代长度相同
    crossover_point = random.randint(1, size - 1)  # 选择一个交叉点

    # 创建子代
    child1 = np.zeros(size, dtype=parent1.dtype)
    child2 = np.zeros(size, dtype=parent2.dtype)

    # 直接复制父代基因到子代
    child1[:crossover_point] = parent1[:crossover_point]
    child1[crossover_point:] = parent2[crossover_point:]

    child2[:crossover_point] = parent2[:crossover_point]
    child2[crossover_point:] = parent1[crossover_point:]

    return child1, child2


def crossover(population, Pc):
    """对种群执行交叉操作"""
    new_population = []
    for i in range(0, len(population), 2):
        print("第" + str(i) + "次交叉操作")
        parent1 = population[i]
        parent2 = population[(i + 1) % len(population)]  # 确保索引不会越界
        if random.random() < Pc:
            child1, child2 = direct_copy_crossover(parent1, parent2)  # 正确解包子代
            new_population.extend([child1, child2])  # 分别添加
        else:
            new_population.extend([parent1, parent2])  # 不进行交叉，直接添加
    print(new_population)
    return new_population

def insertion_mutation(chromosome, Pm):
    """对单个染色体执行插入变异操作，并确保返回numpy数组格式的染色体"""
    if random.random() < Pm:  # 根据变异概率决定是否进行变异
        size = len(chromosome)
        gene_index = random.randint(0, size - 1)  # 随机选择一个基因
        gene = chromosome[gene_index]  # 提取该基因
        insert_index = random.randint(0, size - 1)  # 选择一个插入位置
        # 从列表转换成numpy数组以进行操作
        chromosome_list = list(chromosome)
        # 移除并插入基因
        chromosome_list.pop(gene_index)
        chromosome_list.insert(insert_index, gene)
        # 将列表转换回numpy数组
        chromosome = np.array(chromosome_list)
    return chromosome

def apply_insertion_mutation(population, Pm):
    """对种群中的每个染色体执行插入变异操作，并确保染色体保持为numpy数组格式"""
    new_population = []
    for chromosome in population:
        mutated_chromosome = insertion_mutation(chromosome, Pm)  # 直接传入numpy数组，返回也是numpy数组
        new_population.append(mutated_chromosome)
    return new_population


# 假定已定义的函数：initialize_population, calculate_task_time, adaptive_crossover_mutation_rates, crossover, apply_insertion_mutation, get_agv_task_lists

def fitness_function(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed):
    """
    根据给定的染色体计算适应度值。适应度值是完成所有任务的最长时间的倒数。

    :param chromosome: 染色体，表示任务分配给AGV的方案。
    :param task_positions: 任务的位置列表。
    :param picking_stations: 拣选站的位置列表。
    :param agv_start_positions: AGV的起始位置列表。
    :param agv_speed: AGV的速度。
    :return: 适应度值，为完成所有任务的最长时间的倒数。
    """


    # 计算每辆AGV完成其分配任务的总时间
    agv_total_times = calculate_task_time(task_assignments, task_positions, picking_stations, agv_start_positions, agv_speed)

    # 找到最长的任务完成时间
    max_time = max(agv_total_times)

    # 适应度值为最长完成时间的倒数（目的是最小化最长完成时间）
    fitness = 1/max_time

    return fitness


def simulated_annealing_acceptance(current_fitness, new_fitness, temperature):
    # 如果新解更好，直接接受
    if new_fitness > current_fitness:
        return True
    # 如果新解更差，根据模拟退火准则决定是否接受
    else:
        probability = np.exp((new_fitness - current_fitness) / temperature)
        return random.random() < probability


def select_parents_roulette(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [f / total_fitness for f in fitness_scores]

    # 选择父代
    selected_indices = np.random.choice(len(population), size=2, p=selection_probs, replace=False)
    return population[selected_indices[0]], population[selected_indices[1]]
def genetic_algorithm_with_simulated_annealing(task_positions, picking_stations, agv_start_positions, agv_speed,
                                               population_size, generations, m, elite_size):
    population = initialize_population(population_size, len(task_positions), len(agv_start_positions))
    Pc=0.4
    Pm=0.05
    best_solution = None
    best_fitness = -np.inf
    temperature = 100  # 初始温度
    cooling_rate = 0.6  # 冷却率
    fitness_history = []  # 用于记录每次迭代的最优适应度值
    for generation in range(generations):
        # 评估当前代种群
        fitness_scores = [
            fitness_function(get_agv_task_lists(chromosome), task_positions, picking_stations, agv_start_positions,
                             agv_speed) for chromosome in population]

        # 获取适应度分数并与染色体一起存储
        scored_population = list(zip(population, fitness_scores))
        # 按适应度排序
        scored_population.sort(key=lambda x: x[1], reverse=True)

        # 选择精英个体
        elites = [x[0] for x in scored_population[:elite_size]]

        # 创建新种群并添加精英个体
        new_population = elites[:]

        # 使用轮盘赌选择和交叉填充其余种群
        while len(new_population) < population_size:
            parent1, parent2 = select_parents_roulette(population, fitness_scores)
            if random.random() < Pc:
                child1, child2 = direct_copy_crossover(parent1, parent2)
                new_population.extend([child1, child2])
            else:
                new_population.extend([parent1, parent2])

        # 对除了精英个体外的染色体进行变异
        for i in range(elite_size, len(new_population)):
            new_population[i] = insertion_mutation(new_population[i], Pm)

        population = new_population

        # 评估新种群，更新最优解
        for chromosome in population:
            fitness = fitness_function(get_agv_task_lists(chromosome), task_positions, picking_stations,
                                       agv_start_positions, agv_speed)
            if fitness > best_fitness or simulated_annealing_acceptance(best_fitness, fitness, temperature):
                best_fitness = fitness
                best_solution = chromosome

        fitness_history.append(1/best_fitness)
        temperature *= cooling_rate
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_solution, best_fitness, fitness_history


# 在运行算法时指定精英大小
elite_size = 2  # 例如，保留每代的前两个最优个体
best_solution, best_fitness, fitness_history = genetic_algorithm_with_simulated_annealing(
    task_positions, picking_stations, agv_start_positions, agv_speed, population_size, generations, m, elite_size
)


print("Best Solution:", best_solution)
print("Best Fitness:", best_fitness)
print("time:",1/best_fitness)
order=get_agv_task_lists(best_solution)
order_time=calculate_task_time(order,task_positions, picking_stations, agv_start_positions, agv_speed)
print("order:",order)
print("order_time:",order_time)

# 绘制最优适应度值随迭代过程的变化图
plt.plot(fitness_history)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Best Fitness over Generations')
plt.show()
