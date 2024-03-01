import numpy as np

population_size = 10  # 种群大小
task_count = 20  # 搬运任务数量
agv_count = 5  # AGV数量

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

# 初始化种群
population = initialize_population(population_size, task_count, agv_count)

# 打印初始化种群
for i, chromosome in enumerate(population):
    print(f"染色体 {i+1}: {chromosome}")

