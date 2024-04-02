import numpy as np

task_count = 80  # 搬运任务数量
agv_count = 5  # AGV数量
task_positions = [[112,9],[96,28],[57,16],[19,42],[93,36],[63,42],[64,4],[89,15],[93,42],[40,36],[102,10],[57,22],[107,28],[27,48],[63,42],[90,48],[27,41],[56,48],[90,9],[107,28],[63,32],[79,41],[89,15],[47,43],[103,3],[78,36],[95,21],[76,48],[63,36],[83,3],[57,16],[48,22],[103,22],[83,42],[46,28],[46,48],[85,22],[89,32],[93,42],[48,10],[13,48],[49,3],[41,16],[111,21],[54,9],[50,36],[102,36],[90,9],[66,15],[96,28],[30,42],[65,28],[67,48],[109,16],[57,22],[42,4],[39,28],[27,41],[8,42],[98,16],[48,10],[100,48],[77,29],[100,4],[63,42],[65,10],[105,16],[85,10],[13,42],[57,28],[114,4],[48,16],[95,21],[57,28],[26,48],[90,4],[68,22],[46,28],[104,42],[99,10]]
agv_start_positions = [[3, 1], [5, 1], [7, 1], [9, 1], [11, 1]]
picking_stations = [[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,32],[27,21],[27,21],[27,21],[27,21],[27,21],[27,32],[27,32],[27,32],[27,32],[27,32],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27,21],[27,21],[27,21],[27,21],[27,21],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9],[27, 9]]
n = 6  # AGV一次最多运送的任务数
v = 0.1  # AGV的平均速度
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
agv_count=5
chromosome= [
    [6, 3],
    [54, 0],
    [2, 0],
    [37, 3],
    [0, 0],
    [57, 2],
    [34, 1],
    [42, 1],
    [16, 3],
    [33, 2],
    [36, 4],
    [38, 1],
    [46, 0],
    [55, 1],
    [22, 3],
    [32, 1],
    [35, 0],
    [5, 1],
    [29, 2],
    [65, 1],
    [13, 2],
    [4, 4],
    [23, 1],
    [11, 3],
    [63, 1],
    [19, 4],
    [8, 3],
    [17, 1],
    [18, 3],
    [9, 4],
    [25, 0],
    [40, 4],
    [14, 4],
    [52, 2],
    [56, 2],
    [7, 2],
    [58, 3],
    [15, 2],
    [41, 2],
    [30, 1],
    [45, 3],
    [47, 4],
    [49, 0],
    [64, 3],
    [20, 0],
    [3, 2],
    [39, 3],
    [12, 0],
    [44, 4],
    [24, 0],
    [50, 4],
    [1, 0],
    [27, 2],
    [48, 4],
    [61, 2],
    [43, 4],
    [21, 0],
    [51, 1],
    [62, 4],
    [59, 4],
    [26, 4],
    [10, 3],
    [53, 3],
    [60, 2],
    [28, 2],
    [31, 2]
]

result=get_agv_tasks_from_chromosome(chromosome, agv_count)
time=calculate_agv_time(result, task_positions, agv_start_positions, picking_stations, n, v)
print(result)
print(time)

