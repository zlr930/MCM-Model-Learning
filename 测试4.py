
import matplotlib.pyplot as plt
import numpy as np

task_positions = [[3,5],[5,11],[4,17],[6,20],[24,2],[15,8],[25,8],[22,11],[13,17],[25,20],[6,5],[26,14],[18,11],[22,8],[5,11],[11,14],[15,5],[5,8],[9,2],[11,14],[24,5],[22,11],[3,5],[19,17],[28,5],[18,8],[1,11],[28,5],[21,14],[1,14],[9,5],[26,11],[19,17],[18,8],[14,11],[7,11],[16,14],[2,2],[22,17],[5,14],[19,5],[16,17],[14,14],[11,14],[14,11],[8,17],[7,8],[1,11],[7,8],[13,5],[11,20],[17,20],[28,17],[7,8],[11,11],[17,2],[28,14],[28,8],[21,14],[3,20]]
agv_start_positions = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
picking_stations = [[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[13, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[17, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1],[15, 1]]
n = 6  # AGV一次最多运送的任务数
v = 0.1  # AGV的平均速度
def filter_duplicate_tasks(task_positions):
    """
    移除task_positions中的重复任务位置，并返回筛选后的列表及其大小。

    :param task_positions: 一个列表，包含各个任务的位置。
    :return: 一个元组，包含筛选后的任务位置列表和任务数量。
    """
    # 使用集合（set）来自动移除重复项，但这会丢失原始顺序
    # 如果保持顺序很重要，可以使用下面的方法
    unique_task_positions = []
    [unique_task_positions.append(item) for item in task_positions if item not in unique_task_positions]

    # 更新任务数量
    task_count = len(unique_task_positions)

    return unique_task_positions

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

chromosome=[[24,4],[16 , 2] ,[20 , 2] ,[ 1 , 0] ,[41 , 4] ,[10 , 1], [ 0 , 0],[32  ,2],[ 2 , 3] ,[ 3  ,2] ,[36  ,0], [34  ,1] ,[ 8 , 2] ,[15 , 1] ,[35 , 4] ,[19 , 3] ,[33  ,1], [ 7 , 3], [40 , 4], [39 , 0], [42  ,1] ,[13 , 4] ,[11  ,3], [43 , 0] ,[ 6 , 3] ,[12 ,0] ,[ 4 , 3], [14 , 4] ,[46  ,0] ,[17 , 0] ,[45  ,1] ,[21 , 0] ,[18 , 2], [25 , 1], [28,  4] ,[26 , 4], [22 , 3], [ 5 , 4] ,[23 , 0],[37  ,1], [29 , 3], [ 9 , 2] ,[44  ,1] ,[31 , 1] ,[38 , 0] ,[27 , 3],[30 , 2]]
result=get_agv_tasks_from_chromosome(chromosome, 5)
total_time=calculate_agv_time(result, task_positions, agv_start_positions, picking_stations, n, v)
print(result)
print(total_time)


# 任务的位置坐标
task_positions1 = filter_duplicate_tasks(task_positions)
print(task_positions1)

# AGV初始位置
agv_start_positions = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

# 每辆AGV的任务顺序
tasks = result

# 为每辆AGV分配颜色
colors = ['blue', 'green', 'red', 'cyan', 'magenta']

plt.figure(figsize=(10, 8))

# 绘制每个任务的位置
for position in task_positions1:
    plt.scatter(position[0], position[1], color='black')

# 绘制每辆AGV的搬运路径
for i, agv in enumerate(tasks):
    # 包含AGV的初始位置
    path = [agv_start_positions[i]]
    # 添加AGV的任务位置
    for task in agv:
        path.append(task_positions1[task])
    path = np.array(path)
    plt.plot(path[:,0], path[:,1], color=colors[i], label=f'AGV {i+1}')

plt.legend()
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('AGV Task Routes')
plt.grid(True)

plt.show()
