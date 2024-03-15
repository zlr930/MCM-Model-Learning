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
            print(tasks_this_round)
            tasks = tasks[n:]  # 更新剩余的任务列表

            # 遍历本轮的任务
            for task_index, task in enumerate(tasks_this_round):
                # 任务点到下一个任务点的距离
                task_position = task_positions[task]
                agv_time += manhattan_distance(current_position, task_position) / v
                current_position = task_position

            # 从当前位置到本轮所有任务对应拣货台的距离，选择最短距离的拣货台先去拣货
            picking_stations_this_round = [picking_stations[task] for task in tasks_this_round]
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


# 示例输入和计算每一辆AGV完成任务的总时间
genes = [[4, 3, 1, 0], [6, 7, 5], [2, 8, 9]]
task_positions = [[3, 8], [18, 11], [20, 5], [6, 11], [8, 17], [20, 14], [6, 5], [8, 17], [18, 11], [15, 14]]
agv_start_positions = [[3, 0], [3, 0], [3, 0]]
picking_stations = [[8, 1], [8, 1], [8, 1], [13, 1], [13, 1], [13, 1], [18, 1], [18, 1], [18, 1], [18, 1]]
n = 2  # AGV一次最多运送的任务数
v = 1  # AGV的平均速度

agvs_times = calculate_agv_time(genes, task_positions, agv_start_positions, picking_stations, n, v)

# 输出每一辆AGV完成任务的总时间
for i, time in enumerate(agvs_times):
    print(f"AGV{i + 1} 完成所有任务并返回起始点所需的总时间为：{time:.2f} 单位时间")
