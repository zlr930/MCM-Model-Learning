import heapq


class Node:
    """代表网格中的一个节点"""

    def __init__(self, position, parent=None):
        self.position = position  # 节点在网格中的位置
        self.parent = parent  # 节点的父节点
        self.g = 0  # 从起点到当前节点的实际移动成本
        self.h = 0  # 从当前节点到终点的估计移动成本（启发式成本）
        self.f = 0  # 节点的总成本(f = g + h)

    def __lt__(self, other):
        # 在优先队列中需要比较节点时，根据f值进行比较
        return self.f < other.f


def get_neighbors(position, grid):
    """获取节点的所有可行邻居节点"""
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # 上下左右移动
    neighbors = []
    for d in directions:
        neighbor = (position[0] + d[0], position[1] + d[1])
        if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 0:
            neighbors.append(neighbor)
    return neighbors


def calculate_turn_penalty(parent, current, next_position, turn_penalty):
    """计算拐弯惩罚因子"""
    if not parent:
        return 0  # 如果没有父节点，说明是起点，不存在拐弯
    prev_move = (current.position[0] - parent.position[0], current.position[1] - parent.position[1])
    current_move = (next_position[0] - current.position[0], next_position[1] - current.position[1])
    return turn_penalty if prev_move != current_move else 0


def astar(grid, start, goal, turn_penalty=10):
    """执行A*算法找到最短路径"""
    open_list = []  # 开放列表，用于存储待检查的节点
    closed_list = set()  # 关闭列表，用于存储已检查的节点

    start_node = Node(start)
    heapq.heappush(open_list, start_node)  # 将起点加入开放列表

    while open_list:
        current_node = heapq.heappop(open_list)  # 选择F值最小的节点
        closed_list.add(current_node.position)  # 将当前节点加入到关闭列表

        if current_node.position == goal:
            # 如果找到终点，重构并返回路径
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        neighbors = get_neighbors(current_node.position, grid)
        for next_position in neighbors:
            if next_position in closed_list:
                continue  # 如果节点已在关闭列表中，则跳过

            new_node = Node(next_position, current_node)
            penalty = calculate_turn_penalty(current_node.parent, current_node, next_position, turn_penalty)
            new_node.g = current_node.g + 1 + penalty
            new_node.h = abs(next_position[0] - goal[0]) + abs(next_position[1] - goal[1])
            new_node.f = new_node.g + new_node.h

            if any(n.position == next_position and n.f <= new_node.f for n in open_list):
                continue
            heapq.heappush(open_list, new_node)

    return None  # 如果开放列表为空，搜索失败

# 示例用法
grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]
start = (12, 0)  # 起点坐标
goal = (0, 12)  # 终点坐标
turn_penalty = 1000# 设定拐弯惩罚因子

path = astar(grid, start, goal)
print("最短路径:", path)
