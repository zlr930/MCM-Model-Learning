import heapq
import matplotlib.pyplot as plt
import numpy as np


class Node:
    """代表网格中的一个节点"""

    def __init__(self, position, parent=None):
        self.position = position  # 节点在网格中的位置
        self.parent = parent  # 节点的父节点
        self.g = 0  # 从起点到当前节点的实际移动成本
        self.h = 0  # 从当前节点到终点的估计移动成本（启发式成本）
        self.f = 0  # 节点的总成本(f = g + h)
        self.turns = 0  # 新增：追踪拐弯次数

    def __lt__(self, other):
        # 在比较节点时，先比较总成本f，如果f相同，则比较拐弯次数
        if self.f == other.f:
            return self.turns < other.turns
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
    if not parent:
        return 0, 0  # 没有拐弯，拐弯次数不变
    prev_move = (current.position[0] - parent.position[0], current.position[1] - parent.position[1])
    current_move = (next_position[0] - current.position[0], next_position[1] - current.position[1])
    if prev_move != current_move:
        return turn_penalty, 1  # 发生拐弯，拐弯次数加1
    return 0, 0  # 没有拐弯，拐弯次数不变

# 修改 astar 函数以考虑拐弯次数
def astar(grid, start, goal, turn_penalty):
    open_list = []
    closed_list = set()
    start_node = Node(start)
    heapq.heappush(open_list, start_node)

    while open_list:
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node.position)

        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        neighbors = get_neighbors(current_node.position, grid)
        for next_position in neighbors:
            if next_position in closed_list:
                continue

            new_node = Node(next_position, current_node)
            penalty, turns = calculate_turn_penalty(current_node.parent, current_node, next_position, turn_penalty)
            new_node.g = current_node.g + 1 + penalty
            new_node.h = abs(next_position[0] - goal[0]) + abs(next_position[1] - goal[1])
            new_node.f = new_node.g + new_node.h
            new_node.turns = current_node.turns + turns  # 更新拐弯次数

            if not any(n for n in open_list if n.position == next_position and (n.f < new_node.f or (n.f == new_node.f and n.turns <= new_node.turns))):
                heapq.heappush(open_list, new_node)

    return None
def plot_path(grid, path):
    grid_array = np.array(grid)
    print(grid_array)
    for position in path:
        grid_array[position] = 2
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid_array, cmap="viridis")
    start, goal = path[0], path[-1]
    ax.scatter(start[1], start[0], color='red', label='Start')
    ax.scatter(goal[1], goal[0], color='blue', label='Goal')
    ax.legend()
    ax.axis('off')
    plt.show()

# 示例用法
grid = [
    [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
]
start = (12, 0)  # 起点坐标
goal = (0, 12)  # 终点坐标
turn_penalty = 3# 设定拐弯惩罚因子

path = astar(grid, start, goal,turn_penalty)
print("最短路径:", path)
plot_path(grid, path)
