import heapq


class Node:
    """代表网格中的一个节点"""

    def __init__(self, position, parent=None):
        self.position = position  # 节点在网格中的位置，以元组形式表示(x, y)
        self.parent = parent  # 节点的父节点
        self.g = 0  # 从起点到当前节点的实际移动成本
        self.h = 0  # 从当前节点到终点的估计移动成本（启发式成本）
        self.f = 0  # 节点的总成本(f = g + h)

    def __lt__(self, other):
        # 重载小于操作符，用于在优先队列中按f值排序节点
        return self.f < other.f


def astar(grid, start, goal, turn_penalty=10):
    """执行A*算法找到最短路径"""
    open_list = []  # 开放列表，存储待检查的节点
    closed_list = set()  # 关闭列表，存储已检查的节点

    start_node = Node(start)
    goal_node = Node(goal)
    heapq.heappush(open_list, start_node)  # 将起点加入开放列表

    while open_list:
        current_node = heapq.heappop(open_list)  # 取出并删除开放列表中f值最低的节点
        closed_list.add(current_node.position)  # 将当前节点添加到关闭列表中

        # 如果当前节点是终点，重构并返回路径
        if current_node.position == goal:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # 返回反向路径（从起点到终点）

        # 检查当前节点的所有相邻节点
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0)]:  # 上、下、左、右移动
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # 确保相邻节点在网格范围内，并且不是障碍物
            if node_position[0] > (len(grid) - 1) or node_position[0] < 0 or node_position[1] > (len(grid[0]) - 1) or \
                    node_position[1] < 0:
                continue
            if grid[node_position[0]][node_position[1]] != 0:
                continue
            if node_position in closed_list:
                continue

            # 创建新的节点
            new_node = Node(node_position, current_node)
            new_node.g = current_node.g + 1  # 更新g成本

            # 计算拐点惩罚
            if current_node.parent:
                # 计算横纵坐标的变化
                dx = abs(new_node.position[0] - current_node.parent.position[0])
                dy = abs(new_node.position[1] - current_node.parent.position[1])
                # 如果横纵坐标都发生变化（即乘积不为0），则加入拐点惩罚
                if dx * dy != 0:
                    new_node.g += turn_penalty  # 应用拐点惩罚

            # 计算启发式成本h和总成本f
            new_node.h = (abs(new_node.position[0] - goal_node.position[0]) ** 2) + (
                        abs(new_node.position[1] - goal_node.position[1]) ** 2)
            new_node.f = new_node.g + new_node.h

            # 检查新节点是否已在开放列表中，如果是，则检查是否需要更新其成本
            if add_to_open(open_list, new_node):
                heapq.heappush(open_list, new_node)  # 将新节点加入开放列表

    return None  # 如果没有找到路径，则返回None


def add_to_open(open_list, neighbor):
    """检查是否应该将邻居节点添加到开放列表"""
    for node in open_list:
        if neighbor.position == node.position and neighbor.f >= node.f:
            return False  # 如果开放列表中已存在相同位置且成本不更优的节点，则不添加
    return True

    # 示例代码用法
grid = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
start = (0, 0)
goal = (3, 8)

path = astar(grid, start, goal, turn_penalty=100)
print(path)
