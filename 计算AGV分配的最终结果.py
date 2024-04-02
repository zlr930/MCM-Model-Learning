import numpy as np

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
