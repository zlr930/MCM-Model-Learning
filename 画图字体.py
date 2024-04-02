import matplotlib.pyplot as plt
import matplotlib

# 设置字体为Songti SC，字体大小为12，字体权重为轻
matplotlib.rcParams['font.family'] = 'Heiti TC'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.weight'] = 'light'  # 尝试使用 'normal' 如果 'light' 效果不理想
matplotlib.rcParams['axes.unicode_minus'] = False  # 确保能够显示负号

# 绘图示例
plt.title('图表标题')
plt.xlabel('横轴标题')
plt.ylabel('纵轴标题')
plt.plot([1, 2, 3], [1, 4, 9], label='示例线条')
plt.legend(title='图例')
plt.show()

from matplotlib.font_manager import fontManager
import os

fonts = [f.name for f in fontManager.ttflist if 'Heiti' in f.name or 'Song' in f.name]
print("可用的中文字体：")
for font in fonts:
    print(font)

