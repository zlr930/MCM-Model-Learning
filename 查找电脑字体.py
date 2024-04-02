import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# 动态寻找支持中文的字体
font = FontProperties(fname='/System/Library/Fonts/STHeiti Light.ttc')
# 使用findSystemFonts()函数也可以动态查找系统字体

plt.plot([1, 2, 3], [4, 5, 6], label='示例图例')
plt.legend(prop=font)
plt.show()
