import matplotlib.pyplot as plt
import numpy as np

size = 5
# 返回size个0-1的随机数
a = [0.951118, 0.951466, 0.956, 0.952248, 0.949766]
b = [0.95680, 0.957066, 0.960, 0.956292, 0.952404]
# x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
x = np.arange(size)+1

# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.8, 2
# 每种类型的柱状图宽度
width = total_width / n

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
print(x)

# 画柱状图
plt.bar(x, a, width=width, label="AUROC")
plt.bar(x + width, b, width=width, label="AUPR")
plt.ylim(0.94, 0.965)
# 显示图例
plt.legend()
# 显示柱状图
plt.savefig("test2.jpg")
plt.show()
