import matplotlib.pyplot as plt
import numpy as np

size = 3
# 返回size个0-1的随机数
# a = np.random.random(size)
a = [0.894, 0.937, 0.773]
# b = np.random.random(size)
b = [0.892, 0.906, 0.749]
# c = np.random.random(size)
c = [0.912, 0.933, 0.752]
# d = np.random.random(size)
d = [0.922, 0.938, 0.812]
# e = np.random.random(size)
e = [0.96018, 0.95439, 0.90718]

# x轴坐标, size=5, 返回[0, 1, 2, 3, 4]
x = np.arange(size)
# x= ['EN', 'IC', 'GPCR', 'NC']
# 有a/b/c三种类型的数据，n设置为3
total_width, n = 0.5, 5
# 每种类型的柱状图宽度
width = total_width / n

# 重新设置x轴的坐标
x = x - (total_width - width) / 2
print(x)
x_labels = ['EN', 'IC', 'GPCR']
# plt.xticks(x, x_labels)

# 画柱状图
plt.bar(x, a, width=width, label="MSCMF")
plt.bar(x + width, b, width=width, label="NRLMF")
plt.bar(x + 2*width, c, width=width, label="mk-TCMF")
plt.bar(x + 3*width, d, width=width, label="DNILMF")
plt.bar(x + 4*width, e, width=width, label="DTI-MACF")
plt.ylim(0, 1.4)
plt.ylabel('AUPR',size=15)

plt.xticks([0.0,1.0,2.0],x_labels)


# 显示图例
plt.legend()
# 显示柱状图
plt.savefig("test1.jpg")
plt.show()

