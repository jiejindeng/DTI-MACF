import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

a=[0.4,0.5,0.6,0.7,0.8,0.9,1]
b=[0.93943,0.94279,0.94793,0.95079,0.95404,0.95070,0.94193]

c=[0.4,0.5,0.6,0.7,0.8,0.9,1]
d=[0.94758,0.95090,0.94871,0.95387,0.95708,0.95388,0.95352]

# plt.title('Title')
plt.plot(a, b, color='blue', label='AUROC')
plt.plot(c, d, color='RED', label='AUPR')
plt.legend()  # 显示图例

plt.xlabel('g')
plt.ylabel('value')
plt.savefig("test.jpg")
