import numpy as np
import matplotlib.pyplot as plt

rho = 0.5
Q = 0.1
y = [0.1,]
for i in range(10-1):
    y.append(y[-1]*rho+Q)
x = np.arange(1,len(y)+1)
plt.plot(x,y)
plt.show()
