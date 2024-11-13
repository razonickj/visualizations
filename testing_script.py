import numpy as np
#import matplotlib as mpl
from matplotlib import pyplot as plt

#setup the style
plt.style.use("./njr_c1.mplstyle")
#plt.style.use("./ex_styl.mplstyle")

#get some X data
x = np.array([1,2,3,4,5,6,7,8,9])
y = np.power(x,2)

plt.plot(x,y, label='First')
for k in range(10):
    plt.plot(x, (k+2)*x, label=f"{k+2}")
plt.legend()
plt.xlabel(f"Dummy X axis")
plt.ylabel(f"Dummy Y axis")
plt.show()