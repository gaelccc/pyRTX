from pyRTX.core.analysis_utils import LookupTableND
import numpy as np
import timeit
import matplotlib.pyplot as plt
x = np.linspace(0, np.pi, 100)
y = np.linspace(0, np.pi, 1000)
z = np.linspace(0, np.pi, 100)

Z,Y,X = np.meshgrid(z,y,x)
print(X.shape, Y.shape, Z.shape)
v = np.sin(X)*np.cos(Y)*np.sin(Z)
print(v.shape)

lut = LookupTableND(axes = (x,y, z), values = v)


nx = np.linspace(0, np.pi, 10)
ny = np.linspace(0, np.pi, 10)
nz = np.linspace(0, np.pi, 10)


times = []
execs = 100
print('Running')
for i in range(execs):
    tic = timeit.default_timer()
    newvals = lut[nx, ny, nz]
    toc = timeit.default_timer()
    times.append(toc-tic)

mean = np.mean(times)
std = np.std(times)
print(f'Elapsed time: {mean:.2e}+-{std:.2e}')



#fig, ax = plt.subplots()
#ax.contourf(x,y,Z)
#plt.show()
