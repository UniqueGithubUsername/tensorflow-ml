import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Constants
v_line = 400 #V
v_phase = v_line / math.sqrt(3)
load = 36 #kW
single_load = load/3
power_factor = 0.86
iters = 10
time = np.linspace(0, iters*np.pi, 100*iters)

# noise arrays
noise1 = np.random.normal(0,10,100*iters)
noise2 = np.random.normal(0,10,100*iters)
noise3 = np.random.normal(0,10,100*iters)
noise4 = np.random.normal(0,1,100*iters)
noise5 = np.random.normal(0,1,100*iters)
noise6 = np.random.normal(0,1,100*iters)
# result arrays
v1 = []
v2 = []
v3 = []
i1 = []
i2 = []
i3 = []

for t in time:
    v1.append(v_phase * math.sin(2*math.pi*t))
    v2.append(v_phase * math.sin(2*math.pi*t+2/3*math.pi))
    v3.append(v_phase * math.sin(2*math.pi*t+4/3*math.pi))
    i1.append(single_load / power_factor * 1000 / v_phase)
    i2.append(single_load / power_factor * 1000 / v_phase)
    i3.append(single_load / power_factor * 1000 / v_phase)

# Add noise
v1 = v1+noise1
v2 = v2+noise2
v3 = v3+noise3
i1 = i1+noise4
i2 = i2+noise5
i3 = i3+noise6

# Plotting
plt.plot(time, v1, label='V1')
plt.plot(time, v2, label='V2')
plt.plot(time, v3, label='V3')
plt.show()
plt.plot(time, i1, label='I1')
plt.plot(time, i2, label='I2')
plt.plot(time, i3, label='I3')
plt.show()

# Save results as csv
dict = {
    'time':time,
    'v1':v1,
    'v2':v2,
    'v3':v3,
    'i1':i1,
    'i2':i2,
    'i3':i3,
}

df = pd.DataFrame(dict)
df.to_csv('3phase.csv')