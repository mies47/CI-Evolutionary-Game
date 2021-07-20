import json
import numpy as np
import matplotlib.pyplot as plt

with open('plotting.json', 'r') as f:
    data = json.load(f)

    avg = np.array(data['avg'])
    minimum = np.array(data['min'])
    maximum = np.array(data['max'])

plt.figure(figsize = (12,10))
plt.scatter(avg[:,0], avg[:,1], marker='o')
plt.scatter(minimum[:,0], minimum[:,1], marker='^')
plt.scatter(maximum[:,0], maximum[:,1], marker='v')

plt.show()


