import matplotlib.pyplot as plt
import numpy as np
import json

def open_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

plt.figure()
plt.axis('equal')
plt.ylim(0, 1)
filenames = ['data/motion_primitives/1.0_1.0_0.0.json', 'data/motion_primitives/1.0_1.0_0.2.json', 'data/motion_primitives/1.0_1.0_0.4.json']
filenames2 = ['data/motion_primitives/approach21.0_1.0_0.0.json', 'data/motion_primitives/approach21.0_1.0_0.2.json', 'data/motion_primitives/approach21.0_1.0_0.4.json']
# filenames = ['data/motion_primitives/approach21.0_1.0_0.0.json',
#              'data/motion_primitives/approach21.0_1.0_0.2.json',
#              'data/motion_primitives/approach21.0_1.0_0.4.json',
#              'data/motion_primitives/1.0_1.0_0.0.json',
#              'data/motion_primitives/1.0_1.0_0.2.json',
#              'data/motion_primitives/1.0_1.0_0.4.json']

for filename in filenames:
    points = open_json(filename)['points']
    points = np.array(points)
    plt.plot(points[:, 1], points[:, 0], color='b')
    # plt.plot(-points[:, 1], points[:, 0])

for filename in filenames2:
    points = open_json(filename)['points']
    points = np.array(points)
    plt.plot(points[:, 1], points[:, 0], color='g')
    # plt.plot(-points[:, 1], points[:, 0])

plt.show()
