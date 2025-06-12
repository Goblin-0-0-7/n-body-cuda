import os
from mpl_toolkits.mplot3d import Axes3D
import time
import matplotlib.pyplot as plt

num_part = 10

def loadPositionsStr(filedir):
    with open(filedir, 'r') as f:
        data = f.readlines()
    return data

def extractStepData(strData, num_part):
    steps: list[list[int, int, int]] = []
    step_tot = int(len(strData) / (num_part + 1))
    for i in range(step_tot):
        step = []
        for j in range(num_part):
            line_num = (i+1) + j + i*num_part
            xStr,yStr,zStr,_ = strData[line_num].split(" ")
            step.append([float(xStr), float(yStr), float(zStr)])
        steps.append(step)
    return steps

workdir = os.getcwd()
strData = loadPositionsStr(workdir + "\\log\\lastSimulation.txt")
stepData = extractStepData(strData, num_part)


# Visualize the first step's points in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw a box centered at the origin with side length 1
r = 0.5
box_points = [
    [-r, -r, -r], [ r, -r, -r], [ r,  r, -r], [-r,  r, -r],  # bottom face
    [-r, -r,  r], [ r, -r,  r], [ r,  r,  r], [-r,  r,  r],  # top face
]
edges = [
    (0,1), (1,2), (2,3), (3,0),  # bottom
    (4,5), (5,6), (6,7), (7,4),  # top
    (0,4), (1,5), (2,6), (3,7)   # sides
]
for e in edges:
    x = [box_points[e[0]][0], box_points[e[1]][0]]
    y = [box_points[e[0]][1], box_points[e[1]][1]]
    z = [box_points[e[0]][2], box_points[e[1]][2]]
    ax.plot(x, y, z, color='black')

# Set equal aspect ratio for all axes
ax.set_xlim([-r, r])
ax.set_ylim([-r, r])
ax.set_zlim([-r, r])
ax.set_box_aspect([1,1,1])  # Requires matplotlib >= 3.3


for step in stepData:
    xs = [p[0] for p in step]
    ys = [p[1] for p in step]
    zs = [p[2] for p in step]
    colors = plt.cm.tab20(range(len(xs)))
    ax.scatter(xs, ys, zs, c=colors, s=40)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.draw()
    plt.pause(0.3)  # delta t in seconds
