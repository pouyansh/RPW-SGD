import numpy as np
import math
import torch

import matplotlib.pyplot as plt

dim = 2
rows_num = 8  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
radius = 0.05  # radius of the disks drawn for each center


# Drawing the maintained output distribution
def draw(centers):
    _, ax = plt.subplots()
    for center in centers:
        circle = plt.Circle((center[0], center[1]), radius, color='k')
        ax.add_patch(circle)
    plt.show()
    plt.close()


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num
print(out_centers)
draw(out_centers)
