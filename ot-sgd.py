import math
import torch # type: ignore
import os
import random
import ot # type: ignore
import matplotlib.pyplot as plt # type: ignore

from utils import sample, draw, draw_samples, compute_OT_error

dim = 2
rows_num = 32  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = 900
epoch_num = 30
lr = 0.2  # learning rate
p = 1
batch_size = 5
draw_interval = 10
from_cifar10 = True
beta = 0.4  # RGB significance in cifar10

path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_ot_lr" + str(lr) + "_p" + str(p) + "_bs" + str(batch_size) 
if from_cifar10:
    path += "_cifar10_beta" + str(beta)
path += "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num
colors = torch.FloatTensor([[beta, beta, beta] for _ in range(output_size)]) / 2

_, ax = plt.subplots()
c = colors / beta
c = c.tolist()
draw(out_centers, torch.ones(output_size) / output_size, ax, 0, path, colors=c)

for i in range(epoch_num):
    plt.close()
    _, ax = plt.subplots()
    if from_cifar10:
        arrows = torch.zeros((output_size, 3))
    else:
        arrows = torch.zeros((output_size, dim))
    for _ in range(batch_size):
        samples = sample(sample_size, beta=beta)
        if (i+1) % draw_interval == 0 and not from_cifar10:
            draw_samples(samples, ax)
        
        if from_cifar10:
            samples = torch.cat((out_centers, samples), dim=1)
            points = torch.cat((out_centers, colors), dim=1)
        else:
            points = out_centers
        cost_matrix = torch.cdist(points, samples, p=2)
        cost_matrix = torch.pow(cost_matrix, p)

        a = torch.ones(out_centers.shape[0]) / out_centers.shape[0]
        b = torch.ones(points.shape[0]) / points.shape[0]

        plan = ot.emd(a, b, cost_matrix)
        plan = plan.shape[0] * plan

        if from_cifar10:
            arrows = arrows + torch.matmul(plan, samples)[:, [2, 3, 4]] - colors
        else:
            arrows = arrows + torch.matmul(plan, samples) - out_centers

    # Averaging the arrows computed for each sample
    arrows = arrows / batch_size

    if from_cifar10:
        colors = colors + lr * arrows
        colors[colors < 0] = 0
        colors[colors > beta] = beta
    else:
        out_centers = out_centers + lr * arrows

    if (i+1) % draw_interval == 0:
        c = colors / beta
        c = c.tolist()
        draw(out_centers, a, ax, i + 1, path, colors=c)

out_masses = torch.ones(out_centers.shape[0]) / out_centers.shape[0]


with open(path + "results.txt", 'w') as f:
    f.write(str(compute_OT_error(out_masses, out_centers, sample_size, colors=colors)))
    f.write("\n")
    f.write("\n")
    for center in out_centers:
        f.write(str(float(center[0])) + " " + str(float(center[1])))
        f.write("\n")
