import math
import torch # type: ignore
import os
import random
import ot # type: ignore
import matplotlib.pyplot as plt # type: ignore

from rpw import RPW
from utils import sample, draw, draw_samples

dim = 2
rows_num = 30  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = 900
epoch_num = 80
lr = 0.1  # learning rate
k = 1
p = 1
margin = 0.1  # min dist of the center of the normal distribution from the boundaries of the unit squares 
batch_size = 5
alpha = 0.15  # the parameter in alpha-partial

# Creating folder to save figures
path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_pot_lr" + str(lr) + "_alpha" + str(alpha) + "_p" + str(p) + "_bs" + str(batch_size) + "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num
out_masses = torch.ones(output_size) / output_size

# distribution to learn
mean_x = random.random() * (1 - 2 * margin) + margin
mean_y = random.random() * (1 - 2 * margin) + margin

_, ax = plt.subplots()
draw(out_centers, out_masses, ax, 0, path)

for i in range(epoch_num):
    plt.cla()
    _, ax = plt.subplots()
    arrows = torch.zeros((output_size, dim))
    plans = torch.zeros((output_size, sample_size))
    for _ in range(batch_size):
        samples = sample(mean_x, mean_y, sample_size)
        draw_samples(samples, ax)

        cost_matrix = torch.cdist(out_centers, samples, p=2)
        cost_matrix = torch.pow(cost_matrix, p)
        
        # Adding fake vertices with rpw mass on them
        a = torch.cat((out_masses, torch.FloatTensor([alpha])))
        b = [1 / sample_size for _ in range(sample_size)]
        b.append(alpha)
        b = torch.FloatTensor(b)

        # Adding zero columns as the distances to the fake vertices
        zeros_cols = torch.zeros((cost_matrix.shape[0], 1))
        zeros_rows = torch.zeros((1, cost_matrix.shape[1] + 1))
        cost_matrix = torch.cat((cost_matrix, zeros_cols), dim=-1)
        cost_matrix = torch.cat((cost_matrix, zeros_rows), dim=0)

        # plan = sinkhorn(a, b, cost_matrix)
        plan = ot.emd(a, b, cost_matrix)
        plan = plan[:-1, :-1]  # removing the fake vertices
        plans = plans + plan

        arrows = arrows + torch.matmul(plan, samples) - torch.matmul(torch.diag_embed(torch.sum(plan, dim=1)), out_centers)
        
    # Averaging the plans and arrows computed for each sample
    plans = plans / batch_size
    arrows = arrows / batch_size

    out_centers = out_centers + lr * torch.div(arrows.T, out_masses).T

    out_masses = torch.sum(plans, 1)
    out_masses[torch.logical_and(out_masses>=0, out_masses<=1e-9)] = 1e-9
    out_masses = out_masses / torch.sum(out_masses)

    if i % 1 == 0:
        draw(out_centers, out_masses, ax, i + 1, path)
