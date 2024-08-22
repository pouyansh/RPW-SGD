import math
import torch # type: ignore
import os
import random

from rpw import RPW
from utils import sample, draw, sinkhorn

dim = 2
rows_num = 20  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = 400
radius = 0.01  # radius of the disks drawn for each center
epoch_num = 400
lr = 0.2  # learning rate
k = 1
p = 2
margin = 0.1  # min dist of the center of the normal distribution from the boundaries of the unit squares 

# Creating folder to save figures
path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_rpw_lr" + str(lr) + "_k" + str(k) + "_p" + str(p) + "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num

# distribution to learn
mean_x = random.random() * (1 - 2 * margin) + margin
mean_y = random.random() * (1 - 2 * margin) + margin
print(mean_x, mean_y)

for i in range(epoch_num):
    samples = sample(mean_x, mean_y, sample_size)

    cost_matrix = torch.cdist(out_centers, samples, p=2)
    cost_matrix = torch.pow(cost_matrix, p)

    rpw = RPW(cost_matrix, k=k, p=p)
    
    # Adding fake vertices with rpw mass on them
    a = [1 / cost_matrix.shape[0] for _ in range(cost_matrix.shape[0])]
    a.append(rpw)
    b = [1 / cost_matrix.shape[1] for _ in range(cost_matrix.shape[1])]
    b.append(rpw)
    a = torch.FloatTensor(a)
    b = torch.FloatTensor(b)

    # Adding zero columns as the distances to the fake vertices
    zeros_cols = torch.zeros((cost_matrix.shape[0], 1))
    zeros_rows = torch.zeros((1, cost_matrix.shape[1] + 1))
    cost_matrix = torch.cat((cost_matrix, zeros_cols), dim=-1)
    cost_matrix = torch.cat((cost_matrix, zeros_rows), dim=0)

    plan = sinkhorn(a, b, cost_matrix)
    plan = plan[:-1, :-1]  # removing the fake vertices
    plan = plan.shape[0] * plan

    arrows = torch.matmul(plan, samples) - torch.matmul(torch.diag_embed(torch.sum(plan, dim=1)), out_centers)

    prev_out_centers = torch.clone(out_centers)

    out_centers = out_centers + lr * arrows

    if i % 4 == 0:
        draw(out_centers, prev_out_centers, samples, i, path)
