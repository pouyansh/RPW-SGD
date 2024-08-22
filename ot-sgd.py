import math
import torch # type: ignore
import os
import random

from utils import sample, sinkhorn, draw

dim = 2
rows_num = 20  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = 500
radius = 0.01  # radius of the disks drawn for each center
max_iters = 500  # maximum number of sinkhorn iterations
reg = 0.05  # regularization parameter in sinkhorn algorithm
epoch_num = 400
lr = 0.1  # learning rate
margin = 0.1  # min dist of the center of the normal distribution from the boundaries of the unit squares 
p = 1

path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_ot_lr" + str(lr) + "_p" + str(p) + "/"
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

    a = torch.ones(out_centers.shape[0]) / out_centers.shape[0]
    b = torch.ones(sample_size) / sample_size

    plan = sinkhorn(a, b, cost_matrix)
    plan = plan.shape[0] * plan

    arrows = torch.matmul(plan, samples) - out_centers

    prev_out_centers = torch.clone(out_centers)

    out_centers = out_centers + lr * arrows

    if i % 4 == 0:
        draw(out_centers, prev_out_centers, samples, i, path)
