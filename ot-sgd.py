import math
import torch # type: ignore
import os
import random
import ot # type: ignore
import matplotlib.pyplot as plt # type: ignore

from utils import sample, draw, draw_samples, compute_OT_error

dim = 2
rows_num = 30  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = 900
epoch_num = 300
lr = 0.1  # learning rate
margin = 0.1  # min dist of the center of the normal distribution from the boundaries of the unit squares 
p = 2
batch_size = 5

path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_ot_lr" + str(lr) + "_p" + str(p) + "_bs" + str(batch_size) + "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num

# distribution to learn
mean_x = random.random() * (1 - 2 * margin) + margin
mean_y = random.random() * (1 - 2 * margin) + margin

_, ax = plt.subplots()
draw(out_centers, torch.ones(output_size) / output_size, ax, 0, path)

for i in range(epoch_num):
    plt.close()
    _, ax = plt.subplots()
    arrows = torch.zeros((output_size, dim))
    for _ in range(batch_size):
        samples = sample(mean_x, mean_y, sample_size)
        if i % 3 == 0:
            draw_samples(samples, ax)

        cost_matrix = torch.cdist(out_centers, samples, p=2)
        cost_matrix = torch.pow(cost_matrix, p)

        a = torch.ones(out_centers.shape[0]) / out_centers.shape[0]
        b = torch.ones(sample_size) / sample_size

        plan = ot.emd(a, b, cost_matrix)
        plan = plan.shape[0] * plan

        arrows = arrows + torch.matmul(plan, samples) - out_centers

    # Averaging the arrows computed for each sample
    arrows = arrows / batch_size

    out_centers = out_centers + lr * arrows

    if i % 3 == 0:
        draw(out_centers, a, ax, i + 1, path)

out_masses = torch.ones(out_centers.shape[0]) / out_centers.shape[0]
with open(path + "results.txt", 'w') as f:
    f.write(str(compute_OT_error(out_masses, out_centers, mean_x, mean_y, sample_size)))
    f.write("\n")
    f.write(out_centers)
    f.write("\n")
