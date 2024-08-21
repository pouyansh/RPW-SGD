import math
import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt

from rpw import RPW

dim = 2
rows_num = 20  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = 400
radius = 0.01  # radius of the disks drawn for each center
max_iters = 500  # maximum number of sinkhorn iterations
reg = 0.05  # regularization parameter in sinkhorn algorithm
max_alpha = 0.4  # maximum amount of noise in each sample
epoch_num = 200
lr = 0.5  # learning rate
k = 0.5

path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_rpw_lr" + str(lr) + "_k" + str(k) + "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


# Drawing the maintained output distribution
def draw(centers, prev, samples, epoch):
    _, ax = plt.subplots()
    for center in centers:
        circle = plt.Circle((center[0], center[1]), radius, color='k')
        ax.add_patch(circle)
    for center in prev:
        circle = plt.Circle((center[0], center[1]), radius, color='b', alpha=0.2)
        ax.add_patch(circle) 
    for center in samples:
        circle = plt.Circle((center[0], center[1]), radius, color='r', alpha=0.2)
        ax.add_patch(circle)
    plt.savefig(path + "fig" + str(epoch) + ".png")
    # plt.show()
    plt.close()


def sinkhorn(C, rpw):
    a = [(1-rpw)/C.shape[0] for _ in range(C.shape[0])]
    a.append(rpw)
    b = [(1-rpw)/C.shape[1] for _ in range(C.shape[1])]
    b.append(rpw)
    a = torch.FloatTensor(a)
    b = torch.FloatTensor(b)

    zeros_cols = torch.zeros((C.shape[0], 1))
    zeros_rows = torch.zeros((1, C.shape[1] + 1))

    C1 = torch.cat((C, zeros_cols), dim=-1)
    C2 = torch.cat((C1, zeros_rows), dim=0)

    K = torch.exp(-C2/reg)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max_iters):
        u = a / torch.matmul(K,v)
        v = b / torch.matmul(K.T,u)
    return torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))


# This method draws n samples from the real distribution contaminated with alpha fraction of noise
def sample(n):
    samples = []
    cov = [[0.01, 0], [0, 0.01]]

    alpha = random.random() * max_alpha  # amount of noise in the samples
    noise_mean_x = random.random()
    noise_mean_y = random.random()

    for _ in range(n):
        p = random.random()
        point = [-1, -1]
        if p < 1 - alpha:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                point = np.random.multivariate_normal([mean_x, mean_y], cov, 1)[0]
        else:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                point = np.random.multivariate_normal([noise_mean_x, noise_mean_y], cov, 1)[0]
        samples.append(point)
    samples = np.array(samples)
    return torch.FloatTensor(samples)


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num

# distribution to learn
mean_x = random.random()
mean_y = random.random()
print(mean_x, mean_y)

for i in range(epoch_num):
    samples = sample(sample_size)

    cost_matrix = torch.cdist(out_centers, samples, p=2)

    rpw = RPW(cost_matrix, k=k)
    print(rpw)

    plan = sinkhorn(cost_matrix, rpw)
    plan = plan[:-1, :-1]
    plan = plan.shape[0] * plan

    arrows = torch.matmul(plan, samples) - torch.matmul(torch.diag_embed(torch.sum(plan, dim=1)), out_centers)

    prev_out_centers = torch.clone(out_centers)

    out_centers = out_centers + lr * arrows

    draw(out_centers, prev_out_centers, samples, i)
