import math
import torch
import numpy as np
import random

import matplotlib.pyplot as plt

dim = 2
rows_num = 16  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
radius = 0.01  # radius of the disks drawn for each center
max_iters = 400  # maximum number of sinkhorn iterations
reg = 0.1  # regularization parameter in sinkhorn algorithm
max_alpha = 0.3  # maximum amount of noise in each sample
epoch_num = 200


# Drawing the maintained output distribution
def draw(centers):
    _, ax = plt.subplots()
    for center in centers:
        circle = plt.Circle((center[0], center[1]), radius, color='k')
        ax.add_patch(circle)
    plt.show()
    plt.close()


def sinkhorn(a, b, C):
    K = torch.exp(-C/reg)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(max_iters):
        u = a / torch.matmul(K,v)
        v = b / torch.matmul(K.T,u)
    return torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))


def compute_euclidean_cost_matrix(A, B):
    matrix = [[0 for _ in range(A.shape[0])] for _ in range(B.shape[0])]
    for i in range(A.shape[0]):
        for j in range(B.shape[0]):
            matrix[i][j] = math.sqrt((A[i][0] - B[j][0]) ** 2 + (A[i][1] - B[j][1]) ** 2)
    return torch.tensor(matrix)


def sample(n):
    samples = []
    cov = [[0.01, 0], [0, 0.01]]
    alpha = random.random() * max_alpha
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
    return torch.FloatTensor(samples)


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num

# distribution to learn
mean_x = random.random()
mean_y = random.random()
print(mean_x, mean_y)

for i in range(epoch_num):
    samples = sample(output_size)
    draw(samples)

    cost_matrix = compute_euclidean_cost_matrix(out_centers, samples)


    draw(out_centers)
