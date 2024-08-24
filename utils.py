import random
import numpy as np # type: ignore
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore

max_alpha = 0.3  # maximum amount of noise in each sample
min_alpha = 0.2

# This method draws n samples from the real distribution contaminated with alpha fraction of noise
def sample(mean_x, mean_y, n):
    samples = []
    cov = [[0.01, 0], [0, 0.01]]

    alpha = random.random() * (max_alpha - min_alpha) + min_alpha  # amount of noise in the samples

    # random noise distribution
    noise_mean_x = random.random()
    noise_mean_y = random.random()
    noise_cov = [[0.05, 0], [0, 0.05]]

    for _ in range(n):
        p = random.random()
        point = [-1, -1]
        if p < 1 - alpha:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                point = np.random.multivariate_normal([mean_x, mean_y], cov, 1)[0]
        else:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                point = np.random.multivariate_normal([noise_mean_x, noise_mean_y], noise_cov, 1)[0]
        samples.append(point)
    samples = np.array(samples)
    return torch.FloatTensor(samples)


def sinkhorn(a, b, C, reg=0.05, max_iters=400):
    K = torch.exp(-C/reg)
    u = torch.clone(a)
    v = torch.clone(b)
    for _ in range(max_iters):
        u = a / torch.matmul(K,v)
        v = b / torch.matmul(K.T,u)
    return torch.matmul(torch.diag_embed(u), torch.matmul(K, torch.diag_embed(v)))


# Drawing the maintained output distribution
def draw(centers, masses, prev_centers, real_samples, epoch, path, radius=0.03):
    _, ax = plt.subplots()
    temp_masses = masses / torch.max(masses)
    for i in range(centers.shape[0]):
        circle = plt.Circle((centers[i][0], centers[i][1]), radius, color='k', alpha=float(temp_masses[i]))
        ax.add_patch(circle)
    # for center in prev_centers:
    #     circle = plt.Circle((center[0], center[1]), radius, color='b', alpha=0.2)
    #     ax.add_patch(circle) 
    for center in real_samples:
        circle = plt.Circle((center[0], center[1]), 0.01, color='r', alpha=0.2)
        ax.add_patch(circle)
    plt.savefig(path + "fig" + str(epoch) + ".png")
    plt.close()