import random
import numpy as np  # type: ignore
import torch  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import ot  # type: ignore
import math

from constants import *


def compute_rpw(masses_a, masses_b, costs, k=1, p=1, delta=0.001):
    rpw_guess = 0.5
    range = 0.5

    # Adding zero columns as the distances to the fake vertices
    zeros_cols = torch.zeros((costs.shape[0], 1))
    zeros_rows = torch.zeros((1, costs.shape[1] + 1))
    costs = torch.cat((costs, zeros_cols), dim=-1)
    costs = torch.cat((costs, zeros_rows), dim=0)

    while range > delta:
        # Adding fake vertices with rpw mass on them
        m_a = torch.cat((masses_a, torch.FloatTensor([rpw_guess])))
        m_b = torch.cat((masses_b, torch.FloatTensor([rpw_guess])))

        pot = math.pow(ot.emd2(m_a, m_b, costs, numItermax=1000000), 1 / float(p))

        range /= 2

        if pot > k * rpw_guess:
            rpw_guess += range
        else:
            rpw_guess -= range
    return rpw_guess


# This method draws n samples from the real distribution contaminated with alpha fraction of noise
def sample(n, X_train, mean_x, mean_y, clean=False):
    samples = []
    cov = [[0.01, 0], [0, 0.01]]

    alpha = (
        random.random() * (max_alpha - min_alpha) + min_alpha
    )  # amount of noise in the samples
    if clean:
        alpha = 0

    X = random.choice(X_train)
    X = X / np.sum(X)

    # random noise distribution
    noise_mean_x = random.random()
    noise_mean_y = random.random()
    noise_cov = [[0.05, 0], [0, 0.05]]

    for _ in range(n):
        p = random.random()
        point = [-1, -1]
        if p < 1 - alpha:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                if from_mnist:
                    flat = X.flatten()
                    sample_index = np.random.choice(a=flat.size, p=flat)
                    point = np.divide(
                        np.unravel_index(sample_index, X.shape), X.shape[0]
                    )
                else:
                    point = np.random.multivariate_normal([mean_x, mean_y], cov, 1)[0]
        else:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                point = np.random.multivariate_normal(
                    [noise_mean_x, noise_mean_y], noise_cov, 1
                )[0]
        samples.append(point)
    return torch.FloatTensor(samples)


# Drawing the maintained output distribution
def draw_samples(real_samples, ax, radius=0.01):
    for center in real_samples:
        circle = plt.Circle((center[1], 1 - center[0]), radius, color="r", alpha=0.03)
        ax.add_patch(circle)


# Drawing the maintained output distribution
def draw(centers, masses, ax, epoch, path, colors=[], radius=0.02):
    alpha = 0.75
    temp_masses = masses * alpha / torch.max(masses)
    for i in range(centers.shape[0]):
        circle = plt.Circle(
            (centers[i][1], 1 - centers[i][0]),
            radius,
            color="k",
            alpha=float(temp_masses[i]),
        )
        ax.add_patch(circle)
    plt.savefig(path + "fig" + str(epoch) + ".png")
    plt.close()


def draw_digits(centers, masses, epoch, path):
    image = [[0 for _ in range(28)] for _ in range(28)]
    for i in range(centers.shape[0]):
        image[math.floor(centers[i][0] * 28)][math.floor(centers[i][1] * 28)] += masses[
            i
        ]
    plt.imshow(image, cmap="gray")
    plt.savefig(path + "fig" + str(epoch) + ".png")
    plt.close()


def compute_OT_error(
    out_masses, out_centers, n, sample_num=200, p=2, data=[], mean_x=0, mean_y=0
):
    # Computing the accuracy
    total_error = 0
    for _ in range(sample_num):
        samples = sample(n, data, mean_x, mean_y, clean=True)
        cost_matrix = torch.cdist(out_centers, samples, p=2)
        cost_matrix = torch.pow(cost_matrix, 2)
        b = [1 / n for _ in range(n)]
        b = torch.FloatTensor(b)
        total_error += math.pow(float(ot.emd2(out_masses, b, cost_matrix)), 1 / p)
    avg_error = total_error / sample_num
    return avg_error


def KL(a, b):
    epsilon = 0.00001
    a += epsilon
    a = a.reshape(28 * 28) / np.sum(a)
    b = np.asarray(b.reshape(28 * 28), dtype="float64") + epsilon
    b = b / np.sum(b)

    cost_matrix = np.array(
        [
            [(i - k) ** 2 + (j - l) ** 2 for i in range(28) for j in range(28)]
            for k in range(28)
            for l in range(28)
        ]
    ) / 28 ** 2 / 2
    return np.sum(a * np.log(a / b)), math.pow(float(ot.emd2(a, b, cost_matrix)), 1 / 2)


def compute_KL_error(data, centers, masses):
    data = random.choices(data, k=500)
    image = [[0 for _ in range(28)] for _ in range(28)]
    for i in range(centers.shape[0]):
        image[math.floor(centers[i][0] * 28)][math.floor(centers[i][1] * 28)] += masses[
            i
        ]
    image = np.array(image)
    total_kl, total_ot = 0, 0
    counter = 0
    for d in data:
        kl, ot = KL(image, d)
        total_kl += kl
        total_ot += ot
        print(counter, len(data))
        counter += 1
    return total_kl / len(data), total_ot / len(data)
