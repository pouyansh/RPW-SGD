import random
import numpy as np # type: ignore
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore
import keras # type: ignore
import os
import ot # type: ignore
import math

max_alpha = 0.4  # maximum amount of noise in each sample
min_alpha = 0

cov = [[0.01, 0], [0, 0.01]]
noise_cov = [[0.05, 0], [0, 0.05]]
margin = 0.1  # min dist of the center of the normal distribution from the boundaries of the unit squares 

digit = 6
from_mnist = False

label = 7
from_cifar10 = True
moving = True


def compute_rpw(masses_a, masses_b, costs, k=1, p=1, delta=0.00001):
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

        pot = math.pow(ot.emd2(m_a, m_b, costs, numItermax=1000000), 1/float(p))

        range /= 2

        if pot > k * rpw_guess:
            rpw_guess += range
        else:
            rpw_guess -= range
    return rpw_guess


# This method draws n samples from the real distribution contaminated with alpha fraction of noise
def sample(n, clean=False, beta=0.3):
    samples = []

    if clean:
        alpha = 0
    else:
        alpha = random.random() * (max_alpha - min_alpha) + min_alpha  # amount of noise in the samples

    X = None
    if from_mnist or from_cifar10:
        X = random.choice(X_train)
        if from_cifar10:
            points = [[X[int(i / X.shape[1])][i % X.shape[1]][0], 
                       X[int(i / X.shape[1])][i % X.shape[1]][1], 
                       X[int(i / X.shape[1])][i % X.shape[1]][2]] 
                      for i in range(X.shape[0] * X.shape[1])]
            return torch.FloatTensor(points) * beta / 256
        X = X / np.sum(X)

    # random noise distribution
    noise_mean_x = random.random()
    noise_mean_y = random.random()

    for _ in range(n):
        p = random.random()
        point = [-1, -1]
        if p < 1 - alpha:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                if from_mnist:
                    flat = X.flatten()
                    sample_index = np.random.choice(a=flat.size, p=flat)
                    point = np.divide(np.unravel_index(sample_index, X.shape), X.shape[0])
                elif from_cifar10:
                    flat = X.flatten()
                else:
                    point = np.random.multivariate_normal([mean_x, mean_y], cov, 1)[0]
        else:
            while point[0] < 0 or point[0] > 1 or point[1] < 0 or point[1] > 1:
                point = np.random.multivariate_normal([noise_mean_x, noise_mean_y], noise_cov, 1)[0]
        samples.append(point)
    samples = np.array(samples)
    return torch.FloatTensor(samples)


# Drawing the maintained output distribution
def draw_samples(real_samples, ax, radius=0.01):
    for center in real_samples:
        circle = plt.Circle((center[1], 1 - center[0]), radius, color='r', alpha=0.03)
        ax.add_patch(circle)


# Drawing the maintained output distribution
def draw(centers, masses, ax, epoch, path, colors=[], radius=0.02):
    alpha = 0.75
    if from_cifar10:
        alpha = 1
    temp_masses = masses * alpha / torch.max(masses)
    for i in range(centers.shape[0]):
        if not from_cifar10:
            circle = plt.Circle((centers[i][1], 1 - centers[i][0]), radius, color='k', alpha=float(temp_masses[i]))
        else:
            circle = plt.Circle((centers[i][1], 1 - centers[i][0]), radius, 
                                color=(colors[i][0], colors[i][1], colors[i][2]), alpha=float(temp_masses[i]))
        ax.add_patch(circle)
    plt.savefig(path + "fig" + str(epoch) + ".png")
    plt.close()


def compute_OT_error(out_masses, out_centers, n, sample_num=200, colors=[], p=2):
    # Computing the accuracy
    total_error = 0
    for _ in range(sample_num):
        samples = sample(n, clean=True)
        if from_cifar10:
            if not moving:
                total_error += KL(colors, samples)
            else:
                rows_num = 32
                centers = torch.FloatTensor([[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(rows_num ** 2)]) / rows_num
                samples = torch.cat((centers, samples), dim=1)
                points = torch.cat((out_centers, colors), dim=1)

                cost_matrix = torch.cdist(points, samples, p=2)
                cost_matrix = torch.pow(cost_matrix, p)
                b = torch.ones(samples.shape[0]) / samples.shape[0]
                total_error += math.pow(float(ot.emd2(out_masses, b, cost_matrix)), 1/p)
        else:
            cost_matrix = torch.cdist(out_centers, samples, p=2)
            cost_matrix = torch.pow(cost_matrix, 2)
            b = [1 / n for _ in range(n)]
            b = torch.FloatTensor(b)
            total_error += math.pow(float(ot.emd2(out_masses, b, cost_matrix)), 1/p)
    avg_error = total_error / sample_num
    return avg_error


def KL(a, b):
    epsilon = 0.00001
    a = np.asarray(torch.flatten(a).tolist()) + epsilon
    a = a / np.sum(a)
    b = np.asarray(torch.flatten(b).tolist()) + epsilon
    b = b / np.sum(b)

    return np.sum(a * np.log(a / b))


X_train = None
# distribution to learn
mean_x = random.random() * (1 - 2 * margin) + margin
mean_y = random.random() * (1 - 2 * margin) + margin
if from_mnist:
    (X_train, labels), (_, _) = keras.datasets.mnist.load_data()
    train_filter = np.where((labels == digit))
    X_train = X_train[train_filter]
elif from_cifar10:
    if os.path.exists('./data/cifar10.npy'):
        X_train = np.load('./data/cifar10.npy')
        labels = np.load('./data/cifar10_labels.npy').ravel().astype(int)
    else:
        (X_train, labels), (_, _) = keras.datasets.cifar10.load_data()
        np.save('./data/cifar10.npy', X_train)
        np.save('./data/cifar10_labels.npy', labels)
    train_filter = np.where((labels == label))
    X_train = X_train[train_filter]