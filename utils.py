import random
import numpy as np # type: ignore
import torch # type: ignore
import matplotlib.pyplot as plt # type: ignore
import keras # type: ignore
import os

max_alpha = 0.25  # maximum amount of noise in each sample
min_alpha = 0.05

cov = [[0.01, 0], [0, 0.01]]
noise_cov = [[0.05, 0], [0, 0.05]]

digit = 6
from_mnist = True

label = 7
from_cifar10 = False

# This method draws n samples from the real distribution contaminated with alpha fraction of noise
def sample(mean_x, mean_y, n):
    samples = []

    alpha = random.random() * (max_alpha - min_alpha) + min_alpha  # amount of noise in the samples

    X = None
    if from_mnist or from_cifar10:
        X = random.choice(X_train)
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
def draw(centers, masses, ax, epoch, path, radius=0.02):
    temp_masses = masses * 0.75 / torch.max(masses)
    for i in range(centers.shape[0]):
        circle = plt.Circle((centers[i][1], 1 - centers[i][0]), radius, color='k', alpha=float(temp_masses[i]))
        ax.add_patch(circle)
    plt.savefig(path + "fig" + str(epoch) + ".png")
    plt.close()


X_train = None
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