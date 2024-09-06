import keras # type: ignore
import numpy as np # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
import torch # type: ignore
import ot # type: ignore
import math
from tqdm import tqdm # type: ignore

from utils import compute_rpw, sample

label = 7
betas = [0.3, 1, 3]
ks = [0.001, 0.01, 0.1, 1]
p = 1
rows_num = 32

# Creating folder to save figures
path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_rpw_cifar_test" + str(label) + "_p" + str(p) + "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(rows_num ** 2)]
const_centers = torch.FloatTensor(centers) / rows_num

for i in tqdm(range(10)):
    plt.figure(dpi=600)
    fig, axs = plt.subplots(len(betas), 3 * len(ks) - 1, figsize=(3 * len(ks) - 1, len(betas)), sharex=True, sharey=False, constrained_layout=True)
    for ax in axs.ravel():
        ax.set_axis_off()
    plt.subplots_adjust(left=0.05,
                        bottom=0.05, 
                        right=0.05, 
                        top=0.05, 
                        wspace=0.01, 
                        hspace=0.01)
    X1 = sample(1, beta=1)
    X2 = sample(1, beta=1)

    counter = 1
    for beta in betas:
        for k in ks:
            X1_c = torch.clone(X1)
            X2_c = torch.clone(X2)
            samples1 = torch.cat((const_centers, X1_c * beta), dim=1)
            samples2 = torch.cat((const_centers, X2_c * beta), dim=1)

            cost_matrix = torch.cdist(samples1, samples2, p=2)
            diam = math.sqrt(2 + 3 * beta ** 2)
            cost_matrix = cost_matrix / diam
            cost_matrix = torch.pow(cost_matrix, p)
            masses = torch.ones(samples1.shape[0]) / samples1.shape[0]

            rpw = compute_rpw(masses, masses, cost_matrix, k=k, p=p)

            # Adding fake vertices with rpw mass on them
            a = torch.cat((masses, torch.FloatTensor([rpw])))
            b = torch.cat((masses, torch.FloatTensor([rpw])))

            # Adding zero columns as the distances to the fake vertices
            zeros_cols = torch.zeros((cost_matrix.shape[0], 1))
            zeros_rows = torch.zeros((1, cost_matrix.shape[1] + 1))
            cost_matrix_rpw = torch.cat((cost_matrix, zeros_cols), dim=-1)
            cost_matrix_rpw = torch.cat((cost_matrix_rpw, zeros_rows), dim=0)

            plan = ot.emd(a, b, cost_matrix_rpw, numItermax=1000000)
            plan = plan[:-1, :-1]  # removing the fake vertices

            masses_a = torch.sum(plan, dim=1)
            masses_a = torch.reshape(masses_a / torch.max(masses_a), (32, 32)).unsqueeze_(-1).expand(32,32,3)
            X1_c = X1_c.reshape(32, 32, 3) * masses_a
            masses_b = torch.sum(plan, dim=0)
            masses_b = torch.reshape(masses_b / torch.max(masses_b), (32, 32)).unsqueeze_(-1).expand(32,32,3)
            X2_c = X2_c.reshape(32, 32, 3) * masses_b

            plt.subplot(len(betas), 3 * len(ks) - 1, counter)
            counter += 1
            plt.imshow(X1_c)
            plt.subplot(len(betas), 3 * len(ks) - 1, counter)
            counter += 2
            plt.imshow(X2_c)
        counter -= 1
        

    plt.axis('off')
    plt.savefig(path + "fig" + str(i) + ".png", bbox_inches='tight')
    plt.close()
