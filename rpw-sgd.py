import torch # type: ignore
import os
import sys
import ot # type: ignore
import matplotlib.pyplot as plt # type: ignore
from tqdm import tqdm # type: ignore

from utils import *
from constants import *

# method = rpw / ot
if len(sys.argv) >= 2:  
    method = sys.argv[1]
else:
    method = "rpw"

# parameter p in (p,k)-RPW or p-Wasserstein distance
if len(sys.argv) >= 3:  
    p = int(sys.argv[2])
else:
    p = 2

# parameter k in (p,k)-RPW
if len(sys.argv) >= 4:  
    k = float(sys.argv[3])
else:
    k = 1


# Creating folder to save figures
plots_path = "plots/"
if not os.path.exists(plots_path):
    os.makedirs(plots_path, exist_ok=True)

path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_" + method + "_lr" + str(lr) + "_p" + str(p) + "_bs" + str(batch_size)
if method == "rpw":
    path += "_k" + str(k)
if no_mass_reduce:
    path += "_cmass"
path += "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num
const_centers = torch.FloatTensor(centers) / rows_num
out_masses = torch.ones(output_size) / output_size

_, ax = plt.subplots()
draw(out_centers, out_masses, ax, 0, path)

for i in tqdm(range(epoch_num)):
    plt.close()
    _, ax = plt.subplots()

    # initialization
    arrows = torch.zeros((output_size, dim))
    plans = torch.zeros((output_size, sample_size))

    # epochs
    for _ in range(batch_size):
        samples = sample(sample_size)
        # if (i+1) % draw_interval == 0:
        #     draw_samples(samples, ax)

        points = out_centers

        # Computing the cost matrix between the maintained centers and samples
        cost_matrix = torch.cdist(points, samples, p=2)
        cost_matrix = torch.pow(cost_matrix, p)

        # Computing (p, k)-rpw
        rpw = 0
        if method == "rpw":
            m = samples.shape[0]
            rpw = compute_rpw(out_masses, torch.ones(m) / m, cost_matrix, k=k, p=p)
        
            # Adding fake vertices with rpw mass on them
            a = torch.cat((out_masses, torch.FloatTensor([rpw])))
            m = samples.shape[0]
            b = torch.ones(m + 1) / m
            b[-1] = rpw

            # Adding zero columns as the distances to the fake vertices
            zeros_cols = torch.zeros((cost_matrix.shape[0], 1))
            zeros_rows = torch.zeros((1, cost_matrix.shape[1] + 1))
            cost_matrix_rpw = torch.cat((cost_matrix, zeros_cols), dim=-1)
            cost_matrix_rpw = torch.cat((cost_matrix_rpw, zeros_rows), dim=0)

            plan = ot.emd(a, b, cost_matrix_rpw)
            plan = plan[:-1, :-1]  # removing the fake vertices

            if no_mass_reduce:  
            # in this case, we remove the part of the mass of the sample that is not transported and normalize
            # then again we compute the transport plan
                b = plan.sum(dim=0)
                b = b / b.sum()
                plan = ot.emd(out_masses, b, cost_matrix)

            arrows = arrows + torch.matmul(plan, samples) - torch.matmul(torch.diag_embed(torch.sum(plan, dim=1)), out_centers)
        else: 
            a = torch.ones(out_centers.shape[0]) / out_centers.shape[0]
            b = torch.ones(points.shape[0]) / points.shape[0]

            plan = ot.emd(a, b, cost_matrix)
            plan = plan.shape[0] * plan

            arrows = arrows + torch.matmul(plan, samples) - out_centers

        plans = plans + plan
        
    # Averaging the plans and arrows computed for each sample
    plans = plans / batch_size
    arrows = arrows / batch_size

    if method != "rpw":
        out_centers = out_centers + lr * arrows
    else:
        out_centers = out_centers + lr * torch.div(arrows.T, out_masses).T

    if not no_mass_reduce:
        out_masses = torch.sum(plans, 1)
        out_masses[torch.logical_and(out_masses>=0, out_masses<=1e-9)] = 1e-9
        out_masses = out_masses / torch.sum(out_masses)

    if (i+1) % draw_interval == 0:
        if from_mnist:
            draw_digits(out_centers, out_masses, i+1, path)
        else:
            draw(out_centers, out_masses, ax, i + 1, path)


with open(path + "results.txt", 'w') as f:
    f.write(str(compute_OT_error(out_masses, out_centers, sample_size, p=p)))
    f.write("\n")
    f.write("\n")
    for center in out_centers:
        f.write(str(float(center[0])) + " " + str(float(center[1])))
        f.write("\n")