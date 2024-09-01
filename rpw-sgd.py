import math
import torch # type: ignore
import os
import random
import ot # type: ignore
import matplotlib.pyplot as plt # type: ignore

from rpw import RPW
from utils import sample, draw, draw_samples, compute_OT_error

dim = 2
rows_num = 32  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = output_size
epoch_num = 100
lr = 0.5  # learning rate
k = 1
p = 2
batch_size = 5
no_mass_reduce = True
draw_interval = 1
from_cifar10 = True
beta = 0.5  # RGB significance in cifar10

# Creating folder to save figures
path = "plots/run_"
index = 0
with open("plots/index.txt", 'r') as f:
    index = int(f.read())
with open("plots/index.txt", 'w') as f:
    f.write(str(index + 1))
path += str(index) + "_rpw_lr" + str(lr) + "_k" + str(k) + "_p" + str(p) + "_bs" + str(batch_size)
if no_mass_reduce:
    path += "_cmass"
if from_cifar10:
    path += "_cifar10"
path += "/"
if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)


def compute_rpw(masses_a, masses_b, costs, delta=0.005):
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

        pot = math.pow(ot.emd2(m_a, m_b, costs), 1/float(p))

        range /= 2

        if pot > k * rpw_guess:
            rpw_guess += range
        else:
            rpw_guess -= range
    return rpw_guess


# initialization
centers = [[int(i / rows_num) + 0.5, i % rows_num + 0.5] for i in range(output_size)]
out_centers = torch.FloatTensor(centers) / rows_num
out_masses = torch.ones(output_size) / output_size
colors = torch.FloatTensor([[beta, beta, beta] for _ in range(output_size)]) / 2

_, ax = plt.subplots()
c = colors / beta
c = c.tolist()
draw(out_centers, out_masses, ax, 0, path, colors=c)

for i in range(epoch_num):
    plt.close()
    _, ax = plt.subplots()
    if from_cifar10:
        arrows = torch.zeros((output_size, 3))
    else:
        arrows = torch.zeros((output_size, dim))
    plans = torch.zeros((output_size, sample_size))
    for _ in range(batch_size):
        samples = sample(sample_size, beta=beta)
        if (i+1) % draw_interval == 0:
            draw_samples(samples, ax)

        if from_cifar10:
            samples = torch.cat((out_centers, samples), dim=1)
            points = torch.cat((out_centers, colors), dim=1)
        else:
            points = out_centers
        cost_matrix = torch.cdist(points, samples, p=2)
        cost_matrix = torch.pow(cost_matrix, p)

        # rpw = RPW(out_masses.tolist(), cost_matrix, k=k, p=p)
        rpw = compute_rpw(out_masses, torch.FloatTensor([1 / samples.shape[0] for _ in range(samples.shape[0])]), cost_matrix)
        # print(rpw, rpw_binary)
        
        # Adding fake vertices with rpw mass on them
        a = torch.cat((out_masses, torch.FloatTensor([rpw])))
        b = [1 / samples.shape[0] for _ in range(samples.shape[0])]
        b.append(rpw)
        b = torch.FloatTensor(b)

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

        plans = plans + plan

        if from_cifar10:
            arrows = arrows + torch.matmul(plan, samples)[:, [2, 3, 4]] - torch.matmul(torch.diag_embed(torch.sum(plan, dim=1)), colors)
        else:
            arrows = arrows + torch.matmul(plan, samples) - torch.matmul(torch.diag_embed(torch.sum(plan, dim=1)), out_centers)
        
    # Averaging the plans and arrows computed for each sample
    plans = plans / batch_size
    arrows = arrows / batch_size

    if from_cifar10:
        colors = colors + lr * torch.div(arrows.T, out_masses).T
    else:
        out_centers = out_centers + lr * torch.div(arrows.T, out_masses).T

    if not no_mass_reduce:
        out_masses = torch.sum(plans, 1)
        out_masses[torch.logical_and(out_masses>=0, out_masses<=1e-9)] = 1e-9
        out_masses = out_masses / torch.sum(out_masses)

    if (i+1) % draw_interval == 0:
        c = colors / beta
        c = c.tolist()
        draw(out_centers, out_masses, ax, i + 1, path, colors=c)

with open(path + "results.txt", 'w') as f:
    if not from_cifar10:
        f.write(str(compute_OT_error(out_masses, out_centers, sample_size)))
        f.write("\n")
        f.write("\n")
    for center in out_centers:
        f.write(str(float(center[0])) + " " + str(float(center[1])))
        f.write("\n")