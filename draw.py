import torch
from utils import draw_digits


path = "plots/run_13_ot_lr0.1_p2_bs5_cmass/"

centers = []
with open(path + "results.txt", 'r') as f:
    for line in f:
        split = line.split(" ")
        if len(split) == 2:
            centers.append([float(split[0]), float(split[1])])
draw_digits(torch.FloatTensor(centers), torch.ones(len(centers))/len(centers), 301, path)