import torch  # type: ignore
import numpy as np  # type: ignore
import keras  # type: ignore
from utils import draw_digits, compute_KL_error


path = "plots/run_13_ot_lr0.1_p2_bs5_cmass/"

digit = 7

centers = []
with open(path + "results.txt", "r") as f:
    for line in f:
        split = line.split(" ")
        if len(split) == 2:
            centers.append([float(split[0]), float(split[1])])

(X_train, labels), (_, _) = keras.datasets.mnist.load_data()
train_filter = np.where((labels == digit))
X = X_train[train_filter]

# draw_digits(torch.FloatTensor(centers), torch.ones(len(centers))/len(centers), 301, path)
with open(path + "KL" + str(digit) + ".txt", "w") as f:
    f.write(str(compute_KL_error(X, np.array(centers), torch.ones(len(centers)) / len(centers))))
