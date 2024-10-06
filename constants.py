import math

dim = 2
rows_num = 32  # the code will generate a square of rows_num x rows_num and then tries to adjust their coordinates
output_size = int(math.pow(rows_num, dim))
sample_size = output_size

epoch_num = 300
batch_size = 5
lr = 0.1  # learning rate

from_mnist = True
digit = 6
no_mass_reduce = True

draw_interval = 50

max_alpha = 0.4  # maximum amount of noise in each sample
min_alpha = 0.1

cov = [[0.01, 0], [0, 0.01]]
noise_cov = [[0.05, 0], [0, 0.05]]
margin = 0.1  # min dist of the center of the normal distribution from the boundaries of the unit squares 