import keras
import numpy as np
import os
import matplotlib.pyplot as plt

if os.path.exists('./data/cifar10.npy'):
    data = np.load('./data/cifar10.npy')
    data_labels = np.load('./data/cifar10_labels.npy').ravel().astype(int)
else:
    (data, data_labels), (_, _) = keras.datasets.cifar10.load_data()
    np.save('./data/cifar10.npy', data)
    np.save('./data/cifar10_labels.npy', data_labels)

print(data.shape)
plt.imshow(data[0], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
plt.gray()
plt.show()
plt.close()
