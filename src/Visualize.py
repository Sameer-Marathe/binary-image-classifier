import numpy as np
import matplotlib.pyplot as plt
img_array=np.load("../data/test/field.npy")
for idx, el in enumerate(img_array):
    plt.imshow(np.moveaxis(img_array[idx], 0, -1), cmap='gray')
    plt.show()
