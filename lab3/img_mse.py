import numpy as np
from skimage import io
from sys import argv

if __name__ == '__main__':
    img1 = io.imread(argv[1])
    img2 = io.imread(argv[2])

    mse_loss = np.mean(np.square(img1-img2))
    print('mse_loss = ', mse_loss)
