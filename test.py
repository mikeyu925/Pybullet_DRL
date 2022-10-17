import numpy as np
import math

if __name__ == '__main__':
    k = 0.037
    arctan = np.arctan(k)
    theta = np.arctan(k) * 2 / np.pi * 90
    print(arctan)
    print(theta)