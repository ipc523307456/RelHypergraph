import numpy as np

def l1_distance(matrix1, matrix2):
    return np.sum(np.abs(matrix1 - matrix2))