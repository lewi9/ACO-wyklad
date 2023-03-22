import numpy as np
from scipy.spatial import distance_matrix

from numba import jit

@jit(nopython=True)
def randomPerm(cities, distance):
    samples = 1000000000
    maxPath = 1e10
    for i in range(samples):
        if i%1000000 == 0: print(i)
        permutation = np.random.permutation(cities)
        path = np.sum(np.array([distance[permutation[i-1]][permutation[i]] for i in range(cities)]))

        if path < maxPath:
            maxPath = path
    print(maxPath)

points = np.genfromtxt("Points.txt", delimiter='\t')
dist_mat = distance_matrix(points, points, p=2)
randomPerm(len(points), dist_mat)
    
