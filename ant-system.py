import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import random
from scipy.spatial import distance_matrix

from numba import jit
from numba import int64, float64, types, typed
from numba.experimental import jitclass

spec = [
    ('startCity', int64),
    ('tabooList', int64[:]),     
    ('currentCity', int64), 
]

@jitclass(spec)
class Ant:
    def __init__(self, startCity):
        self.startCity = startCity
        self.tabooList = np.array([self.startCity])
        self.currentCity = self.startCity

@jit(nopython=True)
def rand_choice_nb(arr, prob):
    """
    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
    :return: A random sample from the given array with a given probability.
    """
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]

@jit(nopython=True, parallel=True)
def travelThrough(cities, distance, localPheromone, vaporizeFactor, maxPath, bestPath, q, alpha, beta, pheromoneZero, antsInCity):
    ants = [Ant(i) for i in range(cities) for j in range(antsInCity)]
   
    for i in range(cities-1):
        #newLocalPheromone = np.copy(localPheromone)*(1-vaporizeFactor)
        newLocalPheromone = localPheromone * (1 - vaporizeFactor)
        for ant in ants:
            #availableCities = [x for x in np.arange(cities) if x not in ant.tabooList]
            availableCities = [x for x in np.arange(cities) if x not in ant.tabooList]
            availableCities = np.array(availableCities)
            ## calculate probability to choose the city
            
            sumTotal = 0
            probabilityVector = []
            for city in availableCities:
                sumTotal += localPheromone[min(ant.currentCity, city)][max(ant.currentCity, city)] ** alpha / (distance[ant.currentCity][city] ** beta )
            for city in availableCities:
                probabilityVector.append(localPheromone[min(ant.currentCity, city)][max(ant.currentCity, city)] ** alpha / (distance[ant.currentCity][city] ** beta) / sumTotal)

            probabilityVector = np.array(probabilityVector)
            ## if exploration pick with probability, else pick the best
            if np.random.sample() < q:
                destinationCity = rand_choice_nb(availableCities, probabilityVector)
                 #destinationCity = np.random.choice(availableCities, 1, p=probabilityVector)
            else:
                #destinationCity = availableCities[np.argmax(probabilityVector)]
                destinationCity = availableCities[list(probabilityVector).index(np.amax(probabilityVector))]
            ant.currentCity = destinationCity
            ant.tabooList = np.append(ant.tabooList, np.int64(destinationCity))
            newLocalPheromone[min(ant.tabooList[-2],ant.tabooList[-1])][max(ant.tabooList[-2],ant.tabooList[-1])] += vaporizeFactor*pheromoneZero
        #localPheromone = np.copy(newLocalPheromone)
    
    antsOnRoad = 0
    for ant in ants:

            #path = 0
            ## calculate travel parameter

            #for i in range(cities):
            #    index = i-1
            #    path += distance[ant.tabooList[index]][ant.tabooList[i]]
            path = np.sum(np.array([distance[ant.tabooList[i-1]][ant.tabooList[i]] for i in range(cities)]))
            if path < maxPath:
                maxPath = path
                bestPath = ant.tabooList.copy()
                antsOnRoad = 1
            elif abs(path-maxPath) <= 0.01:
                antsOnRoad += 1
    output = np.array([np.float64(x) for x in bestPath])
    output = np.append(output, np.array(maxPath, dtype="float64"))
    output = np.append(output, np.array(antsOnRoad, dtype="float64"))
    return output




#points = np.random.sample((100,2))*100
#np.savetxt("Points.txt", points, delimiter='\t')

points = np.genfromtxt("Points.txt", delimiter='\t')

cities = len(points)

dist_mat = distance_matrix(points, points, p=2)

maxCycle = 200
antsInCity = 20

alpha = pheromoneWeight = 1
beta = cityVisibility = 0
vaporizeFactor = 0.2
q = exploitationOrExploration = 0.4
pheromoneZero = 1e-3       

distance = dist_mat

final1 = []

## Algorithm
for lap in range(6):
    beta = lap+2
    result = []
    onPath = []
    maxPath = float("inf")
    bestPath = np.array([], dtype="int64")
    ## Build pheromone matrix with initial value
    pheromone = np.ones((cities, cities)) * pheromoneZero
    for k in tqdm.tqdm(range(maxCycle)):

        localPheromone = np.copy(pheromone)    

        ## travel through the cities
        ## check that all roads are the same and add pheromone
        
        output = travelThrough(cities, distance, localPheromone, vaporizeFactor, maxPath, bestPath, q, alpha, beta, pheromoneZero, antsInCity)
        antsOnRoad = int(output[-1])
        maxPath = output[-2]
        bestPath =  np.array(output[:-2], dtype = "int64")
            
        ## update pheromones for pareto path
        pheromone *= (1-vaporizeFactor)
        for i in range(cities):
            index = i-1
            pheromone[min(bestPath[index],bestPath[i])][max(bestPath[index],bestPath[i])] += vaporizeFactor/maxPath
            
        result.append(maxPath)
        onPath.append(antsOnRoad)
        try:
            if result[-1] != result[-2]:
                print(maxPath)
        except:
            pass
    final1.append(result)
        
print(maxPath)    
fig, ax = plt.subplots(1,2,figsize=(20,10))

ax[0].set_title("Minimal path distance")
ax[0].set_ylabel("distance")
ax[0].set_xlabel("iteration")

g = 1;
for i in range(len(final1)):
    g -= 0.33
    if g < 0:
        g = 0
    X = np.linspace(1,len(final1[i]),len(final1[i]))
    ax[0].plot(X,final1[i], c =(0.1*i, g, 0.2*i))

ax[1].set_title("Optimum path")
ax[1].set_ylabel("Y")
ax[1].set_xlabel("X")
ax[1].scatter([x[0] for x in points],[x[1] for x in points])
for i in range(len(bestPath)):
    x1 = points[bestPath[i]][0]
    y1 = points[bestPath[i]][1]
    x2 = points[bestPath[i-1]][0]
    y2 = points[bestPath[i-1]][1]
    ax[1].plot([x1,x2],[y1,y2], c = "green")

plt.savefig(f"antsOptimum")
plt.close()

