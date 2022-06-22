import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote(object):
    """
    SMOTE algorithm implementation.
    Parameters
    ----------
    samples : {array-like}, shape = [n_samples, n_features]
        Training vector, where n_samples in the number of samples and
        n_features is the number of features.
    N : int, optional (default = 50)
        Parameter N, the percentage of n_samples, affects the amount of final
        synthetic samples，which calculated by floor(N/100)*T.
    k : int, optional (default = 5)
        Specify the number for NearestNeighbors algorithms.
    r : int, optional (default = 2)
        Parameter for sklearn.neighbors.NearestNeighbors API.When r = 1, this
        is equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for r = 2. For arbitrary p, minkowski_distance (l_r) is used.
    Examples
    --------
      >>> samples = np.array([[3,1,2], [4,3,3], [1,3,4],
                              [3,3,2], [2,2,1], [1,4,3]])
      >>> smote = Smote(N=200)
      >>> synthetic_points = smote.fit(samples)
      >>> print(synthetic_points)
      [[3.31266454 1.62532908 2.31266454]
       [2.4178394  1.5821606  2.5821606 ]
       [3.354422   2.677211   2.354422  ]
       [2.4169074  2.2084537  1.4169074 ]
       [1.86018171 2.13981829 3.13981829]
       [3.68440949 3.         3.10519684]
       [2.22247957 3.         2.77752043]
       [2.3339721  2.3339721  1.3339721 ]
       [3.31504371 2.65752185 2.31504371]
       [2.54247589 2.54247589 1.54247589]
       [1.33577795 3.83211103 2.83211103]
       [3.85206355 3.04931215 3.        ]]
    """

    def __init__(self, N=50, k=5, r=2):
        # self.N, self.k, self.r, self.newindex
        self.N = N
        self.k = k
        self.r = r
        self.newindex = 0

    def fit(self, samples):
        self.samples = samples
        self.T, self.numattrs = self.samples.shape

        if (self.N < 100):
            np.random.shuffle(self.samples)
            self.T = int(self.N * self.T / 100)
            self.samples = self.samples[0:self.T, :]
            self.N = 100

        if (self.T <= self.k):
            self.k = self.T - 1

        N = int(self.N / 100)
        self.synthetic = np.zeros((self.T * N, self.numattrs))

        neighbors = NearestNeighbors(n_neighbors=self.k + 1,
                                     algorithm='ball_tree',
                                     p=self.r).fit(self.samples)

        for i in range(len(self.samples)):
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)),
                                           return_distance=False)[0][1:]

            self.__populate(N, i, nnarray)

        return self.synthetic

    def __populate(self, N, i, nnarray):
        # 按照倍数N做循环
        for j in range(N):
            nn = random.randint(0, self.k - 1)

            diff = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.uniform(0, 1)
            self.synthetic[self.newindex] = self.samples[i] + gap * diff

            self.newindex += 1
