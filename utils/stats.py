import numpy as np

__author__ = "Brandon Wood"

"""
Welford method of calculating a running mean and variance
reference: https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
"""


class Stats(object):
    def __init__(self, a_k=0.0, q_k=0.0):
        self.a_k = a_k
        self.q_k = q_k
        self.k = None

    def update(self, k, x_k):
        """
        :param k: sample k where k (1..n) is a float
        :param x_k: value x_k
        :return: updated mean(a_k) and q_k
        """
        assert isinstance(k, float)
        self.k = k
        prev_ak = self.a_k
        self.a_k += (x_k - self.a_k) / k
        if k == 1.0:
            self.q_k = 0.0
        else:
            self.q_k += (x_k - prev_ak) * (x_k - self.a_k)

    @property
    def mean(self):
        return self.a_k

    # m2_k is the second moment at sample k
    @property
    def m2_k(self):
        return self.q_k

    @property
    def variance(self):
        if self.k == 1.0:
            return 0.0
        else:
            return self.q_k / (self.k - 1)

    @property
    def stdev(self):
        return (self.q_k / (self.k - 1))**(1.0/2.0)


class ArrayStats(object):
    def __init__(self, array_len):
        self.a_k = np.zeros(array_len)
        self.q_k = np.zeros(array_len)
        self.k = None

    def update(self, k, x_array):
        """
        :param k: sample k where k (1..n) is a float
        :param x_array: array of x_k values
        :return: updated mean(a_k) and q_k
        """
        assert isinstance(k, float)
        self.k = k
        for i, x_k in enumerate(x_array):
            prev_ak = self.a_k[i]
            self.a_k[i] += (x_k - self.a_k[i]) / k
            if k == 1.0:
                self.q_k[i] = 0.0
            else:
                self.q_k[i] += (x_k - prev_ak) * (x_k - self.a_k[i])

    @property
    def means(self):
        return self.a_k

    # m2_k is the second moment at sample k
    @property
    def m2_k(self):
        return self.q_k

    @property
    def variances(self):
        if self.k == 1.0:
            return 0.0
        else:
            return self.q_k / (self.k - 1)

    @property
    def stdevs(self):
        return np.sqrt(self.q_k / (self.k - 1))
