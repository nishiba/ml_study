import numpy as np
from scipy.stats import t
import math
import pandas as pd


class Grubbs:
    @staticmethod
    def test(x, alpha, sofar=np.array([])):
        n = len(x)
        if n < 3:
            return sofar
        mu = np.mean(x)
        sigma = np.std(x)
        xmax = np.max(x)
        xmin = np.min(x)
        z = max(xmax - mu, mu - xmin) / sigma
        if Grubbs.is_outlier(n, z, alpha):
            e = [xmax if xmax - mu > mu - xmin else xmin]
            sofar = np.append(sofar, e)
            return Grubbs.test(np.delete(x, np.argwhere(x == e)), alpha, sofar)
        else:
            return sofar

    @staticmethod
    def is_outlier(n, z, alpha):
        t0 = math.fabs(t.ppf(2 * alpha / n, n - 2))
        s0 = (n - 1) * t0 / math.sqrt(n * (n - 2) + n * t0 ** 2)
        return z > s0


def outlier(x, alpha=0.01):
    if type(x) == pd.core.series.Series:
        return outlier(x.as_matrix(), alpha)
    else:
        return Grubbs.test(x, alpha)


if __name__ == '__main__':
    x = np.random.normal(0, 1, 100)
    x = np.append(x, 10)
    print(outlier(x))
    print(x)
