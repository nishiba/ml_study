import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize


def negative_log_likelihood(y1, y2, m, m1, m2, s1, s2, rho):
    mean = np.array([m1, m2])
    s12 = rho * s1 * s2
    cov = np.array([[s1 * s1, s12], [s12, s2 * s2]])
    v = 0.0
    for yi1, yi2, mi in zip(y1, y2, m):
        if mi == 1:
            v += multivariate_normal.logpdf(np.array([yi1, yi2]), mean=mean, cov=cov)
        else:
            v += norm.logpdf(yi1, m1, s1)
    return -v


if __name__ == '__main__':
    n = 1000
    u1 = 50
    u2 = 50
    v1 = 100
    v2 = 50
    rho = 0.8
    v12 = rho * math.sqrt(v1 * v2)
    mean = np.array([u1, u2])
    cov = np.array([[v1, v12], [v12, v2]])
    y = np.random.multivariate_normal(mean, cov, n)

    data = pd.DataFrame({'y1': y[:, 0], 'y2': y[:, 1]})
    data['m'] = data.apply(lambda x: 1 if x['y1'] > 60 else 0, axis=1)
    data['y2'] = data.apply(lambda x: None if x['m'] == 0 else x['y2'], axis=1)
    pass_idx = data['m'] == 1

    # using regression model
    coeff = np.polyfit(data[pass_idx]['y1'], data[pass_idx]['y2'], 1)
    c0 = coeff[0]
    var_y1 = np.std(data['y1']) ** 2
    var_epsilon = np.std(data['y2'] - np.poly1d(coeff)(data['y1'])) ** 2
    corr = c0 * var_y1 / (math.sqrt(var_y1) * math.sqrt(c0 ** 2 * var_y1 + var_epsilon))
    print('parameters:', coeff)
    print('correlation:', corr)

    # using multivariate normal model.
    cons = (
        {'type': 'ineq', 'fun': lambda theta: 0.9999 - theta[4]},
        {'type': 'ineq', 'fun': lambda theta: theta[4] + 0.9999},
        {'type': 'ineq', 'fun': lambda theta: theta[0:4] - 1},
        {'type': 'ineq', 'fun': lambda theta: 100 - theta[0:4]},
    )
    theta0 = np.array([50, 60, 10, 30, 0.1])
    theta = minimize(lambda theta: negative_log_likelihood(data['y1'], data['y2'], data['m'],
                                                           theta[0], theta[1], theta[2], theta[3], theta[4]),
                     theta0,
                     constraints=cons,
                     method="SLSQP")
    print(theta)

    axis_x = np.linspace(0, 100, 101)
    plt.scatter(y[:, 0], y[:, 1])
    plt.fill_betweenx((-10, 110), 60, facecolor='r', alpha=0.2)
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.plot(axis_x, np.poly1d(coeff)(axis_x))
    plt.xlabel('y1')
    plt.ylabel('y2')
    plt.show()
