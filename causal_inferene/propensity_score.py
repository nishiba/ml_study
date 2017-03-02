import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import logistic
from sklearn import metrics


def random_sampling():
    n = 1000
    parent_school_career = np.random.normal(3, 1, n)
    treated = np.random.binomial(1, 0.5, n)
    math_score = np.random.normal(50, 10, n) + 10 * (parent_school_career - 3)
    data = pd.DataFrame({'math': math_score, 'parent': parent_school_career, 'treated': treated})

    # plot all sampling data
    plt.scatter(data['parent'], data['math'], c=data['treated'])
    plt.show()

    idx1 = data['treated'] == 1
    idx0 = data['treated'] == 0
    math0 = data[idx0]['math']
    math1 = data[idx1]['math']
    print('z=0:', np.average(math0), np.std(math0))
    print('z=1:', np.average(math1), np.std(math1))


def non_random_sampling():
    n = 1000
    treated = np.random.binomial(1, 0.5, n)
    parent_school_career = np.random.normal(2.5, 1, n) + treated
    math_score = np.random.normal(50, 10, n) + 10 * (parent_school_career - 3)
    data = pd.DataFrame({'math': math_score, 'parent': parent_school_career, 'treated': treated})

    # plot all sampling data
    plt.scatter(data['parent'], data['math'], c=data['treated'])
    plt.ylabel('math score')
    plt.xlabel("parent's school career")
    plt.show()

    idx1 = data['treated'] == 1
    idx0 = data['treated'] == 0
    math0 = data[idx0]['math']
    math1 = data[idx1]['math']
    print('z=0:', np.average(math0), np.std(math0))
    print('z=1:', np.average(math1), np.std(math1))


def calculate_score(x, alpha):
    a = np.dot(x, np.array([alpha]).T).flatten()
    s = logistic.cdf(a)
    return s


def log_likelihood(x, z, alpha):
    s = calculate_score(x, alpha)
    p = z * np.log(s) + (1 - z) * np.log(1 - s)
    return np.sum(p)


def get_data():
    n = 1000
    treated = np.random.binomial(1, 0.5, n)
    parent_school_career = np.random.normal(2.5, 1, n) + treated
    math_score = np.random.normal(50, 10, n) + 10 * (parent_school_career - 3)
    data = pd.DataFrame({'math': math_score, 'parent': parent_school_career, 'treated': treated})
    data['constant'] = np.ones(n)
    return data


def add_hidden_variance(data):
    data['math'] = data.apply(lambda x: x['math'] + np.random.normal(10, 5, 1)[0] if x['treated'] == 1 else x['math'],
                              axis=1)
    return data


def calculate_auc(x, y):
    fpr, tpr, thresholds = metrics.roc_curve(y, x)
    print(metrics.auc(fpr, tpr))
    return x


def calculate_propensity_score(data):
    cons = ({'type': 'ineq', 'fun': lambda theta: 10 - theta}, {'type': 'ineq', 'fun': lambda theta: theta + 10},)
    alpha = np.random.uniform(0, 1, 2)
    res = minimize(lambda a: -log_likelihood(data[['parent', 'constant']], data['treated'].as_matrix().flatten(), a),
                   alpha, constraints=cons, method="SLSQP")
    score = calculate_score(data[['parent', 'constant']], res.x)
    return score


def matching(data):
    data['score'] = calculate_propensity_score(data)
    n = 10
    grouping = np.linspace(0, 1, n)
    d = 0.0
    for i in range(0, n - 1):
        y1 = np.average(
            data.query('score >= {0} and score < {1} and treated == 1'.format(grouping[i], grouping[i + 1]))['math'])
        y0 = np.average(
            data.query('score >= {0} and score < {1} and treated == 0'.format(grouping[i], grouping[i + 1]))['math'])
        w = data.query('score >= {0} and score < {1}'.format(grouping[i], grouping[i + 1])).shape[0]
        d += (y1 - y0) * w
    d /= data.shape[0]
    print(d)


def ipw_analysis(data):
    score = calculate_propensity_score(data)
    calculate_auc(score, data['treated'].as_matrix().flatten())
    w0 = (1 - data['treated']) / (1 - score)
    w1 = data['treated'] / score
    y0 = np.sum(w0 * data['math']) / np.sum(w0)
    y1 = np.sum(w1 * data['math']) / np.sum(w1)
    print('y0:', y0)
    print('y1', y1)


if __name__ == '__main__':
    np.random.seed(123)
    # non_random_sampling()
    data = get_data()
    # data = add_hidden_variance(data)
    ipw_analysis(data)
    matching(data)
