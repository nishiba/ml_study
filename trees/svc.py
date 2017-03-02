import numpy as np
import pandas as pd
from sklearn import svm
import nsb
import pydotplus
import matplotlib.pyplot as plt
from sklearn import model_selection
import math


def is_even(x):
    return np.round(x) % 2 == 0


def grid_data_eq(n=1000):
    m = round(math.sqrt(n))
    axis0 = np.linspace(0, 2, m)
    axis1 = np.linspace(0, 2, m)

    x0, x1 = np.meshgrid(axis0, axis1)
    x0 = x0.ravel()
    x1 = x1.ravel()
    y = np.logical_xor(is_even(x0), np.logical_not(is_even(x1))).astype(np.int64)

    return pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})


def grid_data(n=1000):
    x0 = np.random.uniform(0, 2, n)
    x1 = np.random.uniform(0, 2, n)
    y = np.logical_xor(is_even(x0), np.logical_not(is_even(x1))).astype(np.int64)
    c = 0
    for i, a in enumerate(np.random.uniform(0, 1, n)):
        if a < 0.1:
            c += 1
            y[i] = 1 - y[i]

    return pd.DataFrame({'x0': x0, 'x1': x1, 'y': y})


def grid_data_with_dummy(n=1000):
    df = grid_data(n)
    df['x2'] = np.random.uniform(0, 2, n)
    return df


def grid_data_rotated(n=1000):
    df = grid_data(n)
    theta = math.pi / 4
    x0 = math.cos(theta) * df['x0'] - math.sin(theta) * df['x1']
    x1 = math.sin(theta) * df['x0'] + math.cos(theta) * df['x1']
    df['x0'] = x0
    df['x1'] = x1
    return df


def grid_data_eq_rotated(n=1000):
    df = grid_data_eq(n)
    theta = math.pi / 4
    x0 = math.cos(theta) * df['x0'] - math.sin(theta) * df['x1']
    x1 = math.sin(theta) * df['x0'] + math.cos(theta) * df['x1']
    df['x0'] = x0
    df['x1'] = x1
    return df


if __name__ == '__main__':
    np.random.seed(123)
    df = grid_data_rotated()
    features = ['x0', 'x1']
    plt.scatter(df['x0'], df['x1'], c=df['y'])
    plt.savefig('data.png')
    train, test = model_selection.train_test_split(df, train_size=0.8, random_state=123)
    clf = nsb.classifier.calibrate_svc(df[features], df['y'])
    scores = nsb.classifier.calculate_scores(clf, df[features], df['y'])
    print('- accuracy: %0.3f\n- std: %0.3f' % (scores.mean(), scores.std()))
    print(clf)

    clf.fit(train[features], train['y'])
    nsb.plot.scatter_with_boundary(test['x0'], test['x1'], test['y'], clf, 'fig.png')
