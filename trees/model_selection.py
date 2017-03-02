import numpy as np
import pandas as pd
from sklearn import svm
import nsb
import pydotplus
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn import ensemble
from sklearn import tree
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


def output(clf, df, features, target, name):
    print(clf)
    scores = model_selection.cross_val_score(clf, df[features], df[target], cv=10, scoring='roc_auc')
    print(name, 'accuracy: %0.3f, std: %0.3f' % (scores.mean(), scores.std()))
    train, test = model_selection.train_test_split(df, train_size=0.8, random_state=123)
    # clf.fit(train[features], train[target])
    # nsb.plot.scatter_with_boundary(test['x0'], test['x1'], test['y'], clf, name + '_fig.png')


if __name__ == '__main__':
    np.random.seed(123)
    df = grid_data_with_dummy(1000)
    features = ['x0', 'x1', 'x2']
    target = 'y'
    plt.scatter(df['x0'], df['x1'], c=df['y'])
    plt.savefig('data.png')

    clf = nsb.classifier.calibrate_decision_tree(df[features], df[target])
    output(clf, df, features, target, 'decision_tree')
    clf = nsb.classifier.calibrate_svc(df[features], df[target])
    output(clf, df, features, target, 'svc')
    clf = nsb.classifier.calibrate_random_forest(df[features], df[target])
    output(clf, df, features, target, 'random_forest')
    clf = nsb.classifier.calibrate_extra_trees(df[features], df[target])
    output(clf, df, features, target, 'extra_trees')
    clf = nsb.classifier.calibrate_ada_boost(df[features], df[target])
    output(clf, df, features, target, 'ada_boost')
    clf = nsb.classifier.calibrate_gradient_boost(df[features], df[target])
    output(clf, df, features, target, 'gradient_boost')
