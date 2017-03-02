import numpy as np
import pandas as pd
from sklearn import tree
import nsb
import pydotplus
import matplotlib.pyplot as plt
from sklearn import model_selection
import math


def is_even(x):
    return np.round(x) % 2 == 0


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

if __name__ == '__main__':
    np.random.seed(123)
    df = grid_data_rotated(1000)
    features = ['x0', 'x1']
    plt.scatter(df['x0'], df['x1'], c=df['y'])
    plt.savefig('data.png')
    train, test = model_selection.train_test_split(df, train_size=0.8, random_state=123)
    param_grid = {'max_depth': [4, 5, 6, None], 'max_features': [2], 'min_samples_split': [2, 8, 16, 32],
                  'min_samples_leaf': [2, 8, 16, 32], 'max_leaf_nodes': [50]}

    clf = tree.DecisionTreeClassifier()
    gs = model_selection.GridSearchCV(clf, param_grid=param_grid, cv=10, n_jobs=-1)
    gs.fit(df[['x0', 'x1']], df['y'])
    scores = model_selection.cross_val_score(gs.best_estimator_, df[features], df['y'], cv=10, scoring='roc_auc')
    print('accuracy: %0.3f, std: %0.3f' % (scores.mean(), scores.std()))
    print(gs.best_estimator_)

    clf = gs.best_estimator_
    clf.fit(train[features], train['y'])
    nsb.plot.scatter_with_boundary(test['x0'], test['x1'], test['y'], gs.best_estimator_, 'fig.png')
    dot_data = tree.export_graphviz(gs.best_estimator_, out_file=None, feature_names=features)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_png('tree.png')

    print(dict(zip(features, clf.feature_importances_)))
