import pandas as pd
import numpy as np
from nsb import data_util
import re
from sklearn import model_selection
from copy import deepcopy


def duplicated(df):
    dup = pd.DataFrame(index=['count'])
    for c in data_util.columns(df):
        dup[c] = np.array([df.duplicated(c).sum()])
    return dup


def unique_counts(df):
    u = pd.DataFrame(index=['count'])
    for c in data_util.columns(df):
        u[c] = np.array([df[c].value_counts().shape[0]])
    return u


def unique_proportions(df):
    return unique_counts(df) / df.shape[0]


def find_categorical(df, threshold=0.1):
    u = unique_proportions(df) < threshold
    return np.array([c for c in data_util.columns(u) if u[c][0]])


def to_category(df, tags):
    dup = df.copy()
    for t in tags:
        dup[t] = dup[t].astype('category')
    return dup


def add_prefix(df, p):
    d = {}
    for c in data_util.columns(df):
        d[c] = p + c
    return df.rename(columns=d)


def select_numeric(df):
    c = df.select_dtypes(['float64', 'int64']).columns
    return df[c].copy()


def add_new_column(df, expr):
    m = re.findall(r'\((\w*), ([\w#]*), ([\w#]*)\) -> ([\w#]*)', expr)
    if len(m) == 0:
        return
    (op, x, y, name) = m[0]
    if op == 'add':
        df[name] = df[x] + df[y]
    elif op == 'sub':
        df[name] = df[x] - df[y]
    elif op == 'mul':
        df[name] = df[x] * df[y]
    elif op == 'div':
        df[name] = df[x] / df[y]
    elif op == 'dt_seconds':
        df[name] = (pd.to_datetime(df[x]) - pd.to_datetime(df[y])).dt.seconds


def forward_selection(model, features, target, cv=5):
    max_auc = 0.5
    selected = []
    residuals = data_util.columns(features)
    while True:
        this_auc = max_auc
        this_feature = ''
        for t in residuals:
            selected.append(t)
            auc = model_selection.cross_val_score(model, features[selected], target, cv=cv, scoring='roc_auc').mean()
            if auc > this_auc:
                this_auc = auc
                this_feature = t
            selected.pop()
        if this_feature == '':
            return selected
        print(this_auc, this_feature)
        max_auc = this_auc
        selected.append(this_feature)
        residuals.remove(this_feature)


def backward_elimination(model, features, target, cv=5):
    max_auc = model_selection.cross_val_score(model, features, target, cv=cv, scoring='roc_auc').mean()
    residuals = data_util.columns(features)
    while True:
        this_auc = max_auc
        this_feature = ''
        for t in residuals:
            this_set = deepcopy(residuals)
            this_set.remove(t)
            auc = model_selection.cross_val_score(model, features[this_set], target, cv=cv, scoring='roc_auc').mean()
            if auc > this_auc:
                this_auc = auc
                this_feature = t
        if this_feature == '':
            return residuals
        print(this_auc, this_feature)
        max_auc = this_auc
        residuals.remove(this_feature)


if __name__ == '__main__':
    df = pd.DataFrame({'#x': (1, 2), 'y': (3, 4)})
    add_new_column(df, '(add, #x, y) -> z')
    add_new_column(df, '(div, z, y) -> w')
    print(df.head())