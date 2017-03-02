import pandas as pd
import numpy as np
from nsb import data_util


def summary(df):
    missing = df.apply(pd.isnull, axis=0)
    missing['count'] = 1
    return missing.groupby(df.columns.values.tolist()).sum()


def count(df):
    missing = pd.DataFrame(index=['count'])
    for c in data_util.columns(df):
        missing[c] = np.array([pd.isnull(df[c]).sum()])
    return missing


def proportion(df):
    n = df.shape[0]
    c = count(df)
    c = c / n * 100
    c = c.rename(index={'count': '%'})
    return c


if __name__ == '__main__':
    events = pd.read_csv(
        '/Users/nishiba/Documents/work/python_workspace/real_world_machine_learning/event_recommendations/data/events_.csv',
        sep=',')
    feature_tags = ['city', 'state', 'zip', 'country', 'lat', 'lng']
    print(count(events[feature_tags]))
    print(proportion(events[feature_tags]).round(1))
