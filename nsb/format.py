import pandas as pd
import numpy as np


def dataframe(df):
    lst = df.to_csv(sep='|').split('\n')
    if lst[0][0] == '|':
        lst[0] = '-' + lst[0]
    if lst[0][-1] == '|':
        lst[0] += '-'
    lst.insert(1, '|'.join(['---'] * len(lst[0].split('|'))))
    return '\n'.join(lst)


def series(s):
    # m = np.array([s.values].round(2)).T # because round supports only numeric values.
    m = np.array([s.values]).T
    return dataframe(pd.DataFrame(m, columns=[s.name], index=s.index.values))
