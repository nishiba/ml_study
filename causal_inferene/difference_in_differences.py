import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = pd.read_csv('../data/lalonde.csv')
    data['diff'] = data['re78'] - data['re74']
    d0 = data[data.apply(lambda x: x['treat'] == 0, axis=1)]
    d1 = data[data.apply(lambda x: x['treat'] == 1, axis=1)]
    a0 = np.average(d0[['re74', 're78', 'diff']], axis=0)
    a1 = np.average(d1[['re74', 're78', 'diff']], axis=0)
    print(a1[2] - a0[2])
    print(a0)
    print(a1)
    print(d0.head())

    # plt.plot(range(2), np.average(d0[['re74', 're78']], axis=0), c='b')
    # plt.plot(range(2), np.average(d1[['re74', 're78']], axis=0), c='r')
    # plt.show()

