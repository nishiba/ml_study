import numpy as np
import pandas as pd


def sampling():
    n = 1000
    score_base = 600
    score_private = 80
    score_rich = 150
    score_sigma = 50
    rich_ratio = 0.2
    public_ratio = 0.1
    winning_ticket_ratio = 0.4

    data = pd.DataFrame({
        'rich': np.random.binomial(1, rich_ratio, n),
        'lottery': np.random.binomial(1, winning_ticket_ratio, n)
    })
    data['private'] = data.apply(lambda d: 1 if d['rich'] == 1 or d['lottery'] == 1 else 0, axis=1)
    data['private'] = data['private'] * np.random.binomial(1, 1 - public_ratio, n)
    data['score'] = score_base + score_private * data['private'] + score_rich * data['rich'] \
        + np.random.normal(0, score_sigma, n)

    return data.drop(['rich'], axis=1)


def simple_analysis(data):
    d0 = data[data['private'] == 0]
    d1 = data[data.apply(lambda d: d['private'] == 1 and d['lottery'] == 1, axis=1)]
    print(np.average(d1['score']) - np.average(d0['score']))


def instrumental_variable_method(data):
    data0 = data[data['lottery'] == 0]
    data1 = data[data['lottery'] == 1]
    y0 = np.average(data0['score'])
    y1 = np.average(data1['score'])
    d0 = np.average(data0['private'])
    d1 = np.average(data1['private'])
    print((y1 - y0) / (d1 - d0))


if __name__ == '__main__':
    np.random.seed(123)
    data = sampling()
    print(data.head(10))
    print(data.describe())
    simple_analysis(data)
    instrumental_variable_method(data)
    print('hello')