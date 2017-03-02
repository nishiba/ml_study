import pandas as pd
import numpy as np

data = pd.read_csv('../data/lalonde.csv')

d0 = data[data.apply(lambda x: x['treat'] == 0 and x['u74'] == 0, axis=1)]
print(d0)
