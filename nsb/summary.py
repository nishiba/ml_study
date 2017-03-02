import pandas as pd
import nsb


def data(df):
    out = ''
    out += '## shape\n'
    out += str(df.shape)
    out += '\n\n'
    out += '## head\n'
    out += nsb.format.dataframe(df.head())
    out += '\n\n'
    # out += '## columns\n'
    # out += str(nsb.data_util.columns(df))
    # out += '\n\n'
    out += '## data types\n'
    out += nsb.format.series(df.dtypes)
    out += '\n\n'
    out += '## missing\n'
    out += nsb.format.dataframe(nsb.missing.proportion(df).round(2))
    out += '\n'
    # out += nsb.format.dataframe(nsb.missing.summary(df))
    # out += '\n'
    out += '## duplicated\n'
    out += nsb.format.dataframe(nsb.feature.duplicated(df))
    out += '\n'
    out += '## categorical(candidates)\n'
    out += str(nsb.feature.find_categorical(df))
    out += '\n\n'
    out += '## proportion of categorical data\n'
    for c in nsb.data_util.category_columns(df):
        x = df[c].value_counts(normalize=True)
        out += nsb.format.series(df[c].value_counts(normalize=True).round(2))
        out += '\n'
    out += '\n'
    return out


def data_(df):
    out = ''
    out += '## shape\n'
    out += str(df.shape)
    out += '\n\n'
    out += '## head\n'
    out += nsb.format.dataframe(df.head())
    out += '\n\n'
    out += '## columns\n'
    out += str(nsb.data_util.columns(df))
    out += '\n\n'
    out += '## data types\n'
    out += nsb.format.series(df.dtypes)
    out += '\n\n'
    out += '## missing\n'
    out += nsb.format.dataframe(nsb.missing.proportion(df).round(2))
    out += '\n'
    out += nsb.format.dataframe(nsb.missing.summary(df))
    out += '\n'
    out += '## duplicated\n'
    out += nsb.format.dataframe(nsb.feature.duplicated(df))
    out += '\n'
    out += '## categorical(candidates)\n'
    out += str(nsb.feature.find_categorical(df))
    out += '\n\n'
    out += '## proportion of categorical data\n'
    for c in nsb.data_util.category_columns(df):
        x = df[c].value_counts(normalize=True)
        out += nsb.format.series(df[c].value_counts(normalize=True).round(2))
        out += '\n'
    out += '\n'
    return out




