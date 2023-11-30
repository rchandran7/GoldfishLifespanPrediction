import sys
import pandas as pd
import numpy as np


def dummy_variable_col(df, col):
    df_temp = pd.get_dummies(df[col])
    df_temp = df_temp.astype(int)
    df = pd.concat([df, df_temp], axis=1,).reindex(df.index)
    df.drop(col, axis=1, inplace=True)
    return df


if len(sys.argv) == 3:
    file_input = sys.argv[1]
    file_output = sys.argv[2]
else:
    print("Error: Invalid number of arguments.", file=sys.stderr)

df = pd.read_csv(file_input)
df = df.drop(['id', 'color'], axis=1)
df = df.dropna()


df = dummy_variable_col(df, 'habitat')

df['Gender'] = df['Gender'].replace({True: 'male', False: 'female'})

df.to_csv(file_output + '_train.csv', index=False)
