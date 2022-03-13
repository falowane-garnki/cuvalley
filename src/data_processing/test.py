import pandas as pd
import numpy as np
import seq

df = seq.load(csv_path = 'basic.csv')
print(df)

df_train, df_val, df_test = seq.split(df)

print(df_train)

dfs = seq.scale(df_train, df_val, df_test)
print(dfs)

X, Y = seq.make_sequences(df_val)

print(X, X.shape)

print(Y, Y.shape)