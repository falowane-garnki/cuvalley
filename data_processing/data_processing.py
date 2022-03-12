import pandas
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

def load(df=None, csv_path=None):
    """
    Ładuje dataframe z featurami i dokleja kolumnę temp_zuz. 
    Trzeba przekazazać df z wybranymi featurami i kolumną 'czas'
    albo path do .csv z tym df.
    """
    if (csv_path is None) and (df is None):
        raise ValueError("csv_path albo df musisz podac")

    if csv_path:
        df = pd.read_csv(csv_path)
    else:
        pass

    df['czas'] = pd.to_datetime(df['czas'], utc=True)
    df.set_index('czas', inplace=True)

    temp_zuz = pd.read_csv('../data/temp_zuz_fixed.csv')
    temp_zuz['Czas'] = pd.to_datetime(temp_zuz['Czas'], utc=True)
    temp_zuz.set_index('Czas', inplace=True)

    df = df.join(temp_zuz['temp_zuz'])

    # to powinno dzialac ale czasem nie dziala
    #assert (~df['temp_zuz'].isnull()).sum() == len(temp_zuz)

    return df


def split(df, proportions=(0.7, 0.15, 0.15)):
    """
    Dzieli df na wiele według proporcji.

    """
    assert sum(list(proportions)) == 1
    assert 2 <= len(proportions) <= 3

    n = len(df)

    if len(proportions) == 2:
        break_point = int(proportions[0]*n)
        df1 = df.iloc[0:break_point]
        df2 = df.iloc[break_point:]

        return df1, df2

    elif len(proportions) == 3:
        break_point1 = int(proportions[0]*n)
        break_point2 = int((1-proportions[2])*n)
        df1 = df.iloc[0:break_point1]
        df2 = df.iloc[break_point1:break_point2]
        df3 = df.iloc[break_point2:]

        return df1, df2, df3


def scale(*dfs):
    """
    Standaryzacja wszystkich przekazanych dataframow na
    podstawie wartosci sredniej i odchylenie pierwszego z nich.

    Pierwsze przekazujcie train set! Potem test/validation.
    """
    features = dfs[0].columns[dfs[0].columns != 'temp_zuz']

    scaler = StandardScaler().fit(dfs[0][features].values)

    mean = scaler.mean_
    scale = scaler.scale_

    for df in dfs: 
        df[features] = df[features].sub(mean).div(scale)


    return dfs


def make_sequences(df, seq_len=10):
    """
    Tworzy sekwencje i wypluwa je rozdzielając na X i Y.

    seq_len : długość sekwencji, czyli ile zestawów feature'ów

    """
    assert 'temp_zuz' in df.columns

    indicies = np.arange(0,len(df))[~df['temp_zuz'].isnull()]
    indicies = indicies[indicies>=seq_len]

    Y = df['temp_zuz'].iloc[indicies].values

    features = df.columns[df.columns != 'temp_zuz']

    X = np.array([np.array(df[features].iloc[i-seq_len:i]) for i in indicies] )

    assert X.shape[0] == Y.shape[0]

    return X, Y


def aggregate(df, interval):
    """ 
    Agreguje (uśredniając) wyniki po ileś minut (interval).

    Czyli np aggregate(df, 10) powinno grupować po 10 min (uśredniając wyniki).
    Uwaga, zaokrąglajcie czas w górę raczej, np: 15:03 -> 15:05
    """

    df.reset_index(inplace=True)
    assert 'czas' in df.columns
    assert 60 % interval == 0
    if not type(df['czas']) is pd.datetime:
        df['czas'] = pd.to_datetime(df['czas'])

    df['czas'] = df['czas'].dt.round(f'{interval}min')
    df = df.groupby(['czas']).mean().reset_index()
    df.set_index('czas', inplace=True)

    return df


df = pd.read_csv('../sample_data/avg_from_2020_10_01_00_00_00_to_2020_10_01_23_59_00', index_col=False)
df = load(df=df)
df = aggregate(df, 15)
print(df.head(1200))

