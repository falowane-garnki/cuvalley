import pandas as pd
import os

feature_desc_df = pd.read_csv('./data/feature_desc.csv', index_col='name')


def merge_data(dir_path='data/'):
    """Merges feature data into one .csv"""
    files = sorted([f for f in os.listdir(dir_path) if f[:3] == 'avg'])
    merged_df = pd.concat([pd.read_csv(dir_path + f) for f in files])

    merged_df['czas'] = pd.to_datetime(merged_df['czas'])
    assert merged_df['czas'].is_monotonic_increasing

    merged_df.to_csv(dir_path + 'merged.csv', index=False)
    print(merged_df)


def feature_desc(name):
    """Returns feature description for a feature name passed"""
    if type(name)==str:
        return feature_desc_df.loc[name]['desc']
    else:
        return feature_desc_df.loc[name]['desc'].values


def correct_tz_temp_zuz(dir_path='data/'):
    """
    Applies correct time zone to temp_zuz.csv data
    and saves it in 'temp_zuz_fixed.csv'
    """
    temp_zuz = pd.read_csv(dir_path + 'temp_zuz.csv', delimiter=';')

    temp_zuz['Czas'] = pd.to_datetime(temp_zuz['Czas'])
    temp_zuz.set_index('Czas', inplace=True)
    temp_zuz = temp_zuz.tz_localize('Europe/Warsaw', ambiguous='NaT')
    temp_zuz.reset_index(inplace=True)

    holes = temp_zuz['Czas'].isnull()

    fills = {
        600: '2020-10-25 02:00:00+02:00',
        601: '2020-10-25 02:01:00+01:00',
        9220: '2021-10-31 02:00:00+02:00',
        9221: '2021-10-31 02:01:00+01:00',
        9222: '2021-10-31 02:05:00+01:00'
    }

    for i in fills:
        temp_zuz.loc[i, 'Czas'] = fills[i]

    assert sum(temp_zuz['Czas'].isnull()) == 0

    temp_zuz.to_csv(dir_path + 'temp_zuz_fixed.csv', index=False)

merge_data()
