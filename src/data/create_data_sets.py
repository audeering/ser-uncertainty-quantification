from typing import Union, List

import audb
import pandas as pd
from pandas import DataFrame


def create_dataset(data_source: Union[tuple, List],
                   len_subset=None,
                   ) -> List:
    data_bases = []
    for config in data_source:
        if len_subset:
            db = audb.load(
                name=config['name'],
                version=config['version'],
                only_metadata=True,
                full_path=False,
            )
            table_names = [table for table in db.tables.keys()]
            indices = db[table_names[0]].index
            if type(indices) is pd.MultiIndex:  # index consists of (path, stat, end)
                file_names = indices.get_level_values(0).unique()
            else:
                file_names = indices
            if len_subset > len(file_names):
                len_subset = len(file_names)
            files_to_load = file_names[0:len_subset]

            data_bases.append((audb.load(
                name=config['name'],
                version=config['version'],
                format='wav',
                mixdown=True,
                sampling_rate=16000,
                media=files_to_load,
                full_path=True,
                num_workers=8,
            )[config['table']], False))
        elif 'condition' in config and config['condition']:
            db = audb.load(
                name=config['name'],
                version=config['version'],
                only_metadata=True,
                full_path=False,
            )

            files_to_load = crate_df_from_dataset(db, condition=config['condition']).index.get_level_values('file')

            data_bases.append((audb.load(
                name=config['name'],
                version=config['version'],
                format='wav',
                mixdown=True,
                sampling_rate=16000,
                media=files_to_load,
                full_path=True,
            )[config['table']], config['condition']))

        else:
            data_bases.append((audb.load(
                name=config['name'],
                version=config['version'],
                format='wav',
                mixdown=True,
                sampling_rate=16000,
                full_path=True,
            )[config['table']], False))

    return data_bases


def crate_df_from_dataset(db,
                          condition=None,
                          ) -> DataFrame:
    combined_df = pd.DataFrame([])
    if condition:
        column, values = condition
        for table in db.tables:
            # if ('train' in table):
            info = db[table].get(as_segmented=True, allow_nat=False)
            if column in info.columns:
                combined_df = pd.concat([combined_df, info.loc[info[column].isin(values)]])
    else:
        for table in db.tables:
            info = db[table].get(as_segmented=True, allow_nat=False)
            combined_df = pd.concat([combined_df, info])

    return combined_df
