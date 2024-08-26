import os

import audeer
import pandas as pd
from omegaconf import DictConfig
from pandas import DataFrame
from src.data.create_artificial_data import create_white_noise_db

from src.data.create_data_sets import create_dataset


def load_training_data(cfg_data: DictConfig,
                       load_train_set: bool = True,
                       load_dev_set: bool = True,
                       ) -> (DataFrame, DataFrame):
    # columns (with or w/o labels) that are later needed for fine-tuning
    intended_col_lab = [[c, l] for c, l in cfg_data.important_columns_labels.items()] \
        if cfg_data.important_columns_labels else [[None, None]]

    audeer.mkdir(cfg_data.cache_path)
    if load_train_set:
        df_train = prepare_labeled_df(
            cfg_data.data_source_labeled_train,
            os.path.join(cfg_data.cache_path, 'train.csv'),
            intended_labels=intended_col_lab,
            len_subset=cfg_data.len_subset,

        )
    else:
        df_train = pd.DataFrame()

    if load_dev_set:
        df_dev = prepare_labeled_df(
            cfg_data.data_source_labeled_dev,
            os.path.join(cfg_data.cache_path, 'dev.csv'),
            intended_labels=intended_col_lab,
            len_subset=cfg_data.len_subset,
        )
    else:
        df_dev = pd.DataFrame()

    return df_train, df_dev


def load_testing_data(cfg_data: DictConfig) -> DataFrame:
    # columns (with or w/o labels) that are needed for testing
    intended_col_lab = [[c, l] for c, l in cfg_data.important_columns_labels.items()] \
        if cfg_data.important_columns_labels else [[None, None]]
    audeer.mkdir(cfg_data.cache_path)
    if 'data_source_labeled_test' in cfg_data:
        df_test = prepare_labeled_df(
            cfg_data.data_source_labeled_test,
            os.path.join(cfg_data.cache_path, 'test.csv'),
            intended_labels=intended_col_lab,
        )
    elif 'data_source_artificaial_test' in cfg_data:
        df_test = prepare_artificial_df(cfg_data.data_source_artificaial_test)
    else:
        raise NotImplementedError('No datasource selected')

    return df_test


def load_dev_data(cfg_data: DictConfig) -> DataFrame:
    # columns (with or w/o labels) that are needed for testing
    intended_col_lab = [[c, l] for c, l in cfg_data.important_columns_labels.items()] \
        if cfg_data.important_columns_labels else [[None, None]]
    audeer.mkdir(cfg_data.cache_path)
    if 'data_source_labeled_dev' in cfg_data:
        df_test = prepare_labeled_df(
            cfg_data.data_source_labeled_dev,
            os.path.join(cfg_data.cache_path, 'dev.csv'),
            intended_labels=intended_col_lab,
        )
    else:
        raise NotImplementedError('No datasource selected')

    return df_test


def prepare_artificial_df(name):
    if name == 'white_noise':
        db = create_white_noise_db('../../../artifical_dbs')
        df = db['files'].get(as_segmented=True, allow_nat=False)
        return df


def prepare_labeled_df(
        data_source,
        cache_csv,
        intended_labels=None,
        len_subset=None,

) -> DataFrame:
    data_source_labeled = [database for database in data_source]
    data_bases_labeled = create_dataset(data_source_labeled, len_subset=len_subset)

    df_labeled = pd.DataFrame([])
    for db, _ in data_bases_labeled:
        new_df = db.get(as_segmented=True, allow_nat=False)
        # only select rows, containing all the needed data
        if intended_labels:
            clean_df = pd.DataFrame([])
            for column, labels in intended_labels:
                if column in new_df.columns and labels:
                    if clean_df.empty:
                        clean_df = new_df.loc[new_df[column].isin(labels)][[column]]
                    else:
                        clean_df = new_df.loc[new_df[column].isin(labels)][[column]].combine_first(clean_df)
                elif column in new_df.columns:
                    if clean_df.empty:
                        clean_df = new_df[[column]]
                    else:
                        clean_df = new_df[[column]].combine_first(clean_df)
                elif column not in new_df.columns and column not in clean_df.columns:
                    clean_df[column] = None
            new_df = clean_df.astype('object')

        if df_labeled.empty:
            df_labeled = new_df
        else:
            df_labeled = new_df.combine_first(df_labeled)

    df_labeled.to_csv(cache_csv)

    return df_labeled
