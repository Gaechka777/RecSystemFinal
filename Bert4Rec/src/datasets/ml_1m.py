from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from .utils import *

RAW_DATASET_ROOT_FOLDER = '/NOTEBOOK/RecSystem/RecSystemFinal/Bert4Rec/Data/'


def code():
    return 'santander'


def zip_file_content_is_folder():
    return True


def all_raw_file_names():
    return ['README',
            'movies.dat',
            'ratings.dat',
            'users.dat']


def load_ratings_df(data_path, name_file):
    #folder_path = _get_rawdata_folder_path()
    file_path = data_path + f'santander/{name_file}.dat'
    df = pd.read_csv(file_path, sep='::', header=None)
    df.columns = ['uid', 'sid', 'rating', 'timestamp']
    print(df)
    return df


def load_dataset(data_path, name_file, k_labels=3, min_uc=10):
    preprocess(data_path, name_file, k_labels, min_uc)
    dataset_path = _get_preprocessed_dataset_path()
    dataset = pickle.load(dataset_path.open('rb'))
    return dataset


def preprocess(data_path, name_file, k_labels=3, min_uc=10):
    dataset_path = _get_preprocessed_dataset_path()
    if dataset_path.is_file():
        print('Already preprocessed. Skip preprocessing')
        return
    if not dataset_path.parent.is_dir():
        dataset_path.parent.mkdir(parents=True)
    #maybe_download_raw_dataset()
    df = load_ratings_df(data_path, name_file)
    df = make_implicit(df)
    df = filter_triplets(df, 0, min_uc)
    #print('df', df)
    df, umap, smap = densify_index(df)
    train, val, test = split_df(df, len(umap), 'leave_one_out', k_labels)
    dataset = {'train': train,
               'val': val,
               'test': test,
               'umap': umap,
               'smap': smap}
    with dataset_path.open('wb') as f:
        pickle.dump(dataset, f)


def make_implicit(df, min_rating=0):
    print('Turning into implicit ratings')
    df = df[df['rating'] >= min_rating]
    # return df[['uid', 'sid', 'timestamp']]
    #print(len(df))
    return df


def filter_triplets(df, min_sc=0, min_uc=10):
    print('Filtering triplets')
    if min_sc > 0:
        item_sizes = df.groupby('sid').size()
        good_items = item_sizes.index[item_sizes >= min_sc]
        df = df[df['sid'].isin(good_items)]

    if min_uc > 0:
        print(min_uc)
        user_sizes = df.groupby('uid').size()
        good_users = user_sizes.index[user_sizes >= min_uc]
        df = df[df['uid'].isin(good_users)]

    return df


def densify_index(df):
    print('Densifying index')
    umap = {u: i for i, u in enumerate(set(df['uid']))}
    smap = {s: i for i, s in enumerate(set(df['sid']))}
    df['uid'] = df['uid'].map(umap)
    df['sid'] = df['sid'].map(smap)
    return df, umap, smap


def split_df(df, user_count, split='leave_one_out', k_labels=3):
    if split == 'leave_one_out':
        print('Splitting')
        user_group = df.groupby('uid')
        user2items = user_group.apply(lambda d: list(d.sort_values(by='timestamp')['sid']))
        print(user2items)
        print(k_labels)
        train, val, test = {}, {}, {}
        for user in range(user_count):
            items = user2items[user]
            train[user], val[user], test[user] = items[:-2*k_labels], items[-2*k_labels:-k_labels], items[-k_labels:]

        return train, val, test
    elif split == 'holdout':
        print('Splitting')
        np.random.seed(98765)
        eval_set_size = 500

        # Generate user indices
        permuted_index = np.random.permutation(user_count)
        train_user_index = permuted_index[                :-2*eval_set_size]
        val_user_index   = permuted_index[-2*eval_set_size:  -eval_set_size]
        test_user_index  = permuted_index[  -eval_set_size:                ]

        # Split DataFrames
        train_df = df.loc[df['uid'].isin(train_user_index)]
        val_df   = df.loc[df['uid'].isin(val_user_index)]
        test_df  = df.loc[df['uid'].isin(test_user_index)]

        # DataFrame to dict => {uid : list of sid's}
        train = dict(train_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        val   = dict(val_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        test  = dict(test_df.groupby('uid').progress_apply(lambda d: list(d['sid'])))
        return train, val, test
    else:
        raise NotImplementedError


def _get_rawdata_root_path():
    return Path(RAW_DATASET_ROOT_FOLDER)


def _get_rawdata_folder_path():
    root = _get_rawdata_root_path()
    return root.joinpath(code())


def _get_preprocessed_root_path():
    root = _get_rawdata_root_path()
    return root.joinpath('preprocessed')


def _get_preprocessed_folder_path(min_rating=0, min_uc=5, min_sc=0, split='leave_one_out'):
    preprocessed_root = _get_preprocessed_root_path()
    folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-split{}' \
        .format(code(), min_rating, min_uc, min_sc, split)
    return preprocessed_root.joinpath(folder_name)


def _get_preprocessed_dataset_path():
    folder = _get_preprocessed_folder_path()
    return folder.joinpath('dataset.pkl')


