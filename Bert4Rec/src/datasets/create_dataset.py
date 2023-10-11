import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import dask
import dask.dataframe as dd
from scipy.sparse import csr_matrix
import numpy as np
import pickle
from tqdm import tqdm
from collections import Counter
import random
import cmd
import os

cols = ['fetch_date', 'cust_code', 'emp_index', 'country', 'sex', 'age', 'cust_date', 'new_cust', 'cust_seniority',
            'indrel', 'last_date_as_primary', 'cust_type', 'cust_rel', 'residence_index', 'foreigner_index',
            'spouse_index',
            'joining_channel', 'deceased', 'address_type', 'prov_code', 'prov_name', 'activity_index', 'income',
            'segmentation',
            'savings_account', 'guarentees', 'current_account', 'derivative_account', 'payroll_account',
            'junior_account', 'mas_account',
            'perticular_account', 'perticular_plus', 'st_deposit', 'mt_deposits', 'lt_deposits', 'e_account', 'funds',
            'mortgage',
            'pension', 'loan', 'tax', 'credit_card', 'securities', 'home_account', 'payroll', 'pension2', 'direct_debit'
            ]

target_columns = ['savings_account', 'guarentees', 'current_account', 'derivative_account', 'payroll_account',
                      'junior_account', 'mas_account',
                      'perticular_account', 'perticular_plus', 'st_deposit', 'mt_deposits', 'lt_deposits', 'e_account',
                      'funds', 'mortgage',
                      'pension', 'loan', 'tax', 'credit_card', 'securities', 'home_account', 'payroll', 'pension2',
                      'direct_debit']

user_features = ['fetch_date', 'cust_code', 'emp_index', 'country', 'sex', 'age', 'cust_date', 'new_cust',
                     'cust_seniority',
                     'indrel', 'last_date_as_primary', 'cust_type', 'cust_rel', 'residence_index', 'foreigner_index',
                     'spouse_index',
                     'joining_channel', 'deceased', 'address_type', 'prov_code', 'prov_name', 'activity_index',
                     'income', 'segmentation']

def preprocess_missing_values(data):
    data[target_columns] = data[target_columns].fillna(0)
    return data

def create(path, name_file, create = True):
    global user_features

    if os.path.exists(path):
        print('Already create santander.dat')
        return 'skip'
    print('Path init --- ', path)
    print('Start ...')
    data = pd.read_csv(path)

    data.columns = cols
    data = preprocess_missing_values(data)
    data.isna().sum()

    # creating a new dataframe for a previous date
    dummy = pd.DataFrame(
        {
            'cust_code': data.cust_code.unique(),
            'fetch_date': '2014-12-28'
        }
    )

    new_purchases = pd.concat([data[['cust_code', 'fetch_date'] + target_columns], dummy])

    new_purchases = new_purchases.fillna(0)
    new_purchases[target_columns] = new_purchases[target_columns].astype('uint8')
    new_purchases = new_purchases.sort_values(['cust_code', 'fetch_date'])

    vals = np.array(new_purchases[target_columns].values, dtype='int8')
    vals[1:] = vals[1:] - vals[:-1]

    new_purchases[target_columns] = vals
    # Removing the data of '2014-12-28' and '2015-01-28'
    new_purchases = new_purchases[~new_purchases.fetch_date.isin(['2014-12-28', '2015-01-28'])]

    # Some of the products were discontinued so purchase value there becomes less than 0,
    # as we are only interested in purchases we can remove them
    for col in target_columns:
        new_purchases[col][new_purchases[col] < 0] = 0

    # drop all the rows where no new purchase is made
    new_purchases = new_purchases[(new_purchases[target_columns].sum(axis=1) > 0)]

    user_info = user_features
    new_purchases = data[user_info].merge(new_purchases, on=['fetch_date', 'cust_code'], how='right')
    new_purchases['fetch_date'] = pd.to_datetime(new_purchases['fetch_date'])

    np.sort(new_purchases["fetch_date"].unique())

    train_caser = new_purchases

    user_data = new_purchases.drop_duplicates(subset=['cust_code'], keep='last')[
        ['cust_code', 'emp_index', 'country', 'sex', 'age', 'cust_date', 'new_cust', 'cust_seniority',
         'indrel', 'cust_type', 'cust_rel', 'residence_index', 'foreigner_index', 'spouse_index',
         'joining_channel', 'deceased', 'address_type', 'prov_code', 'prov_name', 'activity_index',
         'income', 'segmentation']]

    user_purchases = new_purchases[['cust_code'] + target_columns].groupby(by='cust_code').sum()

    user_features = user_data.merge(user_purchases, on='cust_code', how='right')

    train_caser = train_caser.sort_values(['cust_code', 'fetch_date'])[['cust_code', 'fetch_date'] + target_columns]

    target_dict = {target_columns[i]: i for i in range(len(target_columns))}

    new_representation = {"uid": [], "sid": [], "rating": [], "timestamp": []}

    for index, row in train_caser.iterrows():
        t = row[target_columns]
        for action in list(t[t != 0].index):
            new_representation["uid"].append(row["cust_code"])
            new_representation["sid"].append(target_dict[action])
            new_representation["rating"].append(1)
            new_representation["timestamp"].append(row['fetch_date'])

    train_caser = pd.DataFrame.from_dict(new_representation).sort_values(['uid'])
    train_caser['timestamp'] = pd.Series(
        pd.to_datetime(train_caser['timestamp'], format='%b-%y').values.astype(float)).div(10 ** 9)

    new_path = os.path.dirname(path) + '/' + 'santander'
    print('Creating new path for data --- ', new_path)
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    np.savetxt(new_path + '/' + f'{name_file}.dat', train_caser.values, delimiter='::')
    print('Success to create santander.dat')
