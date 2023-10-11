# There are contained two main functions for preparing Santander dataset: create_subsample and make_interactions_data
import time
import datetime
import pandas as pd
import numpy as np


target_cols = ['ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1', 'ind_cno_fin_ult1',
               'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1',
               'ind_deme_fin_ult1', 'ind_dela_fin_ult1', 'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
               'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
               'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1']
target_cols = target_cols[2:]


def median_income(df):
    """
    Calculate median income and fill Nan by this number
    Parameters
    ----------
    df : pd.DataFrame
        dataset
    """
    df.loc[df.renta.isnull(), 'renta'] = df.renta.median(skipna=True)
    return df


def make_interactions(df, output_interactions_data):
    """

    Parameters
    ----------
    df : pd.DataFrame
        dataset
    output_interactions_data : str
        path for saving interactions data
    """
    df_interactions_vector = df[["fecha_dato", "ncodpers"] + target_cols]
    df_interactions_index = pd.DataFrame(columns=["TIMESTAMP", "USER_ID", "ITEM_ID"])

    def get_user_product(row):
        if not hasattr(get_user_product, "count"):
            get_user_product.count = 0
            get_user_product.count_rows = 0
        for i, c in enumerate(df_interactions_vector.columns[2:]):
            if row[c] == 1:
                pass
                timestamp = int(time.mktime(datetime.datetime.strptime(row["fecha_dato"], "%Y-%m-%d").timetuple()))
                df_interactions_index.loc[get_user_product.count] = [timestamp, row["ncodpers"], i + 1]
                get_user_product.count += 1
        get_user_product.count_rows += 1
        print('progress:', round(get_user_product.count_rows / df_interactions_vector.shape[0], 3) * 100, "%")

    df_interactions_vector.apply(get_user_product, axis=1)
    df_interactions_index["ITEM_ID"] = df_interactions_index["ITEM_ID"].astype(int)
    df_interactions_index["USER_ID"] = df_interactions_index["USER_ID"].astype(int)
    df_interactions_index["TIMESTAMP"] = df_interactions_index["TIMESTAMP"].astype(int)
    print(df_interactions_index)

    # python preprocess_interactions_data.py
    n_rows = df_interactions_index.shape[0]
    ratings = np.array([1] * n_rows)
    df_interactions_index['RATING'] = ratings
    df_interactions_index = df_interactions_index[["USER_ID", "ITEM_ID", "RATING", "TIMESTAMP"]]
    # df_interactions_index = df_interactions_index[["USER_ID", "ITEM_ID", "RATING"]]
    df_interactions_index.to_csv(output_interactions_data, index=False, sep=" ")
    #     torch.save(df_interactions_index, output_interactions_data)
    print('created file', output_interactions_data)


def create_subsample(input_file: str = "train.csv",
                     sample_size: int = 200000,
                     min_data_points: float = 17,
                     seed: int = 0):
    """
    Parameters
    ----------
    input_file : str
        csv file containing the original Santander product recommendation data.
        - can be downloaded from https://www.kaggle.com/c/santander-product-recommendation/data.
    sample_size : int
        number of users to be sub-sampled from the full dataset (if None use the full data).
    min_data_points : float
        filter users having less than min_data_points records (0 if don't want to filter).
        If 'sample_size' is not None or 'min_data_points' > 0, it saves the sub-sampled dataset:
        - dataset_reduced: sub-sampled dataset containing (sample_size) users,
          each of them having at least 'min_data_points' timestamps.
    """
    df = pd.read_csv(input_file)
    assert sample_size > 0, "sample_size param should be > 0"
    assert min_data_points >= 0, "min_data_points param should be >= 0"
    np.random.seed(seed)

    # get distinct
    def distinct_timestamps(x):
        if min_data_points and len(x.fecha_dato.unique()) >= min_data_points:
            return 1
        return np.nan

    df.dropna(subset=['ncodpers'], inplace=True)
    df.dropna(subset=['fecha_dato'], inplace=True)
    df.dropna(subset=['cod_prov'], inplace=True)
    df_users = df.groupby('ncodpers').apply(distinct_timestamps)
    df_users = pd.DataFrame({'ncodpers': df_users.index, 'values': df_users.values})
    df_users.dropna(inplace=True)
    if sample_size:
        print(f"you have took {sample_size} users from {len(df_users)} possible users")
        df_users = df_users.sample(n=sample_size)
    df = df[df.ncodpers.isin(df_users.ncodpers)]
    print("dataset reduced to " + str(df.shape[0]) + " entries")

    if sample_size or min_data_points:
        df.to_csv(str(input_file).split(".csv")[0] + "_reduced.csv", index=False)
    print("process done")


def make_interactions_data(input_file: str = "train_reduced.csv",
                           output_file: str = "interactions_data_train.csv"):
    """
    input_file: csv file containing the original Santander product recommendation data.
    - can be downloaded from https://www.kaggle.com/c/santander-product-recommendation/data.
    It generates the following file.
    - interactions_data.csv: csv file containing users interactions
    """
    df = pd.read_csv(input_file)

    # preprocess
    # provide median income by province
    df = df.groupby('nomprov').apply(median_income)
    df.loc[df.renta.isnull(), "renta"] = df.renta.median(skipna=True)

    # make datasets
    print("making interactions data...")
    make_interactions(df, output_file)
    print("process done")


def train_val_test_split(input_file: str = "preprocessed\\train_ver2.csv",
                         output_path="preprocessed\\"):
    data = pd.read_csv(input_file, sep=' ')

    print("making data split...")

    data = data.sort_values(["USER_ID", 'TIMESTAMP'])
    quants = data["USER_ID"].value_counts()
    low_users = quants[quants < 9].index
    data = data[~data["USER_ID"].isin(low_users)]

    def get_subset(x, phase="test"):
        if phase == "train":
            res = x.loc[x.index[:-6]]
        elif phase == "test":
            res = x.loc[x.index[-3:]]
        else:
            res = x.loc[x.index[-6:-3]]
        return res[["ITEM_ID", "RATING", "TIMESTAMP"]]

    data_test = data.groupby('USER_ID').apply(get_subset, phase="test")
    data_train = data.groupby('USER_ID').apply(get_subset, phase="train")
    data_val = data.groupby('USER_ID').apply(get_subset, phase="val")

    data_test = data_test.reset_index()[["USER_ID", "ITEM_ID", "RATING", "TIMESTAMP"]]
    data_train = data_train.reset_index()[["USER_ID", "ITEM_ID", "RATING", "TIMESTAMP"]]
    data_val = data_val.reset_index()[["USER_ID", "ITEM_ID", "RATING", "TIMESTAMP"]]

    data_test.to_csv(output_path + 'data_test.csv', index=False, header=False, sep=" ")
    data_train.to_csv(output_path + 'data_train.csv', index=False, header=False, sep=" ")
    data_val.to_csv(output_path + 'data_val.csv', index=False, header=False, sep=" ")

    print("process done")
