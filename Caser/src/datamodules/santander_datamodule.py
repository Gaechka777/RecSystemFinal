import pathlib
from typing import Optional, Tuple, List

import torch
import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from datetime import date
from dotmap import DotMap
from dataclasses import dataclass

# from tqdm.notebook import tqdm
# tqdm.pandas()



from src.utils.data_utils import create_subsample, make_interactions_data

# dataset = {'user2dict': user2dict,
#            'train_targets': train_range,
#            'validation_targets': validation_range,
#            'test_targets': test_range,
#            'umap': umap,
#            'smap': smap}
# df, umap, smap = self.densify_index(df)
# user2dict, train_targets, validation_targets, test_targets = self.split_df(df, len(umap))


@dataclass
class Arguments:
    """This dataclass contains add parametrs.
   train_window:"How much to slide the training window to obtain subsequences from the user's entire item sequence"
   max_len: 'Length of the transformer model'
    """
    train_window: int = 100
    max_len: int = 200
        

def densify_index(df):
#     print('Densifying index')
    umap = {u: (i+1) for i, u in enumerate(set(df["USER_ID"]))}
    smap = {s: (i+1) for i, s in enumerate(set(df["ITEM_ID"]))}
    df["USER_ID"] = df["USER_ID"].map(umap)
    df["ITEM_ID"] = df["ITEM_ID"].map(smap)
    return df, umap, smap


def split_df(df):
    def sort_by_time(d):
        d = d.sort_values(by="TIMESTAMP")
        return {'items': list(d["ITEM_ID"]), 'timestamps': list(d["TIMESTAMP"]), 'ratings': list(d["RATING"])}

    min_date = date.fromtimestamp(df["TIMESTAMP"].min())
    user_group = df.groupby("USER_ID")
#     user2dict = user_group.progress_apply(sort_by_time)
    user2dict = user_group.apply(sort_by_time)

    train_ranges = []
    val_positions = []
    test_positions = []
    for user, d in user2dict.items():
        n = len(d['items'])
        train_ranges.append((user, n-2))  # exclusive range
        val_positions.append((user, n-2))
        test_positions.append((user, n-1))
    train_ranges = train_ranges
    validation_ranges = val_positions
    test_ranges = test_positions

    return user2dict, train_ranges, validation_ranges, test_ranges

# class Dataset_RNN(Dataset):
#     """
#     Simple Torch Dataset for many-to-many RNN
#         celled_data: source of data,
#         start_date: start date index,
#         end_date: end date index,
#         periods_forward: number of future periods for a target,
#         history_length: number of past periods for an input,
#         transforms: input data manipulations
#     """
#
#     def __init__(
#         self,
#         celled_data: torch.Tensor,
#         features_list: List[str],
#     ):
#         # clean up data - choose only needed columns
#         self.data = celled_data[:, features_list]
#
#         self.target = self.data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         input_tensor = self.data[idx : idx + self.history_length]
#
#         target = self.target[
#             idx + self.history_length : idx + self.history_length + self.periods_forward
#         ]
#
#         return (
#             input_tensor,
#             target,
#         )



class Dataset_SRS(Dataset):
    """
    Simple Torch Dataset for many-to-many RNN
        celled_data: source of data,
        start_date: start date index,
        end_date: end date index,
        periods_forward: number of future periods for a target,
        history_length: number of past periods for an input,
        transforms: input data manipulations
    """
    def __init__(self, args, dataset, train_ranges):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.users = sorted(self.user2dict.keys())
        self.train_window = args.train_window
        self.max_len = args.max_len
#         self.mask_prob = args.mask_prob
        # self.special_tokens = dataset['special_tokens']
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        # self.rng = rng
        self.train_ranges = train_ranges

        self.index2user_and_offsets = self.populate_indices()

        self.output_timestamps = True
#         # self.output_days = args.dataloader_output_days
        self.output_user = True

        # self.negative_samples = negative_samples

    # def get_rng_state(self):
    #     return self.rng.getstate()
    #
    # def set_rng_state(self, state):
    #     return self.rng.setstate(state)

    def populate_indices(self):
        index2user_and_offsets = {}
        i = 0
        T = self.max_len
        W = self.train_window

        # offset is exclusive
        for user, pos in self.train_ranges:
            if W is None or W == 0:
                offsets = [pos]
            else:
                offsets = list(range(pos, T-1, -W))  # pos ~ T
                if len(offsets) == 0:
                    offsets = [pos]
            for offset in offsets:
                index2user_and_offsets[i] = (user, offset)
                i += 1
        return index2user_and_offsets

    def __len__(self):
        return len(self.index2user_and_offsets)

    def __getitem__(self, index):
        user, offset = self.index2user_and_offsets[index]
        seq = self.user2dict[user]['items']
        beg = max(0, offset-self.max_len)
        end = offset  # exclude offset (meant to be)
        seq = seq[beg:end]

        # tokens = []
        # labels = []
        # for s in seq:
        #     prob = self.rng.random()
        #     if prob < self.mask_prob:
        #         prob /= self.mask_prob
        #
        #         if prob < 0.8:
        #             tokens.append(self.special_tokens.mask)
        #         elif prob < 0.9:
        #             tokens.append(self.rng.randint(1, self.num_items))
        #         else:
        #             tokens.append(s)
        #
        #         labels.append(s)
        #     else:
        #         tokens.append(s)
        #         labels.append(0)

        tokens = seq[-self.max_len:]
        # labels = labels[-self.max_len:]

        padding_len = self.max_len - len(tokens)
        valid_len = len(tokens)

        tokens = [0] * padding_len + tokens
        # labels = [0] * padding_len + labels

        d = {}
        d['tokens'] = torch.LongTensor(tokens)
        # d['labels'] = torch.LongTensor(labels)

        if self.output_timestamps:
            timestamps = self.user2dict[user]['timestamps']
            timestamps = timestamps[beg:end]
            timestamps = [0] * padding_len + timestamps
            d['timestamps'] = torch.LongTensor(timestamps)

        # if self.output_days:
        #     days = self.user2dict[user]['days']
        #     days = days[beg:end]
        #     days = [0] * padding_len + days
        #     d['days'] = torch.LongTensor(days)

        if self.output_user:
            d['users'] = torch.LongTensor([user])
        return d


# class BertEvalDataset(data_utils.Dataset):
#     def __init__(self, args, dataset, negative_samples, positions):
#         self.user2dict = dataset['user2dict']
#         self.positions = positions
#         self.max_len = args.max_len
#         self.num_items = len(dataset['smap'])
#         self.special_tokens = dataset['special_tokens']
#         self.negative_samples = negative_samples
#
#         self.output_timestamps = args.dataloader_output_timestamp
#         self.output_days = args.dataloader_output_days
#         self.output_user = args.dataloader_output_user
#
#     def __len__(self):
#         return len(self.positions)
#
#     def __getitem__(self, index):
#         user, pos = self.positions[index]
#         seq = self.user2dict[user]['items']
#
#         beg = max(0, pos + 1 - self.max_len)
#         end = pos + 1
#         seq = seq[beg:end]
#
#         negs = self.negative_samples[user]
#         answer = [seq[-1]]
#         candidates = answer + negs
#         labels = [1] * len(answer) + [0] * len(negs)
#
#         seq[-1] = self.special_tokens.mask
#         padding_len = self.max_len - len(seq)
#         seq = [0] * padding_len + seq
#
#         tokens = torch.LongTensor(seq)
#         candidates = torch.LongTensor(candidates)
#         labels = torch.LongTensor(labels)
#         d = {'tokens':tokens, 'candidates':candidates, 'labels':labels}
#
#         if self.output_timestamps:
#             timestamps = self.user2dict[user]['timestamps']
#             timestamps = timestamps[beg:end]
#             timestamps = [0] * padding_len + timestamps
#             d['timestamps'] = torch.LongTensor(timestamps)
#
#         if self.output_days:
#             days = self.user2dict[user]['days']
#             days = days[beg:end]
#             days = [0] * padding_len + days
#             d['days'] = torch.LongTensor(days)
#
#         if self.output_user:
#             d['users'] = torch.LongTensor([user])
#         return d



class SantanderDataModule(LightningDataModule):
    """LightningDataModule for Santander dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = "data",
        dataset_name: str = "train_ver2.csv",
        train_val_test_split: Tuple[float] = (0.8, 0.1, 0.1),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        sample_size: int = 2000,
        min_data_points: float = 17,
        seed: int = 0,
        train_window: int = 100,
        max_len: int = 200
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.dataset_name = dataset_name

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_val_test_split = train_val_test_split
        
        self.sample_size = sample_size
        self.min_data_points = min_data_points
        self.seed = seed
        
        self.args = Arguments(train_window, max_len)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        preprocessed_data_path = pathlib.Path(self.data_dir, "preprocessed", self.dataset_name)
        data_path = pathlib.Path(self.data_dir, self.dataset_name)
        if not preprocessed_data_path.is_file():
            create_subsample(
                data_path,
                self.sample_size,
                self.min_data_points,
                self.seed
            )
            make_interactions_data(str(data_path).split(".csv")[0] + "_reduced.csv",
                                   preprocessed_data_path)


    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            preprocessed_data_path = pathlib.Path(self.data_dir, "preprocessed", self.dataset_name)
#             preprocessed_data = torch.load(preprocessed_data_path)
            preprocessed_data = pd.read_csv(preprocessed_data_path,sep = " ")
            
            preprocessed_data, umap, smap = densify_index(preprocessed_data)
            user2dict, train_range, validation_range, test_range = split_df(preprocessed_data)
            
            dataset = {'user2dict': user2dict,
                       'train_targets': train_range,
                       'validation_targets': validation_range,
                       'test_targets': test_range,
                       'umap': umap,
                       'smap': smap}
            
            # now we can free preprocessed_data
            
            
            self.data_train = Dataset_SRS(self.args, dataset, train_range)
            self.data_val = Dataset_SRS(self.args, dataset, validation_range)
            self.data_test = Dataset_SRS(self.args, dataset, test_range)

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )