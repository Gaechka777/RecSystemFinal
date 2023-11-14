import pathlib
from typing import Optional

import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass

from src.utils.data_utils import create_subsample, make_interactions_data, train_val_test_split

from .caser_modules.interactions import Interactions
from .caser_modules.utils import shuffle


@dataclass
class Arguments:
    """This dataclass contains additional parameters.
    """
    sequence_length: int = 5
    target_length: int = 1
    neg_samples: int = 3


class Dataset_caser(Dataset):
    """
    Simple Torch Dataset for Caser.

    Parameters
    ----------

    args : Arguments
        specified parameters for Caser preprocessing
    dataset_path : str
        path to train data
    mod : str
        "train", "test" or "val"
    dataset_path_test : str
        path to test data
    """
    def __init__(self, args, dataset_path, mod="train", dataset_path_test=None):
        self.args = args
        self._candidate = None
        self.sequence_length = args.sequence_length
        self.target_length = args.target_length
        self._neg_samples = args.neg_samples

        self.output_timestamps = True
        self.output_user = True

        self.dataset_interactions = Interactions(dataset_path)
        self.mod = mod

        # convert to sequences, targets and users

        self.dataset_interactions.to_sequence(self.sequence_length, self.target_length)
        self.test_sequence = self.dataset_interactions.test_sequences

        self._num_items = self.dataset_interactions.num_items
        self._num_users = self.dataset_interactions.num_users

        sequences_np = self.dataset_interactions.sequences.sequences
        targets_np = self.dataset_interactions.sequences.targets
        users_np = self.dataset_interactions.sequences.user_ids.reshape(-1, 1)

        users_np, sequences_np, targets_np = shuffle(users_np,
                                                     sequences_np,
                                                     targets_np)
        # we can omit that part for test mode
        negatives_np = self._generate_negative_samples(users_np, self.dataset_interactions, n=self._neg_samples)

        # convert numpy arrays to PyTorch tensors and move it to the corresponding devices
        self.users, self.sequences, self.targets, self.negatives = (torch.from_numpy(users_np).long(),
                                                                    torch.from_numpy(sequences_np).long(),
                                                                    torch.from_numpy(targets_np).long(),
                                                                    torch.from_numpy(negatives_np).long())
        if dataset_path_test:
            self.dataset_interactions_test = Interactions(dataset_path_test,
                                                          user_map=self.dataset_interactions.user_map,
                                                          item_map=self.dataset_interactions.item_map)

    def _generate_negative_samples(self, users, interactions, n):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}

        Parameters
        ----------

        users: array of np.int64
            sequence users
        interactions: :class:`spotlight.interactions.Interactions`
            training instances, used for generate candidates
        n: int
            total number of negatives to sample for each sequence
        """

        users_ = users.squeeze()
        negative_samples = np.zeros((users_.shape[0], n), np.int64)
        if not self._candidate:
            self._candidate = {}
            all_items = np.arange(interactions.num_items - 1) + 1  # 0 for padding
            train = interactions.tocsr()
            for user, row in enumerate(train):
                self._candidate[user] = list(set(all_items) - set(row.indices))

        for i, u in enumerate(users_):
            for j in range(n):
                x = self._candidate[u]
                negative_samples[i, j] = x[
                    np.random.randint(len(x))]

        return negative_samples

    def __len__(self):
        return self.test_sequence.sequences.shape[0] if self.mod != "train" else \
            self.dataset_interactions.sequences.sequences.shape[0]

    def __getitem__(self, index):
        res = ()
        if self.mod == "train":
            data = (self.users, self.sequences, self.targets, self.negatives)
            user, sequence, target, negative_sample = tuple(x[index] for x in data)

            # items_to_predict = torch.cat((target, negative_sample), 1)
            # res = (sequence, user, items_to_predict)

            res = (sequence, user, target, negative_sample)
        elif self.mod == "val":
            sequences_np = self.test_sequence.sequences[index, :]
            sequences_np = np.atleast_2d(sequences_np)

            # if item_ids is None:
            item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long().squeeze()
            item_ids = torch.from_numpy(item_ids).long().squeeze()
            user_id = torch.LongTensor(self.users[index])
            # print(self.users[index], user_id)

            mask = self.dataset_interactions_test.user_ids == index
            test_targets = self.dataset_interactions_test.item_ids[mask]
            test_targets = torch.LongTensor(test_targets)

            res = (sequences, user_id, item_ids, test_targets)
        elif self.mod == "test":
            sequences_np = self.test_sequence.sequences[index, :]
            sequences_np = np.atleast_2d(sequences_np)

            # if item_ids is None:
            item_ids = np.arange(self._num_items).reshape(-1, 1)

            sequences = torch.from_numpy(sequences_np).long().squeeze()
            item_ids = torch.from_numpy(item_ids).long().squeeze()
            user_id = user_id = torch.LongTensor(self.users[index])

            mask = self.dataset_interactions_test.user_ids == index
            test_targets = self.dataset_interactions_test.item_ids[mask]
            test_targets = torch.LongTensor(test_targets)

            res = (sequences, user_id, item_ids, test_targets)

        return res


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

    Parameters
    ----------

    data_dir : str
        dataset directory
    dataset_name : str
        train dataset`s name
    dataset_name_test : str
        test dataset`s name
    batch_size : int
        batch size for dataloader
    num_workers : int
        number of workers for dataloader
    pin_memory : bool
    sample_size : int
        the number of users whose will be taken from dataset
    min_data_points : int
        minimal number of user`s timestamps in sequence
    seed : int
    sequence_length : int
    target_length : int
    neg_samples : int
    """

    def __init__(
            self,
            data_dir: str = "data",
            dataset_name: str = "train_ver2.csv",
            dataset_name_test: str = "train_ver2.csv",
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False,
            sample_size: int = 2000,
            min_data_points: float = 17,
            seed: int = 0,
            sequence_length: int = 5,
            target_length: int = 1,
            neg_samples: int = 3
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.dataset_name_test = dataset_name_test

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        # self.train_val_test_split = train_val_test_split

        self.sample_size = sample_size
        self.min_data_points = min_data_points
        self.seed = seed

        self.args = Arguments(sequence_length, target_length, neg_samples)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def prepare_data(self):
        """Download data if needed.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """
        preprocessed_data_path = pathlib.Path(self.data_dir, "preprocessed", self.dataset_name)
        out_dir = str(pathlib.Path(self.data_dir, "preprocessed")) + '\\'
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
            train_val_test_split(preprocessed_data_path, out_dir)
        else:
            print("preparing data has already done!")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            preprocessed_data_path_train = pathlib.Path(self.data_dir, "preprocessed", "data_train.csv")
            preprocessed_data_path_test = pathlib.Path(self.data_dir, "preprocessed", "data_test.csv")
            preprocessed_data_path_val = pathlib.Path(self.data_dir, "preprocessed", "data_val.csv")

            self.data_train = Dataset_caser(self.args, preprocessed_data_path_train, "train")
            self.data_val = Dataset_caser(self.args, preprocessed_data_path_train, "val", preprocessed_data_path_val)
            self.data_test = Dataset_caser(self.args, preprocessed_data_path_train, "test", preprocessed_data_path_test)
            
            print("Train, validation and test datasets are ready for use!")

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
