import random
from typing import Optional
import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
from src.datasets.create_dataset import create
from src.datasets.santander import load_dataset, _get_preprocessed_folder_path
from .negative_samplers.random import get_negative_samples


class BertTrainDataset(Dataset):
    """
    Simple Torch Train Dataset for Bert.

    Args:
        u2seq: a dictionary of users, where a sequence of tokens (items) is recorded for each user
        max_len: the length of the history filling, since the sequence length is different for users
        mask_prob: the probability of choosing a particular token
        mask_token: mask for the labels we want to predict
        num_items: number of items
        rng: random probability generator
    """
    def __init__(self, u2seq, max_len, mask_prob, mask_token, num_items, rng):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.num_items = num_items
        self.rng = rng

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]

        tokens = []
        labels = []
        for s in seq:
            prob = self.rng.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob

                if prob < 0.8:
                    tokens.append(self.mask_token)
                elif prob < 0.9:
                    tokens.append(self.rng.randint(1, self.num_items))
                else:
                    tokens.append(s)

                labels.append(s)
            else:
                tokens.append(s)
                labels.append(0)

        tokens = tokens[-self.max_len:]
        labels = labels[-self.max_len:]

        mask_len = self.max_len - len(tokens)

        tokens = [0] * mask_len + tokens
        labels = [0] * mask_len + labels
        return torch.LongTensor(tokens), torch.LongTensor(labels)


class BertEvalDataset(Dataset):
    """
    Simple Torch Valid and Test Dataset for Bert.

    Args:
        u2seq: a dictionary of users, where a sequence of tokens (items) is recorded for each user
        users: list of user's keys
        u2answer: answer for each user key
        max_len: the length of the history filling, since the sequence length is different for users
        mask_token: mask for the labels we want to predict
        negative_samples: negative samples
        k_labels: k labels for prediction
        min_uc: min number of items for users

    """
    def __init__(self, u2seq, u2answer, max_len, mask_token, negative_samples, k_labels, min_uc):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.max_len = max_len
        self.mask_token = mask_token
        self.negative_samples = negative_samples
        self.k_labels = k_labels
        self.min_uc = min_uc

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        user = self.users[index]
        seq = self.u2seq[user]
        answer = self.u2answer[user]
        negs = self.negative_samples[user]
        candidates = answer + negs
        labels = [1] * len(answer) + [0] * len(negs)
        zipped = list(zip(candidates, labels))
        random.shuffle(zipped)
        candidates, labels = zip(*zipped)
        seq = seq + list(range(self.mask_token, self.mask_token + self.k_labels))#[self.mask_token, self.mask_token + 1, self.mask_token + 2]
        seq = seq[-self.max_len:]
        padding_len = self.max_len - len(seq)
        seq = [0] * padding_len + seq

        return torch.LongTensor(seq), torch.LongTensor(candidates), torch.LongTensor(labels)


class BertDataModule(LightningDataModule):
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


        Args:
            name_file: name file which was converted from the init file
            data_path: path to data
            init_file: original file
            k_labels: k labels for prediction
            min_uc: min number of items for users
            seed: seed for model
            bert_max_len: the length of the history filling, since the sequence length is different for users
            bert_mask_prob: the probability of choosing a particular token
            train_negative_sample_size: number of negative samples
            train_negative_sampling_seed: seed for random
            test_negative_sample_size: number of negative samples
            test_negative_sampling_seed: seed for random
            train_batch_size: size for train batch
            val_batch_size: size for valid batch
            test_batch_size: size for test batch
            num_workers: number of workers for dataloader
            pin_memory: used as a staging area for transfers from the device to the host
    """
    def __init__(self,
                 name_file,
                 data_path,
                 init_file,
                 k_labels,
                 min_uc,
                 seed,
                 bert_max_len,
                 bert_mask_prob,
                 train_negative_sample_size,
                 train_negative_sampling_seed,
                 test_negative_sample_size,
                 test_negative_sampling_seed,
                 train_batch_size,
                 val_batch_size,
                 test_batch_size,
                 num_workers,
                 pin_memory):
        super().__init__()
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.save_folder = _get_preprocessed_folder_path(data_path)
        self.k_labels = k_labels
        self.min_uc = min_uc
        create(data_path + init_file, name_file)
        dataset = load_dataset(data_path, name_file, k_labels, min_uc)

        self.train = dataset['train']
        self.val = dataset['val']
        self.test = dataset['test']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.user_count = len(self.umap)
        self.item_count = len(self.smap)
        self.cloze_mask_token = self.item_count + 1
        self.train_negative_sample_size = train_negative_sample_size
        self.test_negative_sample_size = test_negative_sample_size
        self.train_negative_sampling_seed = train_negative_sampling_seed
        self.test_negative_sampling_seed = test_negative_sampling_seed

        self.train_negative_samples = get_negative_samples(self.train, self.val, self.test,
                                                           self.user_count, self.item_count,
                                                           self.train_negative_sample_size,
                                                           self.train_negative_sampling_seed,
                                                           self.save_folder)
        self.test_negative_samples = get_negative_samples(self.train, self.val, self.test,
                                                          self.user_count, self.item_count,
                                                          self.test_negative_sample_size,
                                                          self.test_negative_sampling_seed,
                                                          self.save_folder)

        self.max_len = bert_max_len
        self.mask_prob = bert_mask_prob
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size

        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    # define any setup computations

    def prepare_data(self):
        pass

    # download data if applicable

    def setup(self, stage=None):
        pass

    # assign data to `Dataset`(s)

    def train_dataloader(self):
        self.data_train = BertTrainDataset(self.train, self.max_len,
                                           self.mask_prob, self.cloze_mask_token,
                                           self.item_count, self.rng)
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        self.data_val = BertEvalDataset(self.train, self.val, self.max_len,
                                        self.cloze_mask_token,
                                        self.test_negative_samples, self.k_labels, self.min_uc)
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.val_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        self.data_test = BertEvalDataset(self.train, self.test, self.max_len,
                                         self.cloze_mask_token,
                                         self.test_negative_samples, self.k_labels, self.min_uc)
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.test_batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
