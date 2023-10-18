from .caser_modules.caser import Caser
from pytorch_lightning import LightningModule
import torch.nn as nn
from typing import Any, List, Optional
import torch
from src.utils.metrics import calculate_metrics
import pathlib
from dataclasses import dataclass

from src.datamodules.caser_modules.interactions import Interactions


@dataclass
class caser_model_args:
    """This dataclass contains Caser`s parameters.
    """
    L: int = 5
    d: int = 50
    nh: int = 16
    nv: int = 4
    drop: float = 0.5
    ac_conv: str = 'relu'
    ac_fc: str = 'relu'


class CaserModel(LightningModule):
    """
    https://github.com/graytowne/caser_pytorch#model-args-in-train_caserpy

    Parameters
    ----------
    data_dir : directory with dataset
    dataset_name : dataset`s name
    L : length of sequence
    d : number of latent dimensions
    nv : number of vertical filters
    nh : number of horizontal filters
    ac_conv : activation function for convolution layer (i.e., phi_c in paper)
    ac_fc : activation function for fully-connected layer (i.e., phi_a in paper)
    drop : drop ratio when performing dropout
    l2 : float
    learning_rate : float
    """

    def __init__(
            self,
            data_dir,
            dataset_name,
            dataset_name_test,
            L: int = 5,
            d: int = 50,
            nh: int = 16,
            nv: int = 4,
            drop: float = 0.5,
            ac_conv: str = "relu",
            ac_fc: str = "relu",
            l2: float = 1e-6,
            learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.l2 = l2
        self.learning_rate = learning_rate

        preprocessed_data_path_train = pathlib.Path(data_dir, "preprocessed", dataset_name)

        dataset_interactions = Interactions(preprocessed_data_path_train)

        self.num_items = dataset_interactions.num_items + 1  # add 0 padding
        self.num_users = dataset_interactions.num_users

        self.model_args = caser_model_args(L, d, nh, nv, drop, ac_conv, ac_fc)

        self.caser = Caser(self.num_users, self.num_items, self.model_args)

    def forward(self, seq_var, user_var, item_var, for_pred=False):
        x = self.caser(seq_var, user_var, item_var, for_pred)
        return x

    def calculate_loss(self, targets_prediction, negatives_prediction):
        positive_loss = -torch.mean(
            torch.log(torch.sigmoid(targets_prediction)))
        negative_loss = -torch.mean(
            torch.log(1 - torch.sigmoid(negatives_prediction)))
        loss = positive_loss + negative_loss
        return loss

    def step(self, batch: Any, stage: str):

        if stage == 'train':
            batch_sequences, batch_users, batch_targets, batch_negatives = batch

            items_to_predict = torch.cat((batch_targets, batch_negatives), 1)
            items_prediction = self.forward(batch_sequences,
                                            batch_users,
                                            items_to_predict)

            (targets_prediction,
             negatives_prediction) = torch.split(items_prediction,
                                                 [batch_targets.size(1),
                                                  batch_negatives.size(1)], dim=1)

            loss = self.calculate_loss(targets_prediction, negatives_prediction)

            return loss, targets_prediction, batch_targets

        if stage == 'val' or stage == 'test':
            sequences, user, items, test_targets = batch
            outputs = self.forward(sequences,
                                   user,
                                   items,
                                   for_pred=True)

            # loss = calculate_loss(outputs, labels)
            metrics = calculate_metrics(outputs, None,
                                        test_targets)
            return outputs, metrics

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch, 'train')
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        preds, metrics = self.step(batch, 'val')
        # self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "val/Recall@1",
            metrics['Recall@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/Recall@2",
            metrics['Recall@2'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/Recall@3",
            metrics['Recall@3'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/NDCG@1",
            metrics['NDCG@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/NDCG@2",
            metrics['NDCG@2'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/NDCG@3",
            metrics['NDCG@3'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/MRR@1",
            metrics['MRR@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/MRR@2",
            metrics['MRR@2'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "val/MRR@3",
            metrics['MRR@3'],
            on_epoch=True,
            prog_bar=True,
        )
        return {"preds": preds}

    def test_step(self, batch: Any, batch_idx: int):
        preds, metrics = self.step(batch, 'test')
        # self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/Recall@1",
            metrics['Recall@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/Recall@2",
            metrics['Recall@2'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/Recall@3",
            metrics['Recall@3'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/NDCG@1",
            metrics['NDCG@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/NDCG@2",
            metrics['NDCG@2'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/NDCG@3",
            metrics['NDCG@3'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/MRR@1",
            metrics['MRR@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/MRR@2",
            metrics['MRR@2'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/MRR@3",
            metrics['MRR@3'],
            on_epoch=True,
            prog_bar=True,
        )
        return {"preds": preds}

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.caser.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2,
        )
