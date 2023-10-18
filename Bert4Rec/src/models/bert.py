from typing import Any, List
from pytorch_lightning import LightningModule
import torch.nn as nn
import torch
from src.utils.metrics import calculate_loss, calculate_metrics
from .bert_modules.bert import BERT


def get_optimizer(name, model_params, params):
    """

    Args:
        name: имя оптимайзера
        model_params: параметры модели
        params: lr, weight decay

    Returns:
    Возвращем нужную функцию
    """
    optimizers = {
        "adadelta": torch.optim.Adadelta,
        "adagrad": torch.optim.Adagrad,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sparseadam": torch.optim.SparseAdam,
        "adamax": torch.optim.Adamax,
        "asgd": torch.optim.ASGD,
        "lbfgs": torch.optim.LBFGS,
        "nadam": torch.optim.NAdam,
        "radam": torch.optim.RAdam,
        "rmsprop": torch.optim.RMSprop,
        "rprop": torch.optim.Rprop,
        "sgd": torch.optim.SGD
    }
    return optimizers[name](params=model_params, **params)

class BERTModel(LightningModule):
    """
    Класс вызова модели Bert и функции для тренировки в пайплайне с pytorch-lightning
    """
    def __init__(
            self,
            model_init_seed,
            bert_max_len,
            num_items,
            bert_num_blocks,
            bert_num_heads,
            bert_hidden_units,
            bert_dropout,
            lr,
            weight_decay,
            cand_need,
            k_labels
    ):
        super().__init__()
        self.lr = lr
        self.cand_need = cand_need
        self.k_labels = k_labels
        self.weight_decay = weight_decay
        self.bert = BERT(model_init_seed, bert_max_len, num_items, bert_num_blocks,
                         bert_num_heads, bert_hidden_units, bert_dropout)
        self.out = nn.Linear(self.bert.hidden, num_items + 1)

    def forward(self, x):
        x = self.bert(x)
        return self.out(x)

    def step(self, batch: Any, stage: str):
        """

        Args:
            batch: батч, который состоит из последовательности категорий и
            последовательности истинных лейблов,
            в случае валидации и инференса добавляется последовательность кандидатов
            stage: трейн или тест(валид)

        Returns:
        В случае трейна возваращем выход модели, лосс, истинные итемы
        В случае теста - выход модели и метрики
        """

        if stage == 'train':
            seqs, labels = batch
            outputs = self.forward(seqs)
            loss = calculate_loss(outputs, labels)
            return loss, outputs, labels
        if stage == 'val' or stage == 'test':
            seqs, candidates, labels = batch
            outputs = self.forward(seqs)
            metrics = calculate_metrics(outputs, candidates, labels, stage, self.cand_need,
                                        self.k_labels)
            return outputs, metrics

    def training_step(self, batch: Any, batch_idx: int):
        """

        Args:
            batch: батч, который состоит из последовательности категорий и
            последовательности истинных лейблов

        Returns:
        В логи передаем лосс, можно использовать cometa для просмотра качества обучения
        """
        loss, preds, targets = self.step(batch, 'train')
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        """

        Args:
            batch: батч, который состоит из последовательности категорий и
            последовательности истинных лейблов,
            также последовательность кандидатов

        Returns:
        Посчитанные метрики и предсказания
        """
        preds, metrics = self.step(batch, 'val')
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
            "test/Precision@1",
            metrics['Precision@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/Precision@3",
            metrics['Precision@3'],
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

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        """

        Args:
            batch: батч, который состоит из последовательности категорий и
            последовательности истинных лейблов,
            также последовательность кандидатов

        Returns:
        Посчитанные метрики и предсказания
        """
        preds, metrics = self.step(batch, 'test')
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
            "test/Precision@1",
            metrics['Precision@1'],
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "test/Precision@3",
            metrics['Precision@3'],
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

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        pass

    def configure_optimizers(self):
        return torch.optim.Adam(
            params=self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


