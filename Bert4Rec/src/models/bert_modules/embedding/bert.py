import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    Эмбеддинг BERT, которое состоит из нижеприведенных функций
        1. Представление токена: обычная матрица эмбеддингов
        2. Позиционное представление: добавление позиционной информации
        с использованием sin, cos
        2. Эмбеддинг сегмента: добавление segmentinfo предложения, (sent_A:1, cent_B:2)
        (не используем)
        сумма всех этих функций является представлением BERT
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """

        Args:
            vocab_size: общий размер словаря
            embed_size: размер представления токена
            max_len: максимальная длина последовательности
            dropout: процент разреживания
        """

        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
