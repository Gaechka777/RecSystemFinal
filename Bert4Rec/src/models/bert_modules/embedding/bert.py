import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding


class BERTEmbedding(nn.Module):
    """
    Embedding BERT, which consists of the following functions
        1. Token representation: the usual embedding matrix
        2. Positional representation: Adding positional information
        using sin, cos
        3. Embedding a segment: adding segment info sentences, (sent_A:1, cent_B:2) (do not use)

    The sum of all these functions is a representation of BERT
    """

    def __init__(self, vocab_size, embed_size, max_len, dropout=0.1):
        """

        Args:
            vocab_size: the total size of the dictionary
            embed_size: the size of the token representation
            max_len: maximum sequence length
            dropout: value for dropout function
        """

        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence):
        x = self.token(sequence) + self.position(sequence)
        return self.dropout(x)
