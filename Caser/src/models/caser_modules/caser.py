import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import activation_getter


class Caser(nn.Module):
    """
    Convolutional Sequence Embedding Recommendation Model (Caser)[1].

    [1] Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding,
     Jiaxi Tang and Ke Wang , WSDM '18
     https://github.com/graytowne/caser_pytorch#model-args-in-train_caserpy

    Parameters
    ----------

    num_users: int,
        Number of users.
    num_items: int,
        Number of items.
    model_args: args,
        Model-related arguments, like latent dimensions.
    """

    def __init__(self, num_users, num_items, model_args):
        super(Caser, self).__init__()
        self.args = model_args

        # init args
        L = self.args.L
        dims = self.args.d
        self.n_h = self.args.nh
        self.n_v = self.args.nv
        self.drop_ratio = self.args.drop
        self.ac_conv = activation_getter[self.args.ac_conv]
        self.ac_fc = activation_getter[self.args.ac_fc]

        # user and item embeddings
        self.user_embeddings = nn.Embedding(num_users, dims)
        self.item_embeddings = nn.Embedding(num_items, dims)

        # additional information embedding
        #         self.sex_embeddings = nn.Embedding(2, 5)
        #         self.age_embeddings = nn.Embedding(7, 5)
        #         self.occupation_embeddings = nn.Embedding(21, 5)

        # vertical conv layer
        self.conv_v = nn.Conv2d(1, self.n_v, (L, 1))

        # horizontal conv layer
        lengths = [i + 1 for i in range(L)]
        self.conv_h = nn.ModuleList([nn.Conv2d(1, self.n_h, (i, dims)) for i in lengths])

        # fully-connected layer
        self.fc1_dim_v = self.n_v * dims
        self.fc1_dim_h = self.n_h * len(lengths)
        fc1_dim_in = self.fc1_dim_v + self.fc1_dim_h
        # W1, b1 can be encoded with nn.Linear
        self.fc1 = nn.Linear(fc1_dim_in, dims)
        # W2, b2 are encoded with nn.Embedding, as we don't need to compute scores for all items

        # self.W2 = nn.Embedding(num_items, dims + dims + 5)
        self.W2 = nn.Embedding(num_items, dims + dims)
        self.b2 = nn.Embedding(num_items, 1)

        # # fcl for additional information
        # self.fc2 = nn.Linear(5 + 5 + 5, 5)
        # # TransformerEncoderLayer  for additional information
        # self.transformer = nn.TransformerEncoderLayer(d_model=15, nhead=5)

        # dropout
        self.dropout = nn.Dropout(self.drop_ratio)

        # weight initialization
        self.user_embeddings.weight.data.normal_(0, 1.0 / self.user_embeddings.embedding_dim)
        self.item_embeddings.weight.data.normal_(0, 1.0 / self.item_embeddings.embedding_dim)
        #         self.sex_embeddings.weight.data.normal_(0, 1.0 / self.sex_embeddings.embedding_dim)
        #         self.age_embeddings.weight.data.normal_(0, 1.0 / self.age_embeddings.embedding_dim)
        #         self.occupation_embeddings.weight.data.normal_(0, 1.0 / self.occupation_embeddings.embedding_dim)
        self.W2.weight.data.normal_(0, 1.0 / self.W2.embedding_dim)
        self.b2.weight.data.zero_()

        self.cache_x = None

    def forward(self, seq_var, user_var, item_var, add_inf_path = 'datasets/ml1m/users.csv', for_pred=False):
        """
        The forward propagation used to get recommendation scores, given
        triplet (user, sequence, targets) and additional information about users like sex, ages and occupation.

        Parameters
        ----------

        seq_var: torch.FloatTensor with size [batch_size, max_sequence_length]
            a batch of sequence
        user_var: torch.LongTensor with size [batch_size]
            a batch of user
        item_var: torch.LongTensor with size [batch_size]
            a batch of items
        for_pred: boolean, optional
            Train or Prediction. Set to True when evaluation.
        """

        # Embedding Look-up
        item_embs = self.item_embeddings(seq_var).unsqueeze(1)  # use unsqueeze() to get 4-D
        user_emb = self.user_embeddings(user_var).squeeze(1)

        # add_inf = torch.from_numpy(pd.read_csv(add_inf_path).values.astype(np.int32)).int().to("cuda")
        #
        # additional_inf_emb = []
        # for i in user_var:
        #     # print(add_inf[i-1])
        #     # print(self.sex_embeddings(torch.tensor([1])))
        #     additional_inf_emb.append(torch.cat((self.sex_embeddings(add_inf[i-1, 1]),
        #                                          self.age_embeddings(add_inf[i-1, 2]),
        #                                          self.occupation_embeddings(add_inf[i-1, 3])), 0).reshape(1,-1))
        # additional_inf_emb = torch.cat(additional_inf_emb, 0)  # shape = (user_var.shape, 3*5)

        # Convolutional Layers
        out, out_h, out_v = None, None, None
        # vertical conv layer
        if self.n_v:
            out_v = self.conv_v(item_embs)
            out_v = out_v.view(-1, self.fc1_dim_v)  # prepare for fully connect

        # horizontal conv layer
        out_hs = list()
        if self.n_h:
            for conv in self.conv_h:
                conv_out = self.ac_conv(conv(item_embs).squeeze(3))
                pool_out = F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
                out_hs.append(pool_out)
            out_h = torch.cat(out_hs, 1)  # prepare for fully connect

        # Fully-connected Layers
        out = torch.cat([out_v, out_h], 1)
        # apply dropout
        out = self.dropout(out)

        # fully-connected layer
        z = self.ac_fc(self.fc1(out))
        # additional layer
        # add_layer =  self.ac_fc(self.fc2(self.transformer(additional_inf_emb.unsqueeze(0)).squeeze(0)))

        # x = torch.cat([z, user_emb, add_layer], 1)
        x = torch.cat([z, user_emb], 1)

        # if for_pred == False:
        #     for i in range(item_var.shape[0]):
        #         print(item_var[i])
        #         print(self.W2(item_var[i]))
        #         print("-----------------------------------------")
        w2 = self.W2(item_var)
        b2 = self.b2(item_var)

        if for_pred:
            w2 = w2.squeeze()
            b2 = b2.squeeze()
            # print(x.shape)
            # print(w2.shape)
            # print(b2.shape)
            res = (x * w2).sum(1) + b2
        else:
            res = torch.baddbmm(b2, w2, x.unsqueeze(2)).squeeze()

        return res
