"""
For this voice assistant system, we are using most advanced RNN model : Transformers.
In particular, we are applying the Streaming Transformers from the paper 'Wake Word Detection with Streaming Transformers' [1]

Bibliography
@article{wang2021wake,
      title={Wake Word Detection with Streaming Transformers}, 
      author={Yiming Wang and Hang Lv and Daniel Povey and Lei Xie and Sanjeev Khudanpur},
      journal={ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
      year={2021}
}

Some parts of this code is from Michael Nguyen:
https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant.git

and from Aladdin Persson:
https://github.com/aladdinpersson/Machine-Learning-Collection.git
"""

import torch

from numpy.lib.arraypad import pad
from torch.functional import Tensor
from torch.nn import Dropout, Linear, LSTM, LayerNorm, Module, ReLU, Sequential, Sigmoid
from torch.nn.modules.activation import Softmax
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.flatten import Flatten
from torch.nn.modules.pooling import MaxPool2d
from torchsummary import summary


class LSTMBinaryClassifier(Module):
    def __init__(self, feature_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float) -> None:
        super(LSTMBinaryClassifier, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.layernorm = LayerNorm(feature_size)
        self.lstm = LSTM(
            input_size=feature_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.classifier = Sequential(
            Linear(hidden_size, num_classes), Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x.view(-1)


class SelfAttention(Module):
    def __init__(self, embed_size: int, heads: int) -> None:
        """
        Embed_size must be devided by heads
        """
        super(SelfAttention, self).__init__()
        self.emded_size = embed_size
        self.heads = heads
        self.heads_dim = embed_size//heads

        self.values = Linear(self.heads_dim, self.heads_dim, bias=False)
        self.keys = Linear(self.heads_dim, self.heads_dim, bias=False)
        self.queries = Linear(self.heads_dim, self.heads_dim, bias=False)

        self.fc_out = Linear(embed_size, embed_size)

    def foward(self, values: torch.Tensor, keys: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.heads_dim)
        keys = keys.reshape(N, key_len, self.heads, self.heads_dim)
        query = query.reshape(N, query_len, self.heads, self.heads_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(query)

        # Einsum does matrix mult. for query*keys for each training example
        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries_shape:[n:N, q:query_len, h:heads, d:heads_dim]
        # keys_shape:[n:N, k:key_len, h:heads, d:heads_dim]
        # energy_shape:[n:N, h:heads, q:query_len, k:key_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20)'))

        attention = torch.softmax(energy/(self.emded_size)**(1/2), dim=3)

        # Einsum does matrix mult. for softmax*values for each training example
        out = torch.einsum("nhqk,nvhd->nqhd", attention,
                           values).reshape(N, query_len, self.emded_size)
        # attention_shape:[n:N, h: heads, q:query_len, k:key_len]
        # values_shape:[n:N, v:value_len, h:heads, d: heads_dim]
        # out_shape:[n:N, q:query_len, h:heads, d:heads_dim]
        # Then reshape and flattent the last 2 dimensions

        # shape:[n:N, q:query_len, h:heads, d:heads_dim]
        return self.fc_out(out)


class TransformerBlock(Module):
    def __init__(self, embed_size: int, heads: int, dropout, forward_expension: int) -> None:
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)

        self.norm1 = LayerNorm(embed_size)
        self.norm2 = LayerNorm(embed_size)

        self.feed_forward = Sequential(
            Linear(embed_size, forward_expension*embed_size),
            ReLU(),
            Linear(forward_expension*embed_size, embed_size)
        )
        self.dropout = Dropout(dropout)

    def forward(self, value: torch.Tensor, key: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention+query))
        forward = self.feed_forward(attention)
        out = self.dropout(self.norm2(forward+x))
        return out


class CNNNetwork(Module):
    def __init__(self)->None:
        super(CNNNetwork, self).__init__()
        self.conv1 = Sequential(
            Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        self.conv2 = Sequential(
            Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        self.conv3 = Sequential(
            Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            ReLU(),
            MaxPool2d(kernel_size=2)
        )
        # self.conv4=Sequential(
        #    Conv2d(
        #        in_channels=64,
        #        out_channels=128,
        #        kernel_size=3,
        #        stride=1,
        #        padding=2
        #    ),
        #    ReLU(),
        #    MaxPool2d(kernel_size=2)
        # )
        self.flatten = Flatten()
        self.linear = Linear(64*3*15, 2)
        self.softmax = Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x=self.conv4(x)
        x = self.flatten(x)
        x = self.linear(x)
        predictions = self.softmax(x)
        return predictions


def main() -> None:
    cnn = CNNNetwork()
    summary(cnn, (1, 64, 44))


if __name__ == "__main__":
    main()
