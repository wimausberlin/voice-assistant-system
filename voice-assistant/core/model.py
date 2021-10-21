from torch.functional import Tensor
from torch.nn import Dropout, Linear, LSTM, LayerNorm, Module, ReLU, Sequential

import torch
from torch.nn.modules import dropout


class LSTMBinaryClassifier(Module):
    def __init__(self, feature_size: int, hidden_size: int, num_layers: int, num_classes: int, dropout: float) -> None:
        super(LSTMBinaryClassifier, self).__init__()
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = LSTM(
            input_size=feature_size,
            hidden_size=feature_size,
            num_layers=num_layers,   
            dropout=self.dropout,
        )   
        self.classifier = Linear(feature_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layernorm(x)
        out, (hn, cn) = self.lstm(x)
        x = self.classifier(hn[-1])
        x = self.classifier(out)
        return x

class SelfAttention(Module):
    def __init__(self,embed_size:int,heads:int)->None:
        """
        Embed_size must be devided by heads
        """
        super(SelfAttention,self).__init__()
        self.emded_size=embed_size
        self.heads=heads
        self.heads_dim=embed_size//heads

        self.values=Linear(self.heads_dim,self.heads_dim,bias=False)
        self.keys=Linear(self.heads_dim,self.heads_dim,bias=False)
        self.queries=Linear(self.heads_dim,self.heads_dim,bias=False)
        
        self.fc_out=Linear(embed_size,embed_size)
    
    def foward(self,values:torch.Tensor,keys:torch.Tensor,query:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        N=query.shape[0]
        value_len,key_len,query_len=values.shape[1],keys.shape[1],query.shape[1]

        #Split embedding into self.heads pieces
        values=values.reshape(N,value_len,self.heads,self.heads_dim)
        keys=keys.reshape(N,key_len,self.heads,self.heads_dim)
        query=query.reshape(N,query_len,self.heads,self.heads_dim)

        values=self.values(values)
        keys=self.keys(keys)
        queries=self.queries(query)

        # Einsum does matrix mult. for query*keys for each training example
        energy=torch.einsum("nqhd,nkhd->nhqk",queries,keys)
        # queries_shape:[n:N, q:query_len, h:heads, d:heads_dim]
        # keys_shape:[n:N, k:key_len, h:heads, d:heads_dim]
        # energy_shape:[n:N, h:heads, q:query_len, k:key_len]

        if mask is not None:
            energy=energy.masked_fill(mask==0,float('-1e20)'))
        
        attention=torch.softmax(energy/(self.emded_size)**(1/2),dim=3)

        # Einsum does matrix mult. for softmax*values for each training example
        out=torch.einsum("nhqk,nvhd->nqhd",attention,values).reshape(N,query_len,self.emded_size)
        # attention_shape:[n:N, h: heads, q:query_len, k:key_len]
        # values_shape:[n:N, v:value_len, h:heads, d: heads_dim]
        # out_shape:[n:N, q:query_len, h:heads, d:heads_dim]
        # Then reshape and flattent the last 2 dimensions

        return self.fc_out(out) #shape:[n:N, q:query_len, h:heads, d:heads_dim]
        
class TransformerBlock(Module):
    def __init__(self,embed_size:int,heads:int,dropout,forward_expension:int)->None:
        super(TransformerBlock,self).__init__()
        self.attention=SelfAttention(embed_size,heads)

        self.norm1=LayerNorm(embed_size)
        self.norm2=LayerNorm(embed_size)

        self.feed_forward=Sequential(
            Linear(embed_size,forward_expension*embed_size),
            ReLU(),
            Linear(forward_expension*embed_size,embed_size)
        )
        self.dropout=Dropout(dropout)

    def forward(self, value:torch.Tensor,key:torch.Tensor,query:torch.Tensor,mask:torch.Tensor=None)->torch.Tensor:
        attention=self.attention(value,key,query,mask)
        x=self.dropout(self.norm1(attention+query))
        forward=self.feed_forward(attention)
        out=self.dropout(self.norm2(forward+x))
        return out

def main():
    pass

if __name__=="__main__":
    main()