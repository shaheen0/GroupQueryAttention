import torch
import torch.nn as nn
import torch.nn.functional as F
class GroupQueryAttention(nn.Module):
    def __init__(self , K_V_heads , num_Q_heads , d_model):
        super().__init__()
        self.K_V_heads = K_V_heads       # 8
        self.num_Q_heads = num_Q_heads   # 32
        self.d_model = d_model           # 4096
        self.head_dim = d_model // num_Q_heads  # 4906// 32 = 128
        self.kv_dim = self.K_V_heads * self.head_dim   # 8 x 128 = 1024
        assert d_model % num_Q_heads == 0, "d_model must be divisible by num_Q_heads"
        self.WQ = nn.Linear(d_model , d_model) # 4096
        self.WK = nn.Linear(d_model , self.kv_dim)
        self.WV = nn.Linear(d_model , self.kv_dim)

        #now dot linear trasnformation X.WQ
    def forward(self , x):
        Batch, seq_lenght, dim = x.shape
        Q = self.WQ(x)
        K = self.WK(x)
        V = self.WV(x)
        # reshape
        Q = Q.view(Batch , seq_lenght , self.num_Q_heads , self.head_dim).transpose(-2,-3)   # [1,32,10,128]
        K = K.view(Batch , seq_lenght , self.K_V_heads , self.head_dim).transpose(-2,-3)     # [1,8,10,128]  k are 8 and each k have 4 queries
        V = V.view(Batch , seq_lenght , self.K_V_heads , self.head_dim).transpose(-2,-3)     # [1,8,10,128]

        # GQA MAGIC: Repeat K and V so they match the 32 Q heads
        # Every 4 Q heads will share 1 K head and 1 V head.
        Q_Group = self.num_Q_heads // self.num_K_V_heads
        K = K.repeat_interleave(Q_Group , dim =1)
        V = V.repeat_interleave(Q_Group , dim = 1)

        # attention
        attention_score = Q @ K.transpose(-2,-1)
        # scale
        scale = self.head_dim ** 0.5
        attention_score = attention_score / scale
        scaling = F.softmax(attention_score , dim= -1)
        attention_context = scaling @ V     #[B, 32, S, 128]
        # concatination
        context_attention = attention_context.transpose(-2,-3).contiguous().view(Batch , seq_lenght , self.d_model)
        return context_attention


