import torch
from torch import nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, k, heads=8):
        super().init__()
        self.k, self.heads = k, heads
        self.tokeys = nn.Linear(k, k*heads, bias=False)
        self.toqueries = nn.Linear(k, k*heads, bias=False)
        self.tovalues = nn.Linear(k, k*heads, bias=False)
        self.unifyheads = nn.Linear(heads*k, k)

    def forward(self, x):
        b, t, k = x.shape
        h = self.heads
        queries = self.toqueries(x).view(b, t, h, k)
        keys = self.tokeys(x).view(b, t, h, k)
        values = self.tovalues(x).view(b, t, h, k)
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries / (k ** (1 / 4))
        keys = keys / (k ** (1 / 4))
        dot = torch.bmm(queries, keys.transpose(1, 2))
        dot = F.softmax(dot)
        out = torch.bmm(dot, values).view(b, h, t, k)
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)
        return self.unifyheads(out)

class TransformerBlock(nn.Module):
    def __init__(self, k, heads):
        super().__init__()
        self.attention = SelfAttention(k, heads=heads)
        self.norm1 = nn.LayerNorm(k)
        self.norm2 = nn.LayerNorm(k)
        self.ff = nn.Sequential(
            nn.Linear(k, 4 * k),
            nn.ReLU(),
            nn.Linear(4 * k, k)
        )

    def forward(self, x):
        attended = self.attention(x)
        x = self.norm1(attended + x)
        fed = self.ff(x)
        return self.norm2(fed + x)

class Transformer(nn.Module):
    def __init__(self, k, heads, depth, seq_length, vocab_size, num_classes):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, k)
        self.pos_emb = nn.Embedding(seq_length, k)
        tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k=k, heads=heads))
        self.tblocks = nn.Sequential(*tblocks)
        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):
        embedded = self.token_emb(x)
        b, t, k = embedded.shape
        positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)
        x = embedded + positions
        x = self.tblocks(x)
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
