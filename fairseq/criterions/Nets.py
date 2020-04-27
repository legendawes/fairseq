
import torch
import torch.nn as nn


class DenseNet(nn.Module):
    def __init__(self, emb_size, pretrained_emb, hidden_size, out_size, dropout):
        super(DenseNet, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_emb, freeze=False)
        self.hidden = nn.Linear(emb_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.mean(x, dim=1)
        x = self.hidden(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x
