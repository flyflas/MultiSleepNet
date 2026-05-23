import math
import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, dropout=0.2, max_len=30):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, query, key, value):
        q = self.norm1(query)
        k = self.norm1(key)
        v = self.norm1(value)

        out, _ = self.attn(q, k, v)
        x = query + out
        x = x + self.ffn(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.position_single = PositionalEncoding(config.dim_model, 0.1, config.pad_size + 1)
        self.position_multi = PositionalEncoding(config.dim_model * 3, 0.1, config.pad_size + 1)

        encoder_layer_1 = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.num_head,
            dim_feedforward=config.forward_hidden,
            dropout=config.dropout,
            batch_first=True
        )
        encoder_layer_2 = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.num_head,
            dim_feedforward=config.forward_hidden,
            dropout=config.dropout,
            batch_first=True
        )
        encoder_layer_3 = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.num_head,
            dim_feedforward=config.forward_hidden,
            dropout=config.dropout,
            batch_first=True
        )

        self.transformer_encoder_1 = nn.TransformerEncoder(encoder_layer_1, num_layers=config.num_encoder)
        self.transformer_encoder_2 = nn.TransformerEncoder(encoder_layer_2, num_layers=config.num_encoder)
        self.transformer_encoder_3 = nn.TransformerEncoder(encoder_layer_3, num_layers=config.num_encoder)

        self.cross_eeg1_eog = CrossAttentionBlock(config.dim_model, config.num_head, config.dropout)
        self.cross_eeg2_eog = CrossAttentionBlock(config.dim_model, config.num_head, config.dropout)
        self.cross_eog_eeg = CrossAttentionBlock(config.dim_model, config.num_head, config.dropout)

        self.drop = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(config.dim_model * 3)

        encoder_layer_multi = nn.TransformerEncoderLayer(
            d_model=config.dim_model * 3,
            nhead=config.num_head,
            dim_feedforward=config.forward_hidden,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder_multi = nn.TransformerEncoder(
            encoder_layer_multi,
            num_layers=config.num_encoder_multi
        )

        self.fc1 = nn.Sequential(
            nn.Linear(config.pad_size * config.dim_model * 3, config.fc_hidden),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(config.fc_hidden, config.num_classes)

    def forward(self, x):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x3 = x[:, 2]

        x1 = self.position_single(x1)
        x2 = self.position_single(x2)
        x3 = self.position_single(x3)

        x1 = self.transformer_encoder_1(x1)
        x2 = self.transformer_encoder_2(x2)
        x3 = self.transformer_encoder_3(x3)

        x1_eog = self.cross_eeg1_eog(x1, x3, x3)
        x2_eog = self.cross_eeg2_eog(x2, x3, x3)

        eeg_fusion = 0.5 * (x1 + x2)
        x3_eeg = self.cross_eog_eeg(x3, eeg_fusion, eeg_fusion)

        x = torch.cat([x1_eog, x2_eog, x3_eeg], dim=2)

        x = self.drop(x)
        x = self.layer_norm(x)
        residual = x

        x = self.position_multi(x)
        x = self.transformer_encoder_multi(x)
        x = self.layer_norm(x + residual)

        x = x.contiguous().view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x