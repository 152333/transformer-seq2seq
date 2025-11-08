import torch
import torch.nn as nn
import math

def positional_encoding(max_len, d_model, device):
    """正弦余弦位置编码（原始Transformer方案）"""
    pe = torch.zeros(max_len, d_model, device=device)
    position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
    pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
    pe = pe.unsqueeze(0)  # 增加batch维度 (1, max_len, d_model)
    return pe

def create_padding_mask(seq, pad_idx):
    """生成padding掩码（pad位置为1，其他为0）"""
    mask = (seq == pad_idx).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
    return mask  # True表示需要mask

def create_look_ahead_mask(seq_len, device):
    """生成前向掩码（下三角为0，上三角为1，防止Decoder看到未来token）"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    return mask  # (seq_len, seq_len)，True表示需要mask