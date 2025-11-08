import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import positional_encoding, create_padding_mask, create_look_ahead_mask


# --------------------------
# 核心修改：单头注意力（替代多头注意力）
# --------------------------
class SingleHeadAttention(nn.Module):
    """单头注意力机制（移除多头拆分）"""

    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model  # 单头维度 = 模型总维度（无需拆分）

        # 线性映射层（Q、K、V共享权重，维度均为d_model）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 输出映射

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性映射（无需拆分多头，直接处理）
        q = self.w_q(q)  # (batch, seq_q, d_model)
        k = self.w_k(k)  # (batch, seq_k, d_model)
        v = self.w_v(v)  # (batch, seq_k, d_model)

        # 计算缩放点积注意力（与多头核心逻辑一致，但无多头维度）
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, seq_q, seq_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # 屏蔽无效位置
        attn_weights = F.softmax(scores, dim=-1)  # (batch, seq_q, seq_k)
        output = torch.matmul(attn_weights, v)  # (batch, seq_q, d_model)

        # 输出映射
        output = self.w_o(output)
        return output, attn_weights


# --------------------------
# 以下组件仅替换注意力为单头，其他逻辑不变
# --------------------------
class PositionWiseFeedForward(nn.Module):
    """位置-wise前馈网络（保持不变）"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    """Encoder层（使用单头注意力）"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SingleHeadAttention(d_model)  # 替换为单头注意力
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力 + 残差连接 + LayerNorm（逻辑不变）
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # 前馈网络 + 残差连接 + LayerNorm（逻辑不变）
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


class DecoderLayer(nn.Module):
    """Decoder层（使用单头注意力）"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SingleHeadAttention(d_model)  # 掩码自注意力（单头）
        self.cross_attn = SingleHeadAttention(d_model)  # 交叉注意力（单头）
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        # 掩码自注意力（逻辑不变）
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # 交叉注意力（逻辑不变）
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        # 前馈网络（逻辑不变）
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x


class AblationTransformer_SingleHead(nn.Module):
    """消融实验模型：单头注意力Transformer（替代原多头版本）"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 嵌入层（词嵌入 + 位置编码，保持不变）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = positional_encoding(max_len, d_model,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dropout = nn.Dropout(dropout)

        # Encoder堆叠（使用单头注意力的EncoderLayer）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, d_ff, dropout) for _ in range(num_encoder_layers)
        ])

        # Decoder堆叠（使用单头注意力的DecoderLayer）
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, d_ff, dropout) for _ in range(num_decoder_layers)
        ])

        # 输出层（保持不变）
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src_seq, src_mask):
        """Encoder编码源序列（逻辑不变）"""
        src_emb = self.src_embedding(src_seq) * math.sqrt(self.d_model)
        src_emb += self.pos_encoding[:, :src_seq.size(1), :]
        x = self.dropout(src_emb)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x  # (batch_size, src_len, d_model)

    def decode(self, tgt_seq, enc_output, tgt_mask, cross_mask):
        """Decoder解码目标序列（逻辑不变）"""
        tgt_emb = self.tgt_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb += self.pos_encoding[:, :tgt_seq.size(1), :]
        x = self.dropout(tgt_emb)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, cross_mask)
        return x  # (batch_size, tgt_len, d_model)

    def forward(self, src_seq, tgt_seq):
        """前向传播（逻辑不变）"""
        # 生成掩码
        src_mask = create_padding_mask(src_seq, self.pad_idx)
        tgt_pad_mask = create_padding_mask(tgt_seq, self.pad_idx)
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1), device=src_seq.device)
        tgt_mask = tgt_pad_mask | tgt_look_ahead_mask
        cross_mask = src_mask

        # 编码和解码
        enc_output = self.encode(src_seq, src_mask)
        dec_output = self.decode(tgt_seq, enc_output, tgt_mask, cross_mask)

        # 输出预测
        logits = self.fc(dec_output)  # (batch_size, tgt_len, tgt_vocab_size)
        return logits