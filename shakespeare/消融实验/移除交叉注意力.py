import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import positional_encoding, create_padding_mask, create_look_ahead_mask


# 以下基础组件与原始模型一致，仅修改DecoderLayer和最终模型
class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力核心计算（保持不变）"""

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """多头注意力机制（保持不变）"""

    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)

        output, attn_weights = ScaledDotProductAttention()(q, k, v, mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.w_o(output)
        return output, attn_weights


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
    """Encoder层（保持不变）"""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


# --------------------------
# 核心修改：移除交叉注意力的DecoderLayer
# --------------------------
class DecoderLayer_NoCrossAttn(nn.Module):
    """Decoder层（移除交叉注意力，仅保留掩码自注意力和前馈网络）"""

    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        # 仅保留掩码自注意力（移除交叉注意力）
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)

        # 调整归一化层：移除交叉注意力对应的norm2
        self.norm1 = nn.LayerNorm(d_model)  # 对应自注意力的归一化
        self.norm2 = nn.LayerNorm(d_model)  # 对应前馈网络的归一化（原norm3）

        # 调整dropout：移除交叉注意力对应的dropout2
        self.dropout1 = nn.Dropout(dropout)  # 自注意力输出的dropout
        self.dropout2 = nn.Dropout(dropout)  # 前馈网络输出的dropout

    def forward(self, x, enc_output, self_mask, cross_mask):
        # 仅保留掩码自注意力 + 残差连接 + 归一化
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))  # 与原始逻辑一致

        # 移除交叉注意力步骤，直接连接前馈网络
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))  # 原norm3调整为norm2，保证形状不变

        return x  # 输出形状仍为 (batch_size, tgt_len, d_model)


# --------------------------
# 消融实验模型：无交叉注意力的Transformer
# --------------------------
class AblationTransformer_NoCrossAttn(nn.Module):
    """完整Encoder-Decoder Transformer（移除Decoder中的交叉注意力）"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 嵌入层（与原始模型一致）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = positional_encoding(max_len, d_model,
                                                device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dropout = nn.Dropout(dropout)

        # Encoder堆叠（与原始模型一致）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_encoder_layers)
        ])

        # Decoder堆叠（替换为无交叉注意力的DecoderLayer）
        self.decoder_layers = nn.ModuleList([
            DecoderLayer_NoCrossAttn(d_model, n_head, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

        # 输出层（与原始模型一致）
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src_seq, src_mask):
        """Encoder编码（与原始模型一致）"""
        src_emb = self.src_embedding(src_seq) * math.sqrt(self.d_model)
        src_emb += self.pos_encoding[:, :src_seq.size(1), :]
        x = self.dropout(src_emb)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x  # (batch_size, src_len, d_model)

    def decode(self, tgt_seq, enc_output, tgt_mask, cross_mask):
        """Decoder解码（交叉注意力已移除，enc_output和cross_mask仅为兼容参数，实际未使用）"""
        tgt_emb = self.tgt_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb += self.pos_encoding[:, :tgt_seq.size(1), :]
        x = self.dropout(tgt_emb)
        for layer in self.decoder_layers:
            # 传入enc_output和cross_mask但不使用，保证接口兼容
            x = layer(x, enc_output, tgt_mask, cross_mask)
        return x  # (batch_size, tgt_len, d_model)（形状与原始一致）

    def forward(self, src_seq, tgt_seq):
        """前向传播（掩码逻辑与原始一致，确保输入输出形状不变）"""
        src_mask = create_padding_mask(src_seq, self.pad_idx)
        tgt_pad_mask = create_padding_mask(tgt_seq, self.pad_idx)
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1), device=src_seq.device)
        tgt_mask = tgt_pad_mask | tgt_look_ahead_mask
        cross_mask = src_mask  # 仍生成但未使用，保证兼容性

        enc_output = self.encode(src_seq, src_mask)
        dec_output = self.decode(tgt_seq, enc_output, tgt_mask, cross_mask)

        # 输出形状与原始模型完全一致：(batch_size, tgt_len, tgt_vocab_size)
        logits = self.fc(dec_output)
        return logits