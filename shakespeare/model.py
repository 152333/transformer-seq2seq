import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import positional_encoding, create_padding_mask, create_look_ahead_mask

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力核心计算"""
    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, n_head, seq_q, seq_k)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)  # 屏蔽位置置为负无穷
        attn_weights = F.softmax(scores, dim=-1)  # 注意力权重
        output = torch.matmul(attn_weights, v)  # (batch, n_head, seq_q, d_v)
        return output, attn_weights

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head  # 每个头的维度
        self.d_v = d_model // n_head

        # 线性映射层（Q、K、V共享权重）
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)  # 输出映射

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 线性映射并拆分多头
        q = self.w_q(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # (batch, n_head, seq_q, d_k)
        k = self.w_k(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)  # (batch, n_head, seq_k, d_k)
        v = self.w_v(v).view(batch_size, -1, self.n_head, self.d_v).transpose(1, 2)  # (batch, n_head, seq_k, d_v)

        # 计算注意力
        output, attn_weights = ScaledDotProductAttention()(q, k, v, mask)  # (batch, n_head, seq_q, d_v)

        # 拼接多头结果
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch, seq_q, d_model)
        output = self.w_o(output)
        return output, attn_weights

class PositionWiseFeedForward(nn.Module):
    """位置-wise前馈网络"""
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
    """Encoder层：自注意力 + 前馈网络"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # 自注意力 + 残差连接 + LayerNorm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # 前馈网络 + 残差连接 + LayerNorm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x

class DecoderLayer(nn.Module):
    """Decoder层：掩码自注意力 + 交叉注意力 + 前馈网络"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)  # 掩码自注意力（防未来信息）
        self.cross_attn = MultiHeadAttention(d_model, n_head)  # 与Encoder交互的注意力
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask, cross_mask):
        # 掩码自注意力
        attn_output, _ = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        # 交叉注意力（Q=Decoder输出，K=V=Encoder输出）
        attn_output, _ = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(attn_output))
        # 前馈网络
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_output))
        return x

class FullTransformer(nn.Module):
    """完整Encoder-Decoder Transformer"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_head=8,
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048,
                 max_len=512, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        # 嵌入层（词嵌入 + 位置编码）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = positional_encoding(max_len, d_model,
                                               device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.dropout = nn.Dropout(dropout)

        # Encoder堆叠
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_encoder_layers)
        ])

        # Decoder堆叠
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(num_decoder_layers)
        ])

        # 输出层（映射到目标词表）
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src_seq, src_mask):
        """Encoder编码源序列"""
        src_emb = self.src_embedding(src_seq) * math.sqrt(self.d_model)  # 词嵌入缩放
        src_emb += self.pos_encoding[:, :src_seq.size(1), :]  # 加入位置编码
        x = self.dropout(src_emb)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x  # (batch_size, src_len, d_model)

    def decode(self, tgt_seq, enc_output, tgt_mask, cross_mask):
        """Decoder解码目标序列（结合Encoder输出）"""
        tgt_emb = self.tgt_embedding(tgt_seq) * math.sqrt(self.d_model)
        tgt_emb += self.pos_encoding[:, :tgt_seq.size(1), :]
        x = self.dropout(tgt_emb)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, tgt_mask, cross_mask)
        return x  # (batch_size, tgt_len, d_model)

    def forward(self, src_seq, tgt_seq):
        """前向传播：同时处理源序列和目标序列"""
        # 生成掩码
        src_mask = create_padding_mask(src_seq, self.pad_idx)  # 源序列Padding掩码
        tgt_pad_mask = create_padding_mask(tgt_seq, self.pad_idx)  # 目标序列Padding掩码
        tgt_look_ahead_mask = create_look_ahead_mask(tgt_seq.size(1), device=src_seq.device)  # 前向掩码
        tgt_mask = tgt_pad_mask | tgt_look_ahead_mask  # 目标掩码（合并Padding和前向掩码）
        cross_mask = src_mask  # 交叉注意力使用源序列的Padding掩码

        # 编码和解码
        enc_output = self.encode(src_seq, src_mask)
        dec_output = self.decode(tgt_seq, enc_output, tgt_mask, cross_mask)

        # 输出预测
        logits = self.fc(dec_output)  # (batch_size, tgt_len, tgt_vocab_size)
        return logits