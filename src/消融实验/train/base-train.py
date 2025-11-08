import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import os
import math
import editdistance  # 用于计算字符编辑距离（需安装：pip install editdistance）
from model import FullTransformer
from data import get_tinyshakespeare_dataloaders

# 超参数设置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_model = 128
n_head = 2
num_encoder_layers = 2
num_decoder_layers = 2
d_ff = 512
max_len = 128
batch_size = 32
epochs = 20
lr = 5e-4
warmup_steps = 4000
dropout = 0.1
seed = 42

# 固定随机种子
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)

# --------------------------
# 新增：评价指标计算函数
# --------------------------
def calculate_ppl(loss):
    """根据交叉熵损失计算困惑度（PPL）"""
    return math.exp(loss)

def calculate_cer(pred_ids, true_ids, pad_idx):
    """
    计算字符错误率（CER）
    pred_ids: 模型预测的字符索引序列 (batch_size, seq_len)
    true_ids: 真实字符索引序列 (batch_size, seq_len)
    pad_idx: 填充符索引（需忽略）
    """
    cer_sum = 0.0
    for pred, true in zip(pred_ids, true_ids):
        # 去除填充符
        pred_clean = [p for p in pred if p != pad_idx]
        true_clean = [t for t in true if t != pad_idx]
        # 计算编辑距离（插入+删除+替换）
        distance = editdistance.eval(pred_clean, true_clean)
        # 除以真实序列长度（避免除零）
        cer = distance / len(true_clean) if len(true_clean) > 0 else 0.0
        cer_sum += cer
    return cer_sum / len(pred_ids)  # 平均CER


def main():
    # 创建结果目录
    os.makedirs('../results-base', exist_ok=True)

    # 加载数据
    train_loader, val_loader, test_loader, src_vocab_size, tgt_vocab_size, pad_idx = get_tinyshakespeare_dataloaders(
        max_len=max_len,
        batch_size=batch_size
    )
    print(f"数据集加载完成 | 设备: {device} | 词表大小: {src_vocab_size}")

    # 初始化模型
    model = FullTransformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        n_head=n_head,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        d_ff=d_ff,
        max_len=max_len,
        dropout=dropout,
        pad_idx=pad_idx
    ).to(device)

    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    # 优化器与学习率调度
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    def lr_lambda(step):
        step = max(step, 1)
        return min(step **(-0.5), step * (warmup_steps** (-1.5)))
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # --------------------------
    # 新增：记录所有评价指标
    # --------------------------
    # 必选指标：交叉熵损失 + PPL
    train_metrics = {'loss': [], 'ppl': []}
    val_metrics = {'loss': [], 'ppl': [], 'cer': []}  # 验证集额外记录CER
    test_metrics = {'loss': 0.0, 'ppl': 0.0, 'cer': 0.0}  # 测试集最终指标

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段（仅计算损失和PPL）
        model.train()
        train_loss_total = 0.0
        for batch in train_loader:
            src_seq = batch['src_ids'].to(device)
            tgt_input = batch['tgt_input_ids'].to(device)
            tgt_label = batch['tgt_label_ids'].to(device)

            optimizer.zero_grad()
            logits = model(src_seq, tgt_input)  # (batch, tgt_len, vocab)
            loss = criterion(logits.transpose(1, 2), tgt_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item() * src_seq.size(0)

        # 计算训练集指标
        train_loss_avg = train_loss_total / len(train_loader.dataset)
        train_ppl = calculate_ppl(train_loss_avg)
        train_metrics['loss'].append(train_loss_avg)
        train_metrics['ppl'].append(train_ppl)

        # 验证阶段（计算损失、PPL和CER）
        model.eval()
        val_loss_total = 0.0
        val_cer_total = 0.0
        with torch.no_grad():
            for batch in val_loader:
                src_seq = batch['src_ids'].to(device)
                tgt_input = batch['tgt_input_ids'].to(device)
                tgt_label = batch['tgt_label_ids'].to(device)

                logits = model(src_seq, tgt_input)
                loss = criterion(logits.transpose(1, 2), tgt_label)
                val_loss_total += loss.item() * src_seq.size(0)

                # 计算CER：取预测的字符索引（argmax）
                pred_ids = logits.argmax(dim=-1).cpu().numpy()  # (batch, tgt_len)
                true_ids = tgt_label.cpu().numpy()  # 真实标签
                batch_cer = calculate_cer(pred_ids, true_ids, pad_idx)
                val_cer_total += batch_cer * src_seq.size(0)

        # 计算验证集指标
        val_loss_avg = val_loss_total / len(val_loader.dataset)
        val_ppl = calculate_ppl(val_loss_avg)
        val_cer_avg = val_cer_total / len(val_loader.dataset)
        val_metrics['loss'].append(val_loss_avg)
        val_metrics['ppl'].append(val_ppl)
        val_metrics['cer'].append(val_cer_avg)

        # 打印epoch日志
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train: Loss={train_loss_avg:.4f}, PPL={train_ppl:.2f}")
        print(f"Val:   Loss={val_loss_avg:.4f}, PPL={val_ppl:.2f}, CER={val_cer_avg:.4f}\n")

    # --------------------------
    # 新增：测试集最终评估
    # --------------------------
    model.eval()
    test_loss_total = 0.0
    test_cer_total = 0.0
    with torch.no_grad():
        for batch in test_loader:
            src_seq = batch['src_ids'].to(device)
            tgt_input = batch['tgt_input_ids'].to(device)
            tgt_label = batch['tgt_label_ids'].to(device)

            logits = model(src_seq, tgt_input)
            loss = criterion(logits.transpose(1, 2), tgt_label)
            test_loss_total += loss.item() * src_seq.size(0)

            # 计算CER
            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            true_ids = tgt_label.cpu().numpy()
            batch_cer = calculate_cer(pred_ids, true_ids, pad_idx)
            test_cer_total += batch_cer * src_seq.size(0)

    # 计算测试集指标
    test_metrics['loss'] = test_loss_total / len(test_loader.dataset)
    test_metrics['ppl'] = calculate_ppl(test_metrics['loss'])
    test_metrics['cer'] = test_cer_total / len(test_loader.dataset)
    print("="*50)
    print("测试集最终结果：")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test PPL: {test_metrics['ppl']:.2f}")
    print(f"Test CER: {test_metrics['cer']:.4f}")
    print("="*50)

    # --------------------------
    # 新增：保存所有指标曲线
    # --------------------------
    # 1. 损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics['loss'], label='Train Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('../results-base/loss_curve.png')

    # 2. PPL曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics['ppl'], label='Train PPL')
    plt.plot(val_metrics['ppl'], label='Validation PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity (PPL)')
    plt.title('Training and Validation PPL')
    plt.legend()
    plt.savefig('../results-base/ppl_curve.png')

    # 3. 验证集CER曲线
    plt.figure(figsize=(10, 5))
    plt.plot(val_metrics['cer'], label='Validation CER')
    plt.xlabel('Epoch')
    plt.ylabel('Character Error Rate (CER)')
    plt.title('Validation CER')
    plt.legend()
    plt.savefig('../results-base/cer_curve.png')

    print(f"所有指标曲线已保存至 ../results-base 目录")

if __name__ == "__main__":
    main()