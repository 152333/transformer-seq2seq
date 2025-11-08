import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import matplotlib.pyplot as plt
import os
import math
import editdistance
from model import FullTransformer
from data import get_tinyshakespeare_dataloaders

# 超参数设置（保持不变）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
d_model = 128
n_head = 8
num_encoder_layers = 2
num_decoder_layers = 2
d_ff = 512
max_len = 128
batch_size = 32
epochs = 50
lr = 5e-4
warmup_steps = 4000
dropout = 0.1
seed = 42

# 固定随机种子（保持不变）
torch.manual_seed(seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(seed)


# --------------------------
# 新增：文本生成与解码工具
# --------------------------
def ids_to_text(ids, vocab, pad_idx, eos_idx=None):
    """将token索引序列转换为文本字符串"""
    text = []
    for idx in ids:
        if idx == pad_idx:  # 忽略填充符
            break
        if eos_idx is not None and idx == eos_idx:  # 遇到结束符停止
            break
        text.append(vocab[idx])  # 直接用列表索引获取字符（vocab本身就是索引到字符的映射）
    return ''.join(text)


def generate_text(model, src_seq, vocab, max_gen_len, pad_idx, sos_idx, eos_idx, device):
    """
    用贪婪解码生成目标文本
    src_seq: 输入序列 (1, src_len) （batch_size=1，方便单样本生成）
    max_gen_len: 最大生成长度
    sos_idx: 开始符索引
    eos_idx: 结束符索引
    """
    model.eval()
    with torch.no_grad():
        # 初始化生成序列：[SOS]
        generated = torch.tensor([[sos_idx]], dtype=torch.long, device=device)

        for _ in range(max_gen_len):
            # 模型预测：输入src_seq和当前生成的序列
            logits = model(src_seq, generated)  # (1, gen_len, vocab_size)
            # 取最后一个token的预测结果（贪婪选择）
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            # 拼接生成序列
            generated = torch.cat([generated, next_token], dim=1)
            # 若生成结束符，提前停止
            if next_token.item() == eos_idx:
                break

        return generated.squeeze(0).cpu().numpy()  # 转为numpy数组（去掉batch维度）


# 评价指标计算函数（保持不变）
def calculate_ppl(loss):
    return math.exp(loss)


def calculate_cer(pred_ids, true_ids, pad_idx):
    cer_sum = 0.0
    for pred, true in zip(pred_ids, true_ids):
        pred_clean = [p for p in pred if p != pad_idx]
        true_clean = [t for t in true if t != pad_idx]
        distance = editdistance.eval(pred_clean, true_clean)
        cer = distance / len(true_clean) if len(true_clean) > 0 else 0.0
        cer_sum += cer
    return cer_sum / len(pred_ids)


def main():
    os.makedirs('../results-调参-dropout', exist_ok=True)

    # --------------------------
    # 加载数据（新增vocab和特殊符号索引）
    # --------------------------
    # 假设数据加载器返回：词汇表、开始符/结束符索引（需根据实际data.py调整）
    train_loader, val_loader, test_loader, src_vocab_size, tgt_vocab_size, pad_idx, vocab, sos_idx, eos_idx = get_tinyshakespeare_dataloaders(
        max_len=max_len,
        batch_size=batch_size
    )
    print(f"数据集加载完成 | 设备: {device} | 词表大小: {src_vocab_size}")

    # 初始化模型（保持不变）
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

    # 损失函数、优化器、调度器（保持不变）
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)

    def lr_lambda(step):
        step = max(step, 1)
        return min(step ** (-0.5), step * (warmup_steps ** (-1.5)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # 记录指标（保持不变）
    train_metrics = {'loss': [], 'ppl': []}
    val_metrics = {'loss': [], 'ppl': [], 'cer': []}
    test_metrics = {'loss': 0.0, 'ppl': 0.0, 'cer': 0.0}

    # 训练循环（保持不变）
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        for batch in train_loader:
            src_seq = batch['src_ids'].to(device)
            tgt_input = batch['tgt_input_ids'].to(device)
            tgt_label = batch['tgt_label_ids'].to(device)

            optimizer.zero_grad()
            logits = model(src_seq, tgt_input)
            loss = criterion(logits.transpose(1, 2), tgt_label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss_total += loss.item() * src_seq.size(0)

        train_loss_avg = train_loss_total / len(train_loader.dataset)
        train_ppl = calculate_ppl(train_loss_avg)
        train_metrics['loss'].append(train_loss_avg)
        train_metrics['ppl'].append(train_ppl)

        # 验证阶段（保持不变）
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

                pred_ids = logits.argmax(dim=-1).cpu().numpy()
                true_ids = tgt_label.cpu().numpy()
                batch_cer = calculate_cer(pred_ids, true_ids, pad_idx)
                val_cer_total += batch_cer * src_seq.size(0)

        val_loss_avg = val_loss_total / len(val_loader.dataset)
        val_ppl = calculate_ppl(val_loss_avg)
        val_cer_avg = val_cer_total / len(val_loader.dataset)
        val_metrics['loss'].append(val_loss_avg)
        val_metrics['ppl'].append(val_ppl)
        val_metrics['cer'].append(val_cer_avg)

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train: Loss={train_loss_avg:.4f}, PPL={train_ppl:.2f}")
        print(f"Val:   Loss={val_loss_avg:.4f}, PPL={val_ppl:.2f}, CER={val_cer_avg:.4f}\n")

    # 测试集评估（保持不变）
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

            pred_ids = logits.argmax(dim=-1).cpu().numpy()
            true_ids = tgt_label.cpu().numpy()
            batch_cer = calculate_cer(pred_ids, true_ids, pad_idx)
            test_cer_total += batch_cer * src_seq.size(0)

    test_metrics['loss'] = test_loss_total / len(test_loader.dataset)
    test_metrics['ppl'] = calculate_ppl(test_metrics['loss'])
    test_metrics['cer'] = test_cer_total / len(test_loader.dataset)
    print("=" * 50)
    print("测试集最终结果：")
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"Test PPL: {test_metrics['ppl']:.2f}")
    print(f"Test CER: {test_metrics['cer']:.4f}")
    print("=" * 50)

    # --------------------------
    # 新增：样本预测与文本生成展示
    # --------------------------
    print("\n" + "=" * 50)
    print("样本预测结果展示：")
    print("=" * 50)

    # 从测试集中选取3个样本进行展示
    sample_indices = [0, 5, 10]  # 选取第0、5、10个batch中的第一个样本
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx not in sample_indices:
                continue  # 只处理选中的样本

            # 提取单个样本（batch中的第一个）
            src_seq = batch['src_ids'][0:1].to(device)  # (1, src_len)
            tgt_true_ids = batch['tgt_label_ids'][0].cpu().numpy()  # 真实目标序列

            # 1. 用模型生成预测（贪婪解码）
            pred_ids = generate_text(
                model=model,
                src_seq=src_seq,
                vocab=vocab,
                max_gen_len=max_len,
                pad_idx=pad_idx,
                sos_idx=sos_idx,
                eos_idx=eos_idx,
                device=device
            )

            # 2. 转换为文本
            src_text = ids_to_text(src_seq.squeeze(0).cpu().numpy(), vocab, pad_idx)
            true_text = ids_to_text(tgt_true_ids, vocab, pad_idx, eos_idx)
            pred_text = ids_to_text(pred_ids, vocab, pad_idx, eos_idx)

            # 3. 打印结果
            print(f"\n样本 {idx + 1}:")
            print(f"输入文本:  {src_text}")
            print(f"真实文本:  {true_text}")
            print(f"预测文本:  {pred_text}")
            print("-" * 50)

    # 保存指标曲线（保持不变）
    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics['loss'], label='Train Loss')
    plt.plot(val_metrics['loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('../results-调参-dropout/loss_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(train_metrics['ppl'], label='Train PPL')
    plt.plot(val_metrics['ppl'], label='Validation PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity (PPL)')
    plt.title('Training and Validation PPL')
    plt.legend()
    plt.savefig('../results-调参-dropout/ppl_curve.png')

    plt.figure(figsize=(10, 5))
    plt.plot(val_metrics['cer'], label='Validation CER')
    plt.xlabel('Epoch')
    plt.ylabel('Character Error Rate (CER)')
    plt.title('Validation CER')
    plt.legend()
    plt.savefig('../results-调参-dropout/cer_curve.png')

    print(f"\n所有指标曲线已保存至 ../results-调参-dropout 目录")


if __name__ == "__main__":
    main()