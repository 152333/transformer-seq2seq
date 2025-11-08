import torch
import os
import requests
from torch.utils.data import Dataset, DataLoader

# 数据集下载地址
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
# 本地缓存路径
CACHE_DIR = "./data_cache"
LOCAL_FILE_PATH = os.path.join(CACHE_DIR, "input.txt")


class TinyShakespeareSeq2SeqDataset(Dataset):
    """Tiny Shakespeare字符级序列到序列数据集（源：前半段，目标：后半段）"""

    def __init__(self, text, max_len=128, split_ratio=0.5, vocab=None):
        self.max_len = max_len  # 源+目标总长度（字符数）
        self.split_ratio = split_ratio  # 源序列占比
        self.text = text  # 原始文本（单字符串）
        self.chars = list(self.text)  # 按字符拆分

        # 定义特殊符号
        self.bos_char = '<bos>'  # 目标序列开始符（即sos）
        self.eos_char = '<eos>'  # 目标序列结束符
        self.pad_char = '<pad>'  # 填充符号

        # 构建或复用字符词表
        if vocab is None:
            self.vocab = self.build_vocab()
        else:
            self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.char_to_idx = {c: i for i, c in enumerate(self.vocab)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}

        # 特殊符号索引
        self.bos_idx = self.char_to_idx[self.bos_char]  # 开始符索引（sos_idx）
        self.eos_idx = self.char_to_idx[self.eos_char]  # 结束符索引
        self.pad_idx = self.char_to_idx[self.pad_char]  # 填充符索引

        # 生成训练样本
        self.samples = self.generate_samples()

    def build_vocab(self):
        """构建字符级词表（包含所有出现的字符+特殊符号）"""
        unique_chars = list(set(self.chars))  # 数据集中的唯一字符
        special_chars = [self.pad_char, self.bos_char, self.eos_char]  # 特殊符号
        return special_chars + [c for c in unique_chars if c not in special_chars]  # 去重合并

    def generate_samples(self):
        """滑动窗口生成样本：每个窗口拆分为源（前半）和目标（后半）"""
        samples = []
        total_chars = len(self.chars)
        # 滑动窗口步长为max_len//2，增加样本数量
        for i in range(0, total_chars - self.max_len, self.max_len // 2):
            window_chars = self.chars[i:i + self.max_len]  # 截取窗口
            split_idx = int(len(window_chars) * self.split_ratio)  # 拆分点
            src_chars = window_chars[:split_idx]  # 源序列（前半段）
            tgt_chars = window_chars[split_idx:]  # 目标序列（后半段）

            # 转换为索引
            src_ids = [self.char_to_idx[c] for c in src_chars]
            tgt_ids = [self.char_to_idx[c] for c in tgt_chars]

            # 目标序列添加特殊符号
            tgt_input_ids = [self.bos_idx] + tgt_ids  # Decoder输入（带<bos>）
            tgt_label_ids = tgt_ids + [self.eos_idx]  # 目标标签（带<eos>）

            # 填充/截断到固定长度（源和目标各占max_len//2）
            src_len = self.max_len // 2
            tgt_len = self.max_len // 2
            src_ids = self.pad_or_truncate(src_ids, src_len)
            tgt_input_ids = self.pad_or_truncate(tgt_input_ids, tgt_len)
            tgt_label_ids = self.pad_or_truncate(tgt_label_ids, tgt_len)

            samples.append((src_ids, tgt_input_ids, tgt_label_ids))
        return samples

    def pad_or_truncate(self, ids, max_len):
        """填充或截断序列到指定长度"""
        if len(ids) > max_len:
            return ids[:max_len]
        else:
            return ids + [self.pad_idx] * (max_len - len(ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_ids, tgt_input_ids, tgt_label_ids = self.samples[idx]
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_input_ids': torch.tensor(tgt_input_ids, dtype=torch.long),
            'tgt_label_ids': torch.tensor(tgt_label_ids, dtype=torch.long)
        }


def download_and_extract():
    """下载并读取Tiny Shakespeare原始文本"""
    os.makedirs(CACHE_DIR, exist_ok=True)

    if not os.path.exists(LOCAL_FILE_PATH):
        print(f"下载数据集到 {LOCAL_FILE_PATH}...")
        try:
            response = requests.get(DATA_URL, timeout=10)
            response.encoding = 'utf-8'
            with open(LOCAL_FILE_PATH, 'w', encoding='utf-8') as f:
                f.write(response.text)
            print("下载完成")
        except Exception as e:
            print(f"下载失败：{e}")
            print(f"请手动下载文件到 {LOCAL_FILE_PATH}，地址：{DATA_URL}")
            raise

    with open(LOCAL_FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


def split_data(text):
    """拆分训练/验证/测试集（90%训练 / 5%验证 / 5%测试）"""
    i = int(len(text) * 0.9)
    train_text, remaining = text[:i], text[i:]
    i = int(len(remaining) * 0.5)
    val_text, test_text = remaining[:i], remaining[i:]
    return train_text, val_text, test_text


def get_tinyshakespeare_dataloaders(max_len=128, batch_size=32):
    """加载Tiny Shakespeare数据集，返回数据加载器、词表信息及特殊符号索引"""
    full_text = download_and_extract()
    train_text, val_text, test_text = split_data(full_text)

    # 构建数据集
    train_dataset = TinyShakespeareSeq2SeqDataset(
        text=train_text,
        max_len=max_len,
        split_ratio=0.5
    )
    vocab = train_dataset.vocab  # 复用训练集词表

    val_dataset = TinyShakespeareSeq2SeqDataset(
        text=val_text,
        max_len=max_len,
        split_ratio=0.5,
        vocab=vocab
    )
    test_dataset = TinyShakespeareSeq2SeqDataset(
        text=test_text,
        max_len=max_len,
        split_ratio=0.5,
        vocab=vocab
    )

    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 新增返回：词汇表、开始符索引、结束符索引
    return (train_loader, val_loader, test_loader,
            train_dataset.vocab_size,
            train_dataset.vocab_size,
            train_dataset.pad_idx,
            vocab,
            train_dataset.bos_idx,  # 开始符索引（sos_idx）
            train_dataset.eos_idx)