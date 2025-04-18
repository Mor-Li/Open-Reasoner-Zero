import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import numpy as np
from collections import Counter
import re

# GPU 内存监控函数
def print_gpu_memory_stats():
    """打印 GPU 内存使用情况"""
    if torch.cuda.is_available():
        print("\n----- GPU 内存使用情况 -----")
        # 总内存
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        # 当前分配的内存
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        # 当前缓存的内存
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        # 空闲的缓存内存
        free_memory = reserved_memory - allocated_memory
        
        print(f"总 GPU 内存: {total_memory:.2f} GB")
        print(f"已分配内存: {allocated_memory:.2f} GB")
        print(f"缓存内存: {reserved_memory:.2f} GB")
        print(f"空闲缓存内存: {free_memory:.2f} GB")
        print(f"利用率: {(allocated_memory / total_memory) * 100:.2f}%")
        print("------------------------\n")
    else:
        print("无可用 GPU")

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 简单的英法翻译样本 
train_data = [
    ("Hello, how are you?", "Bonjour, comment ça va?"),
    ("I love programming.", "J'aime programmer."),
    ("Where is the library?", "Où est la bibliothèque?"),
    ("My name is John.", "Je m'appelle John."),
    ("What time is it?", "Quelle heure est-il?"),
    ("I am a student.", "Je suis étudiant."),
    ("The weather is nice today.", "Le temps est beau aujourd'hui."),
    ("Do you speak English?", "Parlez-vous anglais?"),
    ("I am hungry.", "J'ai faim."),
    ("Thank you very much.", "Merci beaucoup.")
]

# 简单的分词函数 - 不依赖 torchtext
def tokenize_en(text):
    """简单的英文分词"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # 替换标点符号为空格
    return text.split()

def tokenize_fr(text):
    """简单的法文分词"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)  # 替换标点符号为空格
    return text.split()

# 构建词汇表
class Vocabulary:
    def __init__(self, specials=['<unk>', '<pad>', '<bos>', '<eos>']):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        self.specials = specials
        
        # 添加特殊标记
        for i, token in enumerate(specials):
            self.word2idx[token] = i
            self.idx2word[i] = token
    
    def add_sentence(self, sentence):
        """向词汇表添加一个句子的所有单词"""
        for word in sentence:
            self.word_count[word] += 1
    
    def build(self, min_freq=1):
        """根据词频构建词汇表"""
        # 按频率排序的单词列表 (不包括特殊标记)
        words = [w for w, c in self.word_count.items() if c >= min_freq]
        
        # 更新索引
        for i, word in enumerate(words):
            idx = i + len(self.specials)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def __len__(self):
        return len(self.word2idx)
    
    def __getitem__(self, word):
        """获取单词的索引，如果不存在则返回 <unk> 的索引"""
        return self.word2idx.get(word, self.word2idx['<unk>'])

# 创建词汇表
en_vocab = Vocabulary()
fr_vocab = Vocabulary()

# 处理所有数据并构建词汇表
for src, tgt in train_data:
    en_tokens = tokenize_en(src)
    fr_tokens = tokenize_fr(tgt)
    en_vocab.add_sentence(en_tokens)
    fr_vocab.add_sentence(fr_tokens)

# 构建词汇表
en_vocab.build()
fr_vocab.build()

# 特殊标记索引
PAD_IDX = en_vocab['<pad>']
BOS_IDX = en_vocab['<bos>']
EOS_IDX = en_vocab['<eos>']

# 自定义数据集类
class TranslationDataset(Dataset):
    def __init__(self, data, src_tokenize, tgt_tokenize, src_vocab, tgt_vocab):
        self.data = data
        self.src_tokenize = src_tokenize
        self.tgt_tokenize = tgt_tokenize
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        src_tokens = self.src_tokenize(src)
        tgt_tokens = self.tgt_tokenize(tgt)
        
        src_indices = [BOS_IDX] + [self.src_vocab[token] for token in src_tokens] + [EOS_IDX]
        tgt_indices = [BOS_IDX] + [self.tgt_vocab[token] for token in tgt_tokens] + [EOS_IDX]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

# 数据加载器的整理函数
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    
    # 找到当前批次中最长序列长度
    src_max_len = max([len(s) for s in src_batch])
    tgt_max_len = max([len(t) for t in tgt_batch])
    
    # 手动填充
    padded_src_batch = []
    padded_tgt_batch = []
    
    for src in src_batch:
        padded = torch.cat([src, torch.full((src_max_len - len(src),), PAD_IDX, dtype=torch.long)])
        padded_src_batch.append(padded)
    
    for tgt in tgt_batch:
        padded = torch.cat([tgt, torch.full((tgt_max_len - len(tgt),), PAD_IDX, dtype=torch.long)])
        padded_tgt_batch.append(padded)
    
    return torch.stack(padded_src_batch), torch.stack(padded_tgt_batch)

# 创建数据集和数据加载器
train_dataset = TranslationDataset(train_data, tokenize_en, tokenize_fr, en_vocab, fr_vocab)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Transformer 模型参数
SRC_VOCAB_SIZE = len(en_vocab)
TGT_VOCAB_SIZE = len(fr_vocab)
EMB_SIZE = 32  # 小型化以加速训练
NHEAD = 4
FFN_HID_DIM = 64
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2
DROPOUT = 0.1

print(f"源语言词汇表大小: {SRC_VOCAB_SIZE}")
print(f"目标语言词汇表大小: {TGT_VOCAB_SIZE}")

# 位置编码类
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout, maxlen=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(maxlen).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2) * (-math.log(10000.0) / emb_size))
        pe = torch.zeros(maxlen, 1, emb_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# 构建Transformer模型
class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, emb_size, nhead, 
                 dim_feedforward, num_encoder_layers, num_decoder_layers, dropout_prob):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout_prob,
            batch_first=True
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout_prob)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
                src_padding_mask=None, tgt_padding_mask=None, memory_key_padding_mask=None):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                               src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    
    def encode(self, src, src_mask=None):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)
    
    def decode(self, tgt, memory, tgt_mask=None):
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

# 创建Transformer模型实例
print("\n初始化模型...")
transformer = Seq2SeqTransformer(
    SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, EMB_SIZE, NHEAD,
    FFN_HID_DIM, NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, DROPOUT
).to(device)

# 打印模型内存使用情况
print("\n模型参数量:", sum(p.numel() for p in transformer.parameters()))
print_gpu_memory_stats()

# 构建掩码函数
def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)
    
    src_padding_mask = (src == PAD_IDX).to(device)
    tgt_padding_mask = (tgt == PAD_IDX).to(device)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# 损失函数
loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

# ------------------- 优化器设置（重点部分） -------------------

print("\n初始化优化器...")

# 基本的SGD优化器 (随机梯度下降)
sgd_optimizer = optim.SGD(
    transformer.parameters(),
    lr=0.01,                # 学习率：控制每次参数更新的步长
    momentum=0.9,           # 动量：帮助加速训练并跳出局部最小值
    weight_decay=1e-4,      # 权重衰减：L2正则化，防止过拟合
    dampening=0,            # 阻尼：用于减少动量
    nesterov=True           # 使用Nesterov动量：提供更好的收敛性
)

# Adam优化器 (自适应矩估计)
adam_optimizer = optim.Adam(
    transformer.parameters(),
    lr=0.0001,               # 学习率：通常Adam需要较小的学习率
    betas=(0.9, 0.999),      # β1和β2：一阶和二阶矩估计的指数衰减率
    eps=1e-8,                # ε：添加到分母以提高数值稳定性
    weight_decay=1e-4,       # 权重衰减：L2正则化
    amsgrad=False            # AMSGrad变体：保持学习率更稳定
)

# AdamW优化器 (权重衰减修正的Adam)
adamw_optimizer = optim.AdamW(
    transformer.parameters(),
    lr=0.0001,               # 学习率
    betas=(0.9, 0.999),      # β1和β2
    eps=1e-8,                # ε
    weight_decay=0.01,       # 权重衰减：AdamW中更正确地实现了权重衰减
    amsgrad=False            # AMSGrad变体
)

# RMSprop优化器
rmsprop_optimizer = optim.RMSprop(
    transformer.parameters(),
    lr=0.001,                # 学习率
    alpha=0.99,              # α：平滑常数，控制梯度平方的移动平均
    eps=1e-8,                # ε：数值稳定性常数
    weight_decay=1e-4,       # 权重衰减
    momentum=0,              # 动量
    centered=False           # 是否使用中心化版本
)

# 选择要使用的优化器
optimizer = adamw_optimizer  # 大多数Transformer训练中的常用选择

# 打印优化器内存使用情况
print("\n优化器初始化后的内存使用情况:")
print_gpu_memory_stats()

# 学习率调度器 - 使用带预热的余弦退火
def get_lr_scheduler(optimizer):
    # 预热步数和总步数
    warmup_steps = 100
    total_steps = 2000
    
    def lr_lambda(step):
        # 预热阶段线性增加学习率
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # 预热后使用余弦退火
        return 0.5 * (1.0 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scheduler = get_lr_scheduler(optimizer)

# ------------------- 训练循环 -------------------

def train_epoch(model, optimizer, scheduler, dataloader, loss_fn, epoch):
    model.train()
    losses = 0
    
    for idx, (src, tgt) in enumerate(dataloader):
        # 批次开始前的内存
        if idx == 0:
            print(f"\n批次 {idx} 开始前的内存使用情况:")
            print_gpu_memory_stats()
            
        src = src.to(device)
        tgt = tgt.to(device)
        
        # 将批次数据移到 GPU 后的内存
        if idx == 0:
            print(f"\n批次 {idx} 数据移到 GPU 后的内存使用情况:")
            print_gpu_memory_stats()
        
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)
        
        logits = model(
            src, tgt_input, src_mask, tgt_mask,
            src_padding_mask, tgt_padding_mask, src_padding_mask
        )
        
        # 前向传播后的内存
        if idx == 0:
            print(f"\n批次 {idx} 前向传播后的内存使用情况:")
            print_gpu_memory_stats()
        
        optimizer.zero_grad()
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_output.reshape(-1))
        loss.backward()
        
        # 反向传播后的内存
        if idx == 0:
            print(f"\n批次 {idx} 反向传播后的内存使用情况:")
            print_gpu_memory_stats()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        losses += loss.item()
        
        if idx % 2 == 0:
            print(f"Epoch: {epoch}, Batch: {idx}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return losses / len(dataloader)

# 训练函数
def train(model, optimizer, scheduler, train_dataloader, loss_fn, epochs):
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_loss = train_epoch(model, optimizer, scheduler, train_dataloader, loss_fn, epoch)
        end_time = time.time()
        
        print(f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Epoch time: {(end_time - start_time):.2f}s")
        
        # 每个 epoch 结束后打印内存使用情况
        print(f"\nEpoch {epoch} 结束后的内存使用情况:")
        print_gpu_memory_stats()

# 开始训练
print(f"\n开始训练...")
train(transformer, optimizer, scheduler, train_dataloader, loss_fn, epochs=10)

# 训练结束后打印最终内存使用情况
print("\n训练结束，最终内存使用情况:")
print_gpu_memory_stats()

# 如果您想手动清理内存
print("\n清理内存...")
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print_gpu_memory_stats() 