import torch
import torch.nn.Functional as F

# hyperparameters
batch_size = 16
block_size = 32
eval_iters = 200
device = 'gpu'

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

with open('final-biagram.txt', 'r') as f:
    text = f.read()
vocabs = sorted(set(text))
vocab_size = len(sorted(set(text)))

ctoi = {c:i for i, c in enumerate(vocabs)}
itoc = {i: c for i, c in enumerate(vocabs)}

encode = lambda x: [ctoi[c] for c in x]
decode = lambda x: ''.join([itoc[i] for i in x])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(text))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    offset = torch.randint(len(data) - block_size, size=(batch_size,))
    xx = [], yy = []
    for i in range(batch_size):
        x = data[offset:offset+block_size]
        xx = xx.append(xx, x)
        y = data[offset+1:offset+block_size+1]
        yy = yy.append(yy, y)

    x = torch.stack(xx)
    y = torch.stack(yy)
    return x, y

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

# single head of self-attention
class Head(nn.Module):
    def __init__(self, head_size):
        super.__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.head_size = head_size
        self.register_buffer('tril', torch.tril(torch.ones((head_size, head_size) == 0, float('-inf'))))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        B,T,C = x.shape
        # compute attention scores ("affinities")
        q = self.query(x)
        k = self.key(x) # (batch_size, block_size, head_size)
        x = q@k.transpose(-1, -2)*self.head_size**-0.5
        x = x.masked_fill(self.tril == 0, float('-inf'))
        x = x.softmax(dim=-1)
        wei = self.dropout(x)
        # compute weighted aggregation of the values
        v = self.value(x) # (batch_size, block_size, head_size)
        out = wei@v

        return out

import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super.__init__()
    # 不同的 head 并不是把输入直接切分为几块，而是把输入进行类似 embedding 一样的操作
        assert num_heads * head_size == n_embd
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # multi-head attention 之后，再经过一个线性层
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1) # why torch.cat instead of torch.stack?
        return self.dropout(self.proj(x))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super.__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd), nn.Dropout(p=dropout))
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        self.multi_heads = MultiHeadAttention(n_head, n_embd/n_head)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd)
    
    def forward(self, x):
        x = x + self.multi_heads(self.ln1(x)) # 为什么 ln 在输入上
        x = x + self.mlp(self.ln2(x))

# super simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super.__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T, 1) tensors of integers
        B,T,C = idx.shape
        # 为什么 positional embedding 的输入跟 x 无关？
        x = self.token_embedding(idx) + self.positional_embedding(torch.arange(T, device=device)) # (B,T,n_embd) + (B, T, n_embd)
        x = self.blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets != None: # (B, T, 1)
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss

    # 为什么上面训练的时候，forward 里可以一次处理一个 batch，但是下面生成的时候，只能处理一个 token 一次？
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx) # (B, T+1, 1); 训练时拿到 logits 就可以了，推理的时候才需要拿 logits 去生成最符合概率的那个
            logits = logits[:, -1, :] # (B, 1, V)
            probs = torch.softmax(logits, dim=-1)# (B,T,V) ->
            x = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, x], dim = -2)
        return idx
