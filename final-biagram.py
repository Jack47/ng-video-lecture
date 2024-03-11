import torch

# hyperparameters
batch_size = 16
block_size = 32
eval_iters = 200

with open('final-biagram.txt', 'r') as f:
    text = f.read()
vocabs = sorted(set(text))

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

class Head(nn.Module):
    def __init__(self, head_size):
        pass

    def forward(self, x):
        pass
    
import torch.nn as nn
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        pass

    def forward(self, x):
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        pass
    
    def forward(self, x):
        pass

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        pass
    
    def forward(self, x):
        pass

class BigramLanguageModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, idx, targets=None):
        return logits, loss

    def generate(self, idx, max_new_tokens):
        pass
     