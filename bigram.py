with open('input.txt') as f:
    text = f.read()
print(text[:30])
print("length of dataset in characters: ", len(text))

vocab_size = len(set(text))
chars = sorted(set(text))
print(''.join(chars))
print("vocab_size: ", vocab_size)

# create a mapping from characters to integers
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))
encode = lambda x: [char_to_int[c] for c in x] # take a string and convert it to a list of integers
decode = lambda x: ''.join([int_to_char[c] for c in x]) # take a list of integers and convert it to a string

print("encode of Hello world: ", encode("Hello world"))
print("decode after encode: ", decode(encode("Hello world")))

import torch
data = torch.tensor(encode(text), dtype=torch.long)
print(data[:30])
print(data.shape, data.dtype)

# Let's now split up the data into train and validation sets
n = int(0.9*len(data))
train_data = data[:n] # first 90% will be train, rest val
val_data = data[n:] # last 10% will be val

block_size = 8 # maximum context length

# turn tensor data to 1d int array
# train = train.numpy()
x = train_data[:block_size]
y = train_data[1:block_size+1]

# print blocks in one batch
for t in range(block_size):
    input = x[:t+1]
    target = y[t]
    print(f"when input is {decode(input.tolist())}, target is {decode([target.tolist()])}")

batch_size = 4 # how many independent sequences will we process in parallel
block_size = 8 # maximum context length for predictions?
torch.manual_seed(42)
def get_batch(split: str = 'train'):
    data = train_data if split == 'train' else val_data
    j = torch.randint(len(data)-block_size+1, (batch_size,))
    xx = [data[k:k+block_size] for k in j]
    yy = [data[k+1:k+block_size+1] for k in j] # yy 正好是 xx 的整体右移一个窗口大小

    return torch.stack(xx, dim=0), torch.stack(yy, dim=0)

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets')
print(yb.shape)

for i in range(batch_size):
    for j in range(block_size):
        # import ipdb; ipdb.set_trace()
        print(f"when inputs is {decode(xb[i, 0:j+1].tolist())}, targets: {decode([yb[i,j].tolist()])}") 

print('----')


import torch.nn as nn
from torch.nn import functional as F
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # logits is (B,T,V) tensor of log-probabilities for each token in the vocab
        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            # cross entropy loss
            # change logits and targets shape to fit with cross_entropy
            logits = logits.view(logits.shape[0]*logits.shape[1], logits.shape[2]) # (B,T,V) -> (B*T,V)
            targets = targets.view(targets.shape[0]*targets.shape[1]) # (B,T) -> (B*T,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            logits, _ = self.forward(idx) # (B, T, V)
            # import ipdb; ipdb.set_trace()
            # focus on the last timestamp
            logits = logits[:, -1, :] # (B, T, V) -> (B, V)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)

        return idx

m = BigramLanguageModel(vocab_size=vocab_size)
logits, loss = m(xb, yb)
print(logits.shape)
print(loss)

words = m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=100)
print(f"words with 101 length:\n{decode(words[0].tolist())}") # batch 1
assert len(words[0].tolist()) == 101
# print(f"1 to char: {decode(torch.tensor([0]).tolist())}lll\nlll")

# create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(100):
    xb, yb = get_batch(split='train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if steps % 20 == 0:
        print(f"after {steps} steps, loss: {loss}")

print(f"after 100 steps, loss: {loss.item()}")

## The mathmatical tric in self-attention
# toy example illustrating how matrix multiplication can be used for a "weighted aggregation"
torch.manual_seed(42)
a = torch.tril(torch.ones((3, 3)))
# 1. how to use softmax, sum dim
a = a/a.sum(dim=1, keepdim=True) # 2. why not use softmax(a, dim=1)?
b = torch.randint(0, 3, (3,2)).float()
c = a@b

print('a=')
print(a)
print('b=')
print(b)
print('c=')
print(c)

# consider the following toy examples:
torch.manual_seed(42)
B, T, C = 4, 8, 2 # batch, time steps, channels
x = torch.randn(B, T, C)
x.shape

# We want x[b,t] = mean_{i<=t} x[b,i]
xbow = torch.randn(B, T, C)
xbow1 = torch.Tensor(B, T, C)
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # [1, t, C]
        xbow1[b, t] = xprev.mean(dim=0)

# Version 2: use matrix matmul for a weighted aggregation
wei = torch.tril(torch.ones((T, T)))
wei = wei/wei.sum(dim=1, keepdim=True)
xbow2 = wei@x # [B, T, T] @ [B, T, C] -> [B, T, C]
torch.allclose(xbow1, xbow2)

# Version 3: use softmax
wei = torch.ones((T, T)) # zeros or ones are both ok. 因为下面 softmax 算的是概率
tril = torch.tril(torch.ones(T,T))
wei = wei.masked_fill(tril==0, float('-inf')) # '-inf' means 0 in softmax
wei = wei.softmax(dim=-1)

xbow3 = wei@x # [B, T, T] @ [B, T, C] -> [B, T, C]
torch.allclose(xbow2, xbow3)

# Version 4: use self-attention
torch.manual_seed(42)
B,T,C = 4,8,32
x = torch.randn(B,T,C)
heads = 2
# Let's see a single Head perform self-attention
head_size = C//heads
key = nn.Linear(T, head_size, bias=False)
value = nn.Linear(T, head_size, bias=False)
query = nn.Linear(T, head_size, bias=False)
k = key(x) # (B, T, C) -> (B, T, head_size)
q = query(x) # (B, T, C) -> (B, T, head_size)
v = value(x) # (B, T, C) -> (B, T, head_size)
wei = q@k.transpose(-1, -2) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
wei = wei.mask_fill(torch.tril(torch.ones(T, T)) == 0, float('-inf'))
wei = wei.softmax(dim=-1)

out = wei@v # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)



