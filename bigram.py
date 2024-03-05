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
# import ipdb; ipdb.set_trace()
train_data = data[:n] # first 90% will be train, rest val
val_data = data[n:] # last 10% will be val

block_size = 8 # maximum context length

# turn tensor data to 1d int array
# train = train.numpy()
x = train_data[:block_size]
y = train_data[1:block_size+1]

for t in range(block_size):
    input = x[:t+1]
    target = y[t]
    print(f"when input is {decode(input.tolist())}, target is {decode([target.tolist()])}")

batch_size = 4 # how many independent sequences will we process in parallel
block_size = 8 # maximum context length for predictions?

def get_batch(split: str = 'train'):
    torch.stack()

xb, yb = get_batch('train')
print('inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)

print('----')