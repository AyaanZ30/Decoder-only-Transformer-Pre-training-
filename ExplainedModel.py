import torch
import torch.nn as nn 
from torch.nn import functional as F 

# INITIALIZING HYPERPARAMETERS
batch_size = 64
context_size = 256
epochs = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6 
dropout = 0.2
#====================

torch.manual_seed(1537)

with open('input.txt', 'r', encoding = 'utf-8') as f:
  text = f.read()

# all unique chars present in input.txt
unique_elements = sorted(list(set(text)))
vocab_size = len(unique_elements)

# char ---> int (encoding)   +    int ---> char (decoding)
char_to_index = {c : i for i, c in enumerate(unique_elements)}
index_to_char = {i : c for i, c in enumerate(unique_elements)}

encode = lambda text : [char_to_index[c] for c in text]
decode = lambda li : ''.join([index_to_char[i] for i in li])

data = torch.tensor(encode(text), dtype = torch.long)
n = int(0.9 * len(data))

train_data = data[:n]               # 90% of data used for training
val_data = data[n:]                 # 10% of data used for validation

def get_batches(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint((len(data) - context_size), (batch_size,))
  x = torch.stack([data[i : i + context_size] for i in ix])
  y = torch.stack([data[i + 1 : i + context_size + 1] for i in ix])
  return x, y

# MULTI-HEADED SELF ATTENTION FOR CAPTURING RELEVANT CONTEXT
class Head(nn.Module):              
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(p = dropout)     # to prevent network from overfitting
    
    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x) 
        q = self.query(x)
        # compute attention scores('affinities') (attention formula -> Softmax(QK^T / sqrt(dim))v)
        wei = q @ k.transpose(-2, -1) * C**(-0.5)   # (B, T, C) @ (B, C, T) --> (B, T, T) 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))   # (B, T, T)
        wei = F.softmax(wei, dim = -1)     # (B, T, T)
        wei = self.dropout(wei)            # to avoid random communications
        v = self.value(x)                  # (B, T, C)
        out = wei @ v                      # weighted aggregation of values -> (B, T, C)
        return out 

class MultiHeadAttention(nn.Module):   # multiple self-attention heads running in parallel
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(p = dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)  # along the channels dim
        out = self.dropout(self.proj(out))            # projection is linear transf of the outcome of prev layer
        return out 

#==================================================================

# FORWARD PROPAGATION : (linear layer followed by a non-linearity)
class FeedForward(nn.Module):    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # 32 --> 128
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),        # 128 --> 32 
            nn.Dropout(p = dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
# TRANSFORMER DECODER BLOCK (COMMUNICATION <--> COMPUTATION)
class Block(nn.Module):      
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(head_size, n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)       # normalizing batches (Gaussian dist)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))      # x(batch) --> Normalized(x) --> self-attention implementation 
        x = x + self.ffwd(self.ln2(x))    # x(batch) --> Normalized(x) --> feed-forward implementation
        return x 
#==================================================================

# BUILDING THE MODEL (INTEGRATING SUB-PARTS INTO A MODEL)
class SuperMiniGPT(nn.Module):
  def __init__(self):
    super().__init__()
    self.vocab_size = vocab_size
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
    self.positional_embedding_table = nn.Embedding(context_size, n_embd)    # create embeddings for all chars in the current context
    self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd)               # final normlization layer
    self.lm_head = nn.Linear(n_embd, vocab_size)    # linear layer : token embed ---> logits
  
  def forward(self, idx, targets = None):    # idx, targets are (Batch(B), Time/Context(T)) tensors of integers
    B, T = idx.shape
    token_emb = self.token_embedding_table(idx)   # (B, T, C) ==> (4, 8, 65)
    pos_emb = self.positional_embedding_table(torch.arange(T))  # (T, C)
    x = token_emb + pos_emb               # Very important ==> (B, T, C)
    x = self.blocks(x)                      # (B, T, C) forward prop (refer class above) after implementing self-attention
    logits = self.lm_head(x)   # (B, T, vocab_size)
    
    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)             # input format expected by cross_entropy function 
      targets = targets.view(B*T)  
      loss = F.cross_entropy(logits, targets)
    return logits, loss

#==================================================================
  
# GENERATING NEW TEXT (CHARACTER-BY-CHARACTER)
  def generate(self, idx, max_new_tokens):    # idx => (B, T) matrix of indices in current context
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]    # crop idx to the last context_size token (8th)
        logits, loss = self(idx_cond)    # get the predictions (B, T, C)
        logits = logits[:, -1, :]   # focus only on last time step & obtain form (B, C) 
        probs = F.softmax(input = logits, dim = -1)   # get probabilities of generation from logits
        idx_next = torch.multinomial(input = probs, num_samples = 1)   # at the end, we need one prob per batch, of the most probable next character in seq(batch chunk)
        idx = torch.cat((idx, idx_next), dim = 1)     # (B, T+1)
    return idx

#==================================================================

model = SuperMiniGPT()

# LOSS ESTIMATION FOR TRAINING AND VALIDATION SPLITS
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batches(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

#==================================================================


optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate)

# Training
for iter in range(epochs):
    if (iter % eval_interval) == 0:
        losses = estimate_loss()
        print(f'Step {iter} : Train loss {losses['train']:.4f}, Val loss {losses['val']:.4f}')
    
    xb, yb = get_batches('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()
    
#==================================================================


# Generate from the model
context = torch.zeros((1, 1), dtype = torch.long)
print(decode(model.generate(context, max_new_tokens = 500)[0].tolist()))
