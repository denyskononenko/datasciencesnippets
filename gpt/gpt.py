import torch
import torch.nn as nn
from torch.nn import functional as F

# B, T, C = 4, 8, 32 # batch, time (context depth), channels

# hyperparameters
batch_size = 32
block_size = 8 # the maximum context lenght for predictions
max_iters = 1000
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
eval_interval = 500
embed_size = 32
n_heads = 2
dropout = 0.2

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed_table = nn.Embedding(vocab_size, embed_size)
        self.position_embed_table = nn.Embedding(block_size, embed_size)
        self.blocks = nn.Sequential(
            Block(embed_size, n_heads=4),
            Block(embed_size, n_heads=4),
            Block(embed_size, n_heads=4),
            Block(embed_size, n_heads=4),
            nn.LayerNorm(embed_size),
        )
        self.lm_head = nn.Linear(embed_size, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are (B,  T)
        token_emb = self.token_embed_table(idx) # (B, T, C)
        pos_emb = self.position_embed_table(torch.arange(T, device=device)) # (T, C)
        x = token_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # apply transformer block (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # generate predictions
            logits, _ = self(idx_cond)
            # select last last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class Block(nn.Module):
    """Transformer block: commnication (multi-head self-attention) and computation (Feed Forward Network)"""
    def __init__(self, embed_size, n_heads):
        super().__init__()
        head_size = embed_size // n_heads
        self.sa = MultiHead(n_heads, head_size)
        self.ffn = FFN(embed_size)
        self.pre_ln = nn.LayerNorm(embed_size) # pre transformation layer norm
        self.post_ln = nn.LayerNorm(embed_size) # post transformation layer norm

    def forward(self, x):
        x = x + self.sa(self.pre_ln(x))
        x = x + self.ffn(self.post_ln(x))
        return x
    
class Head(nn.Module):
    """A head of scale dot-product self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # complete attention scores (affinities)
        aff = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        aff = aff.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T) make feature not interact with the past
        aff = F.softmax(aff, dim=-1)
        aff = self.dropout(aff)
        # compute weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = aff @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHead(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FFN(nn.Module):
    "Feed-forward neural network"
    def __init__(self, embed_size):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.ReLU(),
            nn.Linear(4 * embed_size, embed_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.ffn(x)

@torch.no_grad()
def estimate_loss(model, data):
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data[split])
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_batch(data):
    """
    Args:
        data: (torch.tensor) dataset
        batch_size: (int) size of the batch
        block_size: (int) lenght of the context
    """
    indx = torch.randint(len(data) - block_size, (batch_size, ))
    x_batch = torch.stack([data[i : i + block_size] for i in indx])
    y_batch = torch.stack([data[i + 1 : i + block_size + 1] for i in indx])
    return x_batch, y_batch

if __name__ == "__main__":
    # read the text file
    with open('data/hp/01 Harry Potter and the Sorcerers Stone.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    vocab = sorted(list(set(text)))
    vocab_size = len(vocab)
    # encoding of data
    char2token = {c : i for i, c in enumerate(vocab)}
    token2char = {i : c for i, c in enumerate(vocab)}
    encode = lambda s: [char2token[c] for c in s]
    decode = lambda s: ''.join([token2char[int(t)] for t in s])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = len(data)

    train_data = data[: int(0.9 * n)]
    test_data = data[int(0.9 * n) :]
    all_data = {'train': train_data, 'test': test_data}

    # train model
    m = GPT()

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    for i in range(max_iters):
        # sample text to get the training batch data
        xb, yb = get_batch(train_data)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if i % eval_interval == 0:
            losses = estimate_loss(m, all_data)
            print(f"{i} train loss: {losses['train']:.4f}, test loss: {losses['test']:.4f}")

    # generate 
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(context)
    print(decode(m.generate(context, max_new_tokens=100)[0].tolist()))