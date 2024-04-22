import torch
import torch.nn as nn
from torch.nn import functional
import utils

# --- Hyper Parameters ---
batch_size = 32 # Number of independent sequences processed in parallel
block_size = 8 # Maximum context length for prediction
max_iterations = 3000
evaluation_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
evaluation_iterations = 200
# --- Hyper Parameters ---

torch.manual_seed(1337)

contents = utils.read_data('Data')

# Unique characters in data
unique_characters = sorted(list(set(contents)))
vocabulary_size = len(unique_characters)

print('Unique characters in file contents: ', unique_characters)
print('Vocabulary size: ', vocabulary_size)

# Mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(unique_characters) }
itos = { i:ch for i,ch in enumerate(unique_characters) }

# Encoder: string to integers
encode = lambda s: [stoi[c] for c in s]
# Decoder: Integers to string
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(contents), dtype=torch.long)

# First 90% of data is for training
n = int(0.9*len(data))
training_data = data[:n]
# Last 10% of data is for validation
validation_data = data[n:]

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(evaluation_iterations)
        for k in range(evaluation_iterations):
            data = training_data if split == 'train' else validation_data
            X, Y = utils.get_batch(data, batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # Each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Get predictions
            logits, loss = self(idx)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = functional.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocabulary_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iterations):
    # Evaluate the loss on training and validation sets
    if iter % evaluation_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: training loss {losses['train']:.4f}, validation loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = utils.get_batch(training_data, batch_size, block_size, device)

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
