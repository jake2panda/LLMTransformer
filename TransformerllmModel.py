import torch
from torch import nn
from torch.nn import functional as F

torch.manual_seed(1447)

device = "cuda" if torch.cuda.is_available() else "cpu"

class Head(nn.Module):
	def __init__(self, head_size, n_embed, block_size, dropout):
		super().__init__()
		self.key = nn.Linear(n_embed, head_size, bias=False)
		self.query = nn.Linear(n_embed, head_size, bias=False)
		self.value = nn.Linear(n_embed, head_size, bias=False)
		self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		B, T, C = x.shape

		k = self.key(x)
		q = self.query(x)

		w = q @ k.transpose(-2,-1) * C**-0.5
		w = w.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		w = F.softmax(w, dim=-1)
		w = self.dropout(w)

		v = self.value(x)

		output = w @ v

		return output


class MultiheadAttention(nn.Module):
	def __init__(self,head_size, num_head, n_embed, block_size, dropout):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size, n_embed, block_size, dropout) for _ in range(num_head)])
		self.proj = nn.Linear(n_embed, n_embed)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.proj(out)
		out = self.dropout(out)
		return out


class FeedForward(nn.Module):
	def __init__(self, n_embed, dropout):
		super().__init__()
		self.net = nn.Sequential(
				nn.Linear(n_embed, 4 * n_embed),
				nn.ReLU(),
				nn.Linear(4 * n_embed, n_embed),
				nn.Dropout(dropout),
		)

	def forward(self, x):
		return self.net(x)


class Block(nn.Module):
	def __init__(self, n_embed, num_head, block_size, dropout):
		super().__init__()
		head_size = n_embed // num_head
		self.sa = MultiheadAttention(head_size, num_head, n_embed, block_size, dropout)
		self.ffwd = FeedForward(n_embed, dropout)
		self.ln1 = nn.LayerNorm(n_embed)
		self.ln2 = nn.LayerNorm(n_embed)

	def forward(self, x):
		x = x + self.sa(self.ln1(x))
		x = x + self.ffwd(self.ln2(x))

		return x


class TransfomerLLMModel(nn.Module):
  def __init__(self, vocab_size, n_embed, block_size, num_head, dropout):
    super().__init__()
    self.block_size = block_size
    self.token_embed = nn.Embedding(vocab_size, n_embed)
    self.positional_embedding = nn.Embedding(block_size,n_embed)
    #self.sa_head = Head(n_embed)
    # self.mul_head = MultiheadAttention(n_embed//num_head, num_head)
    # self.ffwd_layer = FeedForward(n_embed)

    self.blocks = nn.Sequential(
    		Block(n_embed, num_head, block_size, dropout),
    		Block(n_embed, num_head, block_size, dropout),
    		Block(n_embed, num_head, block_size, dropout),
    		Block(n_embed, num_head, block_size, dropout),
    		Block(n_embed, num_head, block_size, dropout),
    		nn.LayerNorm(n_embed),
    )

    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, input, target=None):

  	B, T = input.shape

  	tok_embed = self.token_embed(input)
  	pos_embed = self.positional_embedding(torch.arange(T, device=device)) # (T, n_embed)

  	x = tok_embed + pos_embed 	# (B, T, n_embed)
  	x = self.blocks(x)
  	logits = self.lm_head(x) 		# (4,8,n_embed/num_head) @ (n_embed, vocab_size) (B, T, vocab_size)

  	if target is None:
  		loss = None
  	else:
  		B, T, C = logits.shape
  		logits = logits.view(B*T, C)
  		target = target.view(B*T)
  		loss = F.cross_entropy(logits, target)

  	return logits, loss

  @torch.no_grad()
  def generate(self,idx, max_new_tokens):
    for _ in range(max_new_tokens):

    	idx_cond = idx[:, -self.block_size:]

    	logits, loss = self(idx_cond) # logits.shape = [1, T+1, 65] Here, T = 1

    	logits = logits[:, -1, :]
    	# logits.shape = [1, 65] this ensure always choose the last one
    	probs = F.softmax(logits, dim=-1) #probs.shape = [1, 65]
    	idx_next = torch.multinomial(probs, num_samples=1) # idx_next.shape = [1,1]

    	#print("[+] idx_next : {}".format(idx_next))
    	idx = torch.cat((idx, idx_next), dim=1)

    return idx

