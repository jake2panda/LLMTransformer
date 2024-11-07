import torch
from torch.nn import functional as F
from torch import nn
from TransformerllmModel import TransfomerLLMModel
from transformers import AutoTokenizer


torch.manual_seed(1447)


# define hyper parameters here

device = "cuda" if torch.cuda.is_available() else "cpu"
n_embed = 64
block_size = 8
num_head = 8
lr = 3e-4
dropout = 0.2





#If anyone wanna use gpt2 tokenizer.You can use any tokenizer you but keep in mind the vocab_size


#tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")




def get_batch(data):
  ix = torch.randint(len(data) - 8, (4,))
  xa = torch.stack([data[i:i+8] for i in ix])
  ya = torch.stack([data[i+1:i+1+8] for i in ix])
  return xa, ya


# If anyone wants to implement average losses in eval() mode use this estimate_loss() function



@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(1)   
        X, Y = get_batch(split)
        logits, loss = model.forward(X, Y)
        losses[0] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


with open("gotest.txt") as f:
	text = f.read()


# Encoding text

# vocab_size = tokenizer.vocab_size


# tokens = tokenizer.encode(text)


############################################################
##################### Here this is character wise encoder and decoder ######################
############################################################


chars = list(set(text))

vocab_size = len(chars)

# print("[+] vocab_size : {}".format(vocab_size))

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}


encode = lambda a : [stoi[q] for q in a]
decode = lambda a : [itos[i] for i in a]

tokens = encode(text)


train = torch.tensor(tokens[:int(len(tokens) * 0.8)])
test = torch.tensor(tokens[int(len(tokens) * 0.8):])


#########################--- END Encoder and Decoder --############


# train = torch.tensor(tokens[:int(len(tokens) * 0.8)])
# test = torch.tensor(tokens[int(len(tokens) * 0.8):])




model = TransfomerLLMModel(vocab_size, n_embed, block_size, num_head, dropout)

model = model.to(device)


# here printing 400 tokens

#using own tokenizer



#Implementing user prompt to ask question to out llm
## this prompt is not like question answer but it is just sequence completing user prompt


"""
If someone want to implement user prompt uncomment user prompt code

"""

# user_prompt = input("Ask-LLM ~$ ")


# encoded_prompt = torch.tensor(encode(user_prompt), dtype=torch.long)

# prompt_len = len(encoded_prompt)

# prompt = encoded_prompt.view(1,prompt_len)


# Generating text before traing the model


#If you use user prompt use this one
#print("".join(decode(model.generate(prompt.to(device), max_new_tokens=400)[0].tolist())))


#if you use without user prompt
print("".join(decode(model.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=400)[0].tolist())))

# this one is for gpt2 ro any other tokenizer

#print("".join(tokenizer.decode(model.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=400)[0].tolist())))


#typical Training loop here using Adam optimizer

optimizer = torch.optim.Adam(model.parameters(), lr)


for epoch in range(10000):
	za, zb = get_batch(train)
	logits,loss = model.forward(za.to(device), zb.to(device))

	if epoch % 1000 == 0:
		print("loss : ", loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()



#After training the model



# using own implemented decoder
0
## if you use user prompt
#print("".join(decode(model.generate(prompt.to(device), max_new_tokens=400)[0].tolist())))


## this one is for without user prompt

print("".join(decode(model.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=400)[0].tolist())))


# this is for any other standard tokenizer

#print("".join(tokenizer.decode(model.generate(torch.zeros((1,1), dtype=torch.long).to(device), max_new_tokens=400)[0].tolist())))


