#coding: utf-8

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
1. `nn.Embedding(vocab_size,emb_dim)`: 定义embedding层
'''

word_to_ix={'hello':0,"world":1}
embeds=nn.Embedding(2,5) # 2 words in vocab, 5 dimentional embeddings
lookup_tensor=torch.LongTensor([word_to_ix['hello']])
hello_embed=embeds(autograd.Variable(lookup_tensor))
print(hello_embed) 