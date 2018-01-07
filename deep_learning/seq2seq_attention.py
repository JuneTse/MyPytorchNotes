#coding: utf-8

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
1. `torch.bmm(batch1,batch2,out=None)`
* 执行batch matrix-matrix product
* batch1和batch2必须是3D的tensor
* 如果batch1是：b*n*m的tensor， batch2是: b*m*p的tensor, out就是b*n*p的tensor
注意： 这个函数不会broadcast, 要使用broadcast，看`torch.matmul()`
    
'''

word_to_ix={'hello':0,"world":1}
embeds=nn.Embedding(2,5) # 2 words in vocab, 5 dimentional embeddings
lookup_tensor=torch.LongTensor([word_to_ix['hello']])
hello_embed=embeds(autograd.Variable(lookup_tensor))
print(hello_embed) 