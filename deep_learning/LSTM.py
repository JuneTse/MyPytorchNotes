#coding: utf-8

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

'''
1. `nn.Embedding(vocab_size,emb_dim)`: 定义embedding层
2. `nn.LSTM(input_size,hidden_size,num_layers,bias,batch_first,dropout,bidirectional)`:
参数:
    * input_size: 输入的维度
    * hidden_sieze: hidden state的维度
    * num_layers: 层数
    * bias: 默认为True,如果False,就不使用bias
    * batch_first: 如果为True, 输入和输出的维度为[batch, seq_len, feature_dim]
    * dropout: 如果不是0, 在输出层加dropout
    * bidirectional: 如果为True, 就变成了双向RNN, 默认为False
输入：
    * input, (h_0,c_0)
    * input: [seq_len,batch, input_size], 
     input也可以是一个packed的变长的序列，详见：`torch.nn.utils.rnn.pack_padded_sequence()`
    * h_0: [num_layers*num_directions* batch* hidden_size]
    * c_0: [num_layers* num_directions* batch_size]
输出：
    * output, (h_n,c_n)
    
3. `nn.LSTMCell(input_size,hidden_size,bias=True)`
LSTM Cell
参数：
    * input_size: 输入维度
    * hidden_size: hidden state 维度
    * bias=True: 是否使用bias
输入：
    * input,(h_0,c_0)
输出：
    * h_1,c_1
    
'''

word_to_ix={'hello':0,"world":1}
embeds=nn.Embedding(2,5) # 2 words in vocab, 5 dimentional embeddings
lookup_tensor=torch.LongTensor([word_to_ix['hello']])
hello_embed=embeds(autograd.Variable(lookup_tensor))
print(hello_embed) 