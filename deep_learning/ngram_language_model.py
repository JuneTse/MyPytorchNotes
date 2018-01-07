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

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = [([test_sentence[i], test_sentence[i + 1]], test_sentence[i + 2])
            for i in range(len(test_sentence) - 2)]
# print the first 3, just so you can see what they look like
print(trigrams[:3])

vocab = set(test_sentence)
word_to_ix = {word: i for i, word in enumerate(vocab)}

class NGramLanguageModeler(nn.Module):
    def __init__(self,vocab_size,emb_dim,context_size):
        super(NGramLanguageModeler,self).__init__()
        self.embeddings=nn.Embedding(vocab_size,emb_dim)
        self.linear1=nn.Linear(context_size*emb_dim,128)
        self.linear2=nn.Linear(128,vocab_size)
        
    def forward(self,inputs):
        embeds=self.embeddings(inputs).view([1,-1])
        out=F.relu(self.linear1(embeds))
        out=self.linear2(out)
        log_probs=F.log_softmax(out)
        return log_probs
        
losses=[]
loss_function=nn.NLLLoss()
model=NGramLanguageModeler(len(vocab),EMBEDDING_DIM,CONTEXT_SIZE)
optimizer=optim.SGD(model.parameters(),lr=0.001)

for epoch in range(10):
    total_loss=torch.Tensor([0])
    for context,target in trigrams:
        # step 1. 处理输入数据
        context_idxs=[word_to_ix[w] for w in context]
        context_var=autograd.Variable(torch.LongTensor(context_idxs))
        
        #清空梯度
        model.zero_grad()
        #step 3. forward
        log_probs=model(context_var)
        #step 4: 计算损失
        loss=loss_function(log_probs,autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        #step 5: 反向传播
        loss.backward()
        optimizer.step()
        
        total_loss+=loss.data
        
    losses.append(total_loss)
print(losses)

        
