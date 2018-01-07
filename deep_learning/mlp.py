#coding:utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

'''
1. 所有的神经网络组件都应该继承`nn.Moudle`并重写`forward` 方法
2. 参数可以使用`.cuda()`和`.cpu()`在GPU和CPU上切换
3. `torch.nn.functional`中定义了很多非线性和其他的函数
4. torch的reshape函数为： `view()`
5. `model.parameters()` 返回模型中的参数
6. `nn.NLLLoss()` 是negative log likelihood loss
7. 注意： pytorch会累加梯度，每次迭代时需要清空梯度: `model.zero_grad()`
8. 调用optimizer的`.step()`方法更新参数
'''

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
        ("Give it to me".split(), "ENGLISH"),
        ("No creo que sea una buena idea".split(), "SPANISH"),
        ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
             ("it is lost on me".split(), "ENGLISH")]
             
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

# 单词转换成唯一的idx
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print(word_to_ix)

VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2

class BoWClassifier(nn.Module):  #继承nn.Module
    def __init__(self,num_labels,vocab_size):
        #调用父类的`nn.Module`函数
        super(BoWClassifier,self).__init__()
        
        #定义需要的参数，这里需要全连接层参数A和b
        #pytorch中的`nn.Linear()` 提供了全连接仿射
        self.linear=nn.Linear(vocab_size,num_labels)
        
        #注意： non-linearity log softmax不需要参数
        
    def forward(self,inputs):
        #把输入传入到linear层
        #然后经过log_softmax
        #`torch.nn.functional`中定义了很多非线性和其他的函数
        return F.log_softmax(self.linear(inputs))
        
def make_bow_vector(sentence,word_to_ix):
    vec=torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]]+=1
    return vec.view(1,-1) #reshape
def make_target(label,label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

model=BoWClassifier(NUM_LABELS,VOCAB_SIZE)  
#输出模型中的参数
for param in model.parameters():
    print(param)
    
#执行模型需要把输入转换成`autograd.Variable`
sample=data[0]
bow_vector=make_bow_vector(sample[0],word_to_ix)
log_probs=model(autograd.Variable(bow_vector))
print(log_probs)
    

# Run on test data before we train, just to see a before-and-after
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Print the matrix column corresponding to "creo"
print(next(model.parameters())[:, word_to_ix["creo"]])

#计算损失
loss_function=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.1)

for epoch in range(100):
    for instance,label in data:
        #注意： pytorch会累加梯度，每次迭代时需要清空梯度
        model.zero_grad()
        #input和label要转换成Variable
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))
        # 前向运行
        log_probs=model(bow_vec)
        # 计算损失
        loss=loss_function(log_probs,target)
        # 计算梯度并优化
        loss.backward()
        optimizer.step()
        
for instance, label in test_data:
    bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
    log_probs = model(bow_vec)
    print(log_probs)

# Index corresponding to Spanish goes up, English goes down!
print(next(model.parameters())[:, word_to_ix["creo"]])