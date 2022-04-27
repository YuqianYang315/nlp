# 位置编码理想的设计：
* 它应该为每个字输出唯一的编码
* 不同长度的句子之间，任何两个字之间的差值应该保持一致（不可单纯用pos/L-1）
* 它的值应该是有界的(希望在[0,1]之间)

$$
  P_t^{(i)} =
\begin{cases}
sin(pos/10000^{2i/d}),  & \text{if $i=2k$} \\
cos(pos/10000^{2i/d}), & \text{if $i=2k+1$}
\end{cases}
$$
i属于[0,dim],t属于[0,seq_len]

随着i越来越大，$1/10000^{2i/d}$越来越小，对于三角函数来说，$周期=2pi/B$，周期越来越大。因此，位置编码矩阵会呈现这样的图像。
![](./pictures/pos.png)

# Encoder整体结构:

经过上面3个步骤，我们已经基本了解了Encoder的主要构成部分，下面我们用公式把一个Encoder block的计算过程整理一下：

1). 字向量与位置编码

X = E m b e d d i n g   L o o k u p ( X ) + P o s i t i o n a l   E n c o d i n g X = Embedding\ Lookup(X) + Positional\ Encoding
X=Embedding Lookup(X)+Positional Encoding

2). 自注意力机制
Q = L i n e a r ( X ) = X W Q K = L i n e a r ( X ) = X W K V = L i n e a r ( X ) = X W V X a t t e n t i o n = S e l f A t t e n t i o n ( Q ,   K ,   V ) Q = Linear(X) = XW_{Q}\\ K = Linear(X) = XW_{K}\\ V = Linear(X) = XW_{V}\\ X_{attention} = SelfAttention(Q, \ K, \ V)
Q=Linear(X)=XW 
Q
​
 
K=Linear(X)=XW 
K
​
 
V=Linear(X)=XW 
V
​
 
X 
attention
​
 =SelfAttention(Q, K, V)

3). self-attention残差连接与Layer Normalization
X a t t e n t i o n = X + X a t t e n t i o n X a t t e n t i o n = L a y e r N o r m ( X a t t e n t i o n ) X_{attention} = X + X_{attention}\\ X_{attention} = LayerNorm(X_{attention})
X 
attention
​
 =X+X 
attention
​
 
X 
attention
​
 =LayerNorm(X 
attention
​
 )

4). 下面进行Encoder block结构图中的第4部分，也就是FeedForward，其实就是两层线性映射并用激活函数激活，比如说R e L U ReLUReLU
X h i d d e n = L i n e a r ( R e L U ( L i n e a r ( X a t t e n t i o n ) ) ) X_{hidden} = Linear(ReLU(Linear(X_{attention})))
X 
hidden
​
 =Linear(ReLU(Linear(X 
attention
​
 )))

5). FeedForward残差连接与Layer Normalization
X h i d d e n = X a t t e n t i o n + X h i d d e n X h i d d e n = L a y e r N o r m ( X h i d d e n ) X_{hidden} = X_{attention} + X_{hidden}\\ X_{hidden} = LayerNorm(X_{hidden})
X 
hidden
​
 =X 
attention
​
 +X 
hidden
​
 
X 
hidden
​
 =LayerNorm(X 
hidden
​
 )

其中
X h i d d e n ∈ R b a t c h _ s i z e   ∗   s e q _ l e n .   ∗   e m b e d _ d i m X_{hidden} \in \mathbb{R}^{batch\_size \ * \ seq\_len. \ * \ embed\_dim}
X 
hidden
​
 ∈R 
batch_size ∗ seq_len. ∗ embed_dim

randrange
inputs_ids
segment_ids
cls：1
sep:2

# 基于huggingface的预训练模型微调

在实例化tokenizer和model的时候
tokenizer，和model可以与特定的模型关联的tokenizer类来创建(方法一)，也可以直接使用AutoTokenizer类来创建（方法二）
* tokenizer类方法可以在定义Dataset()类中getitem（）函数中使用self.tokenizer来实例化，通过返回的dict值获取"input_ids","token_type_ids"（前后句话0/1）,"atthenion_mask"来传入模型。
* model类方法可以在Mymodel()中定义self.bert来实例化

1. 方法一
```
from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertForSequenceClassification
from transformers.modeling_bert import BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForSequenceClassification.from_pretrained('bert-base-cased')
```
2. 方法二
```
from transformers import AutoTokenizer
from transformers import AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
model = AutoModel.from_pretrained('bert-base-cased')
```

注！！！bertmodel返回值可以查看API ，其中hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)，可对它进行多层拼接，也可使模型直接输出last_hidden_state。




# struct bert 
paper:https://arxiv.org/pdf/1908.04577.pdf
tructBERT是阿里在BERT改进上面的一个实践，模型取得了很好的效果，仅次于ERNIE 2.0, 因为ERNIE2.0 采用的改进思路基本相同，都是在pretraining的时候，增加预训练的obejctives
首先我们先看看一个下面英文和中文的两句话：
 ```
i tinhk yuo undresatnd this sentneces.
研表究明，汉字序顺并不定一影阅响读。比如当你看完这句话后，才发这现里的字全是都乱的
```
注意：上面的两个句子都是乱序的！
这个就是structBERT的改进思路的来源。对于一个人来说，字的或者character的顺序不应该是影响模型效果的因素，一个好的LM 模型，需要懂得自己纠错！
此外模型还在NSP的基础上，结合ALBERT的SOP，采用了三分类的方式，预测句子间的顺序。
* StructBERT的模型架构和BERT一样，它改进在于，在现有MLM和NSP任务的情况下，新增了两个预训练目标：Word Structural Objective和Sentence Structural Objective
## Word Structural Objective
输入的句子首先先按照BERT一样mask15%，（80%MASK，10%UNMASK，10%random replacement）。从未被mask的序列中随机选择部分子序列（使用超参数K KK来确定子序列的长度），将子序列中的词序打乱，让模型重建原来的词序(给定subsequence of length K， 希望的结果是把sequence 恢复成正确顺序的likelihood最大。)。
## Sentence Structural Objective
给定句子对（S1,S2），判断S2是S1的下一个句子、上一个句子、毫无关联的句子（三分类问题）

采样时，对于一个句子S，1/3的概率采样S的下一句组成句对，1/3的概率采样S的上一句，1/3的概率随机采样另一个文档的句子组成句对。

# RoBERTa

```
import os
import torch
from transformers.models.bert import tokenization_bert, configuration_bert, modeling_bert

model_dir = "/sentiment_analysis/absa_bert_pair/bert_base_uncased"
my_model_dir = "/sentiment_analysis/absa_bert_pair/mybert"
#tokenizer = tokenization_bert.BertTokenizer.from_pretrained(os.path.join(my_model_dir, 'dic.txt'))
tokenizer = tokenization_bert.BertTokenizer.from_pretrained(model_dir)
print(tokenizer.vocab_size)
config = configuration_bert.BertConfig.from_pretrained(os.path.join(my_model_dir, 'conf.json'))
bert = modeling_bert.BertModel(config)
print(bert.__repr__())
print(bert.embeddings.word_embeddings.weight[0])
state_dict = torch.load(os.path.join(my_model_dir, 'bert.bin'))
state_dict = {key[5:]: val for key, val in state_dict.items()}
#print(state_dict.keys())
bert.load_state_dict(state_dict, strict=False)
print(bert.embeddings.word_embeddings.weight[0])
# # print("----------")
```
