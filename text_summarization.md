# 1.抽取式
## 1.1 Pagerank无监督的方式

1. seq2vec(tf_idf/gensim中使用BM2.5)
2. 计算相似矩阵
3. 构建相似图
4. 计算每个节点的pagerank
5. 输出top


## 1.2 pagerank算法原理：

$PR(A)=\frac{(1-d)}{N}+\frac{(PR(T_1))}{C(T_1)}+\ldots+\frac{(PR(T_N))}{C(T_N)}$

PR(A) 是页面A的PR值
PR(Ti)是页面Ti的PR值，在这里，页面Ti是指向A的所有页面中的某个页面
C(Ti)是页面Ti的出度，也就是Ti指向其他页面的边的个数
d 为阻尼系数，其意义是，在任意时刻，用户到达某页面后并继续向后浏览的概率，并即使一个页面没有其他页面指向它，他的PR值也不会为0。

```
假训练：为了便于计算，我们假设每个页面的PR初始值为1，d为0.5。
下面是迭代计算n轮之后，各个页面的PR值将趋于不变,一般要设置收敛条件：比如上次迭代结果与本次迭代结果小于某个误差，我们结束程序运行；比如还可以设置最大循环次数。
```
## 1.3 textrank是pagerank的变形，使用textrank端到端：
```
from gensim.summarization import summarize
text=" "
print summarize(text)
```
## 1.4 bertsum抽取式有监督的方式
# 2. 生成式
有监督：OpenNMT（neural machine translation）
无监督:OPENAI-gpt

## 2.1 评价指标
### 2.1.1 ROUGE
公式：$ROUGE-N=\frac{共现的字/词的个数}{摘要个数}$
对应的N即为连续字个数。
* 人工摘要句子y：文章内容新颖 （6个字）
* 候选句子摘要句子即机器生成x：这篇文章内容是新颖的（10个字）

重叠的部分是6个字
precision 两个句子重叠部分的n-gram/len(x) 6/10
recall 两个句子重叠部分的n-gram/len(y) 6/6
```
from rouge import Rouge
rouge = Rouge()
_scores = rouge.get_scores(self.pred, self.gold, avg=True)
```
### 2.1.1 BLUE：

公式：$
BLEU=BP⋅exp(\sum_{i=1}^Nw_nlogP_n)
$

n表示N_gram，BP 表示Brevity Penalty最短惩罚因子，$P_n$表示Modified n-gram Precision。
$
  BP =
\begin{cases}
1,  & \text{if c>r} \\
e^{1-r}, & \text{if c≤r}
\end{cases}
$
```
了解：

#NLTKnltk.align.bleu_score模块实现了这里的公式，主要包括三个函数，两个私有函数分别计算P和BP，一个函数整合计算BLEU值

# 计算BLEU值
def bleu(candidate, references, weights)

# （1）私有函数，计算修正的n元精确率（Modified n-gram Precision）
def _modified_precision(candidate, references, n)

# （2）私有函数，计算BP惩罚因子
def _brevity_penalty(candidate, references)

```
eg:

候选译文（Predicted）：

It is a guide to action which ensures that the military always obeys the commands of the party

参考译文（Gold Standard）

1：It is a guide to action that ensures that the military will forever heed Party commands

2：It is the guiding principle which guarantees the military forces always being under the command of the Party

3：It is the practical guide for the army always to heed the directions of the party
* 计算$P_1$:

|词|候选译文|参考译文1|参考译文3|Max|Min
|:----|:----|:----|:----|:----|:----
|the|3|1|4|4|3
|obeys|1|0|0|0|0
|...

首先统计候选译文里每个词出现的次数，然后统计每个词在参考译文中出现的次数，Max表示3个参考译文中的最大值，Min表示候选译文和Max两个的最小值
然后将每个词的Min值相加，将候选译文每个词出现的次数相加，然后两值相除即得

$P_1=\frac{3+0+\ldots}{3+1+\ldots}=.75$

同理$P_2,\ldots,P_N$计算可得。

* 计算$BP$:
由BP的公式可知取值范围是(0,1]，候选句子越短，越接近0。
候选翻译句子长度为18，参考翻译分别为：16，18，16。
所以c=18，r=18（参考翻译中选取长度最接近候选翻译的作为r）
所以$BP=e_0=1$




```
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
reference = [['The', 'cat', 'is', 'on', 'the', 'mat']]
candidate = ['The', 'cat', 'sat', 'on', 'the', 'mat']
smooth = SmoothingFunction()  # 定义平滑函数对象
score = sentence_bleu(reference, candidate, weight=(0.25,0.25, 0.25, 0.25), smoothing_function=smooth.method1)
corpus_score = corpus_bleu([reference], [candidate], smoothing_function=smooth.method1)
```
NLTK 中提供了两种计算BLEU的方法，实际上在sentence_bleu中是调用了corpus_bleu方法，另外要注意reference和candinate连个参数的列表嵌套不要错了，weight参数是设置不同的n−gram的权重，另外weight元组中的数量决定了计算BLEU时，会用几个n−gram，以上面为例，会用1−gram,2−gram,3−gram,4−gram。SmoothingFunction是用来平滑log函数的结果的，防止fn=0时，取对数为负无穷。
