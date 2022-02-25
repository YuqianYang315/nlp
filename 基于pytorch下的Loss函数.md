# 多标签
## sigmoid cross entropy loss
* out和target同维度

sigmoid函数又叫s曲线，又叫logistic函数
sigmoid一般用于二分类问题（这个大家都知道）， 看上图的这个公式就知道，扔进sigmoid的是一个标量，然后吐出一个标量。也就是说只是对一个量把它映射到（0,1），于是也就具有了概率意义。当然，当softmax的输入退化为一个标量时，softmax和sigmoid在形式上就一样了。

## BCEWithLogitsLoss
* out和target同维度

(BCELoss)BCEWithLogitsLoss用于单标签二分类或者多标签二分类,输出和目标的维度是(batch,C)，batch是样本数量，C(1/2)是类别数量，对于每一个batch的C个值，对每个值求sigmoid到0-1之间，所以每个batch的C个值之间是没有关系的，相互独立的，`所以之和不一定为1`。每个C值代表属于一类标签的概率。



# 多分类
## CrossEntropyLoss（）
* CrossEntropyLoss()是output 列是class维为one_hotted编码形式，但target不是one_hotted编码形式

交叉熵：其用来衡量在给定的真是分布下，使用非真实分布所指定的策略消除系统不确定性所需要付出努力的大小。

KL散度：衡量不同策略之间的差异，所以用KL散度来模拟模型缝补的拟合损失

CE (p,q)=-p(x)log(q(x))

H(p)=-p(x)log(p(x))

KL(P,Q)=p(x)log(q(x)/p(x))


CE（P,Q)=KL(P,Q)+H(P)

所以CE（P,Q)>H(P),当P,Q的分布越接近的时候，交叉熵CE(p,q)越越接近H（p）,故而选择交叉熵CE (p,q)作为Loss函数。

基于单个样本（即矩阵一行）来说：

又因为CE (p,q)=-p(x)log(q(x))，将taget数据写成onehot形式，使用onehot的数据向量用来模拟概率P（x）`【你也可以就理解为这个向量里的每一个元素都是一个概率，或者这个向量总体就是一个概率】`，用output矩阵一行模拟q(x)。对于每一个类别维度来说，作分类的时候只有一个y（k）=1，其他均为0，所以最终的结果是CE=-y（k）log(a(k)),k表示不为0的下标！

## nll_loss
* CrossEntropyLoss()是output 列是class维为one_hotted编码形式，但target不是one_hotted编码形式

NLLLoss 的 输入 是一个对数概率向量和一个目标标签. 它不会为我们计算对数概率. 适合网络的最后一层是log_softmax.

NLLLoss 函数输入 input 之前，需要对 input 进行 log_softmax 处理。

损失函数 nn.CrossEntropyLoss() 与 NLLLoss() 相同，事实上`log_softmax+NLLLoss=nn.CrossEntropyLoss`

# 回归或分类
## MSELoss均方误差
* MSELoss target不是one hot的形式，batch_x与batch_y的tensor都是FloatTensor类型。




