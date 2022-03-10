    命名实体识别可以看作是token（字）级别或实体级别的多分类，评估指标还是Precision，Recall和F1值。所以从token和实体两个角度来看，命名实体识别的评测方式分为两种，一是基于所有token标签的评测，二是考虑实体边界+实体类型的评测。基于所有token标签的评测，是一种宽松匹配的方法，就是把所有测试样本的真实标签展成一个列表，把预测标签也展成一个列表，然后直接计算Precision、Recall和F1值。考虑实体边界和实体类型的评测方法，是一种精准匹配的方法，只有当实体边界和实体类别同时被标记正确，才能认为实体识别正确。用的是CoNLL-2000的一个评估脚本，原本是用Perl写的，本demo基于python的修改，支持IOBES格式。


### 解决问题与修改
1. 使用pytorch框架，数据处理使用dataloader（数据按batch最大句子进行padding,没加<start>标签）。

2. 解码时，忽略<pad>标签，计算每个样本的真实长度。

3. 评价指标为f1, recall, precision评价指标为f1, recall, precision。
  
4. 实现两种评价方法，一是基于所有token标签的评测（将输出平铺后使用sklearn），二是考虑实体边界+实体类型的评测(使用conllebal.py)。
 
5. conllebal.py原代码是把文本、真实标签和预测标签用空格拼接，写入一个预测结果文件，再直接加载该文件进行评估，并写入一个评估结果文件。
    本脚本在evaluate函数中把dev数据与预测结果写入txt文件中后调用conllebal.py得到三个评价指标。
  
6. 10个epoch后在dev上的表现为，F1：81.51，recall:85.26,precision:75.98。
