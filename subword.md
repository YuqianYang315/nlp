# subword
很多时候用多了框架的API，觉得分词和生成字典就是调用的事情，不过事情并没有那么简单，比如其中涉及到的未登录词的问题，就对任务性能影响很大。一种很朴素的做法就是将未见过的词编码成#UNK ，有时为了不让字典太大，只会把出现频次大于某个阈值的词丢到字典里边，剩下所有的词都统一编码成#UNK 。对于分词和生成字典方面，常见的方法有：
* 给低频次再设置一个back-off 表，当出现低频次的时候就去查表。这种方法简单直接，若干back-off做的很好的话，对低频词的效果会有很大的提升，但是这种方法依赖于back-off表的质量，而且也没法处理非登录词问题。
* 不做word-level转而使用char-level，既然以词为对象进行建模会有未登录词的问题，那么以单个字母或单个汉字为对象建模不就可以解决了嘛？因为不管是什么词它肯定是由若干个字母组成。这种方法的确可以从源头解决未登录词的问题，但是这种模型粒度太细。
```
未登录词：简单来讲就是在验证集或测试集出现了训练集从来没见到过的单词。
```
下面举例word-level和subword-level的一种直观感受：
```
训练集的词汇: old older oldest smart smarter smartest
word-level 词典: old older oldest smart smarter smartest 长度为 6
subword-level 词典: old smart er est 长度为 4
```
如何将词分解成subword，依据不同的策略，产生了几种主流的方法: Byte Pair Encoding (BPE)、wordpiece 和 Unigram Language Model等。值得一提的是，这几种算法的处理流程跟语言学没有太大的关系，单纯是统计学的解决思路，Subword模型的主要趋势：
* 与单词级别的模型架构相同，但使用的是字符级别的输入
* 采用混合架构，输入主要是字符，但是会混入其他信息
# Byte Pair Encoding（BPE）
算法过程

       比如我们想编码：

              aaabdaaabac

       我们会发现这里的aa出现的词数最高（我们这里只看两个字符的频率），那么用这里没有的字符Z来替代aa：

               ZabdZabac

               Z=aa

       此时，又发现ab出现的频率最高，那么同样的，Y来代替ab：

               ZYdZYac

               Y=ab

               Z=aa

       同样的，ZY出现的频率大，我们用X来替代ZY：

               XdXac

               X=ZY

               Y=ab

               Z=aa

       最后，连续两个字符的频率都为1了，也就结束了。
       解码的时候，就按照相反的顺序更新替换即可。

