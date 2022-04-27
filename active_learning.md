
主动学习是机器学习（更普遍的说是人工智能）的一个子领域，在统计学领域也叫查询学习、最优实验设计”(Active learning (sometimes called “query learning” or “optimal experimental design” in the statistics literature) is a subfield of machine learning and, more generally, artificial intelligence. )。
# 半监督学习和主动学习
在机器学习领域中，根据是否需要样本的标签信息可分为“监督学习”和“无监督学习”。半监督学习和主动学习都是从未标记样例中挑选部分价值量高的样例标注后补充到已标记样例集中来提高分类器精度，降低领域专家的工作量，但二者的学习方式不同：半监督学习一般不需要人工参与，是通过具有一定分类精度的基准分类器实现对未标注样例的自动标注；而主动学习有别于半监督学习的特点之一就是需要将挑选出的高价值样例进行人工准确标注。半监督学习通过用计算机进行自动或半自动标注代替人工标注，虽然有效降低了标注代价，但其标注结果依赖于用部分已标注样例训练出的基准分类器的分类精度，因此并不能保证标注结果完全正确。相比而言，主动学习挑选样例后是人工标注，不会引入错误类标 。
# 主动学习流程
如下图所示为常见的主动学习流程图，属于一个完整的迭代过程，模型可以表示为 A = (C, L, S, Q, U)。其中C表示分类器（1个或者多个）、L表示带标注的样本集、S表示能够标注样本的专家、Q表示当前所使用的查询策略、U表示未标注的样本集。流程图可解释为如下步骤（以分类任务为例）：

（1）选取合适的分类器（网络模型）记为 current_model 、主动选择策略、数据划分为 train_sample（带标注的样本，用于训练模型）、validation_sample（带标注的样本，用于验证当前模型的性能）、active_sample（未标注的数据集，对应于ublabeled pool）；

（2）初始化：随机初始化或者通过迁移学习（source domain）初始化；如果有target domain的标注样本，就通过这些标注样本对模型进行训练；

（3）使用当前模型 current_model 对 active_sample 中的样本进行逐一预测（预测不需要标签），得到每个样本的预测结果。此时可以选择 Uncertainty Strategy 衡量样本的标注价值，预测结果越接近0.5的样本表示当前模型对于该样本具有较高的不确定性，即样本需要进行标注的价值越高。

（4）专家对选择的样本进行标注，并将标注后的样本放至train_sapmle目录下。

（5）使用当前所有标注样本 train_sample对当前模型current_model 进行fine-tuning，更新 current_model；

（6）使用 current_model 对validation_sample进行验证，如果当前模型的性能得到目标或者已不能再继续标注新的样本（没有专家或者没有钱），则结束迭代过程。否则，循环执行步骤（3）-（6）。

![](./pictures/al.jpg)

# 主动学习三种算法
根据不同应用场景下主动学习挑选未标记样本的方式不同，将主动学习算法分为以下三种：
• 查询合成（query synthesis）算法
• 基于流（stream-based）算法
• 基于池（pool-based）算法


基于流的采样策略，这类做法将落在样例空间中的所有未标记样本按顺序根据采样策略决定标记或者丢弃。一般而言，这种采样策略需要逐个将未标记样本的信息含量跟事先设定好的固定阈值做比较，因此，无法得到未标记样本的整体结构分布及样本间的差异。仅适用于入侵检测、信息获取等场景
基于池的采样策略。可以利用样本的不确定性与多样性与代表性。将所有未标记样本视为一个“池”，从样本池中有选择性地标记样本，与基于流的算法相比，其通过计算样本池中所有未标记样本的信息含量并从中挑选出信息含量最好的样本进行标记，避免了设定固定阈值，查询无意义样本的情况，因而成为主动学习领域中研究最广泛的一类算法，在视频检索、文本分类、信息抽取等领域都有具体的应用。

![](./pictures/al_2.jpg)

# 基本采样方法
* 样本的不确定性与多样性与代表性

主要以基于池的样本采样策略为研究对象，研究制定适合关系抽取任务的样本采样策略，在保证模型达到一定性能的同时尽可能降低标注成本，即维护一个未标记样本池，通过主动学习样本策略迭代选择样本标记后训练模型，使得模型的泛化能力得到快速提升，其选择策略一般遵循贪心思想，即每次迭代从未标记样本集中选择某一属性最大(或最小)的样本进行标记。

## 基于不确定性的采样方法
基于不确定性的样本采样方法的选择策略的主要思想是从未标注样本集中选择分类模型给出较低置信度的样本，通过比较模型分类结果中的各样本的置信度大小，判断各待选样本能给分类器带来的信息含量，从未标注样本集中选择能带来最大信息量的样本获取标注加入标注样本集。

具体到文本多标签任务中，由于一条语句有多种可能的标记，可以用语句被预测为每一种类别的置信度来衡量样本的不确定性。multilabel_margin_sampling采用每个样本的预测结果中类别置信度的差作为样本不确定性的衡量标准;实际分类结果中，经常出现置信度的最高两个类别预测概率接近的情况，针对这种情况margin sampling 采用每个样本的预测结果中最大和次大的类别置信度的差作为样本不确定性的衡量标准,显然置信度差值越小的样本的实际类别更难区分，因此选择该类样本获取标注能够给基础模型带来更多的有效信息
```
def multilabel_margin_sampling(top_n, logists,  idxs_unlabeled):
    """
    多标签 的不确定性样本选择策略,比如一个样本(6个类别)经过sigmoid之后：preds为[0.1,0.3,0.5,0.1,0.8,0.74]
    选择策略：对于每个样本计算每个类别预测值与0.5的距离之和，对距离之和排序，距离之和越小说明该样本越不确定
    即：
    |0.1-0.5|+|0.3-0.5|+|0.5-0.5|+|0.1-0.5|+|0.8-0.5|+|0.74-0.5|

    :param top_n:
    :param logists:
    :param embedding:
    :param idx_unlabeled:
    :return:
    """
    preds = F.sigmoid(logists)
    uncertainty = torch.sum(torch.abs(preds - torch.ones_like(preds) * 0.5), dim=-1)
    uncertainty_sorted, idx_sorted = uncertainty.sort()  # 与0.5的距离之和越小则越不确定
    return idxs_unlabeled[idx_sorted[:top_n]]
```
```
def margin_sampling(top_n, logists,  idxs_unlabeled):
    preds = F.softmax(logists, dim=-1)
    preds_sorted, idxs = preds.sort(descending=True)  # 对类别进行排序
    U = preds_sorted[:, 0] - preds_sorted[:, 1]  # 计算概率最高的前两个类别之前的差值，
    residule_sorted, idx_sorted = U.sort()  # 按照差值升序排序，差值越小说明预测效果越差
    return idxs_unlabeled[idx_sorted[:top_n]]
```


## 基于多样性的采样方法
基于不确定的采样方法只考虑单个样本信息量问题，而忽视了选择样本的信息冗余问题，因此采样策略还可以从样本的多样性来考虑。若一个未标记样本与已标记样本中的样本过于接近，那么说明它与其接近的那些已标记样本具有很多相似信息，没有标记价值。因此，基于多样性的采样方法是指优先考虑那些与已标记样本集中所有样本最不相似的未标记样本，将其加入到已标记数据集中会使得该集合中样本的分布尽可能分散。常用的相似度标准有欧几里得距离、皮尔森相关系数，余弦距离等,本任务未涉及。

## 基于代表性的采样方法
基于代表性的采样方法考虑未标记数据集中整体数据分布，选出最具有代表性，能够更好的表示样本空间的样本，以提高基础模型的区分能力，最终达到提高主动学习算法效率的目的。以图为例，图中直线代表决策边界，正方形和三角形代表已标记的两类样本，圆形代表的未标记样本。因为样本A 位于决策边界上，所以它的不确定性最高，但实际上样本B 会给基础模型提供更多有效信息，这是因为样本A 在样本分布中属于孤立点，信息密度低，而样本B 在一定程度上拥有附近未标记样本的共性。
具体操作如下：对未标记样本集中所有样本聚类，将其划分到多个类簇中，使得相同类簇内的样本差异尽可能大，不同类簇间的样本差异尽可能小，再从中选取信息密度最大的样本，即距离类簇中心最近的样本作为本次挑选样本。本任务为涉及。

![](./pictures/al_3.jpg)

# 模型训练思路
整个训练的思路是：
1. 先用初始化的训练集（比如只有100个样本）来训练一个学习器，当学习器经过early stop 步之后验证集没有提升，停止训练，保存最优模型
2. 加载最优模型，对query pool中的样本进行预测，根据样本选择策略选择样本加入训练集
3. 在新加入样本的训练集上重新训练，重复1，2.直到query pool 中样本为零 或者 达到最大epoch

# 本任务针对文本多标签任务，样本选择策略对比效果
进行multilabel_margin_sampling、randon_sample样本选择策略发现在验证集上效果不行。于是对全量数据训练查看模型是否收敛。
在multi_margin_label进行测试的时候发现模型在基础100个数据训练的时候验证集F1就已经表现得比较好（85%以上）可能是数据集较为简单。
log日志保存在/data/aif/yuko/active_learning-main/saved/log/TransformersModel/0424_024836/info.log下，里面有训练过程产生的数据如下

* 第一个epoch
```
2022-04-13 10:16:47,014 - trainer - INFO -     spending time                 : 48.154051542282104
2022-04-13 10:16:47,016 - trainer - INFO -     epoch                         : 1
2022-04-13 10:16:47,016 - trainer - INFO -     loss                          : 0.40383868008852003
2022-04-13 10:16:47,016 - trainer - INFO -     macro_f1                      : 0.1666666666666666
2022-04-13 10:16:47,016 - trainer - INFO -     micro_f1                      : 0.5530000000000002
2022-04-13 10:16:47,016 - trainer - INFO -     sample_f1                     : 0.536
2022-04-13 10:16:47,016 - trainer - INFO -     val_loss                      : 0.3745390142202377
2022-04-13 10:16:47,016 - trainer - INFO -     val_macro_f1                  : 0.12522222222222254
2022-04-13 10:16:47,016 - trainer - INFO -     val_micro_f1                  : 0.5733142857142868
2022-04-13 10:16:47,016 - trainer - INFO -     val_sample_f1                 : 0.5739000000000003
2022-04-13 10:16:48,985 - trainer - INFO - Saving checkpoint: /data/aif/yuko/active_learning-main/saved/models/TransformersModel/0413_101440/checkpoint-epoch1.pth ...
2022-04-13 10:16:48,985 - trainer - INFO - best epoch:1,max val_macro_f1:0.12522222222222254
```
* 第2个epoch---第9个epoch模型F1持续上升，直至第10个epch不上升，加入train中加入新的50个，模型最不确定的数据，qeury池中减少对应的50个数据。
```
2022-04-13 10:26:32,725 - trainer - INFO - Checkpoint loaded. Resume training from epoch 10
2022-04-13 10:27:32,482 - trainer - INFO -     spending time                 : 92.9340386390686
2022-04-13 10:27:32,482 - trainer - INFO -     epoch                         : 17
2022-04-13 10:27:32,483 - trainer - INFO -     loss                          : 0.11669576784595848
2022-04-13 10:27:32,483 - trainer - INFO -     macro_f1                      : 0.308888888888889
2022-04-13 10:27:32,483 - trainer - INFO -     micro_f1                      : 0.9054285714285712
2022-04-13 10:27:32,483 - trainer - INFO -     sample_f1                     : 0.9016666666666666
2022-04-13 10:27:32,483 - trainer - INFO -     val_loss                      : 0.5731101288348437
2022-04-13 10:27:32,483 - trainer - INFO -     val_macro_f1                  : 0.13711111111111135
2022-04-13 10:27:32,483 - trainer - INFO -     val_micro_f1                  : 0.45040952380952476
2022-04-13 10:27:32,483 - trainer - INFO -     val_sample_f1                 : 0.43853333333333433
2022-04-13 10:27:32,483 - trainer - INFO -     num sample in train set       : 150
2022-04-13 10:27:32,483 - trainer - INFO -     num sample in query pool      : 3850
2022-04-13 10:27:34,201 - trainer - INFO - Saving checkpoint: /data/aif/yuko/active_learning-main/saved/models/TransformersModel/0413_101440/checkpoint-epoch17.pth ...
2022-04-13 10:27:34,202 - trainer - INFO - best epoch:10,max val_macro_f1:0.17311111111111188
```
整个过程维护self.idxs_labeled= np.zeros(len(query_dataset), dtype=bool)# 用于记录查询集中被标记过的样本，长度为4000。def _query(self):返回self.idxs_labeled，即若在此次查询中被标记为前五十则对应位置为true。
```
def _query(self):
    idxs_unlabeled = np.arange(len(self.query_pool))[~self.idxs_labeled] # 取出没有被标注的样本下标
    #让模型判断最不确定的50个数据,准备dataloader:
    unlabeled_dataloader = DataLoader(np.array(self.query_pool)[idxs_unlabeled],
                                          batch_size=self.query_pool.batch_size,
                                          shuffle=False,
                                          num_workers=self.query_pool.num_workers,
                                          collate_fn=self.query_pool.collate_fn)

    # 主动学习，对top_n 个query 进行标注
    query_labeled = self.config.init_ftn('active_learning', module_query_strategies)(logists=logists,
                                                                                         idxs_unlabeled=indexes)
    self.idxs_labeled[query_labeled] = True
    return query_labeled
     
```

并且可以通过tensboard进行可视化查看数据：
```
cd /active_learning-main
tensboard:tensorboard --logdir=/active_learning-main
```
返回地址后在浏览器中输入查看：

如下，左图为训练集指标，右图为验证集指标。图形横坐标表示step,纵坐标表示macrof1值。F1值出现明显震荡，在模型刚处理100条数据时，模型性能不断上升，直至f1不变后加入新的模型最不确定的查询数据50条。因此F1值随之下降到最低，再开始不断训练，性能又随之增加。反复如此。因此图形呈现震荡，但由于文本多标签数据集过于简单且标签不均衡，因此在未加入主动学习策略前macrof1就以达比较大的值（90%以上）,microf1却比较低（20%）。
![](./pictures/al4_microf1.jpg)

如下左图为训练集指标，右图为验证集指标。图形横坐标表示step,纵坐标表示loss值。随着step的增加，loss虽有震荡（原因与加入新数据有关）整体呈现下降趋势。
![](./pictures/al5_loss.jpg)

