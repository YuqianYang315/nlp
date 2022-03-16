# HMCN简介
hierarchical classification与普通的classification任务不同，预测标签之间存在分级关系，对应的，预测得到的标签集合需要满足预先设定好的类目树关系。

因此，hierarchical classification需要设计特定的模型结构与损失函数，以便于将类目树的约束添加到训练过程中。

HMCN（hierarchical classification with hierarchical_multilabel_classification_networks）是HMCN-F前馈、HMCN-R循环的结合，融合全局和局部处理方式。发表于ICML 2018。paper:https://proceedings.mlr.press/v80/wehrmann18a/wehrmann18a.pdf

Tencent NeuralNLP-NeuralClassifier库中model实现的是HMCN-F前馈。

# HMCN-F
这版代码实现与论文中的图示还是略有差异（参见下图中的红叉位置）。
为了与图示匹配，将参数调整为self.hierarchical_depth= [0, 384, 384, 384]，self.global2local = [0, 16, 192, 512]，self.hierarchical_class= [10, 100, 200]，对应生成3个globallayer与3个locallayer，具体流程说明如下：

图示中黄底绿字为tensor shape，蓝字为Weight shape；
起始送入tensor名为doc_embedding（图中x），向量维度为128，经过global_layer_0以后维度为384（图中$A_G^1$），进一步通过local_layer中的两次权重变换，维度依次调整为16（图中$A_L^1$）与10（图中$P_L^1$），完成本级类目预测，同时$A_G^1$与x做concatenate后作为global_layer_activation送入下一级类目预测；
重复上述流程三次，即完成示例的三级类目预测；需要说明的是，与论文图示不同，在三级类目预测完成后，在代码中没有将$A_G^3$与x做concatenate，而是将$A_G^3$直接通过linear层调整维度后变为$P_G$；
最后$P_L^1$、$P_L^2$与$P_L^3$做concatenate后与$P_G$做加权融合后，即得到最后的输出$P_F$；
![](./pictures/hmcn1.jpg)

```
# HMCN_F代码实现如下，所需参数配置
# https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/conf/train.json#L125
  "TextRCNN": {
    "kernel_sizes": [
        2,
        3,
        4
    ],
    "num_kernels": 100,
    "top_k_max_pooling": 1,
    "hidden_dimension":64,
    "rnn_type": "GRU",
    "num_layers": 1,
    "bidirectional": true  # 如果为true，则代码中调整hidden_dimension为64*2=128
  },

# https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/conf/train.json#L153
  "HMCN": {
    "hierarchical_depth": [0, 384, 384, 384, 384],
    "global2local": [0, 16, 192, 512, 64]
  },

# 模型init，定义所需的算子
# https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/classification/hmcn.py#L44
        self.local_layers = torch.nn.ModuleList()
        self.global_layers = torch.nn.ModuleList()
        for i in range(1, len(self.hierarchical_depth)):
            self.global_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dimension + self.hierarchical_depth[i-1], self.hierarchical_depth[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.hierarchical_depth[i]),
                    torch.nn.Dropout(p=0.5)
                ))
            self.local_layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(self.hierarchical_depth[i], self.global2local[i]),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(self.global2local[i]),
                    torch.nn.Linear(self.global2local[i], self.hierarchical_class[i-1])
                ))
        self.global_layers.apply(self._init_weight)
        self.local_layers.apply(self._init_weight)
        self.linear = torch.nn.Linear(self.hierarchical_depth[-1], len(dataset.label_map))
        self.linear.apply(self._init_weight)
        self.dropout = torch.nn.Dropout(p=config.train.hidden_layer_dropout)

# 模型forward，定义前向流
# https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/classification/hmcn.py#L101
        global_layer_activation = doc_embedding  # doc_embedding即为提取得到的global特征
        batch_size = doc_embedding.size()[0]
        for i, (local_layer, global_layer) in enumerate(zip(self.local_layers, self.global_layers)):
            local_layer_activation = global_layer(global_layer_activation)
            local_layer_outputs.append(local_layer(local_layer_activation))
            if i < len(self.global_layers)-1:
                global_layer_activation = torch.cat((local_layer_activation, doc_embedding), 1)
            else:
                global_layer_activation = local_layer_activation


        global_layer_output = self.linear(global_layer_activation)
        local_layer_output = torch.cat(local_layer_outputs, 1)
        return global_layer_output, local_layer_output, 0.5 * global_layer_output + 0.5 * local_layer_output 
```
# recursive_regularize函数定义与上层调用
* 通过cal_recursive_regularize计算层次结构各子结点的概率与对应父结点的概率的欧式距离损失来约束模型预测。
cal_recursive_regularize的函数接口为def cal_recursive_regularize(self, paras, hierar_relations, device="cpu”)，在分析具体实现之前，需要理解与该函数相关的参量定义。

cal_recursive_regularize的上层调用在loss函数中(模型仅有一个logits输出，分别添加类别损失与hierarchy惩罚损失)，通过argvs传入hierar_penalty, hierar_paras, hierar_relations三个参数：
* hierar_penalty为recursive损失的权重系数，通过self.conf.task_info.hierar_penalty指定
* hierar_paras来自于model.linear.weight，也就是模型全局线性分类层的权重，对应上图中的linear算子，示例权重维度为(384*310)，其中384为超参、310=10+100+200为三级类目的数量和
* hierar_relations为类目关系，类目关系self.hierar_relations的构建在model_util.py文件def get_hierar_relations(hierar_taxonomy, label_map)中，其中label_map为标签名称与id的映射，hierar_taxonomy为标签层级关系的文本文件，其中每行内容为parent_label \t child_label_0 \t child_label_1 \n，依次读取获得parent_label与children_label，最后构建relation字典，key为parent_label_id，value为所有children_label_ids的列表；效果为将任意深度、任意大小的hierar_taxonomy类目树拆分为仅有parent-children的单一层级关系。向量维度实际为(所有类目标签的总数, linear层计算该ID的权重（长度为384的向量）)。

```
#上一级调用为ClassificationLoss的loss_fn函数入口：
class ClassificationLoss(torch.nn.Module):
    def __init__(self, label_size, class_weight=None,
                 loss_type=LossType.SOFTMAX_CROSS_ENTROPY):
    def forward(self, logits, target,
                use_hierar=False,
                is_multi=False,
                *argvs):
        device = logits.device
        if use_hierar:
            assert self.loss_type in [LossType.BCE_WITH_LOGITS,
                                      LossType.SIGMOID_FOCAL_CROSS_ENTROPY]
            if not is_multi:
                target = target.long()
                target = torch.eye(self.label_size)[target].to(device)
            hierar_penalty, hierar_paras, hierar_relations = argvs[0:3]#这里获取cal_recursive_regularize所需的三个参数！
            return self.criterion(logits, target) + \
                   hierar_penalty * self.cal_recursive_regularize(hierar_paras,
                                                                  hierar_relations,
                                                                  device)


#上上一级调用为loss_fn为入口:
在模型的train/eval函数体中

         for batch in data_loader:
            label_ids = batch[ClassificationDataset.DOC_LABEL].to(self.conf.device)
            label_list = batch[ClassificationDataset.DOC_LABEL_LIST]
            token_ids = batch[ClassificationDataset.DOC_TOKEN].to(self.conf.device)
            token_len = batch[ClassificationDataset.DOC_TOKEN_LEN].to(self.conf.device)
            token_mask = batch.get(ClassificationDataset.DOC_TOKEN_MASK).to(self.conf.device)

            # hierarchical classification
            if self.conf.task_info.hierarchical:
                logits = model(input_ids=token_ids, input_length=token_len, input_mask=token_mask)
                if hasattr(model, "module"):
                    linear_paras = model.module.linear.weight
                else:
                    linear_paras = model.linear.weight
                is_hierar = True
                used_argvs = (self.conf.task_info.hierar_penalty, linear_paras, self.hierar_relations)#这里获取cal_recursive_regularize所需的三个参数！
                loss = self.loss_fn(
                    logits,
                    label_ids.to(self.conf.device),
                    is_hierar,
                    is_multi,
                    *used_argvs)
            # hierarchical classification with hierarchical_multilabel_classification_networks
            # HMCN输出包括层级的输出和全局的输出，相应的损失函数也由local 损失和全局loss组成
            elif self.conf.model_name == "HMCN":
                global_logits, local_logits = model(input_ids=token_ids, input_length=token_len, input_mask=None) #对应图中的global_logits:$P_G$,和local_logits:concatenate后的$P_L^1$、$P_L^2$与$P_L^3$
                # 加权平均后的logits则作为模型的整体前向结果 仅在预测时使用
                logits = self.conf.train.global_loss_weight * global_logits + (1 - self.conf.train.global_loss_weight) * local_logits
                # HMCN的损失计算是将约束分别添加到global prediction与local prediction上
                loss = self.loss_fn(
                    global_logits,
                    label_ids.to(self.conf.device),
                    False,
                    is_multi)
                loss += self.loss_fn(
                    local_logits,
                    label_ids.to(self.conf.device),
                    False,
                    is_multi)
```
# recursive_regularize代码分析
* 核心思想是 父类标签的预测参数与子类标签的预测权重参数应当趋于一致，因此通过L2损失约束对应的权重即可。
```
   def cal_recursive_regularize(self, paras, hierar_relations, device="cpu"):
        """ Only support hierarchical text classification with BCELoss
        """
        recursive_loss = 0.0
        for i in range(len(paras)):  # 遍历所有标签, 0 ~ len(paras)-1代表所有标签ID
            if i not in hierar_relations:  # 当前标签ID是否为parent标签，如不是则不做后续处理
                continue
            children_ids = hierar_relations[I]  # 当前标签ID为parent，返回对应的所有children标签ID
            if not children_ids:  # 如果没有children标签则不做后续处理
                continue
            children_ids_list = torch.tensor(children_ids, dtype=torch.long).to(  # 将list格式的children标签转换为tensor
                device)
            children_paras = torch.index_select(paras, 0, children_ids_list)  # shape (children_num, 384)
            parent_para = torch.index_select(paras, 0,
                                             torch.tensor(i).to(device))  # shape (1, 384)
            parent_para = parent_para.repeat(children_ids_list.size()[0], 1)  # shape (children_num, 384)
            diff_paras = parent_para - children_paras  # shape (children_num, 384)
            diff_paras = diff_paras.view(diff_paras.size()[0], -1)  # shape (children_num, 384)
            recursive_loss += 1.0 / 2 * torch.norm(diff_paras, p=2) ** 2
        return recursive_loss
```
# 总述
* Tencent NeuralNLP-NeuralClassifier库中train函数中包含两种层次分类实现： hierarchical classification和# hierarchical classification with hierarchical_multilabel_classification_networks（HMCN）（代码实现的model是HMCN-F前馈）。前者model输出一个logits，并且使用 recursive_regularize函数作为loss函数的hierar损失，计算层次父子标签向量的L2正则化来约束对应权重。后者model输出global_logits, local_logits，分别计算loss（未使用recursive_regularize）,再相加。
