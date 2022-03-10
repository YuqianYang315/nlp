import os
import logging
import numpy

import torchsnooper
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torchcrf import CRF
from torch.optim import Adam
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pickle
import numpy as np

from conlleval import evaluate_ner


# from torchtext.legacy.data import Field
# from torchtext.legacy import data

class Myconfig():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_path = "ner/data/train.txt"
        self.dev_path = "ner/data/dev.txt"
        self.test_path = "ner/data/test.txt"

        self.batch_size = 128
        self.embedding_size = 256
        self.hidden_size = 128

        self.clip = 5
        self.lr = 0.001
        self.epoch = 10

        self.ner_model_path = "ner/ner.model"
        self.ner_result_path = "ner/result_path.txt"

    pass


# START_TAG = "<start>"
# END_TAG = "<end>"
# PAD = "<pad>"
# UNK = "<unk>"
def build_corpus(data_file, make_vocab=True):
    """读取数据"""

    word_lists = []
    tag_lists = []
    with open(data_file, 'r', encoding='utf-8') as f:
        word_list = []
        tag_list = []
        word_special = []
        tag_special = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                if len(word_list) == len(tag_list):
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                word_list = []
                tag_list = []

        if make_vocab:
            token2id = build_map(word_lists)
            tag2id = build_map(tag_lists)
            return word_lists, tag_lists, token2id, tag2id
            # return [" ".join(i) for i in word_lists], [" ".join(i) for i in tag_lists], token2id, tag2id
        else:
            return word_lists, tag_lists
            # return [" ".join(i) for i in word_lists], tag_lists


def build_map(lists):
    maps = {"PAD": 0, "UNK": 1}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)
    return maps


# def extend_maps(token2id, tag2id, for_crf=True):
#     token2id[UNK] = len(token2id)
#     token2id[PAD] = len(token2id)
#     tag2id[UNK] = len(tag2id)
#     tag2id[PAD] = len(tag2id)
#     # 如果是加了CRF的bilstm  那么还要加入<start> 和 <end>token
#     if for_crf:
#         tag2id[START_TAG] = len(tag2id)
#         tag2id[END_TAG] = len(tag2id)
#
#     return token2id, tag2id


class Mydatasets(Dataset):
    def __init__(self, datas, tags, word_2_index, tag_2_index):
        # 传入data和target路径，进行读取以获得文件，这里可以将文件路径存入config中。assert len(data)=len(target)
        self.datas = datas
        self.tags = tags
        self.word_2_index = word_2_index
        self.tag_2_index = tag_2_index
        assert len(self.datas) == len(self.tags)
        pass

    def __getitem__(self, index):
        # 对数据在这里进行处理：分词，取最大长度等
        # 返回形式一般为input,target,input_len/input2index,target_len/target2index,根据任务看返回什么比较方便
        data = self.datas[index]
        tag = self.tags[index]
        data_index = [self.word_2_index.get(i, self.word_2_index["UNK"]) for i in data]
        tag_index = [self.tag_2_index.get(i, self.tag_2_index["UNK"]) for i in tag]
        if len(data_index) != len(tag_index):
            print("data:{}".format(data))
            print("tag:{}".format(tag))

        return data_index, tag_index
        pass

    def __len__(self):
        # return len(self.input_lines)
        return len(self.tags)
        pass

    def collate_fn(self, batch):
        # for i in batch:
        #     if not len(i[0])==len(i[1]):
        #         print((len(i[0]),len(i[1])))
        #     else:
        #         print("相等")
        batch_data, batch_tag = zip(*batch)

        batch_max_len = max([len(i) for i in batch_data])
        # max_data= batch_data[[len(i) for i in batch_data].index(max([len(i) for i in batch_data]))]
        # tag_data=batch_tag[[len(i) for i in batch_tag].index(max([len(i) for i in batch_tag]))]

        batch_data = [j + [self.tag_2_index["PAD"]] * (batch_max_len - len(j)) for j in batch_data]
        batch_tag = [i + [self.tag_2_index["PAD"]] * (batch_max_len - len(i)) for i in batch_tag]
        # max_batch_data=max([len(i) for i in batch_data])
        # max_batch_tag=max([len(i) for i in batch_tag])

        batch_data = torch.LongTensor(batch_data)
        batch_tag = torch.LongTensor(batch_tag)
        return batch_data, batch_tag

        pass


class Mymodel(nn.Module):
    def __init__(self, corpus_num, embedding_num, hidden_size, class_num, bi=True) -> None:
        super().__init__()
        self.embedding = nn.Embedding(corpus_num, embedding_num)
        self.lstm = nn.LSTM(embedding_num, hidden_size, bidirectional=bi, batch_first=True)
        if bi:
            self.fc = nn.Linear(hidden_size * 2, class_num)
        else:
            self.fc = nn.Linear(hidden_size, class_num)
        pass

        self.cross_loss = nn.CrossEntropyLoss()
        self.crf = CRF(class_num, batch_first=True)

    # @torchsnooper.snoop()
    def forward(self, batch_data):
        embeded = self.embedding(batch_data)
        out, _ = self.lstm(embeded)
        pre = self.fc(out)
        return pre
        pass

    # @torchsnooper.snoop()
    def compute_loss(self, batch_data, batch_label):

        out = self.forward(batch_data)
        loss = -self.crf(out, batch_label)
        return loss

    def decode(self, x):
        out = self.forward(x)
        predicted_index = self.crf.decode(out)
        return predicted_index


# 基于所有token标签的评测
# def evaluate(model,dev_dataloder,config):
#     model.eval()
#     all_pre=[]
#     all_tag=[]
#     for batch_data,batch_tag in dev_dataloder:
#         batch_data = batch_data.to(config.device)
#         pre=model.decode(batch_data)
#         # print("pre:{}".format(type(batch_tag)))
#         pre_ = numpy.array(pre).reshape(-1).tolist()
#         std_ = batch_tag.detach().cpu().numpy().reshape(-1).tolist()
#
#         all_pre.extend(pre_)
#         all_tag.extend(std_)
#     dev_precision = precision_score(all_pre,all_tag, average='macro')
#     dev_recall = recall_score(all_pre,all_tag, average='macro')
#     dev_f1 = f1_score(all_pre,all_tag,average="macro")
#     return dev_precision,dev_recall,dev_f1


# 基于实体边界和实体类型的ner评测，用的是CoNLL-2000的一个评估脚本

# @torchsnooper.snoop()
def evaluate(model, dev_dataloder, config, index_2_tag,index_2_word, test=False):
    with torch.no_grad():
        total_loss = 0
        results = []
        i = 0
        with open(config.ner_result_path, "w", encoding='utf-8') as f:
            for batch_data, batch_tag in dev_dataloder:
                i += 1
                batch_data = batch_data.to(config.device)
                batch_tag = batch_tag.to(config.device)
                loss = model.compute_loss(batch_data, batch_tag)
                total_loss += loss.item()
                pre = model.decode(batch_data)

                """ 忽略<pad>标签，计算每个样本的真实长度 """
                batch_data_lens = [[word for word in sentence if word != 0] for sentence in batch_data.tolist()]
                batch_tag_lens = [[word for word in sentence if word != 0] for sentence in batch_tag.tolist()]
                lens = [len(i) for i in batch_tag_lens]
                pre_lens = [pre[i][:len_] for i, len_ in enumerate(lens)]

                """ id_2_tag"""
                data2 = [[index_2_word[char] for char in sentence] for sentence in batch_data_lens]
                pre_tag = [[index_2_tag[char] for char in sentence] for sentence in pre_lens]
                tag2 = [[index_2_tag[char] for char in sentence] for sentence in batch_tag_lens]
                """ 用CoNLL-2000的实体识别评估脚本, 需要按其要求的格式保存结果，
                即 字-真实标签-预测标签 用空格拼接"""
                for data, tag, pred in zip(data2, tag2, pre_tag):
                    for word in zip(data, tag, pred):
                        f.writelines(" ".join('%s' % id for id in list(word)) + "\n")



        aver_loss = total_loss / i
        """ 用CoNLL-2000的实体识别评估脚本来计算F1值 """
        eval_lines = evaluate_ner(config)

        if test:

            """ 如果是测试，则打印评估结果 """
            for line in eval_lines:
                logging.info(line)

        dev_f1 = float(eval_lines[1].strip().split()[-1]) / 100
        dev_precision = float(eval_lines[1].strip().split()[-3]) / 100
        dev_recall = float(eval_lines[1].strip().split()[-5]) / 100

        return dev_precision, dev_recall, dev_f1



def predict(model,input_str,config,word_2_index,index_2_tag):

    model.load_state_dict(torch.load(config.ner_model_path))
    model.eval()
    input_ = torch.LongTensor([word_2_index.get(i,0) for i in input_str]).unsqueeze(0).to(config.device)
    out=np.array(model.decode(input_)).flatten()
    tags=[index_2_tag[i] for i in out]

    return tags



if __name__ == '__main__':
    print("Loading data")
    config = Myconfig()
    train_data, train_tag, word_2_index, tag_2_index = build_corpus(config.train_path)
    dev_data, dev_tag = build_corpus(config.dev_path, make_vocab=False)
    index_2_tag = dict(zip(tag_2_index.values(), tag_2_index.keys()))
    index_2_word = dict(zip(word_2_index.values(), word_2_index.keys()))

    train_dataset = Mydatasets(train_data, train_tag, word_2_index, tag_2_index)
    train_dataloader = DataLoader(train_dataset, config.batch_size, shuffle=False, collate_fn=train_dataset.collate_fn,
                                  drop_last=True)

    dev_dataset = Mydatasets(dev_data, dev_tag, word_2_index, tag_2_index)
    dev_dataloader = DataLoader(dev_dataset, config.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn,
                                drop_last=False)

    corpus_num = len(word_2_index)
    class_num = len(tag_2_index)

    print("Building model")
    model = Mymodel(corpus_num=corpus_num, embedding_num=config.embedding_size, hidden_size=config.hidden_size,
                    class_num=class_num).to(config.device)
    optimizer = Adam(model.parameters(), lr=config.lr)
    #
    # print("Start training")
    #
    # """ 3: 用early stop 防止过拟合 """
    # dev_best_f1 = float('-inf')
    #
    # for i in range(10):
    #     epoch_loss = 0
    #
    #     bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc="train")
    #
    #     for index, (batch_data, batch_tag) in bar:
    #         model.train()
    #         batch_data = batch_data.to(config.device)
    #         batch_tag = batch_tag.to(config.device)
    #
    #         optimizer.zero_grad()
    #         loss = model.compute_loss(batch_data, batch_tag)
    #
    #         loss.backward()
    #         """ 梯度截断，最大梯度为5 """
    #         nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip)
    #         optimizer.step()
    #         epoch_loss += loss.item()
    #         if index % 100 == 0:
    #             dev_precision, dev_recall, dev_f1 = evaluate(model=model, dev_dataloder=dev_dataloader, config=config,
    #                                                          index_2_tag=index_2_tag, index_2_word=index_2_word)
    #             """ 以acc作为early stop的监控指标 """
    #             if dev_f1 > dev_best_f1:
    #                 dev_best_f1 = dev_f1
    #                 torch.save(model.state_dict(), config.ner_model_path)
    #             # print("epoch:{}\tidx:{}\tloss:{:.3f}\tbestf1:{:.3f}\trecall:{:.3f}\tprecision:{:.3f}".format(i, index, loss.item(),
    #             #                                                                                  dev_best_f1,
    #             #                                                                                  dev_recall,
    #             #                                                                                  dev_precision))
    #             bar.set_description("epoch:{}\tidx:{}\tloss:{:.3f}\tbestf1:{:.3f}\trecall:{:.3f}\tprecision:{:.3f}".format(i, index, loss.item(),dev_best_f1,dev_recall,dev_precision))

    print("Start testing")
    input_str1="那一天我二十一岁，在我一生的黄金时代。我有好多奢望，我想爱、想吃、还想在一瞬间变成天上半明半暗的云" \
              "。后来我才知道，生活就是慢慢受锤的过程，人一天天老下去，奢望也一天天消失，最后变得像挨了锤的牛一样。" \
              "可是我过二十一岁生日时没有预见到这一点，我觉得自己会永远生猛下去，什么也锤不了我。"
    input_str = "你看过许多包公戏"
    pre_tag=predict(model=model,input_str=input_str,config=config,word_2_index=word_2_index,index_2_tag=index_2_tag)
    print(pre_tag)

    # 训练流程
    # 1. 实例化model并传到cuda上,optimizer,Loss
    # 2 遍历dataloader,放到cuda上
    # 3 调用模型
    # 4 计算损失
    # 5 保存模型以及加载
